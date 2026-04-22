"""Arrival forecasts: learn future charging sessions from historical days.

An :class:`ArrivalForecast` is fitted to training days: scenarios on the same
daily clock (same start time, step and length), for example generator days
whose seeds are disjoint from the test seeds. It keeps every training session
and describes the distribution of one day's sessions by a *smoothed
bootstrap*: a sampled day draws its number of sessions from the training
days' counts and each session from the pooled training sessions, then adds
independent Gaussian kernel noise (bandwidth :attr:`ArrivalForecast.bandwidth_steps`)
to its arrival and departure. That is sampling from a kernel density estimate
of the joint distribution of arrival, departure, energy and power; energies,
powers and phase counts are kept as observed.

The forecast-aware MPC variants (:mod:`evcharge.policies.forecast`) use three
views of the future after the current step ``k``:

* :meth:`ArrivalForecast.expected_ghosts`: the expected future fleet. Every
  training session counts with weight ``P(arrival > k) / n_days`` under the
  kernel, and sessions are aggregated into cells of ``bin_steps`` by arrival,
  departure and phase count (a fluid approximation of the expectation);
* :meth:`ArrivalForecast.sample_ghosts`: one sampled future day, aggregated
  the same way;
* :meth:`ArrivalForecast.expected_load_kw`: the expected load of future
  sessions charging at their uniform rate, used as a capacity reserve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evcharge.model import FloatArray, Scenario, ValidationError


@dataclass(frozen=True)
class SessionSample:
    """One training session, on the forecast's step clock.

    Attributes:
        arrival_step: First step the EV is connected.
        departure_step: First step the EV is gone.
        energy_kwh: Energy requested (into the battery).
        max_power_kw: Grid-side power limit (EV and charger).
        efficiency: Grid-to-battery efficiency.
        phases: Phases the EV charges on (1 or 3; 3 on aggregate sites).
    """

    arrival_step: int
    departure_step: int
    energy_kwh: float
    max_power_kw: float
    efficiency: float
    phases: int = 3


@dataclass(frozen=True)
class Ghost:
    """An expected or sampled future session (possibly several EVs aggregated).

    Attributes:
        arrival_step: First step of its window (absolute).
        departure_step: End of its window (absolute, exclusive).
        energy_kwh: Grid-side energy it needs.
        max_power_kw: Grid-side power limit of the aggregate.
        phases: Phase count of its EVs (1 or 3).
    """

    arrival_step: int
    departure_step: int
    energy_kwh: float
    max_power_kw: float
    phases: int = 3


def _normal_cdf(z: FloatArray) -> FloatArray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return np.asarray(0.5 * (1.0 + erf(z / math.sqrt(2.0))), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class ArrivalForecast:
    """Kernel-smoothed empirical model of the sessions of a day.

    Build it with :meth:`fit`.

    Attributes:
        samples: Every training session.
        counts: Number of sessions of each training day.
        n_steps: Steps per day (the horizon length).
        step_minutes: Step length.
        start_minute: Start of the horizon in minutes after midnight.
        dt_h: Step length in hours.
        bandwidth_steps: Standard deviation of the kernel noise on arrival and
            departure times, in steps.
    """

    samples: tuple[SessionSample, ...]
    counts: tuple[int, ...]
    n_steps: int
    step_minutes: int
    start_minute: int
    bandwidth_steps: float

    @property
    def dt_h(self) -> float:
        """Step length in hours."""
        return self.step_minutes / 60.0

    @property
    def n_days(self) -> int:
        """Number of training days."""
        return len(self.counts)

    @property
    def mean_sessions_per_day(self) -> float:
        """Mean number of sessions per training day."""
        return float(np.mean(self.counts))

    @classmethod
    def fit(
        cls, days: Sequence[Scenario], *, bandwidth_steps: float | None = None
    ) -> ArrivalForecast:
        """Learn the forecast from training days.

        Args:
            days: Training scenarios, all on the same clock (start time of day,
                step, number of steps).
            bandwidth_steps: Kernel bandwidth in steps; by default Silverman's
                rule of thumb on the pooled arrival steps,
                ``0.9 * min(sd, IQR / 1.34) * n ** (-1/5)``.

        Raises:
            ValidationError: without days or if their clocks differ.
        """
        if not days:
            raise ValidationError("a forecast needs at least one training day")
        clock = _clock(days[0])
        samples: list[SessionSample] = []
        counts: list[int] = []
        for day in days:
            if _clock(day) != clock:
                raise ValidationError(
                    f"training day {day.name!r} is on a different clock "
                    f"(start minute, step, steps) {_clock(day)} than {clock}"
                )
            counts.append(len(day.sessions))
            for s in day.sessions:
                p_max = day.power_bounds(s)[1]
                samples.append(
                    SessionSample(
                        s.arrival_step,
                        s.departure_step,
                        s.energy_kwh,
                        p_max,
                        s.efficiency,
                        s.phases if day.site.phase_aware else 3,
                    )
                )
        if bandwidth_steps is None:
            bandwidth_steps = _silverman([s.arrival_step for s in samples])
        if bandwidth_steps < 0 or not math.isfinite(bandwidth_steps):
            raise ValidationError("bandwidth_steps must be finite and >= 0")
        start, step, n_steps = clock
        return cls(tuple(samples), tuple(counts), n_steps, step, start, float(bandwidth_steps))

    def check(self, scenario: Scenario) -> None:
        """Raise :class:`ValidationError` unless ``scenario`` is on the forecast's clock."""
        clock = (self.start_minute, self.step_minutes, self.n_steps)
        if _clock(scenario) != clock:
            raise ValidationError(
                f"scenario {scenario.name!r} is not on the forecast's clock "
                f"(start minute, step, steps) {clock}"
            )

    def expected_ghosts(self, now: int, *, bin_steps: int = 4) -> tuple[Ghost, ...]:
        """Expected fleet of the sessions arriving after step ``now``.

        Each training session gets weight ``P(arrival > now) / n_days``, the
        arrival probability under the kernel; its window starts no earlier than
        ``now + 1``. Sessions are then aggregated into cells of ``bin_steps``
        by arrival, departure and phase count.
        """
        arrival = np.array([s.arrival_step for s in self.samples], dtype=np.float64)
        if self.bandwidth_steps > 0.0:
            later = _normal_cdf((arrival - (now + 0.5)) / self.bandwidth_steps)
        else:
            later = (arrival > now).astype(np.float64)
        weights = later / self.n_days
        rows = [
            (max(s.arrival_step, now + 1), s.departure_step, s, float(w))
            for s, w in zip(self.samples, weights, strict=True)
            if w > 1e-6
        ]
        return self._aggregate(rows, bin_steps)

    def sample_ghosts(
        self, rng: np.random.Generator, now: int, *, bin_steps: int = 4
    ) -> tuple[Ghost, ...]:
        """Sessions arriving after step ``now`` in one day sampled from the forecast."""
        n = int(self.counts[int(rng.integers(len(self.counts)))])
        picks = rng.integers(len(self.samples), size=n)
        noise = rng.normal(0.0, self.bandwidth_steps, size=(n, 2)) if n else np.zeros((0, 2))
        rows = []
        for j, (da, dd) in zip(picks, noise, strict=True):
            s = self.samples[int(j)]
            a = int(np.clip(round(s.arrival_step + da), 0, self.n_steps - 1))
            d = int(np.clip(round(s.departure_step + dd), a + 1, self.n_steps))
            if a > now:
                rows.append((a, d, s, 1.0))
        return self._aggregate(rows, bin_steps)

    def expected_load_kw(self, now: int, *, phases: int | None = None) -> FloatArray:
        """Expected load (kW) of sessions arriving after ``now``, each at its uniform rate.

        Every expected session's grid-side energy is spread evenly over its
        window (which starts no earlier than ``now + 1``). With ``phases``,
        only sessions of that phase count are included.
        """
        load = np.zeros(self.n_steps)
        for g in self.expected_ghosts(now, bin_steps=1):
            if phases is not None and g.phases != phases:
                continue
            steps = g.departure_step - g.arrival_step
            load[g.arrival_step : g.departure_step] += g.energy_kwh / (steps * self.dt_h)
        return load

    @property
    def phase_counts(self) -> tuple[int, ...]:
        """Phase counts that occur in the training sessions."""
        return tuple(sorted({s.phases for s in self.samples}))

    def _aggregate(
        self, rows: list[tuple[int, int, SessionSample, float]], bin_steps: int
    ) -> tuple[Ghost, ...]:
        """Sum weighted sessions per (arrival cell, departure cell, phases)."""
        if bin_steps < 1:
            raise ValueError("bin_steps must be >= 1")
        cells: dict[tuple[int, int, int], list[float]] = {}
        for a, d, s, w in rows:
            if d <= a:
                continue
            # energy that fits in the (possibly shortened) window at full power
            grid_kwh = min(s.energy_kwh / s.efficiency, s.max_power_kw * (d - a) * self.dt_h)
            key = (a // bin_steps, d // bin_steps, s.phases)
            acc = cells.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            acc[0] += w
            acc[1] += w * a
            acc[2] += w * d
            acc[3] += w * grid_kwh
            acc[4] += w * s.max_power_kw
        ghosts = []
        for (_, _, phases), (w, wa, wd, energy, power) in sorted(cells.items()):
            a = round(wa / w)
            d = max(a + 1, round(wd / w))
            d = min(d, self.n_steps)
            if d <= a:
                continue
            energy = min(energy, power * (d - a) * self.dt_h)
            ghosts.append(Ghost(a, d, energy, power, phases))
        return tuple(ghosts)


def _clock(sc: Scenario) -> tuple[int, int, int]:
    hz = sc.horizon
    return (hz.start.hour * 60 + hz.start.minute, hz.step_minutes, hz.n_steps)


def _silverman(values: Sequence[int]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    spread = min(sd, float(q75 - q25) / 1.34) if q75 > q25 else sd
    return float(0.9 * spread * x.size ** (-0.2))
