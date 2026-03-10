"""Domain model: time grid, chargers, site, charging sessions, tariff and scenario.

All quantities use SI-ish engineering units that are customary in EV charging:
power in kW, energy in kWh, prices in EUR/kWh and demand charges in EUR/kW.

Time is discretised into equal steps (default 15 minutes).  Power is modelled as
the *average* power over a step, which is also how settlement meters and most
smart-charging back-ends (OCPP ``ChargingProfile`` periods) reason about it.

Every dataclass validates itself on construction and raises
:class:`ValidationError` (a :class:`ValueError`) with an explicit message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
"""One-dimensional (or two-dimensional) array of ``float64`` values."""

POWER_TOL_KW = 1e-6
"""Numerical tolerance for power comparisons (kW)."""

ENERGY_TOL_KWH = 1e-6
"""Numerical tolerance for energy comparisons (kWh)."""

IEC_61851_MIN_CURRENT_A = 6.0
"""Lowest current an AC charger may signal to the EV (IEC 61851-1 PWM, 10 % duty cycle)."""

NOMINAL_PHASE_VOLTAGE_V = 230.0
"""Nominal phase-to-neutral voltage of a European low-voltage grid."""


class ValidationError(ValueError):
    """Raised when model input is inconsistent or physically impossible."""


def ac_power_kw(
    current_a: float, phases: int = 3, voltage_v: float = NOMINAL_PHASE_VOLTAGE_V
) -> float:
    """Return the AC charging power in kW for a per-phase current.

    ``ac_power_kw(6.0)`` gives the IEC 61851 minimum for a three-phase
    connection, about 4.14 kW; ``ac_power_kw(6.0, phases=1)`` about 1.38 kW.
    """
    if phases not in (1, 2, 3):
        raise ValidationError(f"phases must be 1, 2 or 3, got {phases}")
    if current_a < 0 or voltage_v <= 0:
        raise ValidationError("current must be >= 0 and voltage > 0")
    return phases * current_a * voltage_v / 1000.0


MIN_POWER_3PH_KW = ac_power_kw(IEC_61851_MIN_CURRENT_A, phases=3)
"""Minimum charging power of a three-phase EV at 6 A (about 4.14 kW)."""

MIN_POWER_1PH_KW = ac_power_kw(IEC_61851_MIN_CURRENT_A, phases=1)
"""Minimum charging power of a single-phase EV at 6 A (about 1.38 kW)."""


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")


def as_series(name: str, values: object, n_steps: int) -> FloatArray:
    """Convert ``values`` to a finite float64 array of length ``n_steps``.

    Raises:
        ValidationError: if the shape is wrong or any value is NaN/inf.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n_steps:
        raise ValidationError(f"{name} must have exactly {n_steps} values, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class Horizon:
    """Discrete time grid ``[start, start + n_steps * step)``.

    Attributes:
        start: Wall-clock time of the first step (naive or timezone-aware).
        n_steps: Number of control steps.
        step_minutes: Length of one step in minutes (default 15).
    """

    start: datetime
    n_steps: int
    step_minutes: int = 15

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValidationError(f"n_steps must be positive, got {self.n_steps}")
        if not 1 <= self.step_minutes <= 1440:
            raise ValidationError(f"step_minutes must be in [1, 1440], got {self.step_minutes}")

    @classmethod
    def spanning(cls, start: datetime, hours: float, step_minutes: int = 15) -> Horizon:
        """Build a horizon covering ``hours`` hours; must be a whole number of steps."""
        steps = hours * 60.0 / step_minutes
        if steps <= 0 or abs(steps - round(steps)) > 1e-9:
            raise ValidationError(
                f"{hours} h is not a positive whole number of {step_minutes}-minute steps"
            )
        return cls(start=start, n_steps=round(steps), step_minutes=step_minutes)

    @property
    def dt_h(self) -> float:
        """Step length in hours."""
        return self.step_minutes / 60.0

    @property
    def step(self) -> timedelta:
        """Step length as a :class:`~datetime.timedelta`."""
        return timedelta(minutes=self.step_minutes)

    @property
    def end(self) -> datetime:
        """Wall-clock end of the horizon (exclusive)."""
        return self.start + self.n_steps * self.step

    def time_of(self, step: int) -> datetime:
        """Wall-clock start time of ``step`` (``step == n_steps`` gives :attr:`end`)."""
        if not 0 <= step <= self.n_steps:
            raise ValidationError(f"step {step} outside horizon [0, {self.n_steps}]")
        return self.start + step * self.step

    def clock_hours(self) -> FloatArray:
        """Hour-of-day (0-24, fractional) at the start of every step."""
        base = self.start.hour + self.start.minute / 60.0 + self.start.second / 3600.0
        return np.asarray((base + np.arange(self.n_steps) * self.dt_h) % 24.0, dtype=np.float64)

    def _offset_steps(self, t: datetime) -> float:
        if (t.tzinfo is None) != (self.start.tzinfo is None):
            raise ValidationError(
                f"cannot mix naive and timezone-aware datetimes ({t.isoformat()} vs "
                f"{self.start.isoformat()})"
            )
        return (t - self.start).total_seconds() / (60.0 * self.step_minutes)

    def step_at_or_after(self, t: datetime) -> int:
        """Index of the first step starting at or after ``t`` (used for arrivals)."""
        return math.ceil(self._offset_steps(t) - 1e-9)

    def step_at_or_before(self, t: datetime) -> int:
        """Index of the last step boundary at or before ``t`` (used for departures)."""
        return math.floor(self._offset_steps(t) + 1e-9)


@dataclass(frozen=True)
class Charger:
    """A charge point (one connector) with its controllable power range.

    Attributes:
        id: Unique charger identifier.
        max_power_kw: Maximum power the charger can deliver.
        min_power_kw: Lowest non-zero power the charger can signal. For AC
            chargers this follows from the IEC 61851 minimum of 6 A (about
            4.14 kW on three phases). Use 0 for continuously adjustable chargers.
    """

    id: str
    max_power_kw: float
    min_power_kw: float = MIN_POWER_3PH_KW

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("charger id must be a non-empty string")
        _require_finite(f"charger {self.id}: max_power_kw", self.max_power_kw)
        _require_finite(f"charger {self.id}: min_power_kw", self.min_power_kw)
        if self.max_power_kw <= 0:
            raise ValidationError(f"charger {self.id}: max_power_kw must be > 0")
        if not 0 <= self.min_power_kw <= self.max_power_kw:
            raise ValidationError(
                f"charger {self.id}: min_power_kw must be in [0, max_power_kw], "
                f"got {self.min_power_kw} (max {self.max_power_kw})"
            )


@dataclass(frozen=True)
class Site:
    """A site behind one grid connection.

    Attributes:
        grid_limit_kw: Maximum import at the grid connection point (fuse or
            contractual capacity), shared by base load and EV charging.
        chargers: Charge points installed at the site.
    """

    grid_limit_kw: float
    chargers: tuple[Charger, ...]

    def __post_init__(self) -> None:
        _require_finite("grid_limit_kw", self.grid_limit_kw)
        if self.grid_limit_kw <= 0:
            raise ValidationError(f"grid_limit_kw must be > 0, got {self.grid_limit_kw}")
        if not self.chargers:
            raise ValidationError("a site needs at least one charger")
        ids = [c.id for c in self.chargers]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValidationError(f"duplicate charger ids: {', '.join(dupes)}")

    def charger(self, charger_id: str) -> Charger:
        """Look up a charger by id."""
        for c in self.chargers:
            if c.id == charger_id:
                return c
        raise ValidationError(f"unknown charger id {charger_id!r}")


@dataclass(frozen=True)
class Session:
    """One EV charging session (plug-in to plug-out).

    Attributes:
        id: Unique session identifier.
        charger_id: Charger the EV is plugged into.
        arrival_step: First step during which the EV is connected.
        departure_step: First step during which the EV is gone (exclusive).
        energy_kwh: Energy the driver requested, measured into the battery.
        max_power_kw: Maximum power the EV accepts on this charger
            (on-board charger limit as realised on this connection).
        efficiency: Grid-to-battery efficiency in ``(0, 1]``; drawing ``p`` kW
            for ``dt`` hours adds ``efficiency * p * dt`` kWh to the battery.
        min_power_kw: Optional EV-specific minimum power when charging, e.g.
            about 1.38 kW for a single-phase EV at 6 A. Defaults to the
            charger's minimum.
    """

    id: str
    charger_id: str
    arrival_step: int
    departure_step: int
    energy_kwh: float
    max_power_kw: float
    efficiency: float = 0.9
    min_power_kw: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("session id must be a non-empty string")
        where = f"session {self.id}"
        if self.arrival_step < 0:
            raise ValidationError(f"{where}: arrival_step must be >= 0")
        if self.departure_step <= self.arrival_step:
            raise ValidationError(
                f"{where}: departure_step ({self.departure_step}) must be after "
                f"arrival_step ({self.arrival_step}); the plug-in window is shorter than one step"
            )
        for name in ("energy_kwh", "max_power_kw", "efficiency"):
            _require_finite(f"{where}: {name}", float(getattr(self, name)))
        if self.energy_kwh <= 0:
            raise ValidationError(f"{where}: energy_kwh must be > 0")
        if self.max_power_kw <= 0:
            raise ValidationError(f"{where}: max_power_kw must be > 0")
        if not 0 < self.efficiency <= 1:
            raise ValidationError(f"{where}: efficiency must be in (0, 1], got {self.efficiency}")
        if self.min_power_kw is not None:
            _require_finite(f"{where}: min_power_kw", self.min_power_kw)
            if not 0 <= self.min_power_kw <= self.max_power_kw:
                raise ValidationError(f"{where}: min_power_kw must be in [0, max_power_kw]")

    @property
    def dwell_steps(self) -> int:
        """Number of steps the EV is connected."""
        return self.departure_step - self.arrival_step

    def is_connected(self, step: int) -> bool:
        """Whether the EV is plugged in during ``step``."""
        return self.arrival_step <= step < self.departure_step


@dataclass(frozen=True, eq=False)
class Tariff:
    """Time-of-use energy prices and a demand (peak) charge.

    Attributes:
        price_eur_per_kwh: Import price per step (energy + grid energy fee + taxes).
        export_price_eur_per_kwh: Price received for exported energy per step;
            defaults to zero. Must not exceed the import price.
        demand_charge_eur_per_kw: Charge on the highest average import over the
            simulated horizon (EUR per kW). Pro-rate a monthly tariff when you
            simulate shorter periods.
    """

    price_eur_per_kwh: FloatArray
    export_price_eur_per_kwh: FloatArray | None = None
    demand_charge_eur_per_kw: float = 0.0

    def __post_init__(self) -> None:
        price = np.asarray(self.price_eur_per_kwh, dtype=np.float64)
        if price.ndim != 1 or price.size == 0:
            raise ValidationError("price_eur_per_kwh must be a non-empty 1-D series")
        price = as_series("price_eur_per_kwh", price, price.size)
        if self.export_price_eur_per_kwh is None:
            export = as_series("export_price_eur_per_kwh", np.zeros(price.size), price.size)
        else:
            export = as_series(
                "export_price_eur_per_kwh", self.export_price_eur_per_kwh, price.size
            )
        if np.any(export > price + 1e-12):
            bad = int(np.argmax(export > price + 1e-12))
            raise ValidationError(
                f"export price exceeds import price at step {bad} "
                f"({export[bad]:.4f} > {price[bad]:.4f} EUR/kWh)"
            )
        _require_finite("demand_charge_eur_per_kw", self.demand_charge_eur_per_kw)
        if self.demand_charge_eur_per_kw < 0:
            raise ValidationError("demand_charge_eur_per_kw must be >= 0")
        object.__setattr__(self, "price_eur_per_kwh", price)
        object.__setattr__(self, "export_price_eur_per_kwh", export)

    @property
    def export_price(self) -> FloatArray:
        """Export price series (never ``None`` after validation)."""
        assert self.export_price_eur_per_kwh is not None
        return self.export_price_eur_per_kwh

    @property
    def n_steps(self) -> int:
        """Length of the price series."""
        return int(self.price_eur_per_kwh.size)


@dataclass(frozen=True, eq=False)
class Scenario:
    """A complete, validated problem instance.

    Attributes:
        name: Human-readable name.
        horizon: Time grid.
        site: Grid connection and chargers.
        tariff: Prices (one value per step).
        sessions: Charging sessions; sessions on the same charger must not overlap.
        base_load_kw: Non-EV site consumption per step (default: zero).
        pv_kw: On-site PV production per step (default: zero).
    """

    name: str
    horizon: Horizon
    site: Site
    tariff: Tariff
    sessions: tuple[Session, ...]
    base_load_kw: FloatArray | None = None
    pv_kw: FloatArray | None = None
    _bounds: dict[str, tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.horizon.n_steps
        if self.tariff.n_steps != n:
            raise ValidationError(f"tariff has {self.tariff.n_steps} steps but the horizon has {n}")
        base = as_series(
            "base_load_kw", np.zeros(n) if self.base_load_kw is None else self.base_load_kw, n
        )
        pv = as_series("pv_kw", np.zeros(n) if self.pv_kw is None else self.pv_kw, n)
        if np.any(base < 0):
            raise ValidationError("base_load_kw must be >= 0")
        if np.any(pv < 0):
            raise ValidationError("pv_kw must be >= 0")
        over = base - pv - self.site.grid_limit_kw
        if np.any(over > POWER_TOL_KW):
            bad = int(np.argmax(over))
            raise ValidationError(
                f"base load minus PV exceeds the grid limit at step {bad} "
                f"({self.horizon.time_of(bad).isoformat()}): {base[bad] - pv[bad]:.2f} kW > "
                f"{self.site.grid_limit_kw:.2f} kW"
            )
        object.__setattr__(self, "base_load_kw", base)
        object.__setattr__(self, "pv_kw", pv)
        object.__setattr__(self, "sessions", tuple(self.sessions))

        bounds: dict[str, tuple[float, float]] = {}
        by_charger: dict[str, list[Session]] = {}
        for s in self.sessions:
            if s.id in bounds:
                raise ValidationError(f"duplicate session id {s.id!r}")
            charger = self.site.charger(s.charger_id)
            if s.departure_step > n:
                raise ValidationError(
                    f"session {s.id}: departs at step {s.departure_step}, "
                    f"after the horizon end ({n})"
                )
            p_max = min(charger.max_power_kw, s.max_power_kw)
            p_min = charger.min_power_kw if s.min_power_kw is None else s.min_power_kw
            bounds[s.id] = (min(p_min, p_max), p_max)
            by_charger.setdefault(s.charger_id, []).append(s)
        for cid, sessions in by_charger.items():
            ordered = sorted(sessions, key=lambda s: s.arrival_step)
            for a, b in pairwise(ordered):
                if b.arrival_step < a.departure_step:
                    raise ValidationError(f"sessions {a.id} and {b.id} overlap on charger {cid}")
        object.__setattr__(self, "_bounds", bounds)

    @property
    def base_load(self) -> FloatArray:
        """Base load series in kW (zeros if not given)."""
        assert self.base_load_kw is not None
        return self.base_load_kw

    @property
    def pv(self) -> FloatArray:
        """PV production series in kW (zeros if not given)."""
        assert self.pv_kw is not None
        return self.pv_kw

    @property
    def net_base_kw(self) -> FloatArray:
        """Site import without any EV charging: base load minus PV (may be negative)."""
        return np.asarray(self.base_load - self.pv, dtype=np.float64)

    @property
    def ev_headroom_kw(self) -> FloatArray:
        """Power available for EV charging per step: grid limit - base load + PV."""
        return np.asarray(self.site.grid_limit_kw - self.net_base_kw, dtype=np.float64)

    def power_bounds(self, session: Session) -> tuple[float, float]:
        """Return ``(p_min, p_max)`` in kW for a session of this scenario.

        ``p_max`` is the lower of charger and EV limits. ``p_min`` is the lowest
        non-zero power that can be signalled (EV override or charger minimum,
        never above ``p_max``). A session charges at 0 or within ``[p_min, p_max]``.
        """
        try:
            return self._bounds[session.id]
        except KeyError:
            raise ValidationError(f"session {session.id!r} is not part of this scenario") from None

    @property
    def energy_requested_kwh(self) -> float:
        """Total energy requested by all sessions."""
        return float(sum(s.energy_kwh for s in self.sessions))

    def without_sessions(self) -> Scenario:
        """Copy of the scenario with sessions removed (what an online policy may know)."""
        return Scenario(
            name=self.name,
            horizon=self.horizon,
            site=self.site,
            tariff=self.tariff,
            sessions=(),
            base_load_kw=self.base_load,
            pv_kw=self.pv,
        )
