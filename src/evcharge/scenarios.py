"""Seeded synthetic scenarios: workplace, depot and residential charging.

Everything is generated offline from a :class:`numpy.random.Generator`, so a
``(scenario, n_sessions, seed, ...)`` tuple always yields the same instance.

The shapes are stylised but grounded in Nordic practice:

* **Prices** follow a day-ahead-like curve with a morning (about 08:00) and an
  evening (about 18:00) peak and a night trough, at 15-minute resolution as in
  the Nordic day-ahead market since 2025, plus a constant grid energy fee.
  PV export is credited at the spot price only.
* **Chargers** are AC wallboxes whose minimum power follows the IEC 61851 6 A
  rule; single-phase EVs charge at 3.7 kW maximum and 1.38 kW minimum.
* **Energy requests** are capped at 95 % of what the EV could take alone during
  its stay, so every session is individually feasible and unmet energy is
  caused by the shared grid limit, not by impossible requests.
* **Phase-aware sites** (``grid="TN"`` or ``"IT"``) command chargers in amperes:
  11 and 22 kW chargers are three-phase 16 and 32 A units, the 3.7 kW EVs are
  single-phase 16 A cars, and the main fuse defaults to the current at which a
  balanced load reaches the kW limit. Chargers are installed with cyclic phase
  rotation unless ``rotate_phases=False``. All phase options are applied after
  the random draws, so they never change the sessions' times or energies.
* **V2G** (``v2g_share > 0``): that share of the EVs get a battery model and a
  bidirectional charger. Batteries are illustrative (40 to 77 kWh by EV type)
  and arrive 30 to 70 % full; the aggregator may use the range from 20 % (the
  driver's reserve) to 90 %. These draws come from a separate random stream,
  so a scenario without V2G is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypedDict, Unpack

import numpy as np

from evcharge.electrical import (
    NOMINAL_VOLTAGE_V,
    GridType,
    Supply,
    WiringError,
    parse_rotation,
    three_phase_kw_per_a,
    wiring,
)
from evcharge.model import (
    MIN_POWER_1PH_KW,
    MIN_POWER_3PH_KW,
    V2G,
    Charger,
    FloatArray,
    Horizon,
    Scenario,
    Session,
    Site,
    Tariff,
    ValidationError,
)

GRID_FEE_EUR_PER_KWH = 0.04
"""Illustrative grid energy fee plus consumption tax added to the spot price."""

DEFAULT_DEMAND_CHARGE_EUR_PER_KW = 0.3
"""Illustrative demand charge for a one-day horizon (a monthly peak tariff pro-rated)."""

SCENARIO_DATE = datetime(2026, 4, 15)
"""Reference day of the synthetic scenarios (a mid-April weekday)."""


@dataclass(frozen=True)
class _EVType:
    max_power_kw: float
    min_power_kw: float
    share: float
    phases: int = 3
    max_current_a: float = 16.0
    battery_kwh: float = 60.0


DEFAULT_DEGRADATION_EUR_PER_KWH = 0.04
"""Illustrative battery wear per kWh of throughput: about 120 EUR/kWh of pack cost
spread over some 1,500 full cycles (3,000 kWh of throughput per kWh of capacity)."""


ROTATIONS = ("L1L2L3", "L2L3L1", "L3L1L2")
"""Cyclic phase rotations used for the chargers of phase-aware scenarios."""


def _bump(hours: FloatArray, centre: float, width: float) -> FloatArray:
    """Gaussian bump on a 24 h circle."""
    d = (hours - centre + 12.0) % 24.0 - 12.0
    return np.asarray(np.exp(-0.5 * (d / width) ** 2), dtype=np.float64)


def _ar1_noise(rng: np.random.Generator, n: int, sigma: float, rho: float) -> FloatArray:
    """Stationary AR(1) noise with standard deviation ``sigma``."""
    out = np.empty(n)
    innovation = sigma * math.sqrt(1.0 - rho * rho)
    x = rng.normal(0.0, sigma)
    for i in range(n):
        out[i] = x
        x = rho * x + rng.normal(0.0, innovation)
    return out


def day_ahead_spot_price(horizon: Horizon, rng: np.random.Generator) -> FloatArray:
    """Spot-price-like series (EUR/kWh) with morning and evening peaks.

    The curve is a night trough around 03:30, a morning peak around 08:00, a
    shallow solar dip around 13:00 and a higher evening peak around 18:30, with
    autocorrelated multiplicative noise. Levels are illustrative, not a forecast.
    """
    h = horizon.clock_hours()
    shape = (
        0.055
        + 0.045 * _bump(h, 8.0, 1.2)
        + 0.060 * _bump(h, 18.5, 1.6)
        - 0.012 * _bump(h, 13.0, 2.0)
        - 0.015 * _bump(h, 3.5, 2.0)
    )
    noise = _ar1_noise(rng, horizon.n_steps, sigma=0.06, rho=0.9)
    return np.asarray(np.maximum(0.0, shape * (1.0 + noise)), dtype=np.float64)


def pv_profile(horizon: Horizon, kwp: float, rng: np.random.Generator) -> FloatArray:
    """PV output (kW) for a mid-April day at about 60 degrees north.

    A clear-sky bell between about 06:20 and 20:50 local time (peak yield 75 %
    of ``kwp`` near 13:30), scaled by a seeded, slowly varying cloud factor.
    """
    if kwp < 0:
        raise ValidationError("pv kWp must be >= 0")
    if kwp == 0:
        return np.zeros(horizon.n_steps)
    h = horizon.clock_hours()
    sunrise, sunset = 6.33, 20.83
    phase = np.clip((h - sunrise) / (sunset - sunrise), 0.0, 1.0)
    daylight = (h > sunrise) & (h < sunset)
    clear_sky = np.where(daylight, 0.75 * kwp * np.sin(np.pi * phase) ** 1.4, 0.0)
    cloud = np.clip(0.85 + _ar1_noise(rng, horizon.n_steps, sigma=0.15, rho=0.95), 0.3, 1.0)
    return np.asarray(clear_sky * cloud, dtype=np.float64)


def _base_load(
    horizon: Horizon, peak_kw: float, profile: str, rng: np.random.Generator
) -> FloatArray:
    h = horizon.clock_hours()
    if profile == "office":
        occupancy = np.clip(_bump(h, 12.0, 3.0) * 1.3, 0.0, 1.0)
        shape = 0.3 + 0.7 * occupancy
    elif profile == "depot":
        shape = 0.4 + 0.6 * np.clip(_bump(h, 12.0, 3.5) * 1.2, 0.0, 1.0)
    else:  # residential building
        shape = 0.35 + 0.25 * _bump(h, 7.5, 1.0) + 0.4 * _bump(h, 19.0, 2.0)
    noise = _ar1_noise(rng, horizon.n_steps, sigma=0.05, rho=0.8)
    return np.asarray(np.maximum(0.0, peak_kw * shape * (1.0 + noise)), dtype=np.float64)


def _assign_chargers(windows: list[tuple[int, int]]) -> tuple[list[str], int]:
    """Greedy interval colouring: reuse the first free charger, add one if none is free."""
    free_at: list[int] = []
    assignment = [""] * len(windows)
    for i in sorted(range(len(windows)), key=lambda k: windows[k]):
        arrival, departure = windows[i]
        for c, t_free in enumerate(free_at):
            if t_free <= arrival:
                free_at[c] = departure
                assignment[i] = f"CP{c + 1:02d}"
                break
        else:
            free_at.append(departure)
            assignment[i] = f"CP{len(free_at):02d}"
    return assignment, len(free_at)


def _ev_mix(types: tuple[_EVType, ...], single_phase_share: float | None) -> tuple[_EVType, ...]:
    """EV types with the single-phase share overridden (``None``: unchanged)."""
    if single_phase_share is None:
        return types
    if not 0.0 <= single_phase_share <= 1.0:
        raise ValidationError("single_phase_share must be in [0, 1]")
    if not any(t.phases == 1 for t in types):
        types = (*types, _SINGLE_PHASE_VAN)
    shares = [0.0] * len(types)
    for single, total in ((True, single_phase_share), (False, 1.0 - single_phase_share)):
        group = [i for i, t in enumerate(types) if (t.phases == 1) == single]
        weights = [types[i].share for i in group]
        if sum(weights) == 0.0:
            weights = [1.0] * len(group)
        for i, w in zip(group, weights, strict=True):
            shares[i] = total * w / sum(weights)
    return tuple(
        _EVType(t.max_power_kw, t.min_power_kw, shares[i], t.phases, t.max_current_a, t.battery_kwh)
        for i, t in enumerate(types)
    )


@dataclass(frozen=True)
class _Profile:
    label: str
    start_hour: int
    arrival_mean_h: float
    arrival_sd_h: float
    arrival_clip: tuple[float, float]
    departure: Callable[[np.random.Generator, float], float]
    energy_median_kwh: float
    energy_sigma: float
    energy_clip: tuple[float, float]
    charger_kw: float
    ev_types: tuple[_EVType, ...]
    base_profile: str
    base_peak_per_session_kw: float
    grid_kw_per_session: float


def _workplace_departure(rng: np.random.Generator, arrival_h: float) -> float:
    return arrival_h + float(np.clip(rng.normal(8.25, 1.0), 3.0, 11.0))


def _overnight_departure(
    mean_h: float, sd_h: float, clip: tuple[float, float]
) -> Callable[[np.random.Generator, float], float]:
    def departure(rng: np.random.Generator, arrival_h: float) -> float:
        # Next-morning clock time, expressed in hours after the 12:00 horizon start.
        return 24.0 + float(np.clip(rng.normal(mean_h, sd_h), *clip))

    return departure


_EV_MIX_AC = (
    _EVType(11.0, MIN_POWER_3PH_KW, 0.65, 3, 16.0, 64.0),
    _EVType(3.7, MIN_POWER_1PH_KW, 0.15, 1, 16.0, 40.0),
    _EVType(22.0, MIN_POWER_3PH_KW, 0.20, 3, 32.0, 77.0),
)

_SINGLE_PHASE_VAN = _EVType(7.4, MIN_POWER_1PH_KW, 0.0, 1, 32.0, 50.0)
"""Added to profiles without a single-phase EV when ``single_phase_share`` is set."""

PROFILES: dict[str, _Profile] = {
    "workplace": _Profile(
        label="workplace",
        start_hour=0,
        arrival_mean_h=8.25,
        arrival_sd_h=0.8,
        arrival_clip=(6.0, 11.0),
        departure=_workplace_departure,
        energy_median_kwh=7.0,
        energy_sigma=0.55,
        energy_clip=(2.0, 40.0),
        charger_kw=11.0,
        ev_types=_EV_MIX_AC,
        base_profile="office",
        base_peak_per_session_kw=0.25,
        grid_kw_per_session=1.5,
    ),
    "depot": _Profile(
        label="depot",
        start_hour=12,
        arrival_mean_h=17.0 - 12.0,
        arrival_sd_h=1.0,
        arrival_clip=(14.5 - 12.0, 20.5 - 12.0),
        departure=_overnight_departure(6.0 - 12.0, 0.5, (4.5 - 12.0, 8.0 - 12.0)),
        energy_median_kwh=45.0,
        energy_sigma=0.3,
        energy_clip=(15.0, 90.0),
        charger_kw=22.0,
        ev_types=(
            _EVType(22.0, MIN_POWER_3PH_KW, 0.5, 3, 32.0, 75.0),
            _EVType(11.0, MIN_POWER_3PH_KW, 0.5, 3, 16.0, 50.0),
        ),
        base_profile="depot",
        base_peak_per_session_kw=0.5,
        grid_kw_per_session=6.0,
    ),
    "residential": _Profile(
        label="residential",
        start_hour=12,
        arrival_mean_h=17.5 - 12.0,
        arrival_sd_h=1.5,
        arrival_clip=(15.0 - 12.0, 23.0 - 12.0),
        departure=_overnight_departure(7.25 - 12.0, 0.6, (6.0 - 12.0, 9.0 - 12.0)),
        energy_median_kwh=8.0,
        energy_sigma=0.55,
        energy_clip=(2.0, 40.0),
        charger_kw=11.0,
        ev_types=_EV_MIX_AC,
        base_profile="residential",
        base_peak_per_session_kw=0.4,
        grid_kw_per_session=1.25,
    ),
}
"""Built-in scenario profiles, keyed by CLI name."""


def generate(
    kind: str,
    *,
    n_sessions: int = 40,
    seed: int = 7,
    grid_limit_kw: float | None = None,
    step_minutes: int = 15,
    pv_kwp: float = 0.0,
    base_load_peak_kw: float | None = None,
    demand_charge_eur_per_kw: float = DEFAULT_DEMAND_CHARGE_EUR_PER_KW,
    grid: GridType | str | None = None,
    line_limit_a: float | None = None,
    rotate_phases: bool = True,
    single_phase_share: float | None = None,
    v2g_share: float = 0.0,
    degradation_eur_per_kwh: float = DEFAULT_DEGRADATION_EUR_PER_KWH,
) -> Scenario:
    """Generate a synthetic scenario.

    Args:
        kind: ``"workplace"`` (morning arrivals, about 8 h dwell), ``"depot"``
            (fleet returning in the evening, large energies, 22 kW chargers) or
            ``"residential"`` (evening arrivals, departures next morning).
        n_sessions: Number of charging sessions.
        seed: Seed for all random draws.
        grid_limit_kw: Site import limit; defaults to a per-profile kW per session.
        step_minutes: Control step; must divide 60.
        pv_kwp: Installed PV peak power (0 disables PV).
        base_load_peak_kw: Peak of the non-EV base load; defaults to a
            per-profile kW per session.
        demand_charge_eur_per_kw: Demand charge on the horizon's peak import.
        grid: ``"TN"`` or ``"IT"`` for a phase-aware site (``None``: aggregate kW model).
        line_limit_a: Main fuse per line; defaults to the current at which a
            balanced load reaches ``grid_limit_kw``.
        rotate_phases: Install chargers with cyclic phase rotation
            (``L1L2L3``, ``L2L3L1``, ``L3L1L2``) instead of all ``L1L2L3``.
        single_phase_share: Override the share of single-phase EVs
            (phase-aware sites only).
        v2g_share: Share of EVs with a battery model on a bidirectional charger.
        degradation_eur_per_kwh: Battery wear cost of those EVs per kWh of throughput.

    Raises:
        ValidationError: for unknown kinds or invalid parameters.
    """
    if kind not in PROFILES:
        raise ValidationError(f"unknown scenario {kind!r}; choose from {', '.join(PROFILES)}")
    if n_sessions <= 0:
        raise ValidationError("n_sessions must be positive")
    if step_minutes <= 0 or 60 % step_minutes != 0:
        raise ValidationError("step_minutes must divide 60 (e.g. 5, 15, 30, 60)")
    grid_type = None if grid is None else GridType(grid)
    if grid_type is None and (line_limit_a is not None or single_phase_share is not None):
        raise ValidationError("line_limit_a and single_phase_share need a phase-aware grid")
    if not 0.0 <= v2g_share <= 1.0:
        raise ValidationError("v2g_share must be in [0, 1]")
    prof = PROFILES[kind]
    ev_types = _ev_mix(prof.ev_types, single_phase_share)
    rng = np.random.default_rng(seed)
    start = SCENARIO_DATE.replace(hour=prof.start_hour)
    horizon = Horizon.spanning(start, 24.0, step_minutes)
    dt = horizon.dt_h
    steps_per_hour = 60 // step_minutes

    # Draw all session attributes first so the RNG stream does not depend on
    # which draws get rejected or clipped.
    arrivals_h = np.clip(
        rng.normal(prof.arrival_mean_h, prof.arrival_sd_h, n_sessions), *prof.arrival_clip
    )
    departures_h = np.array([prof.departure(rng, float(a)) for a in arrivals_h])
    energies = np.clip(
        rng.lognormal(math.log(prof.energy_median_kwh), prof.energy_sigma, n_sessions),
        *prof.energy_clip,
    )
    shares = np.array([t.share for t in ev_types])
    type_idx = rng.choice(len(ev_types), size=n_sessions, p=shares / shares.sum())
    efficiencies = rng.uniform(0.88, 0.94, n_sessions)

    windows: list[tuple[int, int]] = []
    for a_h, d_h in zip(arrivals_h, departures_h, strict=True):
        a = math.ceil(a_h * steps_per_hour - 1e-9)
        d = min(horizon.n_steps, math.floor(d_h * steps_per_hour + 1e-9))
        windows.append((a, max(d, a + 1)))
    assignment, n_chargers = _assign_chargers(windows)

    limit = grid_limit_kw if grid_limit_kw is not None else prof.grid_kw_per_session * n_sessions
    # three-phase AC chargers: 11 kW is 16 A, 22 kW is 32 A per phase
    charger_a = float(round(prof.charger_kw * 1000.0 / (3 * NOMINAL_VOLTAGE_V)))
    rotations = [ROTATIONS[c % 3] if rotate_phases else ROTATIONS[0] for c in range(n_chargers)]
    if grid_type is None:
        chargers = tuple(
            Charger(f"CP{c + 1:02d}", prof.charger_kw, MIN_POWER_3PH_KW) for c in range(n_chargers)
        )
        supply = None
    else:
        chargers = tuple(
            Charger(
                f"CP{c + 1:02d}",
                3 * NOMINAL_VOLTAGE_V * charger_a / 1000.0,
                phases=3,
                rotation=rotations[c],
                max_current_a=charger_a,
            )
            for c in range(n_chargers)
        )
        per_amp = three_phase_kw_per_a(grid_type, NOMINAL_VOLTAGE_V)
        fuse = line_limit_a if line_limit_a is not None else limit / per_amp
        supply = Supply.uniform(fuse, grid_type)

    sessions = []
    for i in range(n_sessions):
        ev = ev_types[int(type_idx[i])]
        a, d = windows[i]
        eta = float(round(efficiencies[i], 3))
        sid = f"{prof.label[:3].upper()}-{i + 1:03d}"
        if grid_type is None:
            p_max = min(ev.max_power_kw, prof.charger_kw)
            feasible = 0.95 * p_max * eta * (d - a) * dt
            sessions.append(
                Session(
                    id=sid,
                    charger_id=assignment[i],
                    arrival_step=a,
                    departure_step=d,
                    energy_kwh=float(round(min(float(energies[i]), feasible), 2)),
                    max_power_kw=ev.max_power_kw,
                    efficiency=eta,
                    min_power_kw=ev.min_power_kw,
                )
            )
            continue
        charger_index = int(assignment[i][2:]) - 1
        try:
            wires = wiring(
                grid_type,
                NOMINAL_VOLTAGE_V,
                3,
                parse_rotation(rotations[charger_index], 3, grid_type),
                ev.phases,
            )
        except WiringError as exc:
            raise ValidationError(str(exc)) from None
        p_max = wires.kw_per_a * min(ev.max_current_a, charger_a)
        feasible = 0.95 * p_max * eta * (d - a) * dt
        sessions.append(
            Session(
                id=sid,
                charger_id=assignment[i],
                arrival_step=a,
                departure_step=d,
                energy_kwh=float(round(min(float(energies[i]), feasible), 2)),
                max_power_kw=ev.phases * NOMINAL_VOLTAGE_V * ev.max_current_a / 1000.0,
                efficiency=eta,
                phases=ev.phases,
                max_current_a=ev.max_current_a,
            )
        )

    if v2g_share > 0.0:
        sessions, chargers = _with_v2g(
            sessions,
            chargers,
            [ev_types[int(i)] for i in type_idx],
            seed,
            v2g_share,
            degradation_eur_per_kwh,
        )

    base_peak = (
        base_load_peak_kw
        if base_load_peak_kw is not None
        else prof.base_peak_per_session_kw * n_sessions
    )
    if base_peak < 0:
        raise ValidationError("base_load_peak_kw must be >= 0")
    spot = day_ahead_spot_price(horizon, rng)
    tariff = Tariff(
        price_eur_per_kwh=spot + GRID_FEE_EUR_PER_KWH,
        export_price_eur_per_kwh=spot,
        demand_charge_eur_per_kw=demand_charge_eur_per_kw,
    )
    base = _base_load(horizon, base_peak, prof.base_profile, rng)
    pv = pv_profile(horizon, pv_kwp, rng)
    return Scenario(
        name=kind if grid_type is None else f"{kind}-{grid_type.value}",
        horizon=horizon,
        site=Site(grid_limit_kw=limit, chargers=chargers, supply=supply),
        tariff=tariff,
        sessions=tuple(sessions),
        base_load_kw=base,
        pv_kw=pv,
    )


def _with_v2g(
    sessions: list[Session],
    chargers: tuple[Charger, ...],
    types: list[_EVType],
    seed: int,
    share: float,
    degradation: float,
) -> tuple[list[Session], tuple[Charger, ...]]:
    """Give a share of the sessions a battery model and their chargers bidirectionality."""
    rng = np.random.default_rng([seed, 2])  # separate stream: never shifts the main draws
    chosen = rng.random(len(sessions)) < share
    fill = rng.uniform(0.3, 0.7, len(sessions))
    out: list[Session] = []
    two_way: set[str] = set()
    for s, ev, pick, soc in zip(sessions, types, chosen, fill, strict=True):
        if not pick:
            out.append(s)
            continue
        capacity = ev.battery_kwh
        initial = round(float(soc) * capacity, 2)
        ceiling = 0.9 * capacity
        v2g = V2G(
            capacity_kwh=capacity,
            initial_kwh=initial,
            min_kwh=0.2 * capacity,
            max_kwh=ceiling,
            discharge_efficiency=0.9,
            degradation_eur_per_kwh=degradation,
        )
        out.append(replace(s, energy_kwh=round(min(s.energy_kwh, ceiling - initial), 2), v2g=v2g))
        two_way.add(s.charger_id)
    return out, tuple(replace(c, bidirectional=c.id in two_way) for c in chargers)


class ScenarioOptions(TypedDict, total=False):
    """Keyword options shared by the scenario generators (see :func:`generate`)."""

    n_sessions: int
    seed: int
    grid_limit_kw: float | None
    step_minutes: int
    pv_kwp: float
    base_load_peak_kw: float | None
    demand_charge_eur_per_kw: float
    grid: GridType | str | None
    line_limit_a: float | None
    rotate_phases: bool
    single_phase_share: float | None
    v2g_share: float
    degradation_eur_per_kwh: float


def workplace(**options: Unpack[ScenarioOptions]) -> Scenario:
    """Office car park: morning arrivals, about 8 h dwell (see :func:`generate`)."""
    return generate("workplace", **options)


def depot(**options: Unpack[ScenarioOptions]) -> Scenario:
    """Fleet depot: vans return in the evening and need large energies overnight."""
    return generate("depot", **options)


def residential(**options: Unpack[ScenarioOptions]) -> Scenario:
    """Apartment-building garage: evening arrivals, departures next morning."""
    return generate("residential", **options)
