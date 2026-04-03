"""Domain model: time grid, chargers, site, charging sessions, tariff and scenario.

All quantities use SI-ish engineering units that are customary in EV charging:
power in kW, energy in kWh, prices in EUR/kWh and demand charges in EUR/kW.

Time is discretised into equal steps (default 15 minutes).  Power is modelled as
the *average* power over a step, which is also how settlement meters and most
smart-charging back-ends (OCPP ``ChargingProfile`` periods) reason about it.

Sites come in two electrical flavours. Without a :class:`~evcharge.electrical.Supply`
the site is a single aggregate kW limit and sessions are commanded in kW (the
original model). With a supply, the site is phase-aware: every line has a
current limit and AC sessions are commanded in amperes per phase, as real
chargers are; the kW model is the special case of one aggregate resource.
Either way the physics compiles to linear *rows* per step,
``sum_s a[r, s] * x[s, t] <= rhs[t, r]`` over the sessions' setpoints ``x``,
which the simulator enforces and every policy and the optimiser respect.

Every dataclass validates itself on construction and raises
:class:`ValidationError` (a :class:`ValueError`) with an explicit message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from evcharge.electrical import (
    IEC_61851_MIN_CURRENT_A,
    LINES,
    NOMINAL_VOLTAGE_V,
    GridType,
    Supply,
    Wiring,
    WiringError,
    balanced_line_current_a,
    parse_rotation,
    wiring,
)

FloatArray = npt.NDArray[np.float64]
"""One-dimensional (or two-dimensional) array of ``float64`` values."""

POWER_TOL_KW = 1e-6
"""Numerical tolerance for power comparisons (kW)."""

ENERGY_TOL_KWH = 1e-6
"""Numerical tolerance for energy comparisons (kWh)."""

CURRENT_TOL_A = 1e-6
"""Numerical tolerance for current comparisons (A)."""

NOMINAL_PHASE_VOLTAGE_V = NOMINAL_VOLTAGE_V
"""Nominal phase-to-neutral voltage of a European low-voltage grid."""

DEFAULT_CURRENT_STEP_A = 0.1
"""Default setpoint resolution of an AC charger: OCPP 1.6 limits carry one decimal."""


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


def snap_down(value: float, step: float) -> float:
    """Largest multiple of ``step`` not above ``value`` (``value`` itself if ``step`` is 0).

    Values within 1e-6 of a grid point snap to it, so solver noise such as
    ``5.9999999`` A on a 0.1 A grid gives 6.0 A rather than 5.9 A.
    """
    if step <= 0:
        return value
    k = math.floor(value / step + 1e-6)
    return round(k * step, 9)


def snap_up(value: float, step: float) -> float:
    """Smallest multiple of ``step`` not below ``value`` (``value`` itself if ``step`` is 0)."""
    if step <= 0:
        return value
    k = math.ceil(value / step - 1e-6)
    return round(k * step, 9)


def on_grid(value: float, step: float) -> bool:
    """Whether ``value`` is a multiple of ``step`` (always true for ``step == 0``)."""
    if step <= 0:
        return True
    q = value / step
    return abs(q - round(q)) <= 1e-6


@dataclass(frozen=True)
class Control:
    """How one session is commanded: setpoint unit, range and resolution.

    A setpoint ``x`` is 0 (paused), charging in ``[charge_min, charge_max]`` or,
    for bidirectional sessions, discharging with ``-x`` in
    ``[discharge_min, discharge_max]``, always on a grid of ``step``.

    Attributes:
        unit: ``"kW"`` on aggregate sites, ``"A"`` (per phase) on phase-aware sites.
        kw_per_unit: Grid power per setpoint unit (1 for kW; for example 0.69 for a
            three-phase EV on a 230/400 V TN grid, 0.23 for a single-phase one).
        charge_min: Lowest non-zero setpoint (the IEC 61851 6 A minimum).
        charge_max: Highest setpoint (charger, cable and on-board-charger limits).
        step: Setpoint resolution; 0 means continuous.
        discharge_min: Lowest non-zero discharge magnitude.
        discharge_max: Highest discharge magnitude; 0 means the session cannot discharge.
    """

    unit: str
    kw_per_unit: float
    charge_min: float
    charge_max: float
    step: float = 0.0
    discharge_min: float = 0.0
    discharge_max: float = 0.0

    @property
    def can_discharge(self) -> bool:
        """Whether negative (discharging) setpoints are allowed."""
        return self.discharge_max > 0.0

    def minimum(self, setpoint: float) -> float:
        """Lowest non-zero magnitude in the direction of ``setpoint``."""
        return self.discharge_min if setpoint < 0 else self.charge_min

    def snap_down(self, value: float) -> float:
        """Round a non-negative setpoint down to the resolution."""
        return snap_down(value, self.step)

    def snap_up(self, value: float) -> float:
        """Round a non-negative setpoint up to the resolution."""
        return snap_up(value, self.step)


@dataclass(frozen=True)
class Charger:
    """A charge point (one connector) with its controllable range.

    The kW fields describe the charger on aggregate sites. On phase-aware sites
    (``Site.supply`` set) the charger is commanded in amperes per phase and the
    phase fields apply; ``max_power_kw`` then remains an additional power cap.

    Attributes:
        id: Unique charger identifier.
        max_power_kw: Maximum charger power.
        min_power_kw: Lowest non-zero power the charger can signal on an
            aggregate site. The default is the IEC 61851 minimum of 6 A at the
            charger's phase count: about 4.14 kW for three-phase and 1.38 kW for
            single-phase chargers. A single-phase EV on a three-phase charger needs
            its own ``Session.min_power_kw`` (1.38 kW), or use a phase-aware site,
            where the minimum is a current. Use 0 for continuously adjustable
            chargers.
        phases: Phases the charger provides (1 or 3; phase-aware sites only).
        rotation: Site line of each charger conductor, e.g. ``"L2L3L1"``
            (``None``: identity). Single-phase chargers name one line on TN and two
            lines on IT grids, e.g. ``"L2"`` or ``"L2L3"``.
        max_current_a: Per-phase current limit (``None``: derived from
            ``max_power_kw``).
        min_current_a: Lowest non-zero current (IEC 61851: 6 A).
        current_step_a: Setpoint resolution in amperes (0 = continuous).
        bidirectional: Whether the charger can discharge an EV (V2G). A session
            discharges only if its charger is bidirectional and it has a
            :class:`V2G` spec.
    """

    id: str
    max_power_kw: float
    min_power_kw: float = MIN_POWER_3PH_KW
    phases: int = 3
    rotation: str | None = None
    max_current_a: float | None = None
    min_current_a: float = IEC_61851_MIN_CURRENT_A
    current_step_a: float = DEFAULT_CURRENT_STEP_A
    bidirectional: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("charger id must be a non-empty string")
        where = f"charger {self.id}"
        if self.phases == 1 and self.min_power_kw == MIN_POWER_3PH_KW:
            # the default is 6 A on three phases; on one phase 6 A is 1.38 kW
            object.__setattr__(self, "min_power_kw", MIN_POWER_1PH_KW)
        _require_finite(f"{where}: max_power_kw", self.max_power_kw)
        _require_finite(f"{where}: min_power_kw", self.min_power_kw)
        if self.max_power_kw <= 0:
            raise ValidationError(f"{where}: max_power_kw must be > 0")
        if not 0 <= self.min_power_kw <= self.max_power_kw:
            raise ValidationError(
                f"{where}: min_power_kw must be in [0, max_power_kw], "
                f"got {self.min_power_kw} (max {self.max_power_kw})"
            )
        if self.phases not in (1, 3):
            raise ValidationError(f"{where}: phases must be 1 or 3, got {self.phases}")
        if self.max_current_a is not None:
            _require_finite(f"{where}: max_current_a", self.max_current_a)
            if self.max_current_a <= 0:
                raise ValidationError(f"{where}: max_current_a must be > 0")
        for name in ("min_current_a", "current_step_a"):
            value = float(getattr(self, name))
            _require_finite(f"{where}: {name}", value)
            if value < 0:
                raise ValidationError(f"{where}: {name} must be >= 0")


@dataclass(frozen=True)
class Site:
    """A site behind one grid connection.

    Attributes:
        grid_limit_kw: Maximum import at the grid connection point (fuse or
            contractual capacity), shared by base load and EV charging. On
            phase-aware sites it is an additional aggregate power limit.
        chargers: Charge points installed at the site.
        supply: Phase-aware connection (grid type and per-line current limits).
            ``None`` gives the aggregate kW model.
        export_limit_kw: Maximum export at the connection point
            (``None``: ``grid_limit_kw``). EV discharge may not push export above
            it; PV alone never forces EVs to charge.
    """

    grid_limit_kw: float
    chargers: tuple[Charger, ...]
    supply: Supply | None = None
    export_limit_kw: float | None = None

    def __post_init__(self) -> None:
        _require_finite("grid_limit_kw", self.grid_limit_kw)
        if self.grid_limit_kw <= 0:
            raise ValidationError(f"grid_limit_kw must be > 0, got {self.grid_limit_kw}")
        if self.export_limit_kw is not None:
            _require_finite("export_limit_kw", self.export_limit_kw)
            if self.export_limit_kw < 0:
                raise ValidationError(f"export_limit_kw must be >= 0, got {self.export_limit_kw}")
        if not self.chargers:
            raise ValidationError("a site needs at least one charger")
        ids = [c.id for c in self.chargers]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValidationError(f"duplicate charger ids: {', '.join(dupes)}")
        if self.supply is not None:
            for c in self.chargers:
                self.charger_lines(c)

    def charger(self, charger_id: str) -> Charger:
        """Look up a charger by id."""
        for c in self.chargers:
            if c.id == charger_id:
                return c
        raise ValidationError(f"unknown charger id {charger_id!r}")

    @property
    def phase_aware(self) -> bool:
        """Whether the site has per-line current limits (:attr:`supply` is set)."""
        return self.supply is not None

    @property
    def bidirectional(self) -> bool:
        """Whether any charger can discharge (the site then has export rows)."""
        return any(c.bidirectional for c in self.chargers)

    @property
    def export_limit(self) -> float:
        """Export limit in kW (``export_limit_kw`` or ``grid_limit_kw``)."""
        return self.grid_limit_kw if self.export_limit_kw is None else self.export_limit_kw

    def charger_lines(self, charger: Charger) -> tuple[int, ...]:
        """Site lines of the charger's conductors (phase-aware sites only)."""
        if self.supply is None:
            raise ValidationError("the site has no phase-aware supply")
        try:
            return parse_rotation(charger.rotation, charger.phases, self.supply.grid)
        except WiringError as exc:
            raise ValidationError(f"charger {charger.id}: {exc}") from None


@dataclass(frozen=True)
class V2G:
    """Battery and bidirectional capability of one session.

    With a V2G spec the session is modelled by its battery energy: it arrives
    with ``initial_kwh``, must leave with at least ``initial_kwh + energy_kwh``
    (the session's request), and in between its energy stays within
    ``[min_kwh, max_kwh]``, which the EV's battery management enforces. It
    discharges only on a bidirectional charger.

    Attributes:
        capacity_kwh: Usable battery capacity.
        initial_kwh: Battery energy at arrival.
        min_kwh: Lowest energy the aggregator may discharge to (the driver's reserve).
        max_kwh: Highest energy while plugged in (``None``: the capacity); charging stops here.
        max_discharge_kw: Discharge power limit in kW (``None``: the charge limit).
        max_discharge_current_a: Discharge current limit per phase on
            phase-aware sites (``None``: the charge limit).
        discharge_efficiency: Battery-to-grid efficiency in ``(0, 1]``; a
            discharge of ``p`` kW for ``dt`` hours takes ``p * dt / efficiency``
            kWh out of the battery.
        degradation_eur_per_kwh: Cost per kWh of battery throughput, counted on
            both energy charged into and discharged from the battery.
    """

    capacity_kwh: float
    initial_kwh: float
    min_kwh: float = 0.0
    max_kwh: float | None = None
    max_discharge_kw: float | None = None
    max_discharge_current_a: float | None = None
    discharge_efficiency: float = 0.9
    degradation_eur_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        for name in ("capacity_kwh", "initial_kwh", "min_kwh", "discharge_efficiency"):
            _require_finite(f"v2g {name}", float(getattr(self, name)))
        _require_finite("v2g degradation_eur_per_kwh", self.degradation_eur_per_kwh)
        if self.capacity_kwh <= 0:
            raise ValidationError("v2g capacity_kwh must be > 0")
        ceiling = self.capacity_kwh if self.max_kwh is None else self.max_kwh
        _require_finite("v2g max_kwh", ceiling)
        if not 0 <= self.min_kwh <= self.initial_kwh <= ceiling <= self.capacity_kwh:
            raise ValidationError(
                "v2g needs 0 <= min_kwh <= initial_kwh <= max_kwh <= capacity_kwh, got "
                f"{self.min_kwh} <= {self.initial_kwh} <= {ceiling} <= {self.capacity_kwh}"
            )
        if not 0 < self.discharge_efficiency <= 1:
            raise ValidationError("v2g discharge_efficiency must be in (0, 1]")
        if self.degradation_eur_per_kwh < 0:
            raise ValidationError("v2g degradation_eur_per_kwh must be >= 0")
        for name in ("max_discharge_kw", "max_discharge_current_a"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(f"v2g {name}", value)
                if value <= 0:
                    raise ValidationError(f"v2g {name} must be > 0")
        object.__setattr__(self, "max_kwh", ceiling)

    @property
    def ceiling_kwh(self) -> float:
        """``max_kwh`` resolved (never ``None`` after validation)."""
        assert self.max_kwh is not None
        return self.max_kwh


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
        phases: Phases the on-board charger uses (1, 2 or 3; phase-aware sites).
        max_current_a: Per-phase current limit of the on-board charger
            (``None``: derived from ``max_power_kw``; phase-aware sites).
        v2g: Battery state and bidirectional capability (``None``: charge-only,
            and the EV stops when ``energy_kwh`` is delivered). With a spec the
            request may be 0 (leave with the arrival energy).
    """

    id: str
    charger_id: str
    arrival_step: int
    departure_step: int
    energy_kwh: float
    max_power_kw: float
    efficiency: float = 0.9
    min_power_kw: float | None = None
    phases: int = 3
    max_current_a: float | None = None
    v2g: V2G | None = None

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
        if self.v2g is None and self.energy_kwh <= 0:
            raise ValidationError(f"{where}: energy_kwh must be > 0")
        if self.v2g is not None:
            if self.energy_kwh < 0:
                raise ValidationError(f"{where}: energy_kwh must be >= 0")
            target = self.v2g.initial_kwh + self.energy_kwh
            if target > self.v2g.ceiling_kwh + ENERGY_TOL_KWH:
                raise ValidationError(
                    f"{where}: the departure target {target:.2f} kWh (arrival energy plus "
                    f"request) exceeds max_kwh {self.v2g.ceiling_kwh:.2f} kWh"
                )
        if self.max_power_kw <= 0:
            raise ValidationError(f"{where}: max_power_kw must be > 0")
        if not 0 < self.efficiency <= 1:
            raise ValidationError(f"{where}: efficiency must be in (0, 1], got {self.efficiency}")
        if self.min_power_kw is not None:
            _require_finite(f"{where}: min_power_kw", self.min_power_kw)
            if not 0 <= self.min_power_kw <= self.max_power_kw:
                raise ValidationError(f"{where}: min_power_kw must be in [0, max_power_kw]")
        if self.phases not in (1, 2, 3):
            raise ValidationError(f"{where}: phases must be 1, 2 or 3, got {self.phases}")
        if self.max_current_a is not None:
            _require_finite(f"{where}: max_current_a", self.max_current_a)
            if self.max_current_a <= 0:
                raise ValidationError(f"{where}: max_current_a must be > 0")

    @property
    def dwell_steps(self) -> int:
        """Number of steps the EV is connected."""
        return self.departure_step - self.arrival_step

    @property
    def target_kwh(self) -> float | None:
        """Battery energy required at departure for V2G sessions (``None`` otherwise)."""
        if self.v2g is None:
            return None
        return self.v2g.initial_kwh + self.energy_kwh

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
class RowModel:
    """Per-step linear constraints on the setpoints of connected sessions.

    Row ``r`` at step ``t`` reads ``sum_s a[r, s] * x[s, t] <= rhs[t, r]``, where
    ``x`` are setpoints in each session's :class:`Control` unit and ``a`` comes
    from :meth:`Scenario.row_coefficients`. On aggregate sites there is one row
    (site import in kW). Phase-aware sites add one row per line (amperes).

    Attributes:
        names: Row names, e.g. ``"L1 import"`` or ``"site import"``.
        kinds: ``"line"`` (amperes) or ``"site"`` (kW) for each row.
        rhs: Right-hand sides, shape ``(n_steps, n_rows)``, never negative.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    rhs: FloatArray

    @property
    def n_rows(self) -> int:
        """Number of rows per step."""
        return len(self.names)


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
        base_current_a: Non-EV load current per line, shape ``(n_steps, 3)``
            (phase-aware sites; default: ``base_load_kw`` as a balanced load).
        pv_current_a: PV current per line, shape ``(n_steps, 3)`` (phase-aware
            sites; default: ``pv_kw`` as a balanced three-phase inverter).
    """

    name: str
    horizon: Horizon
    site: Site
    tariff: Tariff
    sessions: tuple[Session, ...]
    base_load_kw: FloatArray | None = None
    pv_kw: FloatArray | None = None
    base_current_a: FloatArray | None = None
    pv_current_a: FloatArray | None = None
    _bounds: dict[str, tuple[float, float]] = field(init=False, repr=False)
    _controls: dict[str, Control] = field(init=False, repr=False)
    _wirings: dict[str, Wiring] = field(init=False, repr=False)
    _by_id: dict[str, Session] = field(init=False, repr=False)
    _rows: RowModel = field(init=False, repr=False)

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
        self._init_line_currents(base, pv)

        bounds: dict[str, tuple[float, float]] = {}
        controls: dict[str, Control] = {}
        wirings: dict[str, Wiring] = {}
        by_charger: dict[str, list[Session]] = {}
        for s in self.sessions:
            if s.id in bounds:
                raise ValidationError(f"duplicate session id {s.id!r}")
            if s.departure_step > n:
                raise ValidationError(
                    f"session {s.id}: departs at step {s.departure_step}, "
                    f"after the horizon end ({n})"
                )
            control, wires = self._describe(s)
            controls[s.id] = control
            if wires is not None:
                wirings[s.id] = wires
            k = control.kw_per_unit
            bounds[s.id] = (control.charge_min * k, control.charge_max * k)
            by_charger.setdefault(s.charger_id, []).append(s)
        for cid, sessions in by_charger.items():
            ordered = sorted(sessions, key=lambda s: s.arrival_step)
            for a, b in pairwise(ordered):
                if b.arrival_step < a.departure_step:
                    raise ValidationError(f"sessions {a.id} and {b.id} overlap on charger {cid}")
        object.__setattr__(self, "_bounds", bounds)
        object.__setattr__(self, "_controls", controls)
        object.__setattr__(self, "_wirings", wirings)
        object.__setattr__(self, "_by_id", {s.id: s for s in self.sessions})
        object.__setattr__(self, "_rows", self._build_rows())

    def _init_line_currents(self, base: FloatArray, pv: FloatArray) -> None:
        supply = self.site.supply
        n = self.horizon.n_steps
        if supply is None:
            if self.base_current_a is not None or self.pv_current_a is not None:
                raise ValidationError("per-line currents need a phase-aware site (Site.supply)")
            return
        for name, kw in (("base_current_a", base), ("pv_current_a", pv)):
            raw = getattr(self, name)
            if raw is None:
                per_line = balanced_line_current_a(1.0, supply.grid, supply.voltage_v)
                arr = np.repeat((kw * per_line)[:, None], 3, axis=1)
            else:
                arr = np.asarray(raw, dtype=np.float64)
                if arr.shape != (n, 3):
                    raise ValidationError(
                        f"{name} must have shape ({n}, 3) (steps x lines), got {arr.shape}"
                    )
                if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                    raise ValidationError(f"{name} must be finite and >= 0")
                arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        limits = np.array(supply.line_limit_a)
        b, v = self.base_line_current_a, self.pv_line_current_a
        if supply.grid is GridType.TN:
            imp, exp = b - v, v - b
        else:
            imp, exp = b, v
        for label, load in (("import", imp), ("export", exp)):
            over = load - limits[None, :]
            if np.any(over > CURRENT_TOL_A):
                t, line = np.unravel_index(int(np.argmax(over)), over.shape)
                raise ValidationError(
                    f"non-EV {label} current exceeds the {LINES[line]} limit at step {t} "
                    f"({self.horizon.time_of(int(t)).isoformat()}): "
                    f"{load[t, line]:.1f} A > {limits[line]:.1f} A"
                )

    def _describe(self, s: Session) -> tuple[Control, Wiring | None]:
        charger = self.site.charger(s.charger_id)
        supply = self.site.supply
        v2g = s.v2g if charger.bidirectional else None
        if supply is None:
            p_max = min(charger.max_power_kw, s.max_power_kw)
            p_min = min(charger.min_power_kw if s.min_power_kw is None else s.min_power_kw, p_max)
            d_min = d_max = 0.0
            if v2g is not None:
                cap = v2g.max_discharge_kw
                d_max = p_max if cap is None else min(p_max, cap)
                d_min = min(p_min, d_max)
            return Control("kW", 1.0, p_min, p_max, 0.0, d_min, d_max), None
        lines = self.site.charger_lines(charger)
        try:
            wires = wiring(supply.grid, supply.voltage_v, charger.phases, lines, s.phases)
        except WiringError as exc:
            raise ValidationError(f"session {s.id}: {exc}") from None
        k = wires.kw_per_a
        step = charger.current_step_a
        caps = [charger.max_power_kw / k, s.max_power_kw / k]
        caps += [c for c in (charger.max_current_a, s.max_current_a) if c is not None]
        i_max = snap_down(min(caps), step)
        i_min = charger.min_current_a
        if s.min_power_kw is not None:
            i_min = max(i_min, s.min_power_kw / k)
        i_min = snap_up(i_min, step)
        if i_max <= 0 or i_min > i_max + CURRENT_TOL_A:
            raise ValidationError(
                f"session {s.id}: the EV cannot charge at the {i_min:g} A minimum "
                f"(at most {i_max:g} A on {wires.phases} phase(s))"
            )
        d_min = d_max = 0.0
        if v2g is not None:
            d_caps = [charger.max_power_kw / k]
            d_caps += [c for c in (charger.max_current_a, v2g.max_discharge_current_a) if c]
            d_caps += [] if v2g.max_discharge_kw is None else [v2g.max_discharge_kw / k]
            d_max = snap_down(min(d_caps), step)
            d_min = i_min
            if d_max + CURRENT_TOL_A < d_min:
                raise ValidationError(
                    f"session {s.id}: the EV cannot discharge at the {d_min:g} A minimum "
                    f"(at most {d_max:g} A)"
                )
        return Control("A", k, i_min, i_max, step, d_min, d_max), wires

    def _build_rows(self) -> RowModel:
        net = self.net_base_kw
        site_import = np.maximum(0.0, self.site.grid_limit_kw - net)
        site_export = np.maximum(0.0, self.site.export_limit + net)
        supply = self.site.supply
        two_way = self.site.bidirectional
        if supply is None:
            if not two_way:
                return RowModel(("site import",), ("site",), _frozen(site_import[:, None]))
            return RowModel(
                ("site import", "site export"),
                ("site", "site"),
                _frozen(np.column_stack([site_import, site_export])),
            )
        kappa = 1.0 if supply.grid is GridType.TN else 0.0
        limits = np.array(supply.line_limit_a)[None, :]
        base, pv = self.base_line_current_a, self.pv_line_current_a
        line_import = np.maximum(0.0, limits - base + kappa * pv)
        names: tuple[str, ...] = (*(f"{line} import" for line in LINES), "site import")
        kinds: tuple[str, ...] = ("line", "line", "line", "site")
        columns = [line_import, site_import[:, None]]
        if two_way:
            line_export = np.maximum(0.0, limits - pv + kappa * base)
            names = (*names, *(f"{line} export" for line in LINES), "site export")
            kinds = (*kinds, "line", "line", "line", "site")
            columns += [line_export, site_export[:, None]]
        return RowModel(names, kinds, _frozen(np.column_stack(columns)))

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

    @property
    def base_line_current_a(self) -> FloatArray:
        """Non-EV load current per line, shape ``(n_steps, 3)`` (phase-aware sites)."""
        if self.base_current_a is None:
            raise ValidationError("the site has no phase-aware supply")
        return self.base_current_a

    @property
    def pv_line_current_a(self) -> FloatArray:
        """PV current per line, shape ``(n_steps, 3)`` (phase-aware sites)."""
        if self.pv_current_a is None:
            raise ValidationError("the site has no phase-aware supply")
        return self.pv_current_a

    @property
    def rows(self) -> RowModel:
        """Per-step constraint rows of the site (see :class:`RowModel`)."""
        return self._rows

    def _is_own(self, session: Session) -> bool:
        return self._by_id.get(session.id) == session

    def control(self, session: Session) -> Control:
        """Setpoint unit, range and resolution of a session (see :class:`Control`).

        Works for any session whose charger is on this site, for example a
        forecast EV that has not arrived.
        """
        if self._is_own(session):
            return self._controls[session.id]
        return self._describe(session)[0]

    def wiring(self, session: Session) -> Wiring | None:
        """How the session loads the site lines (``None`` on aggregate sites)."""
        if self._is_own(session):
            return self._wirings.get(session.id)
        return self._describe(session)[1]

    def row_coefficients(self, session: Session) -> FloatArray:
        """Coefficient of a charging setpoint in every row of :attr:`rows`.

        Charging never counts in export rows, so an EV that stops early can
        only lower export (see ``docs/theory.md``).
        """
        wires = self.wiring(session)
        two_way = self.site.bidirectional
        if wires is None:
            return np.array([1.0, 0.0]) if two_way else np.ones(1)
        coef = [*wires.incidence(), wires.kw_per_a]
        return np.array(coef + [0.0] * 4 if two_way else coef)

    def row_discharge_coefficients(self, session: Session) -> FloatArray:
        """Coefficient of a discharge magnitude in every row of :attr:`rows`.

        Discharge is credited against import in kW and, on TN grids, per line
        (collinear currents); on IT grids line imports get no credit.
        """
        n = self.rows.n_rows
        if not self.control(session).can_discharge:
            return np.zeros(n)
        wires = self.wiring(session)
        if wires is None:
            return np.array([-1.0, 1.0])
        assert self.site.supply is not None
        kappa = 1.0 if self.site.supply.grid is GridType.TN else 0.0
        a = list(wires.incidence())
        k = wires.kw_per_a
        return np.array([-kappa * v for v in a] + [-k] + a + [k])

    def power_bounds(self, session: Session) -> tuple[float, float]:
        """Return ``(p_min, p_max)`` in kW for a session of this scenario.

        ``p_max`` is the lower of charger and EV limits. ``p_min`` is the lowest
        non-zero power that can be signalled (EV override or charger minimum,
        never above ``p_max``). A session charges at 0 or within ``[p_min, p_max]``.
        On phase-aware sites these are the current limits times
        :attr:`Control.kw_per_unit`.
        """
        try:
            return self._bounds[session.id]
        except KeyError:
            raise ValidationError(f"session {session.id!r} is not part of this scenario") from None

    def line_currents_a(self, setpoint: FloatArray) -> FloatArray:
        """Modelled current on every line for a setpoint matrix.

        Args:
            setpoint: Setpoints in control units, shape ``(n_sessions, n_steps)``.

        Returns:
            Shape ``(n_steps, 3)``: on TN grids the magnitude of the signed sum of
            base load, PV and EV currents (exact for in-phase currents); on IT
            grids the larger of the load current and the injected current (PV and
            discharge), which bounds the true current for unity-power-factor
            devices (see ``docs/theory.md``).
        """
        supply = self.site.supply
        if supply is None:
            raise ValidationError("line currents need a phase-aware site (Site.supply)")
        x = np.asarray(setpoint, dtype=np.float64)
        if x.shape != (len(self.sessions), self.horizon.n_steps):
            raise ValidationError(
                f"setpoint must have shape ({len(self.sessions)}, {self.horizon.n_steps})"
            )
        incidence = np.array([self._wirings[s.id].incidence() for s in self.sessions]).reshape(
            len(self.sessions), 3
        )
        charge = np.maximum(x, 0.0).T @ incidence
        discharge = np.maximum(-x, 0.0).T @ incidence
        base, pv = self.base_line_current_a, self.pv_line_current_a
        if supply.grid is GridType.TN:
            return np.asarray(np.abs(base - pv + charge - discharge), dtype=np.float64)
        return np.asarray(np.maximum(base + charge, pv + discharge), dtype=np.float64)

    @property
    def energy_requested_kwh(self) -> float:
        """Total energy requested by all sessions."""
        return float(sum(s.energy_kwh for s in self.sessions))

    def without_sessions(self) -> Scenario:
        """Copy of the scenario with sessions removed (what an online policy may know)."""
        return self.with_sessions(())

    def aggregate(self) -> Scenario:
        """The kW-only view of a phase-aware scenario.

        Lines are dropped; every session keeps its power range
        (:meth:`power_bounds`) as kW limits and the site keeps its kW limit. This
        is what a controller that ignores phases believes the site to be.
        Aggregate scenarios are returned unchanged.
        """
        if self.site.supply is None:
            return self
        chargers = tuple(
            Charger(c.id, c.max_power_kw, 0.0, current_step_a=0.0, bidirectional=c.bidirectional)
            for c in self.site.chargers
        )
        sessions = []
        for s in self.sessions:
            p_min, p_max = self.power_bounds(s)
            ctl = self.control(s)
            v2g = s.v2g
            if v2g is not None and ctl.can_discharge:
                v2g = replace(
                    v2g,
                    max_discharge_kw=ctl.discharge_max * ctl.kw_per_unit,
                    max_discharge_current_a=None,
                )
            sessions.append(
                Session(
                    id=s.id,
                    charger_id=s.charger_id,
                    arrival_step=s.arrival_step,
                    departure_step=s.departure_step,
                    energy_kwh=s.energy_kwh,
                    max_power_kw=p_max,
                    efficiency=s.efficiency,
                    min_power_kw=p_min,
                    v2g=v2g,
                )
            )
        return Scenario(
            name=self.name,
            horizon=self.horizon,
            site=Site(self.site.grid_limit_kw, chargers, None, self.site.export_limit_kw),
            tariff=self.tariff,
            sessions=tuple(sessions),
            base_load_kw=self.base_load,
            pv_kw=self.pv,
        )

    def unidirectional(self) -> Scenario:
        """Copy in which no charger can discharge (unidirectional smart charging).

        V2G sessions keep their battery model (arrival energy, departure target
        and ceiling), so this is the like-for-like baseline for V2G.
        """
        chargers = tuple(replace(c, bidirectional=False) for c in self.site.chargers)
        return self.with_site(replace(self.site, chargers=chargers))

    def with_site(self, site: Site) -> Scenario:
        """Copy of the scenario on a different site (same sessions and series)."""
        return Scenario(
            name=self.name,
            horizon=self.horizon,
            site=site,
            tariff=self.tariff,
            sessions=self.sessions,
            base_load_kw=self.base_load,
            pv_kw=self.pv,
            base_current_a=self.base_current_a if site.phase_aware else None,
            pv_current_a=self.pv_current_a if site.phase_aware else None,
        )

    def with_sessions(self, sessions: tuple[Session, ...]) -> Scenario:
        """Copy of the scenario with a different set of sessions."""
        return Scenario(
            name=self.name,
            horizon=self.horizon,
            site=self.site,
            tariff=self.tariff,
            sessions=sessions,
            base_load_kw=self.base_load,
            pv_kw=self.pv,
            base_current_a=self.base_current_a if self.site.phase_aware else None,
            pv_current_a=self.pv_current_a if self.site.phase_aware else None,
        )


def _frozen(values: FloatArray) -> FloatArray:
    out = np.ascontiguousarray(values, dtype=np.float64)
    out.setflags(write=False)
    return out
