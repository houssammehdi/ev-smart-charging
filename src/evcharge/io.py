"""Load user scenarios from JSON.

The format is documented in ``docs/input-format.md``; ``examples/office.json``
is a complete example. In short::

    {
      "name": "my-site",
      "horizon": {"start": "2026-04-15T00:00:00+02:00", "hours": 24, "step_minutes": 15},
      "site": {
        "grid_limit_kw": 40,
        "chargers": [{"id": "CP1", "max_power_kw": 11, "min_power_kw": 4.14}]
      },
      "tariff": {
        "price_eur_per_kwh": {"resolution_minutes": 60, "values": [0.08, ...]},
        "export_price_eur_per_kwh": 0.0,
        "demand_charge_eur_per_kw": 0.3
      },
      "base_load_kw": {"resolution_minutes": 60, "values": [...]},
      "pv_kw": 0,
      "sessions": [
        {"id": "EV1", "charger": "CP1", "arrival": "2026-04-15T07:40:00+02:00",
         "departure": "2026-04-15T16:10:00+02:00", "energy_kwh": 12.5,
         "max_power_kw": 11, "efficiency": 0.9}
      ]
    }

Time series accept a number (constant), a list with one value per step, or an
object ``{"resolution_minutes": m, "values": [...]}`` whose resolution is a
multiple of the step (values are held constant within each interval).
Arrivals are rounded up and departures down to the step grid.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from evcharge.electrical import LINES, NOMINAL_VOLTAGE_V, GridType, Supply, WiringError
from evcharge.model import (
    DEFAULT_CURRENT_STEP_A,
    MIN_POWER_3PH_KW,
    Charger,
    FloatArray,
    Horizon,
    Scenario,
    Session,
    Site,
    Tariff,
    ValidationError,
)


def _obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationError(f"{where}: expected an object")
    return {str(k): v for k, v in value.items()}


def _num(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{where}: expected an integer, got {value!r}")
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _time(value: object, where: str) -> datetime:
    text = _str(value, where)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise ValidationError(f"{where}: not an ISO 8601 timestamp: {text!r}") from None


def _get(obj: dict[str, object], key: str, where: str) -> object:
    if key not in obj:
        raise ValidationError(f"{where}: missing required field {key!r}")
    return obj[key]


def parse_series(value: object, horizon: Horizon, where: str) -> FloatArray:
    """Convert a JSON series (number, list or resolution object) to per-step values."""
    n = horizon.n_steps
    if isinstance(value, int | float) and not isinstance(value, bool):
        return np.full(n, float(value))
    if isinstance(value, list):
        values = [_num(v, f"{where}[{i}]") for i, v in enumerate(value)]
        if len(values) != n:
            raise ValidationError(f"{where}: expected {n} per-step values, got {len(values)}")
        return np.asarray(values, dtype=np.float64)
    obj = _obj(value, where)
    resolution = _int(_get(obj, "resolution_minutes", where), f"{where}.resolution_minutes")
    if resolution <= 0 or resolution % horizon.step_minutes != 0:
        raise ValidationError(
            f"{where}.resolution_minutes ({resolution}) must be a positive multiple of the "
            f"step ({horizon.step_minutes} min)"
        )
    raw = _get(obj, "values", where)
    if not isinstance(raw, list):
        raise ValidationError(f"{where}.values: expected a list")
    values = [_num(v, f"{where}.values[{i}]") for i, v in enumerate(raw)]
    repeat = resolution // horizon.step_minutes
    if len(values) * repeat != n:
        raise ValidationError(
            f"{where}: {len(values)} values at {resolution} min cover "
            f"{len(values) * resolution} min, but the horizon is {n * horizon.step_minutes} min"
        )
    return np.repeat(np.asarray(values, dtype=np.float64), repeat)


def _opt_num(obj: dict[str, object], key: str, where: str) -> float | None:
    value = obj.get(key)
    return None if value is None else _num(value, f"{where}.{key}")


def _parse_supply(value: object) -> Supply:
    obj = _obj(value, "site.supply")
    raw = _get(obj, "line_limit_a", "site.supply")
    if isinstance(raw, list):
        if len(raw) != 3:
            raise ValidationError("site.supply.line_limit_a: expected 3 values (L1, L2, L3)")
        limits = tuple(_num(v, f"site.supply.line_limit_a[{i}]") for i, v in enumerate(raw))
    else:
        limits = (_num(raw, "site.supply.line_limit_a"),) * 3
    grid = _str(obj.get("grid", "TN"), "site.supply.grid").upper()
    if grid not in ("TN", "IT"):
        raise ValidationError(f"site.supply.grid: expected 'TN' or 'IT', got {grid!r}")
    try:
        return Supply(
            (limits[0], limits[1], limits[2]),
            GridType(grid),
            _num(obj.get("voltage_v", NOMINAL_VOLTAGE_V), "site.supply.voltage_v"),
        )
    except WiringError as exc:
        raise ValidationError(f"site.supply: {exc}") from None


def _parse_charger(raw: object, where: str, phase_aware: bool) -> Charger:
    c = _obj(raw, where)
    phases = _int(c.get("phases", 3), f"{where}.phases")
    max_current = _opt_num(c, "max_current_a", where)
    power = c.get("max_power_kw")
    if power is None and phase_aware and max_current is not None:
        max_power = phases * NOMINAL_VOLTAGE_V * max_current / 1000.0
    else:
        max_power = _num(_get(c, "max_power_kw", where), f"{where}.max_power_kw")
    rotation = c.get("rotation")
    try:
        return Charger(
            id=_str(_get(c, "id", where), f"{where}.id"),
            max_power_kw=max_power,
            min_power_kw=_num(c.get("min_power_kw", MIN_POWER_3PH_KW), f"{where}.min_power_kw"),
            phases=phases,
            rotation=None if rotation is None else _str(rotation, f"{where}.rotation"),
            max_current_a=max_current,
            min_current_a=_num(c.get("min_current_a", 6.0), f"{where}.min_current_a"),
            current_step_a=_num(
                c.get("current_step_a", DEFAULT_CURRENT_STEP_A), f"{where}.current_step_a"
            ),
        )
    except ValidationError as exc:
        raise ValidationError(f"{where}: {exc}") from None


def _parse_session(raw: object, where: str, horizon: Horizon) -> Session:
    s = _obj(raw, where)
    arrival = _time(_get(s, "arrival", where), f"{where}.arrival")
    departure = _time(_get(s, "departure", where), f"{where}.departure")
    try:
        return Session(
            id=_str(_get(s, "id", where), f"{where}.id"),
            charger_id=_str(_get(s, "charger", where), f"{where}.charger"),
            arrival_step=max(0, horizon.step_at_or_after(arrival)),
            departure_step=min(horizon.n_steps, horizon.step_at_or_before(departure)),
            energy_kwh=_num(_get(s, "energy_kwh", where), f"{where}.energy_kwh"),
            max_power_kw=_num(_get(s, "max_power_kw", where), f"{where}.max_power_kw"),
            efficiency=_num(s.get("efficiency", 0.9), f"{where}.efficiency"),
            min_power_kw=_opt_num(s, "min_power_kw", where),
            phases=_int(s.get("phases", 3), f"{where}.phases"),
            max_current_a=_opt_num(s, "max_current_a", where),
        )
    except ValidationError as exc:
        raise ValidationError(f"{where}: {exc}") from None


def _parse_line_series(value: object, horizon: Horizon, where: str) -> FloatArray:
    obj = _obj(value, where)
    unknown = sorted(set(obj) - set(LINES))
    if unknown:
        raise ValidationError(f"{where}: unknown line(s) {', '.join(unknown)}; use L1, L2, L3")
    zero = np.zeros(horizon.n_steps)
    columns = [
        parse_series(obj[line], horizon, f"{where}.{line}") if line in obj else zero
        for line in LINES
    ]
    return np.column_stack(columns)


def scenario_from_dict(data: object) -> Scenario:
    """Build a validated :class:`Scenario` from parsed JSON."""
    root = _obj(data, "scenario")
    hz = _obj(_get(root, "horizon", "scenario"), "horizon")
    horizon = Horizon.spanning(
        _time(_get(hz, "start", "horizon"), "horizon.start"),
        _num(_get(hz, "hours", "horizon"), "horizon.hours"),
        _int(hz.get("step_minutes", 15), "horizon.step_minutes"),
    )

    site_obj = _obj(_get(root, "site", "scenario"), "site")
    supply = None if site_obj.get("supply") is None else _parse_supply(site_obj["supply"])
    raw_chargers = _get(site_obj, "chargers", "site")
    if not isinstance(raw_chargers, list):
        raise ValidationError("site.chargers: expected a list")
    chargers = tuple(
        _parse_charger(raw, f"site.chargers[{i}]", supply is not None)
        for i, raw in enumerate(raw_chargers)
    )
    if site_obj.get("grid_limit_kw") is None and supply is not None:
        grid_limit = supply.fuse_equivalent_kw
    else:
        grid_limit = _num(_get(site_obj, "grid_limit_kw", "site"), "site.grid_limit_kw")
    site = Site(grid_limit_kw=grid_limit, chargers=chargers, supply=supply)

    tar = _obj(_get(root, "tariff", "scenario"), "tariff")
    export_raw = tar.get("export_price_eur_per_kwh")
    tariff = Tariff(
        price_eur_per_kwh=parse_series(
            _get(tar, "price_eur_per_kwh", "tariff"), horizon, "tariff.price_eur_per_kwh"
        ),
        export_price_eur_per_kwh=None
        if export_raw is None
        else parse_series(export_raw, horizon, "tariff.export_price_eur_per_kwh"),
        demand_charge_eur_per_kw=_num(
            tar.get("demand_charge_eur_per_kw", 0.0), "tariff.demand_charge_eur_per_kw"
        ),
    )

    raw_sessions = _get(root, "sessions", "scenario")
    if not isinstance(raw_sessions, list):
        raise ValidationError("sessions: expected a list")
    sessions = tuple(
        _parse_session(raw, f"sessions[{i}]", horizon) for i, raw in enumerate(raw_sessions)
    )

    def optional_series(key: str) -> FloatArray | None:
        raw = root.get(key)
        return None if raw is None else parse_series(raw, horizon, key)

    def optional_lines(key: str) -> FloatArray | None:
        raw = root.get(key)
        return None if raw is None else _parse_line_series(raw, horizon, key)

    return Scenario(
        name=_str(root.get("name", "custom"), "name"),
        horizon=horizon,
        site=site,
        tariff=tariff,
        sessions=sessions,
        base_load_kw=optional_series("base_load_kw"),
        pv_kw=optional_series("pv_kw"),
        base_current_a=optional_lines("base_current_a"),
        pv_current_a=optional_lines("pv_current_a"),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate a scenario JSON file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{p}: invalid JSON ({exc})") from None
    return scenario_from_dict(data)
