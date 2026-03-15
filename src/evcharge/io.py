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

from evcharge.model import (
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
    raw_chargers = _get(site_obj, "chargers", "site")
    if not isinstance(raw_chargers, list):
        raise ValidationError("site.chargers: expected a list")
    chargers = []
    for i, raw in enumerate(raw_chargers):
        where = f"site.chargers[{i}]"
        c = _obj(raw, where)
        chargers.append(
            Charger(
                id=_str(_get(c, "id", where), f"{where}.id"),
                max_power_kw=_num(_get(c, "max_power_kw", where), f"{where}.max_power_kw"),
                min_power_kw=_num(c.get("min_power_kw", MIN_POWER_3PH_KW), f"{where}.min_power_kw"),
            )
        )
    site = Site(
        grid_limit_kw=_num(_get(site_obj, "grid_limit_kw", "site"), "site.grid_limit_kw"),
        chargers=tuple(chargers),
    )

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
    sessions = []
    for i, raw in enumerate(raw_sessions):
        where = f"sessions[{i}]"
        s = _obj(raw, where)
        arrival = _time(_get(s, "arrival", where), f"{where}.arrival")
        departure = _time(_get(s, "departure", where), f"{where}.departure")
        min_raw = s.get("min_power_kw")
        try:
            sessions.append(
                Session(
                    id=_str(_get(s, "id", where), f"{where}.id"),
                    charger_id=_str(_get(s, "charger", where), f"{where}.charger"),
                    arrival_step=max(0, horizon.step_at_or_after(arrival)),
                    departure_step=min(horizon.n_steps, horizon.step_at_or_before(departure)),
                    energy_kwh=_num(_get(s, "energy_kwh", where), f"{where}.energy_kwh"),
                    max_power_kw=_num(_get(s, "max_power_kw", where), f"{where}.max_power_kw"),
                    efficiency=_num(s.get("efficiency", 0.9), f"{where}.efficiency"),
                    min_power_kw=None
                    if min_raw is None
                    else _num(min_raw, f"{where}.min_power_kw"),
                )
            )
        except ValidationError as exc:
            raise ValidationError(f"{where}: {exc}") from None

    def optional_series(key: str) -> FloatArray | None:
        raw = root.get(key)
        return None if raw is None else parse_series(raw, horizon, key)

    return Scenario(
        name=_str(root.get("name", "custom"), "name"),
        horizon=horizon,
        site=site,
        tariff=tariff,
        sessions=tuple(sessions),
        base_load_kw=optional_series("base_load_kw"),
        pv_kw=optional_series("pv_kw"),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate a scenario JSON file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{p}: invalid JSON ({exc})") from None
    return scenario_from_dict(data)
