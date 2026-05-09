"""Load charging sessions in the Caltech ACN-Data JSON format.

ACN-Data (Lee, Li and Low 2019) publishes the sessions of the Adaptive
Charging Network sites as JSON: an object whose ``_items`` list holds one
record per session, with fields such as ``sessionID``, ``stationID``,
``connectionTime``, ``disconnectTime``, ``doneChargingTime``, ``kWhDelivered``
and ``userInputs`` (the driver's ``kWhRequested`` and ``requestedDeparture``,
possibly several entries, the last one being the latest). Timestamps are
RFC 1123 strings such as ``"Wed, 25 Apr 2018 11:08:04 GMT"``; ISO 8601 is
accepted too. This module only reads files: it downloads nothing.

The records become sessions of a :class:`~evcharge.model.Scenario` on the
aggregate kW model: one charger per ``stationID``, the energy of each session
is ``kWhDelivered`` (metered at the station, so ``efficiency=1``) or the
driver's ``kWhRequested``, and the departure is the actual ``disconnectTime``
or the driver's ``requestedDeparture``. ``examples/acn-sample.json`` is a small
hand-made file in this format.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from evcharge.model import Charger, Horizon, Scenario, Session, Site, Tariff, ValidationError

EnergySource = Literal["delivered", "requested"]
DepartureSource = Literal["actual", "requested"]


@dataclass(frozen=True)
class AcnRecord:
    """One ACN-Data session record, with the fields the loader uses.

    Attributes:
        session_id: ``sessionID``.
        station_id: ``stationID`` (one charger per station).
        connection: ``connectionTime``.
        disconnect: ``disconnectTime``.
        done_charging: ``doneChargingTime`` (``None`` if missing).
        kwh_delivered: ``kWhDelivered``.
        kwh_requested: ``kWhRequested`` of the latest user input (``None`` if none).
        requested_departure: ``requestedDeparture`` of the latest user input.
    """

    session_id: str
    station_id: str
    connection: datetime
    disconnect: datetime
    done_charging: datetime | None
    kwh_delivered: float
    kwh_requested: float | None
    requested_departure: datetime | None


def _time(value: object, where: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{where}: expected a timestamp string, got {value!r}")
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValidationError(f"{where}: cannot parse timestamp {value!r}") from None
    if when.tzinfo is None:
        raise ValidationError(f"{where}: timestamp {value!r} has no time zone")
    return when


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValidationError(f"{where}: expected a number, got {value!r}")
    return float(value)


def parse_records(data: object) -> list[AcnRecord]:
    """Read the records of a parsed ACN-Data document (an object with ``_items``, or a list)."""
    items = data.get("_items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValidationError("ACN-Data: expected a list of sessions or an object with '_items'")
    records = []
    for i, raw in enumerate(items):
        where = f"_items[{i}]"
        if not isinstance(raw, dict):
            raise ValidationError(f"{where}: expected an object")
        for key in ("sessionID", "stationID", "connectionTime", "disconnectTime", "kWhDelivered"):
            if raw.get(key) is None:
                raise ValidationError(f"{where}: missing {key!r}")
        inputs = raw.get("userInputs") or []
        if not isinstance(inputs, list):
            raise ValidationError(f"{where}.userInputs: expected a list")
        latest = inputs[-1] if inputs else {}
        if not isinstance(latest, dict):
            raise ValidationError(f"{where}.userInputs: expected objects")
        requested = latest.get("kWhRequested")
        departure = latest.get("requestedDeparture")
        done = raw.get("doneChargingTime")
        records.append(
            AcnRecord(
                session_id=str(raw["sessionID"]),
                station_id=str(raw["stationID"]),
                connection=_time(raw["connectionTime"], f"{where}.connectionTime"),
                disconnect=_time(raw["disconnectTime"], f"{where}.disconnectTime"),
                done_charging=None if done is None else _time(done, f"{where}.doneChargingTime"),
                kwh_delivered=_number(raw["kWhDelivered"], f"{where}.kWhDelivered"),
                kwh_requested=None
                if requested is None
                else _number(requested, f"{where}.userInputs.kWhRequested"),
                requested_departure=None
                if departure is None
                else _time(departure, f"{where}.userInputs.requestedDeparture"),
            )
        )
    return records


@dataclass(frozen=True)
class AcnLoad:
    """A scenario built from ACN-Data and what the loader had to drop or change.

    Attributes:
        scenario: The scenario.
        dropped: Session ids left out (window shorter than one step, or no energy).
        capped: Session ids whose energy was capped at what fits their rounded window.
    """

    scenario: Scenario
    dropped: tuple[str, ...]
    capped: tuple[str, ...]


def acn_scenario(
    records: list[AcnRecord],
    *,
    grid_limit_kw: float,
    step_minutes: int = 15,
    charger_kw: float = 6.6,
    charger_min_kw: float = 1.25,
    price_eur_per_kwh: float = 0.1,
    demand_charge_eur_per_kw: float = 0.0,
    energy: EnergySource = "delivered",
    departure: DepartureSource = "actual",
    name: str = "acn",
) -> AcnLoad:
    """Build a scenario from ACN records.

    Arrivals are rounded up and departures down to the step grid (as in the
    JSON loader). The horizon runs from the first arrival's step to the last
    departure's. With ``energy="requested"`` or ``departure="requested"``,
    records without user input fall back to the delivered energy and the
    actual departure; a requested departure later than the actual one is cut
    at the actual one, since the EV is gone.

    Args:
        records: Parsed records (see :func:`parse_records`).
        grid_limit_kw: Site import limit.
        step_minutes: Control step.
        charger_kw: Power limit of every station (ACN stations deliver up to
            32 A at 208 V, about 6.6 kW).
        charger_min_kw: Lowest non-zero power (default: 6 A at 208 V).
        price_eur_per_kwh: Constant import price.
        demand_charge_eur_per_kw: Demand charge on the horizon's peak.
        energy: Session energy from ``kWhDelivered`` or the driver's ``kWhRequested``.
        departure: Departure from ``disconnectTime`` or the driver's ``requestedDeparture``.
        name: Scenario name.

    Raises:
        ValidationError: if no session is left.
    """
    if not records:
        raise ValidationError("ACN-Data: no sessions")
    step = timedelta(minutes=step_minutes)
    first = min(r.connection for r in records)
    midnight = first.replace(hour=0, minute=0, second=0, microsecond=0)
    start = first - (first - midnight) % step

    def departure_of(r: AcnRecord) -> datetime:
        if departure == "requested" and r.requested_departure is not None:
            return min(r.requested_departure, r.disconnect)
        return r.disconnect

    def energy_of(r: AcnRecord) -> float:
        if energy == "requested" and r.kwh_requested is not None:
            return r.kwh_requested
        return r.kwh_delivered

    last = max(departure_of(r) for r in records)
    n_steps = max(1, math.ceil((last - start) / step))
    horizon = Horizon(start, n_steps, step_minutes)
    sessions: list[Session] = []
    dropped: list[str] = []
    capped: list[str] = []
    stations = sorted({r.station_id for r in records})
    for r in sorted(records, key=lambda rec: (rec.connection, rec.session_id)):
        a = horizon.step_at_or_after(r.connection)
        d = min(n_steps, horizon.step_at_or_before(departure_of(r)))
        kwh = energy_of(r)
        if d <= a or kwh <= 0.0:
            dropped.append(r.session_id)
            continue
        fits = charger_kw * (d - a) * horizon.dt_h
        if kwh > fits:
            capped.append(r.session_id)
            kwh = fits
        sessions.append(
            Session(
                id=r.session_id,
                charger_id=r.station_id,
                arrival_step=a,
                departure_step=d,
                energy_kwh=round(kwh, 4),
                max_power_kw=charger_kw,
                efficiency=1.0,
            )
        )
    if not sessions:
        raise ValidationError("ACN-Data: every session is shorter than one step")
    chargers = tuple(Charger(sid, charger_kw, min(charger_min_kw, charger_kw)) for sid in stations)
    scenario = Scenario(
        name=name,
        horizon=horizon,
        site=Site(grid_limit_kw, chargers),
        tariff=Tariff(np.full(n_steps, price_eur_per_kwh), None, demand_charge_eur_per_kw),
        sessions=tuple(sessions),
    )
    return AcnLoad(scenario, tuple(dropped), tuple(capped))


def load_acn(path: str | Path, **options: Any) -> AcnLoad:
    """Read an ACN-Data JSON file and build a scenario (options of :func:`acn_scenario`)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{p}: invalid JSON ({exc})") from None
    return acn_scenario(parse_records(data), **options)
