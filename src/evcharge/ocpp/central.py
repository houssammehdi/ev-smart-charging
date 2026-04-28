"""An OCPP 1.6-J central system that runs an evcharge policy on live sessions.

Charge points connect to ``ws://<host>:<port>/<prefix>/<chargePointId>`` with
the ``ocpp1.6`` subprotocol (the id is the last path segment, as in OCPP-J).
The central system answers BootNotification, Heartbeat, StatusNotification,
Authorize, StartTransaction, MeterValues, StopTransaction and the other
messages a charge point may send, and every ``interval_s`` it

1. describes the site *now* as a :class:`~evcharge.model.Scenario`: each
   active transaction becomes a session connected from step 0 with its
   remaining energy need and departure,
2. asks the policy for the first step (:func:`evcharge.sim.first_step`), and
3. sends each transaction a ``SetChargingProfile`` with a ``TxProfile``: a
   single period with the limit in A per phase (with ``numberPhases``) on
   phase-aware sites, or in W otherwise. Decreases are sent before increases,
   so the site limit also holds while the profiles change.

OCPP 1.6 has no field for the driver's energy need or departure time, so a
session's need is *declared* through the ``id_tags`` table of the
configuration (energy, departure clock time or dwell, phases, EV power) or
*assumed* from ``session_defaults``. Delivered energy is the metered
``Energy.Active.Import.Register`` since ``meterStart`` times the assumed
charging efficiency. See ``docs/ocpp-demo.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from websockets.asyncio.server import Server, ServerConnection
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol

from evcharge.electrical import Supply
from evcharge.io import _parse_charger, _parse_supply
from evcharge.model import (
    ENERGY_TOL_KWH,
    Charger,
    Horizon,
    Scenario,
    Session,
    Site,
    Tariff,
    ValidationError,
)
from evcharge.ocpp.framing import (
    SUBPROTOCOL,
    Call,
    CallError,
    CallResult,
    ErrorCode,
    FramingError,
    parse,
)
from evcharge.ocpp.messages import (
    CHARGE_POINT_STATUS,
    PayloadError,
    as_int,
    as_list,
    as_str,
    as_time,
    energy_register_wh,
    require,
)
from evcharge.policies import POLICY_FACTORIES, make_policy
from evcharge.sim import first_step

log = logging.getLogger(__name__)

RateUnit = Literal["A", "W"]


@dataclass(frozen=True)
class SessionDefaults:
    """What the central system assumes about a session nobody declared.

    Attributes:
        energy_kwh: Energy need into the battery.
        dwell_hours: Time from the start of the transaction to departure.
        phases: Phases the EV charges on.
        max_power_kw: EV power limit.
        efficiency: Grid-to-battery efficiency applied to metered energy.
    """

    energy_kwh: float = 20.0
    dwell_hours: float = 8.0
    phases: int = 3
    max_power_kw: float = 11.0
    efficiency: float = 0.9


@dataclass(frozen=True)
class Declared:
    """A driver's declared need, keyed by idTag in the configuration.

    Attributes:
        energy_kwh: Energy need into the battery.
        departure: Departure clock time ``"HH:MM"`` (the next one after the start).
        dwell_hours: Alternative to ``departure``.
        phases: Phases the EV charges on.
        max_power_kw: EV power limit.
    """

    energy_kwh: float | None = None
    departure: str | None = None
    dwell_hours: float | None = None
    phases: int | None = None
    max_power_kw: float | None = None


@dataclass(frozen=True)
class CentralSystemConfig:
    """Site, tariff and control settings of the central system.

    Attributes:
        grid_limit_kw: Import limit of the site (kW).
        chargers: Known connectors: evcharge chargers keyed ``"<cp>/<connector>"``.
        template: Charger used for connectors that are not listed (its id is replaced).
        supply: Phase-aware connection (``None``: the kW model).
        price_eur_per_kwh: One price, or 24 hourly prices by clock hour.
        demand_charge_eur_per_kw: Demand charge the policy may use.
        base_load_kw: Constant non-EV load behind the same meter.
        policy: Name of an evcharge policy (see ``POLICY_FACTORIES``).
        interval_s: Seconds between control steps.
        step_minutes: Step of the scenarios the policy plans on.
        horizon_hours: Longest planning horizon.
        rate_unit: Unit of the charging profiles (``None``: A on phase-aware
            sites, W otherwise).
        heartbeat_s: Heartbeat interval given in BootNotification.conf.
        defaults: Assumed session parameters.
        id_tags: Declared needs by idTag.
        accept_unknown_tags: Accept idTags that are not in ``id_tags``.
    """

    grid_limit_kw: float
    chargers: dict[str, Charger] = field(default_factory=dict)
    template: Charger = field(default_factory=lambda: Charger("template", 11.0))
    supply: Supply | None = None
    price_eur_per_kwh: tuple[float, ...] = (0.1,)
    demand_charge_eur_per_kw: float = 0.0
    base_load_kw: float = 0.0
    policy: str = "llf"
    interval_s: float = 60.0
    step_minutes: int = 15
    horizon_hours: float = 24.0
    rate_unit: RateUnit | None = None
    heartbeat_s: int = 300
    defaults: SessionDefaults = field(default_factory=SessionDefaults)
    id_tags: dict[str, Declared] = field(default_factory=dict)
    accept_unknown_tags: bool = True

    def __post_init__(self) -> None:
        if self.policy not in POLICY_FACTORIES:
            raise ValidationError(
                f"control.policy must be one of {', '.join(POLICY_FACTORIES)}, got {self.policy!r}"
            )
        if len(self.price_eur_per_kwh) not in (1, 24):
            raise ValidationError("tariff.price_eur_per_kwh needs 1 or 24 hourly values")
        if self.interval_s <= 0 or self.horizon_hours <= 0 or 60 % self.step_minutes:
            raise ValidationError("control: interval and horizon > 0, step must divide 60")

    @property
    def unit(self) -> RateUnit:
        """Charging-profile unit in use."""
        if self.rate_unit is not None:
            return self.rate_unit
        return "A" if self.supply is not None else "W"


def _num(obj: dict[str, object], key: str, default: float) -> float:
    value = obj.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{key}: expected a number, got {value!r}")
    return float(value)


def _opt(obj: dict[str, object], key: str) -> float | None:
    return None if obj.get(key) is None else _num(obj, key, 0.0)


def _section(root: dict[str, object], key: str) -> dict[str, object]:
    value = root.get(key, {})
    if not isinstance(value, dict):
        raise ValidationError(f"{key}: expected an object")
    return {str(k): v for k, v in value.items()}


def config_from_dict(data: object) -> CentralSystemConfig:
    """Build a :class:`CentralSystemConfig` from parsed JSON (see ``docs/ocpp-demo.md``)."""
    if not isinstance(data, dict):
        raise ValidationError("config: expected an object")
    root = {str(k): v for k, v in data.items()}
    site = _section(root, "site")
    supply = None if site.get("supply") is None else _parse_supply(site["supply"])
    raw_chargers = site.get("chargers", [])
    if not isinstance(raw_chargers, list):
        raise ValidationError("site.chargers: expected a list")
    chargers: dict[str, Charger] = {}
    for i, raw in enumerate(raw_chargers):
        c = _parse_charger(raw, f"site.chargers[{i}]", supply is not None)
        key = c.id if "/" in c.id else f"{c.id}/1"
        chargers[key] = replace(c, id=key)
    template_raw: dict[str, object] = {"id": "template", **_section(root, "charger_template")}
    if "max_power_kw" not in template_raw and (
        supply is None or "max_current_a" not in template_raw
    ):
        template_raw["max_power_kw"] = 22.0
    template = _parse_charger(template_raw, "charger_template", supply is not None)
    if site.get("grid_limit_kw") is None and supply is not None:
        grid_limit = supply.fuse_equivalent_kw
    else:
        grid_limit = _num(site, "grid_limit_kw", math.nan)
    tariff = _section(root, "tariff")
    prices = tariff.get("price_eur_per_kwh", 0.1)
    price_list = [
        _num({"price": p}, "price", 0.0) for p in (prices if isinstance(prices, list) else [prices])
    ]
    control = _section(root, "control")
    unit = control.get("rate_unit")
    if unit not in (None, "A", "W"):
        raise ValidationError("control.rate_unit must be 'A' or 'W'")
    d = _section(root, "session_defaults")
    defaults = SessionDefaults(
        energy_kwh=_num(d, "energy_kwh", 20.0),
        dwell_hours=_num(d, "dwell_hours", 8.0),
        phases=int(_num(d, "phases", 3)),
        max_power_kw=_num(d, "max_power_kw", 11.0),
        efficiency=_num(d, "efficiency", 0.9),
    )
    tags: dict[str, Declared] = {}
    for tag, raw in _section(root, "id_tags").items():
        if not isinstance(raw, dict):
            raise ValidationError(f"id_tags.{tag}: expected an object")
        entry = {str(k): v for k, v in raw.items()}
        departure = entry.get("departure")
        if departure is not None and not isinstance(departure, str):
            raise ValidationError(f"id_tags.{tag}.departure: expected 'HH:MM'")
        phases = _opt(entry, "phases")
        tags[tag] = Declared(
            energy_kwh=_opt(entry, "energy_kwh"),
            departure=departure,
            dwell_hours=_opt(entry, "dwell_hours"),
            phases=None if phases is None else int(phases),
            max_power_kw=_opt(entry, "max_power_kw"),
        )
    return CentralSystemConfig(
        grid_limit_kw=grid_limit,
        chargers=chargers,
        template=template,
        supply=supply,
        price_eur_per_kwh=tuple(price_list),
        demand_charge_eur_per_kw=_num(tariff, "demand_charge_eur_per_kw", 0.0),
        base_load_kw=_num(root, "base_load_kw", 0.0) if "base_load_kw" in root else 0.0,
        policy=str(control.get("policy", "llf")),
        interval_s=_num(control, "interval_s", 60.0),
        step_minutes=int(_num(control, "step_minutes", 15)),
        horizon_hours=_num(control, "horizon_hours", 24.0),
        rate_unit="A" if unit == "A" else "W" if unit == "W" else None,
        heartbeat_s=int(_num(root, "heartbeat_s", 300)),
        defaults=defaults,
        id_tags=tags,
        accept_unknown_tags=bool(root.get("accept_unknown_tags", True)),
    )


def load_config(path: str | Path) -> CentralSystemConfig:
    """Read a central-system configuration file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{p}: invalid JSON ({exc})") from None
    return config_from_dict(data)


@dataclass
class Transaction:
    """A live charging transaction and what the controller knows about it.

    Attributes:
        transaction_id: Id given in StartTransaction.conf.
        charge_point: Charge point id.
        connector: Connector id (>= 1).
        id_tag: The idTag that started it.
        started: Start time reported by the charge point.
        meter_start_wh: Energy register at the start.
        meter_wh: Latest energy register reading.
        need_kwh: Energy need into the battery (declared or assumed).
        departure: Expected departure.
        phases: Phases the EV charges on.
        max_power_kw: EV power limit.
        efficiency: Grid-to-battery efficiency.
        declared: Whether the need came from ``id_tags`` (else assumed).
        limit: Last limit sent, in the profile unit (``None``: none yet).
        profile_status: Status of the last SetChargingProfile.conf.
    """

    transaction_id: int
    charge_point: str
    connector: int
    id_tag: str
    started: datetime
    meter_start_wh: float
    meter_wh: float
    need_kwh: float
    departure: datetime
    phases: int
    max_power_kw: float
    efficiency: float
    declared: bool
    limit: float | None = None
    profile_status: str | None = None

    @property
    def delivered_kwh(self) -> float:
        """Energy delivered into the battery so far (metered energy times efficiency)."""
        return self.efficiency * max(0.0, self.meter_wh - self.meter_start_wh) / 1000.0

    @property
    def remaining_kwh(self) -> float:
        """Energy still needed."""
        return max(0.0, self.need_kwh - self.delivered_kwh)

    @property
    def session_id(self) -> str:
        """Id of the session in the controller's scenarios."""
        return f"tx{self.transaction_id}"


class _Connection:
    """One charge point's WebSocket with at most one outstanding CALL."""

    def __init__(self, ws: ServerConnection, charge_point: str, timeout_s: float) -> None:
        self.ws = ws
        self.charge_point = charge_point
        self.timeout_s = timeout_s
        self.lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Future[CallResult | CallError]] = {}

    async def call(self, action: str, payload: dict[str, object]) -> CallResult | CallError:
        """Send a CALL and wait for its answer; never two at once on this connection."""
        async with self.lock:
            message_id = str(uuid.uuid4())
            future: asyncio.Future[CallResult | CallError] = (
                asyncio.get_running_loop().create_future()
            )
            self.pending[message_id] = future
            try:
                await self.ws.send(Call(message_id, action, payload).encode())
                return await asyncio.wait_for(future, self.timeout_s)
            finally:
                self.pending.pop(message_id, None)

    def resolve(self, message: CallResult | CallError) -> None:
        future = self.pending.get(message.message_id)
        if future is None or future.done():
            log.warning("%s: answer to unknown message %s", self.charge_point, message.message_id)
            return
        future.set_result(message)


class CentralSystem:
    """OCPP 1.6-J central system with an evcharge policy in the loop.

    Args:
        config: Site, tariff and control settings.
        clock: Current time (timezone-aware); tests pass an accelerated clock.
        call_timeout_s: How long to wait for a charge point's answer.
    """

    def __init__(
        self,
        config: CentralSystemConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        call_timeout_s: float = 30.0,
    ) -> None:
        self.config = config
        self.clock = clock if clock is not None else (lambda: datetime.now().astimezone())
        self.call_timeout_s = call_timeout_s
        self.connections: dict[str, _Connection] = {}
        self.boot: dict[str, dict[str, object]] = {}
        self.status: dict[tuple[str, int], str] = {}
        self.transactions: dict[int, Transaction] = {}
        self.completed: list[Transaction] = []
        self.steps = 0
        self._next_transaction = 1
        self._wake = asyncio.Event()
        #: total EV power of every plan (kW), in step order
        self.planned_kw: list[float] = []

    # ----------------------------------------------------------- connections

    async def handler(self, ws: ServerConnection) -> None:
        """Serve one charge point connection (a ``websockets`` handler)."""
        path = "" if ws.request is None else ws.request.path.split("?", 1)[0]
        charge_point = path.rstrip("/").rsplit("/", 1)[-1]
        conn = _Connection(ws, charge_point, self.call_timeout_s)
        self.connections[charge_point] = conn
        log.info("%s connected (%s)", charge_point, ws.subprotocol)
        try:
            async for text in ws:
                reply = self.receive(conn, text)
                if reply is not None:
                    await ws.send(reply)
        except ConnectionClosed:
            pass
        finally:
            if self.connections.get(charge_point) is conn:
                del self.connections[charge_point]
            for future in conn.pending.values():
                future.cancel()
            log.info("%s disconnected", charge_point)

    def receive(self, conn: _Connection, text: str | bytes) -> str | None:
        """Handle one incoming message; return the reply to send, if any."""
        try:
            message = parse(text)
        except FramingError as exc:
            log.warning("%s: %s", conn.charge_point, exc)
            if exc.message_id is None:
                return None
            return CallError(exc.message_id, exc.code, str(exc)).encode()
        if isinstance(message, CallResult | CallError):
            conn.resolve(message)
            return None
        try:
            payload = self.dispatch(conn.charge_point, message.action, message.payload)
        except PayloadError as exc:
            return CallError(message.message_id, exc.code, str(exc)).encode()
        except NotImplementedError as exc:
            return CallError(message.message_id, ErrorCode.NOT_IMPLEMENTED, str(exc)).encode()
        return CallResult(message.message_id, payload).encode()

    # -------------------------------------------------------------- messages

    def dispatch(self, cp: str, action: str, payload: dict[str, object]) -> dict[str, object]:
        """Answer one CALL from charge point ``cp``."""
        now = self.clock()
        if action == "BootNotification":
            as_str(require(payload, "chargePointVendor"), "chargePointVendor", 20)
            as_str(require(payload, "chargePointModel"), "chargePointModel", 20)
            self.boot[cp] = payload
            return {
                "status": "Accepted",
                "currentTime": now.isoformat(),
                "interval": self.config.heartbeat_s,
            }
        if action == "Heartbeat":
            return {"currentTime": now.isoformat()}
        if action == "StatusNotification":
            connector = as_int(require(payload, "connectorId"), "connectorId")
            as_str(require(payload, "errorCode"), "errorCode")
            status = as_str(require(payload, "status"), "status")
            if status not in CHARGE_POINT_STATUS:
                raise PayloadError(
                    ErrorCode.PROPERTY_CONSTRAINT_VIOLATION, f"unknown status {status!r}"
                )
            self.status[(cp, connector)] = status
            return {}
        if action == "Authorize":
            tag = as_str(require(payload, "idTag"), "idTag", 20)
            return {"idTagInfo": {"status": self._tag_status(tag)}}
        if action == "StartTransaction":
            return self._start(cp, payload)
        if action == "MeterValues":
            self._meter(cp, payload)
            return {}
        if action == "StopTransaction":
            return self._stop(payload)
        if action in ("DiagnosticsStatusNotification", "FirmwareStatusNotification"):
            as_str(require(payload, "status"), "status")
            return {}
        if action == "DataTransfer":
            as_str(require(payload, "vendorId"), "vendorId", 255)
            return {"status": "UnknownVendorId"}
        raise NotImplementedError(f"action {action!r} is not implemented")

    def _tag_status(self, tag: str) -> str:
        known = tag in self.config.id_tags
        return "Accepted" if known or self.config.accept_unknown_tags else "Invalid"

    def _start(self, cp: str, payload: dict[str, object]) -> dict[str, object]:
        connector = as_int(require(payload, "connectorId"), "connectorId")
        if connector < 1:
            raise PayloadError(ErrorCode.PROPERTY_CONSTRAINT_VIOLATION, "connectorId must be >= 1")
        tag = as_str(require(payload, "idTag"), "idTag", 20)
        meter = as_int(require(payload, "meterStart"), "meterStart")
        started = as_time(require(payload, "timestamp"), "timestamp")
        status = self._tag_status(tag)
        tx_id = self._next_transaction
        self._next_transaction += 1
        if status != "Accepted":
            return {"transactionId": tx_id, "idTagInfo": {"status": status}}
        for old in self._on_connector(cp, connector):
            self._finish(old)
        cfg = self.config
        declared = cfg.id_tags.get(tag)
        d = cfg.defaults
        need = d.energy_kwh
        dwell = timedelta(hours=d.dwell_hours)
        departure = started + dwell
        phases, power = d.phases, d.max_power_kw
        if declared is not None:
            need = d.energy_kwh if declared.energy_kwh is None else declared.energy_kwh
            if declared.departure is not None:
                departure = _next_clock_time(started, declared.departure)
            elif declared.dwell_hours is not None:
                departure = started + timedelta(hours=declared.dwell_hours)
            phases = phases if declared.phases is None else declared.phases
            power = power if declared.max_power_kw is None else declared.max_power_kw
        self.transactions[tx_id] = Transaction(
            transaction_id=tx_id,
            charge_point=cp,
            connector=connector,
            id_tag=tag,
            started=started,
            meter_start_wh=float(meter),
            meter_wh=float(meter),
            need_kwh=need,
            departure=departure,
            phases=phases,
            max_power_kw=power,
            efficiency=d.efficiency,
            declared=declared is not None,
        )
        # plan for the new EV now rather than at the next interval
        self._wake.set()
        return {"transactionId": tx_id, "idTagInfo": {"status": "Accepted"}}

    def _meter(self, cp: str, payload: dict[str, object]) -> None:
        connector = as_int(require(payload, "connectorId"), "connectorId")
        values = as_list(require(payload, "meterValue"), "meterValue")
        tx_raw = payload.get("transactionId")
        reading = energy_register_wh(values)
        if reading is None:
            return
        tx = None
        if tx_raw is not None:
            tx = self.transactions.get(as_int(tx_raw, "transactionId"))
        if tx is None:
            tx = next(iter(self._on_connector(cp, connector)), None)
        if tx is not None:
            tx.meter_wh = max(tx.meter_wh, reading)

    def _stop(self, payload: dict[str, object]) -> dict[str, object]:
        meter = as_int(require(payload, "meterStop"), "meterStop")
        as_time(require(payload, "timestamp"), "timestamp")
        tx_id = as_int(require(payload, "transactionId"), "transactionId")
        tx = self.transactions.get(tx_id)
        if tx is not None:
            tx.meter_wh = max(tx.meter_wh, float(meter))
            self._finish(tx)
        tag = payload.get("idTag")
        if tag is None:
            return {}
        return {"idTagInfo": {"status": self._tag_status(as_str(tag, "idTag", 20))}}

    def _on_connector(self, cp: str, connector: int) -> list[Transaction]:
        where = (cp, connector)
        return [t for t in self.transactions.values() if (t.charge_point, t.connector) == where]

    def _finish(self, tx: Transaction) -> None:
        self.transactions.pop(tx.transaction_id, None)
        self.completed.append(tx)

    # --------------------------------------------------------------- control

    def charger_for(self, tx: Transaction) -> Charger:
        """The evcharge charger of a transaction's connector."""
        key = f"{tx.charge_point}/{tx.connector}"
        known = self.config.chargers.get(key)
        return known if known is not None else replace(self.config.template, id=key)

    def snapshot(self, now: datetime) -> tuple[Scenario, dict[str, Transaction]] | None:
        """The site now as a scenario (``None`` if no transaction needs energy)."""
        cfg = self.config
        active = [t for t in self.transactions.values() if t.remaining_kwh > ENERGY_TOL_KWH]
        if not active:
            return None
        step = timedelta(minutes=cfg.step_minutes)
        start = now.replace(second=0, microsecond=0)
        max_steps = max(1, int(cfg.horizon_hours * 60 // cfg.step_minutes))

        def steps_until(when: datetime) -> int:
            return min(max_steps, max(1, int((when - now) / step)))

        n_steps = max(steps_until(t.departure) for t in active)
        horizon = Horizon(start, n_steps, cfg.step_minutes)
        chargers = tuple(self.charger_for(t) for t in active)
        sessions = tuple(
            Session(
                id=t.session_id,
                charger_id=c.id,
                arrival_step=0,
                departure_step=steps_until(t.departure),
                energy_kwh=t.remaining_kwh,
                max_power_kw=t.max_power_kw,
                efficiency=t.efficiency,
                phases=min(t.phases, c.phases),
            )
            for t, c in zip(active, chargers, strict=True)
        )
        hours = np.array([(start + i * step).hour for i in range(n_steps)])
        prices = np.asarray(cfg.price_eur_per_kwh, dtype=np.float64)
        price = np.full(n_steps, prices[0]) if prices.size == 1 else prices[hours]
        sc = Scenario(
            name="ocpp-live",
            horizon=horizon,
            site=Site(cfg.grid_limit_kw, chargers, cfg.supply),
            tariff=Tariff(price, None, cfg.demand_charge_eur_per_kw),
            sessions=sessions,
            base_load_kw=np.full(n_steps, cfg.base_load_kw),
        )
        return sc, {t.session_id: t for t in active}

    def plan(self) -> dict[int, tuple[float, int]]:
        """Limits for every transaction now: transaction id to ``(limit, numberPhases)``.

        Limits are in the configured unit (A per phase or W); transactions whose
        need is met get 0.
        """
        plan: dict[int, tuple[float, int]] = {
            t.transaction_id: (0.0, t.phases) for t in self.transactions.values()
        }
        snap = self.snapshot(self.clock())
        if snap is None:
            self.planned_kw.append(0.0)
            return plan
        sc, by_session = snap
        commands, violations = first_step(sc, make_policy(self.config.policy))
        if violations:
            log.warning("policy commands corrected: %s", violations)
        total_kw = 0.0
        for s in sc.sessions:
            tx = by_session[s.id]
            ctl = sc.control(s)
            value = commands.get(s.id, 0.0)
            total_kw += value * ctl.kw_per_unit
            phases = min(tx.phases, sc.site.charger(s.charger_id).phases)
            kw = value * ctl.kw_per_unit
            limit: float
            if self.config.unit == "W":
                limit = float(round(1000.0 * kw))
            elif ctl.unit == "A":
                limit = round(value, 1)
            else:
                # kW model, profile in amperes: per-phase current at 230 V, rounded down
                limit = math.floor(10.0 * 1000.0 * kw / (230.0 * phases) + 1e-9) / 10.0
            plan[tx.transaction_id] = (limit, phases)
        self.planned_kw.append(total_kw)
        return plan

    def profile(
        self, tx: Transaction, limit: float, phases: int, now: datetime
    ) -> dict[str, object]:
        """SetChargingProfile.req payload with a one-period TxProfile."""
        period: dict[str, object] = {"startPeriod": 0, "limit": limit}
        if self.config.unit == "A":
            period["numberPhases"] = phases
        return {
            "connectorId": tx.connector,
            "csChargingProfiles": {
                "chargingProfileId": tx.transaction_id,
                "transactionId": tx.transaction_id,
                "stackLevel": 0,
                "chargingProfilePurpose": "TxProfile",
                "chargingProfileKind": "Absolute",
                "chargingSchedule": {
                    "startSchedule": now.isoformat(),
                    "chargingRateUnit": self.config.unit,
                    "chargingSchedulePeriod": [period],
                },
            },
        }

    async def control_step(self) -> None:
        """Plan and send the changed limits: decreases first, then increases."""
        plan = self.plan()
        now = self.clock()
        changes = []
        for tx_id, (limit, phases) in plan.items():
            tx = self.transactions.get(tx_id)
            if tx is None or tx.charge_point not in self.connections:
                continue
            if tx.limit is None or abs(tx.limit - limit) > 1e-9:
                changes.append((tx, limit, phases))
        down = [c for c in changes if c[0].limit is None or c[1] < c[0].limit]
        up = [c for c in changes if c not in down]
        for batch in (down, up):
            await asyncio.gather(*(self._send(tx, value, n, now) for tx, value, n in batch))
        self.steps += 1

    async def _send(self, tx: Transaction, limit: float, phases: int, now: datetime) -> None:
        conn = self.connections.get(tx.charge_point)
        if conn is None:
            return
        try:
            answer = await conn.call("SetChargingProfile", self.profile(tx, limit, phases, now))
        except (TimeoutError, ConnectionClosed, asyncio.CancelledError) as exc:
            log.warning("%s: SetChargingProfile failed: %r", tx.charge_point, exc)
            return
        if isinstance(answer, CallError):
            tx.profile_status = f"CALLERROR {answer.code}"
            return
        tx.profile_status = str(answer.payload.get("status"))
        if tx.profile_status == "Accepted":
            tx.limit = limit

    async def run_control(self, stop: asyncio.Event | None = None) -> None:
        """Run :meth:`control_step` every ``interval_s`` until ``stop`` is set.

        A new transaction triggers a step at once, so an EV does not wait up to
        an interval for its first profile.
        """
        stop = stop if stop is not None else asyncio.Event()
        while not stop.is_set():
            self._wake.clear()
            try:
                await self.control_step()
            except (ValidationError, ValueError, RuntimeError) as exc:
                log.error("control step failed: %s", exc)
            waiters = [asyncio.ensure_future(e.wait()) for e in (stop, self._wake)]
            await asyncio.wait(
                waiters, timeout=self.config.interval_s, return_when=asyncio.FIRST_COMPLETED
            )
            for w in waiters:
                w.cancel()


def _next_clock_time(after: datetime, clock: str) -> datetime:
    """The first moment after ``after`` at wall-clock time ``"HH:MM"``."""
    try:
        t = time.fromisoformat(clock)
    except ValueError:
        raise ValidationError(f"departure {clock!r} is not HH:MM") from None
    candidate = after.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    return candidate if candidate > after else candidate + timedelta(days=1)


def reject_bad_path(prefix: str) -> Callable[[ServerConnection, Request], Response | None]:
    """A ``process_request`` hook that answers 404 unless the path is ``/<prefix>/<id>``."""
    stem = "/" + prefix.strip("/") if prefix.strip("/") else ""

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        path = request.path.split("?", 1)[0]
        rest = path[len(stem) :].strip("/") if path.startswith(stem + "/") else ""
        if not rest or "/" in rest:
            return connection.respond(404, f"connect to {stem}/<chargePointId>\n")
        return None

    return process_request


async def serve(
    system: CentralSystem, host: str = "127.0.0.1", port: int = 9000, prefix: str = "ocpp"
) -> Server:
    """Start the WebSocket server of ``system`` on ``ws://host:port/<prefix>/<id>``.

    Returns the running server; close it with ``server.close()`` and
    ``await server.wait_closed()``.
    """
    return await ws_serve(
        system.handler,
        host,
        port,
        subprotocols=[Subprotocol(SUBPROTOCOL)],
        process_request=reject_bad_path(prefix),
    )
