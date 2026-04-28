"""A simulated OCPP 1.6-J charge point for tests and demos.

:class:`SimulatedChargePoint` connects to a central system, boots, reports its
connector, starts a transaction, meters the energy its EV draws under the
latest ``SetChargingProfile`` limit and stops when the EV is full or its stay
ends. Time can run faster than the wall clock (:class:`ScaledClock`), so a demo
day passes in minutes. It answers ``SetChargingProfile`` (TxProfile, A or W) and
rejects other central-system requests with ``NotImplemented``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from evcharge.ocpp.framing import (
    SUBPROTOCOL,
    Call,
    CallError,
    CallResult,
    ErrorCode,
    FramingError,
    parse,
)

NOMINAL_VOLTAGE_V = 230.0


class ScaledClock:
    """Wall-clock time sped up by ``scale``, shared by a demo's central system and charge points.

    Args:
        scale: Simulated seconds per real second.
        start: Simulated time at creation (default: now, UTC).
    """

    def __init__(self, scale: float = 1.0, start: datetime | None = None) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.scale = scale
        self.start = start if start is not None else datetime.now(UTC)
        self._t0 = time.monotonic()

    def __call__(self) -> datetime:
        """The simulated time now."""
        return self.start + timedelta(seconds=(time.monotonic() - self._t0) * self.scale)

    async def sleep(self, simulated_s: float) -> None:
        """Sleep for ``simulated_s`` simulated seconds."""
        await asyncio.sleep(simulated_s / self.scale)


@dataclass
class EV:
    """The EV plugged into the simulated charge point.

    Attributes:
        energy_kwh: Energy it takes before it is full (grid side).
        max_power_kw: Its power limit.
        phases: Phases it charges on.
        stay_s: Simulated seconds it stays plugged in.
        id_tag: RFID tag presented in Authorize and StartTransaction.
    """

    energy_kwh: float = 10.0
    max_power_kw: float = 11.0
    phases: int = 3
    stay_s: float = 8 * 3600.0
    id_tag: str = "EV-TAG"


@dataclass
class ChargePointLog:
    """What happened on the simulated charge point (for assertions and demos).

    Attributes:
        transaction_id: Id from StartTransaction.conf.
        energy_wh: Energy register now.
        power_kw: Power drawn now.
        limits: Every limit received: ``(unit, limit, numberPhases)``.
        calls_received: Actions the central system called.
        max_outstanding: Largest number of central-system CALLs seen unanswered
            at the same time (OCPP-J allows one).
    """

    transaction_id: int | None = None
    energy_wh: float = 0.0
    power_kw: float = 0.0
    limits: list[tuple[str, float, int | None]] = field(default_factory=list)
    calls_received: list[str] = field(default_factory=list)
    max_outstanding: int = 0


class SimulatedChargePoint:
    """One charge point with one connector and one EV.

    Args:
        url: Central-system URL up to the id, e.g. ``ws://127.0.0.1:9000/ocpp``.
        charge_point_id: Appended to ``url`` as the last path segment.
        ev: The EV that plugs in.
        clock: Shared simulated clock (default: real time).
        meter_interval_s: Simulated seconds between MeterValues.
        plug_in_after_s: Simulated seconds after boot before the EV plugs in.
        answer_delay_s: Wall-clock delay before answering a central-system CALL.
    """

    def __init__(
        self,
        url: str,
        charge_point_id: str,
        ev: EV | None = None,
        *,
        clock: ScaledClock | None = None,
        meter_interval_s: float = 60.0,
        plug_in_after_s: float = 0.0,
        answer_delay_s: float = 0.0,
    ) -> None:
        self.url = url.rstrip("/") + "/" + charge_point_id
        self.charge_point_id = charge_point_id
        self.ev = ev if ev is not None else EV()
        self.clock = clock if clock is not None else ScaledClock()
        self.meter_interval_s = meter_interval_s
        self.plug_in_after_s = plug_in_after_s
        self.answer_delay_s = answer_delay_s
        self.log = ChargePointLog()
        self._ws: ClientConnection | None = None
        self._pending: dict[str, asyncio.Future[CallResult | CallError]] = {}
        self._limit_kw: float = 0.0
        self._outstanding = 0

    def now(self) -> datetime:
        """Simulated time."""
        return self.clock()

    async def _sleep(self, simulated_s: float) -> None:
        await self.clock.sleep(simulated_s)

    async def call(self, action: str, payload: dict[str, object]) -> dict[str, object]:
        """Send one CALL and wait for the CALLRESULT (one outstanding CALL at a time)."""
        assert self._ws is not None
        message_id = str(uuid.uuid4())
        future: asyncio.Future[CallResult | CallError] = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        try:
            await self._ws.send(Call(message_id, action, payload).encode())
            answer = await asyncio.wait_for(future, 30.0)
        finally:
            self._pending.pop(message_id, None)
        if isinstance(answer, CallError):
            raise RuntimeError(f"{action} failed: {answer.code} {answer.description}")
        return answer.payload

    async def _receive(self) -> None:
        assert self._ws is not None
        with contextlib.suppress(ConnectionClosed):
            async for text in self._ws:
                try:
                    message = parse(text)
                except FramingError:
                    continue
                if isinstance(message, CallResult | CallError):
                    future = self._pending.get(message.message_id)
                    if future is not None and not future.done():
                        future.set_result(message)
                    continue
                asyncio.get_running_loop().create_task(self._answer(message))

    async def _answer(self, message: Call) -> None:
        assert self._ws is not None
        self._outstanding += 1
        self.log.max_outstanding = max(self.log.max_outstanding, self._outstanding)
        self.log.calls_received.append(message.action)
        try:
            if self.answer_delay_s:
                await asyncio.sleep(self.answer_delay_s)
            if message.action != "SetChargingProfile":
                reply = CallError(message.message_id, ErrorCode.NOT_IMPLEMENTED, message.action)
                await self._ws.send(reply.encode())
                return
            await self._ws.send(CallResult(message.message_id, self._apply(message)).encode())
        finally:
            self._outstanding -= 1

    def _apply(self, message: Call) -> dict[str, object]:
        profiles = message.payload.get("csChargingProfiles")
        if not isinstance(profiles, dict) or profiles.get("chargingProfilePurpose") != "TxProfile":
            return {"status": "Rejected"}
        schedule = profiles.get("chargingSchedule")
        if not isinstance(schedule, dict):
            return {"status": "Rejected"}
        periods = schedule.get("chargingSchedulePeriod")
        unit = schedule.get("chargingRateUnit")
        if not isinstance(periods, list) or not periods or unit not in ("A", "W"):
            return {"status": "Rejected"}
        first = periods[0]
        if not isinstance(first, dict):
            return {"status": "Rejected"}
        limit = float(first["limit"])
        phases_raw = first.get("numberPhases")
        phases = int(phases_raw) if isinstance(phases_raw, int) else None
        self.log.limits.append((str(unit), limit, phases))
        if unit == "A":
            n = min(self.ev.phases, phases if phases is not None else 3)
            self._limit_kw = limit * NOMINAL_VOLTAGE_V * n / 1000.0
        else:
            self._limit_kw = limit / 1000.0
        return {"status": "Accepted"}

    def _status(self, status: str) -> dict[str, object]:
        return {"connectorId": 1, "errorCode": "NoError", "status": status}

    async def run(self) -> ChargePointLog:
        """Connect, charge one EV and disconnect; return the log."""
        async with connect(self.url, subprotocols=[Subprotocol(SUBPROTOCOL)]) as ws:
            self._ws = ws
            receiver = asyncio.get_running_loop().create_task(self._receive())
            try:
                await self._session()
            finally:
                receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receiver
        return self.log

    async def _session(self) -> None:
        ev = self.ev
        await self.call(
            "BootNotification", {"chargePointVendor": "evcharge", "chargePointModel": "simulated"}
        )
        await self.call("StatusNotification", self._status("Available"))
        await self._sleep(self.plug_in_after_s)
        await self.call("StatusNotification", self._status("Preparing"))
        await self.call("Authorize", {"idTag": ev.id_tag})
        started = self.now()
        conf = await self.call(
            "StartTransaction",
            {
                "connectorId": 1,
                "idTag": ev.id_tag,
                "meterStart": 0,
                "timestamp": started.isoformat(),
            },
        )
        tx = conf.get("transactionId")
        if not isinstance(tx, int):
            raise RuntimeError("StartTransaction.conf without a transactionId")
        self.log.transaction_id = tx
        await self.call("StatusNotification", self._status("Charging"))
        leave = started + timedelta(seconds=ev.stay_s)
        last = self.now()
        while self.now() < leave:
            await self._sleep(self.meter_interval_s)
            now = self.now()
            hours = (now - last).total_seconds() / 3600.0
            last = now
            room_kwh = ev.energy_kwh - self.log.energy_wh / 1000.0
            power = max(0.0, min(self._limit_kw, ev.max_power_kw))
            if room_kwh <= 0.0:
                power = 0.0
            elif power * hours > room_kwh:
                power = room_kwh / hours
            self.log.power_kw = power
            self.log.energy_wh += 1000.0 * power * hours
            await self.call(
                "MeterValues",
                {
                    "connectorId": 1,
                    "transactionId": tx,
                    "meterValue": [
                        {
                            "timestamp": now.isoformat(),
                            "sampledValue": [
                                {
                                    "value": f"{self.log.energy_wh:.0f}",
                                    "measurand": "Energy.Active.Import.Register",
                                    "unit": "Wh",
                                },
                                {
                                    "value": f"{1000.0 * power:.0f}",
                                    "measurand": "Power.Active.Import",
                                    "unit": "W",
                                },
                            ],
                        }
                    ],
                },
            )
            if room_kwh <= 0.0:
                break
        await self.call("StatusNotification", self._status("Finishing"))
        await self.call(
            "StopTransaction",
            {
                "transactionId": tx,
                "meterStop": round(self.log.energy_wh),
                "timestamp": self.now().isoformat(),
                "reason": "EVDisconnected",
                "idTag": ev.id_tag,
            },
        )
        await self.call("StatusNotification", self._status("Available"))
