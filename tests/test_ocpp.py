"""OCPP 1.6-J: framing, payload checks, the central system and an end-to-end run."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("websockets")

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus
from websockets.typing import Subprotocol

from evcharge.cli import main
from evcharge.electrical import Supply
from evcharge.model import Charger, ValidationError
from evcharge.ocpp import (
    EV,
    SUBPROTOCOL,
    Call,
    CallError,
    CallResult,
    CentralSystem,
    CentralSystemConfig,
    Declared,
    ErrorCode,
    config_from_dict,
    load_config,
    parse,
    run_demo,
    serve,
)
from evcharge.ocpp.central import _Connection, _next_clock_time
from evcharge.ocpp.framing import FramingError
from evcharge.ocpp.messages import PayloadError, energy_register_wh

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ocpp-site.json"
NOW = datetime(2026, 4, 15, 18, 0, tzinfo=UTC)
PROTOCOLS = [Subprotocol(SUBPROTOCOL)]


# ------------------------------------------------------------------ framing


def test_messages_round_trip() -> None:
    call = Call("42", "Heartbeat", {})
    assert call.encode() == '[2, "42", "Heartbeat", {}]'
    assert parse(call.encode()) == call
    result = CallResult("42", {"currentTime": "2026-04-15T18:00:00Z"})
    assert parse(result.encode()) == result
    error = CallError("42", ErrorCode.NOT_IMPLEMENTED, "no", {"a": 1})
    assert json.loads(error.encode()) == [4, "42", "NotImplemented", "no", {"a": 1}]
    assert parse(error.encode()) == error
    # unknown error codes are kept as GenericError
    assert parse('[4, "1", "Weird", "", {}]') == CallError("1", ErrorCode.GENERIC_ERROR, "")


@pytest.mark.parametrize(
    ("text", "code", "message_id"),
    [
        ("not json", ErrorCode.PROTOCOL_ERROR, None),
        ('{"a": 1}', ErrorCode.PROTOCOL_ERROR, None),
        ('[2, 7, "Heartbeat", {}]', ErrorCode.PROTOCOL_ERROR, None),
        ('[5, "1", "Heartbeat", {}]', ErrorCode.PROTOCOL_ERROR, "1"),
        ('[2, "1", "Heartbeat"]', ErrorCode.PROTOCOL_ERROR, "1"),
        ('[2, "1", "Heartbeat", []]', ErrorCode.FORMATION_VIOLATION, "1"),
        ('[3, "1", {}, 5]', ErrorCode.PROTOCOL_ERROR, "1"),
        (json.dumps([2, "x" * 37, "Heartbeat", {}]), ErrorCode.PROTOCOL_ERROR, "x" * 37),
    ],
)
def test_malformed_messages(text: str, code: ErrorCode, message_id: str | None) -> None:
    with pytest.raises(FramingError) as info:
        parse(text)
    assert info.value.code == code
    assert info.value.message_id == message_id


def test_energy_register_reading() -> None:
    def mv(*sampled: dict[str, Any]) -> list[object]:
        return [{"timestamp": "2026-04-15T18:00:00Z", "sampledValue": list(sampled)}]

    assert energy_register_wh(mv({"value": "1500"})) == 1500.0  # defaults: register, Wh
    assert energy_register_wh(mv({"value": "2.5", "unit": "kWh"})) == 2500.0
    assert energy_register_wh(mv({"value": "7000", "measurand": "Power.Active.Import"})) is None
    with pytest.raises(PayloadError, match="not a number"):
        energy_register_wh(mv({"value": "lots"}))
    with pytest.raises(PayloadError, match="energy unit"):
        energy_register_wh(mv({"value": "1", "unit": "A"}))


# ------------------------------------------------------------ central system


def system(**kwargs: Any) -> CentralSystem:
    config = CentralSystemConfig(grid_limit_kw=kwargs.pop("grid_limit_kw", 22.0), **kwargs)
    return CentralSystem(config, clock=lambda: NOW)


def answer(cs: CentralSystem, action: str, payload: dict[str, Any], cp: str = "CP1") -> Any:
    conn = _Connection(None, cp, 1.0)  # type: ignore[arg-type]
    reply = cs.receive(conn, Call("m1", action, payload).encode())
    assert reply is not None
    return json.loads(reply)


def start(cs: CentralSystem, cp: str, tag: str, when: datetime = NOW) -> int:
    reply = answer(
        cs,
        "StartTransaction",
        {"connectorId": 1, "idTag": tag, "meterStart": 1000, "timestamp": when.isoformat()},
        cp,
    )
    assert reply[0] == 3
    return int(reply[2]["transactionId"])


def test_boot_heartbeat_status_and_authorize() -> None:
    cs = system(accept_unknown_tags=False, id_tags={"KNOWN": Declared()})
    boot = answer(cs, "BootNotification", {"chargePointVendor": "V", "chargePointModel": "M"})
    assert boot[:2] == [3, "m1"]
    assert boot[2]["status"] == "Accepted"
    assert boot[2]["interval"] == 300
    assert answer(cs, "Heartbeat", {})[2] == {"currentTime": NOW.isoformat()}
    status = {"connectorId": 1, "errorCode": "NoError", "status": "Preparing"}
    assert answer(cs, "StatusNotification", status)[2] == {}
    assert cs.status[("CP1", 1)] == "Preparing"
    assert answer(cs, "Authorize", {"idTag": "KNOWN"})[2]["idTagInfo"]["status"] == "Accepted"
    assert answer(cs, "Authorize", {"idTag": "OTHER"})[2]["idTagInfo"]["status"] == "Invalid"


BOOT = {"chargePointVendor": "V", "chargePointModel": "M"}
STATUS = {"connectorId": 1, "errorCode": "NoError", "status": "Available"}
START = {"connectorId": 1, "idTag": "A", "meterStart": 0, "timestamp": "2026-04-15T18:00:00Z"}


@pytest.mark.parametrize(
    ("action", "payload", "code"),
    [
        ("BootNotification", {"chargePointVendor": "V"}, "FormationViolation"),
        ("BootNotification", {**BOOT, "chargePointVendor": 1}, "TypeConstraintViolation"),
        (
            "BootNotification",
            {**BOOT, "chargePointVendor": "V" * 21},
            "PropertyConstraintViolation",
        ),
        ("StatusNotification", {**STATUS, "status": "Busy"}, "PropertyConstraintViolation"),
        ("StartTransaction", {**START, "connectorId": 0}, "PropertyConstraintViolation"),
        ("StartTransaction", {**START, "timestamp": "soon"}, "TypeConstraintViolation"),
        ("Reset", {"type": "Soft"}, "NotImplemented"),
    ],
)
def test_bad_calls_get_the_right_error(action: str, payload: dict[str, Any], code: str) -> None:
    reply = answer(system(), action, payload)
    assert reply[:3] == [4, "m1", code]


def test_malformed_input_is_answered_only_when_the_id_is_known() -> None:
    cs = system()
    conn = _Connection(None, "CP1", 1.0)  # type: ignore[arg-type]
    assert cs.receive(conn, "garbage") is None
    reply = cs.receive(conn, '[2, "id-7", "Heartbeat"]')
    assert reply is not None
    assert json.loads(reply)[:3] == [4, "id-7", "ProtocolError"]
    # an answer to a CALL we never sent is ignored
    assert cs.receive(conn, '[3, "nobody", {}]') is None


def test_transactions_become_sessions_with_declared_or_assumed_needs() -> None:
    tags = {"ALICE": Declared(energy_kwh=30.0, departure="07:30"), "BOB": Declared(dwell_hours=2.0)}
    cs = system(id_tags=tags)
    a = start(cs, "CP1", "ALICE")
    b = start(cs, "CP2", "BOB")
    c = start(cs, "CP3", "STRANGER")
    alice, bob, stranger = (cs.transactions[t] for t in (a, b, c))
    assert (alice.need_kwh, alice.departure, alice.declared) == (
        30.0,
        datetime(2026, 4, 16, 7, 30, tzinfo=UTC),
        True,
    )
    assert bob.need_kwh == 20.0  # declared dwell, assumed energy
    assert bob.departure == NOW + timedelta(hours=2)
    assert (stranger.declared, stranger.departure) == (False, NOW + timedelta(hours=8))
    # 9 kWh metered at 0.9 efficiency is 8.1 kWh into the battery
    meter = {"value": "10000", "measurand": "Energy.Active.Import.Register", "unit": "Wh"}
    reading = {"timestamp": NOW.isoformat(), "sampledValue": [meter]}
    payload = {"connectorId": 1, "transactionId": a, "meterValue": [reading]}
    assert answer(cs, "MeterValues", payload, "CP1")[2] == {}
    assert alice.delivered_kwh == pytest.approx(8.1)
    snap = cs.snapshot(NOW)
    assert snap is not None
    sc, by_session = snap
    assert [s.energy_kwh for s in sc.sessions] == pytest.approx([30.0 - 8.1, 20.0, 20.0])
    assert [s.departure_step for s in sc.sessions] == [54, 8, 32]  # 13.5 h, 2 h, 8 h
    assert set(by_session) == {f"tx{t}" for t in (a, b, c)}
    stop = {"transactionId": b, "meterStop": 5000, "timestamp": NOW.isoformat(), "idTag": "BOB"}
    assert answer(cs, "StopTransaction", stop, "CP2")[2] == {"idTagInfo": {"status": "Accepted"}}
    assert b not in cs.transactions
    assert cs.completed[-1].meter_wh == 5000.0


def test_plan_respects_the_site_limit_in_watts_and_amperes() -> None:
    cs = system(grid_limit_kw=15.0)
    for cp in ("CP1", "CP2", "CP3"):
        start(cs, cp, "T")
    plan = cs.plan()
    assert sum(limit for limit, _ in plan.values()) <= 15000.0
    assert cs.planned_kw[-1] <= 15.0 + 1e-9
    tx = cs.transactions[1]
    profile = cs.profile(tx, *plan[1], NOW)
    schedule = profile["csChargingProfiles"]["chargingSchedule"]  # type: ignore[index]
    assert schedule["chargingRateUnit"] == "W"
    assert "numberPhases" not in schedule["chargingSchedulePeriod"][0]
    # on a phase-aware site the profile is in A per phase, with numberPhases
    tn = system(
        grid_limit_kw=50.0,
        supply=Supply.uniform(20.0),
        template=Charger("t", 11.0, max_current_a=16.0),
    )
    for cp in ("CP1", "CP2"):
        start(tn, cp, "T")
    amps = tn.plan()
    assert sum(limit for limit, _ in amps.values()) <= 20.0 + 1e-9  # one 20 A line each
    assert all(phases == 3 for _, phases in amps.values())
    period = tn.profile(tn.transactions[1], *amps[1], NOW)["csChargingProfiles"]
    first = period["chargingSchedule"]["chargingSchedulePeriod"][0]  # type: ignore[index]
    assert first["numberPhases"] == 3


def test_one_outstanding_call_per_connection() -> None:
    class Wire:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, text: str) -> None:
            self.sent.append(text)

    async def scenario() -> None:
        wire = Wire()
        conn = _Connection(wire, "CP1", 5.0)  # type: ignore[arg-type]
        first = asyncio.create_task(conn.call("SetChargingProfile", {"n": 1}))
        second = asyncio.create_task(conn.call("SetChargingProfile", {"n": 2}))
        await asyncio.sleep(0.05)
        assert len(wire.sent) == 1  # the second CALL waits for the first answer
        conn.resolve(CallResult(parse(wire.sent[0]).message_id, {"status": "Accepted"}))
        assert (await first).payload == {"status": "Accepted"}  # type: ignore[union-attr]
        await asyncio.sleep(0.05)
        assert len(wire.sent) == 2
        conn.resolve(CallResult(parse(wire.sent[1]).message_id, {"status": "Rejected"}))
        assert (await second).payload == {"status": "Rejected"}  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_configuration_file() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.unit == "A"
    assert cfg.supply is not None
    assert cfg.grid_limit_kw == pytest.approx(3 * 0.23 * 40)  # fuse-equivalent
    assert set(cfg.chargers) == {"CP001/1", "CP002/1", "CP003/1"}
    assert cfg.template.max_power_kw == pytest.approx(11.04)
    assert cfg.id_tags["BOB"].phases == 1
    assert len(cfg.price_eur_per_kwh) == 24
    with pytest.raises(ValidationError, match=r"control\.policy"):
        config_from_dict({"site": {"grid_limit_kw": 10}, "control": {"policy": "magic"}})
    with pytest.raises(ValidationError, match="rate_unit"):
        config_from_dict({"site": {"grid_limit_kw": 10}, "control": {"rate_unit": "kW"}})
    assert _next_clock_time(NOW, "07:30") == datetime(2026, 4, 16, 7, 30, tzinfo=UTC)
    assert _next_clock_time(NOW, "19:00") == datetime(2026, 4, 15, 19, 0, tzinfo=UTC)


# -------------------------------------------------------------- end to end


def test_charge_points_share_the_site_limit_end_to_end() -> None:
    fleet = [
        EV(energy_kwh=6.0, max_power_kw=11.0, stay_s=2 * 3600.0, id_tag=f"T{i}") for i in range(3)
    ]
    config = CentralSystemConfig(grid_limit_kw=15.0, policy="edf", interval_s=0.15)
    result = asyncio.run(
        run_demo(config, fleet, scale=3600.0, arrival_gap_s=300.0, meter_interval_s=120.0)
    )
    assert result.delivered_kwh() == pytest.approx([6.0, 6.0, 6.0], abs=0.01)
    assert result.control_steps >= 3
    assert max(result.planned_kw) <= 15.0 + 1e-9
    assert result.max_site_kw <= 15.0 + 1e-6
    for _, log in result.logs:
        assert log.transaction_id is not None
        assert log.limits  # every EV was controlled
        assert all(unit == "W" for unit, _, _ in log.limits)
        assert log.max_outstanding == 1
        assert set(log.calls_received) == {"SetChargingProfile"}


def test_server_negotiates_ocpp16_and_rejects_bad_requests() -> None:
    async def scenario() -> None:
        cs = system()
        server = await serve(cs, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            url = f"ws://127.0.0.1:{port}"
            async with connect(f"{url}/ocpp/CP9", subprotocols=PROTOCOLS) as ws:
                assert ws.subprotocol == "ocpp1.6"
                await ws.send(Call("b1", "BootNotification", BOOT).encode())
                reply = parse(await ws.recv())
                assert isinstance(reply, CallResult)
                assert reply.message_id == "b1"
                assert "CP9" in cs.boot
            with pytest.raises(InvalidStatus):
                async with connect(f"{url}/ocpp/CP9"):
                    pass  # no ocpp1.6 subprotocol
            with pytest.raises(InvalidStatus, match="404"):
                async with connect(f"{url}/other", subprotocols=PROTOCOLS):
                    pass
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_cli_demo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ocpp-demo", "--chargers", "2", "--grid-limit", "11", "--scale", "7200"]) == 0
    out = capsys.readouterr().out
    assert "2 charge points, site limit 11 kW" in out
    assert "CP002" in out
