# OCPP 1.6-J central system

`evcharge.ocpp` runs any `evcharge` policy against real (or simulated) charge points. It is an
OCPP 1.6-J central system: charge points connect over WebSocket, report their transactions and
meter values, and receive a `SetChargingProfile` every control interval. It needs the `ocpp`
extra:

```bash
pip install -e ".[ocpp]"            # adds websockets
evcharge ocpp-demo --chargers 8 --grid-limit 22 --scale 1200
evcharge ocpp-server --config examples/ocpp-site.json --port 9000
```

## What happens on the wire

Charge points connect to `ws://<host>:<port>/ocpp/<chargePointId>` and must offer the
`ocpp1.6` subprotocol; the id is the last path segment, as OCPP-J specifies. Other paths get
HTTP 404, and a client without the subprotocol is refused during the handshake.

Every message is an OCPP-J array: CALL `[2, id, action, payload]`, CALLRESULT
`[3, id, payload]` or CALLERROR `[4, id, code, description, details]`. The central system

- answers `BootNotification` (Accepted, with the heartbeat interval), `Heartbeat`,
  `StatusNotification`, `Authorize`, `StartTransaction`, `MeterValues`, `StopTransaction`,
  `DataTransfer` (UnknownVendorId) and the two status notifications; any other action gets
  `NotImplemented`;
- answers a malformed message with the CALLERROR the specification prescribes when the message
  id can be read (`ProtocolError` for a broken frame, `FormationViolation` for a missing field,
  `TypeConstraintViolation` for a wrong type, `PropertyConstraintViolation` for a value out of
  range or a string over its maximum length), and otherwise drops it;
- has at most **one outstanding CALL per connection**: its `SetChargingProfile` requests to a
  charge point are serialised, and each waits for its CALLRESULT or CALLERROR (30 s timeout).

## The control loop

Every `control.interval_s` seconds, and at once after a `StartTransaction`:

1. Each active transaction becomes a `Session` connected from step 0, with its remaining energy
   need and its departure (see below), on the charger of its connector.
2. The site as it is now becomes a `Scenario`: the configured grid limit or supply (per-line
   fuses on phase-aware sites), the tariff from now on and the base load.
3. The policy's first step is computed with `evcharge.sim.first_step`, which applies the same
   checks as the simulator (so the property tests about limits carry over).
4. Each transaction whose limit changed receives a `SetChargingProfile` with a `TxProfile`
   (`chargingProfileKind: Absolute`, one period starting now). The limit is in **A per phase
   with `numberPhases`** on phase-aware sites (0.1 A resolution by default) and in **W**
   otherwise; `control.rate_unit` overrides that. Decreases are sent before increases, so the
   site limit also holds while the profiles change.

A transaction that has met its need gets 0 (the charger pauses). A charge point whose profile is
rejected keeps its last accepted limit in the controller's books.

## Where the energy need and the departure come from

OCPP 1.6 has no message for the driver's energy need or departure time. The central system
therefore uses, per idTag:

- **Declared** needs from the configuration's `id_tags` table: `energy_kwh`, `departure` as a
  clock time `"HH:MM"` (the next one after the start) or `dwell_hours`, and optionally `phases`
  and `max_power_kw`. In a deployment these would come from an app or a booking system.
- **Assumed** needs from `session_defaults` for every other idTag (20 kWh, 8 hours, three-phase
  11 kW unless configured).

Delivered energy is the metered `Energy.Active.Import.Register` since `meterStart` times the
assumed charging efficiency (`session_defaults.efficiency`, 0.9). Timestamps without a UTC
offset are read as UTC. The `Transaction` objects record whether a need was declared or assumed.

## Configuration

[`examples/ocpp-site.json`](../examples/ocpp-site.json) is a complete example: a TN garage with
40 A per line, three listed chargers with rotated phases, a template for unlisted connectors
(`"<chargePointId>/<connectorId>"` keys), hourly prices, a 3 kW base load, LLF every 60 s and
two declared drivers. The `site` block uses the scenario format of
[input-format.md](input-format.md) (`supply`, `chargers[]`); `grid_limit_kw` defaults to the
fuse-equivalent power. `control.policy` is any policy of `evcharge compare` except the
forecast-aware ones, which need a history of past days.

## The demo

`evcharge ocpp-demo` runs the central system and one simulated charge point per EV in one
process, on a shared clock that runs `--scale` times faster than real time. The EVs (seeded
needs of 8 to 30 kWh, stays of 3 to 8 hours) plug in 15 simulated minutes apart and declare
their needs; each charge point meters its EV every 5 simulated minutes, obeys its latest
profile and stops the transaction when the EV is full or leaves. It reports the highest planned
and sampled site power and the energy each EV received:

```text
$ evcharge ocpp-demo --chargers 8 --grid-limit 22 --scale 1200
8 charge points, site limit 22 kW, policy llf, 31 control steps
highest planned site power: 22.0 kW
highest sampled site power: 22.0 kW
charge point   need kWh  stay h  got kWh  profiles
CP001              19.3     7.8     19.3         9
CP002              11.2     7.7     11.2         7
CP003              14.9     5.1     14.9         8
CP004              26.2     5.0     26.2        10
CP005              20.1     3.1     20.1         3
CP006              24.6     5.7     24.6        10
CP007              15.3     6.9     15.3         8
CP008              14.7     5.3     14.7         7
```

The run takes about 25 s of wall time. The number of control steps depends on task scheduling
(31 and 34 in two runs); the energies did not change. The simulated charge points start each transaction at
0 W until the first profile arrives (as a charger configured with a zero default profile
would); the controller plans at once after `StartTransaction`.

The integration tests (`tests/test_ocpp.py`) use the same in-repo simulated charge point, so CI
needs no external tools: three EVs sharing a 15 kW site all finish, no plan and no sampled site
power exceeds the limit, every charge point sees at most one outstanding CALL, the server
negotiates `ocpp1.6` and refuses clients without it.

## Load testing with ocpp-kit

The simulator of [ocpp-kit](https://github.com/houssammehdi/ocpp-kit) can open many charge
point connections against the same server, for example

```bash
evcharge ocpp-server --config examples/ocpp-site.json --port 9000
node dist/cli/main.js sim --url ws://localhost:9000/ocpp --count 20 --ramp 5/s
```

It is not a dependency and is not used in CI, and this combination has not been run as part of
this repository's checks. The server expects the charge point id as the last path segment of
the URL.

## Limitations

- **Not a production CSMS.** No TLS or OCPP security profiles, no authorisation lists or cache,
  no persistence (transactions live in memory), no `RemoteStart/Stop`, `Reset`,
  `ChangeConfiguration`, `TriggerMessage` or `GetCompositeSchedule`, and no reservation.
- **Single-period TxProfiles.** Each profile holds the current step's limit only; the plan
  beyond the first step is not sent, so a charge point that loses its connection keeps its last
  limit.
- **No site meter.** The controller does not measure the site: the base load is the configured
  constant and the demand charge is priced against peaks within the planning horizon only
  (the peak already incurred is taken as 0).
- **Trusts the charge points.** Meter readings and timestamps are used as reported. An EV that
  draws less than its limit leaves that capacity unused until the next step.
