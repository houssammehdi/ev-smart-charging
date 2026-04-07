# Scenario JSON format

`evcharge run --input FILE` and `evcharge plot --input FILE` read a site, a tariff and a list of
charging sessions from a JSON file. [`examples/office.json`](../examples/office.json) is a complete,
working example (6 chargers, 8 sessions, hourly prices, base load and PV).

```bash
evcharge run --input examples/office.json
evcharge plot --input examples/office.json --policies uncontrolled mpc optimal --output office.png
```

## Top level

| field           | type          | required | meaning                                                   |
|-----------------|---------------|----------|-----------------------------------------------------------|
| `name`          | string        | no       | label used in the output (default `custom`)               |
| `horizon`       | object        | yes      | simulated period and control step                         |
| `site`          | object        | yes      | grid connection limit and chargers                        |
| `tariff`        | object        | yes      | import/export prices and demand charge                    |
| `sessions`      | array         | yes      | charging sessions                                         |
| `base_load_kw`  | series        | no       | non-EV consumption behind the same meter (default 0)      |
| `pv_kw`         | series        | no       | on-site PV production (default 0)                         |

## `horizon`

| field          | type   | default | meaning                                                         |
|----------------|--------|---------|-----------------------------------------------------------------|
| `start`        | string | -       | ISO 8601 timestamp, naive or with offset (`2026-04-15T00:00:00+02:00`) |
| `hours`        | number | -       | length; must be a whole number of steps                         |
| `step_minutes` | int    | 15      | control step                                                    |

All session timestamps must use the same convention as `start` (all naive or all with an offset).

## `site`

| field                      | type   | default | meaning                                                  |
|----------------------------|--------|---------|----------------------------------------------------------|
| `grid_limit_kw`            | number | -       | maximum import at the connection point (fuse/contract)   |
| `chargers[].id`            | string | -       | unique charger id                                        |
| `chargers[].max_power_kw`  | number | -       | maximum charger power                                    |
| `chargers[].min_power_kw`  | number | 4.14 (1.38 if `phases` is 1) | lowest non-zero power on aggregate sites: 6 A x 230 V x phases per IEC 61851. Use 0 for chargers without a minimum |

Bidirectional chargers and the export limit are described under
[Bidirectional (V2G) sessions](#bidirectional-v2g-sessions).

## Phase-aware sites

Add a `supply` block to `site` and the site is modelled per line, with setpoints in amperes per
phase (see [phases.md](phases.md)):

```jsonc
"site": {
  "grid_limit_kw": 40,                      // optional here: defaults to the fuse-equivalent power
  "supply": {"grid": "IT", "line_limit_a": 80, "voltage_v": 230},
  "chargers": [
    {"id": "P1", "phases": 3, "rotation": "L2L3L1", "max_current_a": 32, "current_step_a": 1},
    {"id": "P2", "phases": 1, "rotation": "L3L1", "max_current_a": 16}
  ]
}
```

| field                        | type   | default | meaning                                               |
|------------------------------|--------|---------|-------------------------------------------------------|
| `supply.grid`                | string | `TN`    | `TN` (230/400 V with neutral) or `IT` (230 V, no neutral) |
| `supply.line_limit_a`        | number or 3 numbers | - | current limit of L1, L2, L3 (main fuse, breaker or contract) |
| `supply.voltage_v`           | number | 230     | voltage across a single-phase load: line-to-neutral (TN) or line-to-line (IT) |
| `chargers[].phases`          | int    | 3       | 1 or 3                                                |
| `chargers[].rotation`        | string | identity | site line of each charger conductor: `L2L3L1`, or OCPP `STR`; single-phase chargers name one line (TN) or two (IT) |
| `chargers[].max_current_a`   | number | from `max_power_kw` | per-phase current limit                   |
| `chargers[].min_current_a`   | number | 6       | lowest non-zero current (IEC 61851)                   |
| `chargers[].current_step_a`  | number | 0.1     | setpoint resolution (OCPP 1.6 limits carry one decimal); 0 = continuous |
| `chargers[].max_power_kw`    | number | phases x 230 V x `max_current_a` | an additional power cap          |

With a `supply`, the top level also accepts per-line currents. Lines that are left out carry 0 A.

```jsonc
"base_current_a": {"L1": <series>, "L2": <series>, "L3": <series>},   // non-EV load per line
"pv_current_a":   {"L1": <series>, "L2": <series>, "L3": <series>}    // PV current per line
```

Without them, `base_load_kw` and `pv_kw` are converted to balanced three-phase currents at
unity power factor. The kW series are always used for energy cost and the peak. The current
series are used for the line limits. [`examples/garage-it.json`](../examples/garage-it.json) is
a complete example.

## `tariff`

| field                      | type   | default | meaning                                                 |
|----------------------------|--------|---------|---------------------------------------------------------|
| `price_eur_per_kwh`        | series | -       | import price (energy + grid energy fee + taxes)         |
| `export_price_eur_per_kwh` | series | 0       | price paid for export; must not exceed the import price |
| `demand_charge_eur_per_kw` | number | 0       | charge on the highest step-average import of the horizon; pro-rate monthly tariffs |

## `sessions[]`

| field          | type   | default          | meaning                                                   |
|----------------|--------|------------------|-----------------------------------------------------------|
| `id`           | string | -                | unique session id                                         |
| `charger`      | string | -                | charger id; sessions on one charger must not overlap      |
| `arrival`      | string | -                | plug-in time; **rounded up** to the next step boundary     |
| `departure`    | string | -                | plug-out time; **rounded down** to the previous boundary   |
| `energy_kwh`   | number | -                | energy requested, measured into the battery               |
| `max_power_kw` | number | -                | EV limit on this connection (e.g. 3.7 for a single-phase 16 A car) |
| `efficiency`   | number | 0.9              | grid-to-battery efficiency in (0, 1]                      |
| `min_power_kw` | number | charger minimum  | EV-specific minimum, e.g. 1.38 for a single-phase car at 6 A |
| `phases`       | int    | 3                | phases the on-board charger uses (phase-aware sites) |
| `max_current_a`| number | from `max_power_kw` | per-phase current limit of the on-board charger (phase-aware sites) |

On an aggregate site the charger's 4.14 kW default minimum is the 6 A minimum of a
*three-phase* EV. A single-phase EV on such a charger can modulate from 1.38 kW, so give it
`"min_power_kw": 1.38`. Otherwise it is treated as an on/off load at its maximum (the minimum is
capped at the EV's maximum). Phase-aware sites have no such caveat: there the minimum is a
current.

The rounding is conservative: a session is only scheduled in steps during which the EV is plugged
in for the whole step. A session shorter than one step after rounding is rejected with an error.

## Bidirectional (V2G) sessions

A session discharges only if its charger is bidirectional **and** it carries a `v2g` block,
which describes the battery (see [v2g.md](v2g.md) for the model and a study):

```jsonc
"site": {
  "grid_limit_kw": 40,
  "export_limit_kw": 20,                                   // optional, default grid_limit_kw
  "chargers": [{"id": "CP1", "max_power_kw": 11, "bidirectional": true}]
},
"sessions": [
  {"id": "EV1", "charger": "CP1", "arrival": "...", "departure": "...",
   "energy_kwh": 10, "max_power_kw": 11,
   "v2g": {"capacity_kwh": 64, "initial_kwh": 30, "min_kwh": 13, "max_kwh": 58,
           "max_discharge_kw": 7.4, "discharge_efficiency": 0.9,
           "degradation_eur_per_kwh": 0.04}}
]
```

| field                        | type   | default | meaning                                                |
|------------------------------|--------|---------|--------------------------------------------------------|
| `site.export_limit_kw`       | number | `grid_limit_kw` | maximum export at the connection point; EV discharge may not push export above it |
| `chargers[].bidirectional`   | bool   | false   | the charger can discharge an EV                        |
| `v2g.capacity_kwh`           | number | -       | usable battery capacity                                |
| `v2g.initial_kwh`            | number | -       | battery energy at arrival                              |
| `v2g.min_kwh`                | number | 0       | floor the aggregator may discharge to (the driver's reserve) |
| `v2g.max_kwh`                | number | `capacity_kwh` | ceiling while plugged in; charging stops there   |
| `v2g.max_discharge_kw`       | number | charge limit | discharge power limit                             |
| `v2g.max_discharge_current_a`| number | charge limit | discharge current limit per phase (phase-aware sites) |
| `v2g.discharge_efficiency`   | number | 0.9     | battery-to-grid efficiency in (0, 1]                   |
| `v2g.degradation_eur_per_kwh`| number | 0       | wear cost per kWh of battery throughput, both directions |

With a `v2g` block, `energy_kwh` is the net energy to add: the EV must leave with at least
`initial_kwh + energy_kwh`, which must not exceed `max_kwh`, and `energy_kwh` may be 0. The
discharge minimum equals the charging minimum (6 A, or `min_power_kw`). The session's
`efficiency` applies to charging, `discharge_efficiency` to discharging.

## Series

Every time series (`price_eur_per_kwh`, `export_price_eur_per_kwh`, `base_load_kw`, `pv_kw`)
accepts one of three forms:

```jsonc
0.12                                                   // constant
[0.10, 0.10, 0.11, ...]                                // exactly one value per step
{"resolution_minutes": 60, "values": [0.10, 0.11, ...]}  // coarser grid, held constant
```

`resolution_minutes` must be a multiple of `step_minutes`, and the values must cover the horizon
exactly (24 hourly values for a 24 h horizon).

## Validation

The loader reports the offending field, for example

```text
evcharge: error: sessions[3]: session EV-04: departure_step (34) must be after arrival_step (34);
the plug-in window is shorter than one step
evcharge: error: base load minus PV exceeds the grid limit at step 68 (2026-04-15T17:00:00+02:00):
38.00 kW > 32.00 kW
```

and exits with status 2.
