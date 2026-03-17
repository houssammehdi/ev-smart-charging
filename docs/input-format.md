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
| `chargers[].min_power_kw`  | number | 4.14    | lowest non-zero power; 6 A x 230 V x 3 per IEC 61851. Use 0 for chargers without a minimum |

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

The rounding is conservative: a session is only scheduled in steps during which the EV is plugged
in for the whole step. A session shorter than one step after rounding is rejected with an error.

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
