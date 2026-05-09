# ev-smart-charging

[![CI](https://github.com/houssammehdi/ev-smart-charging/actions/workflows/ci.yml/badge.svg)](https://github.com/houssammehdi/ev-smart-charging/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**A study of smart EV charging under grid constraints: policies, an exact optimisation
benchmark and a faithful simulator, with the numbers to show what works and what does not.**
A garage with forty chargers behind a 50 kW connection has to decide every 15 minutes which EVs
charge and how fast, without tripping a fuse. `evcharge` models that decision per phase, in
amperes, the way chargers are actually commanded (IEC 61851, OCPP), and compares real-time
heuristics, rolling-horizon MPC with and without an arrival forecast, and a perfect-foresight
LP/MILP optimum that certifies how far every policy is from the best possible schedule.

## What the study finds

All numbers come from scripts in [`examples/`](examples/) and are reproduced in the docs with
their commands.

- **A kW limit is not a grid limit.** A controller that respects a garage's 50 kW limit
  would draw **158 A on a 72 A fuse** when single-phase EVs pile up on one line. Cutting its
  commands after the fact delivers 24 % of the energy; phase-aware LLF delivers 90 %, the
  optimum 97 %, and rotating the chargers' phases lets LLF deliver everything.
  [phases.md](docs/phases.md)
- **With enough capacity, MPC is close to the clairvoyant optimum**: 0.9 to 1.4 % above it on
  average over ten days of each built-in scenario, while LLF pays 8 to 23 % more.
  [experiments.md](docs/experiments.md)
- **Under tight capacity, an arrival forecast helps MPC a lot, but LLF is still the most
  robust.** Learning arrivals from 30 training days cuts MPC's unmet energy on a tight
  residential garage from 11.1 to 0.2 kWh per day (sample-average MPC), yet LLF delivers
  everything there, and forecast-aware MPC still ends 1 to 6 days in 10 short on the tight
  residential and depot cases.
  [experiments.md](docs/experiments.md#2-does-an-arrival-forecast-make-mpc-robust)
- **V2G is worth little unless someone else sets an expensive peak.** With battery wear priced
  at 0.04 EUR/kWh, bidirectional charging saves nothing where the EVs set the peak, 0.3 % with
  a building peak at a 0.3 EUR/kW daily demand charge and 2.4 % at 1.0 EUR/kW. Plain MPC even
  *loses* money with V2G; forecast-aware MPC gains. [v2g.md](docs/v2g.md)
- **Every result is certified.** The LP relaxation is a proven lower bound on the cost of any
  schedule, and the offline optimum's plan is a proven upper bound on its own replay; the
  `gap %` column is measured against the bound. [theory.md](docs/theory.md)

## Features

- **Phase-aware electrical model**: TN (230/400 V) and IT (230 V, no neutral) supplies with
  per-line fuses; chargers with phase rotation (`L1L2L3`, `L2L3L1`, `L3L1L2` or OCPP `RST`,
  `STR`, ...); 1-, 2- and 3-phase EVs; setpoints in amperes per phase with the 6 A minimum and
  a configurable resolution (0.1 A by default). The aggregate kW model remains available, with
  unchanged results.
- **Policies behind one protocol**: `uncontrolled` (FCFS), `equal-share`, `edf`, `llf`,
  `price-aware`, `mpc`, the forecast-aware `mpc-reserve`, `mpc-ev` and `mpc-saa`, and the
  clairvoyant `optimal`. A custom policy is a class with one `decide()` method.
- **Exact optimisation**: LP/MILP with HiGHS (via SciPy), semi-continuous minimum currents,
  relax-and-fix with a certified gap, rounding to the chargers' resolution, and a
  sample-average solver with non-anticipativity.
- **V2G**: battery trajectories, floor and ceiling, discharge limits, both efficiencies,
  degradation per kWh of throughput, export limits and per-line export rows.
- **Enforcing simulator**: arrivals are revealed online; every command is checked against
  physics, battery and grid limits, and each correction is logged. Hypothesis property tests
  check that no built-in policy ever overloads a line, a site limit or a battery.
- **OCPP 1.6-J central system**: runs any policy live, sending `SetChargingProfile` in A per
  phase or W; a simulated charge point and an accelerated-time demo are included.
- **Inputs**: seeded synthetic workplace, depot and residential scenarios (Nordic price shape,
  optional PV and V2G); a documented JSON format; the Caltech ACN-Data session format.
- **Outputs**: comparison tables (text or JSON), plots, and a self-contained HTML report.

## Quickstart

```bash
git clone https://github.com/houssammehdi/ev-smart-charging.git
cd ev-smart-charging
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"        # runtime: numpy + scipy; [plot] adds matplotlib, [ocpp] websockets

evcharge compare --scenario workplace --sessions 40 --seed 7 --grid-limit 60
evcharge compare --scenario residential --grid TN --single-phase-share 0.5 --policies llf mpc optimal
evcharge report --scenario residential --grid IT --v2g-share 0.5 --output report.html
```

Everything runs offline; HiGHS ships inside SciPy.

| command | what it does |
|---|---|
| `evcharge compare` | policies on a synthetic scenario: `--scenario`, `--sessions`, `--seed`, `--grid-limit`, `--grid TN/IT`, `--line-limit`, `--single-phase-share`, `--no-rotation`, `--v2g-share`, `--degradation`, `--pv-kwp`, `--demand-charge`, ...; `--format json` |
| `evcharge run --input FILE` | the same on your own site ([input-format.md](docs/input-format.md)); `--acn FILE --grid-limit KW` reads ACN-Data sessions |
| `evcharge plot`, `evcharge report` | stacked power plots; a self-contained HTML report with KPIs and figures |
| `evcharge ocpp-server`, `evcharge ocpp-demo` | the OCPP 1.6-J central system; a demo with simulated charge points ([ocpp-demo.md](docs/ocpp-demo.md)) |

Forecast-aware policies learn from `--train-days` synthetic days (seeds reserved from
1,000,000, disjoint from test seeds) or from `--history FILE...`.

### Python

```python
from evcharge import compute_metrics, scenarios, simulate
from evcharge.policies import ModelPredictiveControl, OptimalSchedule

sc = scenarios.workplace(n_sessions=40, seed=7, grid_limit_kw=60)
for policy in (ModelPredictiveControl(), OptimalSchedule()):
    m = compute_metrics(simulate(sc, policy))
    print(f"{m.policy:8s} total {m.total_cost_eur:6.2f} EUR  peak {m.peak_import_kw:5.1f} kW")
```

```text
mpc      total  58.71 EUR  peak  46.1 kW
optimal  total  58.03 EUR  peak  39.7 kW
```

A policy is any object with `reset(scenario)` and `decide(observation)`:

```python
from collections.abc import Mapping

from evcharge.policies import Observation, OnlinePolicy
from evcharge.policies.base import priority_fill


class LargestRemainingFirst(OnlinePolicy):
    """Serve the EV with the most energy still to go first."""

    name = "largest-remaining"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        order = sorted(obs.pending(), key=lambda s: -s.remaining_kwh)
        return priority_fill(order, obs.allocation())  # every row, 6 A minimum, resolution
```

`simulate(scenario, LargestRemainingFirst())` runs it; the result's `violations` list shows
every command the simulator had to correct.

## Documentation

| page | contents |
|---|---|
| [theory.md](docs/theory.md) | the LP/MILP; when EDF/LLF are optimal and when not; NP-hardness of the minimum current; proof that the LP bounds every policy; value of information; the phase and V2G models |
| [phases.md](docs/phases.md) | TN/IT, rotation, amperes, what the linear current model guarantees; the kW-limit-versus-line study |
| [experiments.md](docs/experiments.md) | policy comparison on the built-in scenarios; the forecast-aware MPC study; solver scalability |
| [v2g.md](docs/v2g.md) | the battery model and what V2G is worth against unidirectional smart charging |
| [ocpp-demo.md](docs/ocpp-demo.md) | the OCPP 1.6-J central system, its assumptions and the demo |
| [input-format.md](docs/input-format.md) | the scenario JSON format, phase and V2G fields, ACN-Data |

## Architecture

| module | responsibility |
|---|---|
| `electrical.py` | supplies, grid types, phase rotation, wiring and kW per ampere (pure) |
| `model.py` | validated dataclasses (site, chargers, sessions, V2G, tariff) and the capacity rows |
| `capacity.py` | per-step rows: allocation, trimming and proportional enforcement |
| `policies/` | the `Policy` protocol, heuristics, optimum, MPC, forecast-aware MPC, `PhaseBlind` |
| `optim.py` | sparse LP/MILP, relax-and-fix, rounding to the grid, SAA solver, lower bound |
| `forecast.py` | arrival forecasts learned from training days |
| `sim.py` | the online simulator, command enforcement, battery physics, `first_step` |
| `metrics.py`, `experiment.py` | KPIs, comparisons and tables |
| `scenarios.py`, `io.py`, `acn.py` | synthetic generators, JSON loader, ACN-Data loader |
| `ocpp/` | OCPP-J framing, central system, simulated charge point, demo |
| `cli.py`, `plotting.py`, `report.py` | command line, figures, HTML report |

The simulator is the only component that sees future sessions: it passes the full scenario only
to policies that declare `clairvoyant = True` (the offline optimum), so online policies cannot
look ahead by accident.

## Testing

```bash
pip install -e ".[plot,ocpp,dev]"
ruff check . && ruff format --check . && mypy --strict src && pytest
```

Unit tests on hand-computed cases for every policy, the optimiser, the simulator, the metrics,
the loaders and the OCPP messages; hypothesis property tests for physical limits, line limits,
battery bounds and the LP-bound sandwich on random aggregate, phase-aware (TN and IT) and V2G
sites (`EVCHARGE_HYPOTHESIS_EXAMPLES=1000` for a deeper search); and OCPP integration tests
against an in-repo simulated charge point. CI runs the four commands on Python 3.11 and 3.12.
The studies in `examples/` take minutes to an hour and are scripts, not tests.

## Limitations

- Unity power factor for EVs, base load and PV; no neutral-conductor limit; no automatic
  1-/3-phase switching (see [phases.md](docs/phases.md#limitations)).
- EVs follow setpoints exactly and at once: no CC/CV taper, SoC-dependent limits or ramping.
- Prices, base load and PV are known exactly; declared departures and energies are truthful.
  The arrival forecast is learned from synthetic days and is not conditioned on the day so far.
- One demand-charge peak per simulated horizon; V2G wear is linear in throughput.
- Synthetic data with illustrative price, load, PV, battery and wear parameters; real sites come
  in through the JSON or ACN-Data formats.
- The OCPP central system is a research tool, not a production CSMS (no TLS, persistence or
  authorisation lists; see [ocpp-demo.md](docs/ocpp-demo.md#limitations)).

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Houssam Mehdi.
