# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## 0.2.0 - 2026-09-25

### Added

- **Phase-aware electrical model** (`evcharge.electrical`, `Site(supply=...)`): TN and IT
  supplies with per-line current limits, chargers with phase rotation (including OCPP
  `ConnectorPhaseRotation` names), 1-, 2- and 3-phase EVs, setpoints in amperes per phase with
  the IEC 61851 6 A minimum and a configurable resolution (0.1 A default). Every constraint is a
  capacity row shared by the simulator, the heuristics, the LP/MILP and MPC; the kW model is the
  one-row special case. `Scenario.aggregate()` and `PhaseBlind(policy)` give the kW-only view.
- Offline plans are rounded to the chargers' resolution with a rounding MILP, and MPC's first
  step is rounded safely (`finalize`), so replays match plans.
- **V2G**: `V2G` battery spec per session (capacity, arrival energy, floor, ceiling, discharge
  limits, discharge efficiency, degradation per kWh of throughput), bidirectional chargers,
  export limits and per-line export rows; battery trajectories in the LP/MILP with charge and
  discharge complementarity; battery physics and a `soc-limit` violation in the simulator;
  discharged energy and wear in the metrics and tables.
- **Forecast-aware MPC**: `ArrivalForecast` learned from training days (kernel-smoothed
  bootstrap), `mpc-ev` (expected ghost EVs), `mpc-saa` (sample-average approximation with a
  non-anticipative first step, `optim.solve_scenarios`) and `mpc-reserve` (capacity reserve);
  `scenarios.training_days` with reserved seeds; `--train-days` and `--history` options.
- **OCPP 1.6-J central system** (`evcharge.ocpp`, extra `ocpp`): OCPP-J framing and error
  codes, one outstanding CALL per connection, BootNotification, Heartbeat, StatusNotification,
  Authorize, StartTransaction, MeterValues and StopTransaction, `SetChargingProfile` TxProfiles
  in A (with `numberPhases`) or W from any policy; a simulated charge point, `evcharge
  ocpp-server` and `evcharge ocpp-demo`.
- `sim.first_step` for live control with the simulator's enforcement.
- **ACN-Data loader** (`evcharge.acn`, `evcharge run --acn`) with a hand-made sample file.
- **HTML report** (`evcharge report`): one self-contained page with KPIs and embedded figures.
- Generator options `grid`, `line_limit_a`, `rotate_phases`, `single_phase_share`,
  `v2g_share`, `degradation_eur_per_kwh`, and matching CLI flags; JSON fields for supplies,
  per-line currents, bidirectional chargers, export limits and `v2g` blocks.
- Studies: `examples/phase_imbalance.py`, `examples/forecast_mpc.py`, `examples/v2g_value.py`,
  `examples/scalability.py`; documentation `docs/theory.md`, `docs/phases.md`,
  `docs/experiments.md`, `docs/v2g.md`, `docs/ocpp-demo.md`.
- Hypothesis properties for line limits on TN and IT sites, battery bounds, export rows and the
  LP-bound sandwich with V2G, and the safety of the forecast-aware policies.

### Changed

- The README is a front page for the study; results and the model moved to `docs/`.
- `examples/mpc_robustness.py` is replaced by `examples/forecast_mpc.py`.
- JSON output maps NaN metrics (for example line loading on aggregate sites) to `null`.

### Fixed

- A MILP dual bound of exactly 0.0 was treated as missing and replaced by the incumbent.
- Undrawn overshoot energy could look like revenue at negative export prices; it is now priced
  so that the MILP objective bounds the replayed cost from above.
- V2G discharge on phase-aware sites could exceed the EV's charge limit by default.
- A minimum-current step could not top up a V2G battery just below its ceiling.
- The sample-average program dropped the overshoot allowance of all but the first scenario.

## 0.1.0 - 2026-09-25

### Added

- Validated domain model for sites, chargers, sessions, tariffs and time grids.
- Policies: uncontrolled (FCFS), equal share, EDF, LLF, price-aware valley filling, MPC and
  the clairvoyant optimum.
- LP/MILP formulation solved with HiGHS (exact and relax-and-fix), with the LP relaxation as a
  certified lower bound; coefficients below 1e-9 snapped to zero and MILP presolve disabled
  for robustness.
- Discrete-time simulator with online arrivals, command enforcement and a violation log; KPIs.
- Seeded workplace, depot and residential scenario generators; JSON scenario input.
- CLI `evcharge compare | run | plot`; hypothesis property tests; CI on Python 3.11 and 3.12.

