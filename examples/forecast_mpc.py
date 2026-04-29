"""Does an arrival forecast make MPC robust? A seeds x scenarios x grid-limit study.

Plain MPC sees only connected EVs. Under scarce capacity it postpones energy
into cheap slots that later arrivals also need and ends runs short. This study
compares it with three forecast-aware variants, LLF and the clairvoyant
optimum on test seeds 1 to 10 of every built-in profile, at the default grid
limit and at a tight one. The forecasts are learned from 30 training days per
profile generated with reserved seeds (1,000,000 and up), never a test seed.

Policies: ``llf``; ``mpc`` with quick-charge weight 0 and 0.002 (the default);
``mpc-reserve``, ``mpc-ev`` and ``mpc-saa`` (10 sampled days per step), all
with weight 0; ``optimal`` (perfect foresight). Every run is also compared with
the LP-relaxation lower bound.

    python examples/forecast_mpc.py --csv runs.csv          # about 2 hours
    python examples/forecast_mpc.py --tables runs.csv       # tables from the CSV
    python examples/forecast_mpc.py --figure docs/figures/forecast-mpc.png --from runs.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from evcharge import compute_metrics, scenarios, simulate
from evcharge.forecast import ArrivalForecast
from evcharge.optim import relaxation_bound
from evcharge.policies import (
    ExpectedValueMPC,
    LeastLaxityFirst,
    ModelPredictiveControl,
    OptimalSchedule,
    Policy,
    ReserveMPC,
    ScenarioMPC,
)

SEEDS = range(1, 11)
TRAIN_DAYS = 30
SESSIONS = 40
CASES: list[tuple[str, float | None, str]] = [
    ("workplace", 60.0, "default"),
    ("residential", None, "default"),
    ("depot", None, "default"),
    ("workplace", 40.0, "tight"),
    ("residential", 40.0, "tight"),
    ("depot", 200.0, "tight"),
]
POLICIES: list[tuple[str, Callable[[ArrivalForecast], Policy]]] = [
    ("llf", lambda fc: LeastLaxityFirst()),
    ("mpc w=0", lambda fc: ModelPredictiveControl(quick_charge_weight=0.0)),
    ("mpc w=0.002", lambda fc: ModelPredictiveControl()),
    ("mpc-reserve", ReserveMPC),
    ("mpc-ev", ExpectedValueMPC),
    ("mpc-saa K=10", lambda fc: ScenarioMPC(fc, n_scenarios=10)),
    ("optimal", lambda fc: OptimalSchedule()),
]


@dataclass(frozen=True)
class Run:
    """One policy on one test day."""

    kind: str
    limit_kw: float
    regime: str
    seed: int
    policy: str
    requested_kwh: float
    unmet_kwh: float
    total_eur: float
    penalised_eur: float
    bound_eur: float
    runtime_s: float
    steps: int


def run_all(path: str) -> None:
    """Simulate every case, seed and policy; write one CSV row per run."""
    fields = list(Run.__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for kind, limit, regime in CASES:
            history = scenarios.training_days(
                kind, TRAIN_DAYS, n_sessions=SESSIONS, grid_limit_kw=limit
            )
            forecast = ArrivalForecast.fit(history)
            for seed in SEEDS:
                sc = scenarios.generate(kind, n_sessions=SESSIONS, seed=seed, grid_limit_kw=limit)
                bound = relaxation_bound(sc)
                for name, make in POLICIES:
                    policy = make(forecast)
                    started = time.perf_counter()
                    res = simulate(sc, policy)
                    elapsed = time.perf_counter() - started
                    m = compute_metrics(res)
                    row = Run(
                        kind,
                        sc.site.grid_limit_kw,
                        regime,
                        seed,
                        name,
                        m.energy_requested_kwh,
                        m.unmet_kwh,
                        m.total_cost_eur,
                        m.penalised_cost_eur,
                        bound,
                        elapsed,
                        sc.horizon.n_steps,
                    )
                    writer.writerow(asdict(row))
                    fh.flush()
                    print(
                        f"{kind:<12} {sc.site.grid_limit_kw:5.0f} kW seed {seed:2d} {name:<13} "
                        f"unmet {m.unmet_kwh:7.2f} total {m.total_cost_eur:7.2f} {elapsed:6.1f} s",
                        flush=True,
                    )


def load(path: str) -> list[Run]:
    """Read the runs written by :func:`run_all`."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(
            Run(
                r["kind"],
                float(r["limit_kw"]),
                r["regime"],
                int(r["seed"]),
                r["policy"],
                float(r["requested_kwh"]),
                float(r["unmet_kwh"]),
                float(r["total_eur"]),
                float(r["penalised_eur"]),
                float(r["bound_eur"]),
                float(r["runtime_s"]),
                int(r["steps"]),
            )
        )
    return out


def tables(runs: list[Run]) -> None:
    """Markdown tables for docs/experiments.md."""
    names = [name for name, _ in POLICIES]
    print(
        "\n| case | policy | runs short | unmet kWh (mean) | delivered % (min) "
        "| cost / optimum | gap to LP bound % | runtime s (median) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for kind, _, regime in CASES:
        case = [r for r in runs if r.kind == kind and r.regime == regime]
        if not case:
            continue
        limit = case[0].limit_kw
        best = {r.seed: r.total_eur for r in case if r.policy == "optimal"}
        for name in names:
            rs = [r for r in case if r.policy == name]
            short = sum(r.unmet_kwh > 0.01 for r in rs)
            unmet = statistics.mean(r.unmet_kwh for r in rs)
            delivered = min(100.0 * (1.0 - r.unmet_kwh / r.requested_kwh) for r in rs)
            ratio = statistics.mean(r.total_eur / best[r.seed] for r in rs)
            gap = statistics.mean(
                100.0 * (r.penalised_eur - r.bound_eur) / abs(r.bound_eur) for r in rs
            )
            runtime = statistics.median(r.runtime_s for r in rs)
            print(
                f"| {kind} {limit:.0f} kW | {name} | {short}/{len(rs)} | {unmet:.1f} | "
                f"{delivered:.1f} | {ratio:.3f} | {gap:.1f} | {runtime:.1f} |"
            )


def figure(runs: list[Run], path: str) -> None:
    """Unmet energy (tight limits) and cost relative to the optimum (default limits)."""
    import matplotlib.pyplot as plt

    from evcharge.plotting import save_figure

    names = [name for name, _ in POLICIES]
    colors = {"workplace": "#2a78d6", "residential": "#eb6834", "depot": "#1baf7a"}
    markers = {"workplace": "o", "residential": "s", "depot": "D"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True, layout="constrained")
    y = np.arange(len(names))[::-1]
    for kind in colors:
        tight = [r for r in runs if r.kind == kind and r.regime == "tight"]
        default = [r for r in runs if r.kind == kind and r.regime == "default"]
        best = {r.seed: r.total_eur for r in default if r.policy == "optimal"}
        unmet = [np.mean([r.unmet_kwh for r in tight if r.policy == n]) for n in names]
        extra = [
            100.0 * np.mean([r.total_eur / best[r.seed] - 1.0 for r in default if r.policy == n])
            for n in names
        ]
        left.scatter(
            unmet,
            y,
            color=colors[kind],
            marker=markers[kind],
            s=36,
            zorder=3,
            clip_on=False,
            label=kind,
        )
        right.scatter(
            extra, y, color=colors[kind], marker=markers[kind], s=36, zorder=3, label=kind
        )
    left.set_yticks(y, names)
    left.set_xlabel("mean unmet energy per day (kWh), tight limits (40 / 40 / 200 kW)")
    right.set_xlabel("mean total cost above the optimum (%), default limits")
    left.set_xscale("symlog", linthresh=1.0)
    left.set_xlim(left=0.0)
    for ax in (left, right):
        ax.grid(axis="x", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.legend(
        *left.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Forecast-aware MPC versus plain MPC, LLF and the optimum (10 test days per case)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    out = save_figure(fig, path)
    print(f"wrote {out}")


def main() -> None:
    """Run the study, print the tables or draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="run the study and write one row per run to this file")
    parser.add_argument("--tables", metavar="CSV", help="print the tables from a CSV")
    parser.add_argument("--figure", help="write the summary figure to this path")
    parser.add_argument("--from", dest="source", metavar="CSV", help="CSV for --figure")
    args = parser.parse_args()
    if args.csv:
        run_all(args.csv)
        tables(load(args.csv))
    if args.tables:
        tables(load(args.tables))
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        if not args.source:
            parser.error("--figure needs --from CSV")
        figure(load(args.source), args.figure)


if __name__ == "__main__":
    main()
