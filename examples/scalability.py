"""How do the solvers scale? Solve time against fleet size and horizon length.

Residential scenarios (overnight stays of 11 to 20 hours, the longest windows
of the built-in profiles) on the aggregate kW model with the 4.14 kW minimum,
so the MILPs have one binary per session and step in its window. Measured:

* ``LP``: the relaxation (minimum power relaxed), i.e. :func:`relaxation_bound`;
* ``relax-and-fix``: the default strategy of the offline optimum;
* ``exact MILP``: the full MILP, stopped at a 0.1 % relative gap or after 60 s
  (HiGHS reports the gap it reached);
* ``MPC step``: one MPC decision with every EV connected (the worst step:
  the longest local horizon and a binary per EV), via :func:`evcharge.sim.first_step`.

Every point is the median of 3 repeats. The 1-minute load average is printed
with each row; run it on an otherwise idle machine.

    python examples/scalability.py --csv scaling.csv       # about 15 minutes
    python examples/scalability.py --figure docs/figures/scalability.png --from scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import replace

from evcharge import scenarios
from evcharge.model import Charger, Scenario
from evcharge.optim import problem_from_scenario, relaxation_bound, solve_schedule
from evcharge.policies import ModelPredictiveControl
from evcharge.sim import first_step

FLEETS = (10, 25, 50, 100, 200, 500)
STEPS = (60, 30, 15, 5)  # minutes: 24, 48, 96 and 288 steps per day
HORIZON_FLEET = 50
REPEATS = 3
EXACT_LIMIT_S = 60.0
EXACT_GAP = 1e-3


def load1() -> float:
    """1-minute load average."""
    return os.getloadavg()[0]


def all_connected(sc: Scenario) -> Scenario:
    """Every EV on its own charger, plugged in at step 0 (the worst MPC step)."""
    chargers = tuple(Charger(f"X{i:04d}", 11.0, 4.14) for i in range(len(sc.sessions)))
    sessions = tuple(
        replace(s, charger_id=c.id, arrival_step=0)
        for s, c in zip(sc.sessions, chargers, strict=True)
    )
    return replace(sc, site=replace(sc.site, chargers=chargers), sessions=sessions)


def timed(run: Callable[[], str]) -> tuple[float, str]:
    """Median wall time of ``REPEATS`` runs and the note of the last one."""
    times = []
    note = ""
    for _ in range(REPEATS):
        started = time.perf_counter()
        note = run()
        times.append(time.perf_counter() - started)
    return statistics.median(times), note


def measure(n: int, step_minutes: int) -> list[dict[str, object]]:
    """All methods on one scenario size."""
    sc = scenarios.generate("residential", n_sessions=n, seed=7, step_minutes=step_minutes)
    problem = problem_from_scenario(sc)
    rows: list[dict[str, object]] = []

    def lp() -> str:
        relaxation_bound(sc)
        return ""

    def relax_and_fix() -> str:
        sol = solve_schedule(problem, strategy="relax-and-fix")
        return f"{sol.n_binaries} binaries, gap {100 * sol.gap:.2f} %"

    def exact() -> str:
        sol = solve_schedule(
            problem, strategy="exact", time_limit_s=EXACT_LIMIT_S, mip_rel_gap=EXACT_GAP
        )
        return f"{sol.n_binaries} binaries, {sol.status}, gap {100 * sol.gap:.2f} %"

    worst = all_connected(sc)

    def mpc_step() -> str:
        first_step(worst, ModelPredictiveControl())
        return f"{len(worst.sessions)} EVs connected"

    methods: list[tuple[str, Callable[[], str]]] = [
        ("LP", lp),
        ("relax-and-fix", relax_and_fix),
        ("exact MILP", exact),
        ("MPC step", mpc_step),
    ]
    for name, run in methods:
        if name == "exact MILP" and n > 200:
            continue
        before = load1()
        seconds, note = timed(run)
        row = {
            "sessions": n,
            "steps": sc.horizon.n_steps,
            "method": name,
            "seconds": round(seconds, 4),
            "note": note,
            "load1": round(max(before, load1()), 2),
        }
        rows.append(row)
        print(row, flush=True)
    return rows


def run_all(path: str) -> None:
    """Both sweeps; one CSV row per (size, method)."""
    rows = [r for n in FLEETS for r in measure(n, 15)]
    rows += [r for m in STEPS if m != 15 for r in measure(HORIZON_FLEET, m)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path: str) -> list[dict[str, str]]:
    """Rows written by :func:`run_all`."""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tables(rows: list[dict[str, str]]) -> None:
    """Markdown tables for docs/experiments.md."""
    methods = ["LP", "relax-and-fix", "exact MILP", "MPC step"]

    def table(title: str, key: str, subset: list[dict[str, str]]) -> None:
        print(f"\n**{title}** (median of {REPEATS} runs, seconds)\n")
        print(f"| {key} | " + " | ".join(methods) + " | exact MILP result | max load |")
        print("|---:|" + "---:|" * len(methods) + "---|---:|")
        for value in sorted({int(r[key]) for r in subset}):
            at = [r for r in subset if int(r[key]) == value]
            by = {r["method"]: r for r in at}
            cells = [f"{float(by[m]['seconds']):.2f}" if m in by else "-" for m in methods]
            exact = by["exact MILP"]["note"] if "exact MILP" in by else "not run"
            peak = max(float(r["load1"]) for r in at)
            print(f"| {value} | " + " | ".join(cells) + f" | {exact} | {peak:.2f} |")

    table("Fleet size, 96 steps", "sessions", [r for r in rows if r["steps"] == "96"])
    table(
        f"Horizon length, {HORIZON_FLEET} sessions",
        "steps",
        [r for r in rows if r["sessions"] == str(HORIZON_FLEET)],
    )


def figure(rows: list[dict[str, str]], path: str) -> None:
    """Log-log solve time against fleet size and against horizon length."""
    import matplotlib.pyplot as plt

    from evcharge.plotting import save_figure

    styles = {
        "LP": ("#2a78d6", "o"),
        "relax-and-fix": ("#eb6834", "s"),
        "exact MILP": ("#1baf7a", "D"),
        "MPC step": ("#7a5bd1", "^"),
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True, layout="constrained")
    for method, (color, marker) in styles.items():
        for ax, key, subset in (
            (left, "sessions", [r for r in rows if r["steps"] == "96"]),
            (right, "steps", [r for r in rows if r["sessions"] == str(HORIZON_FLEET)]),
        ):
            pts = sorted(
                (int(r[key]), float(r["seconds"])) for r in subset if r["method"] == method
            )
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, color=color, marker=marker, lw=1.6, ms=6, label=method)
    left.set_xlabel("sessions (96 steps of 15 min)")
    right.set_xlabel(f"steps per day ({HORIZON_FLEET} sessions)")
    left.set_ylabel("wall time (s)")
    for ax, ticks in ((left, FLEETS), (right, [24 * 60 // m for m in STEPS])):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.grid(color="#e6e6e6", lw=0.6)
        ax.axhline(EXACT_LIMIT_S, color="#9a9a9a", lw=0.8, ls="--")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.legend(
        *left.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Solve time on residential scenarios (dashed: the exact MILP's 60 s time limit)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    out = save_figure(fig, path)
    print(f"wrote {out}")


def main() -> None:
    """Run the sweeps, print the tables or draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="run the sweeps and write the rows to this file")
    parser.add_argument("--tables", metavar="CSV", help="print the tables from a CSV")
    parser.add_argument("--figure", help="write the figure to this path")
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
