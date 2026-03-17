"""How robust is MPC without an arrival forecast?

Runs MPC with three quick-charge weights, LLF and the clairvoyant optimum on ten
seeds of each built-in scenario, at the default grid limit and at a tighter one,
and prints how often energy was left undelivered and the mean total cost
relative to the optimum.

    python examples/mpc_robustness.py        # about 10 minutes on one core
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from evcharge import compute_metrics, scenarios, simulate
from evcharge.policies import LeastLaxityFirst, ModelPredictiveControl, OptimalSchedule, Policy

SEEDS = range(1, 11)
CASES: list[tuple[str, float | None]] = [
    ("workplace", 60.0),
    ("residential", None),
    ("depot", None),
    ("workplace", 40.0),
    ("residential", 40.0),
    ("depot", 200.0),
]
POLICIES: list[tuple[str, Callable[[], Policy]]] = [
    ("mpc w=0", lambda: ModelPredictiveControl(quick_charge_weight=0.0)),
    ("mpc w=0.002", ModelPredictiveControl),
    ("mpc w=0.01", lambda: ModelPredictiveControl(quick_charge_weight=0.01)),
    ("llf", LeastLaxityFirst),
    ("optimal", OptimalSchedule),
]


def main() -> None:
    """Print one row per (scenario, grid limit, policy)."""
    print(
        f"{'scenario':<12} {'limit':>7}  {'policy':<12} {'runs short':>10} {'unmet kWh':>10}"
        f" {'cost/opt':>9}"
    )
    for kind, limit in CASES:
        unmet: dict[str, list[float]] = {name: [] for name, _ in POLICIES}
        cost: dict[str, list[float]] = {name: [] for name, _ in POLICIES}
        grid = 0.0
        for seed in SEEDS:
            sc = scenarios.generate(kind, n_sessions=40, seed=seed, grid_limit_kw=limit)
            grid = sc.site.grid_limit_kw
            for name, factory in POLICIES:
                m = compute_metrics(simulate(sc, factory()))
                unmet[name].append(m.unmet_kwh)
                cost[name].append(m.total_cost_eur)
        best = np.array(cost["optimal"])
        for name, _ in POLICIES:
            u = np.array(unmet[name])
            ratio = float(np.mean(np.array(cost[name]) / best))
            short = f"{int(np.sum(u > 0.01))}/{len(u)}"
            print(
                f"{kind:<12} {grid:>4.0f} kW  {name:<12} {short:>10} {u.sum():>10.1f} {ratio:>9.3f}"
            )


if __name__ == "__main__":
    main()
