"""Why phase-aware control matters: a kW limit that looks fine can overload a line.

A residential garage (40 EVs, half of them single-phase) behind a main fuse
sized so that a *balanced* load reaches the site's kW limit exactly. A
controller that only knows the kW limit (``PhaseBlind``) is compared with the
phase-aware policies, on an installation where every charger is connected
L1L2L3 ("not rotated") and on one with cyclic phase rotation.

For the kW-only controller two numbers are reported: the line current its plan
would draw with nothing stopping it (the fuse would trip), and what happens
when the simulator enforces the line limit on its commands, which it does like
a crude protective load balancer: every command on the overloaded line is
scaled down in proportion and chargers pushed below 6 A pause.

    python examples/phase_imbalance.py                    # tables, a few minutes
    python examples/phase_imbalance.py --figure docs/figures/phase-imbalance.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from evcharge import compute_metrics, scenarios, simulate
from evcharge.model import FloatArray, Scenario
from evcharge.policies import (
    EqualShare,
    LeastLaxityFirst,
    ModelPredictiveControl,
    OptimalSchedule,
    PhaseBlind,
    Policy,
)

SEED = 7
SESSIONS = 40
SHARE = 0.5


@dataclass(frozen=True)
class Row:
    """One policy on one installation."""

    policy: str
    max_line_a: tuple[float, float, float]
    delivered_pct: float
    unmet_kwh: float
    total_eur: float
    violations: int


def build(grid: str, rotate: bool) -> Scenario:
    """The garage on a TN or IT supply, with or without phase rotation."""
    return scenarios.generate(
        "residential",
        n_sessions=SESSIONS,
        seed=SEED,
        grid=grid,
        single_phase_share=SHARE,
        rotate_phases=rotate,
    )


def kw_only_plan_currents(sc: Scenario) -> FloatArray:
    """Line currents of the kW-only LLF plan with nothing enforcing the lines."""
    plan = simulate(sc.aggregate(), LeastLaxityFirst())
    per_amp = np.array([sc.control(s).kw_per_unit for s in sc.sessions])[:, None]
    return sc.line_currents_a(plan.power_kw / per_amp)


def evaluate(sc: Scenario) -> list[Row]:
    """All rows for one installation."""
    rows = []
    unprotected = kw_only_plan_currents(sc).max(axis=0)
    policies: list[Policy] = [
        PhaseBlind(LeastLaxityFirst()),
        LeastLaxityFirst(),
        EqualShare(),
        ModelPredictiveControl(),
        OptimalSchedule(),
    ]
    for policy in policies:
        res = simulate(sc, policy)
        m = compute_metrics(res)
        assert res.line_current_a is not None
        rows.append(
            Row(
                policy.name,
                tuple(float(v) for v in res.line_current_a.max(axis=0)),  # type: ignore[arg-type]
                m.delivered_pct,
                m.unmet_kwh,
                m.total_cost_eur,
                m.violations,
            )
        )
        if isinstance(policy, PhaseBlind):
            rows.append(
                Row(
                    "llf (kW only), plan before protection",
                    tuple(float(v) for v in unprotected),  # type: ignore[arg-type]
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0,
                )
            )
    return rows


def print_table(title: str, sc: Scenario, rows: list[Row]) -> None:
    """Markdown table for docs/phases.md."""
    assert sc.site.supply is not None
    fuse = sc.site.supply.line_limit_a[0]
    print(f"\n**{title}**: fuse {fuse:.0f} A per line, kW limit {sc.site.grid_limit_kw:.1f} kW\n")
    print("| policy | max L1 / L2 / L3 (A) | delivered % | unmet kWh | total EUR | line cuts |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines = " / ".join(f"{v:.0f}" for v in r.max_line_a)
        if np.isnan(r.delivered_pct):
            print(f"| {r.policy} | **{lines}** | - | - | - | - |")
        else:
            print(
                f"| {r.policy} | {lines} | {r.delivered_pct:.1f} | {r.unmet_kwh:.1f} | "
                f"{r.total_eur:.2f} | {r.violations} |"
            )


def figure(path: str) -> None:
    """Three panels for the TN garage: kW-only plan, phase-aware LLF, rotated."""
    from evcharge.plotting import line_current_figure, save_figure

    flat, rotated = build("TN", rotate=False), build("TN", rotate=True)
    assert flat.site.supply is not None
    fuse = flat.site.supply.line_limit_a[0]
    kw = kw_only_plan_currents(flat)
    aware = simulate(flat, LeastLaxityFirst())
    spread = simulate(rotated, LeastLaxityFirst())
    assert aware.line_current_a is not None
    assert spread.line_current_a is not None
    m_aware, m_spread = compute_metrics(aware), compute_metrics(spread)
    panels = [
        (
            f"kW-only LLF plan, chargers not rotated: up to {kw[:, 0].max():.0f} A on L1 "
            f"behind a {fuse:.0f} A fuse",
            flat,
            kw,
        ),
        (
            f"phase-aware LLF, same installation: L1 held at {fuse:.0f} A, "
            f"{m_aware.delivered_pct:.1f} % of the energy delivered",
            flat,
            aware.line_current_a,
        ),
        (
            f"phase-aware LLF, chargers rotated L1L2L3/L2L3L1/L3L1L2: "
            f"{m_spread.delivered_pct:.1f} % delivered",
            rotated,
            spread.line_current_a,
        ),
    ]
    out = save_figure(line_current_figure(panels), path)
    print(f"\nwrote {out}")


def main() -> None:
    """Print the tables; optionally write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", help="write the TN line-current figure to this path")
    args = parser.parse_args()
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        figure(args.figure)
        return
    for grid in ("TN", "IT"):
        for rotate in (False, True):
            sc = build(grid, rotate)
            label = f"{grid}, {'rotated' if rotate else 'not rotated'}"
            print_table(label, sc, evaluate(sc))


if __name__ == "__main__":
    main()
