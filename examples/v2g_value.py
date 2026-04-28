"""What is V2G worth? Bidirectional versus unidirectional smart charging.

Every EV in these residential scenarios has a battery model (arrival energy,
20 % reserve, 90 % ceiling, departure target). The baseline is the same
scenario with charge-only chargers (``Scenario.unidirectional``), so the only
difference is whether the chargers may discharge. Battery wear is charged per
kWh of throughput in both directions; the charging needed for driving costs
the same wear in both variants, so savings are net of the *extra* wear of
cycling.

Cases (5 seeds each):

* ``ev-peak``: the default residential garage (40 EVs, 50 kW connection); the
  overnight charging of the EVs sets the peak.
* ``building-peak``: 20 EVs in an apartment block whose evening load (60 kW
  base-load peak parameter, 80 kW connection) sets the peak.
* ``building-peak-high-dc``: the same with a demand charge of 1.0 instead of
  0.3 EUR/kW per day.
* ``volatile``: ``ev-peak`` with the spot-price deviations from the daily mean
  tripled (a volatile day; the grid fee is unchanged).

Policies: the offline optimum, plain MPC and the forecast-aware ``mpc-ev``
(its forecast is learned from 30 training days of the same case with seeds
from 1,000,000), each on both variants.

    python examples/v2g_value.py                   # tables, about 20 minutes
    python examples/v2g_value.py --figure docs/figures/v2g-peak-shaving.png
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from evcharge import compute_metrics, scenarios, simulate
from evcharge.forecast import ArrivalForecast
from evcharge.metrics import Metrics
from evcharge.model import Scenario, Tariff
from evcharge.policies import ExpectedValueMPC, ModelPredictiveControl, OptimalSchedule, Policy

SEEDS = (1, 2, 3, 4, 5)
DEGRADATION = (0.0, 0.02, 0.04, 0.08)
DEFAULT_WEAR = scenarios.DEFAULT_DEGRADATION_EUR_PER_KWH

CASES: dict[str, dict[str, float]] = {
    "ev-peak": {"n_sessions": 40},
    "building-peak": {"n_sessions": 20, "base_load_peak_kw": 60.0, "grid_limit_kw": 80.0},
    "building-peak-high-dc": {
        "n_sessions": 20,
        "base_load_peak_kw": 60.0,
        "grid_limit_kw": 80.0,
        "demand_charge_eur_per_kw": 1.0,
    },
    "volatile": {"n_sessions": 40, "spread": 3.0},
}


def volatile(sc: Scenario, factor: float) -> Scenario:
    """Scale the spot price's deviation from its mean by ``factor`` (fee unchanged)."""
    t = sc.tariff
    spot = t.export_price
    fee = t.price_eur_per_kwh - spot
    new_spot = spot.mean() + factor * (spot - spot.mean())
    tariff = Tariff(new_spot + fee, new_spot, t.demand_charge_eur_per_kw)
    return replace(sc, tariff=tariff)


def build(case: str, seed: int, wear: float) -> Scenario:
    """Scenario of one case, every EV with a battery model on a bidirectional charger."""
    opts = CASES[case]
    sc = scenarios.generate(
        "residential",
        seed=seed,
        n_sessions=int(opts["n_sessions"]),
        base_load_peak_kw=opts.get("base_load_peak_kw"),
        grid_limit_kw=opts.get("grid_limit_kw"),
        demand_charge_eur_per_kw=opts.get(
            "demand_charge_eur_per_kw", scenarios.DEFAULT_DEMAND_CHARGE_EUR_PER_KW
        ),
        v2g_share=1.0,
        degradation_eur_per_kwh=wear,
    )
    spread = opts.get("spread", 1.0)
    return volatile(sc, spread) if spread != 1.0 else sc


def run(sc: Scenario, policy: Policy) -> Metrics:
    """Simulate and compute metrics."""
    return compute_metrics(simulate(sc, policy))


@dataclass(frozen=True)
class Pair:
    """Unidirectional and bidirectional metrics of one scenario and policy."""

    uni: Metrics
    v2g: Metrics

    @property
    def saving(self) -> float:
        """Total-cost saving of V2G (EUR, positive = V2G cheaper)."""
        return self.uni.total_cost_eur - self.v2g.total_cost_eur

    @property
    def extra_wear(self) -> float:
        """Battery wear of V2G beyond the wear of the charging for driving."""
        return self.v2g.degradation_cost_eur - self.uni.degradation_cost_eur


def pair(sc: Scenario, make: Callable[[], Policy]) -> Pair:
    """Run one policy on the unidirectional and the bidirectional variant."""
    return Pair(run(sc.unidirectional(), make()), run(sc, make()))


def forecast(case: str) -> ArrivalForecast:
    """Arrival forecast of a case, from 30 training days with reserved seeds."""
    base = scenarios.TRAINING_SEED_BASE
    return ArrivalForecast.fit([build(case, base + i, DEFAULT_WEAR) for i in range(30)])


def spread_text(values: list[float]) -> str:
    """Mean with the range over seeds."""
    return f"{np.mean(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


def main_table() -> None:
    """Default wear: offline optimum, MPC and mpc-ev, unidirectional versus V2G."""
    print(f"\n**Default battery wear ({DEFAULT_WEAR} EUR/kWh), mean [min, max] over 5 seeds**\n")
    print(
        "| case | uni total EUR | V2G saving EUR (optimal) | saving % | peak kW uni -> V2G "
        "| discharged kWh | extra wear EUR | V2G saving EUR (mpc) | V2G saving EUR (mpc-ev) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    worst_unmet = 0.0
    for case in CASES:
        fc = forecast(case)
        days = [build(case, seed, DEFAULT_WEAR) for seed in SEEDS]
        opt = [pair(sc, OptimalSchedule) for sc in days]
        mpc = [pair(sc, ModelPredictiveControl) for sc in days]
        ev = [pair(sc, partial(ExpectedValueMPC, fc)) for sc in days]
        worst_unmet = max(
            [worst_unmet] + [m.unmet_kwh for p in (*opt, *mpc, *ev) for m in (p.uni, p.v2g)]
        )
        uni_total = [p.uni.total_cost_eur for p in opt]
        saving = [p.saving for p in opt]
        pct = 100.0 * float(np.mean(saving)) / float(np.mean(uni_total))
        peaks = (
            f"{np.mean([p.uni.peak_import_kw for p in opt]):.1f} -> "
            f"{np.mean([p.v2g.peak_import_kw for p in opt]):.1f}"
        )
        print(
            f"| {case} | {np.mean(uni_total):.2f} | {spread_text(saving)} | {pct:.1f} | {peaks} | "
            f"{np.mean([p.v2g.discharged_kwh for p in opt]):.1f} | "
            f"{np.mean([p.extra_wear for p in opt]):.2f} | "
            f"{spread_text([p.saving for p in mpc])} | {spread_text([p.saving for p in ev])} |",
            flush=True,
        )
    print(f"\nlargest unmet energy of any run: {worst_unmet:.3f} kWh")


def wear_table() -> None:
    """Offline-optimal V2G saving as a function of the wear cost."""
    print("\n**Offline-optimal V2G saving in EUR per day (discharged kWh) by battery wear**\n")
    print("| case | " + " | ".join(f"{w:g} EUR/kWh" for w in DEGRADATION) + " |")
    print("|---|" + "---:|" * len(DEGRADATION))
    for case in CASES:
        cells = []
        for wear in DEGRADATION:
            pairs = [pair(build(case, seed, wear), OptimalSchedule) for seed in SEEDS]
            saving = float(np.mean([p.saving for p in pairs]))
            dis = float(np.mean([p.v2g.discharged_kwh for p in pairs]))
            cells.append(f"{saving:.2f} ({dis:.0f})")
        print(f"| {case} | " + " | ".join(cells) + " |", flush=True)


def figure(path: str) -> None:
    """Site import with and without V2G on one building-peak day (seed 1)."""
    import matplotlib.pyplot as plt

    from evcharge.plotting import save_figure

    sc = build("building-peak-high-dc", 1, DEFAULT_WEAR)
    uni = simulate(sc.unidirectional(), OptimalSchedule())
    two = simulate(sc, OptimalSchedule())
    m_uni, m_two = compute_metrics(uni), compute_metrics(two)
    hours = np.arange(sc.horizon.n_steps + 1) * sc.horizon.dt_h + sc.horizon.start.hour
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, height_ratios=(3, 2), layout="constrained"
    )

    def steps(values: np.ndarray) -> np.ndarray:
        return np.append(values, values[-1])

    top.fill_between(
        hours, steps(sc.net_base_kw), step="post", color="#dcdcdc", lw=0, label="base load"
    )
    top.step(
        hours, steps(uni.import_kw), where="post", color="#2a78d6", lw=1.6, label="unidirectional"
    )
    top.step(hours, steps(two.import_kw), where="post", color="#eb6834", lw=1.6, label="V2G")
    for m, color in ((m_uni, "#2a78d6"), (m_two, "#eb6834")):
        top.axhline(m.peak_import_kw, color=color, lw=0.8, ls="--")
    top.set_ylabel("site import (kW)")
    top.set_title(
        f"Offline optimum, apartment block with 20 EVs: peak {m_uni.peak_import_kw:.1f} kW "
        f"-> {m_two.peak_import_kw:.1f} kW with V2G",
        loc="left",
        fontsize=10,
    )
    top.legend(loc="upper right", frameon=False, fontsize=9)
    ev = two.power_kw.sum(axis=0)
    bottom.step(
        hours, steps(np.minimum(ev, 0.0)), where="post", color="#eb6834", lw=1.4, label="discharge"
    )
    bottom.step(
        hours, steps(np.maximum(ev, 0.0)), where="post", color="#1baf7a", lw=1.4, label="charge"
    )
    bottom.axhline(0.0, color="#9a9a9a", lw=0.6)
    bottom.set_ylabel("V2G EV power (kW)")
    bottom.set_xlabel("hour of day (next day after 24)")
    bottom.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in (top, bottom):
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    out = save_figure(fig, path)
    print(
        f"wrote {out}: total {m_uni.total_cost_eur:.2f} -> {m_two.total_cost_eur:.2f} EUR, "
        f"discharged {m_two.discharged_kwh:.1f} kWh"
    )


def main() -> None:
    """Print the tables; optionally write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", help="write the peak-shaving figure to this path")
    args = parser.parse_args()
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        figure(args.figure)
        return
    main_table()
    wear_table()


if __name__ == "__main__":
    main()
