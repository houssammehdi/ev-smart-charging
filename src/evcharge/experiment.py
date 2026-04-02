"""Run several policies on one scenario and format the comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evcharge.metrics import Metrics, compute_metrics
from evcharge.model import Scenario
from evcharge.optim import DEFAULT_UNMET_PENALTY_EUR_PER_KWH, relaxation_bound
from evcharge.policies import Policy
from evcharge.sim import SimulationResult, simulate


@dataclass(frozen=True, eq=False)
class Comparison:
    """Results of all policies on one scenario.

    Attributes:
        scenario: The scenario.
        results: One simulation result per policy, in input order.
        metrics: Metrics per policy, in input order.
        lower_bound_eur: LP-relaxation lower bound on the penalised cost.
    """

    scenario: Scenario
    results: tuple[SimulationResult, ...]
    metrics: tuple[Metrics, ...]
    lower_bound_eur: float

    def gap_pct(self, m: Metrics) -> float:
        """Penalised cost of ``m`` above the lower bound, in percent (NaN if the bound is 0)."""
        if abs(self.lower_bound_eur) < 1e-9:
            return float("nan")
        return 100.0 * (m.penalised_cost_eur - self.lower_bound_eur) / abs(self.lower_bound_eur)


def compare(
    scenario: Scenario,
    policies: Sequence[Policy],
    *,
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
) -> Comparison:
    """Simulate every policy on ``scenario`` and compute metrics and the lower bound."""
    results = tuple(simulate(scenario, p) for p in policies)
    metrics = tuple(compute_metrics(r, unmet_penalty=unmet_penalty) for r in results)
    bound = relaxation_bound(scenario, unmet_penalty=unmet_penalty)
    return Comparison(scenario=scenario, results=results, metrics=metrics, lower_bound_eur=bound)


def describe_scenario(scenario: Scenario) -> str:
    """One-line summary of a scenario."""
    hz = scenario.horizon
    extras = []
    supply = scenario.site.supply
    if supply is not None:
        limits = "/".join(f"{v:.0f}" if v == round(v) else f"{v:.1f}" for v in supply.line_limit_a)
        extras.append(f"{supply.grid.value} {limits} A per line")
    if scenario.pv.max() > 0:
        extras.append(f"PV peak {scenario.pv.max():.1f} kW")
    if scenario.base_load.max() > 0:
        extras.append(f"base load peak {scenario.base_load.max():.1f} kW")
    extra = (" | " + " | ".join(extras)) if extras else ""
    return (
        f"scenario {scenario.name} | {len(scenario.sessions)} sessions on "
        f"{len(scenario.site.chargers)} chargers | {scenario.energy_requested_kwh:.1f} kWh "
        f"requested | grid limit {scenario.site.grid_limit_kw:.1f} kW{extra} | "
        f"{hz.start:%Y-%m-%d %H:%M} + {hz.n_steps * hz.step_minutes / 60:g} h @ "
        f"{hz.step_minutes} min | demand charge "
        f"{scenario.tariff.demand_charge_eur_per_kw:.2f} EUR/kW"
    )


_COLUMNS: tuple[tuple[str, str], ...] = (
    ("policy", "<"),
    ("delivered %", ">"),
    ("unmet kWh", ">"),
    ("done %", ">"),
    ("energy EUR", ">"),
    ("peak kW", ">"),
    ("demand EUR", ">"),
    ("total EUR", ">"),
    ("gap %", ">"),
    ("Jain", ">"),
    ("util %", ">"),
    ("viol", ">"),
)


def format_table(comparison: Comparison) -> str:
    """Render the comparison as an aligned plain-text table.

    ``gap %`` is the penalised cost (total cost plus the unmet-energy penalty)
    above the LP-relaxation lower bound. Phase-aware scenarios get a ``line %``
    column: the highest line current relative to its limit.
    """
    columns = list(_COLUMNS)
    phase = comparison.scenario.site.phase_aware
    if phase:
        columns.insert(len(columns) - 1, ("line %", ">"))
    rows = []
    for m in comparison.metrics:
        row = [
            m.policy,
            f"{m.delivered_pct:.2f}",
            f"{m.unmet_kwh:.2f}",
            f"{m.sessions_completed_pct:.1f}",
            f"{m.energy_cost_eur:.2f}",
            f"{m.peak_import_kw:.1f}",
            f"{m.demand_charge_eur:.2f}",
            f"{m.total_cost_eur:.2f}",
            f"{comparison.gap_pct(m):.2f}",
            f"{m.jain_fairness:.3f}",
            f"{m.capacity_utilisation_pct:.1f}",
            str(m.violations),
        ]
        if phase:
            row.insert(len(row) - 1, f"{m.max_line_loading_pct:.1f}")
        rows.append(row)
    headers = [name for name, _ in columns]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        parts = [f"{c:{align}{w}}" for c, (_, align), w in zip(cells, columns, widths, strict=True)]
        return "  ".join(parts).rstrip()

    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(r) for r in rows)
    out.append("")
    out.append(
        f"lower bound (LP relaxation, perfect foresight): {comparison.lower_bound_eur:.2f} EUR"
    )
    return "\n".join(out)
