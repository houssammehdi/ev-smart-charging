"""A self-contained HTML report of a policy comparison.

:func:`html_report` renders the scenario summary, the KPI table, the LP lower
bound and figures (power per policy and, on phase-aware sites, line currents)
into one HTML file. Figures are embedded as base64 PNG, and the page has no
external resources, so it can be mailed or archived as is. Figures need the
``plot`` extra (matplotlib).
"""

from __future__ import annotations

import base64
import html
import io
import math
from datetime import datetime
from typing import TYPE_CHECKING

from evcharge import __version__
from evcharge.experiment import Comparison, describe_scenario
from evcharge.metrics import Metrics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CSS = """
:root { --ink: #1f2328; --muted: #59636e; --line: #d1d9e0; --head: #f6f8fa; --bg: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --ink: #e6edf3; --muted: #9198a1; --line: #3d444d; --head: #151b23; --bg: #0d1117; }
  img { background: #ffffff; }
}
body { font: 15px/1.5 system-ui, sans-serif; color: var(--ink); background: var(--bg);
       max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.5rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; }
p.meta, figcaption, .note { color: var(--muted); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right;
         white-space: nowrap; }
th { background: var(--head); }
th:first-child, td:first-child { text-align: left; }
img { max-width: 100%; height: auto; }
"""

_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("delivered %", "delivered_pct", ".2f"),
    ("unmet kWh", "unmet_kwh", ".2f"),
    ("done %", "sessions_completed_pct", ".1f"),
    ("energy EUR", "energy_cost_eur", ".2f"),
    ("peak kW", "peak_import_kw", ".1f"),
    ("demand EUR", "demand_charge_eur", ".2f"),
    ("total EUR", "total_cost_eur", ".2f"),
    ("Jain", "jain_fairness", ".3f"),
    ("util %", "capacity_utilisation_pct", ".1f"),
    ("violations", "violations", "d"),
)


def _cell(value: float | int, fmt: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return format(value, fmt)


def _table(comparison: Comparison) -> str:
    sc = comparison.scenario
    columns = list(_COLUMNS)
    if any(s.v2g is not None for s in sc.sessions):
        columns.append(("V2G kWh", "discharged_kwh", ".1f"))
        columns.append(("wear EUR", "degradation_cost_eur", ".2f"))
    if sc.site.phase_aware:
        columns.append(("line %", "max_line_loading_pct", ".1f"))
    head = "".join(f"<th>{html.escape(name)}</th>" for name, _, _ in columns)
    rows = []
    for m in comparison.metrics:
        cells = "".join(f"<td>{_cell(getattr(m, attr), fmt)}</td>" for _, attr, fmt in columns)
        gap = comparison.gap_pct(m)
        rows.append(
            f"<tr><td>{html.escape(m.policy)}</td>{cells}<td>{_cell(gap, '.2f')}</td>"
            f"<td>{_cell(m.runtime_s, '.2f')}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>policy</th>'
        f"{head}<th>gap %</th><th>runtime s</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _png(fig: Figure) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _figures(comparison: Comparison) -> list[tuple[str, str]]:
    """``(caption, base64 PNG)`` pairs; empty without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    from evcharge.plotting import line_current_figure, power_figure

    out = [
        (
            "Price (top) and, per policy, stacked EV power over the base load with the site "
            "import and the grid limit.",
            _png(power_figure(comparison.results)),
        )
    ]
    sc = comparison.scenario
    if sc.site.phase_aware:
        panels = [
            (r.policy_name, sc, r.line_current_a)
            for r in comparison.results
            if r.line_current_a is not None
        ]
        if panels:
            out.append(
                (
                    "Current on L1, L2 and L3 per policy; dashed: the line limit.",
                    _png(line_current_figure(panels)),
                )
            )
    return out


def _best(metrics: tuple[Metrics, ...]) -> str:
    served = [m for m in metrics if m.unmet_kwh <= 0.01]
    if not served:
        return "No policy delivered every request."
    best = min(served, key=lambda m: m.total_cost_eur)
    return (
        f"Cheapest policy that delivered every request: <b>{html.escape(best.policy)}</b> "
        f"({best.total_cost_eur:.2f} EUR)."
    )


def html_report(comparison: Comparison, *, title: str | None = None) -> str:
    """Render ``comparison`` as a self-contained HTML page."""
    sc = comparison.scenario
    name = title or f"evcharge report: {sc.name}"
    figures = _figures(comparison)
    figure_html = "".join(
        f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for caption, data in figures
    )
    if not figures:
        figure_html = '<p class="note">Figures need matplotlib (the plot extra).</p>'
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{html.escape(name)}</h1>
<p class="meta">evcharge {__version__}, created {html.escape(created)}</p>
<p>{html.escape(describe_scenario(sc))}</p>
<h2>Results</h2>
{_table(comparison)}
<p>LP lower bound on the penalised cost (perfect foresight):
<b>{comparison.lower_bound_eur:.2f} EUR</b>. <code>gap %</code> is each policy's penalised cost
(total cost plus the unmet-energy penalty) above it. {_best(comparison.metrics)}</p>
<h2>Figures</h2>
{figure_html}
</body>
</html>
"""
