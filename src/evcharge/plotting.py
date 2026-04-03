"""Power and line-current plots (requires the optional ``plot`` extra: matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evcharge.metrics import compute_metrics
from evcharge.model import FloatArray, Scenario
from evcharge.sim import SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e1e0d9"
_BASE = "#c3c2b7"
_EV = "#2a78d6"
_IMPORT = "#eb6834"
_PV = "#1baf7a"
_LINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
"""L1, L2, L3: the first three categorical slots (distinct under colour-vision deficiency)."""


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plotting needs matplotlib; install with: pip install 'ev-smart-charging[plot]'"
        ) from None


def power_figure(results: Sequence[SimulationResult]) -> Figure:
    """Build a figure: price on top, then one stacked power panel per result.

    Each power panel stacks the base load and every EV session, and overlays
    the grid limit (plus PV, i.e. the ceiling for gross consumption) and the
    net site import.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not results:
        raise ValueError("need at least one simulation result to plot")
    sc = results[0].scenario
    if any(r.scenario is not sc for r in results):
        raise ValueError("all results must come from the same scenario")
    xs = _step_edges(sc)

    def steps(series: np.ndarray) -> np.ndarray:
        return np.repeat(series, 2)

    n = len(results)
    fig, axes = plt.subplots(
        n + 1,
        1,
        figsize=(10, 1.6 + 2.6 * n),
        sharex=True,
        gridspec_kw={"height_ratios": [0.6] + [1.0] * n},
        constrained_layout=True,
    )
    price_ax = axes[0]
    price_ax.plot(xs, steps(sc.tariff.price_eur_per_kwh * 100), color=_INK, lw=1.5)
    price_ax.set_ylabel("price\nct/kWh", color=_MUTED)
    price_ax.set_title(f"{sc.name}: import price", loc="left", fontsize=10, color=_INK)

    ceiling = sc.site.grid_limit_kw + sc.pv
    limit_label = "grid limit + PV" if sc.pv.any() else "grid limit"
    ymax = float(
        max(ceiling.max(), max(float((sc.base_load + r.ev_power_kw).max()) for r in results))
    )
    for ax, res in zip(axes[1:], results, strict=True):
        m = compute_metrics(res)
        layers = [steps(sc.base_load)] + [steps(row) for row in res.power_kw if row.any()]
        colors = [_BASE] + [_EV] * (len(layers) - 1)
        ax.stackplot(xs, *layers, colors=colors, edgecolor="white", linewidth=0.3)
        ax.plot(xs, steps(ceiling), color=_INK, lw=1.2, ls="--", label=limit_label)
        ax.plot(xs, steps(res.net_import_kw), color=_IMPORT, lw=1.5, label="net import")
        if sc.pv.any():
            ax.plot(xs, steps(sc.pv), color=_PV, lw=1.2, label="PV")
        ax.set_ylim(min(0.0, float(res.net_import_kw.min()) * 1.05), ymax * 1.3)
        ax.set_ylabel("kW", color=_MUTED)
        ax.set_title(
            f"{res.policy_name}: {m.delivered_pct:.1f} % delivered, "
            f"peak {m.peak_import_kw:.1f} kW, total {m.total_cost_eur:.2f} EUR",
            loc="left",
            fontsize=10,
            color=_INK,
        )
    handles, _ = axes[1].get_legend_handles_labels()
    handles = [
        Patch(color=_BASE, label="base load"),
        Patch(color=_EV, label="EV sessions"),
        *handles,
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=len(handles))
    _style(axes, sc)
    return fig


def _step_edges(scenario: Scenario) -> np.ndarray:
    """Hours since the horizon start at step edges, so step-wise values draw as steps."""
    hz = scenario.horizon
    return np.repeat(np.arange(hz.n_steps + 1) * hz.dt_h, 2)[1:-1]


def _style(axes: Sequence[Axes], scenario: Scenario) -> None:
    """Recessive hairline grid and axes; wall-clock ticks on the last axis."""
    hz = scenario.horizon
    for ax in axes:
        ax.grid(True, color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(_BASE)
        ax.spines["bottom"].set_color(_BASE)
        ax.tick_params(colors=_MUTED, labelsize=8)
    tick_every = 3 if hz.n_steps * hz.dt_h > 12 else 1
    ticks = np.arange(0.0, hz.n_steps * hz.dt_h + 1e-9, tick_every)
    labels = [f"{hz.start + timedelta(hours=float(h)):%H:%M}" for h in ticks]
    axes[-1].set_xticks(ticks, labels)
    axes[-1].set_xlim(0.0, hz.n_steps * hz.dt_h)


def line_current_figure(panels: Sequence[tuple[str, Scenario, FloatArray]]) -> Figure:
    """Current on L1, L2 and L3 over time, one panel per ``(title, scenario, currents)``.

    ``currents`` has shape ``(n_steps, 3)`` (e.g. :attr:`SimulationResult.line_current_a`
    or :meth:`Scenario.line_currents_a` of a plan). Each panel draws its
    scenario's line limit as a dashed threshold. Lines are drawn from L3 (widest,
    underneath) to L1, so lines carrying the same current stay visible, and a
    panel whose lines coincide says so in its title.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("need at least one panel")
    for title, sc, _ in panels:
        if sc.site.supply is None:
            raise ValueError(f"panel {title!r}: the scenario has no phase-aware supply")
    top = max(
        max(float(np.max(c)), *(sc.site.supply.line_limit_a if sc.site.supply else (0.0,)))
        for _, sc, c in panels
    )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 0.6 + 2.3 * len(panels)), sharex=True, constrained_layout=True
    )
    axes_list = list(np.atleast_1d(axes))
    for ax, (title, sc, currents) in zip(axes_list, panels, strict=True):
        assert sc.site.supply is not None
        xs = _step_edges(sc)
        cur = np.asarray(currents, dtype=np.float64)
        same = [
            f"L{a + 1} = L{b + 1}"
            for a, b in ((0, 1), (0, 2), (1, 2))
            if np.allclose(cur[:, a], cur[:, b], atol=1e-6)
        ]
        for line, width in ((2, 3.2), (1, 1.6), (0, 1.6)):
            ax.plot(
                xs,
                np.repeat(cur[:, line], 2),
                color=_LINE_COLORS[line],
                lw=width,
                label=f"L{line + 1}",
            )
        limit = max(sc.site.supply.line_limit_a)
        fuse = f"fuse {limit:.0f} A"
        ax.plot(xs, np.full(xs.size, limit), color=_INK, lw=1.0, ls="--", label=fuse)
        ax.set_ylim(0.0, top * 1.12)
        ax.set_ylabel("A", color=_MUTED)
        suffix = f" ({', '.join(same)})" if same else ""
        ax.set_title(title + suffix, loc="left", fontsize=10, color=_INK)
    handles, labels = axes_list[0].get_legend_handles_labels()
    order = [labels.index(n) for n in ("L1", "L2", "L3")] + [len(labels) - 1]
    axes_list[0].legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        fontsize=8,
        frameon=False,
        ncol=4,
    )
    _style(axes_list, panels[0][1])
    return fig


def save_figure(fig: Figure, path: str | Path, *, dpi: int = 110) -> Path:
    """Write ``fig`` to ``path`` (format from the extension) and close it."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def save_power_plot(
    results: Sequence[SimulationResult], path: str | Path, *, dpi: int = 110
) -> Path:
    """Render :func:`power_figure` to ``path`` (format from the file extension)."""
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    return save_figure(power_figure(results), path, dpi=dpi)
