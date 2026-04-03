from __future__ import annotations

from pathlib import Path

import pytest

from evcharge import scenarios
from evcharge.plotting import line_current_figure, save_figure
from evcharge.policies import LeastLaxityFirst
from evcharge.sim import simulate


def test_line_current_figure(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    sc = scenarios.generate("residential", n_sessions=6, seed=1, grid="TN", rotate_phases=False)
    res = simulate(sc, LeastLaxityFirst())
    assert res.line_current_a is not None
    fig = line_current_figure([("llf", sc, res.line_current_a)])
    title = fig.axes[0].get_title(loc="left")
    assert title.startswith("llf")
    out = save_figure(fig, tmp_path / "lines.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="no phase-aware supply"):
        line_current_figure(
            [("kW", scenarios.generate("residential", n_sessions=2), res.line_current_a)]
        )
    with pytest.raises(ValueError, match="at least one panel"):
        line_current_figure([])
