"""The self-contained HTML report."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evcharge import scenarios
from evcharge.cli import main
from evcharge.experiment import compare
from evcharge.policies import LeastLaxityFirst, OptimalSchedule
from evcharge.report import html_report


def test_report_is_self_contained_with_kpis_and_figures() -> None:
    pytest.importorskip("matplotlib")
    sc = scenarios.generate("workplace", n_sessions=6, seed=2, grid="IT", v2g_share=0.5)
    page = html_report(compare(sc, [LeastLaxityFirst(), OptimalSchedule()]))
    assert page.startswith("<!DOCTYPE html>")
    # power figure and line-current figure, embedded
    assert page.count('src="data:image/png;base64,') == 2
    # no external resources of any kind
    assert not re.search(r'(src|href)="(?!data:)', page)
    for header in ("delivered %", "total EUR", "line %", "V2G kWh", "wear EUR", "gap %"):
        assert f"<th>{header}</th>" in page
    assert "<td>llf</td>" in page
    assert "<td>optimal</td>" in page
    assert "LP lower bound" in page


def test_report_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "report.html"
    args = ["report", "--sessions", "4", "--policies", "edf", "mpc", "--output", str(out)]
    assert main(args) == 0
    assert f"wrote {out}" in capsys.readouterr().out
    page = out.read_text(encoding="utf-8")
    assert "<td>mpc</td>" in page
    assert "Cheapest policy that delivered every request" in page
