from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evcharge.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "office.json"


def test_compare_prints_aligned_table(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare",
            "--scenario",
            "workplace",
            "--sessions",
            "8",
            "--seed",
            "7",
            "--grid-limit",
            "25",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("scenario workplace | 8 sessions")
    header = next(line for line in lines if line.startswith("policy"))
    assert "delivered %" in header
    assert "gap %" in header
    rows = [
        line
        for line in lines
        if line.split()
        and line.split()[0]
        in {"uncontrolled", "equal-share", "edf", "llf", "price-aware", "mpc", "optimal"}
    ]
    assert len(rows) == 7
    # right-aligned numeric columns: every row ends at the same column
    assert len({len(r) for r in rows}) == 1
    assert "lower bound (LP relaxation" in out


def test_compare_json_subset(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare",
            "--scenario",
            "depot",
            "--sessions",
            "4",
            "--policies",
            "edf",
            "optimal",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [r["policy"] for r in payload["results"]] == ["edf", "optimal"]
    assert payload["results"][1]["gap_pct"] == pytest.approx(0.0, abs=0.5)
    assert payload["results"][1]["violations"] == 0


def test_run_with_input_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--input", str(EXAMPLE), "--policies", "llf", "mpc"])
    out = capsys.readouterr().out
    assert code == 0
    assert "office-example" in out
    assert "mpc" in out


def test_plot_writes_png(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "plots" / "p.png"
    code = main(
        [
            "plot",
            "--scenario",
            "residential",
            "--sessions",
            "6",
            "--pv-kwp",
            "10",
            "--policies",
            "uncontrolled",
            "optimal",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "wrote" in capsys.readouterr().out


def test_errors_exit_with_status_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--input", str(tmp_path / "missing.json")]) == 2
    assert "evcharge: error" in capsys.readouterr().err
    assert main(["compare", "--sessions", "0"]) == 2
    with pytest.raises(SystemExit):
        main(["compare", "--policies", "magic"])


def test_compare_phase_aware_site(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "compare",
            "--scenario",
            "residential",
            "--sessions",
            "6",
            "--grid",
            "IT",
            "--line-limit",
            "40",
            "--no-rotation",
            "--single-phase-share",
            "0.5",
            "--policies",
            "llf",
            "optimal",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "IT 40/40/40 A per line" in out
    header = next(line for line in out.splitlines() if line.startswith("policy"))
    assert "line %" in header


def test_json_output_has_no_nan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--sessions", "3", "--policies", "edf", "--format", "json"]) == 0
    text = capsys.readouterr().out
    assert "NaN" not in text
    assert json.loads(text)["results"][0]["max_line_loading_pct"] is None


def test_compare_with_v2g_reports_discharge_and_wear(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["compare", "--scenario", "residential", "--sessions", "4", "--v2g-share", "1"]
    assert main([*args, "--degradation", "0.05", "--policies", "llf", "optimal"]) == 0
    out = capsys.readouterr().out
    assert "4 V2G sessions (bidirectional)" in out
    lines = out.splitlines()
    header = re.split(r"\s{2,}", next(line for line in lines if line.startswith("policy")))
    at = header.index("V2G kWh")
    assert header[at : at + 3] == ["V2G kWh", "wear EUR", "total EUR"]
    llf = next(line for line in lines if line.startswith("llf")).split()
    assert llf[at] == "0.0"  # heuristics never discharge


def test_forecast_policies_learn_from_training_days_or_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["compare", "--sessions", "4", "--step-minutes", "60", "--train-days", "2"]
    assert main([*args, "--policies", "mpc-ev", "mpc-reserve", "mpc-saa"]) == 0
    out = capsys.readouterr().out
    assert all(f"\n{name} " in out for name in ("mpc-ev", "mpc-reserve", "mpc-saa"))
    # a scenario file has no synthetic history: --history is required
    assert main(["run", "--input", str(EXAMPLE), "--policies", "mpc-ev"]) == 2
    assert "need --history" in capsys.readouterr().err
    history = ["--history", str(EXAMPLE), str(EXAMPLE)]
    assert main(["run", "--input", str(EXAMPLE), "--policies", "mpc-reserve", *history]) == 0
    assert "\nmpc-reserve " in capsys.readouterr().out
