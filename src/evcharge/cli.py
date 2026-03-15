"""Command-line interface: ``evcharge compare | run | plot``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from evcharge import __version__
from evcharge.experiment import Comparison, compare, describe_scenario, format_table
from evcharge.io import load_scenario
from evcharge.model import Scenario, ValidationError
from evcharge.optim import SolverError
from evcharge.policies import POLICY_FACTORIES, Policy, make_policy
from evcharge.scenarios import DEFAULT_DEMAND_CHARGE_EUR_PER_KW, PROFILES, generate


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("synthetic scenario")
    g.add_argument("--scenario", choices=sorted(PROFILES), default="workplace")
    g.add_argument("--sessions", type=int, default=40, help="number of sessions (default 40)")
    g.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    g.add_argument(
        "--grid-limit",
        type=float,
        default=None,
        help="site import limit in kW (default: per profile)",
    )
    g.add_argument("--step-minutes", type=int, default=15, help="control step (default 15)")
    g.add_argument("--pv-kwp", type=float, default=0.0, help="installed PV in kWp (default 0)")
    g.add_argument(
        "--base-load-peak",
        type=float,
        default=None,
        help="base load peak in kW (default: per profile)",
    )
    g.add_argument(
        "--demand-charge",
        type=float,
        default=DEFAULT_DEMAND_CHARGE_EUR_PER_KW,
        help=f"EUR per kW of peak import (default {DEFAULT_DEMAND_CHARGE_EUR_PER_KW})",
    )


def _add_policy_args(p: argparse.ArgumentParser, default: Sequence[str]) -> None:
    p.add_argument(
        "--policies",
        nargs="+",
        choices=list(POLICY_FACTORIES),
        default=list(default),
        metavar="POLICY",
        help=f"policies to run (choices: {', '.join(POLICY_FACTORIES)}; "
        f"default: {' '.join(default)})",
    )


def _add_format_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=["table", "json"], default="table", help="output format")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="evcharge",
        description="Compare EV smart-charging policies on grid-constrained sites.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser("compare", help="compare policies on a synthetic scenario")
    _add_scenario_args(p_cmp)
    _add_policy_args(p_cmp, list(POLICY_FACTORIES))
    _add_format_arg(p_cmp)

    p_run = sub.add_parser("run", help="compare policies on a scenario JSON file")
    p_run.add_argument("--input", required=True, help="scenario JSON (see docs/input-format.md)")
    _add_policy_args(p_run, list(POLICY_FACTORIES))
    _add_format_arg(p_run)

    p_plot = sub.add_parser("plot", help="save a stacked power plot (needs the [plot] extra)")
    _add_scenario_args(p_plot)
    p_plot.add_argument("--input", help="scenario JSON instead of a synthetic scenario")
    _add_policy_args(p_plot, ["uncontrolled", "mpc", "optimal"])
    p_plot.add_argument("--output", required=True, help="image path, e.g. docs/workplace.png")
    p_plot.add_argument("--dpi", type=int, default=110, help="resolution (default 110)")
    return parser


def _scenario_from_args(args: argparse.Namespace) -> Scenario:
    if getattr(args, "input", None):
        return load_scenario(args.input)
    return generate(
        args.scenario,
        n_sessions=args.sessions,
        seed=args.seed,
        grid_limit_kw=args.grid_limit,
        step_minutes=args.step_minutes,
        pv_kwp=args.pv_kwp,
        base_load_peak_kw=args.base_load_peak,
        demand_charge_eur_per_kw=args.demand_charge,
    )


def _policies(names: Sequence[str]) -> list[Policy]:
    return [make_policy(n) for n in names]


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _print_comparison(comparison: Comparison, fmt: str) -> None:
    if fmt == "json":
        payload = {
            "scenario": describe_scenario(comparison.scenario),
            "lower_bound_eur": comparison.lower_bound_eur,
            "results": [
                {**m.as_dict(), "gap_pct": _finite_or_none(comparison.gap_pct(m))}
                for m in comparison.metrics
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(describe_scenario(comparison.scenario))
        print()
        print(format_table(comparison))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``evcharge`` command; returns the exit status."""
    args = build_parser().parse_args(argv)
    try:
        scenario = _scenario_from_args(args)
        policies = _policies(args.policies)
        if args.command == "plot":
            from evcharge.plotting import save_power_plot
            from evcharge.sim import simulate

            results = [simulate(scenario, p) for p in policies]
            out = save_power_plot(results, args.output, dpi=args.dpi)
            print(f"wrote {out}")
        else:
            _print_comparison(compare(scenario, policies), args.format)
    except (ValidationError, SolverError, OSError, ImportError) as exc:
        print(f"evcharge: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
