"""Command-line interface: ``evcharge compare | run | plot | ocpp-server | ocpp-demo``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence

from evcharge import __version__
from evcharge.experiment import Comparison, compare, describe_scenario, format_table
from evcharge.forecast import ArrivalForecast
from evcharge.io import load_scenario
from evcharge.model import Scenario, ValidationError
from evcharge.optim import SolverError
from evcharge.policies import FORECAST_POLICY_FACTORIES, POLICY_FACTORIES, Policy, make_policy
from evcharge.scenarios import (
    DEFAULT_DEGRADATION_EUR_PER_KWH,
    DEFAULT_DEMAND_CHARGE_EUR_PER_KW,
    PROFILES,
    ScenarioOptions,
    generate,
    training_days,
)


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
    g.add_argument(
        "--grid",
        choices=["TN", "IT"],
        default=None,
        help="phase-aware site on a TN or IT grid (default: aggregate kW model)",
    )
    g.add_argument(
        "--line-limit",
        type=float,
        default=None,
        help="main fuse per line in A (default: the balanced equivalent of --grid-limit)",
    )
    g.add_argument(
        "--no-rotation",
        action="store_true",
        help="install every charger L1L2L3 instead of rotating phases",
    )
    g.add_argument(
        "--single-phase-share",
        type=float,
        default=None,
        help="share of single-phase EVs on phase-aware sites (default: per profile)",
    )
    g.add_argument(
        "--v2g-share",
        type=float,
        default=0.0,
        help="share of EVs with a battery model on a bidirectional charger (default 0)",
    )
    g.add_argument(
        "--degradation",
        type=float,
        default=DEFAULT_DEGRADATION_EUR_PER_KWH,
        help="battery wear of V2G EVs in EUR per kWh of throughput "
        f"(default {DEFAULT_DEGRADATION_EUR_PER_KWH})",
    )


def _add_policy_args(p: argparse.ArgumentParser, default: Sequence[str]) -> None:
    names = [*POLICY_FACTORIES, *FORECAST_POLICY_FACTORIES]
    p.add_argument(
        "--policies",
        nargs="+",
        choices=names,
        default=list(default),
        metavar="POLICY",
        help=f"policies to run (choices: {', '.join(names)}; default: {' '.join(default)})",
    )
    g = p.add_argument_group("forecast (mpc-reserve, mpc-ev, mpc-saa)")
    g.add_argument(
        "--train-days",
        type=int,
        default=20,
        help="synthetic training days for the forecast (seeds from 1000000; default 20)",
    )
    g.add_argument(
        "--history",
        nargs="+",
        metavar="FILE",
        help="past days as scenario JSON files to learn the forecast from",
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

    p_run = sub.add_parser("run", help="compare policies on a scenario JSON or ACN-Data file")
    source = p_run.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="scenario JSON (see docs/input-format.md)")
    source.add_argument("--acn", help="sessions in the Caltech ACN-Data JSON format")
    acn = p_run.add_argument_group("ACN-Data (with --acn)")
    acn.add_argument("--grid-limit", type=float, default=None, help="site limit in kW")
    acn.add_argument(
        "--acn-energy",
        choices=["delivered", "requested"],
        default="delivered",
        help="session energy: kWhDelivered or the driver's kWhRequested (default delivered)",
    )
    acn.add_argument(
        "--acn-departure",
        choices=["actual", "requested"],
        default="actual",
        help="departure: disconnectTime or the driver's requestedDeparture (default actual)",
    )
    acn.add_argument("--step-minutes", type=int, default=15, help="control step (default 15)")
    _add_policy_args(p_run, list(POLICY_FACTORIES))
    _add_format_arg(p_run)

    p_plot = sub.add_parser("plot", help="save a stacked power plot (needs the [plot] extra)")
    _add_scenario_args(p_plot)
    p_plot.add_argument("--input", help="scenario JSON instead of a synthetic scenario")
    _add_policy_args(p_plot, ["uncontrolled", "mpc", "optimal"])
    p_plot.add_argument("--output", required=True, help="image path, e.g. docs/workplace.png")
    p_plot.add_argument("--dpi", type=int, default=110, help="resolution (default 110)")

    p_srv = sub.add_parser(
        "ocpp-server", help="run an OCPP 1.6-J central system (needs the [ocpp] extra)"
    )
    p_srv.add_argument("--config", required=True, help="site and control settings (JSON)")
    p_srv.add_argument("--host", default="127.0.0.1", help="listen address (default 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=9000, help="TCP port (default 9000)")
    p_srv.add_argument("--prefix", default="ocpp", help="URL path before the charge point id")

    p_demo = sub.add_parser(
        "ocpp-demo",
        help="central system plus simulated charge points on an accelerated clock",
    )
    p_demo.add_argument("--chargers", type=int, default=10, help="charge points (default 10)")
    p_demo.add_argument("--grid-limit", type=float, default=22.0, help="site limit in kW")
    p_demo.add_argument(
        "--policy", choices=list(POLICY_FACTORIES), default="llf", help="policy (default llf)"
    )
    p_demo.add_argument(
        "--scale", type=float, default=600.0, help="simulated seconds per second (default 600)"
    )
    p_demo.add_argument("--seed", type=int, default=1, help="seed of the EV needs (default 1)")
    return parser


def _ocpp_server(args: argparse.Namespace) -> None:
    import asyncio
    import logging

    from evcharge.ocpp import CentralSystem, load_config, serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    system = CentralSystem(load_config(args.config))

    async def main() -> None:
        server = await serve(system, args.host, args.port, args.prefix)
        print(f"listening on ws://{args.host}:{args.port}/{args.prefix}/<chargePointId>")
        try:
            await system.run_control()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(main())


def _ocpp_demo(args: argparse.Namespace) -> None:
    import asyncio

    from evcharge.ocpp import CentralSystemConfig, run_demo
    from evcharge.ocpp.demo import demo_fleet

    if args.chargers < 1 or args.scale <= 0:
        raise ValidationError("--chargers must be >= 1 and --scale > 0")
    config = CentralSystemConfig(
        grid_limit_kw=args.grid_limit, policy=args.policy, interval_s=900.0 / args.scale
    )
    fleet = demo_fleet(args.chargers, args.seed)
    result = asyncio.run(run_demo(config, fleet, scale=args.scale))
    print(
        f"{len(fleet)} charge points, site limit {args.grid_limit:g} kW, policy {args.policy}, "
        f"{result.control_steps} control steps"
    )
    print(f"highest planned site power: {max(result.planned_kw, default=0.0):.1f} kW")
    print(f"highest sampled site power: {result.max_site_kw:.1f} kW")
    print(f"{'charge point':<13} {'need kWh':>9} {'stay h':>7} {'got kWh':>8} {'profiles':>9}")
    for i, (ev, log) in enumerate(result.logs):
        print(
            f"CP{i + 1:03d}{'':<8} {ev.energy_kwh:9.1f} {ev.stay_s / 3600:7.1f} "
            f"{log.energy_wh / 1000:8.1f} {len(log.limits):9d}"
        )


def _options(args: argparse.Namespace) -> ScenarioOptions:
    return ScenarioOptions(
        n_sessions=args.sessions,
        seed=args.seed,
        grid_limit_kw=args.grid_limit,
        step_minutes=args.step_minutes,
        pv_kwp=args.pv_kwp,
        base_load_peak_kw=args.base_load_peak,
        demand_charge_eur_per_kw=args.demand_charge,
        grid=args.grid,
        line_limit_a=args.line_limit,
        rotate_phases=not args.no_rotation,
        single_phase_share=args.single_phase_share,
        v2g_share=args.v2g_share,
        degradation_eur_per_kwh=args.degradation,
    )


def _scenario_from_args(args: argparse.Namespace) -> Scenario:
    if getattr(args, "acn", None):
        from evcharge.acn import load_acn

        if args.grid_limit is None:
            raise ValidationError("--acn needs --grid-limit (kW)")
        loaded = load_acn(
            args.acn,
            grid_limit_kw=args.grid_limit,
            step_minutes=args.step_minutes,
            energy=args.acn_energy,
            departure=args.acn_departure,
        )
        if loaded.dropped or loaded.capped:
            print(
                f"ACN-Data: {len(loaded.dropped)} sessions shorter than one step left out, "
                f"{len(loaded.capped)} capped at what fits their window",
                file=sys.stderr,
            )
        return loaded.scenario
    if getattr(args, "input", None):
        return load_scenario(args.input)
    return generate(args.scenario, **_options(args))


def _forecast(args: argparse.Namespace) -> ArrivalForecast:
    """The forecast of the forecast-aware policies: from --history, else synthetic days."""
    if args.history:
        return ArrivalForecast.fit([load_scenario(f) for f in args.history])
    if getattr(args, "input", None) or getattr(args, "acn", None):
        raise ValidationError("forecast-aware policies need --history for a scenario file")
    return ArrivalForecast.fit(training_days(args.scenario, args.train_days, **_options(args)))


def _policies(args: argparse.Namespace) -> list[Policy]:
    names: Sequence[str] = args.policies
    forecast = _forecast(args) if any(n in FORECAST_POLICY_FACTORIES for n in names) else None
    return [make_policy(n, forecast) for n in names]


def _json_value(value: float | int | str) -> float | int | str | None:
    """JSON has no NaN or infinity: map them to ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_comparison(comparison: Comparison, fmt: str) -> None:
    if fmt == "json":
        payload = {
            "scenario": describe_scenario(comparison.scenario),
            "lower_bound_eur": comparison.lower_bound_eur,
            "results": [
                {
                    **{k: _json_value(v) for k, v in m.as_dict().items()},
                    "gap_pct": _json_value(comparison.gap_pct(m)),
                }
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
        if args.command == "ocpp-server":
            _ocpp_server(args)
            return 0
        if args.command == "ocpp-demo":
            _ocpp_demo(args)
            return 0
        scenario = _scenario_from_args(args)
        policies = _policies(args)
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
