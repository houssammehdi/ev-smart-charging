"""A self-contained OCPP demo: a central system and simulated charge points in one process.

:func:`run_demo` starts the central system on a local port, connects ``n``
simulated charge points whose EVs plug in one after another, runs the control
loop on a shared accelerated clock and reports the site power it saw and the
energy every EV received. ``evcharge ocpp-demo`` is the command-line front end.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, replace

import numpy as np

from evcharge.ocpp.central import CentralSystem, CentralSystemConfig, Declared, serve
from evcharge.ocpp.chargepoint import EV, ChargePointLog, ScaledClock, SimulatedChargePoint


@dataclass(frozen=True)
class DemoResult:
    """What a demo run observed.

    Attributes:
        logs: Per charge point: its EV and its log.
        site_kw: Total EV power sampled once per control interval (kW).
        grid_limit_kw: The configured site limit.
        control_steps: Number of control steps run.
        planned_kw: Total EV power of every plan (kW).
    """

    logs: tuple[tuple[EV, ChargePointLog], ...]
    site_kw: tuple[float, ...]
    grid_limit_kw: float
    control_steps: int
    planned_kw: tuple[float, ...] = ()

    @property
    def max_site_kw(self) -> float:
        """Highest sampled total EV power."""
        return max(self.site_kw, default=0.0)

    def delivered_kwh(self) -> list[float]:
        """Metered energy per EV."""
        return [log.energy_wh / 1000.0 for _, log in self.logs]


def demo_fleet(n: int, seed: int = 1) -> list[EV]:
    """``n`` EVs with seeded needs (8 to 30 kWh) and stays (3 to 8 hours)."""
    rng = np.random.default_rng(seed)
    return [
        EV(
            energy_kwh=round(float(rng.uniform(8.0, 30.0)), 1),
            max_power_kw=11.0,
            phases=3,
            stay_s=float(rng.uniform(3.0, 8.0)) * 3600.0,
            id_tag=f"TAG{i + 1:03d}",
        )
        for i in range(n)
    ]


async def run_demo(
    config: CentralSystemConfig,
    fleet: list[EV],
    *,
    scale: float = 600.0,
    arrival_gap_s: float = 900.0,
    meter_interval_s: float = 300.0,
    port: int = 0,
) -> DemoResult:
    """Run the central system with one simulated charge point per EV.

    Args:
        config: Central-system settings; every EV's need is added to its
            ``id_tags`` as declared (energy and dwell), so the controller knows it.
        fleet: The EVs; EV ``i`` plugs in ``i * arrival_gap_s`` simulated seconds
            after the start.
        scale: Simulated seconds per real second.
        arrival_gap_s: Simulated seconds between plug-ins.
        meter_interval_s: Simulated seconds between MeterValues.
        port: TCP port (0: any free port).
    """
    tags = dict(config.id_tags)
    for ev in fleet:
        tags[ev.id_tag] = Declared(
            energy_kwh=ev.energy_kwh * config.defaults.efficiency,
            dwell_hours=ev.stay_s / 3600.0,
            phases=ev.phases,
            max_power_kw=ev.max_power_kw,
        )
    config = replace(config, id_tags=tags)
    clock = ScaledClock(scale)
    system = CentralSystem(config, clock=clock)
    server = await serve(system, "127.0.0.1", port)
    bound = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{bound}/ocpp"
    points = [
        SimulatedChargePoint(
            url,
            f"CP{i + 1:03d}",
            ev,
            clock=clock,
            meter_interval_s=meter_interval_s,
            plug_in_after_s=i * arrival_gap_s,
        )
        for i, ev in enumerate(fleet)
    ]
    stop = asyncio.Event()
    samples: list[float] = []

    async def sample() -> None:
        while not stop.is_set():
            samples.append(sum(p.log.power_kw for p in points))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), config.interval_s)

    control = asyncio.create_task(system.run_control(stop))
    sampler = asyncio.create_task(sample())
    try:
        logs = await asyncio.gather(*(p.run() for p in points))
    finally:
        stop.set()
        await asyncio.gather(control, sampler)
        server.close()
        await server.wait_closed()
    return DemoResult(
        logs=tuple(zip(fleet, logs, strict=True)),
        site_kw=tuple(samples),
        grid_limit_kw=config.grid_limit_kw,
        control_steps=system.steps,
        planned_kw=tuple(system.planned_kw),
    )
