"""Shared helpers for building small, hand-checkable scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from evcharge.model import Charger, Horizon, Scenario, Session, Site, Tariff

START = datetime(2026, 1, 5, 0, 0)


def make_scenario(
    sessions: Sequence[Session],
    *,
    n_steps: int = 4,
    step_minutes: int = 60,
    grid_limit_kw: float = 100.0,
    prices: Sequence[float] | None = None,
    export_prices: Sequence[float] | None = None,
    demand_charge: float = 0.0,
    base_load: Sequence[float] | None = None,
    pv: Sequence[float] | None = None,
    chargers: Sequence[Charger] | None = None,
    charger_kw: float = 11.0,
    charger_min_kw: float = 0.0,
) -> Scenario:
    """Build a scenario; by default one charger per referenced charger id."""
    if chargers is None:
        ids = sorted({s.charger_id for s in sessions}) or ["C1"]
        chargers = [Charger(i, charger_kw, charger_min_kw) for i in ids]
    return Scenario(
        name="test",
        horizon=Horizon(START, n_steps, step_minutes),
        site=Site(grid_limit_kw, tuple(chargers)),
        tariff=Tariff(
            np.asarray(prices if prices is not None else [0.1] * n_steps, dtype=float),
            None if export_prices is None else np.asarray(export_prices, dtype=float),
            demand_charge,
        ),
        sessions=tuple(sessions),
        base_load_kw=None if base_load is None else np.asarray(base_load, dtype=float),
        pv_kw=None if pv is None else np.asarray(pv, dtype=float),
    )


def session(
    sid: str,
    charger: str,
    arrival: int,
    departure: int,
    energy: float,
    *,
    max_kw: float = 11.0,
    efficiency: float = 1.0,
    min_kw: float | None = None,
) -> Session:
    """Terse session constructor for tests."""
    return Session(sid, charger, arrival, departure, energy, max_kw, efficiency, min_kw)
