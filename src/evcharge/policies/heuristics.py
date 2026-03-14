"""Real-time heuristic policies (no optimisation solver, O(n log n) per step).

All heuristics share two rules that real load-management controllers follow:

* the sum of commands never exceeds the site headroom (grid limit - base load + PV);
* a session is either paused (0 kW) or charged at no less than its minimum
  power (IEC 61851: 6 A), so heuristics decide *who* charges before *how much*.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from evcharge.model import FloatArray, Scenario
from evcharge.policies.base import (
    Observation,
    OnlinePolicy,
    SessionState,
    Setpoints,
    command_bounds,
    priority_fill,
    water_fill,
)


class Uncontrolled(OnlinePolicy):
    """Plug-and-charge baseline: every EV charges at full power as soon as it arrives.

    The only control is the protection a static load balancer provides: when
    the site limit is reached, EVs that plugged in earlier keep full power and
    later arrivals get what is left (first come, first served).
    """

    name = "uncontrolled"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Serve sessions in plug-in order at maximum power."""
        order = sorted(obs.pending(), key=lambda s: (s.session.arrival_step, s.id))
        return priority_fill(order, obs.headroom_kw)


class EqualShare(OnlinePolicy):
    """Split the headroom equally between connected EVs (water-filling).

    EVs that need less than the equal share keep only what they can use and the
    rest is redistributed. When the headroom cannot give every EV its minimum
    power, the least-served EVs (lowest delivered fraction) are admitted first,
    which rotates access over time.
    """

    name = "equal-share"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Water-fill the headroom over the largest admissible set of EVs."""
        pending = sorted(
            obs.pending(),
            key=lambda s: (s.delivered_fraction, s.session.departure_step, s.id),
        )
        bounds = [command_bounds(s) for s in pending]
        budget = max(0.0, obs.headroom_kw)
        # Admission is monotone: adding an EV can only lower the water level.
        admitted = len(pending)
        while admitted > 0:
            caps = [b[1] for b in bounds[:admitted]]
            alloc = water_fill(caps, budget)
            if all(a + 1e-9 >= b[0] and a > 0 for a, b in zip(alloc, bounds, strict=False)):
                break
            admitted -= 1
        out: Setpoints = {}
        if admitted > 0:
            caps = [b[1] for b in bounds[:admitted]]
            for st, a in zip(pending, water_fill(caps, budget), strict=False):
                out[st.id] = a
        left = budget - sum(out.values())
        for st, (lowest, useful) in zip(pending[admitted:], bounds[admitted:], strict=True):
            if left + 1e-9 >= max(lowest, 1e-9):
                cmd = min(useful, left)
                out[st.id] = cmd
                left -= cmd
        return out


class EarliestDeadlineFirst(OnlinePolicy):
    """Earliest departure first: the EV leaving soonest gets full power first."""

    name = "edf"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Serve sessions in order of departure."""
        order = sorted(
            obs.pending(),
            key=lambda s: (s.session.departure_step, s.session.arrival_step, s.id),
        )
        return priority_fill(order, obs.headroom_kw)


class LeastLaxityFirst(OnlinePolicy):
    """Least laxity first: serve the EV with the least slack (time left minus time needed)."""

    name = "llf"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Serve sessions in order of laxity."""
        order = sorted(obs.pending(), key=lambda s: (s.laxity_h, s.session.departure_step, s.id))
        return priority_fill(order, obs.headroom_kw)


class PriceAware(OnlinePolicy):
    """Greedy valley filling: EVs book the cheapest free capacity in their window.

    At every step the plan is rebuilt from scratch: connected EVs, taken in
    least-laxity order, each reserve power in the cheapest steps before their
    departure, using only the forecast headroom not already reserved by EVs
    earlier in the order (ties go to the earlier step). Steps with PV surplus
    are valued at the export price, since charging then only costs the forgone
    feed-in. The reservations of the current step are executed.

    Each EV is planned once per step against the bookings of the EVs before it,
    so the greedy never trades slots between EVs and cannot anticipate EVs
    that have not arrived yet; both can leave energy undelivered.
    """

    name = "price-aware"

    def __init__(self) -> None:
        super().__init__()
        self._price: FloatArray = np.zeros(0)

    def reset(self, scenario: Scenario) -> None:
        """Pre-compute the effective price of every step."""
        super().reset(scenario)
        surplus = scenario.net_base_kw < 0
        self._price = np.asarray(
            np.where(surplus, scenario.tariff.export_price, scenario.tariff.price_eur_per_kwh),
            dtype=np.float64,
        )

    @staticmethod
    def _book(st: SessionState, price: FloatArray, free: FloatArray) -> float:
        """Reserve capacity for ``st`` in ``free`` (updated in place); return the first step."""
        booking = np.zeros(st.steps_left)
        energy_per_kw = st.session.efficiency * st.dt_h
        deficit = st.remaining_kwh
        for offset in np.argsort(price[: st.steps_left], kind="stable"):
            k = int(offset)
            add = min(st.p_max_kw, free[k], deficit / energy_per_kw)
            if add < st.p_min_kw:
                # Too little for the minimum current: book p_min if it fits
                # (the EV stops by itself once full), otherwise skip the step.
                if free[k] + 1e-9 < st.p_min_kw:
                    continue
                add = st.p_min_kw
            if add <= 1e-9:
                continue
            booking[k] = add
            free[k] -= add
            deficit -= add * energy_per_kw
            if deficit <= 1e-9:
                break
        return float(booking[0])

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Rebuild the reservation plan and return its first step."""
        pending = sorted(obs.pending(), key=lambda s: (s.laxity_h, s.session.departure_step, s.id))
        if not pending:
            return {}
        horizon = max(st.steps_left for st in pending)
        free = self.scenario.ev_headroom_kw[obs.step : obs.step + horizon].copy()
        price = self._price[obs.step : obs.step + horizon]
        out: Setpoints = {}
        for st in pending:
            now = self._book(st, price, free)
            if now > 0.0:
                out[st.id] = now
        return out
