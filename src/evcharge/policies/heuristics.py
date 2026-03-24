"""Real-time heuristic policies (no optimisation solver, O(n log n) per step).

All heuristics share the rules that real load-management controllers follow:

* commands never exceed the step's capacity rows: the site headroom (grid
  limit - base load + PV) and, on phase-aware sites, every line's current
  limit, so a line is never overloaded even when many single-phase EVs share it;
* a session is either paused or charged at no less than its minimum (IEC
  61851: 6 A), so heuristics decide *who* charges before *how much*;
* setpoints are multiples of the charger's resolution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from evcharge.model import FloatArray, Scenario
from evcharge.policies.base import (
    Observation,
    OnlinePolicy,
    SessionState,
    Setpoints,
    command_bounds,
    fair_fill,
    priority_fill,
)


class Uncontrolled(OnlinePolicy):
    """Plug-and-charge baseline: every EV charges at full power as soon as it arrives.

    The only control is the protection a static load balancer provides: when
    the site limit (or a line limit) is reached, EVs that plugged in earlier
    keep full power and later arrivals get what is left (first come, first
    served).
    """

    name = "uncontrolled"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Serve sessions in plug-in order at maximum power."""
        order = sorted(obs.pending(), key=lambda s: (s.session.arrival_step, s.id))
        return priority_fill(order, obs.allocation())


class EqualShare(OnlinePolicy):
    """Split the capacity equally between connected EVs (water-filling).

    EVs that need less than the equal share keep only what they can use and the
    rest is redistributed. On phase-aware sites the split is max-min fair over
    all lines: EVs on a saturated line stop rising while the others continue.
    When the capacity cannot give every EV its minimum, the least-served EVs
    (lowest delivered fraction) are admitted first, which rotates access over
    time.
    """

    name = "equal-share"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Water-fill the capacity over the largest admissible set of EVs."""
        pending = sorted(
            obs.pending(),
            key=lambda s: (s.delivered_fraction, s.session.departure_step, s.id),
        )
        bounds = [command_bounds(s) for s in pending]
        alloc = obs.allocation()
        # Admission is monotone: adding an EV can only lower the water level.
        admitted = len(pending)
        levels: list[float] = []
        while admitted > 0:
            caps = [b[1] for b in bounds[:admitted]]
            levels = fair_fill(pending[:admitted], caps, alloc)
            if all(a + 1e-9 >= b[0] and a > 0 for a, b in zip(levels, bounds, strict=False)):
                break
            admitted -= 1
        out: Setpoints = {}
        for st, a in zip(pending[:admitted], levels, strict=False):
            out[st.id] = a
            alloc.take(st.id, a)
        for st, (lowest, useful) in zip(pending[admitted:], bounds[admitted:], strict=True):
            room = st.control.snap_down(alloc.headroom(st.id))
            if room + 1e-9 >= max(lowest, 1e-9):
                cmd = min(useful, room)
                out[st.id] = cmd
                alloc.take(st.id, cmd)
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
        return priority_fill(order, obs.allocation())


class LeastLaxityFirst(OnlinePolicy):
    """Least laxity first: serve the EV with the least slack (time left minus time needed)."""

    name = "llf"

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Serve sessions in order of laxity."""
        order = sorted(obs.pending(), key=lambda s: (s.laxity_h, s.session.departure_step, s.id))
        return priority_fill(order, obs.allocation())


class PriceAware(OnlinePolicy):
    """Greedy valley filling: EVs book the cheapest free capacity in their window.

    At every step the plan is rebuilt from scratch: connected EVs, taken in
    least-laxity order, each reserve setpoints in the cheapest steps before
    their departure, using only the forecast capacity (on every row they load)
    not already reserved by EVs earlier in the order (ties go to the earlier
    step). Steps with PV surplus are valued at the export price, since charging
    then only costs the forgone feed-in. The reservations of the current step
    are executed.

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
    def _book(st: SessionState, coef: FloatArray, price: FloatArray, slack: FloatArray) -> float:
        """Reserve capacity for ``st`` in ``slack`` (updated in place); return step 0's booking."""
        ctl = st.control
        energy_per_unit = st.session.efficiency * ctl.kw_per_unit * st.dt_h
        deficit = st.remaining_kwh
        pos = coef > 0
        now = 0.0
        for offset in np.argsort(price[: st.steps_left], kind="stable"):
            k = int(offset)
            room = float(np.min(slack[k, pos] / coef[pos])) if pos.any() else math.inf
            room = ctl.snap_down(max(0.0, room))
            add = min(ctl.charge_max, room, ctl.snap_up(deficit / energy_per_unit))
            if add < ctl.charge_min:
                # Too little for the minimum current: book the minimum if it fits
                # (the EV stops by itself once full), otherwise skip the step.
                if room + 1e-9 < ctl.charge_min:
                    continue
                add = ctl.charge_min
            if add <= 1e-9:
                continue
            if k == 0:
                now = add
            slack[k] -= coef * add
            deficit -= add * energy_per_unit
            if deficit <= 1e-9:
                break
        return now

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Rebuild the reservation plan and return its first step."""
        pending = sorted(obs.pending(), key=lambda s: (s.laxity_h, s.session.departure_step, s.id))
        if not pending:
            return {}
        horizon = max(st.steps_left for st in pending)
        slack = np.array(self.scenario.rows.rhs[obs.step : obs.step + horizon], dtype=np.float64)
        price = self._price[obs.step : obs.step + horizon]
        rows = obs.rows
        out: Setpoints = {}
        for st in pending:
            now = self._book(st, rows.coefficient(st.id), price, slack)
            if now > 0.0:
                out[st.id] = now
        return out
