"""Policy adapters."""

from __future__ import annotations

from collections.abc import Mapping

from evcharge.model import POWER_TOL_KW, Scenario
from evcharge.policies.base import Observation, Policy, SessionState, Setpoints


class PhaseBlind:
    """Run a policy as if the site had only an aggregate kW limit.

    The inner policy sees the kW-only view of the scenario
    (:meth:`~evcharge.model.Scenario.aggregate`) and a single kW row per step,
    which is how a load balancer that knows only the site's power limit sees a
    site. Its kW setpoints are converted to amperes per phase and rounded down
    to each charger's resolution. Where many single-phase EVs share a line,
    this overloads that line; the simulator then cuts the commands and records
    ``LINE_LIMIT`` violations. It exists to show why phase-aware control
    matters and is not a controller to deploy.

    Args:
        inner: The policy to run without phase information.
    """

    def __init__(self, inner: Policy) -> None:
        self.inner = inner

    @property
    def name(self) -> str:
        """Inner policy's name, marked as kW-only."""
        return f"{self.inner.name} (kW only)"

    @property
    def clairvoyant(self) -> bool:
        """Same foresight as the inner policy."""
        return self.inner.clairvoyant

    def reset(self, scenario: Scenario) -> None:
        """Hand the inner policy the kW-only view of the scenario."""
        self.inner.reset(scenario.aggregate())

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Decide in kW on one kW row, then convert to per-phase amperes."""
        kw_states = tuple(
            SessionState(
                session=st.session,
                delivered_kwh=st.delivered_kwh,
                p_min_kw=st.p_min_kw,
                p_max_kw=st.p_max_kw,
                steps_left=st.steps_left,
                dt_h=st.dt_h,
            )
            for st in obs.sessions
        )
        kw_obs = Observation(obs.step, kw_states, obs.headroom_kw, obs.peak_import_kw)
        kw = self.inner.decide(kw_obs)
        out: Setpoints = {}
        for st in obs.sessions:
            p = float(kw.get(st.id, 0.0))
            if p <= POWER_TOL_KW:
                continue
            ctl = st.control
            out[st.id] = max(ctl.charge_min, ctl.snap_down(p / ctl.kw_per_unit))
        return out

    def __repr__(self) -> str:
        return f"PhaseBlind({self.inner!r})"
