"""Key performance indicators of a simulation run.

Costs are *site-level*: they include the base load, because that is what the
grid connection is billed for. Differences between policies are therefore
differences in what the site pays.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from evcharge.model import FloatArray
from evcharge.optim import DEFAULT_UNMET_PENALTY_EUR_PER_KWH
from evcharge.sim import SimulationResult

COMPLETED_TOL_KWH = 0.01
"""A session counts as completed if at most this much energy is missing."""


def jain_index(values: FloatArray) -> float:
    r"""Jain's fairness index :math:`(\sum x)^2 / (n \sum x^2)` in ``[1/n, 1]``.

    Returns 1.0 for an empty input or when all values are zero (everyone is
    treated identically).
    """
    x = np.asarray(values, dtype=np.float64)
    sq = float(np.sum(x * x))
    if x.size == 0 or sq == 0.0:
        return 1.0
    return float(np.sum(x) ** 2 / (x.size * sq))


@dataclass(frozen=True)
class Metrics:
    """Summary of one run.

    Attributes:
        policy: Policy name.
        energy_requested_kwh: Sum of energy requests.
        energy_delivered_kwh: Requested energy that was delivered (requested minus
            unmet; for V2G sessions energy above the target does not count).
        delivered_pct: Delivered / requested, in percent.
        unmet_kwh: Requested but not delivered energy.
        sessions_completed_pct: Share of sessions missing at most 0.01 kWh.
        energy_cost_eur: Import cost minus export revenue (site level).
        peak_import_kw: Highest average import over one step (>= 0).
        demand_charge_eur: ``demand_charge * peak_import_kw``.
        total_cost_eur: Energy cost plus demand charge plus battery degradation.
        penalised_cost_eur: Total cost plus ``unmet_penalty * unmet_kwh``; the
            objective the optimisation policies minimise.
        jain_fairness: Jain's index of per-session delivered fractions (a
            session that requests nothing counts as fully served unless it
            leaves with less than it arrived with).
        capacity_utilisation_pct: EV energy drawn divided by the EV headroom
            energy available in steps with at least one EV connected.
        load_factor_pct: Mean import divided by peak import.
        violations: Number of corrected commands.
        runtime_s: Wall-clock simulation time (policy computation included).
        max_line_loading_pct: Highest modelled line current relative to its
            limit over all lines and steps (phase-aware sites; NaN otherwise).
        discharged_kwh: Grid-side energy discharged by EVs (V2G).
        degradation_cost_eur: Battery degradation of V2G sessions: their
            ``degradation_eur_per_kwh`` times the energy charged into and
            discharged from the battery (both directions count).
    """

    policy: str
    energy_requested_kwh: float
    energy_delivered_kwh: float
    delivered_pct: float
    unmet_kwh: float
    sessions_completed_pct: float
    energy_cost_eur: float
    peak_import_kw: float
    demand_charge_eur: float
    total_cost_eur: float
    penalised_cost_eur: float
    jain_fairness: float
    capacity_utilisation_pct: float
    load_factor_pct: float
    violations: int
    runtime_s: float
    max_line_loading_pct: float = math.nan
    discharged_kwh: float = 0.0
    degradation_cost_eur: float = 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        """Plain-dict view (JSON serialisable)."""
        return asdict(self)


def compute_metrics(
    result: SimulationResult,
    *,
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
) -> Metrics:
    """Compute :class:`Metrics` for a simulation result."""
    sc = result.scenario
    dt = sc.horizon.dt_h
    tariff = sc.tariff
    requested = np.array([s.energy_kwh for s in sc.sessions], dtype=np.float64)
    unmet = result.unmet_kwh
    req_total = float(requested.sum())
    del_total = float((requested - unmet).sum())

    imp = result.import_kw
    exp = result.export_kw
    energy_cost = float(
        np.sum(tariff.price_eur_per_kwh * imp * dt) - np.sum(tariff.export_price * exp * dt)
    )
    peak = float(max(0.0, imp.max(initial=0.0)))
    demand = tariff.demand_charge_eur_per_kw * peak
    charged = np.maximum(result.power_kw, 0.0).sum(axis=1) * dt
    discharged = np.maximum(-result.power_kw, 0.0).sum(axis=1) * dt
    degradation = 0.0
    for i, s in enumerate(sc.sessions):
        if s.v2g is not None and s.v2g.degradation_eur_per_kwh > 0.0:
            throughput = s.efficiency * charged[i] + discharged[i] / s.v2g.discharge_efficiency
            degradation += s.v2g.degradation_eur_per_kwh * float(throughput)
    total = energy_cost + demand + degradation

    connected = np.zeros(sc.horizon.n_steps, dtype=bool)
    for s in sc.sessions:
        connected[s.arrival_step : s.departure_step] = True
    available = float(np.sum(sc.ev_headroom_kw[connected]) * dt)
    ev_energy = float(charged.sum())
    met = unmet <= COMPLETED_TOL_KWH
    safe = np.where(requested > 0.0, requested, 1.0)
    fractions = np.where(requested > 0.0, 1.0 - unmet / safe, np.where(met, 1.0, 0.0))
    completed = float(np.mean(met) * 100.0) if requested.size else 100.0
    mean_import = float(imp.mean())
    loading = math.nan
    supply = sc.site.supply
    if supply is not None and result.line_current_a is not None:
        limits = np.array(supply.line_limit_a)
        loading = float((result.line_current_a / limits[None, :]).max(initial=0.0) * 100.0)

    return Metrics(
        policy=result.policy_name,
        energy_requested_kwh=req_total,
        energy_delivered_kwh=del_total,
        delivered_pct=100.0 * del_total / req_total if req_total > 0 else 100.0,
        unmet_kwh=float(unmet.sum()),
        sessions_completed_pct=completed,
        energy_cost_eur=energy_cost,
        peak_import_kw=peak,
        demand_charge_eur=demand,
        total_cost_eur=total,
        penalised_cost_eur=total + unmet_penalty * float(unmet.sum()),
        jain_fairness=jain_index(fractions),
        capacity_utilisation_pct=100.0 * ev_energy / available if available > 0 else 0.0,
        load_factor_pct=100.0 * mean_import / peak if peak > 0 else 0.0,
        violations=len(result.violations),
        runtime_s=result.runtime_s,
        max_line_loading_pct=loading,
        discharged_kwh=float(discharged.sum()),
        degradation_cost_eur=degradation,
    )
