"""
Phase 4: Collapse physics and domino dynamics.

Province-level tipping, domino propagation, bridge province detection.
All instability structurally explainable; no arbitrary chaos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.state import RegimeState
from ..core.hierarchical_state import HierarchicalState, array_to_districts
from ..core.factions import recompute_system_state
from ..core.parameters import SystemParameters


# District array indices
I_UNREST = 1
I_ADMIN = 2

# Tipping threshold: province_unrest > this for 3 steps -> critical
UNREST_CRITICAL_THRESHOLD = 0.65
CONSECUTIVE_STEPS_FOR_CRITICAL = 3

# Domino: when province becomes critical, adjacent unrest += this * edge_weight
DOMINO_UNREST_BUMP = 0.1

# National shock when 2+ provinces critical
NATIONAL_SHOCK_HAZARD_BUMP = 0.15
NATIONAL_SHOCK_EXHAUSTION_BUMP = 0.05

# Bridge: provinces with external (between-province) links
BRIDGE_DEGREE_THRESHOLD = 1  # at least 1 inter-province link


def build_province_adjacency(hierarchical: HierarchicalState) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Province adjacency and edge weights from district adjacency.
    P_adj[p,q] = 1 if any district in p is linked to any district in q (p != q).
    P_weight[p,q] = sum of A_ij over i in p, j in q (for domino scaling).
    """
    n_p = hierarchical.n_provinces
    A = hierarchical.adjacency
    prov = hierarchical.province_of_district
    n_d = len(prov)
    P_adj = np.zeros((n_p, n_p), dtype=np.float64)
    P_weight = np.zeros((n_p, n_p), dtype=np.float64)
    for i in range(n_d):
        for j in range(n_d):
            if i == j:
                continue
            w = A[i, j]
            if w <= 0:
                continue
            pi, pj = int(prov[i]), int(prov[j])
            if pi != pj:
                P_adj[pi, pj] = 1.0
                P_weight[pi, pj] += w
    return P_adj, P_weight


def is_bridge_province(province_idx: int, hierarchical: HierarchicalState) -> bool:
    """True if this province has at least one district with a link to another province."""
    A = hierarchical.adjacency
    prov = hierarchical.province_of_district
    n_d = len(prov)
    p = province_idx
    for i in range(n_d):
        if prov[i] != p:
            continue
        for j in range(n_d):
            if A[i, j] > 0 and prov[j] != p:
                return True
    return False


def apply_domino_effects(
    state: RegimeState,
    critical_flags: NDArray[np.uint8],
    province_adjacency: NDArray[np.float64],
    province_weight: NDArray[np.float64],
    params: SystemParameters,
) -> RegimeState:
    """
    When a province is critical: increase unrest in adjacent provinces by
    DOMINO_UNREST_BUMP * edge_weight; reduce national stability (legitimacy) by 0.05 per critical.
    Returns new state. All state variables clipped to safe bounds.
    """
    if state.hierarchical is None:
        return state
    hier = state.hierarchical
    arr = hier.get_district_array().copy()
    prov = hier.province_of_district
    n_p = hier.n_provinces

    # For each critical province p, bump unrest in adjacent q
    for p in range(n_p):
        if not critical_flags[p]:
            continue
        for q in range(n_p):
            if p == q or province_adjacency[p, q] <= 0:
                continue
            edge_w = min(1.0, float(province_weight[p, q]))
            bump = DOMINO_UNREST_BUMP * edge_w
            mask = prov == q
            arr[mask, I_UNREST] = np.clip(arr[mask, I_UNREST] + bump, 0.0, 1.0)

    new_hier = hier.copy_with_districts(array_to_districts(arr))
    state = state.copy_with_hierarchical(new_hier)

    # Reduce national stability: one 0.05 hit per critical province (cap total)
    n_critical = int(np.sum(critical_flags))
    if n_critical > 0:
        stability_hit = min(0.25, 0.05 * n_critical)
        new_leg = float(np.clip(state.system.legitimacy - stability_hit, 0.0, 1.0))
        new_sys = state.system.__class__(
            legitimacy=new_leg,
            cohesion=state.system.cohesion,
            fragmentation=state.system.fragmentation,
            instability=state.system.instability,
            mobilization=state.system.mobilization,
            repression=state.system.repression,
            elite_alignment=state.system.elite_alignment,
            volatility=state.system.volatility,
            exhaustion=state.system.exhaustion,
            state_gdp=state.system.state_gdp,
            pillars=state.system.pillars,
        )
        state = state.copy_with_system(new_sys)

    return recompute_system_state(state, params)


def apply_national_shock(
    state: RegimeState,
    hazard_bump: float = NATIONAL_SHOCK_HAZARD_BUMP,
    exhaustion_bump: float = NATIONAL_SHOCK_EXHAUSTION_BUMP,
) -> RegimeState:
    """When 2+ provinces critical: add to volatility (hazard proxy) and exhaustion. Clipped."""
    vol = min(1.0, state.system.volatility + hazard_bump)
    exh = min(1.0, state.system.exhaustion + exhaustion_bump)
    new_sys = state.system.__class__(
        legitimacy=state.system.legitimacy,
        cohesion=state.system.cohesion,
        fragmentation=state.system.fragmentation,
        instability=state.system.instability,
        mobilization=state.system.mobilization,
        repression=state.system.repression,
        elite_alignment=state.system.elite_alignment,
        volatility=vol,
        exhaustion=exh,
        state_gdp=state.system.state_gdp,
        pillars=state.system.pillars,
    )
    return state.copy_with_system(new_sys)


def apply_exhaustion_admin_decay(
    state: RegimeState,
    params: SystemParameters,
) -> RegimeState:
    """If exhaustion > 0.5: admin_capacity *= (1 - exhaustion). Clipped."""
    if state.hierarchical is None:
        return state
    exh = state.system.exhaustion
    if exh <= 0.5:
        return state
    arr = state.hierarchical.get_district_array().copy()
    arr[:, I_ADMIN] = np.clip(arr[:, I_ADMIN] * (1.0 - exh), 0.0, 1.0)
    new_hier = state.hierarchical.copy_with_districts(array_to_districts(arr))
    return state.copy_with_hierarchical(new_hier)


def apply_exhaustion_unrest_drift(state: RegimeState) -> RegimeState:
    """If exhaustion > 0.7: all district unrest += 0.05. Clipped."""
    if state.hierarchical is None:
        return state
    if state.system.exhaustion <= 0.7:
        return state
    arr = state.hierarchical.get_district_array().copy()
    arr[:, I_UNREST] = np.clip(arr[:, I_UNREST] + 0.05, 0.0, 1.0)
    new_hier = state.hierarchical.copy_with_districts(array_to_districts(arr))
    return state.copy_with_hierarchical(new_hier)
