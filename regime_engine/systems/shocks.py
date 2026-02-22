"""
Spatial shock injection for survival RL.

Types: economic, elite_defection, infrastructure_collapse, external_pressure.
Randomize location (province/district), intensity, duration.
Phase 4: propagate to neighbors; intensity > 0.7 -> instability_counter bump.
"""

from __future__ import annotations

from enum import IntEnum, unique
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.state import RegimeState
from ..core.hierarchical_state import HierarchicalState, array_to_districts
from ..core.parameters import SystemParameters
from .collapse_physics import build_province_adjacency


@unique
class ShockType(IntEnum):
    ECONOMIC = 0
    ELITE_DEFECTION = 1
    INFRASTRUCTURE_COLLAPSE = 2
    EXTERNAL_PRESSURE = 3


def inject_shock(
    state: RegimeState,
    params: SystemParameters,
    rng: np.random.Generator,
    shock_type: ShockType,
    location: Optional[int] = None,
    intensity: Optional[float] = None,
) -> Tuple[RegimeState, Dict[str, Any]]:
    """
    Apply one shock; location = province index (or None for global). Intensity in (0, 1].
    Returns (new_state, shock_info). shock_info["instability_counter_province"] = province index
    if intensity > 0.7 and shock was province-level (caller should bump that province's counter).
    """
    if intensity is None:
        intensity = float(rng.uniform(0.2, 0.6))
    intensity = np.clip(intensity, 0.0, 1.0)
    info: Dict[str, Any] = {"instability_counter_province": None}

    if shock_type == ShockType.ECONOMIC:
        return _shock_economic(state, params, rng, intensity, location, info)
    if shock_type == ShockType.ELITE_DEFECTION:
        return _shock_elite_defection(state, params, intensity)
    if shock_type == ShockType.INFRASTRUCTURE_COLLAPSE:
        return _shock_infrastructure(state, params, rng, location, intensity, info)
    if shock_type == ShockType.EXTERNAL_PRESSURE:
        return _shock_external_pressure(state, params, intensity)
    return state, info


def _shock_economic(
    state: RegimeState,
    params: SystemParameters,
    rng: np.random.Generator,
    intensity: float,
    location: Optional[int],
    info: Dict[str, Any],
) -> Tuple[RegimeState, Dict[str, Any]]:
    """Global GDP and wealth drop; district local_gdp. Phase 4: if province location, add unrest + propagate."""
    from ..core.factions import recompute_system_state

    gdp = max(0.0, state.system.state_gdp - 0.25 * intensity)
    pillars = tuple(float(np.clip(p - 0.1 * intensity, 0.0, 1.0)) for p in state.system.pillars)
    factions = [
        f.copy_with(wealth=float(np.clip(f.wealth - 0.2 * intensity, 0.0, 1.0)))
        for f in state.factions
    ]
    new_sys = state.system.__class__(
        legitimacy=state.system.legitimacy,
        cohesion=state.system.cohesion,
        fragmentation=state.system.fragmentation,
        instability=state.system.instability,
        mobilization=state.system.mobilization,
        repression=state.system.repression,
        elite_alignment=state.system.elite_alignment,
        volatility=state.system.volatility,
        exhaustion=state.system.exhaustion,
        state_gdp=gdp,
        pillars=pillars,
    )
    out = RegimeState(
        factions=factions,
        system=new_sys,
        affinity_matrix=state.affinity_matrix,
        step=state.step,
        hierarchical=state.hierarchical,
    )
    if state.hierarchical is not None:
        arr = state.hierarchical.get_district_array().copy()
        prov = state.hierarchical.province_of_district
        n_p = state.hierarchical.n_provinces
        arr[:, 0] = np.clip(arr[:, 0] - 0.2 * intensity, 0.0, 1.0)
        p = location if location is not None and 0 <= location < n_p else int(rng.integers(0, n_p))
        arr[prov == p, 1] = np.clip(arr[prov == p, 1] + intensity, 0.0, 1.0)
        P_adj, P_weight = build_province_adjacency(state.hierarchical)
        for q in range(n_p):
            if p == q or P_adj[p, q] <= 0:
                continue
            ew = min(1.0, float(P_weight[p, q]))
            arr[prov == q, 1] = np.clip(arr[prov == q, 1] + intensity * 0.5 * ew, 0.0, 1.0)
        if intensity > 0.7:
            info["instability_counter_province"] = p
        out = out.copy_with_hierarchical(
            state.hierarchical.copy_with_districts(array_to_districts(arr))
        )
    return recompute_system_state(out, params), info


def _shock_elite_defection(state: RegimeState, params: SystemParameters, intensity: float) -> Tuple[RegimeState, Dict[str, Any]]:
    """Cohesion hit to dominant faction; fragmentation pressure."""
    from ..core.factions import recompute_system_state

    info = {}
    powers = np.array([f.power for f in state.factions])
    dominant_idx = int(np.argmax(powers))
    factions = list(state.factions)
    f = factions[dominant_idx]
    factions[dominant_idx] = f.copy_with(
        cohesion=float(np.clip(f.cohesion - 0.35 * intensity, 0.0, 1.0))
    )
    new_sys = state.system.__class__(
        legitimacy=state.system.legitimacy,
        cohesion=state.system.cohesion,
        fragmentation=min(1.0 - 1e-6, state.system.fragmentation + 0.15 * intensity),
        instability=state.system.instability,
        mobilization=state.system.mobilization,
        repression=state.system.repression,
        elite_alignment=state.system.elite_alignment,
        volatility=state.system.volatility,
        exhaustion=state.system.exhaustion,
        state_gdp=state.system.state_gdp,
        pillars=state.system.pillars,
    )
    return recompute_system_state(
        state.copy_with_factions(factions).copy_with_system(new_sys), params
    ), info


def _shock_infrastructure(
    state: RegimeState,
    params: SystemParameters,
    rng: np.random.Generator,
    province: Optional[int],
    intensity: float,
    info: Dict[str, Any],
) -> Tuple[RegimeState, Dict[str, Any]]:
    """Admin capacity and local_gdp drop in one province; Phase 4: add unrest + propagate to neighbors."""
    from ..core.factions import recompute_system_state

    if state.hierarchical is None:
        return _shock_economic(state, params, rng, intensity, province, info)
    arr = state.hierarchical.get_district_array().copy()
    prov = state.hierarchical.province_of_district
    n_p = state.hierarchical.n_provinces
    p = province if province is not None and 0 <= province < n_p else int(rng.integers(0, n_p))
    mask = prov == p
    arr[mask, 0] = np.clip(arr[mask, 0] - 0.3 * intensity, 0.0, 1.0)  # local_gdp
    arr[mask, 2] = np.clip(arr[mask, 2] - 0.25 * intensity, 0.0, 1.0)  # admin_capacity
    arr[mask, 1] = np.clip(arr[mask, 1] + intensity, 0.0, 1.0)  # unrest in hit province
    P_adj, P_weight = build_province_adjacency(state.hierarchical)
    for q in range(n_p):
        if p == q or P_adj[p, q] <= 0:
            continue
        ew = min(1.0, float(P_weight[p, q]))
        arr[prov == q, 1] = np.clip(arr[prov == q, 1] + intensity * 0.5 * ew, 0.0, 1.0)
    if intensity > 0.7:
        info["instability_counter_province"] = p
    new_hier = state.hierarchical.copy_with_districts(array_to_districts(arr))
    return recompute_system_state(state.copy_with_hierarchical(new_hier), params), info


def _shock_external_pressure(state: RegimeState, params: SystemParameters, intensity: float) -> Tuple[RegimeState, Dict[str, Any]]:
    """Volatility and instability rise; exhaustion accumulation."""
    from ..core.factions import recompute_system_state

    info = {}
    new_sys = state.system.__class__(
        legitimacy=state.system.legitimacy,
        cohesion=state.system.cohesion,
        fragmentation=state.system.fragmentation,
        instability=min(1.0, state.system.instability + 0.2 * intensity),
        mobilization=state.system.mobilization,
        repression=state.system.repression,
        elite_alignment=state.system.elite_alignment,
        volatility=min(1.0, state.system.volatility + 0.25 * intensity),
        exhaustion=min(1.0, state.system.exhaustion + 0.1 * intensity),
        state_gdp=state.system.state_gdp,
        pillars=state.system.pillars,
    )
    return recompute_system_state(state.copy_with_system(new_sys), params), info


def maybe_inject_shock(
    state: RegimeState,
    params: SystemParameters,
    rng: np.random.Generator,
    shock_prob: float = 0.02,
) -> Tuple[RegimeState, bool, Dict[str, Any]]:
    """
    With probability shock_prob, pick random shock type/location/intensity and apply.
    Returns (new_state, shock_applied, shock_info).
    shock_info["instability_counter_province"] set if intensity > 0.7 and province shock.
    """
    if rng.random() >= shock_prob:
        return state, False, {}
    shock_type = ShockType(rng.integers(0, len(ShockType)))
    location = rng.integers(0, 10) if state.hierarchical is not None else None
    if state.hierarchical is not None and location >= state.hierarchical.n_provinces:
        location = state.hierarchical.n_provinces - 1
    new_state, shock_info = inject_shock(state, params, rng, shock_type, location=location)
    return new_state, True, shock_info
