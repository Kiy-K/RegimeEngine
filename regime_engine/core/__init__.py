"""Core ODE system, state, and integration components."""
from .parameters import SystemParameters
from .state import FactionState, SystemState, RegimeState
from .factions import (
    create_balanced_factions,
    create_dominant_factions,
    recompute_system_state,
)
from .integrator import step, multi_step

# Phase 3 hierarchical (optional)
from .hierarchical_state import (
    DistrictState,
    HierarchicalState,
    create_hierarchical_state,
)
from .topology import (
    build_adjacency_matrix,
    build_province_district_layout,
    diffusion_rate_bound,
)
from .hierarchical_obs import (
    aggregated_district_observation,
    hierarchical_observation_vector,
    top_k_unstable_districts,
)
from .hierarchical_coupling import get_geography_summary

__all__ = [
    "SystemParameters",
    "FactionState",
    "SystemState",
    "RegimeState",
    "create_balanced_factions",
    "create_dominant_factions",
    "recompute_system_state",
    "step",
    "multi_step",
    "DistrictState",
    "HierarchicalState",
    "create_hierarchical_state",
    "build_adjacency_matrix",
    "build_province_district_layout",
    "diffusion_rate_bound",
    "aggregated_district_observation",
    "hierarchical_observation_vector",
    "top_k_unstable_districts",
    "get_geography_summary",
]
