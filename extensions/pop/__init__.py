"""
gravitas_engine.extensions.pop — Vectorized population subsystem for GRAVITAS.

Inspired by Paradox's Clausewitz Engine population mechanics:
  populations are demographic distributions (vectors), not agents.
  All operations are dot products — O(N*P) per step, not O(N*pop_count).

Categorized along four axes:
  Job      : 8 archetypes (SUBSISTENCE, YEOMAN, URBAN_LABORER, ARTISAN,
                            CLERK, MERCHANT, PROFESSIONAL, ELITE)
  Class    : lower / middle / upper (derived from archetype)
  Earning  : per-archetype income vector, mean-normalized
  Ethnicity: per-cluster ethnic share simplex + cultural distance matrix

Public API:
    PopParams      — all hyperparameters for the pop subsystem
    PopWrapper     — gymnasium.Wrapper integrating pops with GravitasEnv
    PopVector      — demographic state for one cluster
    WorldPopState  — collection of PopVectors across all clusters
    PopAggregates  — derived scalars (gini, satisfaction, radical_mass, ...)

Usage:
    from regime_engine.agents.gravitas_env import GravitasEnv
    from gravitas_engine.extensions.pop import PopWrapper, PopParams

    env = GravitasEnv(params=GravitasParams(seed=0), seed=0)
    env = PopWrapper(env, pop_params=PopParams(), seed=0)

    obs, info = env.reset()
    # obs is now original_dim + 5*max_N
    # info["pop"] contains mean demographic aggregates each step
"""

from .pop_params import (
    PopParams,
    N_ARCHETYPES,
    ARCHETYPE_NAMES,
    ARCHETYPE_BASE_INCOME,
    ARCHETYPE_RAD_POTENTIAL,
    ARCHETYPE_POLITICAL_WEIGHT,
    ARCHETYPE_CLASS,
)
from .pop_state import (
    PopVector,
    PopAggregates,
    WorldPopState,
    compute_aggregates,
    initialize_world_pop,
    initialize_pop_vector,
)
from .pop_dynamics import (
    step_pop_vector,
    step_world_pop,
    apply_action_to_pop,
    pop_to_ode_drivers,
    STABILIZE, MILITARIZE, REFORM, PROPAGANDA, INVEST, DECENTRALIZE,
)
from .pop_wrapper import PopWrapper

__all__ = [
    # Params
    "PopParams",
    "N_ARCHETYPES",
    "ARCHETYPE_NAMES",
    "ARCHETYPE_BASE_INCOME",
    "ARCHETYPE_RAD_POTENTIAL",
    "ARCHETYPE_POLITICAL_WEIGHT",
    "ARCHETYPE_CLASS",
    # State
    "PopVector",
    "PopAggregates",
    "WorldPopState",
    "compute_aggregates",
    "initialize_world_pop",
    "initialize_pop_vector",
    # Dynamics
    "step_pop_vector",
    "step_world_pop",
    "apply_action_to_pop",
    "pop_to_ode_drivers",
    # Action constants
    "STABILIZE", "MILITARIZE", "REFORM", "PROPAGANDA", "INVEST", "DECENTRALIZE",
    # Wrapper
    "PopWrapper",
]
