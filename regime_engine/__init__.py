"""
Adaptive Memory-Driven Regime Engine.

A research-grade political dynamics simulation with faction-centric state design,
formal stability guarantees, and a Gymnasium-compatible RL environment.

Public API:
    SystemParameters    — immutable parameter pack
    FactionState        — micro-state for one faction
    SystemState         — 9 derived macro variables
    RegimeState         — top-level state container
    SimulationRunner    — high-level trajectory runner
    RegimeEnv           — RL environment
    CrisisLevel         — crisis severity enumeration
"""

from .core.parameters import SystemParameters
from .core.state import FactionState, RegimeState, SystemState
from .core.factions import (
    create_balanced_factions,
    create_dominant_factions,
    recompute_system_state,
)
from .core.integrator import step, multi_step
from .simulation.runner import SimulationRunner
from .agents.rl_env import RegimeEnv
from .agents.action_space import Action, ActionType
from .systems.crisis_classifier import CrisisLevel, classify
from .systems.hazard import HazardParameters, compute_hazard
from .analysis.metrics import summary_statistics
from .analysis.logging import StateLogger

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
    "SimulationRunner",
    "RegimeEnv",
    "Action",
    "ActionType",
    "CrisisLevel",
    "classify",
    "HazardParameters",
    "compute_hazard",
    "summary_statistics",
    "StateLogger",
]
