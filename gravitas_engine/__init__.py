"""
gravitas_engine — GRAVITAS: Governance Under Recursive And Volatile Instability
Through Adaptive Simulation.

A research-grade governance RL environment with partial observability, media
bias, hierarchical actions, Hawkes shock processes, and non-linear ODE dynamics.

Quick start
-----------
    import gravitas_engine                       # registers GravitasEnv-v0
    import gymnasium as gym

    env = gym.make("GravitasEnv-v0")             # via gymnasium registry
    # — or directly —
    from gravitas_engine import GravitasEnv, GravitasParams
    env = GravitasEnv(params=GravitasParams(seed=42), seed=42)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Primary API (GRAVITAS)
----------------------
    GravitasEnv         — main Gymnasium environment
    GravitasParams      — immutable hyperparameter dataclass
    GravitasWorld       — full world state container
    ClusterState        — per-cluster state (σ, h, r, m, τ, p)
    GlobalState         — global state (E, Φ, Π, Ψ, M, T)
    HierarchicalAction  — decoded action container
    Stance              — action stance enum

Legacy API (regime engine)
--------------------------
    SystemParameters    — legacy parameter pack
    RegimeState         — legacy top-level state
    FactionState        — legacy faction micro-state
    SimulationRunner    — high-level trajectory runner
    RegimeEnv           — legacy RL environment
"""

from __future__ import annotations

# ── Primary GRAVITAS API ─────────────────────────────────────────────────── #
from .agents.gravitas_env import GravitasEnv
from .core.gravitas_params import GravitasParams
from .core.gravitas_state import ClusterState, GlobalState, GravitasWorld
from .agents.gravitas_actions import HierarchicalAction, Stance

# ── Legacy regime-engine API ─────────────────────────────────────────────── #
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
from .analysis.metrics import summary_statistics
from .analysis.logging import StateLogger

# ── Gymnasium registration ───────────────────────────────────────────────── #
try:
    import gymnasium as gym
    gym.register(
        id="GravitasEnv-v0",
        entry_point="gravitas_engine.agents.gravitas_env:GravitasEnv",
        kwargs={},
    )
except Exception:
    pass  # gymnasium not installed or already registered

__version__ = "2.0.0"

__all__ = [
    # GRAVITAS primary
    "GravitasEnv",
    "GravitasParams",
    "GravitasWorld",
    "ClusterState",
    "GlobalState",
    "HierarchicalAction",
    "Stance",
    # Legacy
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
    "summary_statistics",
    "StateLogger",
    "__version__",
]
