"""
Gymnasium-compatible Reinforcement Learning environment.

The RegimeEnv wraps the regime engine into a standard Gymnasium environment so
that RL agents can interact through the canonical step/reset API.

Observation space:
    A flat float32 vector of length (n_factions * 4 + 9):
      - n_factions × 4 micro-state values (P, Rad, Coh, Mem)
      - 9 macro-state values (L, C, F, I, M, R, E, V, Exh)

Action space:
    Discrete: (n_factions × n_action_types) actions encoded as integer indices.
    Action i maps to (faction = i // n_action_types, type = i % n_action_types).

Reward signal:
    r = w_L · L − w_I · I − w_V · V − w_F · F − w_Exh · Exh

Episode termination:
    - CrisisLevel.COLLAPSE reached
    - Exhaustion ≥ 0.95
    - max_steps reached

Truncation:
    - max_steps reached first without collapse
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from ..core.factions import create_balanced_factions, recompute_system_state
from ..core.hierarchical_obs import hierarchical_observation_vector
from ..core.hierarchical_state import create_hierarchical_state
from ..core.integrator import step as rk4_step
from ..core.parameters import SystemParameters
from ..core.state import RegimeState, SystemState
from ..systems.crisis_classifier import ClassifierThresholds, CrisisLevel, classify
from ..systems.events import check_and_apply_events
from .action_space import (
    MAX_INTENSITY,
    Action,
    ActionType,
    action_space_size,
    apply_action,
    null_action,
)

# Number of micro-state variables per faction (P, Rad, Coh, Mem, Wealth)
_MICRO_DIM: int = 5
# Number of base macro-state variables (L, C, F, I, M, R, E, V, Exh, GDP)
_MACRO_DIM: int = 10


class RegimeEnv(gym.Env):
    """Gymnasium-style environment for regime dynamics.

    Attributes:
        params:          System parameters for the engine.
        hazard_thresholds: ClassifierThresholds for crisis detection.
        w_legitimacy:    Reward weight for legitimacy (positive).
        w_instability:   Reward weight for instability (negative).
        w_volatility:    Reward weight for volatility (negative).
        w_fragmentation: Reward weight for fragmentation (negative).
        w_exhaustion:    Reward weight for exhaustion (negative).
        intensity:       Intensity of all agent actions.
        agent_id:        Identifier for the agent.
    """

    # Gymnasium metadata (no gymnasium dependency required — duck-typing)
    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        params: Optional[SystemParameters] = None,
        thresholds: Optional[ClassifierThresholds] = None,
        w_legitimacy: float = 1.0,
        w_instability: float = 0.8,
        w_volatility: float = 0.6,
        w_fragmentation: float = 0.4,
        w_exhaustion: float = 0.5,
        intensity: float = 0.10,
        agent_id: str = "agent_0",
    ) -> None:
        """Initialise the environment.

        Args:
            params:         System parameters (defaults to SystemParameters()).
            thresholds:     Crisis classifier thresholds.
            w_legitimacy:   Positive reward weight for L.
            w_instability:  Penalty weight for I.
            w_volatility:   Penalty weight for V.
            w_fragmentation: Penalty weight for F.
            w_exhaustion:   Penalty weight for Exh.
            intensity:      Action impulse magnitude ∈ [0, MAX_INTENSITY].
            agent_id:       Agent name string.
        """
        self.params: SystemParameters = params if params is not None else SystemParameters()
        self.thresholds: ClassifierThresholds = (
            thresholds if thresholds is not None else ClassifierThresholds()
        )
        self.w_legitimacy = w_legitimacy
        self.w_instability = w_instability
        self.w_volatility = w_volatility
        self.w_fragmentation = w_fragmentation
        self.w_exhaustion = w_exhaustion

        if not 0.0 <= intensity <= MAX_INTENSITY:
            raise ValueError(
                f"intensity must be in [0, {MAX_INTENSITY}], got {intensity}"
            )
        self.intensity = intensity
        self.agent_id = agent_id

        self._n_factions: int = self.params.n_factions
        self._n_action_types: int = action_space_size()

        # Phase 3: hierarchical district observation length (0 if disabled)
        self._hier_obs_dim: int = 0
        if getattr(self.params, "use_hierarchy", False):
            from ..core.hierarchical_obs import observation_space_dim_hierarchical
            self._hier_obs_dim = observation_space_dim_hierarchical(
                include_summary=True, include_top_k=False
            )

        # Micro + Macro + Pillars + Affinity Matrix + optional district summary
        self._obs_dim: int = (
            self._n_factions * _MICRO_DIM
            + _MACRO_DIM
            + self.params.n_pillars
            + self._n_factions ** 2
            + self._hier_obs_dim
        )
        self._total_actions: int = self._n_factions * self._n_action_types

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._total_actions)

        self._state: Optional[RegimeState] = None
        self._rng: np.random.Generator = np.random.default_rng(self.params.seed)

    def update_config(self, **kwargs):
        """Update environment configuration (replaces frozen dataclass)."""
        from dataclasses import replace
        valid_kwargs = {k: v for k, v in kwargs.items() if hasattr(self.params, k)}
        if valid_kwargs:
            self.params = replace(self.params, **valid_kwargs)

    # ------------------------------------------------------------------ #
    # Space descriptions (gymnasium-compatible shapes)                     #
    # ------------------------------------------------------------------ #

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Shape of the observation vector."""
        return (self._obs_dim,)

    @property
    def action_space_n(self) -> int:
        """Total number of discrete actions."""
        return self._total_actions

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: If provided, resets the internal RNG with this seed.
            options: Additional environment options (unused).

        Returns:
            (observation, info_dict)
        """
        if seed is not None:
            super().reset(seed=seed)
            import random
            random.seed(seed)
            np.random.seed(seed)
            self._rng = np.random.default_rng(seed)

        factions = create_balanced_factions(self._n_factions)
        placeholder = SystemState.neutral(self.params.n_pillars)
        aff_mat = tuple(tuple(1.0 if i == j else 0.0 for j in range(self._n_factions)) for i in range(self._n_factions))
        state = RegimeState(factions=factions, system=placeholder, affinity_matrix=aff_mat, step=0)
        if getattr(self.params, "use_hierarchy", False):
            n_pol = self.params.n_pillars
            hier = create_hierarchical_state(
                self.params.n_provinces,
                self.params.districts_per_province,
                n_pol,
            )
            state = state.copy_with_hierarchical(hier)
        self._state = recompute_system_state(state, self.params)

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Apply an action and advance the environment by one timestep.

        Args:
            action: Integer action index in [0, action_space_n).

        Returns:
            (observation, reward, terminated, truncated, info_dict)
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if not 0 <= action < self._total_actions:
            raise ValueError(
                f"Action {action} out of range [0, {self._total_actions})"
            )

        # Decode action index
        faction_idx = action // self._n_action_types
        action_type = ActionType(action % self._n_action_types)

        agent_action = Action(
            action_type=action_type,
            actor_idx=0,  # Assume the RL agent plays as Faction 0
            target_idx=faction_idx,
            intensity=self.intensity,
            agent_id=self.agent_id,
        )

        # Apply action impulse THEN integrate THEN shocks
        state_after_action = apply_action(self._state, agent_action)
        state_after_action = recompute_system_state(state_after_action, self.params)
        integrated = rk4_step(state_after_action, self.params, self._rng)
        self._state = check_and_apply_events(integrated, self.params)

        reward = self._compute_reward()
        crisis = classify(self._state, self.thresholds)
        terminated = self._is_terminated(crisis)
        truncated = self._state.step >= self.params.max_steps and not terminated

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> NDArray[np.float32]:
        """Construct and return the observation vector."""
        assert self._state is not None
        micro_parts = [f.to_array() for f in self._state.factions]
        macro_part = self._state.system.to_array()
        aff_flat = np.array(self._state.affinity_matrix, dtype=np.float64).flatten()
        obs = np.concatenate(micro_parts + [macro_part, aff_flat], axis=0)
        if self._hier_obs_dim > 0:
            hier_obs = hierarchical_observation_vector(
                getattr(self._state, "hierarchical", None),
                include_summary=True,
                include_top_k=False,
            )
            obs = np.concatenate([obs, hier_obs.astype(np.float32)], axis=0)
        return obs.astype(np.float32)

    def _compute_reward(self) -> float:
        """r = w_L·L − w_I·I − w_V·V − w_F·F − w_Exh·Exh."""
        assert self._state is not None
        sys = self._state.system
        return float(
            self.w_legitimacy * sys.legitimacy
            - self.w_instability * sys.instability
            - self.w_volatility * sys.volatility
            - self.w_fragmentation * sys.fragmentation
            - self.w_exhaustion * sys.exhaustion
        )

    def _is_terminated(self, crisis: CrisisLevel) -> bool:
        """End episode on collapse or critical exhaustion."""
        assert self._state is not None
        return (
            crisis == CrisisLevel.COLLAPSE
            or self._state.system.exhaustion >= 0.95
        )

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info dictionary."""
        assert self._state is not None
        crisis = classify(self._state, self.thresholds)
        return {
            "step": self._state.step,
            "crisis_level": crisis.name,
            "crisis_int": int(crisis),
            "exhaustion": self._state.system.exhaustion,
            "legitimacy": self._state.system.legitimacy,
            "instability": self._state.system.instability,
            "volatility": self._state.system.volatility,
        }

    def current_state(self) -> RegimeState:
        """Return the current RegimeState (read-only view).

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._state