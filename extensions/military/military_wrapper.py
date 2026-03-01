"""
military_wrapper.py — Gymnasium wrapper integrating military extension with GravitasEnv.

This wrapper adds tactical military operations to GravitasEngine while maintaining
compatibility with the existing military presence system.

Key features:
  - Extends observation space with military unit information
  - Adds military-specific actions
  - Integrates military rewards with base rewards
  - Maintains compatibility with existing military dynamics
  - Provides victory condition tracking

The military extension works alongside the existing military system:
  - Units contribute to cluster military presence values
  - Unit movement affects cluster stability
  - Unit supply consumption affects cluster resources
  - Military objectives provide additional victory conditions
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import gymnasium_shim as gym
    from gymnasium_shim import spaces

from .military_state import (
    StandardizedUnitParams as MilitaryUnitParams,
    StandardizedMilitaryUnit as MilitaryUnit,
    StandardizedClusterMilitaryState as ClusterMilitaryState,
    StandardizedWorldMilitaryState as WorldMilitaryState,
    StandardizedMilitaryObjective as MilitaryObjective,
    initialize_standardized_military_state as initialize_military_state
)
from .unit_types import MilitaryUnitType
from .military_dynamics import (
    step_military_units, apply_military_action,
    compute_military_reward, check_victory_conditions
)
from .political_interface import (
    compute_military_political_feedback,
    apply_political_military_feedback,
)
# build_adjacency_matrix removed (unused and caused slow imports)

# ─────────────────────────────────────────────────────────────────────────── #
# Military Action Space                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class MilitaryActionSpace:
    """
    Custom action space for military operations.

    Supports both discrete and hierarchical action formats.
    """

    def __init__(self, max_clusters: int = 12):
        self.max_clusters = max_clusters
        self.unit_types = list(MilitaryUnitType)
        self.action_types = ["deploy", "move", "attack", "reinforce", "retreat"]

    def sample(self) -> Dict[str, Any]:
        """Sample a random military action."""
        return {
            'action_type': np.random.choice(self.action_types),
            'target_cluster': np.random.randint(0, self.max_clusters),
            'unit_type': np.random.choice(self.unit_types).name,
            'intensity': np.random.uniform(0.1, 1.0)
        }

    def contains(self, action: Any) -> bool:
        """Check if action is valid."""
        if not isinstance(action, dict):
            return False

        required_keys = {'action_type', 'target_cluster', 'unit_type', 'intensity'}
        if not required_keys.issubset(action.keys()):
            return False

        if action['action_type'] not in self.action_types:
            return False

        if not (0 <= action['target_cluster'] < self.max_clusters):
            return False

        if action['unit_type'] not in [ut.name for ut in self.unit_types]:
            return False

        if not (0.0 <= action['intensity'] <= 1.0):
            return False

        return True

# ─────────────────────────────────────────────────────────────────────────── #
# Military Wrapper                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class MilitaryWrapper(gym.Wrapper):
    """
    Wraps GravitasEnv with tactical military operations.

    The military wrapper adds:
      1. Unit-based military operations (separate from cluster military presence)
      2. Tactical movement between clusters
      3. Combat resolution
      4. Supply and logistics management
      5. Military objectives and victory conditions
      6. Military-specific rewards

    Args:
        env: A GravitasEnv instance
        military_params: MilitaryUnitParams config
        seed: RNG seed for military initialization
        objectives: List of military objectives, or None for defaults
    """

    def __init__(
        self,
        env: gym.Env,
        military_params: Optional[MilitaryUnitParams] = None,
        seed: Optional[int] = None,
        objectives: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(env)
        self.military_params = military_params or MilitaryUnitParams.default()
        self._rng = np.random.default_rng(seed)
        self._objectives = objectives  # store for use in reset()

        # Military state — set in reset()
        self._military_state: Optional[WorldMilitaryState] = None
        self._last_metrics: Optional[Dict[str, Any]] = None
        self._adjacency_matrix: Optional[NDArray[np.bool_]] = None

        # Cache max_N from the wrapped env
        self._max_N = getattr(env, "_max_N", 12)

        # Military action space
        self._military_action_space = MilitaryActionSpace(self._max_N)

        # Extend observation space
        base_shape = self.env.observation_space.shape[0]

        # Military observation dimensions:
        # - Per cluster: unit_count, total_combat_power, supply_level (3 * max_N)
        # - Global: global_supply, reinforcement_pool, objectives_progress (3)
        # - Victory status: completion_percentage, victory_achieved (2)
        military_obs_dim = 3 * self._max_N + 5

        self.observation_space = spaces.Box(
            low=np.concatenate([
                self.env.observation_space.low,
                np.full(military_obs_dim, -np.inf, dtype=np.float32),
            ]),
            high=np.concatenate([
                self.env.observation_space.high,
                np.full(military_obs_dim, np.inf, dtype=np.float32),
            ]),
            dtype=np.float32,
        )

        # Create combined action space (original + military)
        self.action_space = spaces.Dict({
            'base_action': self.env.action_space,
            'military_action': spaces.Dict({
                'action_type': spaces.Discrete(len(self._military_action_space.action_types)),
                'target_cluster': spaces.Discrete(self._max_N),
                'unit_type': spaces.Discrete(len(self._military_action_space.unit_types)),
                'intensity': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            })
        })

    # ── Gymnasium API ──────────────────────────────────────────────────────── #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment and initialize military state."""
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed + 2000)

        # Initialize military state from the freshly reset world
        world = self._get_world()
        N = world.n_clusters if world else self._max_N

        # Use world's actual topology adjacency (boolean)
        if world is not None:
            self._adjacency_matrix = world.adjacency > 0.0
        else:
            self._adjacency_matrix = compute_adjacency_matrix(N, rng=self._rng)

        # Initialize military state
        self._military_state = initialize_military_state(
            n_clusters=N,
            params=self.military_params,
            rng=self._rng,
            objectives=self._objectives,
        )

        # Add initial units to some clusters
        initial_units = self._initialize_starting_units(N)
        if initial_units:
            new_clusters = list(self._military_state.clusters)
            for unit in initial_units:
                cluster_idx = unit.cluster_id
                new_cluster = new_clusters[cluster_idx].add_unit(unit)
                new_clusters[cluster_idx] = new_cluster

            self._military_state = self._military_state.copy_with(
                clusters=tuple(new_clusters),
                next_unit_id=len(initial_units) + 1
            )

        extended_obs = self._extend_obs(obs)
        info["military"] = self._military_info()

        return extended_obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Execute one step with military operations."""
        # Extract base and military actions
        if isinstance(action, dict) and 'base_action' in action and 'military_action' in action:
            base_action = action['base_action']
            military_action = self._decode_military_action(action['military_action'])
        else:
            # Fallback: treat as base action only, no military action
            base_action = action
            military_action = None

        # 1. Core env step
        obs, reward, terminated, truncated, info = self.env.step(base_action)

        # 2. Military ↔ politics pipeline
        military_reward = 0.0
        if self._military_state is not None and self._adjacency_matrix is not None:
            world = self._get_world()
            if world is not None:
                N  = world.n_clusters
                dt = getattr(getattr(self.env, "params", None), "dt", 0.1)

                # 2a. Politics → Military: adjust morale/supply/depot from cluster state
                pol_driven_mil = apply_political_military_feedback(
                    self._military_state,
                    world,
                    world.alliance,
                    N,
                    dt,
                    self._rng,
                )

                # 2b. Military step (actions, movement, combat, objectives)
                prev_mil = pol_driven_mil.copy_with()
                new_military_state, metrics = step_military_units(
                    military_state=pol_driven_mil,
                    world_state=world,
                    adjacency_matrix=self._adjacency_matrix,
                    params=self.military_params,
                    rng=self._rng,
                    military_action=military_action,
                )
                self._last_metrics = metrics

                # 2c. Military → Politics: compute per-cluster political deltas
                deltas = compute_military_political_feedback(
                    new_military_state,
                    world,
                    prev_mil,
                    world.adjacency,
                    world.alliance,
                    N,
                    dt,
                )

                # 2d. Write political deltas back into GravitasWorld clusters
                clusters = list(world.clusters)
                for i in range(N):
                    c = clusters[i]
                    clusters[i] = c.copy_with(
                        sigma=float(np.clip(
                            c.sigma    + deltas["delta_sigma"][i],   0.0, 1.0)),
                        hazard=float(np.clip(
                            c.hazard   + deltas["delta_hazard"][i],  0.0, 5.0)),
                        resource=float(np.clip(
                            c.resource + deltas["delta_resource"][i],0.0, 1.0)),
                        trust=float(np.clip(
                            c.trust + deltas["delta_trust"][i], 0.0, 1.0)),
                        polar=float(np.clip(
                            c.polar + deltas["delta_polar"][i], 0.0, 1.0)),
                    )

                new_world = world.copy_with_clusters(clusters)

                # media_bias is on GravitasWorld directly, not on clusters
                if hasattr(world, "media_bias") and world.media_bias is not None:
                    bias = world.media_bias.copy()
                    bias[:N] = np.clip(
                        bias[:N] + deltas["delta_media_bias"][:N], -1.0, 1.0
                    )
                    new_world = new_world.copy_with_bias(bias)

                # population delta (population field on GravitasWorld)
                if (hasattr(world, "population") and world.population is not None
                        and np.any(deltas["delta_population"] != 0)):
                    pop = world.population.copy()
                    pop[:N] = np.clip(
                        pop[:N] + deltas["delta_population"][:N], 0.0, 1.0
                    )
                    new_world = new_world.copy_with_population(pop)

                # Write the modified world back to the inner env
                self._set_world(new_world)

                self._military_state = new_military_state

                # 3. Military victory check
                victory_status = check_victory_conditions(new_military_state)
                if victory_status['victory_achieved']:
                    terminated = True
                    info['military_victory'] = True

                # 4. Military reward
                if hasattr(self, '_prev_military_state') and self._prev_military_state is not None:
                    military_reward = compute_military_reward(
                        new_military_state,
                        self._prev_military_state,
                        military_action['action_type'] if military_action else None,
                    )

        # Store state for next step's reward calculation
        self._prev_military_state = (
            self._military_state.copy_with() if self._military_state else None
        )

        # 5. Total reward
        total_reward = reward + military_reward

        # 6. Extend observation
        extended_obs = self._extend_obs(obs)

        # 7. Enrich info
        info["military"] = self._military_info()
        info["military_reward"] = military_reward
        if self._last_metrics:
            info.update({f"military_{k}": v for k, v in self._last_metrics.items()})

        return extended_obs, float(total_reward), terminated, truncated, info

    # ── Military State Access ───────────────────────────────────────────────── #

    @property
    def military_state(self) -> Optional[WorldMilitaryState]:
        """Get current military state."""
        return self._military_state

    # ── Internal Helpers ───────────────────────────────────────────────────── #

    def _get_world(self):
        """Get GravitasWorld from wrapped env, traversing wrapper chain."""
        env = self.env
        while env is not None:
            world = getattr(env, "world", None)
            if world is not None:
                return world
            env = getattr(env, "env", None)
        return None

    def _set_world(self, new_world) -> None:
        """Write a modified GravitasWorld back into the inner GravitasEnv."""
        env = self.env
        while env is not None:
            if hasattr(env, "_world") and env._world is not None:
                env._world = new_world
                return
            env = getattr(env, "env", None)

    def _initialize_starting_units(self, n_clusters: int) -> List[MilitaryUnit]:
        """Create initial military units for testing."""
        units = []

        inf_hp   = self.military_params.get_max_hp(MilitaryUnitType.INFANTRY)
        armor_hp = self.military_params.get_max_hp(MilitaryUnitType.ARMOR)

        # Add some infantry to cluster 0
        for i in range(2):
            units.append(MilitaryUnit(
                unit_id=i + 1,
                unit_type=MilitaryUnitType.INFANTRY,
                cluster_id=0,
                hit_points=inf_hp,
                combat_effectiveness=1.0,
                supply_level=0.8,
                experience=0.0,
                morale=0.9,
                objective_id=1
            ))

        # Add armor to cluster 1
        units.append(MilitaryUnit(
            unit_id=3,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=1,
            hit_points=armor_hp,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=2
        ))

        return units

    def _decode_military_action(self, military_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action space format to internal format.

        Accepts either gym-space format (integer indices) or pre-decoded format
        (string action_type, MilitaryUnitType unit_type).
        """
        at = military_action['action_type']
        action_type = (
            at if isinstance(at, str)
            else self._military_action_space.action_types[int(at)]
        )

        ut = military_action['unit_type']
        unit_type = (
            ut if isinstance(ut, MilitaryUnitType)
            else self._military_action_space.unit_types[int(ut)]
        )

        raw_intensity = military_action.get('intensity', 0.5)
        if hasattr(raw_intensity, '__len__'):
            intensity = float(raw_intensity[0])
        else:
            intensity = float(raw_intensity)

        return {
            'action_type':    action_type,
            'target_cluster': int(military_action['target_cluster']),
            'unit_type':      unit_type,
            'intensity':      intensity,
        }

    def _extend_obs(self, base_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Append military features to observation."""
        if self._military_state is None:
            # Return zeros if no military state
            military_obs_dim = 3 * self._max_N + 5
            return np.concatenate([
                base_obs,
                np.zeros(military_obs_dim, dtype=np.float32)
            ])

        # Per-cluster military features
        cluster_features = []
        for i in range(self._max_N):
            if i < len(self._military_state.clusters):
                cluster = self._military_state.clusters[i]
                cluster_features.extend([
                    cluster.unit_count,                    # Unit count
                    cluster.total_combat_power,          # Combat power
                    cluster.supply_depot                  # Supply level
                ])
            else:
                cluster_features.extend([0.0, 0.0, 0.0])  # Padding

        # Global military features
        global_features = [
            self._military_state.global_supply,
            self._military_state.global_reinforcement_pool,
            sum(obj.completion_progress for obj in self._military_state.objectives) / max(1, len(self._military_state.objectives)),
            float(check_victory_conditions(self._military_state)['completion_percentage']),
            1.0 if check_victory_conditions(self._military_state)['victory_achieved'] else 0.0
        ]

        military_obs = np.array(cluster_features + global_features, dtype=np.float32)
        return np.concatenate([base_obs, military_obs])

    def _military_info(self) -> Dict[str, Any]:
        """Build military info dictionary."""
        if self._military_state is None:
            return {
                'total_units': 0,
                'total_combat_power': 0.0,
                'global_supply': 0.0,
                'global_reinforcement_pool': 0.0,
                'objectives_completed': 0,
                'objectives_total': 0,
                'victory_achieved': False,
                'completion_percentage': 0.0
            }

        victory_status = check_victory_conditions(self._military_state)
        return {
            'total_units': self._military_state.total_unit_count,
            'total_combat_power': self._military_state.total_combat_power,
            'global_supply': self._military_state.global_supply,
            'global_reinforcement_pool': self._military_state.global_reinforcement_pool,
            'objectives_completed': sum(1 for obj in self._military_state.objectives if obj.is_completed),
            'objectives_total': len(self._military_state.objectives),
            'victory_achieved': victory_status['victory_achieved'],
            'completion_percentage': victory_status['completion_percentage'],
            'objectives': [
                {
                    'id': obj.objective_id,
                    'name': obj.name,
                    'type': obj.objective_type,
                    'progress': obj.completion_progress,
                    'completed': obj.is_completed
                }
                for obj in self._military_state.objectives
            ]
        }

# ─────────────────────────────────────────────────────────────────────────── #
# Utility Functions                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def create_military_objectives(
    n_clusters: int,
    objective_config: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create military objectives configuration.

    Args:
        n_clusters: Number of clusters in the world
        objective_config: Custom objective configuration, or None for defaults

    Returns:
        List of objective definitions
    """
    if objective_config is not None:
        return objective_config

    # Default objectives
    return [
        {
            'objective_id': 1,
            'name': 'Capture Central Cluster',
            'objective_type': 'capture',
            'target_cluster_id': 0,
            'required_units': 3,
            'reward_value': 25.0,
        },
        {
            'objective_id': 2,
            'name': 'Secure Supply Route',
            'objective_type': 'hold',
            'target_cluster_id': 1,
            'required_units': 2,
            'reward_value': 15.0,
        },
        {
            'objective_id': 3,
            'name': 'Eliminate Enemy Forces',
            'objective_type': 'destroy',
            'target_cluster_id': 2,
            'required_units': 1,
            'reward_value': 20.0,
        }
    ]

def compute_adjacency_matrix(
    n_clusters: int,
    rng: np.random.Generator,
    connectivity: float = 0.7
) -> NDArray[np.bool_]:
    """
    Compute a simple adjacency matrix for military movement.

    Creates a connected graph where clusters are connected with given probability.

    Args:
        n_clusters: Number of clusters
        rng: Random number generator
        connectivity: Probability of connection between clusters (0-1)

    Returns:
        Boolean adjacency matrix (n_clusters x n_clusters)
    """
    # Create a connected graph using Erdős–Rényi model
    A = np.zeros((n_clusters, n_clusters), dtype=bool)

    # Ensure graph is connected by creating a spanning tree first
    for i in range(1, n_clusters):
        parent = rng.integers(0, i)
        A[i, parent] = True
        A[parent, i] = True

    # Add additional random connections
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if not A[i, j] and rng.random() < connectivity:
                A[i, j] = True
                A[j, i] = True

    # Zero diagonal (no self-connections)
    np.fill_diagonal(A, False)

    return A

def create_simple_military_action(
    action_type: str = "deploy",
    target_cluster: int = 0,
    unit_type: str = "INFANTRY",
    intensity: float = 0.5
) -> Dict[str, Any]:
    """
    Create a simple military action for testing.

    Args:
        action_type: Type of military action
        target_cluster: Target cluster ID
        unit_type: Type of military unit
        intensity: Action intensity (0-1)

    Returns:
        Military action dictionary
    """
    return {
        'action_type': action_type,
        'target_cluster': target_cluster,
        'unit_type': unit_type,
        'intensity': intensity
    }
