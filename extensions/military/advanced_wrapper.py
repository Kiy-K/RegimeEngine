"""
advanced_wrapper.py — Advanced military wrapper with sophisticated tactics and AI.

This wrapper extends the basic MilitaryWrapper with advanced features:
  1. Unit Formations and Group Tactics
  2. Command Hierarchy and Chain of Command
  3. Intelligence and Reconnaissance Systems
  4. Advanced Combat Tactics (flanking, ambush, suppression)
  5. Electronic Warfare and Communications
  6. Logistics and Supply Chain Management
  7. Tactical AI for Autonomous Unit Behavior

The advanced wrapper provides a more realistic and sophisticated military
simulation while maintaining compatibility with the basic military system.
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
from .military_dynamics import (
    step_military_units, apply_military_action,
    compute_military_reward, check_victory_conditions
)
from .military_wrapper import MilitaryWrapper, MilitaryActionSpace
from .advanced_tactics import (
    AdvancedTacticsEngine, FormationType, CommandRank,
    IntelligenceType, CombatTactic, SupplyType,
    UnitFormation, CommandStructure, IntelligenceReport,
    TacticalOperation, ElectronicWarfareState, SupplyChain,
    AdvancedClusterMilitaryState, AdvancedWorldMilitaryState
)
# build_adjacency_matrix removed (unused and caused slow imports)

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Military Action Space                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class AdvancedMilitaryActionSpace:
    """
    Extended action space for advanced military operations.

    Supports both basic and advanced military actions.
    """

    def __init__(self, max_clusters: int = 12):
        self.max_clusters = max_clusters
        self.unit_types = list(MilitaryUnitType)
        self.action_types = ["deploy", "move", "attack", "reinforce", "retreat"]
        self.advanced_action_types = [
            "create_formation", "plan_operation", "execute_operation",
            "gather_intel", "establish_command", "setup_supply_chain"
        ]
        self.all_action_types = self.action_types + self.advanced_action_types
        self.formation_types = list(FormationType)
        self.command_ranks = list(CommandRank)
        self.intel_types = list(IntelligenceType)
        self.combat_tactics = list(CombatTactic)
        self.supply_types = list(SupplyType)

    def sample(self) -> Dict[str, Any]:
        """Sample a random military action (basic or advanced)."""
        if np.random.random() < 0.7:  # 70% chance of basic action
            return {
                'action_type': np.random.choice(self.action_types),
                'target_cluster': np.random.randint(0, self.max_clusters),
                'unit_type': np.random.choice(self.unit_types).name,
                'intensity': np.random.uniform(0.1, 1.0)
            }
        else:  # 30% chance of advanced action
            advanced_type = np.random.choice(self.advanced_action_types)
            if advanced_type == "create_formation":
                return {
                    'action_type': advanced_type,
                    'cluster_id': np.random.randint(0, self.max_clusters),
                    'formation_type': np.random.choice(self.formation_types).name,
                    'unit_ids': [np.random.randint(1, 10) for _ in range(2, 5)]
                }
            elif advanced_type == "plan_operation":
                return {
                    'action_type': advanced_type,
                    'tactic': np.random.choice(self.combat_tactics).name,
                    'primary_target': np.random.randint(0, self.max_clusters),
                    'participating_units': [np.random.randint(1, 10) for _ in range(1, 4)]
                }
            elif advanced_type == "execute_operation":
                return {
                    'action_type': advanced_type,
                    'operation_id': np.random.randint(1, 5)
                }
            elif advanced_type == "gather_intel":
                return {
                    'action_type': advanced_type,
                    'cluster_id': np.random.randint(0, self.max_clusters),
                    'intel_type': np.random.choice(self.intel_types).name
                }
            elif advanced_type == "establish_command":
                return {
                    'action_type': advanced_type,
                    'cluster_id': np.random.randint(0, self.max_clusters),
                    'commander_unit_id': np.random.randint(1, 10),
                    'subordinate_unit_ids': [np.random.randint(1, 10) for _ in range(1, 4)],
                    'rank': np.random.choice(self.command_ranks).name
                }
            else:  # setup_supply_chain
                return {
                    'action_type': advanced_type,
                    'cluster_id': np.random.randint(0, self.max_clusters),
                    'connected_depots': [np.random.randint(0, self.max_clusters) for _ in range(1, 3)]
                }

    def contains(self, action: Any) -> bool:
        """Check if action is valid."""
        if not isinstance(action, dict):
            return False

        if 'action_type' not in action:
            return False

        action_type = action['action_type']

        # Basic actions
        if action_type in self.action_types:
            required_keys = {'action_type', 'target_cluster', 'unit_type', 'intensity'}
            if not required_keys.issubset(action.keys()):
                return False

            if not (0 <= action['target_cluster'] < self.max_clusters):
                return False

            if action['unit_type'] not in [ut.name for ut in self.unit_types]:
                return False

            if not (0.0 <= action['intensity'] <= 1.0):
                return False

            return True

        # Advanced actions
        elif action_type in self.advanced_action_types:
            if action_type == "create_formation":
                required_keys = {'action_type', 'cluster_id', 'formation_type', 'unit_ids'}
                if not required_keys.issubset(action.keys()):
                    return False
                if not (0 <= action['cluster_id'] < self.max_clusters):
                    return False
                if action['formation_type'] not in [ft.name for ft in self.formation_types]:
                    return False
                return isinstance(action['unit_ids'], list) and len(action['unit_ids']) >= 2

            elif action_type == "plan_operation":
                required_keys = {'action_type', 'tactic', 'primary_target', 'participating_units'}
                if not required_keys.issubset(action.keys()):
                    return False
                if not (0 <= action['primary_target'] < self.max_clusters):
                    return False
                if action['tactic'] not in [ct.name for ct in self.combat_tactics]:
                    return False
                return isinstance(action['participating_units'], list) and len(action['participating_units']) >= 1

            elif action_type == "execute_operation":
                required_keys = {'action_type', 'operation_id'}
                if not required_keys.issubset(action.keys()):
                    return False
                return isinstance(action['operation_id'], int) and action['operation_id'] > 0

            elif action_type == "gather_intel":
                required_keys = {'action_type', 'cluster_id', 'intel_type'}
                if not required_keys.issubset(action.keys()):
                    return False
                if not (0 <= action['cluster_id'] < self.max_clusters):
                    return False
                return action['intel_type'] in [it.name for it in self.intel_types]

            elif action_type == "establish_command":
                required_keys = {'action_type', 'cluster_id', 'commander_unit_id', 'subordinate_unit_ids', 'rank'}
                if not required_keys.issubset(action.keys()):
                    return False
                if not (0 <= action['cluster_id'] < self.max_clusters):
                    return False
                if action['rank'] not in [cr.name for cr in self.command_ranks]:
                    return False
                return isinstance(action['subordinate_unit_ids'], list)

            elif action_type == "setup_supply_chain":
                required_keys = {'action_type', 'cluster_id', 'connected_depots'}
                if not required_keys.issubset(action.keys()):
                    return False
                if not (0 <= action['cluster_id'] < self.max_clusters):
                    return False
                return isinstance(action['connected_depots'], list)

            else:
                return False

        else:
            return False

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Military Wrapper                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class AdvancedMilitaryWrapper(gym.Wrapper):
    """
    Advanced military wrapper with sophisticated tactics and AI.

    Extends the basic MilitaryWrapper with:
      1. Unit Formations and Group Tactics
      2. Command Hierarchy and Chain of Command
      3. Intelligence and Reconnaissance Systems
      4. Advanced Combat Tactics
      5. Electronic Warfare and Communications
      6. Logistics and Supply Chain Management
      7. Tactical AI for Autonomous Unit Behavior

    The advanced wrapper can work with either basic or advanced military states,
    providing backward compatibility while enabling advanced features.
    """

    def __init__(
        self,
        env: gym.Env,
        military_params: Optional[MilitaryUnitParams] = None,
        seed: Optional[int] = None,
        objectives: Optional[List[Dict[str, Any]]] = None,
        use_advanced_tactics: bool = True,
    ) -> None:
        super().__init__(env)
        self.military_params = military_params or MilitaryUnitParams()
        self._rng = np.random.default_rng(seed)
        self.use_advanced_tactics = use_advanced_tactics

        # Military state — set in reset()
        self._military_state: Optional[WorldMilitaryState] = None
        self._advanced_military_state: Optional[AdvancedWorldMilitaryState] = None
        self._last_metrics: Optional[Dict[str, Any]] = None
        self._adjacency_matrix: Optional[NDArray[np.bool_]] = None

        # Tactics engine
        self._tactics_engine = AdvancedTacticsEngine(self.military_params)

        # Cache max_N from the wrapped env
        self._max_N = getattr(env, "_max_N", 12)

        # Military action space
        self._military_action_space = AdvancedMilitaryActionSpace(self._max_N)

        # Extend observation space
        base_shape = self.env.observation_space.shape[0]

        if use_advanced_tactics:
            # Advanced military observation dimensions:
            # - Per cluster: unit_count, total_combat_power, supply_level, formations, intel_quality (5 * max_N)
            # - Global: global_supply, reinforcement_pool, objectives_progress, ew_status, comm_integrity (5)
            # - Victory status: completion_percentage, victory_achieved (2)
            military_obs_dim = 5 * self._max_N + 7
        else:
            # Basic military observation dimensions (same as basic wrapper)
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
                'action_type': spaces.Discrete(len(self._military_action_space.all_action_types)),
                'target_cluster': spaces.Discrete(self._max_N),
                'unit_type': spaces.Discrete(len(self._military_action_space.unit_types)),
                'intensity': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                # Advanced action parameters
                'cluster_id': spaces.Discrete(self._max_N),
                'formation_type': spaces.Discrete(len(self._military_action_space.formation_types)),
                'unit_ids': spaces.Box(low=0, high=100, shape=(10,), dtype=np.int32),  # Max 10 units
                'tactic': spaces.Discrete(len(self._military_action_space.combat_tactics)),
                'primary_target': spaces.Discrete(self._max_N),
                'participating_units': spaces.Box(low=0, high=100, shape=(10,), dtype=np.int32),  # Max 10 units
                'operation_id': spaces.Discrete(100),  # Max 100 operations
                'intel_type': spaces.Discrete(len(self._military_action_space.intel_types)),
                'commander_unit_id': spaces.Discrete(1000),  # Max unit ID
                'subordinate_unit_ids': spaces.Box(low=0, high=1000, shape=(10,), dtype=np.int32),  # Max 10 units
                'rank': spaces.Discrete(len(self._military_action_space.command_ranks)),
                'connected_depots': spaces.Box(low=0, high=self._max_N, shape=(5,), dtype=np.int32),  # Max 5 depots
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
            self._rng = np.random.default_rng(seed + 3000)

        # Initialize military state from the freshly reset world
        world = self._get_world()
        N = world.n_clusters if world else self._max_N

        # Compute adjacency matrix for movement
        self._adjacency_matrix = self._compute_adjacency_matrix(N, rng=self._rng)

        if self.use_advanced_tactics:
            # Initialize advanced military state
            self._advanced_military_state = self._initialize_advanced_military_state(N)
            self._military_state = None
        else:
            # Initialize basic military state
            self._military_state = initialize_military_state(
                n_clusters=N,
                params=self.military_params,
                rng=self._rng,
                objectives=objectives if objectives is not None else None
            )
            self._advanced_military_state = None

        # Add initial units to some clusters
        initial_units = self._initialize_starting_units(N)
        if initial_units:
            if self.use_advanced_tactics:
                self._advanced_military_state = self._add_units_to_advanced_state(
                    self._advanced_military_state, initial_units
                )
            else:
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
        """Execute one step with advanced military operations."""
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

        # 2. Military step
        if self.use_advanced_tactics and self._advanced_military_state is not None:
            world = self._get_world()
            if world is not None and self._adjacency_matrix is not None:
                # Step military units with advanced tactics
                new_military_state, metrics = self._step_advanced_military_units(
                    self._advanced_military_state,
                    world,
                    self._adjacency_matrix,
                    military_action
                )
                self._advanced_military_state = new_military_state
                self._last_metrics = metrics

                # 3. Check for military victory conditions
                victory_status = check_victory_conditions(new_military_state)
                if victory_status['victory_achieved']:
                    terminated = True
                    info['military_victory'] = True
        elif self._military_state is not None and self._adjacency_matrix is not None:
            world = self._get_world()
            if world is not None:
                # Step military units with basic tactics
                new_military_state, metrics = step_military_units(
                    military_state=self._military_state,
                    world_state=world,
                    adjacency_matrix=self._adjacency_matrix,
                    params=self.military_params,
                    rng=self._rng,
                    military_action=military_action
                )
                self._military_state = new_military_state
                self._last_metrics = metrics

                # 3. Check for military victory conditions
                victory_status = check_victory_conditions(new_military_state)
                if victory_status['victory_achieved']:
                    terminated = True
                    info['military_victory'] = True

        # 4. Military reward
        military_reward = 0.0
        if self.use_advanced_tactics and self._advanced_military_state is not None:
            if (hasattr(self, '_prev_advanced_military_state') and
                self._prev_advanced_military_state is not None):
                military_reward = self._compute_advanced_military_reward(
                    self._advanced_military_state,
                    self._prev_advanced_military_state,
                    military_action['action_type'] if military_action else None
                )
        elif self._military_state is not None:
            if (hasattr(self, '_prev_military_state') and
                self._prev_military_state is not None):
                military_reward = compute_military_reward(
                    self._military_state,
                    self._prev_military_state,
                    military_action['action_type'] if military_action else None
                )

        # Store current state for next step's reward calculation
        if self.use_advanced_tactics:
            self._prev_advanced_military_state = self._advanced_military_state.copy_with() if self._advanced_military_state else None
        else:
            self._prev_military_state = self._military_state.copy_with() if self._military_state else None

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
        """Get current military state (basic)."""
        return self._military_state

    @property
    def advanced_military_state(self) -> Optional[AdvancedWorldMilitaryState]:
        """Get current advanced military state."""
        return self._advanced_military_state

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

    def _compute_adjacency_matrix(
        self,
        n_clusters: int,
        rng: np.random.Generator,
        connectivity: float = 0.7
    ) -> NDArray[np.bool_]:
        """
        Compute a simple adjacency matrix for military movement.

        Creates a connected graph where clusters are connected with given probability.
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

    def _initialize_advanced_military_state(self, n_clusters: int) -> AdvancedWorldMilitaryState:
        """Initialize advanced military state."""
        # Create empty clusters with advanced features
        clusters = tuple(
            AdvancedClusterMilitaryState(
                cluster_id=i,
                units=(),
                supply_depot=10.0,
                is_controlled=False,
                controlling_faction=None,
                reinforcement_timer=0.0,
                formations=(),
                command_structure=None,
                intelligence_reports=(),
                active_operations=(),
                ew_state=None,
                supply_chain=None,
                terrain_advantage=1.0 + 0.2 * self._rng.random(),  # Random terrain advantage
                fog_of_war=0.5  # Initial fog of war
            )
            for i in range(n_clusters)
        )

        # Create default objectives
        objectives = [
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
                'name': 'Hold Supply Route',
                'objective_type': 'hold',
                'target_cluster_id': 1,
                'required_units': 2,
                'reward_value': 15.0,
            }
        ]

        # Convert objectives to MilitaryObjective objects
        military_objectives = tuple(
            MilitaryObjective(
                objective_id=obj['objective_id'],
                name=obj['name'],
                objective_type=obj['objective_type'],
                target_cluster_id=obj['target_cluster_id'],
                required_units=obj['required_units'],
                reward_value=obj['reward_value'],
            )
            for obj in objectives
        )

        return AdvancedWorldMilitaryState(
            clusters=clusters,
            objectives=military_objectives,
            global_supply=100.0,
            global_reinforcement_pool=50.0,
            step=0,
            next_unit_id=1,
            next_formation_id=1,
            next_operation_id=1,
            next_report_id=1,
            global_intelligence={},
            electronic_warfare_active=False,
            communication_network_integrity=1.0
        )

    def _add_units_to_advanced_state(
        self,
        state: AdvancedWorldMilitaryState,
        units: List[MilitaryUnit]
    ) -> AdvancedWorldMilitaryState:
        """Add units to advanced military state."""
        new_clusters = list(state.clusters)

        for unit in units:
            cluster_idx = unit.cluster_id
            new_cluster = new_clusters[cluster_idx].add_unit(unit)
            new_clusters[cluster_idx] = new_cluster

        return state.copy_with(
            clusters=tuple(new_clusters),
            next_unit_id=state.next_unit_id + len(units)
        )

    def _initialize_starting_units(self, n_clusters: int) -> List[MilitaryUnit]:
        """Create initial military units for testing."""
        units = []

        # Add some infantry to cluster 0
        for i in range(2):
            units.append(MilitaryUnit(
                unit_id=i + 1,
                unit_type=MilitaryUnitType.INFANTRY,
                cluster_id=0,
                hit_points=self.military_params.infantry_hp,
                combat_effectiveness=1.0,
                supply_level=0.8,
                experience=0.0,
                morale=0.9,
                objective_id=1  # Assign to first objective
            ))

        # Add armor to cluster 1
        units.append(MilitaryUnit(
            unit_id=3,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=1,
            hit_points=self.military_params.armor_hp,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=2  # Assign to second objective
        ))

        return units

    def _decode_military_action(self, military_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action space format to internal format."""
        action_type_idx = int(military_action['action_type'])
        action_type = self._military_action_space.all_action_types[action_type_idx]

        if action_type in self._military_action_space.action_types:
            # Basic action
            return {
                'action_type': action_type,
                'target_cluster': int(military_action['target_cluster']),
                'unit_type': self._military_action_space.unit_types[int(military_action['unit_type'])],
                'intensity': float(military_action['intensity'][0])
            }
        else:
            # Advanced action
            if action_type == "create_formation":
                return {
                    'action_type': action_type,
                    'cluster_id': int(military_action['cluster_id']),
                    'formation_type': self._military_action_space.formation_types[int(military_action['formation_type'])],
                    'unit_ids': [int(uid) for uid in military_action['unit_ids'] if uid > 0]
                }
            elif action_type == "plan_operation":
                return {
                    'action_type': action_type,
                    'tactic': self._military_action_space.combat_tactics[int(military_action['tactic'])],
                    'primary_target': int(military_action['primary_target']),
                    'participating_units': [int(uid) for uid in military_action['participating_units'] if uid > 0]
                }
            elif action_type == "execute_operation":
                return {
                    'action_type': action_type,
                    'operation_id': int(military_action['operation_id'])
                }
            elif action_type == "gather_intel":
                return {
                    'action_type': action_type,
                    'cluster_id': int(military_action['cluster_id']),
                    'intel_type': self._military_action_space.intel_types[int(military_action['intel_type'])]
                }
            elif action_type == "establish_command":
                return {
                    'action_type': action_type,
                    'cluster_id': int(military_action['cluster_id']),
                    'commander_unit_id': int(military_action['commander_unit_id']),
                    'subordinate_unit_ids': [int(uid) for uid in military_action['subordinate_unit_ids'] if uid > 0],
                    'rank': self._military_action_space.command_ranks[int(military_action['rank'])]
                }
            else:  # setup_supply_chain
                return {
                    'action_type': action_type,
                    'cluster_id': int(military_action['cluster_id']),
                    'connected_depots': [int(cid) for cid in military_action['connected_depots'] if cid >= 0]
                }

    def _step_advanced_military_units(
        self,
        military_state: AdvancedWorldMilitaryState,
        world_state: object,  # GravitasWorld
        adjacency_matrix: NDArray[np.bool_],
        military_action: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AdvancedWorldMilitaryState, Dict[str, Any]]:
        """
        Advanced military step with sophisticated tactics.

        This function orchestrates all advanced military dynamics:
          1. Apply military actions (basic and advanced)
          2. Execute tactical operations
          3. Resolve unit movement with formations
          4. Handle combat resolution with tactics
          5. Update supply and logistics
          6. Update intelligence and reconnaissance
          7. Track objective progress
          8. Compute rewards
        """
        # Get cluster resource levels from GravitasWorld
        cluster_resources = np.array([
            c.resource for c in world_state.clusters
        ])

        # 1. Apply military action if provided
        new_state = military_state
        action_type = None

        if military_action:
            action_type = military_action.get('action_type')
            if action_type in ["deploy", "move", "attack", "reinforce", "retreat"]:
                # Basic military action
                new_state = self._apply_basic_military_action(
                    military_state,
                    action_type,
                    military_action.get('target_cluster', 0),
                    military_action.get('unit_type', MilitaryUnitType.INFANTRY),
                    military_action.get('intensity', 0.5)
                )
            elif action_type == "create_formation":
                # Create formation
                new_state, _ = self._tactics_engine.create_formation(
                    military_state,
                    military_action.get('cluster_id', 0),
                    military_action.get('unit_ids', []),
                    military_action.get('formation_type', FormationType.LINE),
                    self.military_params
                )
            elif action_type == "plan_operation":
                # Plan tactical operation
                new_state, _ = self._tactics_engine.plan_tactical_operation(
                    military_state,
                    military_action.get('tactic', CombatTactic.FRONTAL_ASSAULT),
                    military_action.get('primary_target', 0),
                    military_action.get('participating_units', []),
                    self.military_params
                )
            elif action_type == "execute_operation":
                # Execute tactical operation
                new_state = self._tactics_engine.execute_tactical_operation(
                    military_state,
                    military_action.get('operation_id', 1),
                    self.military_params,
                    self._rng
                )
            elif action_type == "gather_intel":
                # Gather intelligence
                new_state = self._tactics_engine.update_intelligence(
                    military_state,
                    military_action.get('cluster_id', 0),
                    military_action.get('intel_type', IntelligenceType.RECON),
                    self.military_params,
                    self._rng
                )
            elif action_type == "establish_command":
                # Establish command structure
                new_state = self._tactics_engine.establish_command_structure(
                    military_state,
                    military_action.get('cluster_id', 0),
                    military_action.get('commander_unit_id', 1),
                    military_action.get('subordinate_unit_ids', []),
                    military_action.get('rank', CommandRank.SERGEANT)
                )
            elif action_type == "setup_supply_chain":
                # Setup supply chain
                new_state = self._tactics_engine.setup_supply_chain(
                    military_state,
                    military_action.get('cluster_id', 0),
                    military_action.get('connected_depots', [])
                )

        # 2. Execute active tactical operations
        new_state = self._execute_active_operations(new_state)

        # 3. Resolve combat in each cluster with advanced tactics
        new_state = self._resolve_advanced_combat(new_state, cluster_resources)

        # 4. Update supply and logistics with advanced supply chains
        new_state = self._update_advanced_supply(new_state, cluster_resources)

        # 5. Update intelligence (stale reports, etc.)
        new_state = self._update_intelligence_status(new_state)

        # 6. Update objective progress
        new_state = self._update_objective_progress(new_state)

        # 7. Check victory conditions
        victory_status = check_victory_conditions(new_state)

        # 8. Compute military reward
        military_reward = self._compute_advanced_military_reward(
            new_state,
            military_state,
            action_type
        )

        # 9. Advance step
        new_state = new_state.advance_step()

        # 10. Prepare metrics
        metrics = {
            'total_units': new_state.total_unit_count,
            'total_combat_power': new_state.total_combat_power,
            'global_supply': new_state.global_supply,
            'global_reinforcement_pool': new_state.global_reinforcement_pool,
            'objectives_completed': sum(1 for obj in new_state.objectives if obj.is_completed),
            'objectives_total': len(new_state.objectives),
            'military_reward': military_reward,
            'active_operations': sum(len(c.active_operations) for c in new_state.clusters),
            'formations': sum(len(c.formations) for c in new_state.clusters),
            'intelligence_reports': sum(len(c.intelligence_reports) for c in new_state.clusters),
            'communication_integrity': new_state.communication_network_integrity,
            **victory_status
        }

        return new_state, metrics

    def _apply_basic_military_action(
        self,
        world_state: AdvancedWorldMilitaryState,
        action_type: str,
        target_cluster_id: int,
        unit_type: MilitaryUnitType,
        intensity: float,
    ) -> AdvancedWorldMilitaryState:
        """Apply basic military action to advanced state."""
        target_cluster = world_state.get_cluster(target_cluster_id)
        if target_cluster is None:
            return world_state

        new_world = world_state

        if action_type == "deploy":
            # Deploy new units to target cluster
            if world_state.global_reinforcement_pool >= 1.0 * intensity:
                # Create new unit
                new_unit = MilitaryUnit(
                    unit_id=world_state.next_unit_id,
                    unit_type=unit_type,
                    cluster_id=target_cluster_id,
                    hit_points=self.military_params.get_max_hp(unit_type),
                    combat_effectiveness=1.0,
                    supply_level=0.8,
                    experience=0.0,
                    morale=0.9,
                    objective_id=None  # Will be assigned later
                )

                # Add to cluster and update world state
                new_cluster = target_cluster.add_unit(new_unit)
                new_clusters = tuple(
                    new_cluster if c.cluster_id == target_cluster_id else c
                    for c in world_state.clusters
                )

                new_world = world_state.copy_with(
                    clusters=new_clusters,
                    global_reinforcement_pool=world_state.global_reinforcement_pool - 1.0 * intensity,
                    next_unit_id=world_state.next_unit_id + 1
                )

        elif action_type == "move":
            # Move units toward target cluster (simplified for now)
            # In full implementation, this would use pathfinding
            for cluster in world_state.clusters:
                if cluster.cluster_id == target_cluster_id:
                    continue

                # Find units that could move toward target
                for unit in cluster.units:
                    if unit.is_alive and self._rng.random() < intensity * 0.3:
                        # Simple movement: 30% chance to move toward target per step
                        new_unit = unit.copy_with(cluster_id=target_cluster_id)
                        new_cluster = cluster.update_unit(new_unit)
                        new_target = target_cluster.add_unit(new_unit)

                        new_clusters = tuple(
                            new_target if c.cluster_id == target_cluster_id else
                            new_cluster if c.cluster_id == cluster.cluster_id else
                            c
                            for c in world_state.clusters
                        )

                        new_world = world_state.copy_with(clusters=new_clusters)
                        break

        elif action_type == "attack":
            # Boost combat effectiveness of units in target cluster
            new_units = []
            for unit in target_cluster.units:
                if unit.is_alive:
                    new_morale = min(1.0, unit.morale + 0.1 * intensity)
                    new_effectiveness = min(1.0, unit.combat_effectiveness + 0.05 * intensity)
                    new_unit = unit.copy_with(
                        morale=new_morale,
                        combat_effectiveness=new_effectiveness
                    )
                    new_units.append(new_unit)
                else:
                    new_units.append(unit)

            new_cluster = target_cluster.copy_with(units=tuple(new_units))
            new_clusters = tuple(
                new_cluster if c.cluster_id == target_cluster_id else c
                for c in world_state.clusters
            )
            new_world = world_state.copy_with(clusters=new_clusters)

        elif action_type == "reinforce":
            # Reinforce units in target cluster
            new_units = []
            supply_used = 0.0

            for unit in target_cluster.units:
                if unit.is_alive:
                    # HP reinforcement
                    hp_gain = self.military_params.reinforcement_rate * self.military_params.get_max_hp(unit.unit_type) * intensity
                    new_hp = min(self.military_params.get_max_hp(unit.unit_type), unit.hit_points + hp_gain)

                    # Supply replenishment
                    supply_gain = 0.1 * intensity
                    new_supply = min(1.0, unit.supply_level + supply_gain)
                    supply_used += self.military_params.get_supply_cost(unit.unit_type) * supply_gain

                    new_unit = unit.copy_with(
                        hit_points=new_hp,
                        supply_level=new_supply,
                        morale=min(1.0, unit.morale + 0.05 * intensity)
                    )
                    new_units.append(new_unit)
                else:
                    new_units.append(unit)

            # Update supply depot
            depot_consumed = min(target_cluster.supply_depot, supply_used)
            new_supply_depot = target_cluster.supply_depot - depot_consumed

            new_cluster = target_cluster.copy_with(
                units=tuple(new_units),
                supply_depot=new_supply_depot
            )
            new_clusters = tuple(
                new_cluster if c.cluster_id == target_cluster_id else c
                for c in world_state.clusters
            )
            new_world = world_state.copy_with(clusters=new_clusters)

        elif action_type == "retreat":
            # Retreat units from target cluster (move to adjacent clusters)
            adjacent_clusters = [c for c in world_state.clusters
                                if c.cluster_id != target_cluster_id]

            if adjacent_clusters:
                retreat_target = self._rng.choice(adjacent_clusters)

                new_units = []
                for unit in target_cluster.units:
                    if unit.is_alive and self._rng.random() < intensity * 0.5:
                        # 50% chance to retreat per unit
                        new_unit = unit.copy_with(cluster_id=retreat_target.cluster_id)
                        new_units.append(new_unit)
                    else:
                        new_units.append(unit)

                # Update both clusters
                new_target_cluster = target_cluster.copy_with(units=tuple(new_units))
                new_retreat_cluster = retreat_target.copy_with(
                    units=retreat_target.units + tuple(
                        u for u in new_units if u.cluster_id == retreat_target.cluster_id
                    )
                )

                new_clusters = tuple(
                    new_target_cluster if c.cluster_id == target_cluster_id else
                    new_retreat_cluster if c.cluster_id == retreat_target.cluster_id else
                    c
                    for c in world_state.clusters
                )

                new_world = world_state.copy_with(clusters=new_clusters)

        return new_world

    def _execute_active_operations(
        self,
        world_state: AdvancedWorldMilitaryState
    ) -> AdvancedWorldMilitaryState:
        """Execute all active tactical operations."""
        new_world = world_state

        # Find all active operations across all clusters
        for cluster in world_state.clusters:
            for operation in cluster.active_operations:
                if operation.is_active and not operation.is_completed:
                    # Execute the operation
                    new_world = self._tactics_engine.execute_tactical_operation(
                        new_world,
                        operation.operation_id,
                        self.military_params,
                        self._rng
                    )

        return new_world

    def _resolve_advanced_combat(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_resources: NDArray[np.float64]
    ) -> AdvancedWorldMilitaryState:
        """Resolve combat in each cluster with advanced tactics."""
        new_clusters = []

        for cluster in world_state.clusters:
            # Get terrain advantage from cluster
            terrain_advantage = cluster.terrain_advantage

            # Simple combat resolution for now
            # In a full implementation, this would consider formations, tactics, etc.
            updated_cluster = self._resolve_cluster_combat_simple(
                cluster,
                terrain_advantage
            )
            new_clusters.append(updated_cluster)

        return world_state.copy_with(clusters=tuple(new_clusters))

    def _resolve_cluster_combat_simple(
        self,
        cluster_state: AdvancedClusterMilitaryState,
        terrain_advantage: float
    ) -> AdvancedClusterMilitaryState:
        """Simple combat resolution considering formations."""
        # For now, simple implementation: find two strongest opposing units and fight
        # In a full implementation, this would handle multiple factions and complex combat

        alive_units = [u for u in cluster_state.units if u.is_alive]
        if len(alive_units) < 2:
            return cluster_state  # No combat possible

        # Sort by combat power and take top two
        sorted_units = sorted(alive_units, key=lambda u: u.combat_power, reverse=True)
        attacker, defender = sorted_units[0], sorted_units[1]

        # Resolve combat with terrain effects
        updated_attacker, updated_defender = self._resolve_combat_with_terrain(
            attacker, defender, terrain_advantage
        )

        # Update units in cluster
        new_cluster = cluster_state
        new_cluster = new_cluster.update_unit(updated_attacker)
        new_cluster = new_cluster.update_unit(updated_defender)

        return new_cluster

    def _resolve_combat_with_terrain(
        self,
        attacker: MilitaryUnit,
        defender: MilitaryUnit,
        terrain_advantage: float
    ) -> Tuple[MilitaryUnit, MilitaryUnit]:
        """Resolve combat between two units with terrain effects."""
        # Compute combat power with terrain effects
        attacker_power = attacker.combat_power * (1.0 + np.random.normal(0, 0.1))
        defender_power = defender.combat_power * terrain_advantage * (1.0 + np.random.normal(0, 0.1))

        # Damage calculation
        total_damage = attacker_power + defender_power
        attacker_damage = defender_power / total_damage * attacker.combat_power * 0.3
        defender_damage = attacker_power / total_damage * defender.combat_power * 0.3

        # Apply damage with some randomness
        attacker_hp_loss = min(attacker.hit_points * 0.9, attacker_damage * (0.9 + 0.2 * np.random.random()))
        defender_hp_loss = min(defender.hit_points * 0.9, defender_damage * (0.9 + 0.2 * np.random.random()))

        # Update combat effectiveness (degrades with combat)
        attacker_effectiveness = max(0.1, attacker.combat_effectiveness - self.military_params.combat_effectiveness_decay)
        defender_effectiveness = max(0.1, defender.combat_effectiveness - self.military_params.combat_effectiveness_decay)

        # Update units
        updated_attacker = attacker.copy_with(
            hit_points=attacker.hit_points - attacker_hp_loss,
            combat_effectiveness=attacker_effectiveness,
            experience=min(10.0, attacker.experience + 0.1),
            morale=max(0.1, attacker.morale - 0.05)
        )

        updated_defender = defender.copy_with(
            hit_points=defender.hit_points - defender_hp_loss,
            combat_effectiveness=defender_effectiveness,
            experience=min(10.0, defender.experience + 0.1),
            morale=max(0.1, defender.morale - 0.05)
        )

        return updated_attacker, updated_defender

    def _update_advanced_supply(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_resources: NDArray[np.float64]
    ) -> AdvancedWorldMilitaryState:
        """Update supply and logistics with advanced supply chains."""
        new_clusters = []
        total_supply_consumed = 0.0

        for cluster in world_state.clusters:
            cluster_idx = cluster.cluster_id
            resource_level = cluster_resources[cluster_idx]

            # Calculate total supply demand
            supply_demand = cluster.supply_demand(self.military_params)

            # Supply available from depot + global pool
            available_supply = cluster.supply_depot + min(
                world_state.global_supply * 0.1,  # Limit global contribution
                supply_demand * 0.3   # Max 30% from global
            )

            # Resource-based supply bonus
            resource_bonus = resource_level * 2.0

            # Total supply available
            total_available = available_supply + resource_bonus

            # Distribute supplies to units
            new_units = []
            supply_used = 0.0

            for unit in cluster.units:
                if not unit.is_alive:
                    new_units.append(unit)
                    continue

                unit_supply_cost = self.military_params.get_supply_cost(unit.unit_type) * self.military_params.supply_consumption_rate
                supply_allocated = min(unit_supply_cost, total_available * (unit_supply_cost / max(supply_demand, 1e-6)))

                supply_used += supply_allocated
                new_supply_level = min(1.0, unit.supply_level + supply_allocated * 0.1)

                # HP regeneration based on supply
                hp_regen = 0.0
                if new_supply_level > 0.5:  # Only regen with good supply
                    hp_regen = self.military_params.reinforcement_rate * unit.hit_points * new_supply_level

                new_hp = min(
                    self.military_params.get_max_hp(unit.unit_type),
                    unit.hit_points + hp_regen - self.military_params.attrition_rate * unit.hit_points
                )

                new_units.append(unit.copy_with(
                    supply_level=new_supply_level,
                    hit_points=new_hp
                ))

            # Update cluster supply depot
            depot_consumed = min(cluster.supply_depot, supply_demand * 0.5)
            global_consumed = supply_used - depot_consumed

            new_supply_depot = max(0.0, cluster.supply_depot - depot_consumed)
            new_supply_depot += resource_level * 1.0  # Resource regeneration

            # Update supply chain if it exists
            if cluster.supply_chain is not None:
                supply_consumption_by_type = {
                    SupplyType.AMMUNITION: supply_used * 0.4,
                    SupplyType.FUEL: supply_used * 0.2,
                    SupplyType.FOOD: supply_used * 0.2,
                    SupplyType.MEDICAL: supply_used * 0.1,
                    SupplyType.REPAIR: supply_used * 0.05,
                    SupplyType.SPECIALIZED: supply_used * 0.05
                }

                updated_supply_chain = self._tactics_engine.update_supply_chain(
                    world_state,
                    cluster.cluster_id,
                    supply_consumption_by_type,
                    self.military_params
                ).get_cluster(cluster.cluster_id).supply_chain

                new_cluster = cluster.copy_with(
                    units=tuple(new_units),
                    supply_depot=new_supply_depot,
                    supply_chain=updated_supply_chain
                )
            else:
                new_cluster = cluster.copy_with(
                    units=tuple(new_units),
                    supply_depot=new_supply_depot
                )

            new_clusters.append(new_cluster)
            total_supply_consumed += global_consumed

        new_world = world_state.copy_with(
            clusters=tuple(new_clusters),
            global_supply=max(0.0, world_state.global_supply - total_supply_consumed)
        )

        return new_world

    def _update_intelligence_status(
        self,
        world_state: AdvancedWorldMilitaryState
    ) -> AdvancedWorldMilitaryState:
        """Update intelligence reports (mark stale reports, etc.)."""
        new_clusters = []

        for cluster in world_state.clusters:
            # Update intelligence reports
            new_reports = []
            for report in cluster.intelligence_reports:
                # Mark reports as stale if older than 5 steps
                is_stale = (world_state.step - report.last_updated) > 5
                new_report = report.copy_with(is_stale=is_stale)
                new_reports.append(new_report)

            new_cluster = cluster.copy_with(intelligence_reports=tuple(new_reports))
            new_clusters.append(new_cluster)

        return world_state.copy_with(clusters=tuple(new_clusters))

    def _update_objective_progress(
        self,
        world_state: AdvancedWorldMilitaryState
    ) -> AdvancedWorldMilitaryState:
        """Update progress on all military objectives."""
        new_objectives = []

        for objective in world_state.objectives:
            if objective.is_completed:
                new_objectives.append(objective)
                continue

            target_cluster = world_state.get_cluster(objective.target_cluster_id)
            if target_cluster is None:
                new_objectives.append(objective)
                continue

            # Calculate progress based on objective type
            if objective.objective_type == "capture":
                # Progress based on combat power in target cluster
                our_units = sum(1 for u in target_cluster.units
                               if u.is_alive and u.objective_id == objective.objective_id)
                progress = min(1.0, our_units / objective.required_units)
                progress_delta = progress - objective.completion_progress

            elif objective.objective_type == "hold":
                # Progress based on maintaining presence over time
                our_units = sum(1 for u in target_cluster.units
                               if u.is_alive and u.objective_id == objective.objective_id)
                if our_units >= objective.required_units:
                    progress_delta = 1.0 / self.military_params.objective_hold_duration
                else:
                    progress_delta = -0.1  # Lose progress if not holding

            elif objective.objective_type == "destroy":
                # Progress based on eliminating enemy units
                enemy_units = sum(1 for u in target_cluster.units
                                 if u.is_alive and (u.objective_id != objective.objective_id or u.objective_id is None))
                progress = 1.0 - min(1.0, enemy_units / (objective.required_units * 2))
                progress_delta = progress - objective.completion_progress

            else:  # Default: presence-based progress
                our_units = sum(1 for u in target_cluster.units
                               if u.is_alive and u.objective_id == objective.objective_id)
                progress_delta = min(0.1, our_units * 0.05)

            # Update objective
            updated_objective = objective.update_progress(progress_delta, world_state.step)
            new_objectives.append(updated_objective)

        return world_state.copy_with(objectives=tuple(new_objectives))

    def _compute_advanced_military_reward(
        self,
        world_state: AdvancedWorldMilitaryState,
        prev_state: AdvancedWorldMilitaryState,
        action_type: Optional[str] = None,
    ) -> float:
        """
        Compute advanced military reward considering tactics and operations.
        """
        reward = 0.0

        # 1. Objective completion rewards
        for objective in world_state.objectives:
            if objective.is_completed and not any(
                prev_obj.objective_id == objective.objective_id and prev_obj.is_completed
                for prev_obj in prev_state.objectives
            ):
                reward += objective.reward_value
                print(f"🎯 Objective completed: {objective.name} (+{objective.reward_value} reward)")

        # 2. Objective progress rewards
        for obj, prev_obj in zip(world_state.objectives, prev_state.objectives):
            if obj.objective_id == prev_obj.objective_id:
                progress_reward = (obj.completion_progress - prev_obj.completion_progress) * 5.0
                reward += progress_reward

        # 3. Unit survival rewards
        prev_units = sum(c.unit_count for c in prev_state.clusters)
        current_units = sum(c.unit_count for c in world_state.clusters)
        survival_reward = (current_units - prev_units) * 2.0
        reward += survival_reward

        # 4. Combat power rewards
        prev_power = sum(c.total_combat_power for c in prev_state.clusters)
        current_power = sum(c.total_combat_power for c in world_state.clusters)
        power_reward = (current_power - prev_power) * 0.1
        reward += power_reward

        # 5. Formation rewards
        prev_formations = sum(len(c.formations) for c in prev_state.clusters)
        current_formations = sum(len(c.formations) for c in world_state.clusters)
        formation_reward = (current_formations - prev_formations) * 3.0
        reward += formation_reward

        # 6. Intelligence rewards
        prev_intel = sum(len(c.intelligence_reports) for c in prev_state.clusters)
        current_intel = sum(len(c.intelligence_reports) for c in world_state.clusters)
        intel_reward = (current_intel - prev_intel) * 1.0
        reward += intel_reward

        # 7. Operation completion rewards
        prev_operations = sum(len(c.active_operations) for c in prev_state.clusters)
        current_operations = sum(len(c.active_operations) for c in world_state.clusters)
        completed_operations = sum(
            1 for c in world_state.clusters
            for op in c.active_operations
            if op.is_completed and not any(
                prev_op.operation_id == op.operation_id and prev_op.is_completed
                for prev_op in prev_state.get_cluster(c.cluster_id).active_operations if prev_state.get_cluster(c.cluster_id)
            )
        )
        operation_reward = completed_operations * 5.0
        reward += operation_reward

        # 8. Action-specific rewards
        if action_type:
            if action_type == "deploy":
                reward += 1.0 * world_state.total_unit_count * 0.5
            elif action_type == "attack":
                reward += 0.5 * world_state.total_combat_power * 0.1
            elif action_type == "reinforce":
                reward += 0.3 * world_state.total_unit_count
            elif action_type == "create_formation":
                reward += 2.0 * sum(len(c.formations) for c in world_state.clusters)
            elif action_type == "plan_operation":
                reward += 1.5 * sum(len(c.active_operations) for c in world_state.clusters)
            elif action_type == "execute_operation":
                reward += 3.0 * completed_operations
            elif action_type == "gather_intel":
                reward += 1.0 * sum(len(c.intelligence_reports) for c in world_state.clusters)
            elif action_type == "establish_command":
                reward += 2.5 * sum(1 for c in world_state.clusters if c.command_structure is not None)
            elif action_type == "setup_supply_chain":
                reward += 2.0 * sum(1 for c in world_state.clusters if c.supply_chain is not None)

        return float(reward)

    def _extend_obs(self, base_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Append military features to observation."""
        if self.use_advanced_tactics and self._advanced_military_state is not None:
            return self._extend_advanced_obs(base_obs)
        elif self._military_state is not None:
            return self._extend_basic_obs(base_obs)
        else:
            # Return zeros if no military state
            military_obs_dim = 3 * self._max_N + 5
            return np.concatenate([
                base_obs,
                np.zeros(military_obs_dim, dtype=np.float32)
            ])

    def _extend_basic_obs(self, base_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extend observation with basic military features."""
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

    def _extend_advanced_obs(self, base_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extend observation with advanced military features."""
        # Per-cluster military features
        cluster_features = []
        for i in range(self._max_N):
            if i < len(self._advanced_military_state.clusters):
                cluster = self._advanced_military_state.clusters[i]
                cluster_features.extend([
                    cluster.unit_count,                    # Unit count
                    cluster.total_combat_power,          # Combat power (with formation bonuses)
                    cluster.supply_depot,                 # Supply level
                    len(cluster.formations),              # Number of formations
                    1.0 - cluster.fog_of_war              # Intelligence quality (inverse of fog of war)
                ])
            else:
                cluster_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # Padding

        # Global military features
        global_features = [
            self._advanced_military_state.global_supply,
            self._advanced_military_state.global_reinforcement_pool,
            sum(obj.completion_progress for obj in self._advanced_military_state.objectives) / max(1, len(self._advanced_military_state.objectives)),
            self._advanced_military_state.communication_network_integrity,
            1.0 if self._advanced_military_state.electronic_warfare_active else 0.0,
            float(check_victory_conditions(self._advanced_military_state)['completion_percentage']),
            1.0 if check_victory_conditions(self._advanced_military_state)['victory_achieved'] else 0.0
        ]

        military_obs = np.array(cluster_features + global_features, dtype=np.float32)
        return np.concatenate([base_obs, military_obs])

    def _military_info(self) -> Dict[str, Any]:
        """Build military info dictionary."""
        if self.use_advanced_tactics and self._advanced_military_state is not None:
            return self._advanced_military_info()
        elif self._military_state is not None:
            return self._basic_military_info()
        else:
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

    def _basic_military_info(self) -> Dict[str, Any]:
        """Build basic military info dictionary."""
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

    def _advanced_military_info(self) -> Dict[str, Any]:
        """Build advanced military info dictionary."""
        victory_status = check_victory_conditions(self._advanced_military_state)

        # Count formations and operations
        total_formations = sum(len(c.formations) for c in self._advanced_military_state.clusters)
        total_operations = sum(len(c.active_operations) for c in self._advanced_military_state.clusters)
        total_intel = sum(len(c.intelligence_reports) for c in self._advanced_military_state.clusters)
        total_commands = sum(1 for c in self._advanced_military_state.clusters if c.command_structure is not None)
        total_supply_chains = sum(1 for c in self._advanced_military_state.clusters if c.supply_chain is not None)

        return {
            'total_units': self._advanced_military_state.total_unit_count,
            'total_combat_power': self._advanced_military_state.total_combat_power,
            'global_supply': self._advanced_military_state.global_supply,
            'global_reinforcement_pool': self._advanced_military_state.global_reinforcement_pool,
            'objectives_completed': sum(1 for obj in self._advanced_military_state.objectives if obj.is_completed),
            'objectives_total': len(self._advanced_military_state.objectives),
            'victory_achieved': victory_status['victory_achieved'],
            'completion_percentage': victory_status['completion_percentage'],
            'formations': total_formations,
            'active_operations': total_operations,
            'intelligence_reports': total_intel,
            'command_structures': total_commands,
            'supply_chains': total_supply_chains,
            'communication_integrity': self._advanced_military_state.communication_network_integrity,
            'electronic_warfare_active': self._advanced_military_state.electronic_warfare_active,
            'objectives': [
                {
                    'id': obj.objective_id,
                    'name': obj.name,
                    'type': obj.objective_type,
                    'progress': obj.completion_progress,
                    'completed': obj.is_completed
                }
                for obj in self._advanced_military_state.objectives
            ],
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'units': c.unit_count,
                    'combat_power': c.total_combat_power,
                    'supply': c.supply_depot,
                    'formations': len(c.formations),
                    'operations': len(c.active_operations),
                    'intel_reports': len(c.intelligence_reports),
                    'has_command': c.command_structure is not None,
                    'has_supply_chain': c.supply_chain is not None,
                    'terrain_advantage': c.terrain_advantage,
                    'fog_of_war': c.fog_of_war
                }
                for c in self._advanced_military_state.clusters
            ]
        }

# ─────────────────────────────────────────────────────────────────────────── #
# Utility Functions                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def create_advanced_military_action(
    action_type: str = "deploy",
    target_cluster: int = 0,
    unit_type: str = "INFANTRY",
    intensity: float = 0.5,
    use_advanced: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a military action for testing.

    Args:
        action_type: Type of military action
        target_cluster: Target cluster ID
        unit_type: Type of military unit
        intensity: Action intensity (0-1)
        use_advanced: Whether to create an advanced action
        kwargs: Additional parameters for advanced actions

    Returns:
        Military action dictionary
    """
    if not use_advanced:
        return {
            'action_type': action_type,
            'target_cluster': target_cluster,
            'unit_type': unit_type,
            'intensity': intensity
        }
    else:
        # Advanced action
        if action_type == "create_formation":
            return {
                'action_type': action_type,
                'cluster_id': kwargs.get('cluster_id', 0),
                'formation_type': kwargs.get('formation_type', 'LINE'),
                'unit_ids': kwargs.get('unit_ids', [1, 2, 3])
            }
        elif action_type == "plan_operation":
            return {
                'action_type': action_type,
                'tactic': kwargs.get('tactic', 'FRONTAL_ASSAULT'),
                'primary_target': kwargs.get('primary_target', 1),
                'participating_units': kwargs.get('participating_units', [1, 2])
            }
        elif action_type == "execute_operation":
            return {
                'action_type': action_type,
                'operation_id': kwargs.get('operation_id', 1)
            }
        elif action_type == "gather_intel":
            return {
                'action_type': action_type,
                'cluster_id': kwargs.get('cluster_id', 0),
                'intel_type': kwargs.get('intel_type', 'RECON')
            }
        elif action_type == "establish_command":
            return {
                'action_type': action_type,
                'cluster_id': kwargs.get('cluster_id', 0),
                'commander_unit_id': kwargs.get('commander_unit_id', 1),
                'subordinate_unit_ids': kwargs.get('subordinate_unit_ids', [2, 3]),
                'rank': kwargs.get('rank', 'SERGEANT')
            }
        elif action_type == "setup_supply_chain":
            return {
                'action_type': action_type,
                'cluster_id': kwargs.get('cluster_id', 0),
                'connected_depots': kwargs.get('connected_depots', [1, 2])
            }
        else:
            return {
                'action_type': action_type,
                'target_cluster': target_cluster,
                'unit_type': unit_type,
                'intensity': intensity
            }