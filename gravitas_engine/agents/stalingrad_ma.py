"""
StalingradMultiAgentEnv — Adversarial two-player environment for Stalingrad.

Two agents (Axis / Soviet) each control a subset of clusters and compete
for territorial control.  The shared GravitasWorld evolves under both
agents' actions simultaneously.

Architecture:
  StalingradMultiAgentEnv   — core two-player env (not Gym-compatible directly)
  SelfPlayEnv(gym.Env)      — SB3-compatible wrapper: trains ONE side while
                               the opponent runs a frozen policy

Observation:
  Same shape as single-agent GravitasEnv (10*max_N + 8 + action_dim = 143).
  Both agents see the full world state — fog-of-war can be added later.

Action:
  Discrete(N_STANCES) per agent.  Each agent's stance is decoded with
  weights masked to only their controlled clusters.

Rewards:
  - Stability advantage:  mean σ(own clusters) − mean σ(enemy clusters)
  - Exhaustion penalty:   shared (both suffer from global exhaustion)
  - Territory bonus:      holding contested clusters (Mamayev Kurgan)
  - Collapse penalty:     per-cluster collapse on your side
"""

from __future__ import annotations

import os
import warnings
from collections import deque
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import sys
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    import gymnasium_shim as gym          # type: ignore[no-redef]
    from gymnasium_shim import spaces     # type: ignore[no-redef]

from ..core.gravitas_dynamics import rk4_step
from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import ClusterState, GravitasWorld, GlobalState
from ..systems.gravitas_reward import compute_reward
from ..systems.hawkes_shock import apply_shock, sample_shock, update_hawkes
from ..systems.media_bias import distort_observation, update_media_bias
from ..systems.topology import (
    build_topology,
    initialize_cluster_states,
    initialize_global_state,
)
from ..systems.diplomacy import build_alliance, decay_alliance
from ..systems.population import initialize_population, step_population, population_cluster_feedback
from ..systems.economy import initialize_economy, step_economy, economy_cluster_feedback
from .gravitas_actions import (
    HierarchicalAction,
    Stance,
    N_STANCES,
    RESOURCE_COSTS,
    apply_action,
    decode_flat_action,
)

# ─────────────────────────────────────────────────────────────────────────── #
# Constants                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

AXIS   = 0
SOVIET = 1
SIDE_NAMES = {AXIS: "Axis", SOVIET: "Soviet"}

# Default cluster assignments (from stalingrad.yaml agents section)
DEFAULT_AXIS_CLUSTERS   = [0, 1, 2, 4, 5, 8]  # includes Wintergewitter corridor
DEFAULT_SOVIET_CLUSTERS = [3, 6]
CONTESTED_CLUSTERS      = [2]   # Mamayev Kurgan — initially Axis but fiercely contested

_SIGMA_WINDOW = 20


# ─────────────────────────────────────────────────────────────────────────── #
# Side-aware action decoder                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def decode_side_action(
    action_int: int,
    world: GravitasWorld,
    params: GravitasParams,
    rng: np.random.Generator,
    controlled_clusters: List[int],
) -> HierarchicalAction:
    """
    Decode a flat integer action into a HierarchicalAction with weights
    masked to only the agent's controlled clusters.
    """
    stance = Stance(action_int % N_STANCES)
    N      = world.n_clusters
    max_N  = params.n_clusters_max
    c_arr  = world.cluster_array()

    # Compute target scores (same heuristic as single-agent)
    if stance == Stance.MILITARIZE:
        scores = c_arr[:, 1]
    elif stance == Stance.REFORM:
        scores = 1.0 - c_arr[:, 4]
    elif stance == Stance.INVEST:
        scores = 1.0 - c_arr[:, 2]
    elif stance == Stance.STABILIZE:
        scores = 1.0 - c_arr[:, 0]
    elif stance == Stance.PROPAGANDA:
        scores = np.abs(world.media_bias[:N]) if world.media_bias is not None else np.ones(N)
    elif stance == Stance.DECENTRALIZE:
        scores = c_arr[:, 5]
    elif stance == Stance.DIPLOMACY:
        scores = (1.0 - c_arr[:, 4]) + (1.0 - c_arr[:, 0])
    else:  # REDEPLOY
        scores = c_arr[:, 1]

    # Mask: zero out clusters not controlled by this side
    mask = np.zeros(N, dtype=np.float64)
    for idx in controlled_clusters:
        if idx < N:
            mask[idx] = 1.0

    scores = np.clip(scores, 0.0, 5.0) * mask
    total = scores.sum()
    if total > 0:
        w_raw = scores / total
    else:
        # Fallback: uniform over controlled clusters
        w_raw = mask / max(mask.sum(), 1.0)

    weights = np.zeros(max_N)
    weights[:N] = w_raw

    intensity = 0.7
    lt_bias   = 0.5
    prop_load = intensity if stance == Stance.PROPAGANDA else 0.0
    mil_load  = float(np.mean(c_arr[controlled_clusters, 3])) if controlled_clusters else 0.0
    r_cost    = RESOURCE_COSTS[stance] * intensity

    return HierarchicalAction(
        stance=stance,
        weights=weights,
        intensity=intensity,
        lt_bias=lt_bias,
        propaganda_load=prop_load,
        military_load=mil_load,
        resource_cost=r_cost,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Adversarial reward                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_side_reward(
    world: GravitasWorld,
    prev_world: GravitasWorld,
    params: GravitasParams,
    side: int,
    own_clusters: List[int],
    enemy_clusters: List[int],
    contested: List[int],
) -> float:
    """
    Compute reward for one side.

    Components:
      1. Own stability:        mean σ of own clusters              (+)
      2. Enemy pressure:       mean σ of enemy clusters            (−)
      3. Territory advantage:  σ of contested clusters (weighted)  (+)
      4. Exhaustion penalty:   global exhaustion                   (−)
      5. Hazard penalty:       mean hazard on own clusters         (−)
      6. Delta bonuses:        improvement in own σ over prev step (+)
    """
    c_arr  = world.cluster_array()
    c_prev = prev_world.cluster_array()
    g      = world.global_state
    N      = world.n_clusters

    # 1. Own cluster stability
    own_sigma = np.mean(c_arr[own_clusters, 0]) if own_clusters else 0.0

    # 2. Enemy cluster stability (we want this LOW)
    enemy_sigma = np.mean(c_arr[enemy_clusters, 0]) if enemy_clusters else 0.0

    # 3. Contested cluster control
    contested_sigma = np.mean(c_arr[contested, 0]) if contested else 0.0

    # 4. Global exhaustion penalty (both sides suffer)
    exh_penalty = g.exhaustion

    # 5. Own hazard
    own_hazard = np.mean(c_arr[own_clusters, 1]) if own_clusters else 0.0

    # 6. Improvement delta
    own_sigma_prev = np.mean(c_prev[own_clusters, 0]) if own_clusters else 0.0
    delta_sigma = own_sigma - own_sigma_prev

    # Reward assembly
    r = 0.0
    r += 3.0 * own_sigma                # Maintain own stability
    r -= 2.0 * enemy_sigma              # Reduce enemy stability
    r += 1.5 * contested_sigma          # Hold contested ground (both want high σ there)
    r -= 3.0 * exh_penalty              # Exhaustion kills both sides
    r -= 1.5 * own_hazard               # Own hazard is bad
    r += 5.0 * delta_sigma              # Reward improvement

    # Collapse check: big penalty if any own cluster has σ < 0.05
    for idx in own_clusters:
        if idx < N and c_arr[idx, 0] < 0.05:
            r -= 5.0

    return float(r)


# ─────────────────────────────────────────────────────────────────────────── #
# Core multi-agent environment                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class StalingradMultiAgentEnv:
    """
    Two-player adversarial Stalingrad environment.

    NOT a Gym env directly — use SelfPlayEnv for SB3 training.
    This class manages the shared world and dispatches per-side obs/rewards.
    """

    def __init__(
        self,
        params: GravitasParams,
        axis_clusters: Optional[List[int]] = None,
        soviet_clusters: Optional[List[int]] = None,
        contested_clusters: Optional[List[int]] = None,
        initial_clusters: Optional[NDArray] = None,
        initial_alliances: Optional[NDArray] = None,
        seed: int = 42,
    ) -> None:
        self.params = params
        self.axis_clusters   = axis_clusters or DEFAULT_AXIS_CLUSTERS
        self.soviet_clusters = soviet_clusters or DEFAULT_SOVIET_CLUSTERS
        self.contested       = contested_clusters or CONTESTED_CLUSTERS

        self._initial_clusters  = initial_clusters
        self._initial_alliances = initial_alliances
        self._seed = seed
        self._rng  = np.random.default_rng(seed)

        self._max_N = params.n_clusters_max
        self._action_dim = self._max_N + 3

        # Observation / action spaces (same shape as single-agent for model compat)
        obs_dim = 10 * self._max_N + 8 + self._action_dim
        self.observation_space = spaces.Box(
            low=-2.0, high=6.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_STANCES)

        # State
        self._world: Optional[GravitasWorld] = None
        self._prev_world: Optional[GravitasWorld] = None
        self._cur_N: int = 8
        self._sigma_hist: deque = deque(maxlen=_SIGMA_WINDOW)
        self._shock_log: List = []
        self._step_count: int = 0
        self._collapse_cause: Optional[str] = None
        self._volga_reinforced: int = 0

        # Per-side prev actions
        self._prev_actions = {
            AXIS: self._zero_action(),
            SOVIET: self._zero_action(),
        }

    def _zero_action(self) -> HierarchicalAction:
        return HierarchicalAction(
            stance=Stance.STABILIZE,
            weights=np.zeros(self._max_N),
            intensity=0.0, lt_bias=0.5,
            propaganda_load=0.0, military_load=0.0,
            resource_cost=0.0,
        )

    # ── Reset ───────────────────────────────────────────────────────────── #

    def reset(self, seed: Optional[int] = None) -> Dict[int, NDArray[np.float32]]:
        """Reset world. Returns {AXIS: obs, SOVIET: obs}."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        N = len(self.axis_clusters) + len(self.soviet_clusters)
        # Account for uncontrolled clusters (e.g., Romanian sector)
        all_mentioned = set(self.axis_clusters) | set(self.soviet_clusters)
        if self._initial_clusters is not None:
            N = max(N, self._initial_clusters.shape[0])
        else:
            N = max(all_mentioned) + 1 if all_mentioned else 8

        # Topology
        adjacency = self._rng.random((N, N)) < self.params.between_link_prob
        np.fill_diagonal(adjacency, False)
        adjacency = adjacency | adjacency.T
        conflict = self._rng.random((N, N)) < self.params.conflict_link_prob
        np.fill_diagonal(conflict, False)
        conflict = conflict & ~adjacency
        conflict = conflict | conflict.T
        self._cur_N = N

        # Cluster states
        clusters = initialize_cluster_states(N, self.params, self._rng)
        global_state = initialize_global_state(self.params, self._rng)

        if self._initial_clusters is not None:
            clusters = [ClusterState.from_array(self._initial_clusters[i]) for i in range(N)]

        # Alliance
        alliance = build_alliance(N, adjacency, self._rng, self._max_N)
        if self._initial_alliances is not None:
            alliance = self._initial_alliances.copy()

        # Population + economy
        media_bias = self._rng.uniform(-0.05, 0.05, N)
        population = initialize_population(N, self._max_N, self._rng)
        c_arr_init = np.array([c.to_array() for c in clusters])
        economy    = initialize_economy(N, self._max_N, c_arr_init, self._rng)

        self._world = GravitasWorld(
            clusters=tuple(clusters),
            global_state=global_state,
            adjacency=adjacency,
            conflict=conflict,
            media_bias=media_bias,
            shock_rate=self.params.hawkes_base_rate,
            hawkes_sum=0.0,
            alliance=alliance,
            population=population,
            economy=economy,
        )
        self._prev_world = self._world

        # Reset buffers
        self._sigma_hist.clear()
        self._shock_log.clear()
        self._step_count = 0
        self._collapse_cause = None
        self._volga_reinforced = 0
        self._prev_actions = {AXIS: self._zero_action(), SOVIET: self._zero_action()}

        return {
            AXIS:   self._make_observation(AXIS),
            SOVIET: self._make_observation(SOVIET),
        }

    # ── Step ────────────────────────────────────────────────────────────── #

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[
        Dict[int, NDArray[np.float32]],   # obs
        Dict[int, float],                 # rewards
        bool,                             # terminated
        bool,                             # truncated
        Dict[str, Any],                   # info
    ]:
        """
        Both agents submit actions simultaneously.

        actions: {AXIS: int, SOVIET: int}
        """
        assert self._world is not None, "Call reset() first"
        self._prev_world = self._world

        # 1. Decode both sides' actions
        axis_hier = decode_side_action(
            actions[AXIS], self._world, self.params, self._rng, self.axis_clusters,
        )
        soviet_hier = decode_side_action(
            actions[SOVIET], self._world, self.params, self._rng, self.soviet_clusters,
        )

        # 2. Apply both actions to the world (order doesn't matter — disjoint clusters)
        world, _ = apply_action(self._world, axis_hier, self.params, self._rng)
        world, _ = apply_action(world, soviet_hier, self.params, self._rng)

        # 3. Merge propaganda/military loads from both sides
        total_prop_load = axis_hier.propaganda_load + soviet_hier.propaganda_load
        total_mil_load  = (axis_hier.military_load + soviet_hier.military_load) / 2.0
        merged_weights  = axis_hier.weights + soviet_hier.weights

        # 4. RK4 integration
        world = rk4_step(
            world=world,
            params=self.params,
            military_load=total_mil_load,
            propaganda_load=total_prop_load,
            sigma_noise=self.params.sigma_obs_base,
            rng=self._rng,
        )

        # 5. Shock process
        shock = sample_shock(world, self.params, self._rng)
        shock_info: Dict[str, Any] = {}
        if shock is not None:
            world, shock_info = apply_shock(world, shock, self.params, self._rng)
            self._shock_log.append(shock_info)

        new_hawkes_rate, new_hawkes_sum = update_hawkes(
            world.hawkes_sum,
            shock_occurred=(shock is not None),
            params=self.params,
        )
        world = world.copy_with_shock(new_hawkes_rate, new_hawkes_sum)

        # 6. Media bias update (merged propaganda from both sides)
        N = world.n_clusters
        new_bias = update_media_bias(
            world=world,
            propaganda_weights=merged_weights[:N],
            propaganda_intensity=total_prop_load,
            shock_occurred=(shock is not None),
            shock_cluster=(shock.cluster_idx if shock is not None else -1),
            params=self.params,
            rng=self._rng,
        )
        world = world.copy_with_bias(new_bias)

        # 7. Alliance decay + population + economy
        if world.alliance is not None:
            world = world.copy_with_alliance(decay_alliance(world.alliance, self.params))

        c_arr_now = world.cluster_array()
        if world.population is not None:
            new_pop = step_population(
                world.population, c_arr_now, world.alliance, self.params, N, self._rng,
            )
            delta_pop = population_cluster_feedback(
                new_pop, c_arr_now, self.params, N, self.params.dt,
            )
            new_clusters_pop = [
                ClusterState.from_array(np.clip(c_arr_now[i] + delta_pop[i], 0.0, 1.0))
                for i in range(N)
            ]
            world = world.copy_with_clusters(new_clusters_pop).copy_with_population(new_pop)
            c_arr_now = world.cluster_array()

        if world.economy is not None:
            new_eco = step_economy(
                world.economy, c_arr_now,
                world.population, world.adjacency, world.alliance,
                self.params, N, self._rng,
            )
            delta_eco = economy_cluster_feedback(new_eco, c_arr_now, self.params, N, self.params.dt)
            _raw_eco = np.clip(c_arr_now + delta_eco, 0.0, 1.0)
            new_clusters_eco = [
                ClusterState._from_array_fast(_raw_eco[i]) for i in range(N)
            ]
            world = world.copy_with_clusters(new_clusters_eco).copy_with_economy(new_eco)

        # 7b. Conditional Soviet Volga reinforcements
        #     Every 50 steps, if Volga Crossing (cluster 3) is stable and
        #     not under heavy attack, Soviet reserves (cluster 6) receive
        #     +10% military and +5% resources — matching the historical
        #     nightly barge crossings that depended on a secure crossing.
        c_arr_now = world.cluster_array()
        _volga_idx, _reserve_idx = 3, 6
        if (
            self._step_count > 0
            and self._step_count % 50 == 0
            and _volga_idx < N
            and _reserve_idx < N
            and _volga_idx in self.soviet_clusters
            and c_arr_now[_volga_idx, 0] > 0.5    # σ > 0.5: stable control
            and c_arr_now[_volga_idx, 1] < 0.7    # hazard < 0.7: not under heavy fire
        ):
            c_arr_mod = c_arr_now.copy()
            c_arr_mod[_reserve_idx, 3] = min(0.95, c_arr_mod[_reserve_idx, 3] + 0.10)  # +10% military
            c_arr_mod[_reserve_idx, 2] = min(1.00, c_arr_mod[_reserve_idx, 2] + 0.05)  # +5% resources
            new_clusters_reinf = [
                ClusterState.from_array(c_arr_mod[i]) for i in range(N)
            ]
            world = world.copy_with_clusters(new_clusters_reinf)
            self._volga_reinforced = getattr(self, '_volga_reinforced', 0) + 1

        # 8. Advance step
        world = world.advance_step()
        self._world = world
        self._step_count += 1

        # Sigma history
        sigma_now = world.cluster_array()[:N, 0]
        sigma_padded = np.zeros(self._max_N)
        sigma_padded[:N] = sigma_now
        self._sigma_hist.append(sigma_padded)

        # Store prev actions
        self._prev_actions[AXIS]   = axis_hier
        self._prev_actions[SOVIET] = soviet_hier

        # 9. Termination
        terminated, cause = self._check_termination(world)
        if terminated:
            self._collapse_cause = cause
        truncated = (world.global_state.step >= self.params.max_steps)

        # 10. Per-side rewards
        rewards = {
            AXIS: compute_side_reward(
                world, self._prev_world, self.params,
                AXIS, self.axis_clusters, self.soviet_clusters, self.contested,
            ),
            SOVIET: compute_side_reward(
                world, self._prev_world, self.params,
                SOVIET, self.soviet_clusters, self.axis_clusters, self.contested,
            ),
        }

        # 11. Observations
        obs = {
            AXIS:   self._make_observation(AXIS),
            SOVIET: self._make_observation(SOVIET),
        }

        # 12. Info
        c_arr = world.cluster_array()
        info = {
            "step": self._step_count,
            "axis_mean_sigma": float(np.mean(c_arr[self.axis_clusters, 0])),
            "soviet_mean_sigma": float(np.mean(c_arr[self.soviet_clusters, 0])),
            "exhaustion": world.global_state.exhaustion,
            "collapse_cause": cause,
            "shock_occurred": shock is not None,
            "axis_action": int(actions[AXIS]),
            "soviet_action": int(actions[SOVIET]),
            "volga_reinforcements": self._volga_reinforced,
        }

        return obs, rewards, terminated, truncated, info

    # ── Observation builder ─────────────────────────────────────────────── #

    def _make_observation(self, side: int) -> NDArray[np.float32]:
        """
        Build observation for one side.
        Same layout as single-agent GravitasEnv for model compatibility.
        """
        N      = self._cur_N
        max_N  = self._max_N
        world  = self._world
        params = self.params

        prev_act = self._prev_actions[side]
        prev_act_arr = prev_act.to_array()

        raw_obs = distort_observation(world, params, self._rng, prev_act_arr)

        cluster_part = raw_obs[:N * 3]
        global_part  = raw_obs[N*3 : N*3+6]
        bias_part    = raw_obs[N*3+6 : N*3+6+N]
        action_part  = raw_obs[N*3+6+N : N*3+6+N+len(prev_act_arr)]
        step_part    = raw_obs[-1:]

        cluster_padded = np.zeros(max_N * 3, dtype=np.float32)
        cluster_padded[:N*3] = cluster_part

        bias_padded = np.zeros(max_N, dtype=np.float32)
        bias_padded[:N] = bias_part

        shock_rate_arr = np.array([float(world.shock_rate)], dtype=np.float32)

        if world.alliance is not None:
            alliance_net = np.mean(world.alliance[:N, :N], axis=1).astype(np.float32)
        else:
            alliance_net = np.zeros(N, dtype=np.float32)
        alliance_padded = np.zeros(max_N, dtype=np.float32)
        alliance_padded[:N] = alliance_net

        if world.population is not None:
            pop_vals = world.population[:N].astype(np.float32)
        else:
            pop_vals = np.full(N, 0.65, dtype=np.float32)
        pop_padded = np.zeros(max_N, dtype=np.float32)
        pop_padded[:N] = pop_vals

        if world.economy is not None:
            eco_vals = world.economy[:N].astype(np.float32)
        else:
            eco_vals = np.tile([0.5, 0.15, 0.15, 0.5], (N, 1)).astype(np.float32)
        eco_padded = np.zeros((max_N, 4), dtype=np.float32)
        eco_padded[:N] = eco_vals

        obs = np.concatenate([
            cluster_padded,
            global_part,
            bias_padded,
            shock_rate_arr,
            alliance_padded,
            pop_padded,
            eco_padded.flatten(),
            action_part,
            step_part,
        ]).astype(np.float32)

        return obs

    # ── Termination check ───────────────────────────────────────────────── #

    def _check_termination(self, world: GravitasWorld) -> Tuple[bool, Optional[str]]:
        g = world.global_state
        c = world.cluster_array()
        if g.exhaustion >= self.params.collapse_exhaustion:
            return True, "exhaustion_collapse"
        if float(np.mean(c[:, 1])) >= self.params.collapse_hazard_mean:
            return True, "hazard_cascade"
        if g.polarization >= self.params.collapse_polarization:
            return True, "polarization_lock"
        if g.fragmentation >= self.params.collapse_fragmentation:
            return True, "fragmentation"
        return False, None

    @property
    def world(self) -> Optional[GravitasWorld]:
        return self._world


# ─────────────────────────────────────────────────────────────────────────── #
# Self-play wrapper (SB3-compatible Gym env)                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class SelfPlayEnv(gym.Env):
    """
    Wraps StalingradMultiAgentEnv as a standard Gym env for SB3.

    One side ("trainable_side") is controlled by the RL agent.
    The opponent uses a frozen policy (model or random).

    Compatible with RecurrentPPO — same obs/action spaces as GravitasEnv.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        params: GravitasParams,
        trainable_side: int = AXIS,
        opponent_policy: Optional[Any] = None,
        axis_clusters: Optional[List[int]] = None,
        soviet_clusters: Optional[List[int]] = None,
        contested_clusters: Optional[List[int]] = None,
        initial_clusters: Optional[NDArray] = None,
        initial_alliances: Optional[NDArray] = None,
        seed: int = 42,
        opponent_deterministic: bool = True,
    ) -> None:
        super().__init__()

        self._ma_env = StalingradMultiAgentEnv(
            params=params,
            axis_clusters=axis_clusters,
            soviet_clusters=soviet_clusters,
            contested_clusters=contested_clusters,
            initial_clusters=initial_clusters,
            initial_alliances=initial_alliances,
            seed=seed,
        )
        self.trainable_side = trainable_side
        self.opponent_side  = SOVIET if trainable_side == AXIS else AXIS
        self._opponent_policy = opponent_policy
        self._opponent_deterministic = opponent_deterministic

        # Spaces (same as single-agent for model compat)
        self.observation_space = self._ma_env.observation_space
        self.action_space      = self._ma_env.action_space

        # Opponent LSTM states (for RecurrentPPO opponent)
        self._opp_lstm_states = None
        self._opp_episode_starts = np.ones((1,), dtype=bool)
        self._opp_obs: Optional[NDArray] = None

    def set_opponent(self, policy: Any) -> None:
        """Hot-swap the opponent policy (for iterative self-play)."""
        self._opponent_policy = policy
        self._opp_lstm_states = None
        self._opp_episode_starts = np.ones((1,), dtype=bool)

    # ── Gym API ──────────────────────────────────────────────────────────── #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        obs_dict = self._ma_env.reset(seed=seed)
        self._opp_obs = obs_dict[self.opponent_side]
        self._opp_lstm_states = None
        self._opp_episode_starts = np.ones((1,), dtype=bool)
        info = {"side": SIDE_NAMES[self.trainable_side]}
        return obs_dict[self.trainable_side], info

    def step(
        self,
        action: int,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        # Get opponent action
        opp_action = self._get_opponent_action()

        # Build joint action
        actions = {
            self.trainable_side: int(action),
            self.opponent_side:  opp_action,
        }

        # Step the multi-agent env
        obs_dict, rewards, terminated, truncated, info = self._ma_env.step(actions)

        # Store opponent obs for next step
        self._opp_obs = obs_dict[self.opponent_side]
        self._opp_episode_starts = np.zeros((1,), dtype=bool)

        # Add side info
        info["own_reward"]  = rewards[self.trainable_side]
        info["opp_reward"]  = rewards[self.opponent_side]
        info["opp_action"]  = opp_action
        info["side"]        = SIDE_NAMES[self.trainable_side]

        return (
            obs_dict[self.trainable_side],
            rewards[self.trainable_side],
            terminated,
            truncated,
            info,
        )

    def _get_opponent_action(self) -> int:
        """Query the opponent policy for an action."""
        if self._opponent_policy is None:
            # Random opponent
            return int(self.action_space.sample())

        # RecurrentPPO opponent
        obs = self._opp_obs
        if obs is None:
            return int(self.action_space.sample())

        try:
            action, self._opp_lstm_states = self._opponent_policy.predict(
                obs.reshape(1, -1),
                state=self._opp_lstm_states,
                episode_start=self._opp_episode_starts,
                deterministic=self._opponent_deterministic,
            )
            return int(action[0])
        except Exception:
            return int(self.action_space.sample())

    def render(self) -> Optional[str]:
        w = self._ma_env.world
        if w is None:
            return None
        c = w.cluster_array()
        g = w.global_state
        N = self._ma_env._cur_N
        lines = [
            f"╔════════════════ STALINGRAD MA ═══════════════════╗",
            f"║ Step {self._ma_env._step_count:>4}  Exh={g.exhaustion:.3f}  Shocks={len(self._ma_env._shock_log)}",
            f"║ Axis  σ̄={np.mean(c[self._ma_env.axis_clusters, 0]):.3f}  "
            f"Soviet σ̄={np.mean(c[self._ma_env.soviet_clusters, 0]):.3f}",
            f"╠═══════════════════════════════════════════════════╣",
        ]
        for i in range(N):
            side = "AX" if i in self._ma_env.axis_clusters else "SV" if i in self._ma_env.soviet_clusters else "??"
            row = c[i]
            lines.append(
                f"║ [{i}] {side} σ={row[0]:.3f} h={row[1]:.3f} r={row[2]:.3f} m={row[3]:.3f}"
            )
        lines.append("╚═══════════════════════════════════════════════════╝")
        output = "\n".join(lines)
        print(output)
        return output
