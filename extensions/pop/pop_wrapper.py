"""
PopWrapper — Gymnasium wrapper integrating the pop subsystem with GravitasEnv.

Architecture:
  GravitasEnv (unchanged)
      ↓  obs, reward, world
  PopWrapper
      ↓  step_world_pop()  — update pop demographics
      ↓  inject drivers    — adjust cluster hazard/trust/sigma post-RK4
      ↓  extend obs        — append 5*max_N pop aggregates
      ↓  extend reward     — add pop satisfaction/inequality/tension terms

The core env is NEVER modified. All pop logic lives in this wrapper.

Observation extension:
  Original obs:  (4*max_N + 8 + action_dim,)
  Extended obs:  original + (5*max_N,)
  New dims:      [gini, mean_sat, radical_mass, fractionalization, ethnic_tension]
                 per cluster, zero-padded to max_N

Reward extension:
  R_pop = w_sat * mean_satisfaction
        - w_ineq * mean_gini
        - w_tension * mean_ethnic_tension
  Added to base GravitasEnv reward each step.

ODE driver injection:
  After GravitasEnv's RK4 step, PopWrapper reads the ODE drivers
  computed from pop aggregates and applies small adjustments to the
  GravitasWorld cluster states:
    h_i  += hazard_boost[i]    (radical pops raise hazard)
    τ_i  += trust_boost[i]     (satisfied pops build trust)
    σ_i  -= sigma_drag[i]      (class tension drags stability)
  Global Π and Φ receive mean pop pressure terms.
  This is equivalent to soft coupling without modifying the core ODEs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

from .pop_params import PopParams
from .pop_state import WorldPopState, PopAggregates, initialize_world_pop
from .pop_dynamics import step_world_pop, STABILIZE
from .pop_shock import apply_custom_shocks, register_custom_shocks


# ─────────────────────────────────────────────────────────────────────────── #
# PopWrapper                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class PopWrapper(gym.Wrapper):
    """
    Wraps GravitasEnv with a vectorized population subsystem.

    The pop system is categorized along four axes:
      - Job      : encoded in archetype (SUBSISTENCE, LABORER, ARTISAN, etc.)
      - Class    : lower / middle / upper (derived from archetype)
      - Earning  : per-archetype income vector (mean-normalized)
      - Ethnicity: per-cluster ethnic share simplex + cultural distance matrix

    All state is represented as fixed-size vectors (simplex distributions
    and bounded scalars). No per-pop simulation — pure matrix operations.
    Cost is O(N * P) per step where P=8 archetypes, N≤12 clusters.

    Args:
        env:        A GravitasEnv instance.
        pop_params: PopParams config. Uses defaults if None.
        seed:       RNG seed for pop initialization.
    """

    def __init__(
        self,
        env: gym.Env,
        pop_params: Optional[PopParams] = None,
        seed: Optional[int] = None,
        custom_shocks: Optional[List[Dict[str, Any]]] = None,
        nation_to_cluster: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(env)
        self.pop_params = pop_params or PopParams()
        self._rng       = np.random.default_rng(seed)
        
        # Custom shocks
        self.custom_shocks = custom_shocks or []
        self.nation_to_cluster = nation_to_cluster or {}
        register_custom_shocks(self.custom_shocks)

        # Pop state — set in reset()
        self._world_pop: Optional[WorldPopState] = None
        self._last_drivers: Optional[Dict[str, NDArray]] = None

        # Cache max_N from the wrapped env
        self._max_N = getattr(env, "_max_N", 12)
        
        # Aggregate caching
        self._cached_aggregates: Optional[Dict[str, float]] = None
        self._aggregates_dirty: bool = True

        # Extend observation space
        base_shape   = self.env.observation_space.shape[0]
        pop_obs_dim  = 5 * self._max_N   # 5 aggregates × max clusters
        extended_dim = base_shape + pop_obs_dim

        self.observation_space = spaces.Box(
            low  = np.concatenate([
                self.env.observation_space.low,
                np.full(pop_obs_dim, -0.1, dtype=np.float32),
            ]),
            high = np.concatenate([
                self.env.observation_space.high,
                np.ones(pop_obs_dim, dtype=np.float32),
            ]),
            dtype = np.float32,
        )

        # Action space unchanged
        self.action_space = self.env.action_space

    # ── Gymnasium API ──────────────────────────────────────────────────────── #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:

        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed + 1000)

        # Initialize pop state from the freshly reset world
        world = self._get_world()
        N     = world.n_clusters if world else self._max_N
        c_arr = world.cluster_array() if world else None

        cluster_resources = c_arr[:N, 2] if c_arr is not None else None
        cluster_trusts    = c_arr[:N, 4] if c_arr is not None else None

        self._world_pop = initialize_world_pop(
            n_clusters        = N,
            params            = self.pop_params,
            rng               = self._rng,
            cluster_resources = cluster_resources,
            cluster_trusts    = cluster_trusts,
        )
        self._last_drivers = None
        self._cached_aggregates = None
        self._aggregates_dirty = True

        extended_obs = self._extend_obs(obs)
        info["pop"] = self._pop_info()
        return extended_obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:

        # ── 1. Core env step ─────────────────────────────────────────────── #
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ── 2. Decode action for pop effects ─────────────────────────────── #
        stance, intensity, weights = self._decode_action(action)

        # ── 3. Step pop dynamics ─────────────────────────────────────────── #
        world = self._get_world()
        if world is not None and self._world_pop is not None:
            self._world_pop, drivers = step_world_pop(
                world_pop      = self._world_pop,
                world          = world,
                stance         = stance,
                intensity      = intensity,
                action_weights = weights,
                params         = self.pop_params,
                rng            = self._rng,
            )
            self._last_drivers = drivers
            self._aggregates_dirty = True

            # ── 4. Inject ODE drivers back into world state ─────────────── #
            self._inject_drivers(drivers)
            
            # ── 5. Apply custom shocks (if any) ─────────────────────────── #
            if self.custom_shocks and self._rng.random() < 0.1:  # 10% chance per step
                # Select 1-3 random shocks to apply
                n_shocks = min(3, max(1, int(len(self.custom_shocks) * 0.3)))
                selected_shocks = self._rng.choice(self.custom_shocks, n_shocks, replace=False)
                world, self._world_pop = apply_custom_shocks(
                    world, self._world_pop, selected_shocks, self._rng
                )
                self._set_world(world)
                self._aggregates_dirty = True

        # ── 5. Pop reward extension ───────────────────────────────────────── #
        pop_reward = self._compute_pop_reward()
        total_reward = reward + pop_reward

        # ── 6. Extend observation ─────────────────────────────────────────── #
        extended_obs = self._extend_obs(obs)

        # ── 7. Enrich info ─────────────────────────────────────────────────── #
        info["pop"]        = self._pop_info()
        info["pop_reward"] = pop_reward

        return extended_obs, float(total_reward), terminated, truncated, info

    # ── Pop state access ───────────────────────────────────────────────────── #

    @property
    def world_pop(self) -> Optional[WorldPopState]:
        return self._world_pop

    # ── Internal helpers ───────────────────────────────────────────────────── #

    def _get_world(self):
        """Get GravitasWorld from wrapped env, traversing wrapper chain."""
        env = self.env
        while env is not None:
            world = getattr(env, "world", None)
            if world is not None:
                return world
            env = getattr(env, "env", None)
        return None

    def _decode_action(
        self,
        action: Any,
    ) -> Tuple[int, float, NDArray[np.float64]]:
        """
        Decode action into (stance_int, intensity, per-cluster weights).
        Handles both Discrete(6) flat actions and hierarchical Dict actions.
        """
        max_N = self._max_N

        if isinstance(action, dict):
            stance    = int(action["stance"]) % 6
            alloc     = np.asarray(action.get("allocation", np.ones(max_N + 2) / (max_N + 2)))
            intensity = float(np.clip(alloc[max_N], 0.0, 1.0))
            weights   = np.clip(alloc[:max_N], 0.0, 1.0)
            w_sum     = weights.sum()
            weights   = weights / w_sum if w_sum > 1e-9 else np.ones(max_N) / max_N
        else:
            stance    = int(action) % 6
            intensity = 0.6    # default intensity for flat actions
            weights   = np.ones(max_N, dtype=np.float64) / max_N

        return stance, intensity, weights.astype(np.float64)

    def _inject_drivers(self, drivers: Dict[str, NDArray]) -> None:
        """
        Apply pop ODE drivers as soft adjustments to the GravitasWorld state.

        Uses copy_with_clusters() and copy_with_global() — the actual
        GravitasWorld API (not a generic copy_with).
        """
        world = self._get_world()
        if world is None:
            return

        N     = world.n_clusters
        g     = world.global_state

        hazard_boost = drivers["hazard_boost"][:N]
        trust_boost  = drivers["trust_boost"][:N]
        sigma_drag   = drivers["sigma_drag"][:N]
        dt = 0.01

        # ── Per-cluster updates ──────────────────────────────────────────── #
        from gravitas_engine.core.gravitas_state import ClusterState
        new_clusters = []
        for i, cluster in enumerate(world.clusters[:N]):
            new_clusters.append(ClusterState(
                sigma    = float(np.clip(cluster.sigma  - sigma_drag[i]   * dt, 0.0, 1.0)),
                hazard   = float(np.clip(cluster.hazard + hazard_boost[i] * dt, 0.0, 3.0)),
                resource = cluster.resource,
                military = cluster.military,
                trust    = float(np.clip(cluster.trust  + trust_boost[i]  * dt, 0.0, 1.0)),
                polar    = cluster.polar,
            ))

        # ── Global updates ───────────────────────────────────────────────── #
        mean_pol = float(np.mean(drivers["pol_boost"][:N]))
        mean_phi = float(np.mean(drivers["phi_boost"][:N]))

        from gravitas_engine.core.gravitas_state import GlobalState
        new_global = GlobalState(
            exhaustion    = g.exhaustion,
            fragmentation = float(np.clip(g.fragmentation + mean_phi * dt, 0.0, 1.0)),
            polarization  = float(np.clip(g.polarization  + mean_pol * dt, 0.0, 1.0)),
            coherence     = g.coherence,
            military_str  = g.military_str,
            trust         = g.trust,
            step          = g.step,
        )

        # ── Reconstruct using actual GravitasWorld API ───────────────────── #
        new_world = world.copy_with_clusters(new_clusters).copy_with_global(new_global)
        self._set_world(new_world)

    def _set_world(self, new_world) -> None:
        """Push a new GravitasWorld into the wrapped env, traversing wrapper chain."""
        env = self.env
        while env is not None:
            if hasattr(env, "_world"):
                env._world = new_world
                return
            env = getattr(env, "env", None)

    def _extend_obs(self, base_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Append 5*max_N pop aggregate features (cached)."""
        if self._world_pop is None:
            pop_obs = np.zeros(5 * self._max_N, dtype=np.float32)
        else:
            if not self._aggregates_dirty and self._cached_aggregates is not None:
                pop_obs = self._world_pop.obs_flat(self._max_N, cached_aggs=self._cached_aggregates)
            else:
                pop_obs = self._world_pop.obs_flat(self._max_N)
                self._cached_aggregates = self._world_pop.get_aggregates()
                self._aggregates_dirty = False

        return np.concatenate([base_obs, pop_obs]).astype(np.float32)

    def _compute_pop_reward(self) -> float:
        """
        Compute the pop reward extension.

        R_pop = w_sat   * mean_satisfaction       (positive)
              - w_ineq  * mean_gini               (penalty)
              - w_tens  * mean_ethnic_tension      (penalty)
        """
        if self._world_pop is None:
            return 0.0

        mean_agg = self._world_pop.mean_aggregates()
        p        = self.pop_params

        r_pop = (
            p.w_pop_satisfaction * mean_agg.mean_satisfaction
            - p.w_pop_inequality * mean_agg.gini
            - p.w_ethnic_tension * mean_agg.ethnic_tension
        )
        return float(r_pop)

    def _pop_info(self) -> Dict[str, float]:
        """Build the info dict section for pop aggregates."""
        if self._world_pop is None:
            return {}
        agg = self._world_pop.mean_aggregates()
        return agg.to_dict()
