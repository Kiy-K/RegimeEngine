"""
GravitasEnv — Main Gymnasium environment for GRAVITAS.

"Governance as control of chaos under distorted perception and power trade-offs."

Architecture:
  reset()  → initialize world, topology, bias; return obs
  step()   → decode action → apply → RK4 integrate → shock → bias update
           → compute reward → check termination → return (obs, r, done, info)

Flat action space (for PPO baseline):
  Discrete(6)  — one per stance
  Allocation and intensity are determined by the heuristic decoder.

Full hierarchical action space (for HRL):
  Use GravitasEnv(hierarchical=True); action = (stance, weights, θ, γ)

Observation space:
  Box(float32, shape=(4N + 8 + action_dim,))
  4N = N×3 cluster obs + N bias estimate
  8  = 6 global vars + step_frac + shock_rate
  action_dim = N + 3 (prev_action embedding)
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import sys, os
    # Add the project root so the shim is findable
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    import gymnasium_shim as gym          # type: ignore[no-redef]
    from gymnasium_shim import spaces     # type: ignore[no-redef]

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_dynamics import compute_hazard, rk4_step
from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import GravitasWorld, GlobalState
from ..systems.gravitas_reward import RewardBreakdown, compute_reward
from ..systems.hawkes_shock import apply_shock, sample_shock, update_hawkes
from ..systems.media_bias import distort_observation, update_media_bias
from ..systems.topology import (
    build_topology,
    initialize_cluster_states,
    initialize_global_state,
)
from .gravitas_actions import (
    HierarchicalAction,
    Stance,
    N_STANCES,
    apply_action,
    decode_flat_action,
)

GravitasConfig = GravitasParams

# Rolling window for smoothness term
_SIGMA_WINDOW = 20


class GravitasEnv(gym.Env):
    """
    GRAVITAS — research-grade governance RL environment.

    Partial observability:
      True state (σ, h, τ, p, m, Π, Φ, E, Ψ) is never directly seen.
      Agent sees distorted observations shaped by media_bias and coherence noise.

    Key non-linearities:
      - Hawkes shock arrivals (memory of past shocks)
      - Superlinear hazard tipping (κ_h, κ_p > 1)
      - Exhaustion gates all recovery (high E → system freezes)
      - Propaganda raises perceived stability but increases true polarization
      - Military is a loan against future exhaustion and polarization
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        params: Optional[GravitasParams] = None,
        hierarchical: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.params       = params or GravitasParams()
        self.hierarchical = hierarchical
        self.render_mode  = render_mode

        # RNG
        _seed = seed if seed is not None else self.params.seed
        self._rng = np.random.default_rng(_seed)

        # Placeholders (set in reset)
        self._world:       Optional[GravitasWorld]        = None
        self._prev_world:  Optional[GravitasWorld]        = None
        self._prev_action: Optional[HierarchicalAction]   = None
        self._sigma_hist:  deque                          = deque(maxlen=_SIGMA_WINDOW)
        self._shock_log:   List[Dict[str, Any]]           = []
        self._reward_log:  List[RewardBreakdown]          = []
        self._collapse_cause: Optional[str]               = None

        # We need to know N to define observation/action spaces.
        # Use params.n_clusters as the fixed maximum for space definition;
        # actual N is randomized per episode but padded to this max.
        self._max_N = self.params.n_clusters_max
        self._cur_N = self.params.n_clusters   # updated in reset

        self._action_dim = self._max_N + 3   # N weights + intensity + lt_bias + stance_enc

        # Define spaces
        self._define_spaces()

    # ──────────────────────────────────────────────────────────────────────── #
    # Space definitions                                                         #
    # ──────────────────────────────────────────────────────────────────────── #

    def _define_spaces(self) -> None:
        """
        Observation: Box(float32, obs_dim)
        Action:      Discrete(6) for flat mode
        """
        N = self._max_N
        # 4N + 8 + action_dim
        # = N*3 (cluster obs) + N (bias est) + 6 (global) + 1 (step_frac) + 1 (shock_rate)
        # + action_dim (prev_action)
        obs_dim = 4 * N + 8 + self._action_dim
        self.observation_space = spaces.Box(
            low=-2.0, high=6.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        if self.hierarchical:
            # For HRL: (stance, N weights + θ + γ)
            # Worker outputs continuous allocation; manager picks stance
            self.action_space = spaces.Dict({
                "stance":    spaces.Discrete(N_STANCES),
                "allocation": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self._max_N + 2,),
                    dtype=np.float32,
                ),
            })
        else:
            # Flat: Discrete(6) → one stance per step
            self.action_space = spaces.Discrete(N_STANCES)

    def update_config(self, **kwargs):
        """Update environment parameters (replaces frozen dataclass)."""
        from dataclasses import replace
        valid_kwargs = {}
        for k, v in kwargs.items():
            if k == "shock_prob_per_step" and not hasattr(self.params, k):
                if hasattr(self.params, "hawkes_base_rate"):
                    valid_kwargs["hawkes_base_rate"] = v
            elif hasattr(self.params, k):
                valid_kwargs[k] = v
        
        if valid_kwargs:
            self.params = replace(self.params, **valid_kwargs)

    # ──────────────────────────────────────────────────────────────────────── #
    # Reset                                                                     #
    # ──────────────────────────────────────────────────────────────────────── #

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:

        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build fresh topology
        N, adjacency, conflict = build_topology(self.params, self._rng)
        self._cur_N = N

        # Initialize states
        clusters     = initialize_cluster_states(N, self.params, self._rng)
        global_state = initialize_global_state(self.params, self._rng)

        # Initial media bias: near-zero with small random perturbation
        media_bias = self._rng.uniform(-0.05, 0.05, N)

        # Build world
        self._world = GravitasWorld(
            clusters=tuple(clusters),
            global_state=global_state,
            adjacency=adjacency,
            conflict=conflict,
            media_bias=media_bias,
            shock_rate=self.params.hawkes_base_rate,
            hawkes_sum=0.0,
        )
        self._prev_world = self._world

        # Reset episode buffers
        self._sigma_hist.clear()
        self._shock_log.clear()
        self._reward_log.clear()
        self._collapse_cause = None

        # Dummy previous action (zeros)
        zero_weights = np.zeros(self._max_N)
        self._prev_action = HierarchicalAction(
            stance=Stance.STABILIZE,
            weights=zero_weights,
            intensity=0.0, lt_bias=0.5,
            propaganda_load=0.0, military_load=0.0,
            resource_cost=0.0,
        )

        obs  = self._make_observation()
        info = self._make_info(reward=None)
        return obs, info

    # ──────────────────────────────────────────────────────────────────────── #
    # Step                                                                      #
    # ──────────────────────────────────────────────────────────────────────── #

    def step(
        self,
        action: Any,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:

        assert self._world is not None, "Call reset() before step()"

        self._prev_world = self._world

        # ── 1. Decode action ────────────────────────────────────────────── #
        hier_action = self._decode_action(action)

        # ── 2. Apply immediate action effects ───────────────────────────── #
        world, _ = apply_action(self._world, hier_action, self.params, self._rng)

        # ── 3. RK4 integration (deterministic backbone + SDE noise) ──────── #
        world = rk4_step(
            world=world,
            params=self.params,
            military_load=hier_action.military_load,
            propaganda_load=hier_action.propaganda_load,
            sigma_noise=self.params.sigma_obs_base,
            rng=self._rng,
        )

        # ── 4. Shock process ─────────────────────────────────────────────── #
        shock      = sample_shock(world, self.params, self._rng)
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

        # ── 5. Media bias update ─────────────────────────────────────────── #
        N = world.n_clusters
        new_bias = update_media_bias(
            world=world,
            propaganda_weights=hier_action.weights[:N],
            propaganda_intensity=(
                hier_action.propaganda_load if hier_action.stance == Stance.PROPAGANDA else 0.0
            ),
            shock_occurred=(shock is not None),
            shock_cluster=(shock.cluster_idx if shock is not None else -1),
            params=self.params,
            rng=self._rng,
        )
        world = world.copy_with_bias(new_bias)

        # ── 6. Advance step counter ──────────────────────────────────────── #
        world = world.advance_step()
        self._world = world

        # ── 7. Update sigma history for smoothness term ──────────────────── #
        sigma_now = world.cluster_array()[:N, 0]
        # Pad to max_N for consistent history shape
        sigma_padded = np.zeros(self._max_N)
        sigma_padded[:N] = sigma_now
        self._sigma_hist.append(sigma_padded)
        sigma_history = np.array(self._sigma_hist)

        # ── 8. Compute reward ────────────────────────────────────────────── #
        action_was_noop = (hier_action.intensity == 0.0)
        breakdown = compute_reward(
            world=world,
            prev_world=self._prev_world,
            params=self.params,
            sigma_history=sigma_history,
            action_was_noop=action_was_noop,
        )
        self._reward_log.append(breakdown)
        self._prev_action = hier_action

        # ── 9. Termination check ─────────────────────────────────────────── #
        terminated, cause = self._check_termination(world)
        if terminated:
            self._collapse_cause = cause
        truncated = (world.global_state.step >= self.params.max_steps)

        # ── 10. Build outputs ────────────────────────────────────────────── #
        obs  = self._make_observation()
        info = self._make_info(reward=breakdown, shock=shock_info, cause=cause)

        return obs, float(breakdown.total), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────── #
    # Internal helpers                                                          #
    # ──────────────────────────────────────────────────────────────────────── #

    def _decode_action(self, action: Any) -> HierarchicalAction:
        if self.hierarchical:
            stance_int  = int(action["stance"])
            alloc       = np.asarray(action["allocation"], dtype=np.float64)
            N           = self._cur_N
            weights_raw = np.clip(alloc[:N], 0.0, 1.0)
            w_sum       = weights_raw.sum()
            weights     = (weights_raw / w_sum) if w_sum > 0 else np.ones(N) / N
            # Pad weights to max_N
            weights_pad = np.zeros(self._max_N)
            weights_pad[:N] = weights
            intensity   = float(np.clip(alloc[self._max_N], 0.0, 1.0))
            lt_bias     = float(np.clip(alloc[self._max_N + 1], 0.0, 1.0))
            stance      = Stance(stance_int % N_STANCES)
            prop_load   = intensity if stance == Stance.PROPAGANDA else 0.0
            mil_load    = float(np.mean(self._world.cluster_array()[:N, 3]))
            from .gravitas_actions import RESOURCE_COSTS
            return HierarchicalAction(
                stance=stance, weights=weights_pad,
                intensity=intensity, lt_bias=lt_bias,
                propaganda_load=prop_load, military_load=mil_load,
                resource_cost=RESOURCE_COSTS[stance] * intensity,
            )
        else:
            return decode_flat_action(
                int(action), self._world, self.params, self._rng
            )

    def _make_observation(self) -> NDArray[np.float32]:
        """Build padded observation vector (padded to max_N clusters)."""
        N      = self._cur_N
        max_N  = self._max_N
        world  = self._world
        params = self.params

        # Distorted cluster obs: (N, 3) → padded to (max_N, 3)
        prev_act_arr = self._prev_action.to_array()  # (max_N + 3,)

        raw_obs = distort_observation(world, params, self._rng, prev_act_arr)
        # raw_obs layout: [N*3, 6, N, action_dim, 1]
        # We need to pad N-dependent slices to max_N

        cluster_part = raw_obs[:N * 3]          # (N*3,)
        global_part  = raw_obs[N*3 : N*3+6]     # (6,)
        bias_part    = raw_obs[N*3+6 : N*3+6+N] # (N,)
        action_part  = raw_obs[N*3+6+N : N*3+6+N+len(prev_act_arr)]
        step_part    = raw_obs[-1:]              # (1,)

        # Pad cluster and bias to max_N
        cluster_padded = np.zeros(max_N * 3, dtype=np.float32)
        cluster_padded[:N*3] = cluster_part

        bias_padded = np.zeros(max_N, dtype=np.float32)
        bias_padded[:N] = bias_part

        # Shock rate as extra global signal
        shock_rate_arr = np.array(
            [float(world.shock_rate)], dtype=np.float32
        )

        obs = np.concatenate([
            cluster_padded,  # max_N * 3
            global_part,     # 6
            bias_padded,     # max_N
            shock_rate_arr,  # 1
            action_part,     # action_dim
            step_part,       # 1
        ]).astype(np.float32)

        return obs

    def _check_termination(
        self, world: GravitasWorld
    ) -> Tuple[bool, Optional[str]]:
        """Check all collapse conditions. Returns (terminated, cause_str)."""
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

    def _make_info(
        self,
        reward: Optional[RewardBreakdown] = None,
        shock: Optional[Dict[str, Any]] = None,
        cause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the info dict returned from step/reset."""
        world = self._world
        g     = world.global_state
        c     = world.cluster_array()
        N     = self._cur_N

        info: Dict[str, Any] = {
            # True state signals (for analysis, not used by policy)
            "true_mean_sigma":    float(np.mean(c[:N, 0])),
            "true_mean_hazard":   float(np.mean(c[:N, 1])),
            "true_mean_trust":    float(np.mean(c[:N, 4])),
            "true_mean_polar":    float(np.mean(c[:N, 5])),
            "true_mean_military": float(np.mean(c[:N, 3])),
            # Global
            "exhaustion":         g.exhaustion,
            "fragmentation":      g.fragmentation,
            "polarization":       g.polarization,
            "coherence":          g.coherence,
            "military_str":       g.military_str,
            "trust":              g.trust,
            "shock_rate":         world.shock_rate,
            # Episode
            "step":               g.step,
            "n_clusters":         N,
            "n_shocks":           len(self._shock_log),
            "collapse_cause":     cause,
            # Derived signals
            "military_load":      float(np.mean(c[:N, 3])),
            "trust_delta":        float(
                g.trust - (self._prev_world.global_state.trust if self._prev_world else g.trust)
            ),
            "media_bias_mean":    float(np.mean(np.abs(world.media_bias[:N]))),
        }

        if reward is not None:
            info.update(reward.to_dict())
        if shock:
            info["last_shock"] = shock

        return info

    # ──────────────────────────────────────────────────────────────────────── #
    # Render                                                                    #
    # ──────────────────────────────────────────────────────────────────────── #

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi" or self._world is None:
            return None

        world = self._world
        g     = world.global_state
        c     = world.cluster_array()
        N     = self._cur_N
        lines = [
            f"╔═══════════════════════ GRAVITAS ══════════════════════╗",
            f"║  Step {g.step:>4}  │  N={N}  │  Shocks={len(self._shock_log):>3}        ║",
            f"╠═══════════════════════════════════════════════════════╣",
            f"║  Exhaustion   {g.exhaustion:.3f}  │  Fragmentation {g.fragmentation:.3f}   ║",
            f"║  Polarization {g.polarization:.3f}  │  Coherence     {g.coherence:.3f}   ║",
            f"║  Military_str {g.military_str:.3f}  │  Trust         {g.trust:.3f}   ║",
            f"║  Shock_rate   {world.shock_rate:.4f}                            ║",
            f"╠═══════════════════════════════════════════════════════╣",
            f"║  Cluster  σ      h      r      m      τ      p        ║",
        ]
        for i in range(N):
            row = c[i]
            lines.append(
                f"║   [{i:>2}]  "
                f"{row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}  "
                f"{row[3]:.3f}  {row[4]:.3f}  {row[5]:.3f}  ║"
            )
        lines.append(f"╚═══════════════════════════════════════════════════════╝")
        output = "\n".join(lines)
        print(output)
        return output

    # ──────────────────────────────────────────────────────────────────────── #
    # Properties for external analysis                                         #
    # ──────────────────────────────────────────────────────────────────────── #

    @property
    def world(self) -> Optional[GravitasWorld]:
        return self._world

    @property
    def shock_log(self) -> List[Dict[str, Any]]:
        return list(self._shock_log)

    @property
    def reward_log(self) -> List[RewardBreakdown]:
        return list(self._reward_log)

    @property
    def collapse_cause(self) -> Optional[str]:
        return self._collapse_cause
