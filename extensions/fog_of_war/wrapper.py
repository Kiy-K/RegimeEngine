"""
FogOfWarWrapper — Observation distortion layer for GravitasEnv.

Adds structured Gaussian noise scaled by current polarization level,
simulating Clausewitz's "fog of war": the agent perceives reality through
a filter that degrades as systemic polarization increases.

Bugs fixed vs original:
  1. Now a proper gymnasium.Wrapper (calls super().__init__), not a manual duck-type.
  2. Removed reference to env.regime (doesn't exist on GravitasEnv).
  3. Removed reference to env.state (GravitasEnv exposes env.world, not env.state).
  4. reset() now correctly unpacks (obs, info) from the Gymnasium API.
  5. step() now correctly unpacks 5-tuple (obs, reward, terminated, truncated, info).
  6. _apply_bias() no longer clips to [0,1]; GravitasEnv obs legitimately spans [-2, 6].
  7. Polarization is read from env.world.global_state, not env.state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
except ImportError:
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import gymnasium_shim as gym  # type: ignore[no-redef]


class FogOfWarWrapper(gym.Wrapper):
    """
    Observation wrapper that adds polarization-scaled Gaussian noise.

    The noise magnitude grows with systemic polarization — when society is
    more polarized, information quality degrades. This forces the agent to
    learn to act under compounding uncertainty, not just fixed-level noise.

    Note: This wrapper adds noise ON TOP of the media bias already baked into
    GravitasEnv's observations. They serve different purposes:
      - GravitasEnv internal bias: structured, directional, per-cluster distortion
      - FogOfWarWrapper:           unstructured, isotropic chaos noise

    Args:
        env:        A GravitasEnv instance (or subclass).
        bias_scale: Base noise magnitude. Total noise std = bias_scale * (1 + Pi).
    """

    def __init__(self, env: gym.Env, bias_scale: float = 0.1) -> None:
        super().__init__(env)   # registers self.env, copies observation/action spaces
        self.bias_scale = bias_scale

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        # FIX 4: unpack (obs, info) — not just obs
        obs, info = self.env.reset(seed=seed, options=options)
        return self._apply_bias(obs), info

    def step(
        self,
        action: Any,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        # FIX 5: unpack 5-tuple — gymnasium returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_bias(obs), reward, terminated, truncated, info

    def _apply_bias(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Add Gaussian noise scaled by current systemic polarization.

        Noise std = bias_scale * (1 + Pi), Pi in [0,1].
        Result is clipped to observation_space bounds [-2, 6], NOT to [0,1].
        GravitasEnv hazard observations can legitimately reach ~6 before normalization.
        """
        polarization = self._get_polarization()
        noise_std    = self.bias_scale * (1.0 + polarization)
        noise        = np.random.normal(0.0, noise_std, size=obs.shape).astype(np.float32)
        obs_noisy    = obs + noise

        # FIX 6: clip to declared space bounds, not [0,1]
        low  = self.observation_space.low
        high = self.observation_space.high
        return np.clip(obs_noisy, low, high)

    def _get_polarization(self) -> float:
        """
        Read true systemic polarization from the underlying GravitasEnv.
        FIX 3: use env.world.global_state — env.state does not exist.
        """
        world = getattr(self.env, "world", None)
        if world is None:
            return 0.0
        gs = getattr(world, "global_state", None)
        if gs is None:
            return 0.0
        return float(getattr(gs, "polarization", 0.0))
