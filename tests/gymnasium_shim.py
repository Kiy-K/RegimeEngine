"""
gymnasium_shim.py — Minimal Gymnasium-compatible shim for environments
that run in contexts where gymnasium is not installed.

Provides:
  gym.Env, gym.spaces.Box, gym.spaces.Discrete, gym.spaces.Dict
with enough API surface to let GravitasEnv run and be tested.

Usage:
  This module is auto-imported by gravitas_env.py if gymnasium is absent.
  For actual RL training (SB3 / other frameworks), install gymnasium>=0.29.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class Space:
    """Base space."""
    def sample(self) -> Any:
        raise NotImplementedError


class Box(Space):
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low   = np.full(shape, low, dtype=dtype)
        self.high  = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> NDArray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class Discrete(Space):
    def __init__(self, n: int):
        self.n = n
        self.shape = ()

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))


class Dict(Space):
    def __init__(self, spaces_dict: dict):
        self.spaces = spaces_dict

    def sample(self) -> dict:
        return {k: v.sample() for k, v in self.spaces.items()}


class Env:
    """Minimal Env base compatible with GravitasEnv API."""
    observation_space: Space
    action_space:      Space
    metadata:          dict = {}

    def reset(self, seed=None, options=None) -> Tuple[Any, dict]:
        raise NotImplementedError

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        raise NotImplementedError

    def render(self) -> Optional[str]:
        return None

    def close(self) -> None:
        pass


# Expose as gymnasium-compatible namespace
class _Spaces:
    Box = Box
    Discrete = Discrete
    Dict = Dict


spaces = _Spaces()
