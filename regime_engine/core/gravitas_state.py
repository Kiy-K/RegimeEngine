"""
GravitasState — Immutable state containers for the GRAVITAS environment.

Three tiers:
  ClusterState   — per-cluster variables (one per cluster)
  GlobalState    — system-wide scalar variables
  GravitasWorld  — top-level container binding both tiers + topology

All fields are validated and clipped at construction.
The state is designed so that:
  - The agent never directly observes ClusterState
  - The agent observes ObservedState = distort(GravitasWorld, media_bias)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────── #
# Cluster-level state                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class ClusterState:
    """
    Per-cluster true state. None of these are directly observed.

    σ  — structural stability      [0,1]
    h  — hazard index              [0,∞)   (can exceed 1 before collapse clamp)
    r  — resource level            [0,1]
    m  — military presence         [0,1]
    τ  — institutional trust       [0,1]
    p  — local polarization        [0,1]
    """

    sigma: float    # stability
    hazard: float   # hazard index
    resource: float # resource level
    military: float # deployed military fraction
    trust: float    # institutional trust
    polar: float    # local polarization

    def __post_init__(self) -> None:
        for name, lo, hi in [
            ("sigma",    0.0, 1.0),
            ("hazard",   0.0, 5.0),  # allow > 1 before clamp
            ("resource", 0.0, 1.0),
            ("military", 0.0, 1.0),
            ("trust",    0.0, 1.0),
            ("polar",    0.0, 1.0),
        ]:
            v = getattr(self, name)
            if not (lo <= v <= hi):
                object.__setattr__(self, name, float(np.clip(v, lo, hi)))

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [self.sigma, self.hazard, self.resource,
             self.military, self.trust, self.polar],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "ClusterState":
        a = np.clip(arr, 0.0, 1.0)
        return cls(
            sigma=float(a[0]),
            hazard=float(np.clip(arr[1], 0.0, 5.0)),
            resource=float(a[2]),
            military=float(a[3]),
            trust=float(a[4]),
            polar=float(a[5]),
        )

    def copy_with(self, **kw: float) -> "ClusterState":
        vals = {
            "sigma": self.sigma, "hazard": self.hazard,
            "resource": self.resource, "military": self.military,
            "trust": self.trust, "polar": self.polar,
        }
        vals.update(kw)
        return ClusterState(**vals)

    def to_dict(self) -> Dict[str, float]:
        return {
            "sigma": self.sigma, "hazard": self.hazard,
            "resource": self.resource, "military": self.military,
            "trust": self.trust, "polar": self.polar,
        }

N_CLUSTER_VARS = 6  # σ, h, r, m, τ, p


# ─────────────────────────────────────────────────────────────────────────── #
# Global dynamic state                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class GlobalState:
    """
    System-wide scalar variables.

    E   — global exhaustion          [0,1]
    Phi — fragmentation probability  [0,1)
    Pi  — systemic polarization      [0,1]
    Psi — information coherence      [0,1]
    M   — total military strength    [0,1]
    T   — aggregate institutional trust  [0,1]
    """

    exhaustion:    float   # E
    fragmentation: float   # Φ
    polarization:  float   # Π
    coherence:     float   # Ψ
    military_str:  float   # M (global capacity)
    trust:         float   # T (aggregate)
    step:          int = 0

    def __post_init__(self) -> None:
        for name in ("exhaustion", "fragmentation", "polarization",
                     "coherence", "military_str", "trust"):
            v = getattr(self, name)
            clamped = float(np.clip(v, 0.0, 1.0))
            object.__setattr__(self, name, clamped)

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.exhaustion, self.fragmentation, self.polarization,
            self.coherence, self.military_str, self.trust,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64], step: int = 0) -> "GlobalState":
        a = np.clip(arr, 0.0, 1.0)
        return cls(
            exhaustion=float(a[0]), fragmentation=float(a[1]),
            polarization=float(a[2]), coherence=float(a[3]),
            military_str=float(a[4]), trust=float(a[5]),
            step=step,
        )

    def copy_with(self, **kw) -> "GlobalState":
        vals = {
            "exhaustion": self.exhaustion, "fragmentation": self.fragmentation,
            "polarization": self.polarization, "coherence": self.coherence,
            "military_str": self.military_str, "trust": self.trust,
            "step": self.step,
        }
        vals.update(kw)
        return GlobalState(**vals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exhaustion": self.exhaustion, "fragmentation": self.fragmentation,
            "polarization": self.polarization, "coherence": self.coherence,
            "military_str": self.military_str, "trust": self.trust,
            "step": self.step,
        }

N_GLOBAL_VARS = 6  # E, Φ, Π, Ψ, M, T


# ─────────────────────────────────────────────────────────────────────────── #
# Full world state                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class GravitasWorld:
    """
    Complete immutable world state.

    clusters     — tuple of ClusterState (length N)
    global_state — GlobalState scalars
    adjacency    — (N,N) proximity/trade weights (symmetric)
    conflict     — (N,N) conflict linkage c_ij (subset of adjacency)
    media_bias   — (N,) per-cluster distortion B_t ∈ [-β_max, β_max]
    shock_rate   — current Hawkes intensity λ(t)
    hawkes_sum   — running Σ exp(-β(t-t_k)) for Hawkes update
    """

    clusters:     Tuple[ClusterState, ...]
    global_state: GlobalState
    adjacency:    NDArray[np.float64]  # (N, N)
    conflict:     NDArray[np.float64]  # (N, N)
    media_bias:   NDArray[np.float64]  # (N,)
    shock_rate:   float
    hawkes_sum:   float

    def __post_init__(self) -> None:
        N = len(self.clusters)
        assert self.adjacency.shape == (N, N), \
            f"adjacency must be ({N},{N}), got {self.adjacency.shape}"
        assert self.conflict.shape == (N, N), \
            f"conflict must be ({N},{N}), got {self.conflict.shape}"
        assert self.media_bias.shape == (N,), \
            f"media_bias must be ({N},), got {self.media_bias.shape}"

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def cluster_array(self) -> NDArray[np.float64]:
        """(N, N_CLUSTER_VARS) array of all cluster states."""
        return np.array([c.to_array() for c in self.clusters], dtype=np.float64)

    def copy_with_clusters(self, clusters: List[ClusterState]) -> "GravitasWorld":
        return GravitasWorld(
            clusters=tuple(clusters),
            global_state=self.global_state,
            adjacency=self.adjacency,
            conflict=self.conflict,
            media_bias=self.media_bias,
            shock_rate=self.shock_rate,
            hawkes_sum=self.hawkes_sum,
        )

    def copy_with_global(self, g: GlobalState) -> "GravitasWorld":
        return GravitasWorld(
            clusters=self.clusters,
            global_state=g,
            adjacency=self.adjacency,
            conflict=self.conflict,
            media_bias=self.media_bias,
            shock_rate=self.shock_rate,
            hawkes_sum=self.hawkes_sum,
        )

    def copy_with_bias(self, bias: NDArray[np.float64]) -> "GravitasWorld":
        return GravitasWorld(
            clusters=self.clusters,
            global_state=self.global_state,
            adjacency=self.adjacency,
            conflict=self.conflict,
            media_bias=np.asarray(bias, dtype=np.float64),
            shock_rate=self.shock_rate,
            hawkes_sum=self.hawkes_sum,
        )

    def copy_with_shock(self, rate: float, h_sum: float) -> "GravitasWorld":
        return GravitasWorld(
            clusters=self.clusters,
            global_state=self.global_state,
            adjacency=self.adjacency,
            conflict=self.conflict,
            media_bias=self.media_bias,
            shock_rate=float(np.clip(rate, 0.0, 1.0)),
            hawkes_sum=float(h_sum),
        )

    def advance_step(self) -> "GravitasWorld":
        return self.copy_with_global(
            self.global_state.copy_with(step=self.global_state.step + 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "global": self.global_state.to_dict(),
            "media_bias": self.media_bias.tolist(),
            "shock_rate": self.shock_rate,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Observed state (what the agent sees)                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ObservedState:
    """
    Distorted view of the world as seen by the agent.

    cluster_obs  — (N, 3): distorted [σ̂, ĥ, r̂] per cluster
    global_obs   — (6,):   distorted global vars + own bias estimate
    bias_est     — (N,):   agent's noisy estimate of B_t
    prev_action  — (N+3,): previous action (always accurate)
    step_frac    — scalar: t / max_steps (time awareness)
    """

    cluster_obs: NDArray[np.float64]   # (N, 3)
    global_obs:  NDArray[np.float64]   # (6,)
    bias_est:    NDArray[np.float64]   # (N,)
    prev_action: NDArray[np.float64]   # (action_dim,)
    step_frac:   float

    def to_flat(self) -> NDArray[np.float32]:
        """Flatten everything into a single vector for the policy."""
        return np.concatenate([
            self.cluster_obs.flatten(),
            self.global_obs,
            self.bias_est,
            self.prev_action,
            np.array([self.step_frac], dtype=np.float64),
        ]).astype(np.float32)
