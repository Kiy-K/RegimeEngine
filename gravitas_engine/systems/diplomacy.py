"""
Diplomacy — Inter-cluster alliance dynamics for GRAVITAS.

Alliance matrix A ∈ ℝ^(N×N), symmetric, diagonal=0, values ∈ [-1, +1]:
  +1  fully allied  : shared stability, resources, trust
   0  neutral       : no diplomatic effect
  -1  openly hostile: trust erosion, polarization spread, hazard cascade

Effects wired into the cluster ODE:
  dσᵢ/dt  += ν_A   · Σⱼ max(A_ij, 0) · (σⱼ - σᵢ)   (allied σ diffusion)
  drᵢ/dt  += ν_R   · Σⱼ max(A_ij, 0) · (rⱼ - rᵢ)   (allied resource sharing)
  dτᵢ/dt  -= α_H   · Σⱼ max(-A_ij,0) · τᵢ           (hostile trust erosion)
  dpᵢ/dt  += α_H   · Σⱼ max(-A_ij,0) · (1 - pᵢ)     (hostile polarization spread)

Alliance decay (every step):
  A_ij  ← A_ij · (1 - decay)   (drift toward neutral)

DIPLOMACY action:
  Weights w select which bilateral links to strengthen.
  A_ij  += shift · wᵢ · wⱼ   (clamped to [-1, +1], symmetric)
  Immediate trust boost in allied clusters.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams


def alliance_cluster_derivatives(
    c_arr: NDArray[np.float64],      # (N, 6): [σ, h, r, m, τ, p]
    alliance: NDArray[np.float64],   # (N, N) ∈ [-1, +1]
    params: GravitasParams,
) -> NDArray[np.float64]:
    """
    Return per-cluster derivative contribution (N, 6) from diplomatic relations.
    Only columns 0 (σ), 2 (r), 4 (τ), 5 (p) are non-zero.
    """
    N = c_arr.shape[0]
    deriv = np.zeros_like(c_arr)

    sigma    = c_arr[:, 0]
    resource = c_arr[:, 2]
    trust    = c_arr[:, 4]
    polar    = c_arr[:, 5]

    pos = np.maximum(alliance, 0.0)   # allied weights  (≥ 0)
    neg = np.maximum(-alliance, 0.0)  # hostile weights (≥ 0)

    # ── Allied stability diffusion ──────────────────────────────────────── #
    # Each cluster gains σ from stronger allies, loses to weaker ones
    deriv[:, 0] = params.nu_alliance * (pos @ sigma - pos.sum(axis=1) * sigma)

    # ── Allied resource sharing ──────────────────────────────────────────── #
    deriv[:, 2] = params.nu_res_alliance * (pos @ resource - pos.sum(axis=1) * resource)

    # ── Hostile trust erosion ────────────────────────────────────────────── #
    deriv[:, 4] = -params.alpha_hostility * neg.sum(axis=1) * trust

    # ── Hostile polarization spread ──────────────────────────────────────── #
    # Enemies fuel local polarization
    deriv[:, 5] = params.alpha_hostility * (neg @ polar) * (1.0 - polar)

    # Batch-clip to physical bounds (derivative cannot push state outside [0,1])
    np.clip(deriv, -c_arr, 1.0 - c_arr, out=deriv)

    return deriv


def decay_alliance(
    alliance: NDArray[np.float64],
    params: GravitasParams,
) -> NDArray[np.float64]:
    """
    Decay alliance values toward neutral (0) each step.
    Returns updated (N, N) alliance matrix.
    """
    return alliance * (1.0 - params.alliance_decay)


def apply_diplomacy_action(
    alliance: NDArray[np.float64],   # current (N, N)
    weights: NDArray[np.float64],    # (N,) cluster weights from action
    intensity: float,                # action intensity θ
    params: GravitasParams,
    N: int,
) -> NDArray[np.float64]:
    """
    Strengthen bilateral alliances for clusters selected by weights.

    The shift is proportional to wᵢ · wⱼ (both parties must be engaged).
    Only the active-cluster block is modified.
    Returns updated (N, N) alliance matrix.
    """
    w = weights[:N]
    shift = params.diplomacy_shift * intensity

    # Outer product: how much to shift each pair
    delta = np.outer(w, w) * shift
    np.fill_diagonal(delta, 0.0)   # no self-alliance

    new_alliance = np.clip(alliance[:N, :N] + delta, -1.0, 1.0)

    # Write back into full matrix (zero-padded outside active block)
    result = alliance.copy()
    result[:N, :N] = new_alliance
    return result


def build_alliance(
    N: int,
    adjacency: NDArray[np.float64],
    rng: np.random.Generator,
    max_N: int,
) -> NDArray[np.float64]:
    """
    Build initial alliance matrix for a new episode.

    Adjacent clusters start with a small positive bias (neighbours tend to
    cooperate on stability); all others start near neutral with tiny noise.
    Matrix is symmetric with zero diagonal.

    Returns: (max_N, max_N) array, active block is [:N, :N].
    """
    alliance = np.zeros((max_N, max_N), dtype=np.float64)

    # For active clusters, seed from adjacency with small positive bias
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if adjacency[i, j] > 0.05:
                # Neighbouring nations start slightly cooperative
                alliance[i, j] = float(rng.uniform(0.05, 0.20))
            else:
                # Non-adjacent: small random perturbation (could be neutral or mildly hostile)
                alliance[i, j] = float(rng.uniform(-0.10, 0.05))

    # Symmetrise
    for i in range(N):
        for j in range(i + 1, N):
            avg = (alliance[i, j] + alliance[j, i]) * 0.5
            alliance[i, j] = avg
            alliance[j, i] = avg

    return alliance
