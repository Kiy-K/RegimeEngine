"""
MediaBias — Perception distortion layer for GRAVITAS.

The agent never observes true state. It observes:
    obs_t = f(true_state, media_bias_t) + structured_noise

Media bias:
  - Drifts autonomously (rho_B memory)
  - Reacts to agent's propaganda actions
  - Amplified by incoherence and shocks
  - Partially observable: agent sees a noisy estimate of B_t

Three emergent bias regimes (not scripted, arising from dynamics):
  1. Optimism bias (B > 0): agent underestimates risk → delayed intervention
  2. Panic bias   (B < 0): agent overestimates risk → over-militarizes
  3. Polarized    (B bimodal): systematic blind spots in specific clusters
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import GravitasWorld


# ─────────────────────────────────────────────────────────────────────────── #
# Bias update                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def update_media_bias(
    world: GravitasWorld,
    propaganda_weights: NDArray[np.float64],  # (N,) allocation weights
    propaganda_intensity: float,              # θ ∈ [0,1]
    shock_occurred: bool,
    shock_cluster: int,                       # index of affected cluster (-1 if global)
    params: GravitasParams,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    B_{t+1,i} = ρ_B · Bᵢ
              + φ_prop · propaganda_effect_i
              + φ_incoherence · (1 - Ψ) · ξᵢ
              + φ_shock · shock_magnitude · ζᵢ

    The propaganda effect is directional: the agent tries to make things look
    better (B → positive), but high polarization reduces precision.

    Returns new bias vector clipped to [-β_max, β_max].
    """
    N       = world.n_clusters
    B       = world.media_bias.copy()
    Pi      = world.global_state.polarization
    Psi     = world.global_state.coherence
    beta_max = params.beta_max_bias
    autonomy = params.media_autonomy

    # ── 1. Auto-regression (memory) ──────────────────────────────────────── #
    B_new = params.rho_bias * B

    # ── 2. Propaganda effect ─────────────────────────────────────────────── #
    # Agent pushes bias toward positive (hiding risk).
    # High polarization → noise drowns signal → propaganda less effective.
    # High media autonomy → agent has less control.
    if propaganda_intensity > 0.0:
        precision       = (1.0 - Pi) * (1.0 - autonomy)
        noise_prop      = rng.standard_normal(N) * Pi * 0.2
        propaganda_dir  = propaganda_weights * propaganda_intensity * precision
        B_new += params.phi_bias_prop * (propaganda_dir + noise_prop)

    # ── 3. Incoherence-driven noise ──────────────────────────────────────── #
    xi    = rng.standard_normal(N)
    B_new += params.phi_bias_incoherence * (1.0 - Psi) * xi

    # ── 4. Shock amplification ───────────────────────────────────────────── #
    if shock_occurred:
        zeta  = rng.standard_normal(N)
        if shock_cluster >= 0 and shock_cluster < N:
            # Shock cluster gets larger bias jump; neighbours get smaller
            bump               = np.zeros(N)
            bump[shock_cluster] = 1.0
            # Propagate to conflict neighbours
            bump += 0.3 * world.conflict[shock_cluster]
            bump = np.clip(bump, 0.0, 1.0)
        else:
            bump = np.ones(N)
        B_new += params.phi_bias_shock * bump * np.abs(zeta)

    # ── 5. Clip to bounds ────────────────────────────────────────────────── #
    return np.clip(B_new, -beta_max, beta_max)


# ─────────────────────────────────────────────────────────────────────────── #
# Observation distortion                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def distort_observation(
    world: GravitasWorld,
    params: GravitasParams,
    rng: np.random.Generator,
    prev_action: NDArray[np.float64],
) -> NDArray[np.float32]:
    """
    Build the distorted observation vector the agent receives.

    Distortion model:
        σ̂ᵢ = σᵢ · (1 + Bᵢ) + εᵢ         (stability appears better/worse)
        ĥᵢ = hᵢ · (1 - Bᵢ) + εᵢ          (hazard appears lower/higher)
        r̂ᵢ = rᵢ + εᵢ                      (resources: less distorted)

    Global variables are also distorted by coherence-scaled noise.
    The bias estimate B̂ is a noisy version of the true B.

    Observation layout:
        [σ̂, ĥ, r̂ per cluster   (N×3)]
        [Ê, Φ̂, Π̂, Ψ̂, M̂, T̂   (6)]
        [B̂₁…B̂ₙ                (N)]
        [prev_action             (action_dim)]
        [step_frac               (1)]

    Total dim = N*3 + 6 + N + action_dim + 1 = 4N + 7 + action_dim
    """
    N   = world.n_clusters
    B   = world.media_bias
    Psi = world.global_state.coherence
    step = world.global_state.step

    # ── Cluster observations ─────────────────────────────────────────────── #
    c_arr = world.cluster_array()   # (N, 6)
    sigma    = c_arr[:, 0]
    hazard   = c_arr[:, 1]
    resource = c_arr[:, 2]

    noise_scale = params.sigma_obs_base + params.sigma_obs_base * (1.0 - Psi)
    eps = rng.standard_normal((N, 3)) * noise_scale

    sigma_obs    = np.clip(sigma * (1.0 + B) + eps[:, 0], 0.0, 1.5)
    hazard_obs   = np.clip(hazard * (1.0 - B) + np.abs(eps[:, 1]), 0.0, 5.0)
    resource_obs = np.clip(resource + eps[:, 2], 0.0, 1.0)

    cluster_obs = np.stack([sigma_obs, hazard_obs, resource_obs], axis=1)  # (N,3)

    # ── Global observations ──────────────────────────────────────────────── #
    g_arr  = world.global_state.to_array()   # (6,)
    g_noise = rng.standard_normal(6) * noise_scale * 0.5
    global_obs = np.clip(g_arr + g_noise, 0.0, 1.0)

    # ── Bias estimate (agent's noisy view of own information environment) ── #
    bias_noise = rng.standard_normal(N) * noise_scale * (1.0 - Psi) * 2.0
    bias_est   = np.clip(B + bias_noise, -params.beta_max_bias, params.beta_max_bias)

    # ── Normalise hazard obs to [0,1] for network input ──────────────────── #
    hazard_obs_norm = np.clip(hazard_obs / 3.0, 0.0, 1.0)
    cluster_obs[:, 1] = hazard_obs_norm

    # ── Step fraction ─────────────────────────────────────────────────────── #
    step_frac = np.array([step / max(1, params.max_steps)], dtype=np.float64)

    # ── Concatenate ──────────────────────────────────────────────────────── #
    obs = np.concatenate([
        cluster_obs.flatten(),   # N×3
        global_obs,              # 6
        bias_est,                # N
        prev_action,             # action_dim
        step_frac,               # 1
    ]).astype(np.float32)

    return obs
