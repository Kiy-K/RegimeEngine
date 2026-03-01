"""
Population — Per-cluster population dynamics for GRAVITAS.

Population P_i ∈ [0, 1] (normalized carrying capacity):
  P_i = 1.0  →  cluster at full sustainable population
  P_i = 0.0  →  depopulated (displacement, death, famine)

Growth model (logistic with state-dependent rates):
  dP_i/dt =  r_pop  · P_i · (1 - P_i) · σ_i · r_i        # logistic growth
           - μ_pop  · h_i · P_i                            # hazard mortality
           - ν_pop  · m_i · (1 - σ_i) · P_i               # military depopulation
           - γ_pol  · p_i · P_i                            # polarization → emigration
           + δ_ally · Σⱼ max(A_ij,0) · (P_j - P_i) * r_j # allied migration inflow

Population feedback into cluster dynamics (applied after RK4):
  σ_i  +=  pop_sigma_lift * (P_i - 0.5)                   # populated → more stable
  r_i  +=  pop_resource_boost * P_i * (1 - r_i) * 0.01    # labour generates resources
  τ_i  +=  pop_trust_boost * P_i * (1 - h_i/5) * 0.01     # stable pop builds institutions
  h_i  +=  pop_hazard_cost * (1 - P_i) * 0.05             # depopulation → instability signal

Global feedback:
  Φ (fragmentation) +=  pop_frag_cost * mean(1 - P_i)
  Π (polarization)  +=  pop_pol_cost  * std(P_i)          # unequal pop → systemic tension
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams


# ─────────────────────────────────────────────────────────────────────────── #
# Population ODE derivatives                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def population_derivatives(
    population: NDArray[np.float64],      # (N,) ∈ [0,1]
    c_arr: NDArray[np.float64],           # (N, 6): [σ, h, r, m, τ, p]
    alliance: Optional[NDArray[np.float64]],  # (N, N) or None
    params: GravitasParams,
    N: int,
) -> NDArray[np.float64]:
    """
    Compute dP/dt for each cluster.  Returns (N,).
    """
    P = population[:N]
    sigma    = c_arr[:, 0]
    hazard   = c_arr[:, 1]
    resource = c_arr[:, 2]
    military = c_arr[:, 3]
    polar    = c_arr[:, 5]

    # ── Logistic growth: stable + resourced → population grows ─────────── #
    growth = params.r_pop * P * (1.0 - P) * sigma * resource

    # ── Hazard mortality ────────────────────────────────────────────────── #
    mortality = params.mu_pop * (hazard / 5.0) * P   # normalize hazard to [0,1]

    # ── Military depopulation (occupation, collateral damage) ───────────── #
    military_loss = params.nu_pop * military * (1.0 - sigma) * P

    # ── Polarization → emigration ───────────────────────────────────────── #
    emigration = params.gamma_pop * polar * P

    # ── Allied migration inflow ─────────────────────────────────────────── #
    inflow = np.zeros(N, dtype=np.float64)
    if alliance is not None:
        pos = np.maximum(alliance[:N, :N], 0.0)
        # Refugees flow from low-population to high-population allied clusters,
        # scaled by destination resource (can it absorb them?)
        pop_diff = P[:, None] - P[None, :]   # (N, N): positive if others have more
        inflow = params.delta_pop_migration * np.sum(
            pos * pop_diff * resource[:, None], axis=1
        )

    d_pop = growth - mortality - military_loss - emigration + inflow
    return np.clip(d_pop, -P, 1.0 - P)


# ─────────────────────────────────────────────────────────────────────────── #
# Population feedback into cluster state (post-RK4 impulse)                  #
# ─────────────────────────────────────────────────────────────────────────── #

def population_cluster_feedback(
    population: NDArray[np.float64],   # (N,)
    c_arr: NDArray[np.float64],        # (N, 6)
    params: GravitasParams,
    N: int,
    dt: float,
) -> NDArray[np.float64]:
    """
    Small per-step impulse on cluster variables from population level.
    Returns delta (N, 6) to add to cluster array.
    """
    P = population[:N]
    hazard   = c_arr[:, 1]

    delta = np.zeros_like(c_arr)
    # Populated clusters are more stable
    delta[:, 0] = params.pop_sigma_lift * (P - 0.5) * dt
    # Labour generates small resource recovery
    delta[:, 2] = params.pop_resource_boost * P * (1.0 - c_arr[:, 2]) * dt
    # Stable population builds institutional trust
    delta[:, 4] = params.pop_trust_boost * P * (1.0 - hazard / 5.0) * dt
    # Depopulated zones signal instability
    delta[:, 1] = params.pop_hazard_cost * (1.0 - P) * dt

    return delta


# ─────────────────────────────────────────────────────────────────────────── #
# Population step (Euler, called after RK4)                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def step_population(
    population: NDArray[np.float64],          # (max_N,)
    c_arr: NDArray[np.float64],               # (N, 6)
    alliance: Optional[NDArray[np.float64]],  # (max_N, max_N) or None
    params: GravitasParams,
    N: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Advance population by one Euler step.  Returns updated (max_N,) array.
    """
    d_pop = population_derivatives(population, c_arr, alliance, params, N)

    new_pop = population.copy()
    new_pop[:N] = np.clip(population[:N] + params.dt * d_pop, 0.0, 1.0)

    # Small stochastic noise
    if rng is not None:
        noise = 0.005 * np.sqrt(params.dt) * rng.standard_normal(N)
        new_pop[:N] = np.clip(new_pop[:N] + noise, 0.0, 1.0)

    return new_pop


# ─────────────────────────────────────────────────────────────────────────── #
# Initializer                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def initialize_population(
    N: int,
    max_N: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Initialize per-cluster population.  Returns (max_N,) array.
    Active clusters [0:N] drawn from U[0.45, 0.85].
    """
    pop = np.zeros(max_N, dtype=np.float64)
    pop[:N] = rng.uniform(0.45, 0.85, N)
    return pop
