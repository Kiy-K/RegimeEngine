"""
GravitasDynamics — Transition kernel for GRAVITAS.

Contains all ODE derivative functions and the RK4 integrator.
Every interaction is non-linear. Every variable has delayed consequences.

Key design principles:
  - No variable evolves independently
  - Military effects are split: immediate (in apply_action) vs accumulated (in ODEs)
  - Cascade hazard uses conflict topology c_ij, not proximity A_ij
  - Exhaustion gates all recovery terms (high exhaustion → everything freezes)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .gravitas_params import GravitasParams
from .gravitas_state import ClusterState, GlobalState, GravitasWorld, N_CLUSTER_VARS


# ─────────────────────────────────────────────────────────────────────────── #
# Hazard index (algebraic — re-derived each step, not integrated)             #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_hazard(
    sigma: NDArray[np.float64],
    trust: NDArray[np.float64],
    polar: NDArray[np.float64],
    conflict: NDArray[np.float64],
    sys_pol: float,
    params: GravitasParams,
) -> NDArray[np.float64]:
    """
    hᵢ = γ₁·(1-σᵢ)^κ_h · pᵢ^κ_p
       + γ₂·(1-τᵢ)·Π
       + γ₃·Σⱼ cᵢⱼ·hⱼ·(1-σⱼ)     [cascade; solved iteratively]

    We approximate the cascade term in one forward pass (sufficient for dt=0.01).
    """
    # Local hazard (no cascade yet)
    h_local = (
        params.gamma_h1 * np.power(np.clip(1.0 - sigma, 0.0, 1.0), params.kappa_h)
                        * np.power(polar, params.kappa_p)
        + params.gamma_h2 * (1.0 - trust) * sys_pol
    )
    # One-step cascade approximation
    cascade = params.gamma_h3 * (conflict @ (h_local * (1.0 - sigma)))
    return np.clip(h_local + cascade, 0.0, 5.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Cluster-level ODE derivatives  d(cluster)/dt                                #
# ─────────────────────────────────────────────────────────────────────────── #

def cluster_derivatives(
    arr: NDArray[np.float64],      # (N, 6): [σ, h, r, m, τ, p]
    hazard: NDArray[np.float64],   # (N,)  pre-computed
    adjacency: NDArray[np.float64],# (N, N)
    g: GlobalState,
    params: GravitasParams,
) -> NDArray[np.float64]:
    """
    Compute d(cluster_array)/dt.  Returns (N, 6).

    Index mapping: 0=σ, 1=h(unused/overwritten), 2=r, 3=m, 4=τ, 5=p
    """
    N = arr.shape[0]
    sigma    = arr[:, 0]
    resource = arr[:, 2]
    military = arr[:, 3]
    trust    = arr[:, 4]
    polar    = arr[:, 5]

    E   = g.exhaustion
    Pi  = g.polarization
    Phi = g.fragmentation
    act = 1.0 - E     # activity gate: high exhaustion freezes everything

    # ── dσ/dt: stability ────────────────────────────────────────────────── #
    # Spatial diffusion of stability across proximity edges
    deg    = np.sum(adjacency, axis=1)
    lap_s  = (adjacency @ sigma) - sigma * deg
    d_sigma = (
        params.alpha_sigma * act * (
            (1.0 - polar) * (1.0 - Phi) - params.beta_sigma * hazard * sigma
        )
        + params.nu_sigma * lap_s
    )
    d_sigma = np.clip(d_sigma, -sigma, 1.0 - sigma)

    # ── dr/dt: resource ─────────────────────────────────────────────────── #
    d_resource = (
        params.alpha_res * (1.0 - resource)            # natural recovery
        - params.hazard_res_cost * hazard * resource    # hazard drain
    ) * act
    d_resource = np.clip(d_resource, -resource, 1.0 - resource)

    # ── dm/dt: military presence decay ──────────────────────────────────── #
    # Military presence decays naturally (requires active upkeep)
    d_military = -params.military_decay * military
    d_military = np.clip(d_military, -military, 1.0 - military)

    # ── dτ/dt: institutional trust ──────────────────────────────────────── #
    # Reform builds trust; military in unstable zones destroys it; deprivation erodes it
    d_trust = (
        - params.military_tau_cost * military * (1.0 - sigma)
        - params.deprivation_tau_cost * (1.0 - resource) * hazard
        - params.tau_decay * trust
    ) * act
    # Reform action adds to trust (handled in apply_action, not here)
    d_trust = np.clip(d_trust, -trust, 1.0 - trust)

    # ── dp/dt: local polarization ────────────────────────────────────────── #
    d_polar = (
        params.alpha_pol * Phi * (1.0 - trust)          # fragmentation drives it
        + params.media_pol_coeff * np.abs(arr[:, 0] * 0)  # placeholder: bias injected in env
        - params.beta_pol * trust * (1.0 - polar)       # trust depolarizes
    )
    d_polar = np.clip(d_polar, -polar, 1.0 - polar)

    deriv = np.zeros_like(arr)
    deriv[:, 0] = d_sigma
    deriv[:, 1] = 0.0       # hazard is re-derived algebraically, not integrated
    deriv[:, 2] = d_resource
    deriv[:, 3] = d_military
    deriv[:, 4] = d_trust
    deriv[:, 5] = d_polar
    return deriv


# ─────────────────────────────────────────────────────────────────────────── #
# Global ODE derivatives  d(global)/dt                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def global_derivatives(
    g: GlobalState,
    hazard: NDArray[np.float64],
    military: NDArray[np.float64],
    trust: NDArray[np.float64],
    params: GravitasParams,
    military_load: float,     # mean(mᵢ)
    propaganda_load: float,   # agent's propaganda intensity this step
) -> NDArray[np.float64]:
    """
    Compute d(global_array)/dt.  Returns (6,): [E, Φ, Π, Ψ, M, T]
    """
    E   = g.exhaustion
    Phi = g.fragmentation
    Pi  = g.polarization
    Psi = g.coherence
    M   = g.military_str
    T   = g.trust

    mean_h   = float(np.mean(hazard))
    mean_m   = military_load
    mean_tau = float(np.mean(trust))

    # ── dE/dt: exhaustion ────────────────────────────────────────────────── #
    accumulation = Pi * (1.0 - Psi) * (1.0 - E) + params.military_exh_coeff * mean_m * M
    recovery     = params.beta_exh * (1.0 - Pi) * Psi * E
    d_E = params.alpha_exh * (accumulation - recovery)
    d_E = float(np.clip(d_E, -E, 1.0 - E))

    # ── dΦ/dt: fragmentation probability ─────────────────────────────────── #
    d_Phi = (
        params.alpha_phi * mean_h * (1.0 - mean_tau)
        + params.military_phi_coeff * mean_m ** 2 * E
        - params.beta_phi * mean_tau * (1.0 - Phi)
    )
    d_Phi = float(np.clip(d_Phi, -Phi, 1.0 - Phi))

    # ── dΠ/dt: systemic polarization ─────────────────────────────────────── #
    d_Pi = (
        params.alpha_pol * Phi * (1.0 - T)
        + params.propaganda_pol_coeff * propaganda_load * (1.0 - T)  # hidden cost
        - params.beta_pol * T * (1.0 - Pi)
    )
    d_Pi = float(np.clip(d_Pi, -Pi, 1.0 - Pi))

    # ── dΨ/dt: information coherence ─────────────────────────────────────── #
    d_Psi = (
        params.psi_recovery * (1.0 - Psi)
        - params.psi_propaganda_cost * propaganda_load
    )
    d_Psi = float(np.clip(d_Psi, -Psi, 1.0 - Psi))

    # ── dM/dt: military strength ──────────────────────────────────────────── #
    # Military capacity recovers slowly; heavy use depletes it
    d_M = 0.01 * (1.0 - M) - 0.05 * mean_m * M
    d_M = float(np.clip(d_M, -M, 1.0 - M))

    # ── dT/dt: aggregate institutional trust ──────────────────────────────── #
    d_T = (
        params.alpha_tau * mean_tau            # aggregate from cluster level
        - params.tau_decay * T
        - params.military_tau_cost * mean_m * (1.0 - mean_tau)
    )
    d_T = float(np.clip(d_T, -T, 1.0 - T))

    return np.array([d_E, d_Phi, d_Pi, d_Psi, d_M, d_T], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────── #
# RK4 integrator                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _eval_derivatives(
    c_arr: NDArray[np.float64],
    g: GlobalState,
    hazard: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    params: GravitasParams,
    military_load: float,
    propaganda_load: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate (d_cluster, d_global) at a given state point."""
    trust = c_arr[:, 4]
    g_deriv = global_derivatives(g, hazard, c_arr[:, 3], trust, params,
                                  military_load, propaganda_load)
    c_deriv = cluster_derivatives(c_arr, hazard, adjacency, g, params)
    return c_deriv, g_deriv


def _arr_to_global(arr: NDArray[np.float64], step: int) -> GlobalState:
    return GlobalState.from_array(arr, step=step)


def rk4_step(
    world: GravitasWorld,
    params: GravitasParams,
    military_load: float,
    propaganda_load: float,
    sigma_noise: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> GravitasWorld:
    """
    Advance world by one RK4 step (dt = params.dt).

    Hazard is re-derived at each sub-step (algebraic, not integrated).
    SDE noise is applied to σ and r after the deterministic RK4.
    """
    from typing import Optional  # local import to avoid circular at module level
    dt     = params.dt
    half   = dt * 0.5
    step   = world.global_state.step

    c0     = world.cluster_array()
    g0_arr = world.global_state.to_array()
    A      = world.adjacency
    C      = world.conflict

    # ─── k1 ─────────────────────────────────────────────────────────────── #
    h0 = compute_hazard(c0[:,0], c0[:,4], c0[:,5], C,
                        world.global_state.polarization, params)
    dc1, dg1 = _eval_derivatives(c0, world.global_state, h0, A, params,
                                  military_load, propaganda_load)

    # ─── k2 ─────────────────────────────────────────────────────────────── #
    c2     = np.clip(c0 + half * dc1, 0.0, 1.0)
    c2[:,1]= 0.0  # hazard re-derived
    g2     = _arr_to_global(np.clip(g0_arr + half * dg1, 0.0, 1.0), step)
    h2     = compute_hazard(c2[:,0], c2[:,4], c2[:,5], C, g2.polarization, params)
    dc2, dg2 = _eval_derivatives(c2, g2, h2, A, params,
                                  military_load, propaganda_load)

    # ─── k3 ─────────────────────────────────────────────────────────────── #
    c3     = np.clip(c0 + half * dc2, 0.0, 1.0)
    c3[:,1]= 0.0
    g3     = _arr_to_global(np.clip(g0_arr + half * dg2, 0.0, 1.0), step)
    h3     = compute_hazard(c3[:,0], c3[:,4], c3[:,5], C, g3.polarization, params)
    dc3, dg3 = _eval_derivatives(c3, g3, h3, A, params,
                                  military_load, propaganda_load)

    # ─── k4 ─────────────────────────────────────────────────────────────── #
    c4     = np.clip(c0 + dt * dc3, 0.0, 1.0)
    c4[:,1]= 0.0
    g4     = _arr_to_global(np.clip(g0_arr + dt * dg3, 0.0, 1.0), step)
    h4     = compute_hazard(c4[:,0], c4[:,4], c4[:,5], C, g4.polarization, params)
    dc4, dg4 = _eval_derivatives(c4, g4, h4, A, params,
                                  military_load, propaganda_load)

    # ─── Weighted sum ────────────────────────────────────────────────────── #
    factor  = dt / 6.0
    c_new   = np.clip(c0 + factor * (dc1 + 2*dc2 + 2*dc3 + dc4), 0.0, 1.0)
    g_new_a = np.clip(g0_arr + factor * (dg1 + 2*dg2 + 2*dg3 + dg4), 0.0, 1.0)

    # ─── SDE noise on σ (stability) and r (resource) ────────────────────── #
    if sigma_noise > 0.0 and rng is not None:
        noise_scale = sigma_noise * np.sqrt(dt)
        c_new[:, 0] += noise_scale * rng.standard_normal(c_new.shape[0])
        c_new[:, 2] += noise_scale * 0.5 * rng.standard_normal(c_new.shape[0])
        c_new = np.clip(c_new, 0.0, 1.0)

    # ─── Re-derive hazard at new state ──────────────────────────────────── #
    g_new  = _arr_to_global(g_new_a, step)
    h_new  = compute_hazard(c_new[:,0], c_new[:,4], c_new[:,5], C,
                             g_new.polarization, params)
    c_new[:, 1] = h_new   # store in array (column 1 = hazard)

    # ─── Reconstruct cluster states ─────────────────────────────────────── #
    new_clusters = [
        ClusterState.from_array(c_new[i]) for i in range(c_new.shape[0])
    ]
    new_global = GlobalState.from_array(g_new_a, step=step)

    return world.copy_with_clusters(new_clusters).copy_with_global(new_global)


# needed by rk4_step
from typing import Optional  # noqa: E402  (placed here to avoid circular at top)
