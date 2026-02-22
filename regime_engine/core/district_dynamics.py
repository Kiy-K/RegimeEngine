"""
District-level SDE dynamics (Phase 3).

Fully integrated district derivatives: local terms, unrest diffusion,
capital flow, and policy effect. All variables kept in [0,1] via decay
and derivative clipping.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .admin_lag import effective_policy_at_districts
from .hierarchical_state import N_DISTRICT_VARS, array_to_districts, districts_to_array
from .parameters import SystemParameters
from .state import RegimeState

# Index into district state array
I_GDP, I_UNREST, I_ADMIN, I_FRAG, I_IMPL, I_MEM = range(N_DISTRICT_VARS)


def unrest_diffusion_term(
    local_unrest: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    nu: float,
) -> NDArray[np.float64]:
    """ν Σ_j A_ij (u_j - u_i)."""
    # Laplacian: (A @ u) - u * deg;  here we want sum_j A_ij (u_j - u_i) = (A @ u)_i - u_i * (A @ 1)_i
    deg = np.sum(adjacency, axis=1)
    lap_u = (adjacency @ local_unrest) - local_unrest * deg
    return nu * lap_u


def capital_flow_term(
    local_gdp: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    mu: float,
) -> NDArray[np.float64]:
    """μ Σ_j A_ij (g_j - g_i)."""
    deg = np.sum(adjacency, axis=1)
    lap_g = (adjacency @ local_gdp) - local_gdp * deg
    return mu * lap_g


def compute_district_derivatives(
    district_array: NDArray[np.float64],
    pipeline_buffers: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    regime_gdp: float,
    regime_instability: float,
    regime_fragmentation: float,
    regime_exhaustion: float,
    params: SystemParameters,
) -> NDArray[np.float64]:
    """Compute d(district_state)/dt for all districts. Shape (N_D, N_DISTRICT_VARS).

    Bounded: derivatives are clipped so that Euler sub-steps cannot push
    state outside [0, 1]. Decay terms ensure dissipativity.
    """
    n_d = district_array.shape[0]
    g = district_array[:, I_GDP]
    u = district_array[:, I_UNREST]
    a = district_array[:, I_ADMIN]
    d = district_array[:, I_FRAG]
    e = district_array[:, I_IMPL]
    m = district_array[:, I_MEM]

    # ----- Local GDP -----
    flow_cap = capital_flow_term(g, adjacency, params.mu_capital_flow)
    growth = params.alpha_local_gdp * (regime_gdp - g)
    pull = params.beta_local_gdp * (1.0 - u) * (1.0 - regime_exhaustion) - params.beta_local_gdp * g
    drag = params.gamma_local_gdp * u * g
    dg = growth + pull - drag + flow_cap
    dg = np.clip(dg, -g, 1.0 - g)

    # ----- Local unrest -----
    diff_u = unrest_diffusion_term(u, adjacency, params.nu_diffusion)
    push = params.alpha_local_unrest * (regime_instability * (1.0 - e) - u)
    mem_term = params.beta_local_unrest * m * (1.0 - a)
    decay_u = params.delta_unrest * u
    du = push + diff_u + mem_term - decay_u
    du = np.clip(du, -u, 1.0 - u)

    # ----- Admin capacity -----
    baseline_a = 0.7
    da = params.alpha_admin * (baseline_a - a) - params.zeta_admin * u * a
    da = np.clip(da, -a, 1.0 - a)

    # ----- Factional dominance (tracks regime F) -----
    dd = params.alpha_factional_dom * (regime_fragmentation - d)
    dd = np.clip(dd, -d, 1.0 - d)

    # ----- Implementation efficiency base (we store e as the effective value;
    #       in a fuller model we'd store e_base and derive e = e_base * (1 - η_exh*Exh) * (1 - η_d*d).
    #       Here we evolve e directly with decay from Exh and d.)
    e_target = (1.0 - params.eta_exh_impl * regime_exhaustion) * (1.0 - params.eta_frag_impl * d)
    de = params.alpha_impl_base * (e_target - e) - params.psi_impl * u * e
    de = np.clip(de, -e, 1.0 - e)

    # ----- Local memory -----
    dm = params.alpha_local_mem * u * (1.0 - m) - params.beta_local_mem * m
    dm = np.clip(dm, -m, 1.0 - m)

    out = np.zeros_like(district_array)
    out[:, I_GDP] = dg
    out[:, I_UNREST] = du
    out[:, I_ADMIN] = da
    out[:, I_FRAG] = dd
    out[:, I_IMPL] = de
    out[:, I_MEM] = dm
    return out


def compute_pipeline_derivative(
    pipeline_buffers: NDArray[np.float64],
    regime_policy: NDArray[np.float64],
    admin_capacity: NDArray[np.float64],
    params: SystemParameters,
) -> NDArray[np.float64]:
    """Wrapper that calls admin_lag.pipeline_derivative."""
    from .admin_lag import pipeline_derivative
    return pipeline_derivative(
        pipeline_buffers, regime_policy, admin_capacity, params
    )


def regime_policy_from_state(state: RegimeState, n_policy_dims: int) -> NDArray[np.float64]:
    """Build regime policy vector from current state (e.g. pillars + legitimacy).
    Used as target for pipeline. Shape (n_policy_dims,)."""
    # Use pillars if available, else repeat legitimacy
    p = np.array(state.system.pillars, dtype=np.float64)
    if len(p) >= n_policy_dims:
        return p[:n_policy_dims]
    out = np.zeros(n_policy_dims, dtype=np.float64)
    out[:len(p)] = p
    out[len(p):] = state.system.legitimacy
    return out
