"""
RK4 numerical integrator for the regime engine ODE system.

Integration pipeline per step:
  1. Recompute macro variables from current faction state.
  2. Evaluate k1, k2, k3, k4.
  3. Apply weighted sum: Δx = (dt/6)(k1 + 2k2 + 2k3 + k4).
  4. SDE Noise: Add Wiener process (sigma * sqrt(dt) * N(0,1)) to Power and Radicalization.
  5. Clip all micro-state values to [0, 1].
  6. Normalise power shares so Σ P_i = 1.
  7. Re-derive all algebraic macro variables.
  8. Increment step counter.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .admin_lag import pipeline_derivative
from .district_dynamics import (
    I_ADMIN,
    compute_district_derivatives,
    regime_policy_from_state,
)
from .dynamics import compute_all_derivatives
from .factions import recompute_system_state
from .hierarchical_state import (
    HierarchicalState,
    array_to_districts,
)
from .parameters import SystemParameters
from .state import FactionState, RegimeState, SystemState


def _extract_arrays(
    state: RegimeState,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, ...], ...],
]:
    """Extract all integration variables from a RegimeState."""
    pillars = np.array(state.system.pillars, dtype=np.float64)
    if len(pillars) == 0:
        pillars = np.zeros(0)  # should not happen if neutral() creates it
    
    return (
        state.get_faction_powers(),
        state.get_faction_radicalizations(),
        state.get_faction_cohesions(),
        state.get_faction_memories(),
        state.get_faction_wealths(),
        state.system.exhaustion,
        state.system.state_gdp,
        pillars,
        state.affinity_matrix,
    )


def _build_state_from_arrays(
    powers: NDArray[np.float64],
    rads: NDArray[np.float64],
    cohs: NDArray[np.float64],
    mems: NDArray[np.float64],
    wealths: NDArray[np.float64],
    exh: float,
    gdp: float,
    pillars: NDArray[np.float64],
    aff_matrix: tuple[tuple[float, ...], ...],
    reference: RegimeState,
    params: SystemParameters,
) -> RegimeState:
    """Construct a temporary RegimeState from raw arrays for k-step evaluation."""
    powers_c = np.clip(powers, 0.0, 1.0)
    rads_c = np.clip(rads, 0.0, 1.0)
    cohs_c = np.clip(cohs, 0.0, 1.0)
    mems_c = np.clip(mems, 0.0, 1.0)
    wealths_c = np.clip(wealths, 0.0, 1.0)
    
    exh_c = float(np.clip(exh, 0.0, 1.0))
    gdp_c = float(np.clip(gdp, 0.0, 1.0))
    pillars_c = tuple(float(x) for x in np.clip(pillars, 0.0, 1.0))

    # Renormalise power shares
    power_sum = float(np.sum(powers_c))
    if power_sum > 0.0:
        powers_c = powers_c / power_sum
    else:
        powers_c = np.full_like(powers_c, 1.0 / len(powers_c))

    factions = [
        FactionState(
            power=float(powers_c[i]),
            radicalization=float(rads_c[i]),
            cohesion=float(cohs_c[i]),
            memory=float(mems_c[i]),
            wealth=float(wealths_c[i]),
        )
        for i in range(len(powers_c))
    ]

    placeholder_system = SystemState(
        legitimacy=0.5,
        cohesion=0.25,
        fragmentation=0.0,
        instability=0.0,
        mobilization=0.0,
        repression=0.5,
        elite_alignment=0.5,
        volatility=0.0,
        exhaustion=exh_c,
        state_gdp=gdp_c,
        pillars=pillars_c,
    )

    # Note logic: if aff_matrix is passed with len>0, we just enforce [-1, 1] clip
    if aff_matrix:
        aff_c = []
        for row in aff_matrix:
            aff_c.append(tuple(float(np.clip(x, -1.0, 1.0)) for x in row))
        aff_matrix = tuple(aff_c)

    intermediate = RegimeState(
        factions=factions,
        system=placeholder_system,
        affinity_matrix=aff_matrix,
        step=reference.step,
        hierarchical=getattr(reference, "hierarchical", None),
    )
    return recompute_system_state(intermediate, params)


def _step_hierarchical(
    state: RegimeState,
    s2: RegimeState,
    s3: RegimeState,
    s4: RegimeState,
    new_state: RegimeState,
    params: SystemParameters,
    dt: float,
) -> RegimeState:
    """RK4 step for district state and pipeline buffers; attach to new_state."""
    hier = state.hierarchical
    if hier is None:
        return new_state
    half_dt = dt * 0.5
    D0 = hier.get_district_array()
    P0 = hier.pipeline_buffers.copy()
    A = hier.adjacency
    n_pol = P0.shape[1]

    def _reg(reg_state: RegimeState):
        return (
            reg_state.system.state_gdp,
            reg_state.system.instability,
            reg_state.system.fragmentation,
            reg_state.system.exhaustion,
        )

    def _d_deriv(d_arr: NDArray[np.float64], p_buf: NDArray[np.float64], reg_state: RegimeState):
        gdp, inst, frag, exh = _reg(reg_state)
        return compute_district_derivatives(
            d_arr, p_buf, A, gdp, inst, frag, exh, params
        )

    def _p_deriv(p_buf: NDArray[np.float64], d_arr: NDArray[np.float64], reg_state: RegimeState):
        regime_pol = regime_policy_from_state(reg_state, n_pol)
        return pipeline_derivative(p_buf, regime_pol, d_arr[:, I_ADMIN], params)

    dD1 = _d_deriv(D0, P0, state)
    dP1 = _p_deriv(P0, D0, state)

    D2 = np.clip(D0 + half_dt * dD1, 0.0, 1.0)
    P2 = P0 + half_dt * dP1
    dD2 = _d_deriv(D2, P2, s2)
    dP2 = _p_deriv(P2, D2, s2)

    D3 = np.clip(D0 + half_dt * dD2, 0.0, 1.0)
    P3 = P0 + half_dt * dP2
    dD3 = _d_deriv(D3, P3, s3)
    dP3 = _p_deriv(P3, D3, s3)

    D4 = np.clip(D0 + dt * dD3, 0.0, 1.0)
    P4 = P0 + dt * dP3
    dD4 = _d_deriv(D4, P4, s4)
    dP4 = _p_deriv(P4, D4, s4)

    factor = dt / 6.0
    D_new = D0 + factor * (dD1 + 2.0 * dD2 + 2.0 * dD3 + dD4)
    P_new = P0 + factor * (dP1 + 2.0 * dP2 + 2.0 * dP3 + dP4)

    # SDE noise on district unrest (and optionally local_gdp)
    if params.sigma_noise > 0:
        dt_sqrt = np.sqrt(dt)
        n_d = D_new.shape[0]
        D_new[:, 1] += params.sigma_noise * 0.5 * dt_sqrt * np.random.randn(n_d)

    D_new = np.clip(D_new, 0.0, 1.0)
    P_new = np.clip(P_new, 0.0, 1.0)

    new_districts = array_to_districts(D_new)
    new_hier = hier.copy_with_districts(new_districts).copy_with_pipeline_buffers(P_new)
    return new_state.copy_with_hierarchical(new_hier)


def _apply_euler_maruyama(
    powers: NDArray[np.float64],
    rads: NDArray[np.float64],
    params: SystemParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply SDE noise to base RK4 integration.
    dX_t = f(X_t)dt + sigma * dW_t
    """
    if params.sigma_noise <= 0.0:
        return powers, rads
        
    n = len(powers)
    dt_sqrt = np.sqrt(params.dt)
    noise_p = params.sigma_noise * dt_sqrt * np.random.randn(n)
    noise_r = params.sigma_noise * dt_sqrt * np.random.randn(n)
    
    return powers + noise_p, rads + noise_r


def step(state: RegimeState, params: SystemParameters) -> RegimeState:
    """Advance the system by one time step using the RK4 method."""
    dt = params.dt
    half_dt = dt * 0.5

    P0, Rad0, Coh0, Mem0, W0, Exh0, Gdp0, Pil0, Aff0 = _extract_arrays(state)
    if not Aff0:
        Aff0 = tuple(tuple(1.0 if i == j else 0.0 for j in range(len(P0))) for i in range(len(P0)))

    # ------------------------------------------------------------------ k1 --
    dP1, dRad1, dCoh1, dMem1, dW1, dExh1, dGdp1, dPil1, dAff1 = compute_all_derivatives(state, params)

    # Helper for affinities
    def add_aff(aff0, daff, hdt):
        return tuple(
            tuple(aff0[i][j] + hdt * daff[i][j] for j in range(len(aff0)))
            for i in range(len(aff0))
        )

    # ------------------------------------------------------------------ k2 --
    s2 = _build_state_from_arrays(
        P0 + half_dt * dP1, Rad0 + half_dt * dRad1, Coh0 + half_dt * dCoh1, Mem0 + half_dt * dMem1, W0 + half_dt * dW1,
        Exh0 + half_dt * dExh1, Gdp0 + half_dt * dGdp1, Pil0 + half_dt * dPil1, add_aff(Aff0, dAff1, half_dt),
        state, params,
    )
    dP2, dRad2, dCoh2, dMem2, dW2, dExh2, dGdp2, dPil2, dAff2 = compute_all_derivatives(s2, params)

    # ------------------------------------------------------------------ k3 --
    s3 = _build_state_from_arrays(
        P0 + half_dt * dP2, Rad0 + half_dt * dRad2, Coh0 + half_dt * dCoh2, Mem0 + half_dt * dMem2, W0 + half_dt * dW2,
        Exh0 + half_dt * dExh2, Gdp0 + half_dt * dGdp2, Pil0 + half_dt * dPil2, add_aff(Aff0, dAff2, half_dt),
        state, params,
    )
    dP3, dRad3, dCoh3, dMem3, dW3, dExh3, dGdp3, dPil3, dAff3 = compute_all_derivatives(s3, params)

    # ------------------------------------------------------------------ k4 --
    s4 = _build_state_from_arrays(
        P0 + dt * dP3, Rad0 + dt * dRad3, Coh0 + dt * dCoh3, Mem0 + dt * dMem3, W0 + dt * dW3,
        Exh0 + dt * dExh3, Gdp0 + dt * dGdp3, Pil0 + dt * dPil3, add_aff(Aff0, dAff3, dt),
        state, params,
    )
    dP4, dRad4, dCoh4, dMem4, dW4, dExh4, dGdp4, dPil4, dAff4 = compute_all_derivatives(s4, params)

    # ----------------------------------------------------------------- RK4 weighted sum --
    factor = dt / 6.0
    new_P = P0 + factor * (dP1 + 2.0 * dP2 + 2.0 * dP3 + dP4)
    new_Rad = Rad0 + factor * (dRad1 + 2.0 * dRad2 + 2.0 * dRad3 + dRad4)
    new_Coh = Coh0 + factor * (dCoh1 + 2.0 * dCoh2 + 2.0 * dCoh3 + dCoh4)
    new_Mem = Mem0 + factor * (dMem1 + 2.0 * dMem2 + 2.0 * dMem3 + dMem4)
    new_W = W0 + factor * (dW1 + 2.0 * dW2 + 2.0 * dW3 + dW4)
    
    new_Exh = Exh0 + factor * (dExh1 + 2.0 * dExh2 + 2.0 * dExh3 + dExh4)
    new_Gdp = Gdp0 + factor * (dGdp1 + 2.0 * dGdp2 + 2.0 * dGdp3 + dGdp4)
    new_Pil = Pil0 + factor * (dPil1 + 2.0 * dPil2 + 2.0 * dPil3 + dPil4)
    
    new_Aff = tuple(
        tuple(Aff0[i][j] + factor * (dAff1[i][j] + 2*dAff2[i][j] + 2*dAff3[i][j] + dAff4[i][j]) for j in range(len(Aff0)))
        for i in range(len(Aff0))
    )

    # Apply SDE Noise (Euler-Maruyama step)
    new_P, new_Rad = _apply_euler_maruyama(new_P, new_Rad, params)

    new_state = _build_state_from_arrays(
        new_P, new_Rad, new_Coh, new_Mem, new_W, new_Exh, new_Gdp, new_Pil, new_Aff,
        state, params
    ).advance_step()

    # Phase 3: hierarchical district + pipeline RK4 step
    if getattr(params, "use_hierarchy", False) and getattr(state, "hierarchical", None) is not None:
        new_state = _step_hierarchical(
            state, s2, s3, s4, new_state, params, dt
        )

    return new_state


def multi_step(
    state: RegimeState,
    params: SystemParameters,
    n_steps: int,
) -> RegimeState:
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    current = state
    for _ in range(n_steps):
        current = step(current, params)
    return current