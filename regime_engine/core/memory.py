"""
Grievance memory dynamics.

Each faction accumulates and forgets grievances via a linear ODE:

    dMem_i/dt = α_Mem · [I · (1 − Coh_i) − β_Mem · Mem_i]

Stability proof (embedded):
  - When Mem_i = 1:
      dMem_i/dt = α_Mem · [I·(1−Coh_i) − β_Mem]
               ≤ α_Mem · [1 − β_Mem]  (since I, 1−Coh_i ≤ 1)
    With β_Mem > 1 this is strictly negative → Mem_i cannot exceed 1.
  - When Mem_i = 0:
      dMem_i/dt = α_Mem · I · (1 − Coh_i) ≥ 0 → Mem_i cannot go below 0.
  - Steady state: Mem*_i = I·(1−Coh_i)/β_Mem < 1/β_Mem < 1.

Memory is intentionally NOT gated by Exhaustion: historical trauma persists
even in exhausted societies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import SystemParameters
from .state import RegimeState


def compute_memory_derivatives(
    state: RegimeState,
    params: SystemParameters,
) -> NDArray[np.float64]:
    """Compute dMem_i/dt for all factions.

    Equation:
        dMem_i/dt = α_Mem · [I · (1 − Coh_i) − β_Mem · Mem_i]

    The raw derivative is clipped to the interval [−Mem_i, 1−Mem_i] so that
    Euler or sub-step evaluations cannot push Mem_i outside [0, 1] in a single
    call.

    Args:
        state:  Current regime state.
        params: System parameters (alpha_mem, beta_mem).

    Returns:
        Array of dMem_i/dt values, shape (n_factions,).
    """
    cohesions = state.get_faction_cohesions()
    memories = state.get_faction_memories()
    instability = state.system.instability

    raw = params.alpha_mem * (
        instability * (1.0 - cohesions) - params.beta_mem * memories
    )
    # Conservative clipping: sub-step evaluations (RK4 k2/k3/k4) may
    # receive memories slightly outside [0, 1]; clip to safe range.
    lower = -np.clip(memories, 0.0, 1.0)
    upper = 1.0 - np.clip(memories, 0.0, 1.0)
    return np.clip(raw, lower, upper)


def memory_steady_state(
    instability: float,
    cohesion: float,
    params: SystemParameters,
) -> float:
    """Compute the analytical steady-state Mem* for given I and Coh_i.

    Formula:
        Mem* = I · (1 − Coh_i) / β_Mem

    Since β_Mem > 1 (enforced by SystemParameters validation), Mem* < 1.

    Args:
        instability: System instability I ∈ [0, 1].
        cohesion:    Faction cohesion Coh_i ∈ [0, 1].
        params:      System parameters.

    Returns:
        Steady-state memory value ∈ [0, 1).
    """
    raw = instability * (1.0 - cohesion) / params.beta_mem
    return float(np.clip(raw, 0.0, 1.0))


def memory_time_constant(params: SystemParameters) -> float:
    """Return τ = 1 / β_Mem (characteristic decay time).

    Args:
        params: System parameters.

    Returns:
        Decay time constant (> 0).
    """
    return 1.0 / params.beta_mem


def memory_convergence_steps(
    params: SystemParameters,
    tolerance: float = 0.01,
) -> int:
    """Estimate the number of simulation steps for memory to converge.

    Uses the exponential decay formula: t = −τ · ln(tolerance).
    Converts to steps via params.dt.

    Args:
        params:    System parameters.
        tolerance: Convergence tolerance (0 < tol < 1).

    Returns:
        Estimated integer number of steps.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError(
            f"tolerance must be in (0, 1), got {tolerance}"
        )
    tau = memory_time_constant(params)
    continuous_time = -tau * float(np.log(tolerance))
    return max(1, int(np.ceil(continuous_time / params.dt)))


def validate_memory_bounds(memories: NDArray[np.float64]) -> bool:
    """Return True iff every element of memories lies in [0, 1].

    Args:
        memories: Array of Mem_i values.

    Returns:
        Boolean validity flag.
    """
    return bool(np.all((memories >= 0.0) & (memories <= 1.0)))