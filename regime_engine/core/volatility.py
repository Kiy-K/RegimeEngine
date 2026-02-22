"""
Volatility dynamics.

System volatility is a derived (algebraic) macro variable computed entirely
from the current faction distribution and exhaustion level:

    V = tanh(κ_V · M · (1 − Exh) · (1 + I))

where:
    M   = mobilization (power-weighted grievance memory × radicalization)
    I   = instability  (power-weighted radicalization × lack of cohesion)
    Exh = exhaustion   (time-evolved societal fatigue)

Stability proof:
  - tanh maps ℝ → (−1, 1); for non-negative argument → [0, 1).
  - As Exh → 1: argument → 0 → V → 0.  Exhaustion fully suppresses cascade.
  - As M, I → 1 and Exh → 0:
        V → tanh(2 · κ_V) < 1  for any finite κ_V.
  - No positive feedback: V does not appear in M or I equations.

V therefore satisfies 0 ≤ V < 1 strictly for all time.
"""

from __future__ import annotations

import numpy as np

from .parameters import SystemParameters
from .state import RegimeState


def compute_volatility(
    mobilization: float,
    instability: float,
    exhaustion: float,
    kappa_v: float,
) -> float:
    """V = tanh(κ_V · M · (1 − Exh) · (1 + I)).

    Args:
        mobilization: Current M ∈ [0, 1].
        instability:  Current I ∈ [0, 1].
        exhaustion:   Current Exh ∈ [0, 1].
        kappa_v:      Amplification factor (> 0).

    Returns:
        Volatility V ∈ [0, 1).
    """
    argument = kappa_v * mobilization * (1.0 - exhaustion) * (1.0 + instability)
    return float(np.clip(np.tanh(argument), 0.0, 1.0))


def compute_volatility_from_state(
    state: RegimeState, params: SystemParameters
) -> float:
    """Compute volatility directly from a RegimeState.

    This is a convenience wrapper that reads M, I, Exh from an already-computed
    SystemState.  The SystemState must be current (call recompute_system_state
    before this if factions have changed).

    Args:
        state:  Current regime state (system.mobilization, instability, exhaustion used).
        params: System parameters (kappa_v).

    Returns:
        Volatility V ∈ [0, 1).
    """
    return compute_volatility(
        mobilization=state.system.mobilization,
        instability=state.system.instability,
        exhaustion=state.system.exhaustion,
        kappa_v=params.kappa_v,
    )


def volatility_upper_bound(kappa_v: float) -> float:
    """Theoretical upper bound of V given worst-case inputs M=1, I=1, Exh=0.

    V_max = tanh(2 · κ_V) < 1.

    Args:
        kappa_v: Amplification factor.

    Returns:
        Strict upper bound on V.
    """
    return float(np.tanh(2.0 * kappa_v))