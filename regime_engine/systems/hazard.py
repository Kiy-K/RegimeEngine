"""
Hazard rate computation.

The hazard rate H ∈ [0, 1] models the instantaneous probability of a critical
regime event (coup, collapse, revolution, transition).  It combines:

  - Regime instability (I)
  - Volatility cascade (V)
  - Power fragmentation (F)
  - Exhaustion (Exh) — high exhaustion moderates acute hazard

Hazard equation:
    H_raw = I^ω_I · V^ω_V · F^ω_F · (1 − δ_Exh · Exh)
    H     = clip(H_raw, 0, 1)

where ω_I, ω_V, ω_F ∈ (0, 1] are convexity weights and δ_Exh ∈ [0, 1] is the
exhaustion moderation factor.

The parametric form uses a product of power-law terms so that hazard is zero
whenever any single critical factor is zero, and saturates gracefully when all
factors are high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.parameters import SystemParameters
from ..core.state import RegimeState


@dataclass(frozen=True)
class HazardParameters:
    """Weights governing the hazard rate formula.

    Attributes:
        omega_instability:  Convexity weight for instability factor (0, 1].
        omega_volatility:   Convexity weight for volatility factor (0, 1].
        omega_fragmentation: Convexity weight for fragmentation factor (0, 1].
        delta_exhaustion:   Exhaustion moderation factor [0, 1].
    """

    omega_instability: float = 0.50
    omega_volatility: float = 0.35
    omega_fragmentation: float = 0.15
    delta_exhaustion: float = 0.40
    kappa_clustering: float = 0.30
    """Phase 3: weight for district unrest clustering in hazard (0 = disable)."""

    def __post_init__(self) -> None:
        """Validate all weights are in (0, 1] or [0, 1] as appropriate."""
        for name, value in {
            "omega_instability": self.omega_instability,
            "omega_volatility": self.omega_volatility,
            "omega_fragmentation": self.omega_fragmentation,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if not 0.0 <= self.delta_exhaustion <= 1.0:
            raise ValueError(
                f"delta_exhaustion must be in [0, 1], got {self.delta_exhaustion}"
            )


def _nonlinear_unrest_hazard_component(unrest_mean: float) -> float:
    """Phase 4: tipping behavior instead of smooth linear increase.
    unrest_mean in [0, 1]; returns multiplicative component in [0, 1].
    """
    u = float(np.clip(unrest_mean, 0.0, 1.0))
    if u < 0.3:
        return u
    if u < 0.6:
        return u ** 1.5
    return u ** 2.5


def compute_hazard(
    state: RegimeState,
    hazard_params: HazardParameters,
    unrest_mean: float | None = None,
    clustering_index: float | None = None,
) -> float:
    """Compute instantaneous hazard rate H ∈ [0, 1].

    Base: H_base = I^ω_I · V^ω_V · F^ω_F · (1 − δ_Exh · Exh)

    Phase 4 nonlinear escalation:
    - Unrest component: if unrest_mean provided, multiply by nonlinear f(unrest_mean):
      f(u) = u if u<0.3, u^1.5 if u<0.6, u^2.5 else.
    - If clustering_index > 0.4: amplify by (1 + 2 * clustering_index).
      Else: (1 + κ_clus · clustering) as before.

    Returns H clipped to [0, 1] (can exceed 1.0 before clip for collapse condition hazard > 1.2).
    """
    inst = state.system.instability
    vol = state.system.volatility
    frag = state.system.fragmentation
    exh = state.system.exhaustion

    base = (
        inst ** hazard_params.omega_instability
        * vol ** hazard_params.omega_volatility
        * frag ** hazard_params.omega_fragmentation
    )
    exhaustion_brake = 1.0 - hazard_params.delta_exhaustion * exh
    raw = base * exhaustion_brake

    # Phase 3/4: clustering and unrest
    clust = clustering_index
    if clust is None and getattr(state, "hierarchical", None) is not None:
        from ..core.hierarchical_coupling import unrest_clustering_index
        u = state.hierarchical.get_district_array()[:, 1]
        clust = unrest_clustering_index(u, state.hierarchical.adjacency)

    if clust is not None:
        if clust > 0.4:
            raw = raw * (1.0 + 2.0 * float(clust))
        else:
            raw = raw * (1.0 + hazard_params.kappa_clustering * float(clust))

    # Phase 4: nonlinear unrest component (tipping)
    if unrest_mean is not None:
        raw = raw * (0.5 + 0.5 * _nonlinear_unrest_hazard_component(unrest_mean))

    return float(np.clip(raw, 0.0, 2.0))  # allow > 1 for collapse check hazard > 1.2


def compute_cumulative_hazard(
    state_sequence: list,
    hazard_params: HazardParameters,
    dt: float,
) -> float:
    """Integrate hazard rate over a trajectory to get cumulative hazard.

    Uses the rectangle rule since steps are uniform.

    H_cum = Σ H(t_k) · dt

    Args:
        state_sequence: Ordered list of RegimeState objects.
        hazard_params:  Hazard weighting parameters.
        dt:             Time step between consecutive states.

    Returns:
        Cumulative hazard ≥ 0 (not bounded above; survival prob = exp(−H_cum)).
    """
    if not state_sequence:
        return 0.0
    total = sum(
        compute_hazard(s, hazard_params) for s in state_sequence
    )
    return total * dt


def compute_survival_probability(cumulative_hazard: float) -> float:
    """Convert cumulative hazard to survival probability via exp(−H_cum).

    Args:
        cumulative_hazard: Non-negative cumulative hazard value.

    Returns:
        Survival probability ∈ (0, 1].
    """
    if cumulative_hazard < 0.0:
        raise ValueError(
            f"cumulative_hazard must be >= 0, got {cumulative_hazard}"
        )
    return float(np.exp(-cumulative_hazard))