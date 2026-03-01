"""
GravitasReward — Composite reward engine for GRAVITAS.

Design axioms:
  1. No single component dominates — governance requires balancing all of them.
  2. Unsustainable stabilization is punished harder than instability.
  3. Resilience growth (Δ trust, Δ coherence) is rewarded separately from
     current level — encourages investment, not just maintenance.
  4. Smoothness term penalizes volatility spikes, pushing agent toward
     dynamic equilibrium rather than boom-bust cycles.
  5. The propaganda trap: PROPAGANDA raises short-term R_stability (bias hides
     risk) but raises true polarization → kills R_polarization next episode.
     The agent must discover this asymmetry through experience.

R_t = w_stab  · R_stability
    + w_frag  · R_fragmentation
    + w_pol   · R_polarization
    + w_exh   · R_exhaustion
    + w_res   · R_resilience
    + w_smooth· R_smoothness
    - w_unsus · P_unsustainable
    + R_abstain_bonus                 (sparse; rewarded for deliberate inaction)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import GravitasWorld


# ─────────────────────────────────────────────────────────────────────────── #
# Component breakdown (for logging / analysis)                                #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class RewardBreakdown:
    stability:       float
    fragmentation:   float
    polarization:    float
    exhaustion:      float
    resilience:      float
    smoothness:      float
    unsustainable:   float
    total:           float
    abstain_bonus:   float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "r_stability":     self.stability,
            "r_fragmentation": self.fragmentation,
            "r_polarization":  self.polarization,
            "r_exhaustion":    self.exhaustion,
            "r_resilience":    self.resilience,
            "r_smoothness":    self.smoothness,
            "r_unsustainable": self.unsustainable,
            "r_abstain":       self.abstain_bonus,
            "r_total":         self.total,
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Per-component reward functions                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _r_stability(world: GravitasWorld) -> float:
    """
    R_stability = mean_i(σᵢ · τᵢ)

    Both structural stability AND trust must be high.
    Military-imposed stability without trust scores poorly here.
    Range: [0, 1]
    """
    c = world.cluster_array()
    sigma = c[:, 0]
    trust = c[:, 4]
    return float(np.mean(sigma * trust))


def _r_fragmentation(world: GravitasWorld) -> float:
    """
    R_fragmentation = 1 - Φ

    Direct penalty for fragmentation probability.
    Range: [0, 1)
    """
    return 1.0 - world.global_state.fragmentation


def _r_polarization(world: GravitasWorld, params: GravitasParams) -> float:
    """
    R_polarization = exp(-κ · Π)

    Exponential decay — at Π=0.5 this is already ~60% of max.
    At Π>0.8, near-zero. Strongly discourages polarization accumulation.
    Range: (0, 1]
    """
    kappa = 3.0   # steepness; matches design spec §V
    return float(np.exp(-kappa * world.global_state.polarization))


def _r_exhaustion(world: GravitasWorld, params: GravitasParams) -> float:
    """
    R_exhaustion = -max(0, E - threshold)²

    No penalty below threshold; quadratic above.
    Catches the military overuse pattern from the v1 analysis (67% exhaustion collapse).
    Range: (-∞, 0]  but practically ≥ -0.16 for E ≤ 1
    """
    overflow = max(0.0, world.global_state.exhaustion - params.exhaustion_threshold)
    return -overflow ** 2


def _r_resilience(world: GravitasWorld, prev_world: GravitasWorld) -> float:
    """
    R_resilience = Δ(aggregate_trust) + Δ(coherence)

    Rewards improvement, not just current level.
    Agent must invest in structural capacity, not just react to hazards.
    Range: [-2, 2] (practically much smaller per step)
    """
    delta_trust = (world.global_state.trust - prev_world.global_state.trust)
    delta_psi   = (world.global_state.coherence - prev_world.global_state.coherence)
    return float(delta_trust + delta_psi)


def _r_smoothness(
    world: GravitasWorld,
    prev_world: GravitasWorld,
    sigma_history: NDArray[np.float64],   # (window, N) rolling σ values
) -> float:
    """
    R_smoothness = -Var_clusters(σᵢ) - |ΔΠ|

    Two terms:
      1. Cross-cluster stability variance: penalizes uneven stability
         (some clusters thriving, others collapsing — governance inequality)
      2. Polarization spike: penalizes sudden Π jumps

    Range: (-∞, 0]
    """
    sigma_now = world.cluster_array()[:, 0]
    cross_var  = float(np.var(sigma_now))

    delta_pol = abs(world.global_state.polarization - prev_world.global_state.polarization)

    # Temporal volatility: std of mean σ over rolling window
    if sigma_history.shape[0] > 1:
        temp_vol = float(np.std(np.mean(sigma_history, axis=1)))
    else:
        temp_vol = 0.0

    return -(cross_var + delta_pol + 0.5 * temp_vol)


def _p_unsustainable(world: GravitasWorld) -> float:
    """
    P_unsustainable = mean(mᵢ) · E · Π

    Penalizes the simultaneous combination of:
      - High military presence
      - High exhaustion
      - High polarization
    This triple combination is the most dangerous failure mode:
    military holding together a system that is burning out and fracturing.
    Range: [0, 1]
    """
    c = world.cluster_array()
    mean_m = float(np.mean(c[:, 3]))
    E = world.global_state.exhaustion
    Pi = world.global_state.polarization
    return mean_m * E * Pi


def _abstain_bonus(
    action_was_noop: bool,
    world: GravitasWorld,
    params: GravitasParams,
) -> float:
    """
    Sparse bonus for deliberate inaction when the system is stable.

    The agent should discover that sometimes the best action is to do nothing.
    Reward: small positive when σ_mean > 0.7, E < 0.3, and no action was taken.
    This teaches the agent that governance is not always intervention.
    """
    if not action_was_noop:
        return 0.0
    c = world.cluster_array()
    mean_sigma = float(np.mean(c[:, 0]))
    if mean_sigma > 0.70 and world.global_state.exhaustion < 0.30:
        return 0.10
    return 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Composite reward                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_reward(
    world: GravitasWorld,
    prev_world: GravitasWorld,
    params: GravitasParams,
    sigma_history: NDArray[np.float64],
    action_was_noop: bool = False,
) -> RewardBreakdown:
    """
    Compute full composite reward and return component breakdown.

    All components are normalized to similar scales before weighting.
    The total is not bounded — VecNormalize in training handles scale.
    """
    r_stab   = _r_stability(world)
    r_frag   = _r_fragmentation(world)
    r_pol    = _r_polarization(world, params)
    r_exh    = _r_exhaustion(world, params)
    r_res    = _r_resilience(world, prev_world)
    r_smooth = _r_smoothness(world, prev_world, sigma_history)
    p_unsus  = _p_unsustainable(world)
    r_abs    = _abstain_bonus(action_was_noop, world, params)

    total = (
        params.w_stability      * r_stab
        + params.w_fragmentation * r_frag
        + params.w_polarization  * r_pol
        + params.w_exhaustion    * r_exh
        + params.w_resilience    * r_res
        + params.w_smoothness    * r_smooth
        - params.w_unsustainable * p_unsus
        + r_abs
    )

    return RewardBreakdown(
        stability=r_stab,
        fragmentation=r_frag,
        polarization=r_pol,
        exhaustion=r_exh,
        resilience=r_res,
        smoothness=r_smooth,
        unsustainable=p_unsus,
        total=float(total),
        abstain_bonus=r_abs,
    )
