"""
Regime metrics.

Computes summary statistics over a trajectory of RegimeState objects.
All metric functions accept a list of RegimeState and return scalar or dict
values.  No side effects.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.state import RegimeState
from ..systems.crisis_classifier import ClassifierThresholds, CrisisLevel, classify


def mean_legitimacy(trajectory: List[RegimeState]) -> float:
    """Mean legitimacy L averaged over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Mean L ∈ [0, 1].
    """
    if not trajectory:
        return 0.0
    return float(np.mean([s.system.legitimacy for s in trajectory]))


def mean_instability(trajectory: List[RegimeState]) -> float:
    """Mean instability I averaged over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Mean I ∈ [0, 1].
    """
    if not trajectory:
        return 0.0
    return float(np.mean([s.system.instability for s in trajectory]))


def mean_fragmentation(trajectory: List[RegimeState]) -> float:
    """Mean fragmentation F averaged over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Mean F ∈ [0, 1).
    """
    if not trajectory:
        return 0.0
    return float(np.mean([s.system.fragmentation for s in trajectory]))


def mean_volatility(trajectory: List[RegimeState]) -> float:
    """Mean volatility V averaged over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Mean V ∈ [0, 1).
    """
    if not trajectory:
        return 0.0
    return float(np.mean([s.system.volatility for s in trajectory]))


def mean_exhaustion(trajectory: List[RegimeState]) -> float:
    """Mean exhaustion Exh averaged over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Mean Exh ∈ [0, 1].
    """
    if not trajectory:
        return 0.0
    return float(np.mean([s.system.exhaustion for s in trajectory]))


def max_radicalization(trajectory: List[RegimeState]) -> float:
    """Maximum Rad_i across all factions and all timesteps.

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        Maximum radicalization encountered ∈ [0, 1].
    """
    if not trajectory:
        return 0.0
    return float(
        max(
            max(f.radicalization for f in s.factions)
            for s in trajectory
        )
    )


def power_entropy(trajectory: List[RegimeState]) -> List[float]:
    """Shannon entropy of the power distribution at each step.

    H = −Σ P_i · log(P_i + ε),  ε = 1e-10 to avoid log(0).

    Args:
        trajectory: Ordered list of RegimeState objects.

    Returns:
        List of entropy values, one per state.
    """
    eps = 1e-10
    entropies = []
    for state in trajectory:
        powers = state.get_faction_powers()
        h = float(-np.sum(powers * np.log(powers + eps)))
        entropies.append(h)
    return entropies


def crisis_fraction(
    trajectory: List[RegimeState],
    thresholds: ClassifierThresholds,
    level: CrisisLevel,
) -> float:
    """Fraction of steps at or above a given crisis level.

    Args:
        trajectory: Ordered list of RegimeState objects.
        thresholds: Crisis classifier thresholds.
        level:      Minimum crisis level to count.

    Returns:
        Fraction ∈ [0, 1].
    """
    if not trajectory:
        return 0.0
    count = sum(
        1 for s in trajectory if classify(s, thresholds) >= level
    )
    return count / len(trajectory)


def summary_statistics(
    trajectory: List[RegimeState],
    thresholds: ClassifierThresholds,
) -> Dict[str, float]:
    """Compute a comprehensive summary over a trajectory.

    Args:
        trajectory: Ordered list of RegimeState objects.
        thresholds: Crisis classifier thresholds.

    Returns:
        Dictionary of metric name → scalar value.
    """
    return {
        "n_steps": float(len(trajectory)),
        "mean_legitimacy": mean_legitimacy(trajectory),
        "mean_instability": mean_instability(trajectory),
        "mean_fragmentation": mean_fragmentation(trajectory),
        "mean_volatility": mean_volatility(trajectory),
        "mean_exhaustion": mean_exhaustion(trajectory),
        "max_radicalization": max_radicalization(trajectory),
        "crisis_fraction_tension": crisis_fraction(
            trajectory, thresholds, CrisisLevel.TENSION
        ),
        "crisis_fraction_crisis": crisis_fraction(
            trajectory, thresholds, CrisisLevel.CRISIS
        ),
        "crisis_fraction_collapse": crisis_fraction(
            trajectory, thresholds, CrisisLevel.COLLAPSE
        ),
        "final_legitimacy": trajectory[-1].system.legitimacy if trajectory else 0.0,
        "final_exhaustion": trajectory[-1].system.exhaustion if trajectory else 0.0,
    }
