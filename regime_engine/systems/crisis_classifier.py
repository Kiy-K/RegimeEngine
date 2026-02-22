"""
Crisis classifier.

Maps the current regime state to a discrete crisis label using deterministic,
threshold-based rules derived from the macro variables.  No stochastic
components — the classifier is a pure function of SystemState.

Crisis taxonomy (ordered by severity):

  STABLE        — system is healthy; no crisis signals
  TENSION       — elevated instability or radicalization below crisis threshold
  MOBILIZATION  — mobilization pressure building; protest/strike risk
  FRAGMENTATION — power distribution destabilising; factional breakdown risk
  VOLATILITY    — cascade volatility; risk of rapid state change
  CRISIS        — multiple concurrent signals at crisis level
  COLLAPSE      — exhaustion + cascade; regime functional breakdown

Classification logic uses a priority ladder: a state can satisfy multiple
labels, but the highest-severity matching label is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Dict, List

from ..core.state import RegimeState


@unique
class CrisisLevel(IntEnum):
    """Ordered crisis severity levels (higher integer = more severe)."""

    STABLE = 0
    TENSION = 1
    MOBILIZATION = 2
    FRAGMENTATION = 3
    VOLATILITY = 4
    CRISIS = 5
    COLLAPSE = 6


@dataclass(frozen=True)
class ClassifierThresholds:
    """Threshold parameters for the crisis classifier.

    Each field defines a lower bound that the corresponding macro variable
    must reach to contribute to a crisis signal.

    Attributes:
        tension_instability:     I threshold for TENSION label.
        mobilization_mob:        M threshold for MOBILIZATION label.
        fragmentation_frag:      F threshold for FRAGMENTATION label.
        volatility_vol:          V threshold for VOLATILITY label.
        crisis_instability:      I threshold for CRISIS label.
        crisis_volatility:       V threshold for CRISIS label (combined).
        collapse_exhaustion:     Exh threshold for COLLAPSE label.
        collapse_volatility:     V threshold for COLLAPSE label (combined).
    """

    tension_instability: float = 0.30
    mobilization_mob: float = 0.35
    fragmentation_frag: float = 0.45
    volatility_vol: float = 0.55
    crisis_instability: float = 0.60
    crisis_volatility: float = 0.50
    collapse_exhaustion: float = 0.85
    collapse_volatility: float = 0.40

    def __post_init__(self) -> None:
        """Validate all thresholds lie in (0, 1)."""
        for name, value in self.__dict__.items():
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"ClassifierThresholds.{name} must be in (0, 1), got {value}"
                )


def classify(
    state: RegimeState,
    thresholds: ClassifierThresholds,
) -> CrisisLevel:
    """Return the highest-severity crisis label for the current state.

    The classifier evaluates each label from highest to lowest severity and
    returns the first match.  If no threshold is exceeded, returns STABLE.

    Args:
        state:      Current regime state (system must be current).
        thresholds: Classifier threshold parameters.

    Returns:
        Most severe applicable CrisisLevel.
    """
    sys = state.system

    # COLLAPSE — exhaustion × volatility cascade
    if (
        sys.exhaustion >= thresholds.collapse_exhaustion
        and sys.volatility >= thresholds.collapse_volatility
    ):
        return CrisisLevel.COLLAPSE

    # CRISIS — severe instability + high volatility
    if (
        sys.instability >= thresholds.crisis_instability
        and sys.volatility >= thresholds.crisis_volatility
    ):
        return CrisisLevel.CRISIS

    # VOLATILITY — cascade risk
    if sys.volatility >= thresholds.volatility_vol:
        return CrisisLevel.VOLATILITY

    # FRAGMENTATION — structural breakdown
    if sys.fragmentation >= thresholds.fragmentation_frag:
        return CrisisLevel.FRAGMENTATION

    # MOBILIZATION — pressure building
    if sys.mobilization >= thresholds.mobilization_mob:
        return CrisisLevel.MOBILIZATION

    # TENSION — early warning
    if sys.instability >= thresholds.tension_instability:
        return CrisisLevel.TENSION

    return CrisisLevel.STABLE


def classify_trajectory(
    state_sequence: List[RegimeState],
    thresholds: ClassifierThresholds,
) -> List[CrisisLevel]:
    """Classify every state in a trajectory.

    Args:
        state_sequence: Ordered list of RegimeState objects.
        thresholds:     Classifier threshold parameters.

    Returns:
        List of CrisisLevel labels, same length as state_sequence.
    """
    return [classify(s, thresholds) for s in state_sequence]


def crisis_distribution(
    labels: List[CrisisLevel],
) -> Dict[CrisisLevel, float]:
    """Compute the fraction of time spent at each crisis level.

    Args:
        labels: List of CrisisLevel labels from classify_trajectory().

    Returns:
        Dictionary mapping each CrisisLevel to its frequency in [0, 1].
    """
    if not labels:
        return {level: 0.0 for level in CrisisLevel}
    n = len(labels)
    counts: Dict[CrisisLevel, int] = {level: 0 for level in CrisisLevel}
    for label in labels:
        counts[label] += 1
    return {level: count / n for level, count in counts.items()}


def max_crisis_reached(labels: List[CrisisLevel]) -> CrisisLevel:
    """Return the most severe crisis level encountered in a trajectory.

    Args:
        labels: List of CrisisLevel labels.

    Returns:
        Maximum CrisisLevel observed.
    """
    if not labels:
        return CrisisLevel.STABLE
    return max(labels)