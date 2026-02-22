"""Systems: hazard rate, crisis classification, collapse physics, shocks."""
from .hazard import HazardParameters, compute_hazard, compute_survival_probability
from .crisis_classifier import (
    CrisisLevel,
    ClassifierThresholds,
    classify,
    classify_trajectory,
    crisis_distribution,
)
from .collapse_physics import (
    build_province_adjacency,
    is_bridge_province,
    apply_domino_effects,
    apply_national_shock,
)

__all__ = [
    "HazardParameters",
    "compute_hazard",
    "compute_survival_probability",
    "CrisisLevel",
    "ClassifierThresholds",
    "classify",
    "classify_trajectory",
    "crisis_distribution",
    "build_province_adjacency",
    "is_bridge_province",
    "apply_domino_effects",
    "apply_national_shock",
]
