"""
Multi-scale coupling: district-level state → regime macro variables.

Macro stability depends on micro turbulence. The district layer is not decorative.

- Global fragmentation: weighted function of district factional_dominance.
- Regime exhaustion: accumulates from weighted district stress (local_unrest × (1 - admin_capacity)).
- Volatility: increases when district unrest *variance* is high (variance is danger).
- Hazard: amplified by spatial *clustering* of unrest (clustering > mean unrest).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .hierarchical_state import HierarchicalState
from .parameters import SystemParameters


def district_stress(local_unrest: NDArray[np.float64], admin_capacity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stress per district: high unrest and low admin capacity."""
    return local_unrest * (1.0 - admin_capacity)


def district_weights(n_districts: int) -> NDArray[np.float64]:
    """Uniform weights (1/N_D) for aggregation. Can be replaced by population weights."""
    return np.full(n_districts, 1.0 / n_districts, dtype=np.float64)


def global_fragmentation_from_districts(
    hierarchical: HierarchicalState,
    regime_fragmentation: float,
    params: SystemParameters,
) -> float:
    """Blend regime F with district-level factional_dominance.

    F_global = (1 - w) * F_regime + w * weighted_mean(district_factional_dominance).
    Keeps F in [0, 1).
    """
    if hierarchical.n_districts == 0:
        return regime_fragmentation
    arr = hierarchical.get_district_array()
    d = arr[:, 3]  # factional_dominance
    w = district_weights(hierarchical.n_districts)
    frag_district = float(np.clip(np.sum(w * d), 0.0, 1.0))
    # Blend: 0.7 regime + 0.3 district (avoid hard override)
    blend = 0.7 * regime_fragmentation + 0.3 * frag_district
    return float(np.clip(blend, 0.0, 1.0))


def exhaustion_increment_from_districts(
    hierarchical: HierarchicalState,
    params: SystemParameters,
) -> float:
    """Extra d(Exh)/dt term from district stress: α_exh_dist * mean(stress)."""
    if hierarchical.n_districts == 0:
        return 0.0
    arr = hierarchical.get_district_array()
    u = arr[:, 1]
    a = arr[:, 2]
    stress = district_stress(u, a)
    w = district_weights(hierarchical.n_districts)
    mean_stress = float(np.sum(w * stress))
    return params.alpha_exh_district_stress * mean_stress


def volatility_bump_from_districts(
    hierarchical: HierarchicalState,
    params: SystemParameters,
) -> float:
    """Additive term for volatility from district unrest variance.

    Returns value in [0, 1] to add (or multiply) into volatility formula.
    """
    if hierarchical.n_districts == 0:
        return 0.0
    arr = hierarchical.get_district_array()
    u = arr[:, 1]
    var_u = float(np.var(u))
    # tanh so bounded
    return float(np.clip(np.tanh(params.kappa_var_volatility * var_u), 0.0, 1.0))


def unrest_clustering_index(
    local_unrest: NDArray[np.float64],
    adjacency: NDArray[np.float64],
) -> float:
    """Spatial clustering of unrest: Σ_ij A_ij u_i u_j normalized.
    High when adjacent districts are both high. Clustering > mean unrest for hazard."""
    n = len(local_unrest)
    if n == 0:
        return 0.0
    total = np.sum(adjacency * np.outer(local_unrest, local_unrest))
    u_sum = float(np.sum(local_unrest)) + 1e-9
    raw = total / (u_sum * u_sum + 1e-9)
    return float(np.clip(raw, 0.0, 1.0))


def get_geography_summary(
    hierarchical: HierarchicalState,
    params: SystemParameters | None = None,
) -> dict:
    """Summary of political geography for logging/CLI: feel pressure building.

    Returns a dict so that when stepping the system you can see:
    - Unrest forming in one province (province_unrest_means, top_unstable_province)
    - Capital distribution (province_gdp_means, unrest_variance)
    - Admin lag / stress (province_admin_means, mean_district_stress)
    - Volatility roots (unrest_variance, volatility_bump if params)
    - Hazard amplification from clustering (clustering_index, hazard_amplification)

    Variance is danger. Clustering > mean unrest.
    """
    if hierarchical.n_districts == 0:
        return {
            "province_unrest_means": [],
            "province_gdp_means": [],
            "province_admin_means": [],
            "unrest_variance": 0.0,
            "clustering_index": 0.0,
            "mean_district_stress": 0.0,
            "top_unstable_province": -1,
            "volatility_bump": 0.0,
            "hazard_amplification": 1.0,
        }
    arr = hierarchical.get_district_array()
    prov = hierarchical.province_of_district
    n_provinces = hierarchical.n_provinces
    u_arr = arr[:, 1]   # unrest
    g_arr = arr[:, 0]   # local_gdp
    a_arr = arr[:, 2]   # admin_capacity

    province_unrest_means = [
        float(np.mean(u_arr[prov == p])) for p in range(n_provinces)
    ]
    province_gdp_means = [
        float(np.mean(g_arr[prov == p])) for p in range(n_provinces)
    ]
    province_admin_means = [
        float(np.mean(a_arr[prov == p])) for p in range(n_provinces)
    ]

    unrest_variance = float(np.var(u_arr))
    clustering_index = unrest_clustering_index(u_arr, hierarchical.adjacency)
    stress = district_stress(u_arr, a_arr)
    mean_stress = float(np.mean(stress))
    top_unstable_province = int(np.argmax(province_unrest_means))

    volatility_bump = 0.0
    if params is not None:
        volatility_bump = volatility_bump_from_districts(hierarchical, params)

    # Hazard amplification factor from clustering (actual H in hazard.py uses HazardParameters.kappa_clustering)
    hazard_amplification = 1.0 + 0.3 * clustering_index

    return {
        "province_unrest_means": province_unrest_means,
        "province_gdp_means": province_gdp_means,
        "province_admin_means": province_admin_means,
        "unrest_variance": unrest_variance,
        "clustering_index": clustering_index,
        "mean_district_stress": mean_stress,
        "top_unstable_province": top_unstable_province,
        "volatility_bump": volatility_bump,
        "hazard_amplification": hazard_amplification,
    }
