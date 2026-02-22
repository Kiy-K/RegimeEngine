"""
RL observation builders for hierarchical (Phase 3) regime state.

Provides aggregated observation vector for PPO and optional district summary
statistics (mean, variance, top-k unstable) without blowing up observation dimension.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .hierarchical_state import HierarchicalState, N_DISTRICT_VARS

# Indices into district array
I_GDP, I_UNREST, I_ADMIN, I_FRAG, I_IMPL, I_MEM = 0, 1, 2, 3, 4, 5


def aggregated_district_observation(
    hierarchical: HierarchicalState,
    include_var: bool = True,
) -> NDArray[np.float64]:
    """Fixed-size aggregated district vector for PPO.

    Returns: [mean_gdp, mean_unrest, mean_admin, mean_frag, mean_impl, mean_mem]
             + [var_gdp, var_unrest, var_admin, var_frag, var_impl, var_mem] if include_var.
    Length: 6 or 12.
    """
    arr = hierarchical.get_district_array()
    if arr.size == 0:
        return np.zeros(12 if include_var else 6, dtype=np.float64)
    means = np.mean(arr, axis=0)
    if not include_var:
        return means
    vars_ = np.var(arr, axis=0)
    return np.concatenate([means, vars_])


def top_k_unstable_districts(
    hierarchical: HierarchicalState,
    k: int = 5,
) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Indices and local_unrest values of the k districts with highest unrest.

    Returns:
        (indices shape (k,), values shape (k,)). If n_districts < k, padded with -1 and 0.
    """
    if hierarchical.n_districts == 0:
        return np.full(k, -1, dtype=np.intp), np.zeros(k, dtype=np.float64)
    u = hierarchical.get_district_array()[:, I_UNREST]
    n = len(u)
    k_use = min(k, n)
    top_idx = np.argsort(u)[::-1][:k_use]
    top_vals = u[top_idx]
    out_idx = np.full(k, -1, dtype=np.intp)
    out_vals = np.zeros(k, dtype=np.float64)
    out_idx[:k_use] = top_idx
    out_vals[:k_use] = top_vals
    return out_idx, out_vals


def hierarchical_observation_vector(
    hierarchical: Optional[HierarchicalState],
    include_summary: bool = True,
    include_top_k: bool = False,
    top_k: int = 5,
) -> NDArray[np.float64]:
    """Single flat observation vector from hierarchical state.

    - If hierarchical is None: returns empty array shape (0,).
    - Otherwise: [aggregated_means_and_vars (12)] and optionally [top_k values (top_k)].
    Avoids full N_D*6 to keep dimension moderate.

    Args:
        hierarchical: Phase 3 state or None.
        include_summary: Include mean and variance of all 6 district vars (length 12).
        include_top_k: Append unrest values of top-k unstable districts (length top_k).
        top_k: Number of top unstable districts to include.

    Returns:
        Float64 vector.
    """
    if hierarchical is None:
        return np.zeros(0, dtype=np.float64)
    parts = []
    if include_summary:
        parts.append(aggregated_district_observation(hierarchical, include_var=True))
    if include_top_k:
        _, top_vals = top_k_unstable_districts(hierarchical, k=top_k)
        parts.append(top_vals)
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def observation_space_dim_hierarchical(
    include_summary: bool = True,
    include_top_k: bool = False,
    top_k: int = 5,
) -> int:
    """Dimension of the hierarchical part of the observation space."""
    dim = 0
    if include_summary:
        dim += 12
    if include_top_k:
        dim += top_k
    return dim
