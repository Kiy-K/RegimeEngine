"""
Spatial topology for the hierarchical regime engine (Phase 3).

Builds province and district layout and the district adjacency matrix A
for diffusion (unrest, capital flow). A is symmetric, non-negative, zero diagonal.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def build_province_district_layout(
    n_provinces: int,
    districts_per_province: int | List[int],
) -> Tuple[List[int], int]:
    """Return list of district counts per province and total districts.

    Args:
        n_provinces: Number of provinces (5–10).
        districts_per_province: Either a single int (same for all) or a list
            of length n_provinces with counts in [5, 20] each.

    Returns:
        (counts_per_province, N_D) where N_D = total number of districts.
    """
    if not 5 <= n_provinces <= 10:
        raise ValueError(f"n_provinces must be in [5, 10], got {n_provinces}")
    if isinstance(districts_per_province, int):
        if not 5 <= districts_per_province <= 20:
            raise ValueError(
                f"districts_per_province must be in [5, 20], got {districts_per_province}"
            )
        counts = [districts_per_province] * n_provinces
    else:
        if len(districts_per_province) != n_provinces:
            raise ValueError(
                f"districts_per_province length must be {n_provinces}, got {len(districts_per_province)}"
            )
        counts = []
        for i, c in enumerate(districts_per_province):
            if not 5 <= c <= 20:
                raise ValueError(
                    f"districts_per_province[{i}] must be in [5, 20], got {c}"
                )
            counts.append(int(c))
    n_districts = sum(counts)
    return counts, n_districts


def district_to_province(counts_per_province: List[int]) -> NDArray[np.intp]:
    """Return array of length N_D: district index -> province index (0-based)."""
    n_d = sum(counts_per_province)
    out = np.empty(n_d, dtype=np.intp)
    idx = 0
    for p, c in enumerate(counts_per_province):
        out[idx : idx + c] = p
        idx += c
    return out


def build_adjacency_matrix(
    counts_per_province: List[int],
    connect_within_province: bool = True,
    connect_between_provinces: bool = False,
    between_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Build district adjacency matrix A (N_D x N_D).

    - Symmetric, A_ii = 0, A_ij >= 0.
    - If connect_within_province: districts in the same province are connected
      (e.g. 1 if adjacent in a ring or all-to-all within province).
    - If connect_between_provinces and between_weight > 0: add weak links
      between provinces (e.g. one link per province pair or border districts).

    Default: within-province ring (each district connected to 2 neighbors)
    plus optional all-to-all within province with weight 1. Here we use
    all-to-all within province (weight 1) so that diffusion is well-defined.

    Returns:
        A: (N_D, N_D) float64, symmetric, non-negative, zero diagonal.
    """
    n_d = sum(counts_per_province)
    A = np.zeros((n_d, n_d), dtype=np.float64)
    idx = 0
    for p, c in enumerate(counts_per_province):
        # Within province: all-to-all (excluding self)
        if connect_within_province and c > 1:
            for i in range(idx, idx + c):
                for j in range(idx, idx + c):
                    if i != j:
                        A[i, j] = 1.0
        idx += c

    if connect_between_provinces and between_weight > 0 and len(counts_per_province) > 1:
        # Connect "border" districts: last district of each province to first of next
        idx = 0
        for p in range(len(counts_per_province) - 1):
            n_cur = counts_per_province[p]
            n_next = counts_per_province[p + 1]
            last_in_p = idx + n_cur - 1
            first_next = idx + n_cur
            A[last_in_p, first_next] += between_weight
            A[first_next, last_in_p] += between_weight
            idx += n_cur

    # Ensure symmetry and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def diffusion_rate_bound(adjacency: NDArray[np.float64]) -> float:
    """Max diffusion rate for stability: 1 / (2 * max_i sum_j A_ij)."""
    row_sums = np.sum(adjacency, axis=1)
    max_deg = float(np.max(row_sums))
    if max_deg <= 0:
        return 1.0
    return 0.5 / max_deg


def build_randomized_adjacency(
    counts_per_province: List[int],
    rng: np.random.Generator,
    connect_within: bool = True,
    between_prob: float = 0.4,
    between_weight: float = 0.5,
) -> NDArray[np.float64]:
    """Build adjacency with random between-province links so topology varies per episode.

    Within-province: all-to-all as in build_adjacency_matrix.
    Between-province: each pair of border districts (last of province p, first of p+1)
    is connected with probability between_prob; weight between_weight.
    Additional random cross-links can be added for more heterogeneity.

    Agents must not memorize adjacency; they must learn spatial principles.
    """
    n_d = sum(counts_per_province)
    A = np.zeros((n_d, n_d), dtype=np.float64)
    idx = 0
    for p, c in enumerate(counts_per_province):
        if connect_within and c > 1:
            for i in range(idx, idx + c):
                for j in range(idx, idx + c):
                    if i != j:
                        A[i, j] = 1.0
        idx += c

    # Random between-province: border links with probability between_prob
    n_p = len(counts_per_province)
    idx = 0
    for p in range(n_p - 1):
        n_cur = counts_per_province[p]
        first_next = idx + n_cur
        last_in_p = first_next - 1
        if rng.random() < between_prob:
            A[last_in_p, first_next] = between_weight
            A[first_next, last_in_p] = between_weight
        idx += n_cur

    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A
