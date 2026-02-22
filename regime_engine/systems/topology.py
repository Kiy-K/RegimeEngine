"""
TopologyBuilder — Randomized cluster topology for GRAVITAS.

Each episode draws a fresh topology with:
  - adjacency  A_ij: proximity / trade weights (symmetric, in [0,1])
  - conflict   C_ij: conflict linkage (subset of A, asymmetric in degree)

Topology properties that prevent trivial exploitation:
  - Number of clusters is random per episode (n_min to n_max)
  - Edge structure changes every episode (no fixed graph to memorize)
  - Bridge clusters emerge naturally and create cascade choke-points
  - Conflict linkage is sparse but creates high-leverage instability paths
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams


def build_topology(
    params: GravitasParams,
    rng: np.random.Generator,
) -> Tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """
    Build randomized episode topology.

    Returns:
        n_clusters  — number of clusters drawn for this episode
        adjacency   — (N, N) symmetric weight matrix (0 diagonal)
        conflict    — (N, N) conflict linkage matrix (0 diagonal)
    """
    # Sample N for this episode
    N = int(rng.integers(params.n_clusters_min, params.n_clusters_max + 1))

    # ── Proximity/trade adjacency (Erdős–Rényi + spatial decay) ─────────── #
    # Place clusters in 2D unit square; weight by proximity
    positions = rng.uniform(0.0, 1.0, (N, 2))
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    # Edge exists if random < link_prob; weight by spatial proximity
    edge_mask = rng.random((N, N)) < params.between_link_prob
    np.fill_diagonal(edge_mask, False)
    # Symmetrize
    edge_mask = edge_mask | edge_mask.T

    # Weight inversely proportional to distance (nearby = stronger link)
    raw_weights = np.exp(-3.0 * dist) * edge_mask.astype(float)
    # Normalize rows to sum ≤ 1 (diffusion stability)
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    adjacency = raw_weights / row_sums
    np.fill_diagonal(adjacency, 0.0)

    # Ensure graph connectivity: add minimum spanning edges if disconnected
    adjacency = _ensure_connected(adjacency, edge_mask, N, rng)

    # ── Conflict linkage (sparse subset of adjacency) ────────────────────── #
    # Conflict edges are a subset of existing proximity edges
    conflict_mask = (edge_mask) & (rng.random((N, N)) < params.conflict_link_prob)
    np.fill_diagonal(conflict_mask, False)
    # Conflict is directional — asymmetric in magnitude
    conflict_raw  = rng.uniform(0.2, 0.8, (N, N)) * conflict_mask.astype(float)
    # Normalize so cascade term stays bounded
    c_row_sums = conflict_raw.sum(axis=1, keepdims=True)
    c_row_sums = np.where(c_row_sums == 0, 1.0, c_row_sums)
    conflict   = conflict_raw / c_row_sums * conflict_mask.astype(float)
    np.fill_diagonal(conflict, 0.0)

    return N, adjacency.astype(np.float64), conflict.astype(np.float64)


def _ensure_connected(
    adjacency: NDArray[np.float64],
    edge_mask: NDArray[np.bool_],
    N: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Ensure the graph is connected by adding edges where needed.
    Uses BFS to find disconnected components and bridges them.
    """
    # BFS to find connected components
    visited = np.zeros(N, dtype=bool)
    components = []

    for start in range(N):
        if visited[start]:
            continue
        comp = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            for nb in range(N):
                if not visited[nb] and edge_mask[node, nb]:
                    queue.append(nb)
        components.append(comp)

    # If already connected, return as-is
    if len(components) == 1:
        return adjacency

    # Bridge components by adding weak edges
    A = adjacency.copy()
    for ci in range(len(components) - 1):
        # Pick one node from each adjacent pair of components
        src = rng.choice(components[ci])
        tgt = rng.choice(components[ci + 1])
        weight = float(rng.uniform(0.05, 0.20))
        A[src, tgt] = weight
        A[tgt, src] = weight

    # Re-normalize rows
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    A = A / row_sums
    np.fill_diagonal(A, 0.0)
    return A


def initialize_cluster_states(
    N: int,
    params: GravitasParams,
    rng: np.random.Generator,
) -> list:
    """
    Initialize cluster states for episode start.
    States are randomized but biased toward moderate instability
    (not random chaos, not perfect calm — the agent needs something to manage).
    """
    from ..core.gravitas_state import ClusterState

    clusters = []
    for _ in range(N):
        sigma    = float(rng.uniform(0.45, 0.80))   # moderate stability
        resource = float(rng.uniform(0.40, 0.85))
        trust    = float(rng.uniform(0.35, 0.75))
        polar    = float(rng.uniform(0.10, 0.45))
        military = float(rng.uniform(0.00, 0.15))   # low initial deployment
        # Hazard derived from state (set to 0 here; dynamics.compute_hazard called first step)
        hazard   = float(rng.uniform(0.05, 0.30))
        clusters.append(ClusterState(
            sigma=sigma, hazard=hazard, resource=resource,
            military=military, trust=trust, polar=polar,
        ))
    return clusters


def initialize_global_state(
    params: GravitasParams,
    rng: np.random.Generator,
) -> object:
    """Initialize GlobalState for episode start."""
    from ..core.gravitas_state import GlobalState
    return GlobalState(
        exhaustion=float(rng.uniform(0.05, 0.20)),
        fragmentation=float(rng.uniform(0.05, 0.20)),
        polarization=float(rng.uniform(0.10, 0.35)),
        coherence=float(rng.uniform(0.60, 0.90)),
        military_str=float(rng.uniform(0.50, 0.90)),
        trust=float(rng.uniform(0.40, 0.70)),
        step=0,
    )
