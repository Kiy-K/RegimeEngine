"""
TopologyVisualizer — Network graph visualization for GravitasWorld topology.

Bugs fixed vs original:
  1. ClusterState has no x/y position attributes. Positions are now derived
     from the adjacency matrix using spectral layout (networkx spring layout
     seeded for reproducibility), or passed explicitly.
  2. state.conflict_links does not exist. GravitasWorld exposes:
       world.adjacency  — (N,N) proximity/trade weight matrix
       world.conflict   — (N,N) conflict linkage matrix
     Both are now read directly as numpy arrays.
  3. Added cluster state annotation (stability σ, hazard h) as node color
     and labels, making the visualization actually useful for analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx
    _HAS_PLOT_LIBS = True
except ImportError:
    _HAS_PLOT_LIBS = False


def plot_topology(
    world: Any,
    ax: Optional[Any] = None,
    show_conflict: bool = True,
    show_proximity: bool = True,
    title: Optional[str] = None,
) -> Any:
    """
    Visualize GravitasWorld cluster topology as a network graph.

    Node color encodes cluster stability σ (green = stable, red = unstable).
    Node size encodes hazard h (larger = higher hazard).
    Green edges = proximity/trade links.
    Red edges   = conflict linkage (thicker = stronger conflict).

    Args:
        world:          A GravitasWorld instance (regime_engine.core.gravitas_state).
        ax:             Optional matplotlib Axes. Created if None.
        show_conflict:  Draw conflict edges (default True).
        show_proximity: Draw proximity edges (default True).
        title:          Plot title. Defaults to "Cluster Topology (Step N)".

    Returns:
        The matplotlib Axes object.
    """
    if not _HAS_PLOT_LIBS:
        raise ImportError(
            "matplotlib and networkx are required for topology visualization. "
            "Install with: pip install matplotlib networkx"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # ── Extract data from GravitasWorld ──────────────────────────────────── #
    # FIX 1 & 2: read from actual GravitasWorld attributes

    # clusters is a tuple of ClusterState objects
    clusters   = getattr(world, "clusters", [])
    N          = len(clusters)
    adjacency  = getattr(world, "adjacency",  np.zeros((N, N)))   # (N,N) trade weights
    conflict   = getattr(world, "conflict",   np.zeros((N, N)))   # (N,N) conflict weights
    step       = getattr(getattr(world, "global_state", None), "step", "?")

    # Per-cluster metrics for visual encoding
    sigmas  = np.array([getattr(c, "sigma",   0.5) for c in clusters])
    hazards = np.array([getattr(c, "hazard",  0.0) for c in clusters])
    trust   = np.array([getattr(c, "trust",   0.5) for c in clusters])

    # ── Build NetworkX graph ─────────────────────────────────────────────── #
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)

    proximity_edges: List[Tuple[int, int, float]] = []
    conflict_edges:  List[Tuple[int, int, float]] = []

    for i in range(N):
        for j in range(i + 1, N):
            w_prox = float(adjacency[i, j])
            w_conf = float(conflict[i, j]) + float(conflict[j, i])
            if show_proximity and w_prox > 0.01:
                proximity_edges.append((i, j, w_prox))
                G.add_edge(i, j, weight=w_prox, etype="proximity")
            if show_conflict and w_conf > 0.01:
                conflict_edges.append((i, j, w_conf))

    # ── Layout: spring layout seeded for reproducibility ─────────────────── #
    # FIX 1: no x/y on ClusterState — derive layout from graph structure
    if len(G.edges) > 0:
        pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5)
    else:
        # Fallback: circular if no edges
        pos = nx.circular_layout(G)

    # ── Node visuals ──────────────────────────────────────────────────────── #
    # Color by stability (green=stable, red=unstable)
    node_colors = [cm.RdYlGn(s) for s in sigmas]
    # Size by hazard (min 300, max 2000)
    node_sizes  = [300 + 1700 * float(np.clip(h, 0.0, 1.0)) for h in hazards]
    labels      = {i: f"{i}\nσ={sigmas[i]:.2f}\nh={hazards[i]:.2f}" for i in range(N)}

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    # ── Proximity edges (green, thin) ─────────────────────────────────────── #
    if show_proximity and proximity_edges:
        prox_edgelist = [(i, j) for i, j, _ in proximity_edges]
        prox_weights  = [w * 3 for _, _, w in proximity_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=prox_edgelist, width=prox_weights,
            edge_color="green", alpha=0.4, ax=ax, style="dashed",
        )

    # ── Conflict edges (red, thick) ────────────────────────────────────────── #
    if show_conflict and conflict_edges:
        conf_edgelist = [(i, j) for i, j, _ in conflict_edges]
        conf_weights  = [w * 5 for _, _, w in conflict_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=conf_edgelist, width=conf_weights,
            edge_color="red", alpha=0.7, ax=ax,
        )

    # ── Labels and legend ─────────────────────────────────────────────────── #
    _title = title or f"Cluster Topology (Step {step})"
    ax.set_title(_title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Simple legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="High stability (σ→1)"),
        Patch(facecolor="red",   alpha=0.7, label="Low stability  (σ→0)"),
        Line2D([0], [0], color="green", linestyle="dashed", label="Proximity edge"),
        Line2D([0], [0], color="red",   linewidth=3,        label="Conflict edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    return ax


def plot_global_timeseries(
    world_history: List[Any],
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot global state variables over time from a list of GravitasWorld snapshots.

    Args:
        world_history: List of GravitasWorld objects, one per step.
        ax:            Optional matplotlib Axes.

    Returns:
        The matplotlib Axes object.
    """
    if not _HAS_PLOT_LIBS:
        raise ImportError("matplotlib required for time-series visualization.")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    steps        = []
    exhaustions  = []
    polarizations = []
    fragmentations = []
    coherences   = []

    for world in world_history:
        gs = getattr(world, "global_state", None)
        if gs is None:
            continue
        steps.append(getattr(gs, "step", len(steps)))
        exhaustions.append(getattr(gs, "exhaustion",    0.0))
        polarizations.append(getattr(gs, "polarization", 0.0))
        fragmentations.append(getattr(gs, "fragmentation", 0.0))
        coherences.append(getattr(gs, "coherence",     1.0))

    ax.plot(steps, exhaustions,   label="Exhaustion (E)",      color="orange", linewidth=2)
    ax.plot(steps, polarizations, label="Polarization (Π)",    color="red",    linewidth=2)
    ax.plot(steps, fragmentations,label="Fragmentation (Φ)",   color="purple", linewidth=2)
    ax.plot(steps, coherences,    label="Coherence (Ψ)",       color="blue",   linewidth=2)
    ax.axhline(0.6, color="orange", linestyle="--", alpha=0.5, label="Exhaustion threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value [0,1]")
    ax.set_title("Global State Variables Over Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    return ax
