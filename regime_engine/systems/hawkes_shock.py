"""
HawkesShocks — Self-exciting shock process for GRAVITAS.

Shock arrival follows a Hawkes process:
    λ(t) = λ₀ + Σ_{t_k < t} α · exp(-β · (t - t_k))

Past shocks raise the probability of future shocks.
Shock magnitude is Pareto-distributed (heavy tail — rare but catastrophic).

Five shock types with distinct propagation signatures:
  ECONOMIC            — resource drain, GDP cascade, memory spike
  MILITARY_COUP       — trust collapse in target cluster
  INFORMATION_WAR     — bias jump + coherence drop
  EXTERNAL_PRESSURE   — global polarization surge
  NATURAL_DISASTER    — resource destruction, localized hazard spike
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import ClusterState, GlobalState, GravitasWorld


# ─────────────────────────────────────────────────────────────────────────── #
# Shock type                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@unique
class ShockType(IntEnum):
    ECONOMIC           = 0
    MILITARY_COUP      = 1
    INFORMATION_WAR    = 2
    EXTERNAL_PRESSURE  = 3
    NATURAL_DISASTER   = 4


@dataclass
class ShockEvent:
    shock_type:    ShockType
    magnitude:     float       # Pareto-drawn, in [x_min, ∞)
    cluster_idx:   int         # primary affected cluster (-1 = global)
    step:          int


# ─────────────────────────────────────────────────────────────────────────── #
# Hawkes intensity update                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def update_hawkes(
    hawkes_sum: float,
    shock_occurred: bool,
    params: GravitasParams,
) -> Tuple[float, float]:
    """
    One-step Hawkes update.

    After each step (whether or not a shock occurred):
        new_sum = hawkes_sum · exp(-β · dt) + shock_occurred · 1.0
    λ(t) = λ₀ + α · new_sum

    Returns (new_shock_rate, new_hawkes_sum).
    """
    decay    = np.exp(-params.hawkes_beta * params.dt)
    new_sum  = hawkes_sum * decay + (1.0 if shock_occurred else 0.0)
    new_rate = params.hawkes_base_rate + params.hawkes_alpha * new_sum
    return float(np.clip(new_rate, 0.0, 1.0)), float(new_sum)


# ─────────────────────────────────────────────────────────────────────────── #
# Shock arrival check                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def sample_shock(
    world: GravitasWorld,
    params: GravitasParams,
    rng: np.random.Generator,
) -> Optional[ShockEvent]:
    """
    Check whether a shock arrives this step and sample it if so.
    Returns a ShockEvent or None.
    """
    if rng.random() >= world.shock_rate * params.dt:
        return None

    # Magnitude: Pareto(α, x_min)
    u = rng.random()
    mag = params.shock_pareto_xmin * (1.0 - u) ** (-1.0 / params.shock_pareto_alpha)
    mag = float(np.clip(mag, params.shock_pareto_xmin, 3.0))

    shock_type  = ShockType(rng.integers(0, len(ShockType)))
    cluster_idx = int(rng.integers(0, world.n_clusters))

    return ShockEvent(
        shock_type=shock_type,
        magnitude=mag,
        cluster_idx=cluster_idx,
        step=world.global_state.step,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Shock application                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_shock(
    world: GravitasWorld,
    shock: ShockEvent,
    params: GravitasParams,
    rng: np.random.Generator,
) -> Tuple[GravitasWorld, Dict[str, Any]]:
    """
    Apply a shock event to the world state.
    Returns (new_world, info_dict).
    Each shock type has a distinct signature; no shock is purely local.
    """
    N   = world.n_clusters
    mag = shock.magnitude
    idx = shock.cluster_idx
    clusters = list(world.clusters)
    g = world.global_state
    info: Dict[str, Any] = {"type": shock.shock_type.name, "magnitude": mag,
                             "cluster": idx}

    if shock.shock_type == ShockType.ECONOMIC:
        # Resource drain in affected cluster + neighbours, memory spike
        c = clusters[idx]
        clusters[idx] = c.copy_with(
            resource=float(np.clip(c.resource - 0.3 * mag, 0.0, 1.0)),
            sigma=float(np.clip(c.sigma - 0.1 * mag, 0.0, 1.0)),
        )
        # Propagate resource drain to conflict neighbours
        for j in range(N):
            if j == idx: continue
            w = world.conflict[idx, j]
            if w > 0:
                cj = clusters[j]
                clusters[j] = cj.copy_with(
                    resource=float(np.clip(cj.resource - 0.15 * mag * w, 0.0, 1.0))
                )
        g = g.copy_with(
            exhaustion=float(np.clip(g.exhaustion + 0.04 * mag, 0.0, 1.0))
        )

    elif shock.shock_type == ShockType.MILITARY_COUP:
        # Trust collapses in target cluster; fragmentation spikes
        c = clusters[idx]
        clusters[idx] = c.copy_with(
            trust=float(np.clip(c.trust - 0.5 * mag, 0.0, 1.0)),
            polar=float(np.clip(c.polar + 0.3 * mag, 0.0, 1.0)),
        )
        g = g.copy_with(
            fragmentation=float(np.clip(g.fragmentation + 0.10 * mag, 0.0, 0.999)),
            polarization=float(np.clip(g.polarization + 0.08 * mag, 0.0, 1.0)),
        )

    elif shock.shock_type == ShockType.INFORMATION_WAR:
        # Coherence drops; bias vector gets a large jump
        g = g.copy_with(
            coherence=float(np.clip(g.coherence - params.psi_shock_cost * mag, 0.0, 1.0)),
            polarization=float(np.clip(g.polarization + 0.06 * mag, 0.0, 1.0)),
        )
        # Bias jump: random direction per cluster, scaled by mag
        bias_jump = rng.standard_normal(N) * 0.20 * mag
        new_bias = np.clip(world.media_bias + bias_jump, -params.beta_max_bias, params.beta_max_bias)
        world = world.copy_with_bias(new_bias)
        info["bias_jump"] = bias_jump.tolist()

    elif shock.shock_type == ShockType.EXTERNAL_PRESSURE:
        # Global polarization surge + exhaustion
        g = g.copy_with(
            polarization=float(np.clip(g.polarization + 0.15 * mag, 0.0, 1.0)),
            exhaustion=float(np.clip(g.exhaustion + 0.06 * mag, 0.0, 1.0)),
            fragmentation=float(np.clip(g.fragmentation + 0.05 * mag, 0.0, 0.999)),
        )
        # Each cluster gets a small polarization bump
        for i in range(N):
            c = clusters[i]
            clusters[i] = c.copy_with(
                polar=float(np.clip(c.polar + 0.10 * mag * rng.random(), 0.0, 1.0))
            )

    elif shock.shock_type == ShockType.NATURAL_DISASTER:
        # Localized: heavy resource + stability hit; spreads via proximity A
        c = clusters[idx]
        clusters[idx] = c.copy_with(
            resource=float(np.clip(c.resource - 0.4 * mag, 0.0, 1.0)),
            sigma=float(np.clip(c.sigma - 0.3 * mag, 0.0, 1.0)),
        )
        for j in range(N):
            if j == idx: continue
            w = world.adjacency[idx, j]
            if w > 0:
                cj = clusters[j]
                clusters[j] = cj.copy_with(
                    resource=float(np.clip(cj.resource - 0.1 * mag * w, 0.0, 1.0))
                )

    new_world = world.copy_with_clusters(clusters).copy_with_global(g)
    return new_world, info
