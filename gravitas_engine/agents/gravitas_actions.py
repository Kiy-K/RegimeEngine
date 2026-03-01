"""
GravitasActions — Hierarchical action space for GRAVITAS.

Level 1: Strategic stance (discrete × 7)
Level 2: Allocation (continuous)
  - cluster weights w ∈ Δ^N  (simplex)
  - intensity θ ∈ [0,1]
  - long-term bias γ ∈ [0,1]  (0=short-term, 1=long-term)

For flat RL (SB3 PPO), we encode as a flat Discrete action space:
  action = stance_idx  (0–6)
  allocation is determined by a fixed heuristic (highest hazard clusters)

For HRL, the manager picks stance; the worker picks (w, θ, γ).

Each stance applies immediate effects then sets ODE modifier flags
that shape the dynamics for the current step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams
from ..core.gravitas_state import ClusterState, GlobalState, GravitasWorld


# ─────────────────────────────────────────────────────────────────────────── #
# Stance definitions                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@unique
class Stance(IntEnum):
    STABILIZE    = 0
    MILITARIZE   = 1
    REFORM       = 2
    PROPAGANDA   = 3
    INVEST       = 4
    DECENTRALIZE = 5
    DIPLOMACY    = 6
    REDEPLOY     = 7  # inter-cluster unit transfer along adjacency edges


N_STANCES = len(Stance)

# Resource costs per stance per step (fraction of available resource pool)
RESOURCE_COSTS: Dict[int, float] = {
    Stance.STABILIZE:    0.10,
    Stance.MILITARIZE:   0.05,
    Stance.REFORM:       0.08,
    Stance.PROPAGANDA:   0.03,
    Stance.INVEST:       0.12,
    Stance.DECENTRALIZE: 0.06,
    Stance.DIPLOMACY:    0.04,
    Stance.REDEPLOY:     0.02,   # cheap: moving troops, not deploying new ones
}


@dataclass
class HierarchicalAction:
    """
    Decoded hierarchical action.

    stance           — strategic stance (Stance enum)
    weights          — (N,) cluster allocation weights (simplex)
    intensity        — θ ∈ [0,1]
    lt_bias          — γ ∈ [0,1]: 0 = short-term, 1 = long-term
    propaganda_load  — propaganda intensity (0 if not PROPAGANDA)
    military_load    — military intensity (0 if not MILITARIZE)
    resource_cost    — total resources consumed
    """
    stance:          Stance
    weights:         NDArray[np.float64]
    intensity:       float
    lt_bias:         float
    propaganda_load: float
    military_load:   float
    resource_cost:   float

    def to_array(self) -> NDArray[np.float64]:
        """Flat representation for prev_action embedding."""
        return np.concatenate([
            np.array([float(self.stance) / (N_STANCES - 1)]),
            self.weights,
            np.array([self.intensity, self.lt_bias]),
        ])


# ─────────────────────────────────────────────────────────────────────────── #
# Flat → hierarchical decode                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def decode_flat_action(
    action_int: int,
    world: GravitasWorld,
    params: GravitasParams,
    rng: np.random.Generator,
) -> HierarchicalAction:
    """
    Decode a flat integer action (0–5) into a HierarchicalAction.

    Allocation heuristic:
      - MILITARIZE  → weight toward highest-hazard clusters
      - REFORM      → weight toward lowest-trust clusters
      - INVEST      → weight toward lowest-resource clusters
      - STABILIZE   → weight toward lowest-stability clusters
      - PROPAGANDA  → weight toward highest-bias-magnitude clusters
      - DECENTRALIZE → weight toward highest-polarization clusters
    """
    stance    = Stance(action_int % N_STANCES)
    N         = world.n_clusters
    max_N     = params.n_clusters_max   # fixed-size weight vector for embedding
    c_arr     = world.cluster_array()

    # Compute target scores for allocation
    if stance == Stance.MILITARIZE:
        scores = c_arr[:, 1]                         # hazard
    elif stance == Stance.REFORM:
        scores = 1.0 - c_arr[:, 4]                  # low trust
    elif stance == Stance.INVEST:
        scores = 1.0 - c_arr[:, 2]                  # low resource
    elif stance == Stance.STABILIZE:
        scores = 1.0 - c_arr[:, 0]                  # low stability
    elif stance == Stance.PROPAGANDA:
        scores = np.abs(world.media_bias)            # high bias magnitude
    elif stance == Stance.DECENTRALIZE:
        scores = c_arr[:, 5]                         # high polarization
    elif stance == Stance.DIPLOMACY:
        # Weight toward low-trust, low-stability clusters (diplomatic priority)
        scores = (1.0 - c_arr[:, 4]) + (1.0 - c_arr[:, 0])
    else:  # REDEPLOY
        # Weight toward highest-hazard clusters (move troops to danger zones)
        scores = c_arr[:, 1]   # hazard

    # Softmax over scores for soft allocation
    scores    = np.clip(scores, 0.0, 5.0)
    exp_s     = np.exp(scores - scores.max())
    w_raw     = exp_s / exp_s.sum()

    # Always pad weights to n_clusters_max so prev_action embedding is fixed-size
    weights = np.zeros(max_N)
    weights[:N] = w_raw

    intensity = 0.7   # fixed for flat mode; worker would tune this
    lt_bias   = 0.5

    prop_load = intensity if stance == Stance.PROPAGANDA else 0.0
    mil_load  = float(np.mean(c_arr[:, 3]))

    r_cost = RESOURCE_COSTS[stance] * intensity

    return HierarchicalAction(
        stance=stance,
        weights=weights,
        intensity=intensity,
        lt_bias=lt_bias,
        propaganda_load=prop_load,
        military_load=mil_load,
        resource_cost=r_cost,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Action application                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_action(
    world: GravitasWorld,
    action: HierarchicalAction,
    params: GravitasParams,
    rng: np.random.Generator,
) -> Tuple[GravitasWorld, float]:
    """
    Apply immediate action effects to the world state.
    Returns (new_world, actual_resource_cost).

    Long-term costs (exhaustion, polarization accumulation) are handled in
    the ODE dynamics via military_load and propaganda_load passed to rk4_step.
    This function handles only the instantaneous impulses.
    """
    N        = world.n_clusters
    clusters = list(world.clusters)
    g        = world.global_state
    w        = action.weights       # (N,)
    θ        = action.intensity
    E        = g.exhaustion
    act      = 1.0 - E             # activity gate

    stance = action.stance

    if stance == Stance.STABILIZE:
        # Boost stability in weighted clusters; costs resources
        for i in range(N):
            c = clusters[i]
            delta = params.alpha_sigma * 2.0 * w[i] * θ * act
            clusters[i] = c.copy_with(
                sigma=float(np.clip(c.sigma + delta, 0.0, 1.0)),
            )

    elif stance == Stance.MILITARIZE:
        # Immediate hazard reduction + military presence; long-term: ODEs handle costs
        cap = params.max_military_total
        total_deployed = sum(c.military for c in clusters)
        for i in range(N):
            c = clusters[i]
            # Can't exceed global military capacity
            headroom = min(cap - total_deployed, w[i] * θ * g.military_str)
            headroom = max(0.0, headroom)
            new_mil  = float(np.clip(c.military + headroom, 0.0, 1.0))
            total_deployed += new_mil - c.military  # keep running total current
            # Immediate hazard suppression
            new_h    = float(np.clip(
                c.hazard - params.military_hazard_reduction * headroom * act,
                0.0, 5.0
            ))
            new_sig  = float(np.clip(
                c.sigma + params.military_sigma_boost * headroom * act,
                0.0, 1.0
            ))
            clusters[i] = c.copy_with(hazard=new_h, sigma=new_sig, military=new_mil)

    elif stance == Stance.REFORM:
        # Gradual trust boost; no immediate hazard effect
        for i in range(N):
            c = clusters[i]
            delta = params.alpha_tau * w[i] * θ * (1.0 - g.polarization) * act
            clusters[i] = c.copy_with(
                trust=float(np.clip(c.trust + delta, 0.0, 1.0))
            )

    elif stance == Stance.PROPAGANDA:
        # Shifts perceived stability (handled by media_bias update)
        # Direct effect: small trust boost (short-term), true polarization rises (in ODE)
        for i in range(N):
            c = clusters[i]
            # Small legitimacy boost from state communication
            clusters[i] = c.copy_with(
                trust=float(np.clip(c.trust + 0.01 * w[i] * θ, 0.0, 1.0))
            )

    elif stance == Stance.INVEST:
        # Resource injection; compounding if trust is high
        for i in range(N):
            c = clusters[i]
            multiplier = 1.0 + 0.5 * c.trust  # trust multiplier
            delta = params.invest_res_boost * w[i] * θ * multiplier * act
            clusters[i] = c.copy_with(
                resource=float(np.clip(c.resource + delta, 0.0, 1.0))
            )

    elif stance == Stance.DECENTRALIZE:
        # Reduces fragmentation + polarization long-term; immediate stability cost
        g = g.copy_with(
            fragmentation=float(np.clip(
                g.fragmentation - 0.05 * θ * act * (1.0 - g.polarization),
                0.0, 0.999
            )),
            polarization=float(np.clip(
                g.polarization - 0.03 * θ * act,
                0.0, 1.0
            )),
        )
        # Short-term stability dip (restructuring is disruptive)
        for i in range(N):
            c = clusters[i]
            clusters[i] = c.copy_with(
                sigma=float(np.clip(c.sigma - 0.02 * w[i] * θ, 0.0, 1.0))
            )

    elif stance == Stance.DIPLOMACY:
        # Strengthen bilateral alliances for clusters selected by weights.
        # Immediate effect: small trust boost in cooperative clusters.
        if world.alliance is not None:
            from ..systems.diplomacy import apply_diplomacy_action
            new_alliance = apply_diplomacy_action(
                world.alliance, w, θ, params, N
            )
        else:
            new_alliance = None

        for i in range(N):
            c = clusters[i]
            if new_alliance is not None:
                # Trust bonus proportional to sum of positive new alliances
                net_ally = float(np.sum(np.maximum(0.0, new_alliance[i, :N])))
                delta_trust = 0.01 * w[i] * θ * net_ally * act
                clusters[i] = c.copy_with(
                    trust=float(np.clip(c.trust + delta_trust, 0.0, 1.0))
                )

        new_world = world.copy_with_clusters(clusters).copy_with_global(g)
        if new_alliance is not None:
            new_world = new_world.copy_with_alliance(new_alliance)
        return new_world, action.resource_cost

    elif stance == Stance.REDEPLOY:
        # Transfer military presence along adjacency edges.
        # Clusters with w[i] < mean_w are DONORS; w[i] > mean_w are RECEIVERS.
        # Transfer is proportional to edge weight and donor surplus.
        # Transit cost: small stability dip in donor cluster.
        A    = world.adjacency          # (N, N) — normalized edge weights
        mean_w = float(w[:N].mean())
        transfer_rate = 0.15 * θ * act  # max fraction moved per step

        donor_surplus   = np.maximum(0.0, mean_w - w[:N])   # donor has low weight
        receiver_demand = np.maximum(0.0, w[:N] - mean_w)   # receiver has high weight

        # For each donor→receiver adjacent pair, compute transfer
        delta_mil = np.zeros(N, dtype=np.float64)
        for i in range(N):   # donor
            if donor_surplus[i] < 1e-6:
                continue
            src_mil = clusters[i].military
            for j in range(N):  # receiver
                if i == j or A[i, j] < 1e-6:
                    continue
                if receiver_demand[j] < 1e-6:
                    continue
                # Amount to transfer: proportional to edge weight and demands
                amount = transfer_rate * A[i, j] * donor_surplus[i] * src_mil
                amount = float(np.clip(amount, 0.0, src_mil * 0.25))  # max 25% per donor
                delta_mil[i] -= amount
                delta_mil[j] += amount

        cap           = params.max_military_total
        total_current = sum(c.military for c in clusters)
        total_add     = sum(max(0.0, d) for d in delta_mil)
        scale         = min(1.0, max(0.0, (cap - (total_current - sum(max(0.0, -d) for d in delta_mil))) / max(total_add, 1e-9)))

        for i in range(N):
            c = clusters[i]
            new_mil = float(np.clip(c.military + scale * delta_mil[i], 0.0, 1.0))
            # Donor clusters: small stability dip (troops leaving creates void)
            sigma_delta = -0.01 * max(0.0, -delta_mil[i]) if delta_mil[i] < 0 else 0.0
            clusters[i] = c.copy_with(
                military=new_mil,
                sigma=float(np.clip(c.sigma + sigma_delta, 0.0, 1.0)),
            )

    new_world = world.copy_with_clusters(clusters).copy_with_global(g)
    return new_world, action.resource_cost
