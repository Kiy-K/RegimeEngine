"""
political_interface.py — Bidirectional military ↔ politics feedback for GRAVITAS.

This module is the connective tissue between the military extension and the
political core engine.  Politics determines everything; this module enforces it.

╔══════════════════════════════════════════════════════════════════════════════╗
║  MILITARY → POLITICS                                                         ║
║  Each military unit present in a cluster exerts a per-step political impulse ║
║  drawn from unit_types.POLITICAL_EFFECTS.  Combat units suppress hazard      ║
║  proportionally to their presence; losses spike hazard and erode trust.      ║
║  Encircled clusters (all adjacent edges held by hostile forces) suffer a     ║
║  supply blockade: resource declines and hazard climbs.                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  POLITICS → MILITARY                                                         ║
║  Low cluster trust drains unit morale each step.                             ║
║  High polarization reduces the effective reinforcement pool.                 ║
║  High hazard raises unit supply consumption.                                 ║
║  Cluster resource level sets the supply-depot replenishment rate.            ║
║  Alliance strength between clusters enables joint-supply sharing.            ║
║  Population provides the manpower ceiling for new unit deployments.          ║
║  Media bias increases fog-of-war for intelligence units.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .unit_types import MilitaryUnitType, UnitRole, get_unit_role, POLITICAL_EFFECTS
from .military_state import (
    StandardizedUnitParams as MilitaryUnitParams,
    StandardizedMilitaryUnit as MilitaryUnit,
    StandardizedClusterMilitaryState as ClusterMilitaryState,
    StandardizedWorldMilitaryState as WorldMilitaryState,
)

# ─────────────────────────────────────────────────────────────────────────── #
# Type alias for the GravitasWorld (avoid circular import)                    #
# ─────────────────────────────────────────────────────────────────────────── #

GravitasWorld = object   # runtime type; duck-typed access only


# ─────────────────────────────────────────────────────────────────────────── #
# Political-effect coefficients                                               #
# ─────────────────────────────────────────────────────────────────────────── #

# Generic combat presence: per-unit hazard suppression per step
_COMBAT_HAZARD_SUPPRESSION = 0.015   # each alive combat unit lowers hazard by this much
_COMBAT_POLAR_PENALTY      = 0.003   # but raises polarization (occupation effect)

# Losses
_LOSS_HAZARD_SPIKE  = 0.12   # hazard jump per unit killed since last step
_LOSS_TRUST_DRAIN   = 0.04   # trust loss per unit killed

# Blockade (encirclement)
_BLOCKADE_RESOURCE_DRAIN = 0.02   # per-step resource drop in encircled cluster
_BLOCKADE_HAZARD_RISE    = 0.03   # per-step hazard rise in encircled cluster

# Politics → military
_TRUST_MORALE_SLOPE      = 0.08   # morale drain rate when trust is below threshold
_POLAR_REINF_PENALTY     = 0.25   # fraction of reinforcement pool lost at max polar
_HAZARD_SUPPLY_MULT      = 0.05   # extra supply consumption per hazard unit
_RESOURCE_DEPOT_REFILL   = 0.06   # depot refill coefficient from cluster resource
_ALLIANCE_SUPPLY_SHARE   = 0.04   # per-allied-cluster supply share per step
_POP_DEPLOY_CEILING      = 8.0    # max units deployable when pop = 1.0
_MEDIA_BIAS_FOW_MULT     = 0.20   # how much media bias inflates fog-of-war


# ─────────────────────────────────────────────────────────────────────────── #
# Helper: cluster index ↔ WorldMilitaryState                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _alive_units_in(cluster: ClusterMilitaryState) -> List[MilitaryUnit]:
    return [u for u in cluster.units if u.is_alive]


def _unit_count_by_role(units: List[MilitaryUnit], role: UnitRole) -> int:
    return sum(1 for u in units if get_unit_role(u.unit_type) == role)


# ─────────────────────────────────────────────────────────────────────────── #
# MILITARY → POLITICS  (returns per-cluster delta arrays)                     #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_military_political_feedback(
    military_state: WorldMilitaryState,
    world: GravitasWorld,            # GravitasWorld duck-typed
    prev_military_state: Optional[WorldMilitaryState],
    adjacency: NDArray[np.float64],  # (N, N) topology
    alliance: Optional[NDArray[np.float64]],   # (N, N) alliance matrix
    N: int,
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute per-cluster political deltas caused by military presence.

    Returns a dict of (N,) arrays:
      delta_sigma, delta_hazard, delta_resource, delta_military,
      delta_trust, delta_polar, delta_media_bias, delta_population
    """
    delta_sigma      = np.zeros(N)
    delta_hazard     = np.zeros(N)
    delta_resource   = np.zeros(N)
    delta_trust      = np.zeros(N)
    delta_polar      = np.zeros(N)
    delta_media_bias = np.zeros(N)
    delta_population = np.zeros(N)

    clusters_pol = list(world.clusters)  # ClusterState list

    for i in range(N):
        mil_cluster = military_state.get_cluster(i)
        if mil_cluster is None:
            continue

        alive = _alive_units_in(mil_cluster)
        if not alive:
            continue

        # ── 1. Per-unit political type effects ───────────────────────────── #
        for unit in alive:
            effects = POLITICAL_EFFECTS.get(unit.unit_type, {})
            effectiveness = unit.combat_effectiveness * unit.morale
            for field, base_delta in effects.items():
                scaled = base_delta * effectiveness * dt
                if field == "sigma":
                    delta_sigma[i]      += scaled
                elif field == "hazard":
                    delta_hazard[i]     += scaled
                elif field == "resource":
                    delta_resource[i]   += scaled
                elif field == "trust":
                    delta_trust[i]      += scaled
                elif field == "polar":
                    delta_polar[i]      += scaled
                elif field == "media_bias":
                    # intelligence units shrink bias magnitude, propaganda expands it
                    current_bias = clusters_pol[i].polar  # proxy; actual bias in world
                    delta_media_bias[i] += scaled
                elif field == "population":
                    delta_population[i] += scaled

        # ── 2. Generic combat presence: suppresses hazard, raises polarization #
        n_combat = _unit_count_by_role(alive, UnitRole.INFANTRY) \
                 + _unit_count_by_role(alive, UnitRole.ARMOR) \
                 + _unit_count_by_role(alive, UnitRole.SPECIAL_FORCES)
        avg_eff = float(np.mean([u.combat_effectiveness for u in alive]))
        delta_hazard[i] -= _COMBAT_HAZARD_SUPPRESSION * n_combat * avg_eff * dt
        delta_polar[i]  += _COMBAT_POLAR_PENALTY * n_combat * avg_eff * dt

        # ── 3. Unit losses → hazard spike + trust drain ───────────────────── #
        if prev_military_state is not None:
            prev_cluster = prev_military_state.get_cluster(i)
            if prev_cluster is not None:
                prev_alive = sum(1 for u in prev_cluster.units if u.is_alive)
                losses = max(0, prev_alive - len(alive))
                delta_hazard[i] += _LOSS_HAZARD_SPIKE * losses * dt
                delta_trust[i]  -= _LOSS_TRUST_DRAIN  * losses * dt

    # ── 4. Encirclement / blockade ─────────────────────────────────────── #
    # A cluster is "blockaded" if every adjacent cluster has more hostile
    # (low-alliance) military presence than the cluster itself.
    total_power = np.array([
        sum(u.combat_power for u in _alive_units_in(military_state.get_cluster(i) or
            type('_', (), {'units': ()})()))
        for i in range(N)
    ], dtype=np.float64)

    for i in range(N):
        neighbors = np.where(adjacency[i, :N] > 0)[0]
        if len(neighbors) == 0:
            continue
        # hostile = neighbor has higher power AND weak/negative alliance
        hostile_count = 0
        for j in neighbors:
            ally_strength = float(alliance[i, j]) if alliance is not None else 0.0
            if total_power[j] > total_power[i] * 1.2 and ally_strength < -0.1:
                hostile_count += 1
        if hostile_count == len(neighbors):   # fully encircled
            delta_resource[i] -= _BLOCKADE_RESOURCE_DRAIN * dt
            delta_hazard[i]   += _BLOCKADE_HAZARD_RISE    * dt

    # clamp all deltas to prevent single-step explosions
    np.clip(delta_hazard,     -0.5,  0.5, out=delta_hazard)
    np.clip(delta_sigma,      -0.2,  0.2, out=delta_sigma)
    np.clip(delta_resource,   -0.2,  0.2, out=delta_resource)
    np.clip(delta_trust,      -0.2,  0.2, out=delta_trust)
    np.clip(delta_polar,      -0.2,  0.2, out=delta_polar)
    np.clip(delta_media_bias, -0.3,  0.3, out=delta_media_bias)
    np.clip(delta_population, -0.1,  0.1, out=delta_population)

    return {
        "delta_sigma":      delta_sigma,
        "delta_hazard":     delta_hazard,
        "delta_resource":   delta_resource,
        "delta_trust":      delta_trust,
        "delta_polar":      delta_polar,
        "delta_media_bias": delta_media_bias,
        "delta_population": delta_population,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# POLITICS → MILITARY  (modifies WorldMilitaryState in-place immutably)      #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_political_military_feedback(
    military_state: WorldMilitaryState,
    world: GravitasWorld,
    alliance: Optional[NDArray[np.float64]],
    N: int,
    dt: float,
    rng: np.random.Generator,
) -> WorldMilitaryState:
    """
    Apply GravitasWorld political state as constraints and modifiers on military state.

    Effects applied:
      - Low cluster trust    → unit morale drain
      - High polarization    → reinforcement pool penalty
      - High hazard          → supply consumption spike
      - Cluster resource     → supply depot replenishment
      - Alliance strength    → supply sharing between allied clusters
      - Population           → deployment ceiling (max reinforcement_pool)
      - Media bias           → fog-of-war increase (observation noise)
    """
    clusters_pol = list(world.clusters)
    new_clusters  = list(military_state.clusters)

    for i in range(N):
        mil_cluster = military_state.get_cluster(i)
        if mil_cluster is None or i >= len(clusters_pol):
            continue

        pol = clusters_pol[i]
        alive = _alive_units_in(mil_cluster)
        if not alive:
            # Still replenish supply depot from resource base
            new_depot = min(
                mil_cluster.supply_depot + _RESOURCE_DEPOT_REFILL * pol.resource * dt,
                10.0  # hard cap
            )
            idx = next((k for k, c in enumerate(new_clusters) if c.cluster_id == i), None)
            if idx is not None:
                new_clusters[idx] = mil_cluster.copy_with(supply_depot=new_depot)
            continue

        # ── a. Trust → morale drain ───────────────────────────────────────── #
        trust_deficit = max(0.0, 0.4 - pol.trust)   # below 40% trust → drain
        morale_drain  = _TRUST_MORALE_SLOPE * trust_deficit * dt

        # ── b. Hazard → supply consumption boost ─────────────────────────── #
        supply_mult = 1.0 + _HAZARD_SUPPLY_MULT * pol.hazard

        # ── c. Resource → depot refill ────────────────────────────────────── #
        depot_gain = _RESOURCE_DEPOT_REFILL * pol.resource * dt

        new_units: List[MilitaryUnit] = []
        for unit in mil_cluster.units:
            if not unit.is_alive:
                new_units.append(unit)
                continue

            new_morale  = float(np.clip(unit.morale - morale_drain, 0.0, 1.0))
            # Effectiveness decays with morale
            new_eff = float(np.clip(
                unit.combat_effectiveness * (0.99 + 0.01 * new_morale),
                0.0, 1.0
            ))
            # Supply depletes faster under hazard
            supply_cost = supply_mult * 0.005 * dt
            new_supply  = float(np.clip(unit.supply_level - supply_cost, 0.0, 1.0))

            new_units.append(unit.copy_with(
                morale=new_morale,
                combat_effectiveness=new_eff,
                supply_level=new_supply,
            ))

        # ── d. Depot replenishment ────────────────────────────────────────── #
        new_depot = float(np.clip(
            mil_cluster.supply_depot + depot_gain,
            0.0, 10.0
        ))

        idx = next((k for k, c in enumerate(new_clusters) if c.cluster_id == i), None)
        if idx is not None:
            new_clusters[idx] = mil_cluster.copy_with(
                units=tuple(new_units),
                supply_depot=new_depot,
            )

    # ── e. Alliance supply sharing ────────────────────────────────────────── #
    if alliance is not None:
        for i in range(N):
            mil_i = next((c for c in new_clusters if c.cluster_id == i), None)
            if mil_i is None:
                continue
            for j in range(N):
                if i == j:
                    continue
                ally_strength = max(0.0, float(alliance[i, j]))
                if ally_strength < 0.1:
                    continue
                mil_j = next((c for c in new_clusters if c.cluster_id == j), None)
                if mil_j is None:
                    continue
                # Share supply from richer to poorer depot
                if mil_j.supply_depot > mil_i.supply_depot + 0.5:
                    transfer = _ALLIANCE_SUPPLY_SHARE * ally_strength * dt
                    transfer = min(transfer, mil_j.supply_depot * 0.1)
                    idx_i = next((k for k, c in enumerate(new_clusters) if c.cluster_id == i), None)
                    idx_j = next((k for k, c in enumerate(new_clusters) if c.cluster_id == j), None)
                    if idx_i is not None and idx_j is not None:
                        new_clusters[idx_i] = new_clusters[idx_i].copy_with(
                            supply_depot=new_clusters[idx_i].supply_depot + transfer
                        )
                        new_clusters[idx_j] = new_clusters[idx_j].copy_with(
                            supply_depot=max(0.0, new_clusters[idx_j].supply_depot - transfer)
                        )

    # ── f. Polarization → reinforcement pool penalty ─────────────────────── #
    global_polar = float(np.mean([c.polar for c in clusters_pol[:N]]))
    polar_penalty = _POLAR_REINF_PENALTY * global_polar
    new_pool = float(np.clip(
        military_state.global_reinforcement_pool * (1.0 - polar_penalty * dt),
        0.0, None
    ))

    # ── g. Population → deployment ceiling on reinforcement pool ─────────── #
    if hasattr(world, "population") and world.population is not None:
        mean_pop = float(np.mean(world.population[:N]))
    else:
        mean_pop = 0.65
    pop_ceiling = _POP_DEPLOY_CEILING * mean_pop
    new_pool = min(new_pool, pop_ceiling)

    # ── h. Economy → supply depot bonus + debt penalty + draft ceiling ─────── #
    if hasattr(world, "economy") and world.economy is not None:
        eco = world.economy
        eco_params = getattr(world, "_params", None)
        # Fallback constants if params not accessible
        industry_bonus_rate  = 0.30   # eco_industry_supply_bonus default
        debt_reinf_penalty   = 0.25   # eco_debt_reinf_penalty default
        working_age_frac     = 0.62
        draft_eligible_frac  = 0.15

        for i in range(N):
            if i >= eco.shape[0]:
                break
            G_i = float(eco[i, 0])   # GDP index
            D_i = float(eco[i, 2])   # Debt ratio
            I_i = float(eco[i, 3])   # Industrial capacity

            mil_cluster = next((c for c in new_clusters if c.cluster_id == i), None)
            if mil_cluster is None:
                continue

            # Industrial capacity boosts supply depot refill beyond base rate
            industry_boost = industry_bonus_rate * I_i * dt
            new_depot = float(np.clip(
                mil_cluster.supply_depot + industry_boost, 0.0, 10.0
            ))

            # War-time industry: high military + high industry → extra supply
            total_power = sum(
                u.hit_points * u.combat_effectiveness
                for u in mil_cluster.units if u.is_alive
            )
            if total_power > 0.5 and I_i > 0.4:
                war_production = 0.10 * I_i * dt
                new_depot = float(np.clip(new_depot + war_production, 0.0, 10.0))

            idx = next((k for k, c in enumerate(new_clusters) if c.cluster_id == i), None)
            if idx is not None:
                new_clusters[idx] = new_clusters[idx].copy_with(supply_depot=new_depot)

        # Mean debt reduces global reinforcement pool
        mean_debt = float(np.mean(eco[:N, 2]))
        debt_penalty = debt_reinf_penalty * mean_debt
        new_pool = float(np.clip(new_pool * (1.0 - debt_penalty * dt), 0.0, None))

        # Draft pool: per-cluster hard ceiling based on population×demographics
        if hasattr(world, "population") and world.population is not None:
            pop = world.population
            c_arr_for_draft = np.array([[c.sigma, c.hazard, c.resource, c.military, c.trust, c.polar]
                                        for c in clusters_pol[:N]])
            # draft_pool_i = pop_i × working_age_frac × draft_eligible_frac × mobilization
            for i in range(N):
                pop_i   = float(pop[i]) if i < len(pop) else 0.65
                sigma_i = float(c_arr_for_draft[i, 0])
                G_i     = float(eco[i, 0]) if i < eco.shape[0] else 0.5
                mobilization = draft_eligible_frac * (
                    1.0 + 0.5 * (1.0 - sigma_i) * (1.0 - G_i)
                )
                mobilization = min(mobilization, draft_eligible_frac * 2.0)
                draft_ceiling = pop_i * working_age_frac * mobilization * 100.0  # in unit-count space
                # Count alive units in this cluster
                mil_cluster = next((c for c in new_clusters if c.cluster_id == i), None)
                if mil_cluster is None:
                    continue
                alive_count = sum(1 for u in mil_cluster.units if u.is_alive)
                if alive_count > draft_ceiling:
                    # Mark excess units as over-extended (reduce their morale)
                    idx = next((k for k, c in enumerate(new_clusters) if c.cluster_id == i), None)
                    if idx is None:
                        continue
                    sorted_units = list(new_clusters[idx].units)
                    alive_units  = [u for u in sorted_units if u.is_alive]
                    excess       = int(alive_count - draft_ceiling)
                    penalised    = []
                    for j, u in enumerate(sorted_units):
                        if u.is_alive and excess > 0:
                            penalised.append(u.copy_with(morale=float(np.clip(u.morale - 0.05 * dt, 0.0, 1.0))))
                            excess -= 1
                        else:
                            penalised.append(u)
                    new_clusters[idx] = new_clusters[idx].copy_with(units=tuple(penalised))

    return military_state.copy_with(
        clusters=tuple(new_clusters),
        global_reinforcement_pool=new_pool,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Tactic political effects — applied once when a tactic is ordered            #
# ─────────────────────────────────────────────────────────────────────────── #

from .advanced_tactics import CombatTactic

# Immediate cluster-level political impulse when a tactic is executed.
# Format: {tactic: {field: delta}}   (applied ONCE, not per-step)
TACTIC_POLITICAL_IMPULSES: Dict[CombatTactic, Dict[str, float]] = {
    CombatTactic.COUNTERINSURGENCY: {
        "hazard":   -0.12,
        "polar":    +0.10,    # crackdown polarizes society
        "trust":    -0.05,
    },
    CombatTactic.PEACEKEEPING_OP: {
        "hazard":   -0.08,
        "trust":    +0.10,
        "polar":    -0.06,
        "sigma":    +0.04,
    },
    CombatTactic.BLOCKADE: {
        "resource": -0.12,
        "hazard":   +0.10,
        "polar":    +0.08,
    },
    CombatTactic.OCCUPATION: {
        "sigma":    -0.08,
        "trust":    -0.12,
        "polar":    +0.10,
        "resource": +0.05,   # occupier extracts resources
    },
    CombatTactic.HEARTS_AND_MINDS: {
        "trust":    +0.15,
        "polar":    -0.12,
        "hazard":   -0.05,
        "sigma":    +0.06,
    },
    CombatTactic.SCORCHED_EARTH: {
        "resource": -0.25,
        "hazard":   +0.20,
        "population": -0.15,
        "sigma":    -0.10,
    },
}


def apply_tactic_political_impulse(
    tactic: CombatTactic,
    target_cluster_id: int,
    world: GravitasWorld,
    N: int,
) -> GravitasWorld:
    """
    Apply the one-time political impulse for a tactic to GravitasWorld.
    Returns the modified world.
    """
    impulse = TACTIC_POLITICAL_IMPULSES.get(tactic)
    if impulse is None or target_cluster_id >= N:
        return world

    clusters = list(world.clusters)
    c = clusters[target_cluster_id]

    new_sigma    = float(np.clip(c.sigma    + impulse.get("sigma",    0.0), 0.0, 1.0))
    new_hazard   = float(np.clip(c.hazard   + impulse.get("hazard",   0.0), 0.0, 5.0))
    new_resource = float(np.clip(c.resource + impulse.get("resource", 0.0), 0.0, 1.0))
    new_trust    = float(np.clip(c.trust    + impulse.get("trust",    0.0), 0.0, 1.0))
    new_polar    = float(np.clip(c.polar + impulse.get("polar", 0.0), 0.0, 1.0))

    clusters[target_cluster_id] = c.copy_with(
        sigma=new_sigma,
        hazard=new_hazard,
        resource=new_resource,
        trust=new_trust,
        polar=new_polar,
    )

    new_world = world.copy_with_clusters(clusters)

    # Population impulse
    pop_delta = impulse.get("population", 0.0)
    if pop_delta != 0.0 and world.population is not None:
        new_pop = world.population.copy()
        new_pop[target_cluster_id] = float(np.clip(
            new_pop[target_cluster_id] + pop_delta, 0.0, 1.0
        ))
        new_world = new_world.copy_with_population(new_pop)

    return new_world
