"""
military_dynamics.py — Call of War native game dynamics engine.

Handles all per-step game logic:
  1. Action application (produce, upgrade, move, attack, build, research, reinforce, retreat)
  2. Production queue progression
  3. Building construction progression
  4. Research progression
  5. Combat resolution (delegates to cow_combat.resolve_cow_combat)
  6. Supply consumption & regeneration (nonlinear)
  7. Morale & fatigue dynamics
  8. Objective tracking
  9. Cluster ownership
  10. Reward computation & victory conditions

All costs scale nonlinearly to prevent RL exploitation.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from .cow_combat import (
    CowUnit, CowArmy, CowUnitType, CowUnitCategory, CowArmorClass,
    CowTerrain, CowDoctrine, CowProductionCost, CowBuildingType,
    CowResearchProject, RESEARCH_TREE, BUILDING_COSTS, BUILDING_BUILD_TIME,
    CATEGORY_BUILDING_REQ, MAX_BUILDING_LEVEL, required_building_level,
    get_unit_stats, get_doctrine_mod, resolve_cow_combat, _gen_uid,
    production_cost as base_production_cost,
    upgrade_cost as base_upgrade_cost,
    nonlinear_production_cost, nonlinear_supply_drain,
    combat_fatigue_factor, morale_cascade,
)
from .military_state import (
    CowBuilding, ProductionQueueItem, ResearchSlot, MilitaryObjective,
    CowClusterState, CowFactionState, CowWorldState, CowExternalModifiers,
    can_afford, deduct, cost_to_array,
    N_RESOURCES, MAX_PRODUCTION_QUEUE, MAX_CONCURRENT_RESEARCH,
)
from .physics_bridge import (
    PhysicsModifiers,
    step_physics_for_world, init_physics_for_world,
    get_cluster_physics_mods, compute_physics_combat_mult,
    compute_physics_attrition, compute_physics_morale_effect,
    physics_to_obs, physics_obs_size,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Action Types                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class ActionType(Enum):
    NOOP     = 0
    PRODUCE  = 1   # Start producing a unit at a cluster
    UPGRADE  = 2   # Upgrade an existing unit to next level
    MOVE     = 3   # Move faction's units from one cluster to another
    ATTACK   = 4   # Move units into enemy cluster (triggers combat)
    BUILD    = 5   # Construct or upgrade a building
    RESEARCH = 6   # Start a research project
    REINFORCE = 7  # Heal units in a cluster (costs resources)
    RETREAT  = 8   # Pull units out of a contested cluster

N_ACTION_TYPES = 9
# Index of each CowUnitType in a flat list (for action space encoding)
UNIT_TYPE_LIST: List[CowUnitType] = list(CowUnitType)
N_UNIT_TYPES = len(UNIT_TYPE_LIST)
BUILDING_TYPE_LIST: List[CowBuildingType] = list(CowBuildingType)
N_BUILDING_TYPES = len(BUILDING_TYPE_LIST)


# ─────────────────────────────────────────────────────────────────────────── #
# Action Application                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_action(
    world: CowWorldState,
    faction_id: int,
    action_type: int,
    source_cluster: int,
    target_cluster: int,
    unit_type_idx: int,
    aux_idx: int,
    rng: np.random.Generator,
) -> Tuple[CowWorldState, float]:
    """Apply a single faction action. Returns (world, action_reward).

    action_type: ActionType enum value
    source_cluster: cluster id for the action origin
    target_cluster: cluster id for the action target (move/attack/retreat)
    unit_type_idx: index into UNIT_TYPE_LIST (for produce/upgrade/research)
    aux_idx: auxiliary index (building type for BUILD, unit level for PRODUCE)
    """
    faction = world.factions.get(faction_id)
    if faction is None:
        return world, -0.01

    at = action_type
    reward = 0.0

    if at == ActionType.NOOP.value:
        pass

    elif at == ActionType.PRODUCE.value:
        reward = _apply_produce(world, faction, source_cluster, unit_type_idx, aux_idx)

    elif at == ActionType.UPGRADE.value:
        reward = _apply_upgrade(world, faction, source_cluster, unit_type_idx)

    elif at == ActionType.MOVE.value:
        reward = _apply_move(world, faction, source_cluster, target_cluster)

    elif at == ActionType.ATTACK.value:
        reward = _apply_move(world, faction, source_cluster, target_cluster)

    elif at == ActionType.BUILD.value:
        reward = _apply_build(world, faction, source_cluster, aux_idx)

    elif at == ActionType.RESEARCH.value:
        reward = _apply_research(world, faction, unit_type_idx, aux_idx)

    elif at == ActionType.REINFORCE.value:
        reward = _apply_reinforce(world, faction, source_cluster)

    elif at == ActionType.RETREAT.value:
        reward = _apply_move(world, faction, source_cluster, target_cluster)

    else:
        reward = -0.01  # invalid action penalty

    return world, reward


def _apply_produce(
    world: CowWorldState, faction: CowFactionState,
    cluster_id: int, unit_type_idx: int, level: int,
) -> float:
    if cluster_id < 0 or cluster_id >= world.n_clusters:
        return -0.02
    level = max(1, min(4, level))
    if unit_type_idx < 0 or unit_type_idx >= N_UNIT_TYPES:
        return -0.02

    ut = UNIT_TYPE_LIST[unit_type_idx]
    cluster = world.clusters[cluster_id]

    # Must own the cluster
    if cluster.owner_faction != faction.faction_id:
        return -0.02

    # Must have researched this level
    if not faction.has_researched(ut, level):
        return -0.02

    # Building & queue check
    if not cluster.can_produce(ut, level):
        return -0.02

    # Compute nonlinear cost (scaled by government military_cost modifier)
    base_cost = base_production_cost(ut, level, faction.doctrine)
    n_same = faction.count_of_type(world, ut)
    n_total = faction.total_units(world)
    n_over = max(0, world.faction_cluster_count(faction.faction_id) - 5)
    actual_cost = nonlinear_production_cost(base_cost, n_same, n_total, n_over)

    # Apply external production cost modifier (government)
    ext = faction.external_modifiers
    if ext.production_cost_mult != 1.0:
        scaled = cost_to_array(actual_cost) * ext.production_cost_mult
        actual_cost = CowProductionCost(
            rations=scaled[0], steel=scaled[1], ammo=scaled[2],
            fuel=scaled[3], medical=scaled[4])

    if not can_afford(faction.resources, actual_cost):
        return -0.02

    # Deduct resources
    faction.resources = deduct(faction.resources, actual_cost)

    # Add to production queue (speed scaled by spirit industry_mod)
    stats = get_unit_stats(ut, level)
    speed_mult = max(0.2, ext.production_speed_mult)
    build_time = stats.build_time * (1.0 + 0.2 * len(cluster.production_queue)) / speed_mult
    cluster.production_queue.append(ProductionQueueItem(
        unit_type=ut, level=level,
        remaining_steps=build_time, total_steps=build_time,
        cost_paid=actual_cost,
    ))

    return 0.05  # small reward for valid production order


def _apply_upgrade(
    world: CowWorldState, faction: CowFactionState,
    cluster_id: int, unit_type_idx: int,
) -> float:
    if cluster_id < 0 or cluster_id >= world.n_clusters:
        return -0.02
    cluster = world.clusters[cluster_id]
    if cluster.owner_faction != faction.faction_id:
        return -0.02

    # Find first matching alive unit of this type that can be upgraded
    if unit_type_idx < 0 or unit_type_idx >= N_UNIT_TYPES:
        return -0.02
    ut = UNIT_TYPE_LIST[unit_type_idx]

    target_unit = None
    for u in cluster.units_of_faction(faction.faction_id):
        if u.stats.unit_type == ut and u.stats.level < 4:
            target_level = u.stats.level + 1
            if faction.has_researched(ut, target_level):
                target_unit = u
                break

    if target_unit is None:
        return -0.02

    target_level = target_unit.stats.level + 1
    cost = base_upgrade_cost(ut, target_level, faction.doctrine)
    if not can_afford(faction.resources, cost):
        return -0.02

    # Pay and upgrade
    faction.resources = deduct(faction.resources, cost)
    new_stats = get_unit_stats(ut, target_level)
    hp_frac = target_unit.hp_fraction
    doctrine_hp_mod = 1.0 + get_doctrine_mod(faction.doctrine, new_stats.category).hp_mod
    new_hp = new_stats.hitpoints * doctrine_hp_mod * hp_frac

    # Replace the unit in-place
    idx = next(i for i, u in enumerate(cluster.units) if u.uid == target_unit.uid)
    cluster.units[idx] = CowUnit(
        uid=target_unit.uid, stats=new_stats, hp=new_hp,
        faction_id=faction.faction_id, cluster_id=cluster_id,
        experience=target_unit.experience, morale=target_unit.morale,
    )

    return 0.10  # reward for successful upgrade


def _apply_move(
    world: CowWorldState, faction: CowFactionState,
    source: int, target: int,
) -> float:
    if source < 0 or source >= world.n_clusters:
        return -0.02
    if target < 0 or target >= world.n_clusters:
        return -0.02
    if source == target:
        return -0.01
    if not world.adjacency[source, target]:
        return -0.02

    src_cluster = world.clusters[source]
    tgt_cluster = world.clusters[target]

    # Move all faction's units from source to target
    to_move = [u for u in src_cluster.units if u.faction_id == faction.faction_id and u.is_alive]
    if not to_move:
        return -0.02

    # Remove from source, add to target
    remaining = [u for u in src_cluster.units if u.faction_id != faction.faction_id or not u.is_alive]
    src_cluster.units = remaining + [u for u in src_cluster.units if not u.is_alive and u.faction_id == faction.faction_id]

    for u in to_move:
        u.cluster_id = target
        tgt_cluster.units.append(u)

    # Clean up source dead units
    src_cluster.units = [u for u in src_cluster.units if u.is_alive or u.faction_id != faction.faction_id]

    return 0.02


def _apply_build(
    world: CowWorldState, faction: CowFactionState,
    cluster_id: int, building_type_idx: int,
) -> float:
    if cluster_id < 0 or cluster_id >= world.n_clusters:
        return -0.02
    if building_type_idx < 0 or building_type_idx >= N_BUILDING_TYPES:
        return -0.02

    cluster = world.clusters[cluster_id]
    if cluster.owner_faction != faction.faction_id:
        return -0.02

    bt = BUILDING_TYPE_LIST[building_type_idx]
    building = cluster.buildings[bt]

    if building.is_upgrading:
        return -0.02  # already building
    if building.level >= MAX_BUILDING_LEVEL:
        return -0.02

    target_level = building.level + 1
    costs = BUILDING_COSTS[bt]
    if target_level - 1 >= len(costs):
        return -0.02

    cost = costs[target_level - 1]
    if not can_afford(faction.resources, cost):
        return -0.02

    faction.resources = deduct(faction.resources, cost)

    build_time = BUILDING_BUILD_TIME[bt][target_level - 1]
    building.is_upgrading = True
    building.construction_progress = 0.0
    # Store target level — we'll apply on completion
    building._target_level = target_level  # type: ignore
    building._build_time = build_time      # type: ignore

    return 0.05


def _apply_research(
    world: CowWorldState, faction: CowFactionState,
    unit_type_idx: int, level: int,
) -> float:
    if unit_type_idx < 0 or unit_type_idx >= N_UNIT_TYPES:
        return -0.02
    level = max(2, min(4, level))
    ut = UNIT_TYPE_LIST[unit_type_idx]

    if not faction.can_start_research(ut, level):
        return -0.02

    proj = RESEARCH_TREE[(ut, level)]
    faction.resources = deduct(faction.resources, proj.cost)
    faction.research_slots.append(ResearchSlot(
        project=proj, remaining_steps=proj.time, total_steps=proj.time,
    ))

    return 0.05


def _apply_reinforce(
    world: CowWorldState, faction: CowFactionState,
    cluster_id: int,
) -> float:
    if cluster_id < 0 or cluster_id >= world.n_clusters:
        return -0.02
    cluster = world.clusters[cluster_id]
    if cluster.owner_faction != faction.faction_id:
        return -0.02

    units = cluster.units_of_faction(faction.faction_id)
    if not units:
        return -0.02

    # Heal cost: 0.5 rations + 0.3 medical per unit
    heal_cost = CowProductionCost(
        rations=0.5 * len(units), steel=0, ammo=0, fuel=0,
        medical=0.3 * len(units))
    if not can_afford(faction.resources, heal_cost):
        return -0.02

    faction.resources = deduct(faction.resources, heal_cost)

    healed = 0
    for u in units:
        if u.hp < u.stats.hitpoints:
            heal_amt = u.stats.hitpoints * 0.15  # heal 15% of max HP
            old_hp = u.hp
            u.hp = min(u.stats.hitpoints, u.hp + heal_amt)
            if u.hp > old_hp:
                healed += 1

    return 0.03 * healed


# ─────────────────────────────────────────────────────────────────────────── #
# Per-Step Subsystem Updates                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _step_production(world: CowWorldState) -> None:
    """Advance production queues, spawn completed units."""
    for cluster in world.clusters:
        completed = []
        remaining = []
        for item in cluster.production_queue:
            item.remaining_steps -= 1.0
            if item.remaining_steps <= 0:
                completed.append(item)
            else:
                remaining.append(item)
        cluster.production_queue = remaining

        for item in completed:
            fid = cluster.owner_faction
            if fid is None:
                continue
            faction = world.factions.get(fid)
            doctrine = faction.doctrine if faction else None
            stats = get_unit_stats(item.unit_type, item.level)
            max_hp = stats.hitpoints
            if doctrine:
                max_hp *= (1.0 + get_doctrine_mod(doctrine, stats.category).hp_mod)
            uid = _gen_uid()
            u = CowUnit(uid=uid, stats=stats, hp=max_hp,
                         faction_id=fid, cluster_id=cluster.cluster_id)
            cluster.units.append(u)


def _step_buildings(world: CowWorldState) -> None:
    """Advance building construction."""
    for cluster in world.clusters:
        for building in cluster.buildings.values():
            if not building.is_upgrading:
                continue
            bt = getattr(building, '_build_time', 5.0)
            building.construction_progress += 1.0 / max(bt, 1.0)
            if building.construction_progress >= 1.0:
                building.construction_progress = 1.0
                building.level = getattr(building, '_target_level', building.level + 1)
                building.is_upgrading = False


def _step_research(world: CowWorldState) -> None:
    """Advance research and unlock completed projects."""
    for faction in world.factions.values():
        completed = []
        remaining = []
        for slot in faction.research_slots:
            slot.remaining_steps -= 1.0
            if slot.remaining_steps <= 0:
                completed.append(slot)
            else:
                remaining.append(slot)
        faction.research_slots = remaining
        for slot in completed:
            proj = slot.project
            faction.researched = faction.researched | frozenset({(proj.unit_type, proj.target_level)})


def _step_supply(world: CowWorldState) -> None:
    """Consume and regenerate supply per cluster. Nonlinear drain."""
    for cluster in world.clusters:
        n_alive = len(cluster.alive_units)
        drain = nonlinear_supply_drain(n_alive)
        cluster.supply = max(0.0, cluster.supply - drain)
        cluster.supply = min(50.0, cluster.supply + cluster.supply_regen)

        # Low supply → HP attrition
        if cluster.supply < 1.0 and n_alive > 0:
            attrition = 0.5 * (1.0 - cluster.supply)
            for u in cluster.alive_units:
                u.hp = max(0.01, u.hp - attrition)


def _step_combat(world: CowWorldState, rng: np.random.Generator) -> Dict[int, int]:
    """Resolve combat in all contested clusters. Returns {cluster_id: losses}."""
    losses_by_cluster: Dict[int, int] = {}

    for cluster in world.clusters:
        alive = cluster.alive_units
        if len(alive) < 2:
            cluster.recent_combat_rounds = max(0, cluster.recent_combat_rounds - 1)
            continue

        # Group by faction
        factions: Dict[int, List[CowUnit]] = {}
        for u in alive:
            factions.setdefault(u.faction_id, []).append(u)

        fids = sorted(factions.keys())
        if len(fids) < 2:
            cluster.recent_combat_rounds = max(0, cluster.recent_combat_rounds - 1)
            continue

        cluster.recent_combat_rounds += 1
        fatigue = combat_fatigue_factor(cluster.recent_combat_rounds)

        # Build armies
        armies: Dict[int, CowArmy] = {}
        for fid in fids:
            armies[fid] = CowArmy(factions[fid], fid, cluster.cluster_id)

        before_counts = {fid: army.unit_count for fid, army in armies.items()}

        # Pairwise combat
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                a_fid, d_fid = fids[i], fids[j]
                atk, dfn = armies.get(a_fid), armies.get(d_fid)
                if atk is None or dfn is None:
                    continue
                if atk.unit_count == 0 or dfn.unit_count == 0:
                    continue

                a_doc = world.factions[a_fid].doctrine if a_fid in world.factions else None
                d_doc = world.factions[d_fid].doctrine if d_fid in world.factions else None

                # External combat effectiveness modifiers
                a_ext = world.factions[a_fid].external_modifiers if a_fid in world.factions else CowExternalModifiers()
                d_ext = world.factions[d_fid].external_modifiers if d_fid in world.factions else CowExternalModifiers()

                # Physics combat modifiers (if physics enabled)
                phys_atk_mult = 1.0
                phys_def_mult = 1.0
                if world.physics_modifiers:
                    pmods = get_cluster_physics_mods(world.physics_modifiers, cluster.cluster_id)
                    phys_atk_mult = compute_physics_combat_mult(pmods, a_fid, is_attacker=True)
                    phys_def_mult = compute_physics_combat_mult(pmods, d_fid, is_attacker=False)

                # Winter terrain penalty (negated if winter_adapted)
                winter_penalty = 1.0
                if cluster.terrain == CowTerrain.MOUNTAINS:
                    # Rough approximation: mountains = winter-like
                    if not a_ext.winter_adapted:
                        winter_penalty = 0.85

                # Partisan bonus: defender gets bonus if fighting on own territory
                partisan_fort = 0.0
                if d_ext.partisan_bonus and cluster.owner_faction == d_fid:
                    partisan_fort = 0.10  # +10% fortification equivalent

                # ── Trait: Breakthrough ── attacker breakthrough reduces fort
                base_fort = (cluster.fortification if cluster.owner_faction == d_fid else 0.0) + partisan_fort
                atk_breakthrough = max(u.stats.traits.breakthrough for u in atk.alive_units) if atk.alive_units else 0.0
                effective_fort = base_fort * max(0.0, 1.0 - atk_breakthrough)

                ua, ud, _log = resolve_cow_combat(
                    atk, dfn,
                    terrain=cluster.terrain,
                    atk_doctrine=a_doc,
                    def_doctrine=d_doc,
                    fortification=effective_fort,
                    defender_is_home=(cluster.owner_faction == d_fid),
                    rng=rng,
                )

                # ── Trait: Suppression ── attacker suppression degrades defender HP
                atk_supp = sum(u.stats.traits.suppression for u in atk.alive_units) / max(1, atk.unit_count)
                def_supp = sum(u.stats.traits.suppression for u in dfn.alive_units) / max(1, dfn.unit_count)
                if atk_supp > 0:
                    for u in ud.units:
                        if u.is_alive:
                            u.hp = max(0.01, u.hp * (1.0 - min(0.15, atk_supp * 0.3)))
                if def_supp > 0:
                    for u in ua.units:
                        if u.is_alive:
                            u.hp = max(0.01, u.hp * (1.0 - min(0.15, def_supp * 0.3)))

                # Apply combat effectiveness multipliers (external + physics)
                a_total_eff = a_ext.combat_effectiveness * winter_penalty * phys_atk_mult
                d_total_eff = d_ext.combat_effectiveness * phys_def_mult
                if a_total_eff != 1.0:
                    for u in ua.units:
                        if u.is_alive:
                            u.morale = min(1.0, u.morale + (a_total_eff - 1.0) * 0.1)
                if d_total_eff != 1.0:
                    for u in ud.units:
                        if u.is_alive:
                            u.morale = min(1.0, u.morale + (d_total_eff - 1.0) * 0.1)

                # Apply fatigue to damage results
                if fatigue < 1.0:
                    for u in ua.units:
                        u.hp = max(0.0, u.hp * (1.0 - (1.0 - fatigue) * 0.3))
                    for u in ud.units:
                        u.hp = max(0.0, u.hp * (1.0 - (1.0 - fatigue) * 0.3))

                armies[a_fid] = ua
                armies[d_fid] = ud

        # Write updated units back
        all_updated: Dict[int, CowUnit] = {}
        for army in armies.values():
            for u in army.units:
                all_updated[u.uid] = u

        total_losses = 0
        # Track dead units per faction for trait-aware morale
        dead_by_faction: Dict[int, list] = {fid: [] for fid in fids}
        for i, u in enumerate(cluster.units):
            if u.uid in all_updated:
                old_alive = u.is_alive
                cluster.units[i] = all_updated[u.uid]
                if not all_updated[u.uid].is_alive and old_alive:
                    total_losses += 1
                    dead_by_faction.setdefault(u.faction_id, []).append(u)

        # Remove dead units
        cluster.units = [u for u in cluster.units if u.is_alive]

        # Trait-aware morale cascade: sum morale_on_death of killed units
        for fid in fids:
            dead_units = dead_by_faction.get(fid, [])
            if not dead_units:
                continue
            # Base cascade from loss ratio
            base_drop = morale_cascade(len(dead_units), before_counts.get(fid, 1))
            # Trait bonus: elite / high-value deaths hurt more
            trait_drop = sum(u.stats.traits.morale_on_death for u in dead_units)
            m_drop = base_drop + trait_drop
            for u in cluster.units:
                if u.faction_id == fid:
                    u.morale = max(0.1, u.morale - m_drop)

        # XP gain for survivors (elite units gain faster)
        for u in cluster.units:
            if u.is_alive:
                xp_gain = 0.15 if u.stats.traits.elite else 0.1
                u.experience = min(10.0, u.experience + xp_gain)

        losses_by_cluster[cluster.cluster_id] = total_losses

    return losses_by_cluster


def _step_ownership(world: CowWorldState) -> None:
    """Update cluster ownership based on dominant faction."""
    for cluster in world.clusters:
        dom = cluster.dominant_faction()
        if dom is not None and dom != cluster.owner_faction:
            # Only change if dominant faction is significantly stronger
            if cluster.owner_faction is not None:
                owner_str = cluster.faction_strength(cluster.owner_faction)
                dom_str = cluster.faction_strength(dom)
                if dom_str > owner_str * 1.5:  # need 50% strength advantage
                    # Update faction controlled_clusters
                    if cluster.owner_faction in world.factions:
                        old_f = world.factions[cluster.owner_faction]
                        if cluster.cluster_id in old_f.controlled_clusters:
                            old_f.controlled_clusters.remove(cluster.cluster_id)
                    cluster.owner_faction = dom
                    if dom in world.factions:
                        world.factions[dom].controlled_clusters.append(cluster.cluster_id)
            else:
                cluster.owner_faction = dom
                if dom in world.factions:
                    world.factions[dom].controlled_clusters.append(cluster.cluster_id)


def _step_income(world: CowWorldState) -> None:
    """Add resource income to factions based on controlled clusters."""
    for faction in world.factions.values():
        n_controlled = len(faction.controlled_clusters)
        # Income scales with sqrt of controlled territory (diminishing returns)
        territory_mult = min(2.0, np.sqrt(max(1, n_controlled) / 3.0))
        # Apply external resource income modifier (spirit + government)
        income_mult = faction.external_modifiers.resource_income_mult
        faction.resources += faction.base_income * territory_mult * income_mult
        # Cap resources
        faction.resources = np.minimum(faction.resources, 200.0)


def _step_objectives(world: CowWorldState) -> None:
    """Update objective progress."""
    for obj in world.objectives:
        if obj.is_completed:
            continue
        cluster = world.clusters[obj.target_cluster_id]
        strength = cluster.faction_strength(obj.faction_id)

        if obj.objective_type == "capture":
            if cluster.owner_faction == obj.faction_id:
                obj.completion_progress = min(1.0, obj.completion_progress + 0.1)
            else:
                obj.completion_progress = max(0.0, obj.completion_progress - 0.05)

        elif obj.objective_type == "hold":
            if cluster.owner_faction == obj.faction_id and strength >= obj.required_strength * 0.5:
                obj.hold_steps += 1
                obj.completion_progress = min(1.0, obj.hold_steps / max(obj.hold_required, 1))
            else:
                obj.hold_steps = max(0, obj.hold_steps - 1)
                obj.completion_progress = min(1.0, obj.hold_steps / max(obj.hold_required, 1))

        if obj.completion_progress >= 0.99:
            obj.is_completed = True


def _step_global_morale(world: CowWorldState) -> None:
    """Update faction global morale based on losses and territory."""
    for faction in world.factions.values():
        total_hp = world.faction_total_hp(faction.faction_id)
        n_clusters = world.faction_cluster_count(faction.faction_id)

        # External morale modifier from national spirit
        ext_morale = faction.external_modifiers.morale_mod

        # Morale recovers slowly, drops on losses
        if total_hp > 0 and n_clusters > 0:
            faction.global_morale = min(1.0, faction.global_morale + 0.01 + ext_morale * 0.1)
        else:
            faction.global_morale = max(0.1, faction.global_morale - 0.05 + ext_morale * 0.05)


# ─────────────────────────────────────────────────────────────────────────── #
# Physics Integration                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _step_physics(world: CowWorldState, rng: np.random.Generator) -> None:
    """Advance physics simulation one step and apply effects to CoW state.

    Skipped gracefully if physics is not initialised on the world state.
    Effects applied:
      - Weather attrition (HP drain from exposure)
      - Movement speed modifier (stored on cluster for action validation)
      - Production weather penalty (reduces income)
      - Morale effects from weather/supply conditions
      - Updated physics_modifiers dict on world state
    """
    if world.physics_states is None or world.los_state is None:
        return

    # Step physics engine
    new_phys, new_los, modifiers = step_physics_for_world(
        world.physics_states, world.los_state, world,
        step=world.step, rng=rng,
        config=world.physics_config,
    )
    world.physics_states = new_phys
    world.los_state = new_los
    world.physics_modifiers = modifiers

    # Apply per-cluster physics effects
    for cluster in world.clusters:
        cid = cluster.cluster_id
        mods = get_cluster_physics_mods(modifiers, cid)

        # Weather attrition: HP drain for all units in this cluster
        for u in cluster.alive_units:
            winterized = False
            if u.faction_id in world.factions:
                winterized = world.factions[u.faction_id].external_modifiers.winter_adapted
            attrition = compute_physics_attrition(mods, u.faction_id, winterized)
            # Trait: winter_hardened units take half cold attrition
            if u.stats.traits.winter_hardened and not winterized:
                attrition *= 0.5
            if attrition > 0:
                u.hp = max(0.01, u.hp - attrition * u.stats.hitpoints)

        # Morale effect from weather/supply
        morale_delta = compute_physics_morale_effect(mods)
        if morale_delta != 0:
            for u in cluster.alive_units:
                u.morale = float(np.clip(u.morale + morale_delta, 0.1, 1.0))

    # Apply production weather penalty to income
    # (reduces next step's effective income via resource penalty)
    for faction in world.factions.values():
        avg_prod_mult = 1.0
        n = 0
        for cid in faction.controlled_clusters:
            pm = get_cluster_physics_mods(modifiers, cid)
            avg_prod_mult += pm.production_weather_mult
            n += 1
        if n > 0:
            avg_prod_mult /= n
        # Apply as a small resource adjustment (weather slows production)
        if avg_prod_mult < 1.0:
            penalty = (1.0 - avg_prod_mult) * faction.base_income * 0.5
            faction.resources = np.maximum(0, faction.resources - penalty)


# ─────────────────────────────────────────────────────────────────────────── #
# Main Step Function                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def step_world(
    world: CowWorldState,
    faction_actions: Dict[int, Tuple[int, int, int, int, int]],
    rng: np.random.Generator,
) -> Tuple[CowWorldState, Dict[int, float], Dict[str, Any]]:
    """Advance world state by one step.

    Args:
        world: Current world state (mutated in-place for efficiency)
        faction_actions: {faction_id: (action_type, source, target, unit_type_idx, aux)}
        rng: Random number generator

    Returns:
        (world, {faction_id: reward}, metrics_dict)
    """
    # Snapshot for reward computation
    prev_hp = {fid: world.faction_total_hp(fid) for fid in world.factions}
    prev_clusters = {fid: world.faction_cluster_count(fid) for fid in world.factions}
    prev_objectives = {obj.objective_id: obj.is_completed for obj in world.objectives}

    # 1. Apply faction actions
    action_rewards: Dict[int, float] = {fid: 0.0 for fid in world.factions}
    for fid, action in faction_actions.items():
        at, src, tgt, ut_idx, aux = action
        _, ar = apply_action(world, fid, at, src, tgt, ut_idx, aux, rng)
        action_rewards[fid] += ar

    # 2. Step subsystems
    _step_production(world)
    _step_buildings(world)
    _step_research(world)
    _step_supply(world)

    # 3. Combat
    combat_losses = _step_combat(world, rng)

    # 4. Post-combat updates
    _step_ownership(world)
    _step_income(world)
    _step_objectives(world)
    _step_global_morale(world)

    # 5. Physics step (if physics enabled)
    _step_physics(world, rng)

    # 6. Advance step counter
    world.step += 1

    # 7. Compute rewards
    rewards = compute_reward(world, prev_hp, prev_clusters, prev_objectives, action_rewards)

    # 8. Metrics
    metrics = {
        "step": world.step,
        "combat_clusters": len(combat_losses),
        "total_losses": sum(combat_losses.values()),
    }
    for fid in world.factions:
        metrics[f"f{fid}_hp"] = round(world.faction_total_hp(fid), 1)
        metrics[f"f{fid}_clusters"] = world.faction_cluster_count(fid)
        metrics[f"f{fid}_resources"] = world.factions[fid].resources.sum().round(1)
    if world.physics_modifiers:
        # Aggregate physics metrics
        avg_temp = np.mean([m.temperature_c for m in world.physics_modifiers.values()])
        avg_vis = np.mean([m.visibility for m in world.physics_modifiers.values()])
        metrics["avg_temperature_c"] = round(avg_temp, 1)
        metrics["avg_visibility"] = round(avg_vis, 2)

    return world, rewards, metrics


# ─────────────────────────────────────────────────────────────────────────── #
# Reward Computation                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

# Reward weights
R_OBJECTIVE      = 5.0    # per objective completed
R_CLUSTER_GAIN   = 1.0    # per cluster captured
R_CLUSTER_LOSS   = -1.5   # per cluster lost
R_HP_RATIO       = 0.3    # scaled by HP change ratio
R_ANNIHILATION   = 10.0   # enemy has 0 units
R_LOSS           = -10.0  # we have 0 units


def compute_reward(
    world: CowWorldState,
    prev_hp: Dict[int, float],
    prev_clusters: Dict[int, int],
    prev_objectives: Dict[int, bool],
    action_rewards: Dict[int, float],
) -> Dict[int, float]:
    """Compute per-faction rewards."""
    rewards: Dict[int, float] = {}

    for fid in world.factions:
        r = action_rewards.get(fid, 0.0)

        # Objective completion
        for obj in world.objectives:
            if obj.faction_id == fid and obj.is_completed and not prev_objectives.get(obj.objective_id, False):
                r += R_OBJECTIVE

        # Cluster changes
        cur_c = world.faction_cluster_count(fid)
        prev_c = prev_clusters.get(fid, 0)
        r += R_CLUSTER_GAIN * max(0, cur_c - prev_c)
        r += R_CLUSTER_LOSS * max(0, prev_c - cur_c)

        # HP ratio change
        cur_hp = world.faction_total_hp(fid)
        p_hp = prev_hp.get(fid, 1.0)
        if p_hp > 0:
            hp_change = (cur_hp - p_hp) / p_hp
            r += R_HP_RATIO * hp_change

        # Terminal conditions
        enemy_fids = [f for f in world.factions if f != fid]
        enemy_hp = sum(world.faction_total_hp(ef) for ef in enemy_fids)
        if cur_hp <= 0:
            r += R_LOSS
        elif enemy_hp <= 0:
            r += R_ANNIHILATION

        rewards[fid] = r

    return rewards


# ─────────────────────────────────────────────────────────────────────────── #
# Victory Conditions                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def check_victory(world: CowWorldState) -> Dict[str, Any]:
    """Check if any faction has won.

    Returns:
        {"done": bool, "winner": Optional[int], "reason": str}
    """
    for fid, faction in world.factions.items():
        enemy_fids = [f for f in world.factions if f != fid]

        # Win: all enemies eliminated
        if all(world.faction_total_hp(ef) <= 0 for ef in enemy_fids):
            return {"done": True, "winner": fid, "reason": "annihilation"}

        # Win: all objectives completed
        faction_objs = [o for o in world.objectives if o.faction_id == fid]
        if faction_objs and all(o.is_completed for o in faction_objs):
            return {"done": True, "winner": fid, "reason": "objectives"}

    # Check for faction with 0 units (they lose but game continues if >2 factions)
    alive_factions = [fid for fid in world.factions if world.faction_total_hp(fid) > 0]
    if len(alive_factions) <= 1:
        winner = alive_factions[0] if alive_factions else None
        return {"done": True, "winner": winner, "reason": "last_standing"}

    return {"done": False, "winner": None, "reason": ""}


# ─────────────────────────────────────────────────────────────────────────── #
# Observation helpers                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def world_to_obs_array(
    world: CowWorldState,
    faction_id: int,
    max_clusters: int = 12,
    max_units_per_cluster: int = 20,
) -> NDArray[np.float32]:
    """Flatten world state into a fixed-size observation vector for RL.

    Layout per cluster (max_clusters):
      [terrain(1), owner_is_self(1), owner_is_enemy(1), owner_is_neutral(1),
       n_friendly(1), n_enemy(1), friendly_hp(1), enemy_hp(1),
       supply(1), fortification(1),
       building_levels(5),  # barracks, factory, airfield, bunker, depot
       production_queue_len(1),
       ...]
    = 15 per cluster

    Global:
      [resources(5), global_morale(1), step_normalized(1),
       n_research_slots_used(1), n_controlled_clusters(1)]
    = 9

    Total: max_clusters * 15 + 9
    """
    per_cluster = 15
    global_size = 9
    obs = np.zeros(max_clusters * per_cluster + global_size, dtype=np.float32)

    faction = world.factions.get(faction_id)
    enemy_fids = [f for f in world.factions if f != faction_id]

    for i, cluster in enumerate(world.clusters):
        if i >= max_clusters:
            break
        off = i * per_cluster

        # Terrain (normalized to 0-1)
        obs[off + 0] = cluster.terrain.value / 4.0

        # Ownership
        obs[off + 1] = 1.0 if cluster.owner_faction == faction_id else 0.0
        obs[off + 2] = 1.0 if cluster.owner_faction in enemy_fids else 0.0
        obs[off + 3] = 1.0 if cluster.owner_faction is None else 0.0

        # Unit counts and HP
        friendly = cluster.units_of_faction(faction_id)
        enemy = [u for fid in enemy_fids for u in cluster.units_of_faction(fid)]
        obs[off + 4] = len(friendly) / 20.0
        obs[off + 5] = len(enemy) / 20.0
        obs[off + 6] = sum(u.hp for u in friendly) / 200.0
        obs[off + 7] = sum(u.hp for u in enemy) / 200.0

        # Supply & fortification
        obs[off + 8] = cluster.supply / 50.0
        obs[off + 9] = cluster.fortification

        # Building levels
        for j, bt in enumerate(BUILDING_TYPE_LIST):
            obs[off + 10 + j] = cluster.building_level(bt) / 3.0

    # Global
    g_off = max_clusters * per_cluster
    if faction:
        obs[g_off:g_off + 5] = np.minimum(faction.resources / 50.0, 4.0)
        obs[g_off + 5] = faction.global_morale
        obs[g_off + 6] = min(1.0, world.step / 200.0)
        obs[g_off + 7] = len(faction.research_slots) / MAX_CONCURRENT_RESEARCH
        obs[g_off + 8] = world.faction_cluster_count(faction_id) / max(world.n_clusters, 1)

    # Append physics observations if physics is enabled
    if world.physics_states is not None and world.los_state is not None:
        phys_obs = physics_to_obs(world.physics_states, world.los_state, max_clusters)
        obs = np.concatenate([obs, phys_obs])

    return obs


def obs_size(max_clusters: int = 12, physics_enabled: bool = False) -> int:
    base = max_clusters * 15 + 9
    if physics_enabled:
        base += physics_obs_size(max_clusters)
    return base
