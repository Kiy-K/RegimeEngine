"""
military_state.py — Call of War native military state.

Full rewrite: CoW-native buildings, production queues, research, faction
economy, and cluster garrison state.  No backward compatibility with the
old StandardizedMilitaryUnit / StandardizedClusterMilitaryState system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, FrozenSet

import numpy as np
from numpy.typing import NDArray

from .cow_combat import (
    CowUnit, CowArmy, CowUnitType, CowUnitCategory, CowArmorClass,
    CowTerrain, CowDoctrine, CowProductionCost, CowBuildingType,
    CowResearchProject, RESEARCH_TREE, BUILDING_COSTS, BUILDING_BUILD_TIME,
    CATEGORY_BUILDING_REQ, MAX_BUILDING_LEVEL, required_building_level,
    get_unit_stats, get_doctrine_mod, create_unit, reset_uid_counter,
    nonlinear_production_cost, nonlinear_supply_drain,
    production_cost as base_production_cost,
    upgrade_cost as base_upgrade_cost,
)

# ─────────────────────────────────────────────────────────────────────────── #
# Resource Vector                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

N_RESOURCES = 5  # rations, steel, ammo, fuel, medical
RES_RATIONS, RES_STEEL, RES_AMMO, RES_FUEL, RES_MEDICAL = range(5)


def cost_to_array(c: CowProductionCost) -> NDArray[np.float64]:
    return np.array([c.rations, c.steel, c.ammo, c.fuel, c.medical], dtype=np.float64)


def can_afford(resources: NDArray[np.float64], cost: CowProductionCost) -> bool:
    return bool(np.all(resources >= cost_to_array(cost) - 1e-9))


def deduct(resources: NDArray[np.float64], cost: CowProductionCost) -> NDArray[np.float64]:
    return np.maximum(0.0, resources - cost_to_array(cost))

# ─────────────────────────────────────────────────────────────────────────── #
# Building                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowBuilding:
    """A building in a cluster."""
    building_type: CowBuildingType
    level: int = 0               # 0 = not built, 1-3 = operational
    construction_progress: float = 0.0  # 0→1 while building, 1 = done
    is_upgrading: bool = False

    @property
    def is_operational(self) -> bool:
        return self.level > 0 and self.construction_progress >= 1.0

    @property
    def fortification_value(self) -> float:
        if self.building_type != CowBuildingType.BUNKER or not self.is_operational:
            return 0.0
        return 0.10 * self.level  # L1=10%, L2=20%, L3=30%

    def supply_regen_bonus(self) -> float:
        if self.building_type != CowBuildingType.SUPPLY_DEPOT or not self.is_operational:
            return 0.0
        return 0.5 * self.level  # L1 +0.5, L2 +1.0, L3 +1.5 per step

    def can_produce_category(self, cat: CowUnitCategory) -> bool:
        req = CATEGORY_BUILDING_REQ.get(cat)
        return (req is not None and req == self.building_type
                and self.is_operational)

    def max_producible_level(self) -> int:
        if not self.is_operational:
            return 0
        return min(self.level, MAX_BUILDING_LEVEL)


# ─────────────────────────────────────────────────────────────────────────── #
# Production & Research Queues                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ProductionQueueItem:
    unit_type: CowUnitType
    level: int
    remaining_steps: float
    total_steps: float
    cost_paid: CowProductionCost

    @property
    def progress(self) -> float:
        return max(0.0, 1.0 - self.remaining_steps / max(self.total_steps, 0.01))

MAX_PRODUCTION_QUEUE = 3


@dataclass
class ResearchSlot:
    project: CowResearchProject
    remaining_steps: float
    total_steps: float

    @property
    def progress(self) -> float:
        return max(0.0, 1.0 - self.remaining_steps / max(self.total_steps, 0.01))

MAX_CONCURRENT_RESEARCH = 2


# ─────────────────────────────────────────────────────────────────────────── #
# Objective                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MilitaryObjective:
    objective_id: int
    name: str
    objective_type: str       # "capture", "hold", "destroy"
    target_cluster_id: int
    required_strength: float  # total HP required to capture/hold
    reward_value: float
    faction_id: int
    completion_progress: float = 0.0
    is_completed: bool = False
    hold_steps: int = 0       # how many steps held so far
    hold_required: int = 5    # how many steps must be held


# ─────────────────────────────────────────────────────────────────────────── #
# Cluster State                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowClusterState:
    """State of a single map cluster/province."""
    cluster_id: int
    terrain: CowTerrain
    owner_faction: Optional[int]

    # Garrison
    units: List[CowUnit] = field(default_factory=list)

    # Buildings (keyed by CowBuildingType)
    buildings: Dict[CowBuildingType, CowBuilding] = field(default_factory=dict)

    # Production queue (max MAX_PRODUCTION_QUEUE items)
    production_queue: List[ProductionQueueItem] = field(default_factory=list)

    # Local supply pool (consumed by garrison, replenished by depot + global)
    supply: float = 10.0

    # Rounds of combat this cluster has seen recently (decays over time)
    recent_combat_rounds: int = 0

    def __post_init__(self):
        # Ensure all building types exist
        for bt in CowBuildingType:
            if bt not in self.buildings:
                self.buildings[bt] = CowBuilding(bt, level=0, construction_progress=1.0)

    @property
    def alive_units(self) -> List[CowUnit]:
        return [u for u in self.units if u.is_alive]

    @property
    def fortification(self) -> float:
        bunker = self.buildings.get(CowBuildingType.BUNKER)
        return bunker.fortification_value if bunker else 0.0

    @property
    def supply_regen(self) -> float:
        base = 0.5  # natural regen
        depot = self.buildings.get(CowBuildingType.SUPPLY_DEPOT)
        if depot:
            base += depot.supply_regen_bonus()
        return base

    def units_of_faction(self, fid: int) -> List[CowUnit]:
        return [u for u in self.alive_units if u.faction_id == fid]

    def faction_strength(self, fid: int) -> float:
        return sum(u.hp for u in self.units_of_faction(fid))

    def dominant_faction(self) -> Optional[int]:
        strengths: Dict[int, float] = {}
        for u in self.alive_units:
            strengths[u.faction_id] = strengths.get(u.faction_id, 0.0) + u.hp
        if not strengths:
            return self.owner_faction
        return max(strengths, key=strengths.get)

    def building_level(self, bt: CowBuildingType) -> int:
        b = self.buildings.get(bt)
        return b.level if b and b.is_operational else 0

    def can_produce(self, unit_type: CowUnitType, level: int) -> bool:
        stats = get_unit_stats(unit_type, level)
        req_building = CATEGORY_BUILDING_REQ.get(stats.category)
        if req_building is None:
            return False
        b = self.buildings.get(req_building)
        if b is None or not b.is_operational:
            return False
        if b.level < required_building_level(level):
            return False
        if len(self.production_queue) >= MAX_PRODUCTION_QUEUE:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────── #
# External Modifiers (from national spirit + government)                       #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowExternalModifiers:
    """Modifiers injected from regime-level systems (national spirit, government).

    These scale CoW military dynamics without changing core combat code.
    All multipliers default to neutral (1.0); all additive mods default to 0.
    """
    # Production
    production_cost_mult: float = 1.0       # from government.military_cost
    production_speed_mult: float = 1.0      # from spirit.industry_mod (higher = faster)

    # Combat
    combat_effectiveness: float = 1.0       # combined spirit + government
    morale_mod: float = 0.0                 # additive spirit.morale_mod

    # Economy
    resource_income_mult: float = 1.0       # from spirit.resource_production_mod

    # Terrain special flags
    winter_adapted: bool = False            # from spirit.winter_adapted
    partisan_bonus: bool = False            # from spirit.partisan_bonus
    scorched_earth: bool = False            # from spirit.scorched_earth

    # Organisation (regime-level feedback)
    org_factor: float = 1.0                 # 0.5-1.0 from org_action_factor
    hazard_resistance: float = 0.0          # from spirit.hazard_resistance

    # Exhaustion
    exhaustion_rate_mod: float = 0.0        # from spirit.exhaustion_rate_mod
    exhaustion_recovery_mod: float = 0.0    # from spirit.exhaustion_recovery_mod

    # Cohesion / propaganda
    cohesion_mod: float = 0.0               # from spirit.cohesion_mod
    propaganda_mod: float = 0.0             # from spirit.propaganda_mod


def merge_modifiers(a: CowExternalModifiers, b: CowExternalModifiers) -> CowExternalModifiers:
    """Combine two modifier sources (e.g. spirit + government)."""
    return CowExternalModifiers(
        production_cost_mult=a.production_cost_mult * b.production_cost_mult,
        production_speed_mult=a.production_speed_mult * b.production_speed_mult,
        combat_effectiveness=a.combat_effectiveness * b.combat_effectiveness,
        morale_mod=a.morale_mod + b.morale_mod,
        resource_income_mult=a.resource_income_mult * b.resource_income_mult,
        winter_adapted=a.winter_adapted or b.winter_adapted,
        partisan_bonus=a.partisan_bonus or b.partisan_bonus,
        scorched_earth=a.scorched_earth or b.scorched_earth,
        org_factor=a.org_factor * b.org_factor,
        hazard_resistance=a.hazard_resistance + b.hazard_resistance,
        exhaustion_rate_mod=a.exhaustion_rate_mod + b.exhaustion_rate_mod,
        exhaustion_recovery_mod=a.exhaustion_recovery_mod + b.exhaustion_recovery_mod,
        cohesion_mod=a.cohesion_mod + b.cohesion_mod,
        propaganda_mod=a.propaganda_mod + b.propaganda_mod,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Faction State                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowFactionState:
    """Per-faction global state."""
    faction_id: int
    name: str
    doctrine: CowDoctrine

    # Global resources (rations, steel, ammo, fuel, medical)
    resources: NDArray[np.float64] = field(
        default_factory=lambda: np.array([20.0, 15.0, 10.0, 10.0, 5.0], dtype=np.float64))

    # Resource income per step (before sector production multipliers)
    base_income: NDArray[np.float64] = field(
        default_factory=lambda: np.array([3.0, 2.0, 1.5, 1.0, 0.5], dtype=np.float64))

    # Research slots (max MAX_CONCURRENT_RESEARCH)
    research_slots: List[ResearchSlot] = field(default_factory=list)

    # Completed research: set of (CowUnitType, level)
    researched: FrozenSet[Tuple[CowUnitType, int]] = field(
        default_factory=lambda: frozenset({(ut, 1) for ut in CowUnitType}))

    # Controlled cluster ids
    controlled_clusters: List[int] = field(default_factory=list)

    # Morale modifier (global, affects all units)
    global_morale: float = 1.0

    # External modifiers from national spirit + government
    external_modifiers: CowExternalModifiers = field(default_factory=CowExternalModifiers)

    def has_researched(self, unit_type: CowUnitType, level: int) -> bool:
        return (unit_type, level) in self.researched

    def max_researched_level(self, unit_type: CowUnitType) -> int:
        return max((lvl for ut, lvl in self.researched if ut == unit_type), default=0)

    def can_start_research(self, unit_type: CowUnitType, level: int) -> bool:
        if self.has_researched(unit_type, level):
            return False
        if len(self.research_slots) >= MAX_CONCURRENT_RESEARCH:
            return False
        proj = RESEARCH_TREE.get((unit_type, level))
        if proj is None:
            return False
        if proj.prerequisite and not self.has_researched(*proj.prerequisite):
            return False
        if not can_afford(self.resources, proj.cost):
            return False
        return True

    def total_units(self, world: 'CowWorldState') -> int:
        return sum(
            len(world.clusters[cid].units_of_faction(self.faction_id))
            for cid in range(len(world.clusters))
        )

    def count_of_type(self, world: 'CowWorldState', ut: CowUnitType) -> int:
        return sum(
            1 for cid in range(len(world.clusters))
            for u in world.clusters[cid].units_of_faction(self.faction_id)
            if u.stats.unit_type == ut
        )


# ─────────────────────────────────────────────────────────────────────────── #
# World State                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowWorldState:
    """Top-level game state."""
    clusters: List[CowClusterState]
    factions: Dict[int, CowFactionState]
    objectives: List[MilitaryObjective]
    adjacency: NDArray[np.bool_]   # (n_clusters, n_clusters) boolean
    step: int = 0

    # Physics integration (populated by military_wrapper if physics enabled)
    physics_states: Optional[List[Any]] = None   # List[SectorPhysicsState]
    los_state: Optional[Any] = None              # LOSState
    physics_config: Optional[Any] = None         # MapPhysicsConfig
    physics_modifiers: Optional[Dict[int, Any]] = None  # {cluster_id: PhysicsModifiers}

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def get_cluster(self, cid: int) -> CowClusterState:
        return self.clusters[cid]

    def get_faction(self, fid: int) -> CowFactionState:
        return self.factions[fid]

    def neighbors(self, cid: int) -> List[int]:
        return [j for j in range(self.n_clusters) if self.adjacency[cid, j]]

    def all_units_of_faction(self, fid: int) -> List[CowUnit]:
        out = []
        for c in self.clusters:
            out.extend(c.units_of_faction(fid))
        return out

    def faction_total_hp(self, fid: int) -> float:
        return sum(u.hp for u in self.all_units_of_faction(fid))

    def faction_cluster_count(self, fid: int) -> int:
        return sum(1 for c in self.clusters if c.owner_faction == fid)


# ─────────────────────────────────────────────────────────────────────────── #
# Initialization Helpers                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def init_world_state(
    n_clusters: int,
    faction_configs: List[Dict[str, Any]],
    adjacency: NDArray[np.bool_],
    objectives: Optional[List[Dict[str, Any]]] = None,
    cluster_terrains: Optional[List[CowTerrain]] = None,
    cluster_owners: Optional[List[Optional[int]]] = None,
    cluster_buildings: Optional[List[Dict[str, int]]] = None,
) -> CowWorldState:
    """Build a CowWorldState from config dicts (called at reset)."""
    reset_uid_counter(1)

    # Clusters
    terrains = cluster_terrains or [CowTerrain.PLAINS] * n_clusters
    owners = cluster_owners or [None] * n_clusters
    clusters = []
    for i in range(n_clusters):
        bldgs: Dict[CowBuildingType, CowBuilding] = {}
        if cluster_buildings and i < len(cluster_buildings):
            for bname, blvl in cluster_buildings[i].items():
                bt = CowBuildingType[bname]
                bldgs[bt] = CowBuilding(bt, level=blvl, construction_progress=1.0)
        clusters.append(CowClusterState(
            cluster_id=i, terrain=terrains[i],
            owner_faction=owners[i], buildings=bldgs,
        ))

    # Factions
    factions: Dict[int, CowFactionState] = {}
    for fc in faction_configs:
        fid = fc["faction_id"]
        res = np.array(fc.get("resources", [20, 15, 10, 10, 5]), dtype=np.float64)
        inc = np.array(fc.get("income", [3, 2, 1.5, 1, 0.5]), dtype=np.float64)
        factions[fid] = CowFactionState(
            faction_id=fid,
            name=fc["name"],
            doctrine=CowDoctrine[fc.get("doctrine", "AXIS")],
            resources=res, base_income=inc,
            controlled_clusters=fc.get("controlled_clusters", []),
        )

    # Objectives
    objs = []
    if objectives:
        for od in objectives:
            objs.append(MilitaryObjective(
                objective_id=od["objective_id"],
                name=od["name"],
                objective_type=od.get("objective_type", "capture"),
                target_cluster_id=od["target_cluster_id"],
                required_strength=od.get("required_strength", 50.0),
                reward_value=od.get("reward_value", 25.0),
                faction_id=od["faction_id"],
                hold_required=od.get("hold_required", 5),
            ))

    return CowWorldState(
        clusters=clusters, factions=factions,
        objectives=objs, adjacency=adjacency, step=0,
    )


def spawn_initial_units(
    world: CowWorldState,
    unit_specs: Dict[int, List[Dict[str, Any]]],
) -> CowWorldState:
    """Spawn initial units per-cluster from YAML-style spec.

    unit_specs: {cluster_id: [{unit_type: "INFANTRY", count: 3,
                                hp_frac: 0.9, faction: 1, level: 1}, ...]}
    """
    from .cow_combat import cow_type_from_legacy, _gen_uid
    for cid, specs in unit_specs.items():
        if cid >= len(world.clusters):
            continue
        cluster = world.clusters[cid]
        for spec in specs:
            ut_name = spec["unit_type"]
            cow_ut = cow_type_from_legacy(ut_name) or CowUnitType.MILITIA
            lvl = spec.get("level", 1)
            count = spec.get("count", 1)
            fid = spec.get("faction", 0)
            hp_frac = spec.get("hp_frac", 1.0)
            doctrine = world.factions[fid].doctrine if fid in world.factions else None

            stats = get_unit_stats(cow_ut, lvl)
            max_hp = stats.hitpoints
            if doctrine:
                max_hp *= (1.0 + get_doctrine_mod(doctrine, stats.category).hp_mod)

            for _ in range(count):
                uid = _gen_uid()
                u = CowUnit(
                    uid=uid, stats=stats,
                    hp=max_hp * hp_frac,
                    faction_id=fid, cluster_id=cid,
                )
                cluster.units.append(u)
    return world