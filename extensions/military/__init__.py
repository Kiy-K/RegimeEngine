"""
Military Extension for GravitasEngine — Call of War native system.

Full rewrite: CoW-native combat, production, research, buildings, and
expanded RL action space.  No backward compatibility with the old
StandardizedMilitaryUnit / StandardizedClusterMilitaryState system.

Legacy modules (advanced_tactics, advanced_wrapper, military_extensions,
political_interface) are NOT imported here.  They reference the old state
classes and will be updated or removed in a follow-up pass.
"""

# ── Core combat engine ────────────────────────────────────────────────────
from .cow_combat import (
    CowArmorClass, CowTerrain, CowDoctrine, CowUnitCategory, CowUnitType,
    CowDamageTable, CowTerrainMods, CowSpeedMods, CowProductionCost,
    UnitTraits, CowUnitStats, CowUnit, CowArmy,
    resolve_cow_combat, create_unit as cow_create_unit,
    create_army as cow_create_army, production_cost as cow_production_cost,
    upgrade_cost as cow_upgrade_cost, get_unit_stats as cow_get_unit_stats,
    cow_type_from_legacy, reset_uid_counter as cow_reset_uid,
    COW_UNIT_REGISTRY, DoctrineModifier, get_doctrine_mod,
    CowBuildingType, CowResearchProject, RESEARCH_TREE,
    BUILDING_COSTS, BUILDING_BUILD_TIME,
    nonlinear_production_cost, nonlinear_supply_drain,
    combat_fatigue_factor, morale_cascade,
)

# ── CoW-native state ─────────────────────────────────────────────────────
from .military_state import (
    CowBuilding, ProductionQueueItem, ResearchSlot, MilitaryObjective,
    CowClusterState, CowFactionState, CowWorldState,
    CowExternalModifiers, merge_modifiers,
    init_world_state, spawn_initial_units,
    can_afford, deduct, cost_to_array,
    N_RESOURCES, MAX_PRODUCTION_QUEUE, MAX_CONCURRENT_RESEARCH,
)

# ── Dynamics engine ───────────────────────────────────────────────────────
from .military_dynamics import (
    step_world, apply_action, compute_reward, check_victory,
    ActionType, N_ACTION_TYPES, N_UNIT_TYPES, N_BUILDING_TYPES,
    world_to_obs_array, obs_size,
)

# ── Physics bridge ──────────────────────────────────────────────────────
from .physics_bridge import (
    PhysicsModifiers, COW_TERRAIN_TO_PHYSICS, PHYSICS_TERRAIN_TO_COW,
    COW_UNIT_TYPE_MAP, cow_unit_to_physics_key,
    extract_cluster_modifiers, extract_all_modifiers,
    cow_units_to_counts, build_unit_counts_per_sector, build_combat_sectors,
    init_physics_for_world, step_physics_for_world,
    physics_obs_size, physics_to_obs,
    get_cluster_physics_mods, compute_physics_combat_mult,
    compute_physics_attrition, compute_physics_morale_effect,
)

# ── Gymnasium wrapper ─────────────────────────────────────────────────────
from .military_wrapper import MilitaryWrapper

__all__ = [
    # Combat engine
    "CowArmorClass", "CowTerrain", "CowDoctrine", "CowUnitCategory",
    "CowUnitType", "CowDamageTable", "CowTerrainMods", "CowSpeedMods",
    "CowProductionCost", "UnitTraits", "CowUnitStats", "CowUnit", "CowArmy",
    "resolve_cow_combat", "cow_create_unit", "cow_create_army",
    "cow_production_cost", "cow_upgrade_cost", "cow_get_unit_stats",
    "cow_type_from_legacy", "cow_reset_uid",
    "COW_UNIT_REGISTRY", "DoctrineModifier", "get_doctrine_mod",
    "CowBuildingType", "CowResearchProject", "RESEARCH_TREE",
    "nonlinear_production_cost", "nonlinear_supply_drain",
    "combat_fatigue_factor", "morale_cascade",
    # State
    "CowBuilding", "ProductionQueueItem", "ResearchSlot", "MilitaryObjective",
    "CowClusterState", "CowFactionState", "CowWorldState",
    "CowExternalModifiers", "merge_modifiers",
    "init_world_state", "spawn_initial_units",
    "can_afford", "deduct", "cost_to_array",
    # Dynamics
    "step_world", "apply_action", "compute_reward", "check_victory",
    "ActionType", "N_ACTION_TYPES", "N_UNIT_TYPES", "N_BUILDING_TYPES",
    "world_to_obs_array", "obs_size",
    # Physics bridge
    "PhysicsModifiers", "COW_TERRAIN_TO_PHYSICS", "PHYSICS_TERRAIN_TO_COW",
    "COW_UNIT_TYPE_MAP", "cow_unit_to_physics_key",
    "extract_cluster_modifiers", "extract_all_modifiers",
    "cow_units_to_counts", "build_unit_counts_per_sector", "build_combat_sectors",
    "init_physics_for_world", "step_physics_for_world",
    "physics_obs_size", "physics_to_obs",
    "get_cluster_physics_mods", "compute_physics_combat_mult",
    "compute_physics_attrition", "compute_physics_morale_effect",
    # Wrapper
    "MilitaryWrapper",
]
