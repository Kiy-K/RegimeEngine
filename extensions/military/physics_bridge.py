"""
physics_bridge.py — Bridge between physics.py and the CoW-native military system.

Maps physics terrain/weather/supply/LOS state into per-cluster modifiers that
the CoW military dynamics can consume each step.  Keeps physics.py and
military_dynamics.py decoupled — all coupling lives here.

Key exports:
  PhysicsModifiers          — per-cluster modifier struct
  COW_TERRAIN_MAP           — CowTerrain ↔ TerrainType mapping
  COW_UNIT_TYPE_MAP         — CowUnitType → physics unit-type string
  extract_cluster_modifiers — SectorPhysicsState → PhysicsModifiers
  cow_units_to_counts       — CowUnit list → Dict[str, int] for physics
  init_physics_for_world    — initialise physics states from CowWorldState + config
  step_physics_for_world    — advance physics one step, return updated states
  physics_to_obs            — flatten physics + LOS into RL observation vector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .cow_combat import CowTerrain, CowUnitType, CowUnitCategory
from .physics import (
    TerrainType, SoilType, WeatherCondition, WeatherState, Season, TimeOfDay,
    TerrainState, TerrainFeatures, SupplyState, SectorPhysicsState,
    MapPhysicsConfig, LOSState,
    MOVEMENT_COST, DEFENSE_BONUS, COVER_FACTOR, VISIBILITY_MODIFIER,
    WEATHER_MOVEMENT_PENALTY, WEATHER_AIR_SUPPORT_FACTOR,
    FUEL_CONSUMPTION, AMMO_CONSUMPTION_PER_COMBAT, SPARE_PARTS_CONSUMPTION,
    compute_combat_effectiveness, compute_artillery_effectiveness,
    compute_air_support_effectiveness, compute_movement_speed,
    compute_penetration_factor, compute_elevation_advantage,
    compute_supply_consumption, compute_supply_delivery,
    step_weather, step_all_physics,
    build_los_state, update_los_for_weather, get_los_obs,
    initialize_physics, initialize_sector_physics, load_map_physics,
    get_physics_obs,
    PHYSICS_OBS_PER_SECTOR, LOS_OBS_DIM,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Terrain Mapping: CowTerrain (4 types) ↔ TerrainType (27 types)            #
# ─────────────────────────────────────────────────────────────────────────── #

# CowTerrain → best-fit TerrainType (for physics lookups when detailed terrain unavailable)
COW_TERRAIN_TO_PHYSICS: Dict[CowTerrain, TerrainType] = {
    CowTerrain.PLAINS:    TerrainType.OPEN,
    CowTerrain.FOREST:    TerrainType.FOREST,
    CowTerrain.URBAN:     TerrainType.URBAN,
    CowTerrain.MOUNTAINS: TerrainType.MOUNTAIN,
}

# TerrainType → CowTerrain (many-to-few, for when physics creates sectors)
PHYSICS_TERRAIN_TO_COW: Dict[TerrainType, CowTerrain] = {
    TerrainType.OPEN:          CowTerrain.PLAINS,
    TerrainType.FOREST:        CowTerrain.FOREST,
    TerrainType.DENSE_FOREST:  CowTerrain.FOREST,
    TerrainType.JUNGLE:        CowTerrain.FOREST,
    TerrainType.URBAN:         CowTerrain.URBAN,
    TerrainType.URBAN_DENSE:   CowTerrain.URBAN,
    TerrainType.RUINS:         CowTerrain.URBAN,
    TerrainType.MOUNTAIN:      CowTerrain.MOUNTAINS,
    TerrainType.HILLS:         CowTerrain.PLAINS,
    TerrainType.MARSH:         CowTerrain.PLAINS,
    TerrainType.DESERT_SANDY:  CowTerrain.PLAINS,
    TerrainType.DESERT_ROCKY:  CowTerrain.PLAINS,
    TerrainType.TUNDRA:        CowTerrain.PLAINS,
    TerrainType.RIVER:         CowTerrain.PLAINS,
    TerrainType.LAKE:          CowTerrain.PLAINS,
    TerrainType.COASTAL:       CowTerrain.PLAINS,
    TerrainType.ROAD:          CowTerrain.PLAINS,
    TerrainType.BRIDGE:        CowTerrain.PLAINS,
    TerrainType.FORTIFIED:     CowTerrain.URBAN,
    TerrainType.AIRFIELD:      CowTerrain.PLAINS,
    TerrainType.PORT:          CowTerrain.URBAN,
    TerrainType.RAIL_YARD:     CowTerrain.PLAINS,
    TerrainType.FARMLAND:      CowTerrain.PLAINS,
    TerrainType.STEPPE:        CowTerrain.PLAINS,
    TerrainType.TAIGA:         CowTerrain.FOREST,
    TerrainType.BOCAGE:        CowTerrain.FOREST,
    TerrainType.VOLCANIC:      CowTerrain.MOUNTAINS,
}


# ─────────────────────────────────────────────────────────────────────────── #
# Unit Type Mapping: CowUnitType → physics string key                        #
# ─────────────────────────────────────────────────────────────────────────── #

COW_UNIT_TYPE_MAP: Dict[CowUnitType, str] = {
    # Infantry (line / specialist)
    CowUnitType.MILITIA:             "MILITIA",
    CowUnitType.INFANTRY:            "INFANTRY",
    CowUnitType.MOTORIZED_INFANTRY:  "MOTORIZED_INFANTRY",
    CowUnitType.MECHANIZED_INFANTRY: "MECHANIZED_INFANTRY",
    CowUnitType.COMMANDOS:           "SPECIAL_FORCES",
    CowUnitType.PARATROOPERS:        "INFANTRY",
    CowUnitType.GUARDS_INFANTRY:     "INFANTRY",
    CowUnitType.SKI_TROOPS:          "INFANTRY",
    CowUnitType.CAVALRY:             "CAVALRY",
    CowUnitType.PENAL_BATTALION:     "MILITIA",
    CowUnitType.RECON_INFANTRY:      "INFANTRY",
    CowUnitType.MOUNTAIN_TROOPS:     "INFANTRY",
    CowUnitType.SHOCK_TROOPS:        "INFANTRY",
    CowUnitType.ENGINEER:            "INFANTRY",
    CowUnitType.SNIPER_TEAM:         "SPECIAL_FORCES",
    # Ordnance
    CowUnitType.ANTI_TANK:           "ANTI_TANK",
    CowUnitType.ARTILLERY:           "ARTILLERY",
    CowUnitType.SP_ARTILLERY:        "ARTILLERY",
    CowUnitType.ANTI_AIR:            "ANTI_AIR",
    CowUnitType.SP_ANTI_AIR:         "ANTI_AIR",
    CowUnitType.MORTAR:              "ARTILLERY",
    CowUnitType.ROCKET_ARTILLERY:    "ROCKET_ARTILLERY",
    # Tanks / Vehicles
    CowUnitType.ARMORED_CAR:         "ARMORED_CAR",
    CowUnitType.LIGHT_TANK:          "LIGHT_TANK",
    CowUnitType.MEDIUM_TANK:         "MEDIUM_TANK",
    CowUnitType.HEAVY_TANK:          "HEAVY_TANK",
    CowUnitType.TANK_DESTROYER:      "TANK_DESTROYER",
    CowUnitType.ASSAULT_GUN:         "TANK_DESTROYER",
    CowUnitType.FLAME_TANK:          "LIGHT_TANK",
    CowUnitType.SUPPLY_TRUCK:        "MOTORIZED_INFANTRY",
    # Aircraft
    CowUnitType.INTERCEPTOR:         "FIGHTER",
    CowUnitType.TACTICAL_BOMBER:     "CAS",
    CowUnitType.ATTACK_BOMBER:       "CAS",
    CowUnitType.STRATEGIC_BOMBER:    "STRATEGIC_BOMBER",
}

def cow_unit_to_physics_key(ut: CowUnitType) -> str:
    """Map a CowUnitType to the physics.py string key."""
    return COW_UNIT_TYPE_MAP.get(ut, "INFANTRY")


# ─────────────────────────────────────────────────────────────────────────── #
# PhysicsModifiers — per-cluster modifiers extracted from physics state      #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class PhysicsModifiers:
    """
    Per-cluster modifiers computed from physics state each step.

    These feed into military_dynamics to make CoW combat, production,
    movement, and morale respond to weather, terrain detail, and supply.
    """
    # Combat
    attack_effectiveness: float = 1.0    # multiplier on attacker damage
    defense_effectiveness: float = 1.0   # multiplier on defender damage
    artillery_effectiveness: float = 1.0 # multiplier on artillery damage
    air_support_factor: float = 1.0      # multiplier on air unit damage
    elevation_advantage: float = 1.0     # attacker elevation bonus

    # Movement
    movement_speed_mult: float = 1.0     # overall movement speed modifier
    vehicle_mobility: float = 1.0        # vehicle-specific mobility

    # Weather effects
    temperature_c: float = 10.0
    equipment_reliability: float = 1.0   # [0.15, 1.0]
    weather_attrition: float = 0.0       # per-step HP drain from exposure
    visibility: float = 1.0              # [0, 1] affects detection + accuracy

    # Supply
    supply_ratio_axis: float = 1.0       # [0, 1] overall supply health
    supply_ratio_soviet: float = 1.0
    fuel_available_axis: bool = True      # False if fuel < 10 tons
    fuel_available_soviet: bool = True
    ammo_available_axis: bool = True
    ammo_available_soviet: bool = True

    # Terrain detail
    cover_factor: float = 0.1           # [0, 1] concealment
    fortification: float = 0.0          # [0, 1] built defenses
    entrenchment: float = 0.0           # [0, 1] dug-in bonus
    mines_density: float = 0.0          # [0, 1] mine threat
    defense_bonus: float = 0.0          # total terrain defense bonus

    # Environmental
    is_mud_season: bool = False
    is_extreme_cold: bool = False
    is_night: bool = False
    snow_depth_cm: float = 0.0

    # Production
    production_weather_mult: float = 1.0  # weather slows factory output


# ─────────────────────────────────────────────────────────────────────────── #
# Extract modifiers from physics state                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def extract_cluster_modifiers(
    state: SectorPhysicsState,
    winterized_axis: bool = False,
    winterized_soviet: bool = True,
) -> PhysicsModifiers:
    """
    Convert a SectorPhysicsState into PhysicsModifiers for one cluster.

    The winterized flags control equipment reliability per faction.
    """
    t = state.terrain
    w = state.weather
    sa = state.supply_axis
    ss = state.supply_soviet

    # Combat effectiveness (faction-agnostic base; per-faction winterized applied later)
    atk_eff = compute_combat_effectiveness(t, w, sa, is_attacker=True, winterized=winterized_axis)
    def_eff = compute_combat_effectiveness(t, w, ss, is_attacker=False, winterized=winterized_soviet)
    arty_eff = compute_artillery_effectiveness(w, t, distance_km=5.0)
    air_eff = compute_air_support_effectiveness(w, t, has_air_superiority=False)

    # Movement
    inf_speed = compute_movement_speed("INFANTRY", t, w, sa)
    veh_speed = compute_movement_speed("ARMOR", t, w, sa)

    # Equipment reliability (use worse of two for global modifier)
    equip_axis = w.equipment_reliability(winterized=winterized_axis)
    equip_soviet = w.equipment_reliability(winterized=winterized_soviet)
    avg_equip = (equip_axis + equip_soviet) / 2.0

    # Production weather penalty: extreme weather slows production
    prod_mult = 1.0
    if w.is_extreme_cold:
        prod_mult *= 0.7
    if w.condition in (WeatherCondition.BLIZZARD, WeatherCondition.HEAVY_SNOW):
        prod_mult *= 0.8
    if w.condition == WeatherCondition.MUD:
        prod_mult *= 0.85

    return PhysicsModifiers(
        attack_effectiveness=atk_eff,
        defense_effectiveness=def_eff,
        artillery_effectiveness=arty_eff,
        air_support_factor=air_eff,
        elevation_advantage=1.0,  # set per-engagement
        movement_speed_mult=inf_speed,
        vehicle_mobility=veh_speed,
        temperature_c=w.temperature_c,
        equipment_reliability=avg_equip,
        weather_attrition=w.attrition_rate,
        visibility=w.visibility_factor,
        supply_ratio_axis=sa.supply_ratio,
        supply_ratio_soviet=ss.supply_ratio,
        fuel_available_axis=sa.fuel_tons >= 10,
        fuel_available_soviet=ss.fuel_tons >= 10,
        ammo_available_axis=sa.ammo_tons >= 10,
        ammo_available_soviet=ss.ammo_tons >= 10,
        cover_factor=t.cover_factor,
        fortification=t.fortification,
        entrenchment=t.entrenchment,
        mines_density=t.mines_density,
        defense_bonus=t.defense_bonus,
        is_mud_season=w.mud_factor > 0.3,
        is_extreme_cold=w.is_extreme_cold,
        is_night=w.time_of_day == TimeOfDay.NIGHT,
        snow_depth_cm=w.snow_depth_cm,
        production_weather_mult=prod_mult,
    )


def extract_all_modifiers(
    physics_states: List[SectorPhysicsState],
    winterized_axis: bool = False,
    winterized_soviet: bool = True,
) -> Dict[int, PhysicsModifiers]:
    """Extract PhysicsModifiers for every cluster from the physics states."""
    return {
        s.sector_id: extract_cluster_modifiers(s, winterized_axis, winterized_soviet)
        for s in physics_states
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Unit counting bridge                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def cow_units_to_counts(
    units: list,  # List[CowUnit]
    faction_id: int,
) -> Dict[str, int]:
    """
    Convert a list of CowUnit objects to {physics_key: count} for one faction.

    Used by compute_supply_consumption and step_all_physics.
    """
    counts: Dict[str, int] = {}
    for u in units:
        if u.faction_id != faction_id:
            continue
        key = cow_unit_to_physics_key(u.stats.unit_type)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_unit_counts_per_sector(
    world,  # CowWorldState
) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Build the unit_counts_per_sector dict for step_all_physics.

    Returns: {sector_id: {"axis": {unit_type: count}, "soviet": {unit_type: count}}}
    Assumes faction 0 = axis, faction 1 = soviet.
    """
    result: Dict[int, Dict[str, Dict[str, int]]] = {}
    for cluster in world.clusters:
        cid = cluster.cluster_id
        axis_counts = cow_units_to_counts(cluster.units, faction_id=0)
        soviet_counts = cow_units_to_counts(cluster.units, faction_id=1)
        result[cid] = {"axis": axis_counts, "soviet": soviet_counts}
    return result


def build_combat_sectors(world) -> List[int]:
    """
    Determine which clusters are in active combat (both sides have units).
    """
    combat = []
    for cluster in world.clusters:
        factions_present = set()
        for u in cluster.units:
            factions_present.add(u.faction_id)
            if len(factions_present) > 1:
                combat.append(cluster.cluster_id)
                break
    return combat


# ─────────────────────────────────────────────────────────────────────────── #
# Initialization                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def init_physics_for_world(
    world,  # CowWorldState
    physics_config: Optional[MapPhysicsConfig] = None,
    yaml_path: Optional[str] = None,
) -> Tuple[List[SectorPhysicsState], LOSState, Optional[MapPhysicsConfig]]:
    """
    Initialize physics states and LOS for a CowWorldState.

    Priority:
      1. If physics_config is provided, use it
      2. If yaml_path is provided, load from YAML
      3. Fall back to Moscow defaults (initialize_sector_physics)
    """
    if physics_config is not None:
        states = initialize_physics(physics_config)
    elif yaml_path is not None:
        physics_config = load_map_physics(yaml_path)
        states = initialize_physics(physics_config)
    else:
        states = initialize_sector_physics(world.n_clusters)

    # Build LOS
    terrain_states = [s.terrain for s in states]
    los = build_los_state(terrain_states)

    return states, los, physics_config


def step_physics_for_world(
    physics_states: List[SectorPhysicsState],
    los: LOSState,
    world,  # CowWorldState
    step: int,
    rng: np.random.Generator,
    config: Optional[MapPhysicsConfig] = None,
    sabotage_factors: Optional[Dict[int, float]] = None,
) -> Tuple[List[SectorPhysicsState], LOSState, Dict[int, PhysicsModifiers]]:
    """
    Advance physics one step and return updated states + modifiers.

    Returns:
        (new_physics_states, new_los, cluster_modifiers)
    """
    # Build unit counts and combat sectors from world state
    unit_counts = build_unit_counts_per_sector(world)
    combat_sectors = build_combat_sectors(world)

    # Determine winterization from config or defaults
    winterized_axis = False
    winterized_soviet = True
    if config and config.factions:
        winterized_axis = config.factions.get("axis", {}).get("winterized", False)
        winterized_soviet = config.factions.get("soviet", {}).get("winterized", True)

    # Step all physics
    new_states = step_all_physics(
        physics_states, step, rng,
        combat_sectors=combat_sectors,
        unit_counts_per_sector=unit_counts,
        sabotage_factors=sabotage_factors,
        config=config,
    )

    # Update LOS for weather
    weather_states = [s.weather for s in new_states]
    new_los = update_los_for_weather(los, weather_states)

    # Extract per-cluster modifiers
    modifiers = extract_all_modifiers(new_states, winterized_axis, winterized_soviet)

    return new_states, new_los, modifiers


# ─────────────────────────────────────────────────────────────────────────── #
# Observation vectors                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def physics_obs_size(max_clusters: int = 12) -> int:
    """Total observation dims for physics + LOS."""
    return PHYSICS_OBS_PER_SECTOR * max_clusters + LOS_OBS_DIM


def physics_to_obs(
    physics_states: List[SectorPhysicsState],
    los: LOSState,
    max_clusters: int = 12,
) -> NDArray[np.float32]:
    """
    Flatten physics + LOS state into a single observation vector.

    Layout: [physics_obs(20 * max_clusters) | los_obs(78)]
    Total: 20 * max_clusters + 78
    """
    phys = get_physics_obs(physics_states, max_N=max_clusters)
    los_vec = get_los_obs(los, max_N=max_clusters)
    return np.concatenate([phys, los_vec])


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers for dynamics integration                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def get_cluster_physics_mods(
    modifiers: Dict[int, PhysicsModifiers],
    cluster_id: int,
) -> PhysicsModifiers:
    """Get physics modifiers for a cluster, with safe default."""
    return modifiers.get(cluster_id, PhysicsModifiers())


def compute_physics_combat_mult(
    mods: PhysicsModifiers,
    faction_id: int,
    is_attacker: bool,
) -> float:
    """
    Compute a single combat multiplier from physics for a faction in a cluster.

    Combines equipment reliability, weather, visibility, supply, and terrain.
    """
    mult = 1.0

    # Base effectiveness from physics combat model
    if is_attacker:
        mult *= mods.attack_effectiveness
    else:
        mult *= mods.defense_effectiveness

    # Supply penalty per faction
    if faction_id == 0:  # Axis
        if not mods.ammo_available_axis:
            mult *= 0.5
        if not mods.fuel_available_axis:
            mult *= 0.7
    else:  # Soviet
        if not mods.ammo_available_soviet:
            mult *= 0.5
        if not mods.fuel_available_soviet:
            mult *= 0.7

    # Mines slow and damage attackers
    if is_attacker and mods.mines_density > 0:
        mult *= (1.0 - mods.mines_density * 0.3)

    # Night combat penalty
    if mods.is_night:
        mult *= 0.85

    return float(np.clip(mult, 0.1, 2.5))


def compute_physics_attrition(
    mods: PhysicsModifiers,
    faction_id: int,
    winterized: bool = False,
) -> float:
    """
    Compute per-step weather attrition HP loss for units in a cluster.

    Returns absolute HP drain per step per unit.
    """
    base = mods.weather_attrition
    # Winterized units take less cold attrition
    if winterized and mods.is_extreme_cold:
        base *= 0.4
    # Low supply increases attrition
    supply_ratio = mods.supply_ratio_axis if faction_id == 0 else mods.supply_ratio_soviet
    if supply_ratio < 0.3:
        base += 0.003  # starvation / exposure
    return float(base)


def compute_physics_morale_effect(mods: PhysicsModifiers) -> float:
    """
    Compute per-step morale adjustment from physics conditions.

    Negative = morale drain, positive = morale boost.
    """
    effect = 0.0
    if mods.is_extreme_cold:
        effect -= 0.01
    if mods.is_mud_season:
        effect -= 0.005
    if mods.is_night:
        effect -= 0.002
    if mods.snow_depth_cm > 50:
        effect -= 0.003
    # Good supply boosts morale
    avg_supply = (mods.supply_ratio_axis + mods.supply_ratio_soviet) / 2.0
    if avg_supply > 0.7:
        effect += 0.002
    elif avg_supply < 0.3:
        effect -= 0.005
    return float(effect)
