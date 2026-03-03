"""
cow_combat.py — Call of War (WW2) style combat resolution system.

Implements the full Call of War combat model:
  1. Four armor classes: Unarmored, Light Armor, Heavy Armor, Aircraft
  2. Per-armor-class damage tables for each unit type
  3. Stack efficiency (max 10 effective units per armor class)
  4. HP-based damage efficiency (100% at full -> 20% near death)
  5. Attack vs Defence stance with separate damage values
  6. Terrain combat modifiers per unit type
  7. Doctrine bonuses (Axis / Allied / Comintern / Pan-Asian)
  8. +/-20% random damage factor
  9. Home defence bonus (+15% damage, +15% damage reduction)
  10. Fortification-based damage reduction
  11. Damage distribution by target army's armor class composition
  12. Unit levels 1-4 with scaling stats
  13. Production costs mapped to 5 resource types

Reference: https://wiki.callofwar.com/wiki/COMBAT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Enums                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class CowArmorClass(Enum):
    UNARMORED   = 0
    LIGHT_ARMOR = 1
    HEAVY_ARMOR = 2
    AIRCRAFT    = 3

class CowTerrain(Enum):
    PLAINS    = 0
    FOREST    = 1
    MOUNTAINS = 2
    URBAN     = 3
    WATER     = 4

class CowDoctrine(Enum):
    AXIS       = 0
    ALLIED     = 1
    COMINTERN  = 2
    PAN_ASIAN  = 3

class CowUnitCategory(Enum):
    INFANTRY = 0
    ORDNANCE = 1
    TANKS    = 2
    AIRCRAFT = 3
    NAVAL    = 4
    SECRET   = 5

class CowUnitType(Enum):
    # Infantry (line / specialist)
    MILITIA = auto(); INFANTRY = auto(); MOTORIZED_INFANTRY = auto()
    MECHANIZED_INFANTRY = auto(); COMMANDOS = auto(); PARATROOPERS = auto()
    GUARDS_INFANTRY = auto(); SKI_TROOPS = auto(); CAVALRY = auto()
    PENAL_BATTALION = auto()
    RECON_INFANTRY = auto(); MOUNTAIN_TROOPS = auto()
    SHOCK_TROOPS = auto(); ENGINEER = auto(); SNIPER_TEAM = auto()
    # Ordnance
    ANTI_TANK = auto(); ARTILLERY = auto(); SP_ARTILLERY = auto()
    ANTI_AIR = auto(); SP_ANTI_AIR = auto(); MORTAR = auto()
    ROCKET_ARTILLERY = auto()
    # Tanks / Vehicles
    ARMORED_CAR = auto(); LIGHT_TANK = auto(); MEDIUM_TANK = auto()
    HEAVY_TANK = auto(); TANK_DESTROYER = auto(); ASSAULT_GUN = auto()
    FLAME_TANK = auto(); SUPPLY_TRUCK = auto()
    # Aircraft
    INTERCEPTOR = auto(); TACTICAL_BOMBER = auto()
    ATTACK_BOMBER = auto(); STRATEGIC_BOMBER = auto()


# ─────────────────────────────────────────────────────────────────────────── #
# Stat Data Structures                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class CowDamageTable:
    """Damage vs each armor class for attack and defence stances."""
    atk_vs_unarmored: float = 0.0; atk_vs_light: float = 0.0
    atk_vs_heavy: float = 0.0;    atk_vs_aircraft: float = 0.0
    def_vs_unarmored: float = 0.0; def_vs_light: float = 0.0
    def_vs_heavy: float = 0.0;    def_vs_aircraft: float = 0.0

    def get_attack(self, ac: CowArmorClass) -> float:
        return (self.atk_vs_unarmored, self.atk_vs_light,
                self.atk_vs_heavy, self.atk_vs_aircraft)[ac.value]

    def get_defence(self, ac: CowArmorClass) -> float:
        return (self.def_vs_unarmored, self.def_vs_light,
                self.def_vs_heavy, self.def_vs_aircraft)[ac.value]

@dataclass(frozen=True)
class CowTerrainMods:
    plains: float = 1.0; forest: float = 1.0
    mountains: float = 1.0; urban: float = 1.0
    def get(self, t: CowTerrain) -> float:
        return (self.plains, self.forest, self.mountains, self.urban,
                1.0)[min(t.value, 4)]

@dataclass(frozen=True)
class CowSpeedMods:
    plains: float = 1.0; forest: float = 1.0
    mountains: float = 1.0; urban: float = 1.0
    def get(self, t: CowTerrain) -> float:
        return (self.plains, self.forest, self.mountains, self.urban,
                1.0)[min(t.value, 4)]

@dataclass(frozen=True)
class CowProductionCost:
    """Maps to Gravitas 5-resource: rations, steel, ammo, fuel, medical."""
    rations: float = 0.0; steel: float = 0.0; ammo: float = 0.0
    fuel: float = 0.0; medical: float = 0.0

@dataclass(frozen=True)
class UnitTraits:
    """Extended per-type traits for physics-driven simulation.

    Every unit type gets unique traits that interact with the physics engine:
    terrain specialisation, weather adaptation, engineering capabilities,
    combat specials, logistics footprint, and morale impact.
    """
    # ── Identity ──
    elite: bool = False                    # elite status (XP gain, enemy priority)
    morale_on_death: float = 0.02          # faction morale drop per unit killed

    # ── Recon & Detection ──
    recon_range: int = 0                   # extra sectors visible (0 = none)

    # ── Terrain Specialisation (additive, stacks with terrain_mods) ──
    plains_bonus: float = 0.0
    forest_bonus: float = 0.0
    mountain_bonus: float = 0.0
    urban_bonus: float = 0.0

    # ── Weather Adaptation ──
    winter_hardened: bool = False           # halves cold / blizzard attrition
    mud_resistant: bool = False             # halves mud movement penalty
    night_fighter: bool = False             # halves night combat penalty

    # ── Engineering ──
    can_entrench: bool = False              # can build field fortifications
    mine_clearing: float = 0.0             # mine clearing rate per step (0-1)
    mine_laying: float = 0.0               # mine laying rate per step   (0-1)
    bridge_building: bool = False           # can build pontoon bridges

    # ── Combat Specials ──
    suppression: float = 0.0               # reduces enemy return fire     (0-0.4)
    breakthrough: float = 0.0              # bonus vs fortified positions  (0-0.3)
    ambush: float = 0.0                    # bonus on 1st round in stealth (0-0.5)
    anti_structure: float = 0.0            # bonus vs buildings            (0-0.5)

    # ── Logistics ──
    supply_consumption: float = 1.0        # multiplier on supply drain
    fuel_consumption: float = 1.0          # multiplier on fuel usage
    can_resupply: bool = False             # extends supply to adjacent units


@dataclass(frozen=True)
class CowUnitStats:
    """Complete stat block for a unit type at a given level."""
    name: str; unit_type: CowUnitType; category: CowUnitCategory
    armor_class: CowArmorClass; level: int; hitpoints: float
    base_speed: float; attack_range: float; damage: CowDamageTable
    terrain_mods: CowTerrainMods; speed_mods: CowSpeedMods
    cost: CowProductionCost; build_time: float = 1.0
    stealth_terrain: Tuple[CowTerrain, ...] = ()
    ignores_defense_bonus: bool = False
    reveals_stealth: bool = False
    traits: UnitTraits = UnitTraits()


# ─────────────────────────────────────────────────────────────────────────── #
# Unit Registry Builder                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _s(base: float, level: int, g: float = 0.20) -> float:
    """Scale stat by level."""
    return base * (1.0 + g * (level - 1))

# Compact stat definitions: (armor_class, hp, speed, range,
#   atk: [unar, light, heavy, air], def: [unar, light, heavy, air],
#   terrain: [pl, fo, mt, ur], speed_mod: [pl, fo, mt, ur],
#   cost: [rat, stl, ammo, fuel, med], build, stealth_terrain, ignores_def, reveals)
_UNIT_DEFS: Dict[CowUnitType, Dict] = {
    # ══════════════════════════════════════════════════════════════════════ #
    #  INFANTRY — line & specialist                                        #
    # ══════════════════════════════════════════════════════════════════════ #

    # Militia — cheap local defenders, guerrilla knowledge, expendable
    CowUnitType.MILITIA: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=12, spd=3, rng=0,
        atk=[1.5, 0.3, 0.1, 0], df=[2.0, 0.5, 0.1, 0],
        tm=[0.75, 1.2, 1.15, 1.0], sm=[1.0, 0.7, 0.4, 0.9],
        cost=[1, 0.5, 0, 0, 0], bt=0.5,
        stealth=(CowTerrain.FOREST, CowTerrain.MOUNTAINS, CowTerrain.URBAN),
        traits=dict(morale_on_death=0.005, can_entrench=True,
                    forest_bonus=0.10, mountain_bonus=0.05)),

    # Infantry — standard line, can entrench, reliable everywhere
    CowUnitType.INFANTRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=20, spd=4, rng=0,
        atk=[3.0, 1.0, 0.3, 0], df=[3.5, 1.2, 0.4, 0],
        tm=[0.95, 1.1, 1.15, 1.2], sm=[1.0, 0.8, 0.5, 0.9],
        cost=[2, 1, 0, 0, 0], bt=0.8,
        traits=dict(morale_on_death=0.015, can_entrench=True)),

    # Motorized Infantry — fast road-mobile, reveals stealth, fuel hungry
    CowUnitType.MOTORIZED_INFANTRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=22, spd=8, rng=0,
        atk=[3.5, 1.2, 0.4, 0], df=[2.8, 1.0, 0.3, 0],
        tm=[1.2, 0.85, 0.65, 1.05], sm=[1.3, 0.55, 0.3, 0.9],
        cost=[2, 1.5, 0, 1, 0], bt=1.0, reveals=True,
        traits=dict(recon_range=1, fuel_consumption=1.3, morale_on_death=0.02)),

    # Mechanized Infantry — IFV mounted, light armor, combined arms
    CowUnitType.MECHANIZED_INFANTRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.LIGHT_ARMOR,
        hp=28, spd=7, rng=0,
        atk=[4.0, 2.0, 0.8, 0], df=[3.5, 1.8, 0.6, 0],
        tm=[1.15, 0.8, 0.55, 1.1], sm=[1.1, 0.5, 0.25, 0.8],
        cost=[2, 0, 1, 1.5, 0], bt=1.2,
        traits=dict(fuel_consumption=1.5, breakthrough=0.1, morale_on_death=0.025)),

    # Commandos — elite spec-ops, stealth everywhere, ignores defense, ambush masters
    CowUnitType.COMMANDOS: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=15, spd=4, rng=0,
        atk=[3.0, 2.5, 1.5, 0], df=[2.0, 1.5, 0.8, 0],
        tm=[0.9, 1.3, 1.25, 1.15], sm=[1.0, 0.9, 0.6, 0.9],
        cost=[3, 1, 0, 0, 1], bt=1.5,
        stealth=(CowTerrain.PLAINS, CowTerrain.FOREST, CowTerrain.MOUNTAINS, CowTerrain.URBAN),
        ign_def=True,
        traits=dict(elite=True, morale_on_death=0.08, night_fighter=True,
                    ambush=0.3, breakthrough=0.15, anti_structure=0.1)),

    # Paratroopers — air-droppable elite, versatile, forest stealth
    CowUnitType.PARATROOPERS: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=16, spd=4, rng=0,
        atk=[2.8, 1.5, 0.5, 0], df=[2.5, 1.2, 0.4, 0],
        tm=[1.0, 1.2, 1.1, 1.05], sm=[1.0, 0.8, 0.5, 0.9],
        cost=[3, 1.5, 0, 0, 1], bt=1.3,
        stealth=(CowTerrain.FOREST,),
        traits=dict(elite=True, morale_on_death=0.06, ambush=0.15)),

    # Guards Infantry — ELITE LINE: +35% urban, +15% forest, devastating vs soft targets
    #   Higher cost, losing them is a serious morale blow
    CowUnitType.GUARDS_INFANTRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=26, spd=4.5, rng=0,
        atk=[4.5, 1.5, 0.5, 0], df=[5.0, 2.0, 0.7, 0],
        tm=[1.05, 1.15, 1.2, 1.35], sm=[1.0, 0.8, 0.5, 0.9],
        cost=[4, 2, 1, 0, 0.5], bt=1.5,
        traits=dict(elite=True, morale_on_death=0.06, can_entrench=True,
                    urban_bonus=0.35, forest_bonus=0.15, suppression=0.1)),

    # Ski Troops — winter specialists, fast in snow/forest/mountains, winter hardened
    CowUnitType.SKI_TROOPS: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=16, spd=6, rng=0,
        atk=[2.5, 0.8, 0.2, 0], df=[2.8, 1.0, 0.2, 0],
        tm=[1.05, 1.3, 1.35, 0.8], sm=[1.2, 1.3, 1.0, 0.7],
        cost=[2, 1, 0, 0, 0.5], bt=0.9,
        stealth=(CowTerrain.FOREST, CowTerrain.MOUNTAINS),
        traits=dict(winter_hardened=True, morale_on_death=0.03, ambush=0.1,
                    forest_bonus=0.1, mountain_bonus=0.15)),

    # Cavalry — fast recon, plains specialist, mud-resistant (horses work in mud)
    CowUnitType.CAVALRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=18, spd=7, rng=0,
        atk=[2.5, 0.8, 0.2, 0], df=[2.0, 0.6, 0.1, 0],
        tm=[1.25, 0.75, 0.55, 0.65], sm=[1.3, 0.7, 0.3, 0.8],
        cost=[2, 0.5, 0, 0, 0], bt=0.7, reveals=True,
        traits=dict(recon_range=2, mud_resistant=True, morale_on_death=0.02,
                    plains_bonus=0.15)),

    # Penal Battalion — expendable, high attack, terrible defense, zero morale cost
    CowUnitType.PENAL_BATTALION: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=14, spd=4, rng=0,
        atk=[3.5, 1.2, 0.5, 0], df=[1.5, 0.5, 0.2, 0],
        tm=[1.0, 1.0, 1.0, 1.1], sm=[1.0, 0.8, 0.5, 0.9],
        cost=[1, 0, 0.5, 0, 0], bt=0.4,
        traits=dict(morale_on_death=0.0, breakthrough=0.1, mine_clearing=0.1)),

    # Recon Infantry — light scouts, fast, stealthy, high recon, minimal combat
    CowUnitType.RECON_INFANTRY: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=12, spd=6, rng=0,
        atk=[1.5, 0.5, 0.1, 0], df=[1.8, 0.5, 0.1, 0],
        tm=[1.0, 1.2, 1.1, 1.0], sm=[1.1, 0.9, 0.6, 0.9],
        cost=[1.5, 0.5, 0, 0, 0], bt=0.6,
        stealth=(CowTerrain.FOREST, CowTerrain.MOUNTAINS, CowTerrain.URBAN),
        reveals=True,
        traits=dict(recon_range=2, night_fighter=True, ambush=0.15,
                    morale_on_death=0.01, supply_consumption=0.6)),

    # Mountain Troops — alpine specialists, +40% mountain, +20% forest, winter adapted
    CowUnitType.MOUNTAIN_TROOPS: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=22, spd=4, rng=0,
        atk=[3.5, 1.2, 0.4, 0], df=[4.0, 1.5, 0.5, 0],
        tm=[0.85, 1.2, 1.4, 1.0], sm=[0.9, 1.0, 1.0, 0.8],
        cost=[3, 1, 0.5, 0, 0.5], bt=1.1,
        stealth=(CowTerrain.MOUNTAINS,),
        traits=dict(elite=True, winter_hardened=True, can_entrench=True,
                    morale_on_death=0.04, mountain_bonus=0.40, forest_bonus=0.20,
                    ambush=0.15)),

    # Shock Troops — breakthrough infantry, high attack, suppression, elite
    CowUnitType.SHOCK_TROOPS: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=22, spd=4, rng=0,
        atk=[4.5, 2.0, 0.8, 0], df=[3.0, 1.2, 0.4, 0],
        tm=[1.1, 1.0, 0.9, 1.2], sm=[1.0, 0.7, 0.5, 0.9],
        cost=[3.5, 1.5, 1, 0, 0.5], bt=1.3,
        traits=dict(elite=True, breakthrough=0.25, suppression=0.2,
                    morale_on_death=0.05, urban_bonus=0.1)),

    # Engineer — combat engineer: fortify, mine clear/lay, bridge, anti-structure
    CowUnitType.ENGINEER: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=18, spd=3.5, rng=0,
        atk=[2.5, 0.8, 0.3, 0], df=[3.0, 1.0, 0.3, 0],
        tm=[0.9, 1.0, 1.0, 1.2], sm=[1.0, 0.8, 0.5, 0.9],
        cost=[2, 1.5, 0.5, 0, 0], bt=1.0,
        traits=dict(can_entrench=True, mine_clearing=0.3, mine_laying=0.2,
                    bridge_building=True, anti_structure=0.3, morale_on_death=0.02)),

    # Sniper Team — precision anti-personnel, stealth everywhere, suppression
    CowUnitType.SNIPER_TEAM: dict(
        cat=CowUnitCategory.INFANTRY, ac=CowArmorClass.UNARMORED,
        hp=8, spd=3, rng=2,
        atk=[4.0, 0.5, 0.1, 0], df=[0.5, 0.2, 0.05, 0],
        tm=[0.9, 1.3, 1.2, 1.4], sm=[1.0, 0.9, 0.6, 0.9],
        cost=[2, 0.5, 0.5, 0, 0.5], bt=1.2,
        stealth=(CowTerrain.PLAINS, CowTerrain.FOREST, CowTerrain.MOUNTAINS, CowTerrain.URBAN),
        traits=dict(elite=True, night_fighter=True, suppression=0.15,
                    ambush=0.4, morale_on_death=0.03, supply_consumption=0.4,
                    urban_bonus=0.2, forest_bonus=0.15)),

    # ══════════════════════════════════════════════════════════════════════ #
    #  ORDNANCE — fire support                                             #
    # ══════════════════════════════════════════════════════════════════════ #

    # Anti-Tank Gun — ambush anti-armor, entrenched, deadly in forest/urban
    CowUnitType.ANTI_TANK: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.UNARMORED,
        hp=14, spd=3.5, rng=0,
        atk=[0.5, 4.0, 3.5, 0], df=[0.8, 4.5, 4.0, 0],
        tm=[0.9, 1.2, 1.1, 1.2], sm=[1.0, 0.7, 0.4, 0.9],
        cost=[1.5, 1.5, 1, 0, 0], bt=0.8,
        stealth=(CowTerrain.FOREST, CowTerrain.URBAN),
        traits=dict(ambush=0.2, can_entrench=True, morale_on_death=0.02)),

    # Artillery — long range, suppression, area denial
    CowUnitType.ARTILLERY: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.UNARMORED,
        hp=10, spd=3, rng=3,
        atk=[4.0, 2.5, 3.0, 0], df=[0.5, 0.3, 0.3, 0],
        tm=[1.0, 1.1, 1.2, 0.9], sm=[1.0, 0.6, 0.3, 0.8],
        cost=[1.5, 2, 1.5, 0, 0], bt=1.2,
        traits=dict(suppression=0.25, morale_on_death=0.03,
                    anti_structure=0.15)),

    # SP Artillery — self-propelled, mobile fire support
    CowUnitType.SP_ARTILLERY: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.LIGHT_ARMOR,
        hp=16, spd=6, rng=3,
        atk=[4.5, 3.0, 3.5, 0], df=[1.0, 0.8, 0.5, 0],
        tm=[1.1, 1.0, 0.8, 0.9], sm=[1.1, 0.5, 0.2, 0.8],
        cost=[0, 2, 2, 1.5, 0], bt=1.4,
        traits=dict(suppression=0.2, fuel_consumption=1.3, morale_on_death=0.03,
                    anti_structure=0.15)),

    # Anti-Air Gun — ground-based air denial
    CowUnitType.ANTI_AIR: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.UNARMORED,
        hp=12, spd=3.5, rng=0,
        atk=[0.5, 0.3, 0.1, 5.0], df=[0.8, 0.5, 0.2, 5.5],
        tm=[1.0, 1.0, 1.0, 1.1], sm=[1.0, 0.7, 0.4, 0.9],
        cost=[1.5, 1.5, 1, 0, 0], bt=0.7,
        traits=dict(morale_on_death=0.02, can_entrench=True)),

    # SP Anti-Air — mobile air defense screen
    CowUnitType.SP_ANTI_AIR: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.LIGHT_ARMOR,
        hp=18, spd=6, rng=0,
        atk=[1.0, 0.5, 0.2, 5.5], df=[1.2, 0.8, 0.3, 6.0],
        tm=[1.0, 0.9, 0.7, 1.0], sm=[1.1, 0.5, 0.2, 0.8],
        cost=[0, 2, 1.5, 1, 0], bt=1.0,
        traits=dict(fuel_consumption=1.2, morale_on_death=0.02)),

    # Mortar — short range indirect, mountain/urban specialist, suppression
    CowUnitType.MORTAR: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.UNARMORED,
        hp=10, spd=3.5, rng=1.5,
        atk=[3.0, 1.0, 0.5, 0], df=[0.5, 0.3, 0.1, 0],
        tm=[1.0, 1.1, 1.3, 1.15], sm=[1.0, 0.7, 0.5, 0.9],
        cost=[1, 1, 1, 0, 0], bt=0.6,
        traits=dict(suppression=0.15, can_entrench=True, morale_on_death=0.015,
                    mountain_bonus=0.1, urban_bonus=0.05)),

    # Rocket Artillery — massed barrage, highest suppression, inaccurate
    CowUnitType.ROCKET_ARTILLERY: dict(
        cat=CowUnitCategory.ORDNANCE, ac=CowArmorClass.UNARMORED,
        hp=10, spd=3.5, rng=4,
        atk=[5.0, 2.0, 1.5, 0], df=[0.3, 0.2, 0.1, 0],
        tm=[1.0, 0.9, 0.8, 0.9], sm=[1.0, 0.6, 0.3, 0.8],
        cost=[1.5, 2, 2.5, 0, 0], bt=1.3,
        traits=dict(suppression=0.35, morale_on_death=0.03,
                    anti_structure=0.2)),

    # ══════════════════════════════════════════════════════════════════════ #
    #  TANKS / VEHICLES                                                    #
    # ══════════════════════════════════════════════════════════════════════ #

    # Armored Car — fast RECON vehicle, reveals stealth, fuel efficient
    CowUnitType.ARMORED_CAR: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.LIGHT_ARMOR,
        hp=20, spd=10, rng=0,
        atk=[2.5, 1.5, 0.3, 0], df=[3.0, 2.0, 0.5, 0],
        tm=[1.25, 0.65, 0.4, 0.85], sm=[1.3, 0.5, 0.2, 0.9],
        cost=[0, 1.5, 0.5, 1, 0], bt=0.7, reveals=True,
        traits=dict(recon_range=2, fuel_consumption=0.8, morale_on_death=0.015,
                    plains_bonus=0.1)),

    # Light Tank — fast flanker, recon capable, good on plains
    CowUnitType.LIGHT_TANK: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.LIGHT_ARMOR,
        hp=25, spd=8, rng=0,
        atk=[3.0, 3.0, 1.0, 0], df=[2.0, 2.0, 0.5, 0],
        tm=[1.3, 0.7, 0.4, 0.8], sm=[1.2, 0.5, 0.2, 0.8],
        cost=[0, 2, 1, 2, 0], bt=1.2,
        traits=dict(recon_range=1, breakthrough=0.1, fuel_consumption=1.2,
                    morale_on_death=0.025, plains_bonus=0.1)),

    # Medium Tank — main battle tank, balanced offensive/defensive
    CowUnitType.MEDIUM_TANK: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.HEAVY_ARMOR,
        hp=35, spd=6, rng=0,
        atk=[3.5, 4.0, 2.5, 0], df=[2.5, 3.0, 1.5, 0],
        tm=[1.3, 0.6, 0.3, 0.7], sm=[1.1, 0.4, 0.15, 0.7],
        cost=[0, 3, 2, 2.5, 0], bt=1.5,
        traits=dict(breakthrough=0.15, fuel_consumption=1.5,
                    morale_on_death=0.035)),

    # Heavy Tank — breakthrough monster, highest armor, slow, fuel guzzler, ELITE
    CowUnitType.HEAVY_TANK: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.HEAVY_ARMOR,
        hp=50, spd=4, rng=0,
        atk=[4.0, 4.5, 3.5, 0], df=[3.5, 4.0, 3.0, 0],
        tm=[1.2, 0.5, 0.2, 0.6], sm=[1.0, 0.3, 0.1, 0.6],
        cost=[0, 4, 2.5, 3, 1], bt=2.0,
        traits=dict(elite=True, breakthrough=0.25, fuel_consumption=2.0,
                    morale_on_death=0.06, suppression=0.1)),

    # Tank Destroyer — ambush anti-armor, devastating from concealment
    CowUnitType.TANK_DESTROYER: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.HEAVY_ARMOR,
        hp=35, spd=5, rng=0,
        atk=[1.5, 3.5, 5.0, 0], df=[2.0, 4.0, 5.5, 0],
        tm=[1.1, 1.0, 0.6, 0.9], sm=[1.0, 0.4, 0.15, 0.7],
        cost=[0, 3, 2, 2, 0.5], bt=1.5,
        stealth=(CowTerrain.FOREST, CowTerrain.URBAN),
        traits=dict(ambush=0.25, fuel_consumption=1.3, morale_on_death=0.03)),

    # Assault Gun — infantry support, anti-structure
    CowUnitType.ASSAULT_GUN: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.HEAVY_ARMOR,
        hp=30, spd=5.5, rng=0,
        atk=[4.0, 3.0, 2.0, 0], df=[3.0, 2.5, 1.5, 0],
        tm=[1.2, 0.7, 0.3, 0.85], sm=[1.0, 0.4, 0.15, 0.7],
        cost=[0, 2.5, 1.5, 2, 0], bt=1.3,
        traits=dict(anti_structure=0.2, suppression=0.1, fuel_consumption=1.3,
                    morale_on_death=0.025)),

    # Flame Tank — anti-fortification terror weapon, short range, high suppression
    CowUnitType.FLAME_TANK: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.LIGHT_ARMOR,
        hp=22, spd=5, rng=0,
        atk=[5.0, 1.5, 0.3, 0], df=[1.5, 1.0, 0.3, 0],
        tm=[0.9, 1.1, 0.7, 1.4], sm=[1.0, 0.5, 0.2, 0.8],
        cost=[0, 2, 0, 2.5, 0], bt=1.2,
        traits=dict(anti_structure=0.5, breakthrough=0.3, suppression=0.3,
                    fuel_consumption=2.5, morale_on_death=0.03,
                    urban_bonus=0.25)),

    # Supply Truck — unarmed logistics vehicle, can resupply adjacent sectors
    CowUnitType.SUPPLY_TRUCK: dict(
        cat=CowUnitCategory.TANKS, ac=CowArmorClass.UNARMORED,
        hp=10, spd=8, rng=0,
        atk=[0, 0, 0, 0], df=[0.2, 0.1, 0, 0],
        tm=[1.0, 0.7, 0.4, 0.9], sm=[1.2, 0.5, 0.2, 0.9],
        cost=[1, 0.5, 0, 1, 0], bt=0.5,
        traits=dict(can_resupply=True, supply_consumption=0.0,
                    fuel_consumption=0.8, morale_on_death=0.005)),

    # ══════════════════════════════════════════════════════════════════════ #
    #  AIRCRAFT                                                            #
    # ══════════════════════════════════════════════════════════════════════ #

    # Interceptor — air superiority, fast, recon
    CowUnitType.INTERCEPTOR: dict(
        cat=CowUnitCategory.AIRCRAFT, ac=CowArmorClass.AIRCRAFT,
        hp=15, spd=12, rng=5,
        atk=[1.5, 1.0, 0.3, 5.0], df=[0.5, 0.3, 0.1, 4.0],
        tm=[1, 1, 1, 1], sm=[1, 1, 1, 1],
        cost=[0, 2, 0, 2, 1], bt=1.0, reveals=True,
        traits=dict(recon_range=3, fuel_consumption=2.0, morale_on_death=0.04)),

    # Tactical Bomber — close air support, devastating vs soft targets
    CowUnitType.TACTICAL_BOMBER: dict(
        cat=CowUnitCategory.AIRCRAFT, ac=CowArmorClass.AIRCRAFT,
        hp=18, spd=10, rng=6,
        atk=[5.0, 2.0, 1.0, 0.5], df=[1.0, 0.5, 0.3, 0.5],
        tm=[1, 1, 1, 1], sm=[1, 1, 1, 1],
        cost=[0, 2.5, 0, 2.5, 1], bt=1.3,
        traits=dict(suppression=0.2, fuel_consumption=2.0,
                    morale_on_death=0.04)),

    # Attack Bomber — anti-armor / anti-ship specialist
    CowUnitType.ATTACK_BOMBER: dict(
        cat=CowUnitCategory.AIRCRAFT, ac=CowArmorClass.AIRCRAFT,
        hp=14, spd=11, rng=5,
        atk=[2.0, 3.5, 4.5, 1.0], df=[0.5, 0.8, 1.0, 1.0],
        tm=[1, 1, 1, 1], sm=[1, 1, 1, 1],
        cost=[0, 2, 0, 2.5, 1], bt=1.2,
        traits=dict(fuel_consumption=2.0, morale_on_death=0.04)),

    # Strategic Bomber — area bombing, anti-infrastructure, long range
    CowUnitType.STRATEGIC_BOMBER: dict(
        cat=CowUnitCategory.AIRCRAFT, ac=CowArmorClass.AIRCRAFT,
        hp=22, spd=8, rng=8,
        atk=[6.0, 3.0, 2.0, 0.3], df=[0.5, 0.3, 0.2, 0.3],
        tm=[1, 1, 1, 1], sm=[1, 1, 1, 1],
        cost=[0, 3, 0, 3, 1.5], bt=1.8,
        traits=dict(anti_structure=0.4, suppression=0.25, fuel_consumption=2.5,
                    morale_on_death=0.05)),
}


def _build_stats(ut: CowUnitType, d: Dict, level: int) -> CowUnitStats:
    a, df = d['atk'], d['df']
    traits_kw = d.get('traits', {})
    return CowUnitStats(
        name=ut.name.replace('_', ' ').title(), unit_type=ut,
        category=d['cat'], armor_class=d['ac'], level=level,
        hitpoints=_s(d['hp'], level, 0.15),
        base_speed=d['spd'], attack_range=d['rng'],
        damage=CowDamageTable(
            atk_vs_unarmored=_s(a[0], level), atk_vs_light=_s(a[1], level),
            atk_vs_heavy=_s(a[2], level), atk_vs_aircraft=_s(a[3], level),
            def_vs_unarmored=_s(df[0], level), def_vs_light=_s(df[1], level),
            def_vs_heavy=_s(df[2], level), def_vs_aircraft=_s(df[3], level),
        ),
        terrain_mods=CowTerrainMods(*d['tm']),
        speed_mods=CowSpeedMods(*d['sm']),
        cost=CowProductionCost(*d['cost']),
        build_time=_s(d.get('bt', 1.0), level, 0.10),
        stealth_terrain=d.get('stealth', ()),
        ignores_defense_bonus=d.get('ign_def', False),
        reveals_stealth=d.get('reveals', False),
        traits=UnitTraits(**traits_kw),
    )


COW_UNIT_REGISTRY: Dict[Tuple[CowUnitType, int], CowUnitStats] = {
    (ut, lvl): _build_stats(ut, d, lvl)
    for ut, d in _UNIT_DEFS.items()
    for lvl in range(1, 5)
}


def get_unit_stats(unit_type: CowUnitType, level: int = 1) -> CowUnitStats:
    return COW_UNIT_REGISTRY[(unit_type, max(1, min(4, level)))]


# ─────────────────────────────────────────────────────────────────────────── #
# Doctrine Modifiers                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class DoctrineModifier:
    hp_mod: float = 0.0; damage_mod: float = 0.0
    speed_mod: float = 0.0; cost_mod: float = 0.0

DOCTRINE_MODIFIERS: Dict[CowDoctrine, Dict[CowUnitCategory, DoctrineModifier]] = {
    CowDoctrine.AXIS: {
        CowUnitCategory.INFANTRY: DoctrineModifier(hp_mod=0.05, damage_mod=0.05),
        CowUnitCategory.TANKS:    DoctrineModifier(hp_mod=0.10, damage_mod=0.10, speed_mod=0.10),
        CowUnitCategory.AIRCRAFT: DoctrineModifier(damage_mod=0.05, speed_mod=0.05),
    },
    CowDoctrine.ALLIED: {
        CowUnitCategory.AIRCRAFT: DoctrineModifier(hp_mod=0.10, damage_mod=0.10),
        CowUnitCategory.TANKS:    DoctrineModifier(hp_mod=0.05),
    },
    CowDoctrine.COMINTERN: {
        CowUnitCategory.INFANTRY: DoctrineModifier(hp_mod=0.10, damage_mod=0.10, cost_mod=-0.10),
        CowUnitCategory.ORDNANCE: DoctrineModifier(damage_mod=0.15, cost_mod=-0.05),
    },
    CowDoctrine.PAN_ASIAN: {
        CowUnitCategory.INFANTRY: DoctrineModifier(hp_mod=0.15, damage_mod=0.05, cost_mod=-0.15),
    },
}

def get_doctrine_mod(doctrine: CowDoctrine, cat: CowUnitCategory) -> DoctrineModifier:
    return DOCTRINE_MODIFIERS.get(doctrine, {}).get(cat, DoctrineModifier())


# ─────────────────────────────────────────────────────────────────────────── #
# Runtime Unit & Army                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class CowUnit:
    """Live unit instance with current HP."""
    uid: int; stats: CowUnitStats; hp: float
    faction_id: int = 0; cluster_id: int = 0
    experience: float = 0.0; morale: float = 1.0

    @property
    def is_alive(self) -> bool: return self.hp > 0.01

    @property
    def hp_fraction(self) -> float:
        return max(0.0, min(1.0, self.hp / self.stats.hitpoints)) if self.stats.hitpoints > 0 else 0.0

    @property
    def damage_efficiency(self) -> float:
        """Linear 100% (full) -> 20% (near death)."""
        return 0.20 + 0.80 * self.hp_fraction

    @property
    def speed(self) -> float:
        s = self.stats.base_speed
        if self.hp_fraction < 0.5: s *= 0.5
        return s

    def effective_damage(self, stance: str, armor: CowArmorClass,
                         terrain: CowTerrain = CowTerrain.PLAINS,
                         doctrine: Optional[CowDoctrine] = None) -> float:
        d = self.stats.damage
        base = d.get_attack(armor) if stance == "attack" else d.get_defence(armor)
        base *= self.stats.terrain_mods.get(terrain)
        base *= self.damage_efficiency
        if doctrine:
            base *= (1.0 + get_doctrine_mod(doctrine, self.stats.category).damage_mod)
        base *= (1.0 + 0.02 * self.experience)
        return base

    def apply_damage(self, amount: float) -> 'CowUnit':
        return CowUnit(uid=self.uid, stats=self.stats, hp=max(0, self.hp - amount),
                        faction_id=self.faction_id, cluster_id=self.cluster_id,
                        experience=self.experience, morale=self.morale)

    def heal(self, amount: float) -> 'CowUnit':
        return CowUnit(uid=self.uid, stats=self.stats,
                        hp=min(self.stats.hitpoints, self.hp + amount),
                        faction_id=self.faction_id, cluster_id=self.cluster_id,
                        experience=self.experience, morale=self.morale)

    def gain_xp(self, amount: float = 0.1) -> 'CowUnit':
        return CowUnit(uid=self.uid, stats=self.stats, hp=self.hp,
                        faction_id=self.faction_id, cluster_id=self.cluster_id,
                        experience=min(10.0, self.experience + amount),
                        morale=self.morale)

    def to_dict(self) -> Dict:
        return {"uid": self.uid, "type": self.stats.unit_type.name,
                "level": self.stats.level, "hp": round(self.hp, 2),
                "max_hp": round(self.stats.hitpoints, 2),
                "armor": self.stats.armor_class.name, "faction": self.faction_id,
                "cluster": self.cluster_id, "xp": round(self.experience, 2)}


@dataclass
class CowArmy:
    """Group of units — the fundamental combat entity."""
    units: List[CowUnit]; faction_id: int = 0; cluster_id: int = 0

    @property
    def alive_units(self) -> List[CowUnit]:
        return [u for u in self.units if u.is_alive]

    @property
    def unit_count(self) -> int: return len(self.alive_units)

    @property
    def total_hp(self) -> float: return sum(u.hp for u in self.alive_units)

    @property
    def movement_speed(self) -> float:
        a = self.alive_units
        return min(u.speed for u in a) if a else 0.0

    def armor_class_composition(self) -> Dict[CowArmorClass, float]:
        alive = self.alive_units
        if not alive: return {ac: 0.0 for ac in CowArmorClass}
        counts = {ac: 0 for ac in CowArmorClass}
        for u in alive: counts[u.stats.armor_class] += 1
        total = len(alive)
        return {ac: c / total for ac, c in counts.items()}

    def compute_damage_potential(self, stance: str,
                                  terrain: CowTerrain = CowTerrain.PLAINS,
                                  doctrine: Optional[CowDoctrine] = None,
                                  ) -> Dict[CowArmorClass, float]:
        """Sum damage vs each armor class — top 10 per class (stack limit)."""
        pot = {ac: 0.0 for ac in CowArmorClass}
        for tgt_ac in CowArmorClass:
            dl = sorted([u.effective_damage(stance, tgt_ac, terrain, doctrine)
                         for u in self.alive_units], reverse=True)
            pot[tgt_ac] = sum(dl[:10])
        return pot

    def summary(self) -> Dict:
        return {"faction": self.faction_id, "cluster": self.cluster_id,
                "units": self.unit_count, "hp": round(self.total_hp, 1),
                "speed": round(self.movement_speed, 1)}


# ─────────────────────────────────────────────────────────────────────────── #
# Combat Resolution                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

RANDOM_FACTOR = 0.20
HOME_DEF_BONUS = 0.15


def resolve_cow_combat(
    attacker: CowArmy, defender: CowArmy,
    terrain: CowTerrain = CowTerrain.PLAINS,
    atk_doctrine: Optional[CowDoctrine] = None,
    def_doctrine: Optional[CowDoctrine] = None,
    fortification: float = 0.0,
    defender_is_home: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[CowArmy, CowArmy, Dict]:
    """Resolve one round of CoW-style combat.

    Phases: 1) damage potential  2) random factor  3) protection
            4) distribute by armor class  5) apply to units
    """
    if rng is None: rng = np.random.default_rng()
    log: Dict[str, Any] = {"terrain": terrain.name}

    if attacker.unit_count == 0 or defender.unit_count == 0:
        log["result"] = "no_combat"; return attacker, defender, log

    # 1. Damage potential
    atk_pot = attacker.compute_damage_potential("attack", terrain, atk_doctrine)
    def_pot = defender.compute_damage_potential("defence", terrain, def_doctrine)
    if defender_is_home:
        for ac in CowArmorClass: def_pot[ac] *= (1.0 + HOME_DEF_BONUS)

    # 2. Random factor
    ar, dr = (1 + rng.uniform(-RANDOM_FACTOR, RANDOM_FACTOR),
              1 + rng.uniform(-RANDOM_FACTOR, RANDOM_FACTOR))
    for ac in CowArmorClass:
        atk_pot[ac] *= ar; def_pot[ac] *= dr

    # 3. Protection
    prot = min(0.75, fortification + (HOME_DEF_BONUS if defender_is_home else 0))
    any_ignores = any(u.stats.ignores_defense_bonus for u in attacker.alive_units)
    if not any_ignores:
        for ac in CowArmorClass: atk_pot[ac] *= (1.0 - prot)

    # 4. Distribute by composition
    def_comp = defender.armor_class_composition()
    atk_comp = attacker.armor_class_composition()
    dmg_to_def = {ac: atk_pot[ac] * def_comp[ac] for ac in CowArmorClass}
    dmg_to_atk = {ac: def_pot[ac] * atk_comp[ac] for ac in CowArmorClass}

    # 5. Apply
    new_def = _apply_dmg(defender.alive_units, dmg_to_def)
    new_atk = _apply_dmg(attacker.alive_units, dmg_to_atk)
    new_def = [u.gain_xp(0.1) for u in new_def]
    new_atk = [u.gain_xp(0.1) for u in new_atk]

    ua = CowArmy(new_atk, attacker.faction_id, attacker.cluster_id)
    ud = CowArmy(new_def, defender.faction_id, defender.cluster_id)

    log.update({
        "atk_hp": round(ua.total_hp, 1), "def_hp": round(ud.total_hp, 1),
        "atk_losses": attacker.unit_count - ua.unit_count,
        "def_losses": defender.unit_count - ud.unit_count,
        "result": ("attacker_wins" if ud.unit_count == 0
                    else "defender_wins" if ua.unit_count == 0 else "ongoing"),
    })
    return ua, ud, log


def _apply_dmg(units: List[CowUnit], dmg_by_ac: Dict[CowArmorClass, float]) -> List[CowUnit]:
    by_ac: Dict[CowArmorClass, List[CowUnit]] = {ac: [] for ac in CowArmorClass}
    for u in units:
        if u.is_alive: by_ac[u.stats.armor_class].append(u)
    result = []
    for ac in CowArmorClass:
        grp = by_ac[ac]; td = dmg_by_ac.get(ac, 0.0)
        if not grp or td <= 0: result.extend(grp); continue
        per = td / len(grp)
        for u in grp:
            nu = u.apply_damage(per)
            if nu.hp_fraction < 0.5 and per >= nu.hp:
                nu = CowUnit(u.uid, u.stats, 0.0, u.faction_id, u.cluster_id, u.experience, u.morale)
            result.append(nu)
    return result


# ─────────────────────────────────────────────────────────────────────────── #
# Factory Helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

_next_uid = 0

def _gen_uid() -> int:
    global _next_uid; _next_uid += 1; return _next_uid

def reset_uid_counter(start: int = 0) -> None:
    global _next_uid; _next_uid = start

def create_unit(unit_type: CowUnitType, level: int = 1,
                faction_id: int = 0, cluster_id: int = 0,
                doctrine: Optional[CowDoctrine] = None) -> CowUnit:
    stats = get_unit_stats(unit_type, level)
    hp = stats.hitpoints
    if doctrine:
        hp *= (1.0 + get_doctrine_mod(doctrine, stats.category).hp_mod)
    return CowUnit(_gen_uid(), stats, hp, faction_id, cluster_id)

def create_army(composition: List[Tuple[CowUnitType, int, int]],
                faction_id: int = 0, cluster_id: int = 0,
                doctrine: Optional[CowDoctrine] = None) -> CowArmy:
    """Create army from [(CowUnitType, level, count), ...]."""
    units = []
    for ut, lvl, cnt in composition:
        for _ in range(cnt):
            units.append(create_unit(ut, lvl, faction_id, cluster_id, doctrine))
    return CowArmy(units, faction_id, cluster_id)

def production_cost(unit_type: CowUnitType, level: int = 1,
                    doctrine: Optional[CowDoctrine] = None) -> CowProductionCost:
    c = get_unit_stats(unit_type, level).cost
    if doctrine:
        f = 1.0 + get_doctrine_mod(doctrine, get_unit_stats(unit_type, level).category).cost_mod
        return CowProductionCost(c.rations*f, c.steel*f, c.ammo*f, c.fuel*f, c.medical*f)
    return c

def upgrade_cost(unit_type: CowUnitType, target_level: int,
                 doctrine: Optional[CowDoctrine] = None) -> CowProductionCost:
    """50% of target level cost (CoW rule)."""
    c = production_cost(unit_type, target_level, doctrine)
    return CowProductionCost(c.rations*.5, c.steel*.5, c.ammo*.5, c.fuel*.5, c.medical*.5)


# ─────────────────────────────────────────────────────────────────────────── #
# Legacy Mapping                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

_LEGACY_MAP = {
    "MILITIA": CowUnitType.MILITIA, "INFANTRY": CowUnitType.INFANTRY,
    "MOTORIZED_INFANTRY": CowUnitType.MOTORIZED_INFANTRY,
    "MECHANIZED_INFANTRY": CowUnitType.MECHANIZED_INFANTRY,
    "COMMANDOS": CowUnitType.COMMANDOS, "PARATROOPERS": CowUnitType.PARATROOPERS,
    "GUARDS_INFANTRY": CowUnitType.GUARDS_INFANTRY,
    "SKI_TROOPS": CowUnitType.SKI_TROOPS, "CAVALRY": CowUnitType.CAVALRY,
    "PENAL_BATTALION": CowUnitType.PENAL_BATTALION,
    "RECON_INFANTRY": CowUnitType.RECON_INFANTRY,
    "MOUNTAIN_TROOPS": CowUnitType.MOUNTAIN_TROOPS,
    "SHOCK_TROOPS": CowUnitType.SHOCK_TROOPS,
    "ENGINEER": CowUnitType.ENGINEER,
    "SNIPER_TEAM": CowUnitType.SNIPER_TEAM,
    "ANTI_TANK": CowUnitType.ANTI_TANK, "ARTILLERY": CowUnitType.ARTILLERY,
    "SELF_PROPELLED_ARTILLERY": CowUnitType.SP_ARTILLERY,
    "ANTI_AIR": CowUnitType.ANTI_AIR, "MORTAR": CowUnitType.MORTAR,
    "ROCKET_ARTILLERY": CowUnitType.ROCKET_ARTILLERY,
    "ARMORED_CAR": CowUnitType.ARMORED_CAR,
    "LIGHT_TANK": CowUnitType.LIGHT_TANK,
    "MEDIUM_TANK": CowUnitType.MEDIUM_TANK,
    "HEAVY_TANK": CowUnitType.HEAVY_TANK,
    "TANK_DESTROYER": CowUnitType.TANK_DESTROYER,
    "ASSAULT_GUN": CowUnitType.ASSAULT_GUN,
    "FLAME_TANK": CowUnitType.FLAME_TANK,
    "SUPPLY_TRUCK": CowUnitType.SUPPLY_TRUCK,
    "FIGHTER": CowUnitType.INTERCEPTOR, "CAS": CowUnitType.ATTACK_BOMBER,
    "STRATEGIC_BOMBER": CowUnitType.STRATEGIC_BOMBER,
    # Legacy aliases → closest new type
    "SNIPER": CowUnitType.SNIPER_TEAM,
    "STORM_TROOPERS": CowUnitType.SHOCK_TROOPS,
    "FLAMETHROWER": CowUnitType.FLAME_TANK,
    "HALF_TRACK": CowUnitType.ARMORED_CAR,
    "HEAVY_ARTILLERY": CowUnitType.ARTILLERY,
    "LOGISTICS": CowUnitType.SUPPLY_TRUCK,
    "SCOUTS": CowUnitType.RECON_INFANTRY,
    "ALPINE": CowUnitType.MOUNTAIN_TROOPS,
    "GEBIRGSJAGER": CowUnitType.MOUNTAIN_TROOPS,
    "SAPPER": CowUnitType.ENGINEER,
    "COMBAT_ENGINEER": CowUnitType.ENGINEER,
    "SP_ARTILLERY": CowUnitType.SP_ARTILLERY,
    "SP_ANTI_AIR": CowUnitType.SP_ANTI_AIR,
    "TACTICAL_BOMBER": CowUnitType.TACTICAL_BOMBER,
    "ATTACK_BOMBER": CowUnitType.ATTACK_BOMBER,
    "INTERCEPTOR": CowUnitType.INTERCEPTOR,
}

def cow_type_from_legacy(name: str) -> Optional[CowUnitType]:
    return _LEGACY_MAP.get(name)


# ─────────────────────────────────────────────────────────────────────────── #
# Buildings & Production Infrastructure                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class CowBuildingType(Enum):
    BARRACKS     = 0   # Produces infantry
    FACTORY      = 1   # Produces tanks & ordnance
    AIRFIELD     = 2   # Produces aircraft
    BUNKER       = 3   # Fortification (reduces incoming damage)
    SUPPLY_DEPOT = 4   # Increases local supply regeneration

# Max building level
MAX_BUILDING_LEVEL = 3

# Building cost per level: (rations, steel, ammo, fuel, medical)
BUILDING_COSTS: Dict[CowBuildingType, List[CowProductionCost]] = {
    CowBuildingType.BARRACKS: [
        CowProductionCost(3, 2, 0, 0, 0),    # L1
        CowProductionCost(5, 4, 0, 0, 1),    # L2
        CowProductionCost(8, 7, 0, 0, 2),    # L3
    ],
    CowBuildingType.FACTORY: [
        CowProductionCost(2, 5, 1, 1, 0),
        CowProductionCost(3, 9, 2, 2, 0),
        CowProductionCost(5, 14, 3, 3, 1),
    ],
    CowBuildingType.AIRFIELD: [
        CowProductionCost(2, 4, 0, 2, 0),
        CowProductionCost(3, 7, 0, 4, 1),
        CowProductionCost(5, 11, 0, 6, 2),
    ],
    CowBuildingType.BUNKER: [
        CowProductionCost(1, 3, 0, 0, 0),
        CowProductionCost(2, 6, 0, 0, 0),
        CowProductionCost(4, 10, 0, 0, 0),
    ],
    CowBuildingType.SUPPLY_DEPOT: [
        CowProductionCost(2, 1, 0, 1, 0),
        CowProductionCost(4, 2, 0, 2, 0),
        CowProductionCost(7, 4, 0, 3, 1),
    ],
}

# Build time per level (in steps)
BUILDING_BUILD_TIME: Dict[CowBuildingType, List[float]] = {
    CowBuildingType.BARRACKS:     [3.0, 5.0, 8.0],
    CowBuildingType.FACTORY:      [4.0, 7.0, 11.0],
    CowBuildingType.AIRFIELD:     [5.0, 8.0, 12.0],
    CowBuildingType.BUNKER:       [2.0, 4.0, 6.0],
    CowBuildingType.SUPPLY_DEPOT: [2.0, 3.0, 5.0],
}

# Which building is required to produce which unit category, at what level
CATEGORY_BUILDING_REQ: Dict[CowUnitCategory, CowBuildingType] = {
    CowUnitCategory.INFANTRY: CowBuildingType.BARRACKS,
    CowUnitCategory.ORDNANCE: CowBuildingType.FACTORY,
    CowUnitCategory.TANKS:    CowBuildingType.FACTORY,
    CowUnitCategory.AIRCRAFT: CowBuildingType.AIRFIELD,
}

# Unit level requires building level >= unit level (capped at MAX_BUILDING_LEVEL)
def required_building_level(unit_level: int) -> int:
    return max(1, min(MAX_BUILDING_LEVEL, unit_level))


# ─────────────────────────────────────────────────────────────────────────── #
# Research Tree                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class CowResearchProject:
    """A single research node."""
    unit_type: CowUnitType
    target_level: int
    cost: CowProductionCost
    time: float  # steps
    prerequisite: Optional[Tuple[CowUnitType, int]] = None  # (type, level) needed first

def _research_cost(unit_type: CowUnitType, level: int) -> CowProductionCost:
    """Research cost = 1.5x production cost of unit at that level."""
    c = get_unit_stats(unit_type, level).cost
    return CowProductionCost(c.rations*1.5, c.steel*1.5, c.ammo*1.5,
                             c.fuel*1.5, c.medical*1.5)

def _research_time(unit_type: CowUnitType, level: int) -> float:
    return get_unit_stats(unit_type, level).build_time * 3.0

def build_research_tree() -> Dict[Tuple[CowUnitType, int], CowResearchProject]:
    """Build full research tree: every unit type at levels 2-4."""
    tree = {}
    for ut in CowUnitType:
        for lvl in range(2, 5):
            prereq = (ut, lvl - 1) if lvl > 2 else None
            tree[(ut, lvl)] = CowResearchProject(
                unit_type=ut, target_level=lvl,
                cost=_research_cost(ut, lvl),
                time=_research_time(ut, lvl),
                prerequisite=prereq,
            )
    return tree

RESEARCH_TREE: Dict[Tuple[CowUnitType, int], CowResearchProject] = build_research_tree()


# ─────────────────────────────────────────────────────────────────────────── #
# Nonlinear Cost Scaling (anti-exploit)                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def nonlinear_production_cost(
    base_cost: CowProductionCost,
    n_existing_same_type: int,
    n_total_units: int,
    overextension_clusters: int = 0,
) -> CowProductionCost:
    """Apply nonlinear scaling to prevent RL exploit of mass-producing one type.

    Mechanisms:
      - Same-type penalty: cost *= (1 + 0.12 * n_existing)^1.3
      - Army size penalty: cost *= (1 + 0.03 * n_total)^1.1
      - Overextension:     cost *= (1 + 0.05 * extra_clusters)
    """
    type_mult = (1.0 + 0.12 * n_existing_same_type) ** 1.3
    army_mult = (1.0 + 0.03 * n_total_units) ** 1.1
    ext_mult  = 1.0 + 0.05 * max(0, overextension_clusters)
    m = type_mult * army_mult * ext_mult
    return CowProductionCost(
        base_cost.rations * m, base_cost.steel * m, base_cost.ammo * m,
        base_cost.fuel * m, base_cost.medical * m,
    )

def nonlinear_supply_drain(n_units: int, base_per_unit: float = 0.15) -> float:
    """Supply drain scales super-linearly with army size."""
    return base_per_unit * n_units ** 1.2

def combat_fatigue_factor(rounds_fought_recently: int) -> float:
    """Units that fight many rounds in a row suffer fatigue: 0-1 penalty."""
    return max(0.0, 1.0 - 0.08 * max(0, rounds_fought_recently - 2))

def morale_cascade(losses_this_step: int, army_size: int) -> float:
    """Morale drop from casualties. Nonlinear — catastrophic losses cause rout."""
    if army_size <= 0:
        return 0.0
    loss_ratio = losses_this_step / army_size
    return min(0.5, (loss_ratio ** 0.8) * 0.6)
