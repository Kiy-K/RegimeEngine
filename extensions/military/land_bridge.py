"""
land_bridge.py — Lightweight land combat bridge for Air Strip One.

Reuses the existing CoW combat system (cow_combat.py) with 30+ unit types,
terrain modifiers, armor classes, and Lanchester-style resolution.

This bridge provides:
  1. Per-sector garrison state (list of CoW units per cluster)
  2. Initialization with historically-accurate 1984 starting forces
  3. Land combat resolution when invasions establish beachheads
  4. Unit production/training integration with manpower system
  5. Garrison summary for LLM turn reports

Unit types available (from cow_combat.py):
  Infantry: MILITIA, INFANTRY, MOTORIZED_INFANTRY, MECHANIZED_INFANTRY,
            COMMANDOS, PARATROOPERS, GUARDS_INFANTRY, SHOCK_TROOPS, ENGINEER
  Ordnance: ANTI_TANK, ARTILLERY, SP_ARTILLERY, ANTI_AIR, MORTAR, ROCKET_ARTILLERY
  Tanks:    ARMORED_CAR, LIGHT_TANK, MEDIUM_TANK, HEAVY_TANK, TANK_DESTROYER
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .cow_combat import (
    CowUnit, CowArmy, CowUnitType, CowUnitCategory, CowArmorClass,
    CowTerrain, CowDoctrine, resolve_cow_combat, create_unit, create_army,
    reset_uid_counter,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Terrain mapping (Air Strip One sectors → CoW terrain)                      #
# ═══════════════════════════════════════════════════════════════════════════ #

SECTOR_TERRAIN: Dict[int, CowTerrain] = {
    # Oceania — British Isles
    0: CowTerrain.URBAN,       # London
    1: CowTerrain.PLAINS,      # Dover
    2: CowTerrain.PLAINS,      # Portsmouth
    3: CowTerrain.PLAINS,      # Southampton
    4: CowTerrain.PLAINS,      # Canterbury
    5: CowTerrain.PLAINS,      # Brighton
    6: CowTerrain.URBAN,       # Bristol
    7: CowTerrain.PLAINS,      # Plymouth
    8: CowTerrain.MOUNTAINS,   # Cardiff (Welsh valleys)
    9: CowTerrain.URBAN,       # Birmingham
    10: CowTerrain.URBAN,      # Manchester
    11: CowTerrain.PLAINS,     # Liverpool
    12: CowTerrain.PLAINS,     # Leeds
    13: CowTerrain.PLAINS,     # Norwich
    14: CowTerrain.URBAN,      # Edinburgh
    15: CowTerrain.URBAN,      # Glasgow
    16: CowTerrain.PLAINS,     # Dublin
    17: CowTerrain.URBAN,      # Belfast
    # Eurasia — France + Benelux
    18: CowTerrain.PLAINS,     # Calais
    19: CowTerrain.PLAINS,     # Dunkirk
    20: CowTerrain.PLAINS,     # Le Havre
    21: CowTerrain.PLAINS,     # Cherbourg
    22: CowTerrain.PLAINS,     # Amiens
    23: CowTerrain.URBAN,      # Rouen
    24: CowTerrain.PLAINS,     # Lille
    25: CowTerrain.URBAN,      # Brussels
    26: CowTerrain.URBAN,      # Antwerp
    27: CowTerrain.URBAN,      # Paris
    28: CowTerrain.PLAINS,     # Orleans
    29: CowTerrain.URBAN,      # Lyon
    30: CowTerrain.PLAINS,     # Brest
    31: CowTerrain.PLAINS,     # Bordeaux
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Land State                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class LandWorld:
    """Land garrison state for all sectors."""
    garrisons: Dict[int, List[CowUnit]] = field(default_factory=dict)
    production_queues: Dict[int, List[Tuple[CowUnitType, int, int]]] = field(
        default_factory=dict)  # faction_id → [(type, level, turns_left)]

    def garrison(self, cluster_id: int) -> List[CowUnit]:
        return self.garrisons.get(cluster_id, [])

    def alive_garrison(self, cluster_id: int) -> List[CowUnit]:
        return [u for u in self.garrison(cluster_id) if u.is_alive]

    def faction_strength(self, faction_id: int) -> float:
        total = 0.0
        for units in self.garrisons.values():
            for u in units:
                if u.is_alive and u.faction_id == faction_id:
                    total += u.hp
        return total

    def cluster_strength(self, cluster_id: int, faction_id: int) -> float:
        return sum(u.hp for u in self.alive_garrison(cluster_id)
                   if u.faction_id == faction_id)

    def cluster_unit_count(self, cluster_id: int, faction_id: int) -> int:
        return sum(1 for u in self.alive_garrison(cluster_id)
                   if u.faction_id == faction_id)


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization — historically-based 1984 starting forces                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_land(cluster_owners: Dict[int, int]) -> LandWorld:
    """Create starting garrisons for both factions.

    Oceania (faction 0): Larger garrison, defensive posture
      - London: Guards Infantry + AA + Artillery (capital defense)
      - Dover/Portsmouth/Southampton: Infantry + Anti-Tank (Channel front)
      - Other sectors: Militia + Infantry (rear guard)

    Eurasia (faction 1): Offensive posture, armor-heavy
      - Calais/Dunkirk/Le Havre: Infantry + Tanks (invasion staging)
      - Paris: Guards + Artillery + AA (HQ defense)
      - Other sectors: Infantry + Motorized (mobile reserves)
    """
    reset_uid_counter(10000)  # don't collide with naval/air IDs
    world = LandWorld()

    for cid, owner in cluster_owners.items():
        units: List[CowUnit] = []

        if owner == 0:  # Oceania
            if cid == 0:  # London — capital defense
                units.extend([create_unit(CowUnitType.GUARDS_INFANTRY, 2, 0, cid) for _ in range(3)])
                units.extend([create_unit(CowUnitType.ANTI_AIR, 2, 0, cid) for _ in range(2)])
                units.extend([create_unit(CowUnitType.ARTILLERY, 2, 0, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.ENGINEER, 1, 0, cid))
            elif cid in (1, 2, 3, 4, 5):  # Channel front
                units.extend([create_unit(CowUnitType.INFANTRY, 2, 0, cid) for _ in range(3)])
                units.append(create_unit(CowUnitType.ANTI_TANK, 2, 0, cid))
                units.append(create_unit(CowUnitType.ARTILLERY, 1, 0, cid))
            elif cid in (9, 10):  # Industrial heartland
                units.extend([create_unit(CowUnitType.INFANTRY, 1, 0, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.ANTI_AIR, 1, 0, cid))
            else:  # Rear sectors
                units.extend([create_unit(CowUnitType.MILITIA, 1, 0, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.INFANTRY, 1, 0, cid))

        elif owner == 1:  # Eurasia
            if cid in (18, 19, 20, 21):  # Channel staging areas
                units.extend([create_unit(CowUnitType.INFANTRY, 2, 1, cid) for _ in range(3)])
                units.append(create_unit(CowUnitType.MEDIUM_TANK, 1, 1, cid))
                units.append(create_unit(CowUnitType.ARTILLERY, 1, 1, cid))
            elif cid == 27:  # Paris — HQ
                units.extend([create_unit(CowUnitType.GUARDS_INFANTRY, 2, 1, cid) for _ in range(3)])
                units.extend([create_unit(CowUnitType.ANTI_AIR, 2, 1, cid) for _ in range(2)])
                units.extend([create_unit(CowUnitType.ARTILLERY, 2, 1, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.HEAVY_TANK, 1, 1, cid))
            elif cid in (25, 26):  # Benelux — mobile reserves
                units.extend([create_unit(CowUnitType.MOTORIZED_INFANTRY, 1, 1, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.LIGHT_TANK, 1, 1, cid))
            else:  # Rear sectors
                units.extend([create_unit(CowUnitType.MILITIA, 1, 1, cid) for _ in range(2)])
                units.append(create_unit(CowUnitType.INFANTRY, 1, 1, cid))

        world.garrisons[cid] = units

    return world


# ═══════════════════════════════════════════════════════════════════════════ #
# Land Combat Step — resolves battles in contested sectors                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_land(
    world: LandWorld,
    cluster_owners: Dict[int, int],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[LandWorld, Dict[str, Any]]:
    """Step all land garrisons. Resolve combat in contested sectors."""
    feedback: Dict[str, Any] = {
        "battles": 0, "units_lost_0": 0, "units_lost_1": 0,
    }

    for cid in list(world.garrisons.keys()):
        alive = world.alive_garrison(cid)
        if not alive:
            continue

        # Check if both factions have units in same sector
        factions_present = set(u.faction_id for u in alive)
        if len(factions_present) < 2:
            continue

        # Battle! Separate by faction
        f0_units = [u for u in alive if u.faction_id == 0]
        f1_units = [u for u in alive if u.faction_id == 1]

        owner = cluster_owners.get(cid, 0)
        terrain = SECTOR_TERRAIN.get(cid, CowTerrain.PLAINS)

        # Determine attacker/defender
        if owner == 0:
            atk_army = CowArmy(f1_units, 1, cid)
            def_army = CowArmy(f0_units, 0, cid)
            defender_home = True
        else:
            atk_army = CowArmy(f0_units, 0, cid)
            def_army = CowArmy(f1_units, 1, cid)
            defender_home = True

        # Resolve one round of combat
        new_atk, new_def, log = resolve_cow_combat(
            atk_army, def_army,
            terrain=terrain,
            fortification=0.1,  # basic field fortifications
            defender_is_home=defender_home,
            rng=rng,
        )

        # Write back
        survivors = new_atk.units + new_def.units
        world.garrisons[cid] = [u for u in survivors if u.is_alive]

        feedback["battles"] += 1
        feedback["units_lost_0"] += log.get("def_losses" if owner == 0 else "atk_losses", 0)
        feedback["units_lost_1"] += log.get("atk_losses" if owner == 0 else "def_losses", 0)

        # Sector changes owner if defender completely destroyed
        if log.get("result") == "attacker_wins":
            new_owner = 1 if owner == 0 else 0
            cluster_owners[cid] = new_owner

    return world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Summary for LLM turn reports                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def land_summary(
    world: LandWorld,
    faction_id: int,
    cluster_owners: Dict[int, int],
    cluster_names: List[str],
) -> str:
    """Generate garrison summary for turn reports."""
    parts = []

    # Count total units
    total_units = 0
    total_hp = 0.0
    for cid, owner in cluster_owners.items():
        if owner != faction_id:
            continue
        units = world.alive_garrison(cid)
        own = [u for u in units if u.faction_id == faction_id]
        total_units += len(own)
        total_hp += sum(u.hp for u in own)

    parts.append(f"Land forces: {total_units} units ({total_hp:.0f} HP)")

    # Key garrisons
    garrisons = []
    for cid, owner in cluster_owners.items():
        if owner != faction_id:
            continue
        own = [u for u in world.alive_garrison(cid) if u.faction_id == faction_id]
        if len(own) >= 3:
            name = cluster_names[cid] if cid < len(cluster_names) else f"C{cid}"
            # Count by category
            inf = sum(1 for u in own if u.stats.category == CowUnitCategory.INFANTRY)
            tank = sum(1 for u in own if u.stats.category == CowUnitCategory.TANKS)
            ord_ = sum(1 for u in own if u.stats.category == CowUnitCategory.ORDNANCE)
            garrisons.append(f"{name}({inf}I/{tank}T/{ord_}A)")

    if garrisons:
        parts.append("Strong points: " + ", ".join(garrisons[:5]))

    # Weakly defended sectors
    weak = []
    for cid, owner in cluster_owners.items():
        if owner != faction_id:
            continue
        own = [u for u in world.alive_garrison(cid) if u.faction_id == faction_id]
        if len(own) <= 1:
            name = cluster_names[cid] if cid < len(cluster_names) else f"C{cid}"
            weak.append(name)
    if weak:
        parts.append(f"WEAK GARRISONS ({len(weak)}): {', '.join(weak[:4])}")

    return " | ".join(parts)
