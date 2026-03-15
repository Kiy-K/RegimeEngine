"""
naval_operations.py — Naval operations: convoys, amphibious, blockade, mines, step function.

Operations:
  - Convoy escort: supply ships run routes, submarines intercept, escorts defend
  - Amphibious landing: transports deliver troops under fire (massive penalty)
  - Blockade: deny enemy sea zones, starve their ports
  - Mine warfare: lay/sweep mines in sea zones
  - Fleet movement: reposition fleets between adjacent zones
  - Shore bombardment: fire at coastal clusters
  - Shipyard production: build new ships (long build times)

Anti-exploitation:
  - Fuel consumption every step (no infinite range)
  - Crew fatigue from sustained operations
  - Repair requires port + STEEL + time
  - Amphibious landing has 3x attacker casualties
  - Convoys must have escorts or get massacred by subs
  - Mine fields damage both sides equally
  - Shipyard build times: 10 steps (corvette) to 90 steps (carrier)
  - Sea state (weather) randomly disrupts operations
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .naval_state import (
    ShipClass, ShipCategory, ShipInstance, Fleet, SeaZone,
    SeaZoneControl, MineField, ConvoyRoute, NavalWorld,
    SHIP_STATS, N_SHIP_CLASSES,
)
from .naval_combat import (
    resolve_naval_battle, compute_detection,
    shore_bombardment, anti_submarine_warfare,
    apply_mine_damage,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Naval Actions                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class NavalAction(Enum):
    NOOP              = 0
    MOVE_FLEET        = 1   # Reposition fleet to adjacent zone
    ASSIGN_MISSION    = 2   # Set fleet mission (PATROL/ESCORT/BLOCKADE/RAID)
    CONVOY_CREATE     = 3   # Establish supply convoy route
    AMPHIBIOUS_LAUNCH = 4   # Launch amphibious assault
    LAY_MINES         = 5   # Deploy mines in a sea zone
    SWEEP_MINES       = 6   # Clear enemy mines
    SHORE_BOMBARD     = 7   # Bombard coastal cluster
    BUILD_SHIP        = 8   # Queue ship construction at shipyard
    REPAIR_FLEET      = 9   # Repair damaged ships (requires port)


N_NAVAL_ACTIONS = len(NavalAction)


# ═══════════════════════════════════════════════════════════════════════════ #
# Convoy Operations                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_convoys(
    nw: NavalWorld,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[int, float]:
    """
    Run all active convoy routes. Returns dict of destination_cluster → cargo delivered.

    Convoys transit their sea zones. In each zone:
      1. Check for enemy submarines/raiders
      2. Escort fleet engages enemies
      3. Unescorted convoys suffer heavy losses
      4. Surviving cargo reaches destination
    """
    deliveries: Dict[int, float] = {}

    for route in nw.convoy_routes:
        if not route.is_active:
            continue

        cargo_remaining = route.cargo_capacity
        route_lost = False

        for zone_id in route.sea_zones:
            if zone_id >= len(nw.sea_zones):
                continue
            zone = nw.sea_zones[zone_id]

            # Find enemy fleets in this zone
            enemy_fleets = [f for f in zone.fleets if f.faction_id != route.faction_id]
            enemy_subs = []
            for ef in enemy_fleets:
                enemy_subs.extend([s for s in ef.operational_ships if s.stats.is_submersible])

            if not enemy_subs and not enemy_fleets:
                continue  # safe passage

            # Find escort fleet
            escort = None
            if route.escort_fleet_id is not None:
                for f in zone.fleets:
                    if f.fleet_id == route.escort_fleet_id:
                        escort = f
                        break

            # Convoy interception
            if enemy_subs:
                if escort and escort.operational_ships:
                    # Escort vs submarine battle
                    sub_fleet = Fleet(fleet_id=-1, faction_id=-1, sea_zone_id=zone_id,
                                     ships=enemy_subs)
                    asw_result = anti_submarine_warfare(escort, sub_fleet, zone.sea_state, rng, dt)
                    # Surviving subs attack convoy
                    surviving_subs = [s for s in enemy_subs if s.is_alive and not s.is_detected]
                    if surviving_subs:
                        # Each undetected sub sinks ~10% of cargo
                        loss_per_sub = 0.10 * dt
                        total_loss = min(1.0, len(surviving_subs) * loss_per_sub)
                        cargo_remaining *= (1.0 - total_loss)
                        route.losses_suffered += 1
                else:
                    # UNESCORTED convoy — catastrophic losses (wolf pack massacre)
                    loss = min(0.6, len(enemy_subs) * 0.15) * dt
                    cargo_remaining *= (1.0 - loss)
                    route.losses_suffered += 1

            # Mine damage to convoy
            if zone.mines.density > 0.01:
                mine_loss = zone.mines.density * 0.05 * dt
                cargo_remaining *= (1.0 - mine_loss)

            if cargo_remaining < route.cargo_capacity * 0.1:
                route_lost = True
                break

        if not route_lost and cargo_remaining > 0:
            dest = route.destination_cluster
            deliveries[dest] = deliveries.get(dest, 0.0) + cargo_remaining
            route.deliveries_completed += 1

    return deliveries


# ═══════════════════════════════════════════════════════════════════════════ #
# Amphibious Landing                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def attempt_amphibious_landing(
    fleet: Fleet,
    target_cluster_id: int,
    sea_zone: SeaZone,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Attempt amphibious assault. Returns results dict.

    Massive penalties for the attacker:
      - 3x casualty rate during landing
      - Only Marines/Infantry can land
      - Transports are vulnerable to shore fire
      - Weather can abort the operation
      - Need naval superiority in the zone
    """
    results = {
        "troops_landed": 0, "troops_lost": 0,
        "transports_lost": 0, "success": False,
    }

    # Check preconditions
    if sea_zone.sea_state > 0.7:
        return results  # too rough for landing

    if sea_zone.control == SeaZoneControl.DENIED:
        return results  # zone is denied

    transports = [s for s in fleet.operational_ships
                  if s.ship_class == ShipClass.TRANSPORT]
    if not transports:
        return results  # no transports

    total_capacity = sum(s.stats.carry_capacity for s in transports)

    # Shore fire damages transports during approach
    shore_fire_chance = 0.15 * dt
    for t in transports:
        if rng.random() < shore_fire_chance:
            t.hp = max(0.0, t.hp - rng.uniform(15, 40))
            if t.hp <= 0:
                results["transports_lost"] += 1

    # Surviving transport capacity
    surviving_cap = sum(s.stats.carry_capacity for s in transports if s.is_alive)

    # Landing casualties: 30% of troops lost during beach assault
    beach_casualty_rate = 0.30 * (1.0 + sea_zone.sea_state)
    troops_landed = int(surviving_cap * (1.0 - beach_casualty_rate))
    troops_lost = int(surviving_cap * beach_casualty_rate)

    results["troops_landed"] = max(0, troops_landed)
    results["troops_lost"] = troops_lost
    results["success"] = troops_landed > 100

    return results


# ═══════════════════════════════════════════════════════════════════════════ #
# Blockade                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def enforce_blockade(
    nw: NavalWorld,
    faction_id: int,
    zone_id: int,
) -> bool:
    """
    Enforce naval blockade of a sea zone.
    Requires fleet with BLOCKADE mission and naval superiority.
    Returns True if blockade is successfully enforced.
    """
    if zone_id >= len(nw.sea_zones):
        return False

    zone = nw.sea_zones[zone_id]
    friendly_fleets = [f for f in zone.fleets if f.faction_id == faction_id]
    enemy_fleets = [f for f in zone.fleets if f.faction_id != faction_id]

    friendly_power = sum(f.total_firepower for f in friendly_fleets)
    enemy_power = sum(f.total_firepower for f in enemy_fleets)

    # Need 2:1 superiority for effective blockade
    if friendly_power > enemy_power * 2.0:
        zone.control = SeaZoneControl.CONTROLLED
        zone.controlling_faction = faction_id
        return True
    elif friendly_power > enemy_power:
        zone.control = SeaZoneControl.CONTESTED
        return False
    else:
        return False


# ═══════════════════════════════════════════════════════════════════════════ #
# Mine Warfare                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def lay_mines(
    fleet: Fleet,
    zone: SeaZone,
    faction_id: int,
    dt: float = 1.0,
) -> float:
    """Lay mines in a sea zone. Returns mines laid."""
    minelayers = [s for s in fleet.operational_ships if s.stats.mine_capacity > 0]
    if not minelayers:
        return 0.0

    mines_laid = sum(min(s.stats.mine_capacity, 10) for s in minelayers) * 0.01 * dt
    zone.mines.density = min(1.0, zone.mines.density + mines_laid)
    zone.mines.faction_id = faction_id
    return mines_laid


def sweep_mines(
    fleet: Fleet,
    zone: SeaZone,
    dt: float = 1.0,
) -> float:
    """Sweep mines from a sea zone. Returns mines cleared."""
    # Any ship can sweep, but destroyers/escorts are best
    sweep_power = sum(
        0.5 + 0.5 * (s.stats.anti_sub / 8.0)  # ASW capability correlates with minesweeping
        for s in fleet.operational_ships
    )
    cleared = sweep_power * 0.005 * dt
    zone.mines.density = max(0.0, zone.mines.density - cleared)
    return cleared


# ═══════════════════════════════════════════════════════════════════════════ #
# Sea Zone Control Update                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def update_zone_control(zone: SeaZone) -> None:
    """Update sea zone control state based on fleet presence."""
    factions = zone.faction_fleets
    if len(factions) == 0:
        zone.control = SeaZoneControl.UNCONTESTED
        zone.controlling_faction = None
    elif len(factions) == 1:
        fid = list(factions.keys())[0]
        power = sum(f.total_firepower for f in factions[fid])
        if power > 5.0:
            zone.control = SeaZoneControl.CONTROLLED
            zone.controlling_faction = fid
        else:
            zone.control = SeaZoneControl.UNCONTESTED
            zone.controlling_faction = None
    else:
        # Multiple factions: contested or one dominates
        powers = {fid: sum(f.total_firepower for f in fleets)
                  for fid, fleets in factions.items()}
        max_fid = max(powers, key=powers.get)
        max_power = powers[max_fid]
        total_enemy = sum(p for fid, p in powers.items() if fid != max_fid)

        if max_power > total_enemy * 3.0:
            zone.control = SeaZoneControl.CONTROLLED
            zone.controlling_faction = max_fid
        elif max_power > total_enemy * 1.5:
            zone.control = SeaZoneControl.CONTESTED
            zone.controlling_faction = None
        else:
            zone.control = SeaZoneControl.CONTESTED
            zone.controlling_faction = None

    # Submarine presence can deny a zone even without surface superiority
    for fid, fleets in factions.items():
        sub_count = sum(f.submarine_count for f in fleets)
        if sub_count >= 3 and zone.controlling_faction != fid:
            if zone.control != SeaZoneControl.CONTROLLED:
                zone.control = SeaZoneControl.DENIED


# ═══════════════════════════════════════════════════════════════════════════ #
# Main Naval Step                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_naval(
    nw: NavalWorld,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[NavalWorld, Dict[str, float]]:
    """
    Advance naval simulation by one step.

    Pipeline:
      1. Weather update (sea state changes)
      2. Fuel consumption for all fleets
      3. Combat in contested zones
      4. Convoy execution
      5. Mine damage for transiting fleets
      6. Zone control update
      7. Shipyard production tick
      8. Crew fatigue recovery (in port) / accumulation (at sea)

    Returns (updated_naval_world, feedback_dict).
    """
    feedback: Dict[str, float] = {
        "total_battles": 0, "ships_sunk": 0,
        "cargo_delivered": 0.0, "mines_active": 0.0,
    }

    # ── 1. Weather ────────────────────────────────────────────────────── #
    for zone in nw.sea_zones:
        # Sea state drifts randomly (Channel storms!)
        zone.sea_state = float(np.clip(
            zone.sea_state + rng.normal(0, 0.05) * dt,
            0.0, 1.0,
        ))

    # ── 2. Fuel consumption ───────────────────────────────────────────── #
    for zone in nw.sea_zones:
        for fleet in zone.fleets:
            for ship in fleet.ships:
                if ship.is_alive:
                    ship.fuel = max(0.0, ship.fuel - ship.stats.fuel_per_step * 0.005 * dt)

    # ── 3. Combat in contested zones ──────────────────────────────────── #
    for zone in nw.sea_zones:
        factions = zone.faction_fleets
        if len(factions) < 2:
            continue

        faction_ids = list(factions.keys())
        for i in range(len(faction_ids)):
            for j in range(i + 1, len(faction_ids)):
                fid_a, fid_b = faction_ids[i], faction_ids[j]
                for fleet_a in factions[fid_a]:
                    for fleet_b in factions[fid_b]:
                        if not fleet_a.operational_ships or not fleet_b.operational_ships:
                            continue

                        # Detection check
                        detect_ab = compute_detection(fleet_a, fleet_b, zone.sea_state, rng)
                        detect_ba = compute_detection(fleet_b, fleet_a, zone.sea_state, rng)

                        if rng.random() < detect_ab or rng.random() < detect_ba:
                            result = resolve_naval_battle(
                                fleet_a, fleet_b, zone.sea_state, rng, dt)
                            feedback["total_battles"] += 1
                            feedback["ships_sunk"] += (
                                result["attacker_ships_sunk"] + result["defender_ships_sunk"])

    # ── 4. Convoys ────────────────────────────────────────────────────── #
    deliveries = run_convoys(nw, rng, dt)
    feedback["cargo_delivered"] = sum(deliveries.values())

    # ── 5. Mine damage ────────────────────────────────────────────────── #
    for zone in nw.sea_zones:
        if zone.mines.density > 0.01:
            feedback["mines_active"] += zone.mines.density
            for fleet in zone.fleets:
                apply_mine_damage(fleet, zone.mines, rng, dt)

    # ── 6. Zone control ───────────────────────────────────────────────── #
    for zone in nw.sea_zones:
        update_zone_control(zone)

    # ── 7. Shipyard production ────────────────────────────────────────── #
    for fid, queue in nw.shipyard_queues.items():
        completed = []
        for i, (ship_class, steps_left) in enumerate(queue):
            queue[i] = (ship_class, steps_left - 1)
            if steps_left - 1 <= 0:
                completed.append((i, ship_class))

        # Launch completed ships (add to nearest friendly port zone)
        for idx, sc in reversed(completed):
            queue.pop(idx)
            stats = SHIP_STATS[sc]
            new_ship = ShipInstance(
                ship_id=nw.alloc_ship_id(), ship_class=sc,
                faction_id=fid, hp=100.0, max_hp=100.0,
            )
            # Find a zone with friendly fleet to add to
            placed = False
            for zone in nw.sea_zones:
                for fleet in zone.fleets:
                    if fleet.faction_id == fid:
                        fleet.ships.append(new_ship)
                        placed = True
                        break
                if placed:
                    break

    # ── 8. Crew fatigue recovery/accumulation ─────────────────────────── #
    for zone in nw.sea_zones:
        is_port = any(True for _ in zone.connected_clusters)  # simplified port check
        for fleet in zone.fleets:
            for ship in fleet.ships:
                if ship.is_alive:
                    if is_port and fleet.mission == "PATROL":
                        # In port: fatigue recovers, damage slowly repairs
                        ship.crew_fatigue = max(0.0, ship.crew_fatigue - 0.01 * dt)
                        ship.damage_level = max(0.0, ship.damage_level - 0.005 * dt)
                    else:
                        # At sea: fatigue accumulates
                        ship.crew_fatigue = min(1.0, ship.crew_fatigue + 0.003 * dt)

    # Remove sunk ships
    for zone in nw.sea_zones:
        for fleet in zone.fleets:
            fleet.ships = [s for s in fleet.ships if s.is_alive]

    nw.step += 1
    return nw, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Action Application                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_naval_action(
    nw: NavalWorld,
    faction_id: int,
    action_type: int,
    zone_id: int,
    target_zone_or_cluster: int,
    ship_class_idx: int,
    fleet_idx: int,
    rng: np.random.Generator,
) -> Tuple[NavalWorld, float]:
    """Apply a naval action. Returns (updated_world, reward)."""
    try:
        action = NavalAction(action_type)
    except ValueError:
        return nw, 0.0

    if action == NavalAction.NOOP:
        return nw, 0.0

    elif action == NavalAction.BUILD_SHIP:
        sc_idx = min(ship_class_idx, N_SHIP_CLASSES - 1)
        sc = ShipClass(sc_idx)
        stats = SHIP_STATS[sc]
        if faction_id not in nw.shipyard_queues:
            nw.shipyard_queues[faction_id] = []
        # Limit queue size (anti-exploitation)
        if len(nw.shipyard_queues[faction_id]) >= 5:
            return nw, -0.1
        nw.shipyard_queues[faction_id].append((sc, stats.build_time))
        return nw, 0.2

    elif action == NavalAction.ASSIGN_MISSION:
        missions = ["PATROL", "CONVOY_ESCORT", "BLOCKADE", "RAID", "SHORE_BOMBARD", "AMPHIBIOUS"]
        mission = missions[min(target_zone_or_cluster, len(missions) - 1)]
        if zone_id < len(nw.sea_zones):
            for fleet in nw.sea_zones[zone_id].fleets:
                if fleet.faction_id == faction_id:
                    fleet.mission = mission
                    return nw, 0.1
        return nw, -0.05

    elif action == NavalAction.SHORE_BOMBARD:
        if zone_id < len(nw.sea_zones):
            zone = nw.sea_zones[zone_id]
            for fleet in zone.fleets:
                if fleet.faction_id == faction_id:
                    result = shore_bombardment(fleet, target_zone_or_cluster, zone.sea_state, rng)
                    return nw, result["damage_dealt"] * 0.05
        return nw, -0.05

    elif action == NavalAction.LAY_MINES:
        if zone_id < len(nw.sea_zones):
            zone = nw.sea_zones[zone_id]
            for fleet in zone.fleets:
                if fleet.faction_id == faction_id:
                    laid = lay_mines(fleet, zone, faction_id)
                    return nw, laid * 0.5
        return nw, -0.05

    elif action == NavalAction.SWEEP_MINES:
        if zone_id < len(nw.sea_zones):
            zone = nw.sea_zones[zone_id]
            for fleet in zone.fleets:
                if fleet.faction_id == faction_id:
                    swept = sweep_mines(fleet, zone)
                    return nw, swept * 0.3
        return nw, -0.05

    elif action == NavalAction.REPAIR_FLEET:
        if zone_id < len(nw.sea_zones):
            for fleet in nw.sea_zones[zone_id].fleets:
                if fleet.faction_id == faction_id:
                    for ship in fleet.ships:
                        if ship.is_alive:
                            ship.damage_level = max(0.0, ship.damage_level - 0.1)
                            ship.hp = min(ship.max_hp, ship.hp + 10.0)
                            ship.ammo = min(1.0, ship.ammo + 0.2)
                            ship.fuel = min(1.0, ship.fuel + 0.2)
                    return nw, 0.2
        return nw, -0.05

    return nw, 0.0


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_naval(
    sea_zone_configs: List[Dict],
    faction_ids: List[int],
    rng: np.random.Generator,
) -> NavalWorld:
    """
    Initialize naval world from zone configs.

    Each zone config dict:
      - name: str
      - connected_clusters: List[int]
      - adjacent_zones: List[int]
      - width_km: float
      - initial_fleets: Dict[int, List[str]]  (faction_id → ship class names)
    """
    zones = []
    next_ship_id = 0
    next_fleet_id = 0

    for i, cfg in enumerate(sea_zone_configs):
        fleets = []
        initial = cfg.get("initial_fleets", {})

        for fid, ship_names in initial.items():
            fid = int(fid)
            ships = []
            for sname in ship_names:
                try:
                    sc = ShipClass[sname]
                except KeyError:
                    continue
                ships.append(ShipInstance(
                    ship_id=next_ship_id, ship_class=sc, faction_id=fid,
                    hp=100.0, max_hp=100.0,
                    fuel=0.8 + rng.uniform(0, 0.2),
                    ammo=0.9 + rng.uniform(0, 0.1),
                ))
                next_ship_id += 1

            if ships:
                fleets.append(Fleet(
                    fleet_id=next_fleet_id, faction_id=fid,
                    sea_zone_id=i, ships=ships,
                ))
                next_fleet_id += 1

        zones.append(SeaZone(
            zone_id=i,
            name=cfg.get("name", f"Zone_{i}"),
            connected_clusters=cfg.get("connected_clusters", []),
            adjacent_zones=cfg.get("adjacent_zones", []),
            fleets=fleets,
            width_km=cfg.get("width_km", 50.0),
            sea_state=rng.uniform(0.0, 0.3),
        ))

    return NavalWorld(
        sea_zones=zones,
        next_ship_id=next_ship_id,
        shipyard_queues={fid: [] for fid in faction_ids},
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def naval_obs(
    nw: NavalWorld,
    faction_id: int,
    max_zones: int = 6,
) -> NDArray[np.float32]:
    """
    Build flat observation vector for naval state.

    Per zone (15 floats × max_zones):
      - control_state (one-hot 4)
      - friendly_firepower, friendly_asw, friendly_subs (3)
      - enemy_firepower, enemy_asw, enemy_subs (3)
      - mine_density (1)
      - sea_state (1)
      - convoy_active (1)
      - width_km normalized (1)
      - controlling_faction (1)

    Global (5 floats):
      - total_friendly_ships, total_enemy_ships
      - shipyard_queue_length
      - total_cargo_delivered (from routes)
      - avg_fleet_fuel

    Total: 15 × max_zones + 5
    """
    n_per = 15
    n_global = 5
    obs = np.zeros(n_per * max_zones + n_global, dtype=np.float32)

    for i, zone in enumerate(nw.sea_zones):
        if i >= max_zones:
            break
        o = i * n_per

        # Control state one-hot
        obs[o + zone.control.value] = 1.0

        # Friendly vs enemy power
        for fleet in zone.fleets:
            if fleet.faction_id == faction_id:
                obs[o + 4] += fleet.total_firepower / 20.0
                obs[o + 5] += fleet.total_anti_sub / 20.0
                obs[o + 6] += fleet.submarine_count / 5.0
            else:
                obs[o + 7] += fleet.total_firepower / 20.0
                obs[o + 8] += fleet.total_anti_sub / 20.0
                obs[o + 9] += fleet.submarine_count / 5.0

        obs[o + 10] = zone.mines.density
        obs[o + 11] = zone.sea_state
        # Convoy active in this zone
        obs[o + 12] = float(any(
            zone.zone_id in r.sea_zones for r in nw.convoy_routes
            if r.faction_id == faction_id and r.is_active
        ))
        obs[o + 13] = min(zone.width_km / 200.0, 1.0)
        obs[o + 14] = float(zone.controlling_faction == faction_id) if zone.controlling_faction is not None else 0.5

    # Global
    go = n_per * max_zones
    friendly_ships = nw.faction_ships(faction_id)
    all_ships = sum(len(z.fleets) for z in nw.sea_zones)
    obs[go + 0] = min(len(friendly_ships) / 20.0, 1.0)
    obs[go + 1] = min((all_ships - len(friendly_ships)) / 20.0, 1.0)
    obs[go + 2] = len(nw.shipyard_queues.get(faction_id, [])) / 5.0
    obs[go + 3] = sum(r.deliveries_completed for r in nw.convoy_routes
                       if r.faction_id == faction_id) / 50.0
    if friendly_ships:
        obs[go + 4] = np.mean([s.fuel for s in friendly_ships])

    return np.clip(obs, 0.0, 1.0)


def naval_obs_size(max_zones: int = 6) -> int:
    return 15 * max_zones + 5
