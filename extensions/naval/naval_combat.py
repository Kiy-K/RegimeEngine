"""
naval_combat.py — Naval combat resolution engine.

Combat model based on modified Lanchester's Square Law:
  Casualties_A = k_B × N_B² / (k_A × N_A + k_B × N_B)
  where k = effectiveness × firepower × (1 - sea_state_penalty)

Key mechanics:
  - Range brackets: long (capital guns), medium (cruisers), close (torpedoes)
  - Initiative: faster fleet shoots first at long range
  - Submarine stealth: undetected subs get free torpedo salvo
  - Carrier air strikes: launch before gun range, vulnerable to AA
  - Shore bombardment: ships fire at land targets (reduced accuracy)
  - Mine damage: probabilistic damage when transiting mined zones
  - Weather: storms reduce accuracy, speed, and detection
  - Ammo depletion: prolonged combat drains magazines
  - Crew fatigue: multi-step combat degrades performance

Anti-exploitation:
  - Command span penalty for oversized fleets (>12 ships)
  - Torpedo tubes need time to reload (1 salvo per combat step)
  - Submarines revealed after attacking (lose stealth)
  - Carriers can't launch in heavy seas (sea_state > 0.7)
  - Damaged ships fight at reduced effectiveness (not binary alive/dead)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .naval_state import (
    ShipClass, ShipCategory, ShipInstance, Fleet, SeaZone,
    MineField, SHIP_STATS, SeaZoneControl,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Detection                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_detection(
    observer_fleet: Fleet,
    target_fleet: Fleet,
    sea_state: float,
    rng: np.random.Generator,
) -> float:
    """
    Compute probability that observer detects target fleet.

    Detection depends on:
      - Observer's sub_detection capability
      - Target's surface_detection (visibility)
      - Sea state (storms reduce detection)
      - Target submersibility (submarines are hard to find)

    Returns detection probability [0, 1].
    """
    if not observer_fleet.operational_ships or not target_fleet.operational_ships:
        return 0.0

    # Observer capability: best detector in fleet
    obs_detect = max(s.stats.sub_detection for s in observer_fleet.operational_ships)

    # Target visibility: average visibility of target fleet
    target_vis = target_fleet.avg_detection_risk

    # Submarine modifier: subs are hard to detect
    sub_fraction = target_fleet.submarine_count / max(len(target_fleet.operational_ships), 1)
    stealth_mod = 1.0 - 0.7 * sub_fraction  # subs reduce detection by up to 70%

    # Weather reduces detection
    weather_mod = 1.0 - 0.5 * sea_state

    # Base detection probability
    base_prob = (obs_detect / 10.0) * (target_vis / 10.0) * stealth_mod * weather_mod

    return float(np.clip(base_prob, 0.02, 0.98))


# ═══════════════════════════════════════════════════════════════════════════ #
# Naval Battle Resolution                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def resolve_naval_battle(
    attacker: Fleet,
    defender: Fleet,
    sea_state: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Resolve one step of naval combat between two fleets.

    Uses modified Lanchester Square Law with:
      - Range brackets (long/medium/close)
      - Initiative from speed advantage
      - Submarine torpedo salvos
      - Weather penalties
      - Command span penalties
      - Ammo depletion

    Returns dict with battle results:
      - attacker_losses_hp, defender_losses_hp
      - attacker_ships_sunk, defender_ships_sunk
      - submarines_revealed (attacker subs that fired and lost stealth)
    """
    results = {
        "attacker_losses_hp": 0.0,
        "defender_losses_hp": 0.0,
        "attacker_ships_sunk": 0,
        "defender_ships_sunk": 0,
        "submarines_revealed": 0,
    }

    atk_ships = attacker.operational_ships
    def_ships = defender.operational_ships
    if not atk_ships or not def_ships:
        return results

    # ── Weather penalty ──────────────────────────────────────────────── #
    weather_mult = max(0.3, 1.0 - 0.5 * sea_state)

    # ── Command span penalty ─────────────────────────────────────────── #
    atk_cmd = 1.0 - attacker.command_span_penalty
    def_cmd = 1.0 - defender.command_span_penalty

    # ── Phase 1: Submarine torpedo salvo (undetected subs fire first) ── #
    atk_subs = [s for s in atk_ships if s.stats.is_submersible and not s.is_detected]
    def_subs = [s for s in def_ships if s.stats.is_submersible and not s.is_detected]

    for sub in atk_subs:
        if sub.ammo > 0.1:
            torp_dmg = sub.stats.torpedo * sub.combat_effectiveness * weather_mult * dt
            # Target largest non-submarine ship
            targets = [s for s in def_ships if s.is_alive and not s.stats.is_submersible]
            if targets:
                target = max(targets, key=lambda s: s.max_hp)
                actual_dmg = torp_dmg * (3.0 + rng.uniform(-1.0, 1.0))  # torpedo hit is devastating
                target.hp = max(0.0, target.hp - actual_dmg)
                results["defender_losses_hp"] += actual_dmg
                sub.ammo = max(0.0, sub.ammo - 0.15)  # torpedo tubes need reloading
                sub.is_detected = True  # firing reveals position
                results["submarines_revealed"] += 1
                if target.hp <= 0:
                    results["defender_ships_sunk"] += 1

    for sub in def_subs:
        if sub.ammo > 0.1:
            torp_dmg = sub.stats.torpedo * sub.combat_effectiveness * weather_mult * dt
            targets = [s for s in atk_ships if s.is_alive and not s.stats.is_submersible]
            if targets:
                target = max(targets, key=lambda s: s.max_hp)
                actual_dmg = torp_dmg * (3.0 + rng.uniform(-1.0, 1.0))
                target.hp = max(0.0, target.hp - actual_dmg)
                results["attacker_losses_hp"] += actual_dmg
                sub.ammo = max(0.0, sub.ammo - 0.15)
                sub.is_detected = True
                results["submarines_revealed"] += 1
                if target.hp <= 0:
                    results["attacker_ships_sunk"] += 1

    # ── Phase 2: Surface gunnery (Lanchester model) ──────────────────── #
    atk_surface = [s for s in atk_ships if s.is_alive and not s.stats.is_submersible]
    def_surface = [s for s in def_ships if s.is_alive and not s.stats.is_submersible]

    if atk_surface and def_surface:
        # Aggregate combat power
        atk_power = sum(s.stats.firepower * s.combat_effectiveness for s in atk_surface)
        def_power = sum(s.stats.firepower * s.combat_effectiveness for s in def_surface)

        atk_power *= weather_mult * atk_cmd
        def_power *= weather_mult * def_cmd

        # Initiative: faster fleet gets accuracy bonus
        atk_speed = max(s.stats.speed for s in atk_surface)
        def_speed = max(s.stats.speed for s in def_surface)
        speed_advantage = (atk_speed - def_speed) / 40.0  # normalized
        atk_init = 1.0 + 0.15 * max(0, speed_advantage)
        def_init = 1.0 + 0.15 * max(0, -speed_advantage)

        # Lanchester damage exchange
        total_power = atk_power * atk_init + def_power * def_init + 0.01
        atk_damage_dealt = (atk_power * atk_init / total_power) * atk_power * 0.3 * dt
        def_damage_dealt = (def_power * def_init / total_power) * def_power * 0.3 * dt

        # Add randomness
        atk_damage_dealt *= rng.uniform(0.7, 1.3)
        def_damage_dealt *= rng.uniform(0.7, 1.3)

        # Distribute damage to defender's ships (prioritize largest)
        _distribute_damage(def_surface, atk_damage_dealt, results, "defender", rng)
        _distribute_damage(atk_surface, def_damage_dealt, results, "attacker", rng)

    # ── Phase 3: ASW (anti-submarine warfare) ────────────────────────── #
    # Detected submarines can be depth-charged
    detected_atk_subs = [s for s in atk_ships if s.is_alive and s.stats.is_submersible and s.is_detected]
    detected_def_subs = [s for s in def_ships if s.is_alive and s.stats.is_submersible and s.is_detected]

    if detected_def_subs and def_surface:
        asw_power = sum(s.stats.anti_sub * s.combat_effectiveness for s in atk_surface) * weather_mult
        for sub in detected_def_subs:
            if asw_power > 0:
                # Depth charge attack: probabilistic
                kill_prob = min(0.5, asw_power * 0.05 * dt)
                if rng.random() < kill_prob:
                    sub.hp = 0.0
                    results["defender_ships_sunk"] += 1
                else:
                    sub.hp = max(0.0, sub.hp - asw_power * 2.0 * dt * rng.uniform(0.3, 1.0))
                    if sub.hp <= 0:
                        results["defender_ships_sunk"] += 1

    if detected_atk_subs and atk_surface:
        asw_power = sum(s.stats.anti_sub * s.combat_effectiveness for s in def_surface) * weather_mult
        for sub in detected_atk_subs:
            if asw_power > 0:
                kill_prob = min(0.5, asw_power * 0.05 * dt)
                if rng.random() < kill_prob:
                    sub.hp = 0.0
                    results["attacker_ships_sunk"] += 1
                else:
                    sub.hp = max(0.0, sub.hp - asw_power * 2.0 * dt * rng.uniform(0.3, 1.0))
                    if sub.hp <= 0:
                        results["attacker_ships_sunk"] += 1

    # ── Phase 4: Ammo depletion + fatigue ────────────────────────────── #
    for s in atk_ships + def_ships:
        if s.is_alive:
            s.ammo = max(0.0, s.ammo - 0.03 * dt)
            s.crew_fatigue = min(1.0, s.crew_fatigue + 0.02 * dt)
            s.fuel = max(0.0, s.fuel - s.stats.fuel_per_step * 0.01 * dt)

    # ── Experience gain for survivors ────────────────────────────────── #
    for s in atk_ships + def_ships:
        if s.is_alive:
            s.experience = min(1.0, s.experience + 0.005 * dt)

    return results


def _distribute_damage(
    ships: List[ShipInstance],
    total_damage: float,
    results: Dict[str, float],
    side: str,
    rng: np.random.Generator,
) -> None:
    """Distribute damage across ships, weighted by size (armor absorbs proportionally)."""
    if not ships or total_damage <= 0:
        return

    # Weight by armor (armored ships absorb more hits)
    weights = np.array([max(0.1, s.stats.armor) for s in ships])
    weights /= weights.sum()

    for i, ship in enumerate(ships):
        ship_dmg = total_damage * weights[i]
        # Armor reduces damage
        armor_reduction = ship.stats.armor / 12.0  # 0-0.83
        actual_dmg = ship_dmg * (1.0 - armor_reduction * 0.5)
        ship.hp = max(0.0, ship.hp - actual_dmg)
        ship.damage_level = min(1.0, ship.damage_level + actual_dmg / ship.max_hp * 0.3)
        results[f"{side}_losses_hp"] += actual_dmg
        if ship.hp <= 0:
            results[f"{side}_ships_sunk"] += 1


# ═══════════════════════════════════════════════════════════════════════════ #
# Anti-Submarine Warfare                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

def anti_submarine_warfare(
    asw_fleet: Fleet,
    submarine_fleet: Fleet,
    sea_state: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Dedicated ASW sweep. Escort fleet hunts enemy submarines.
    Returns dict with subs_detected, subs_damaged, subs_sunk.
    """
    results = {"subs_detected": 0, "subs_damaged": 0, "subs_sunk": 0}

    asw_ships = [s for s in asw_fleet.operational_ships if s.stats.anti_sub > 2.0]
    subs = [s for s in submarine_fleet.operational_ships if s.stats.is_submersible]

    if not asw_ships or not subs:
        return results

    total_asw = sum(s.stats.anti_sub * s.combat_effectiveness for s in asw_ships)
    weather_mod = max(0.3, 1.0 - 0.4 * sea_state)

    for sub in subs:
        if not sub.is_detected:
            # Detection attempt
            detect_chance = (total_asw / 20.0) * weather_mod * dt
            detect_chance *= (1.0 - 0.3 * sub.experience)  # experienced subs evade better
            if rng.random() < min(0.6, detect_chance):
                sub.is_detected = True
                results["subs_detected"] += 1

        if sub.is_detected:
            # Depth charge attack
            attack_power = total_asw * weather_mod * 0.3 * dt
            damage = attack_power * rng.uniform(0.5, 1.5)
            sub.hp = max(0.0, sub.hp - damage)
            results["subs_damaged"] += 1
            if sub.hp <= 0:
                results["subs_sunk"] += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════ #
# Shore Bombardment                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def shore_bombardment(
    fleet: Fleet,
    target_cluster_id: int,
    sea_state: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Fleet bombards a coastal cluster from sea.

    Returns:
      - damage_dealt: total damage to cluster (maps to hazard/resource drain)
      - ammo_consumed: ammunition used
      - accuracy: hit percentage (reduced by weather, range)
    """
    bombard_ships = [s for s in fleet.operational_ships
                     if s.stats.can_shore_bombard and s.ammo > 0.05]
    if not bombard_ships:
        return {"damage_dealt": 0.0, "ammo_consumed": 0.0, "accuracy": 0.0}

    total_firepower = sum(s.stats.firepower * s.combat_effectiveness for s in bombard_ships)

    # Shore bombardment is less accurate than naval gunnery
    base_accuracy = 0.4
    weather_penalty = 0.3 * sea_state
    accuracy = max(0.1, base_accuracy - weather_penalty)

    damage = total_firepower * accuracy * dt * rng.uniform(0.7, 1.3)

    # Consume ammo
    for s in bombard_ships:
        s.ammo = max(0.0, s.ammo - 0.05 * dt)

    return {
        "damage_dealt": damage,
        "ammo_consumed": len(bombard_ships) * 0.05 * dt,
        "accuracy": accuracy,
    }


# ═══════════════════════════════════════════════════════════════════════════ #
# Mine Damage                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_mine_damage(
    fleet: Fleet,
    mines: MineField,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> int:
    """
    Apply mine damage to a fleet transiting a mined zone.
    Returns number of ships damaged.
    """
    if mines.density <= 0.01:
        return 0

    ships_hit = 0
    for ship in fleet.operational_ships:
        # Hit probability based on mine density, ship size, and speed
        hit_chance = mines.density * 0.1 * (ship.stats.surface_detection / 10.0) * dt
        if ship.stats.is_submersible:
            hit_chance *= mines.anti_sub / mines.anti_ship if mines.anti_ship > 0 else 0.5

        if rng.random() < min(0.3, hit_chance):
            damage = (mines.anti_ship if not ship.stats.is_submersible else mines.anti_sub)
            damage *= rng.uniform(10.0, 40.0)
            ship.hp = max(0.0, ship.hp - damage)
            ship.damage_level = min(1.0, ship.damage_level + 0.2)
            ships_hit += 1

    return ships_hit
