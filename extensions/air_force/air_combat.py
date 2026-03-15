"""
air_combat.py — Air combat resolution, bombing, CAS, naval strikes, airborne ops.

Combat model:
  - Air superiority: fighter vs fighter (Lanchester with experience modifier)
  - Interception: radar-detected bombers scramble interceptors
  - Strategic bombing: payload × accuracy × (1 - flak attrition - fighter interception)
  - Close air support: ground_attack power boosts friendly ground forces
  - Anti-ship strike: dive bombers + torpedo bombers vs naval targets
  - Airborne drop: transport aircraft deliver paratroopers behind enemy lines
  - Flak: ground-based AA attrites all aircraft over hostile territory

Anti-exploitation:
  - Sortie limits per step (crew fatigue, maintenance)
  - Escort range: fighters can only escort within their range_km
  - Flak always damages bombers regardless of air superiority
  - Weather/cloud grounds aircraft or reduces accuracy
  - Pilot experience: green pilots have 3x loss rate vs aces
  - Night missions: -50% accuracy, +30% flak evasion
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(fn): return fn
        if args and callable(args[0]): return args[0]
        return _wrap

from .air_state import (
    AircraftType, AircraftRole, AircraftSquadron, AirWing, AirZone,
    AirZoneControl, AirWorld, AIRCRAFT_STATS,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Numba JIT kernels — per-squadron vectorized air combat                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@njit(cache=True)
def _air_lanchester_kernel(
    atk_power_arr, atk_speed_arr, def_power_arr, def_speed_arr,
    weather_mod, dt, rng_val,
):
    """
    Lanchester air combat on flat arrays.
    Returns (atk_damage_dealt, def_damage_dealt).
    """
    atk_power = 0.0
    for i in range(atk_power_arr.shape[0]):
        atk_power += atk_power_arr[i]
    def_power = 0.0
    for i in range(def_power_arr.shape[0]):
        def_power += def_power_arr[i]

    atk_power *= weather_mod
    def_power *= weather_mod

    # Speed initiative
    max_atk = 0.0
    for i in range(atk_speed_arr.shape[0]):
        if atk_speed_arr[i] > max_atk:
            max_atk = atk_speed_arr[i]
    max_def = 0.0
    for i in range(def_speed_arr.shape[0]):
        if def_speed_arr[i] > max_def:
            max_def = def_speed_arr[i]

    speed_adv = (max_atk - max_def) / 800.0
    atk_init = 1.0 + 0.2 * max(0.0, speed_adv)
    def_init = 1.0 + 0.2 * max(0.0, -speed_adv)

    total = atk_power * atk_init + def_power * def_init + 0.01
    atk_dmg = (atk_power * atk_init / total) * atk_power * 0.15 * dt
    def_dmg = (def_power * def_init / total) * def_power * 0.15 * dt

    atk_dmg *= rng_val
    def_dmg *= (2.0 - rng_val)

    return atk_dmg, def_dmg


@njit(cache=True)
def _distribute_air_damage_kernel(strength, power_share, total_damage, fuel, fuel_cost):
    """
    Distribute air combat damage across squadrons weighted by power share.
    Modifies strength and fuel in-place. Returns total_loss.
    """
    n = strength.shape[0]
    total_loss = 0.0
    for i in range(n):
        loss = total_damage * power_share[i] * 0.03
        strength[i] = max(0.0, strength[i] - loss)
        fuel[i] = max(0.0, fuel[i] - fuel_cost[i] * 0.01)
        total_loss += loss
    return total_loss


# ═══════════════════════════════════════════════════════════════════════════ #
# Air Detection                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_air_detection(
    radar_coverage: float,
    cloud_cover: float,
    attacker_stealth: float,
    rng: np.random.Generator,
) -> float:
    """
    Detection probability for incoming air raid.
    High radar + clear skies = high detection = defender can scramble.
    """
    base = radar_coverage * 0.7 + 0.3  # min 30% detection
    weather_mod = 1.0 - 0.4 * cloud_cover
    stealth_mod = 1.0 - 0.3 * attacker_stealth
    return float(np.clip(base * weather_mod * stealth_mod, 0.1, 0.98))


# ═══════════════════════════════════════════════════════════════════════════ #
# Air Battle (fighter vs fighter)                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def resolve_air_battle(
    attacker_squadrons: List[AircraftSquadron],
    defender_squadrons: List[AircraftSquadron],
    cloud_cover: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Resolve air-to-air combat between fighter squadrons.

    Uses modified Lanchester with:
      - Experience multiplier (aces are devastating)
      - Speed advantage (faster fighter gets initiative)
      - Weather penalty (clouds reduce accuracy)
      - Sortie consumption

    Returns: attacker_losses, defender_losses, air_kills_attacker, air_kills_defender
    """
    results = {
        "attacker_losses": 0.0, "defender_losses": 0.0,
        "attacker_kills": 0, "defender_kills": 0,
    }

    atk_fighters = [sq for sq in attacker_squadrons
                    if sq.stats.role == AircraftRole.FIGHTER and sq.can_sortie]
    def_fighters = [sq for sq in defender_squadrons
                    if sq.stats.role == AircraftRole.FIGHTER and sq.can_sortie]

    if not atk_fighters and not def_fighters:
        return results

    # Flatten squadron stats to arrays for JIT kernel
    weather_mod = max(0.4, 1.0 - 0.3 * cloud_cover)
    atk_pw = np.array([sq.combat_power * sq.stats.air_attack for sq in atk_fighters], dtype=np.float64)
    def_pw = np.array([sq.combat_power * sq.stats.air_attack for sq in def_fighters], dtype=np.float64)
    atk_sp = np.array([sq.stats.speed for sq in atk_fighters], dtype=np.float64)
    def_sp = np.array([sq.stats.speed for sq in def_fighters], dtype=np.float64)

    rng_val = rng.uniform(0.6, 1.4)
    atk_damage, def_damage = _air_lanchester_kernel(
        atk_pw, atk_sp, def_pw, def_sp, weather_mod, dt, rng_val)

    # Distribute losses via JIT kernel (flatten, compute, write back)
    atk_power_total = float(atk_pw.sum()) + 0.01
    def_power_total = float(def_pw.sum()) + 0.01

    # Defender takes atk_damage
    def_str = np.array([sq.strength for sq in def_fighters], dtype=np.float64)
    def_share = def_pw / def_power_total
    def_fuel = np.array([sq.fuel for sq in def_fighters], dtype=np.float64)
    def_fcost = np.array([sq.stats.fuel_per_sortie for sq in def_fighters], dtype=np.float64)
    def_loss = _distribute_air_damage_kernel(def_str, def_share, atk_damage, def_fuel, def_fcost)
    for i, sq in enumerate(def_fighters):
        sq.strength = def_str[i]
        sq.fuel = def_fuel[i]
        sq.sorties_today += 1
    results["defender_losses"] += def_loss

    # Attacker takes def_damage
    atk_str = np.array([sq.strength for sq in atk_fighters], dtype=np.float64)
    atk_share = atk_pw / atk_power_total
    atk_fuel = np.array([sq.fuel for sq in atk_fighters], dtype=np.float64)
    atk_fcost = np.array([sq.stats.fuel_per_sortie for sq in atk_fighters], dtype=np.float64)
    atk_loss = _distribute_air_damage_kernel(atk_str, atk_share, def_damage, atk_fuel, atk_fcost)
    for i, sq in enumerate(atk_fighters):
        sq.strength = atk_str[i]
        sq.fuel = atk_fuel[i]
        sq.sorties_today += 1
    results["attacker_losses"] += atk_loss

    # Experience gain for survivors
    for sq in atk_fighters + def_fighters:
        if sq.strength > 0.1:
            sq.experience = min(1.0, sq.experience + 0.003 * dt)
            sq.crew_fatigue = min(1.0, sq.crew_fatigue + 0.02 * dt)

    return results


# ═══════════════════════════════════════════════════════════════════════════ #
# Strategic Bombing                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def strategic_bombing(
    bomber_squadrons: List[AircraftSquadron],
    escort_squadrons: List[AircraftSquadron],
    target_zone: AirZone,
    defender_fighters: List[AircraftSquadron],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Strategic bombing raid on a target cluster.

    Pipeline:
      1. Radar detection → defender scrambles interceptors
      2. Escort fighters engage defenders (if in range)
      3. Surviving bombers face flak
      4. Remaining bombers deliver payload

    Returns:
      - damage_dealt: destruction to target cluster
      - bomber_losses: fraction of bombers lost
      - civilian_casualties: collateral damage (affects morale/unrest)
      - infrastructure_damage: damage to target's infrastructure
    """
    results = {
        "damage_dealt": 0.0, "bomber_losses": 0.0,
        "civilian_casualties": 0.0, "infrastructure_damage": 0.0,
        "escort_losses": 0.0, "defender_losses": 0.0,
    }

    bombers = [sq for sq in bomber_squadrons
               if sq.stats.role == AircraftRole.BOMBER and sq.can_sortie]
    escorts = [sq for sq in escort_squadrons
               if sq.stats.can_escort and sq.can_sortie]

    if not bombers:
        return results

    # ── 1. Radar detection ────────────────────────────────────────────── #
    detection = compute_air_detection(
        target_zone.radar_coverage, target_zone.cloud_cover,
        attacker_stealth=0.2, rng=rng,
    )

    # ── 2. Interception (escorts vs defenders) ────────────────────────── #
    if detection > 0.3 and defender_fighters:
        scrambled = [sq for sq in defender_fighters if sq.can_sortie
                     and rng.random() < detection]
        if scrambled and escorts:
            battle = resolve_air_battle(escorts, scrambled, target_zone.cloud_cover, rng, dt)
            results["escort_losses"] = battle["attacker_losses"]
            results["defender_losses"] = battle["defender_losses"]

        # Unescorted bombers face direct interception
        surviving_defenders = [sq for sq in (scrambled if detection > 0.3 else [])
                               if sq.strength > 0.1]
        if surviving_defenders:
            intercept_power = sum(sq.combat_power * sq.stats.air_attack
                                  for sq in surviving_defenders)
            for bsq in bombers:
                interception_loss = intercept_power * 0.01 * dt * rng.uniform(0.5, 1.5)
                bsq.strength = max(0.0, bsq.strength - interception_loss)
                results["bomber_losses"] += interception_loss

    # ── 3. Flak attrition ─────────────────────────────────────────────── #
    flak = target_zone.flak_density
    for bsq in bombers:
        if bsq.strength > 0.05:
            flak_loss = flak * 0.03 * dt * rng.uniform(0.5, 1.5)
            # Armor reduces flak damage
            flak_loss *= max(0.3, 1.0 - bsq.stats.armor / 5.0)
            # Cloud cover helps evade flak
            flak_loss *= max(0.5, 1.0 - 0.3 * target_zone.cloud_cover)
            bsq.strength = max(0.0, bsq.strength - flak_loss)
            results["bomber_losses"] += flak_loss

    # ── 4. Bomb delivery ──────────────────────────────────────────────── #
    surviving_payload = sum(bsq.combat_power * bsq.stats.bombing
                            for bsq in bombers if bsq.strength > 0.05)

    # Accuracy: reduced by weather, improved by experience
    avg_exp = np.mean([bsq.experience for bsq in bombers]) if bombers else 0.0
    accuracy = max(0.15, 0.5 + 0.3 * avg_exp - 0.3 * target_zone.cloud_cover)

    damage = surviving_payload * accuracy * dt * rng.uniform(0.6, 1.4)
    results["damage_dealt"] = damage
    results["infrastructure_damage"] = damage * 0.3
    results["civilian_casualties"] = damage * 0.2 * (1.0 - accuracy)  # inaccuracy = collateral

    # Consume ammo + fuel
    for bsq in bombers:
        bsq.ammo = max(0.0, bsq.ammo - 0.20)
        bsq.fuel = max(0.0, bsq.fuel - bsq.stats.fuel_per_sortie * 0.01)
        bsq.sorties_today += 1
        bsq.crew_fatigue = min(1.0, bsq.crew_fatigue + 0.03 * dt)

    return results


# ═══════════════════════════════════════════════════════════════════════════ #
# Close Air Support                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def close_air_support(
    cas_squadrons: List[AircraftSquadron],
    target_zone: AirZone,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    CAS mission over a ground battle.

    Returns:
      - ground_damage: damage dealt to enemy ground forces
      - cas_losses: losses from flak
      - suppression: enemy morale/effectiveness reduction
    """
    cas = [sq for sq in cas_squadrons
           if sq.stats.role in (AircraftRole.CAS, AircraftRole.BOMBER) and sq.can_sortie]
    if not cas:
        return {"ground_damage": 0.0, "cas_losses": 0.0, "suppression": 0.0}

    total_ga = sum(sq.combat_power * sq.stats.ground_attack for sq in cas)

    # Flak attrition (CAS flies low — more vulnerable)
    flak = target_zone.flak_density
    total_losses = 0.0
    for sq in cas:
        flak_loss = flak * 0.05 * dt * rng.uniform(0.5, 1.5)
        flak_loss *= max(0.4, 1.0 - sq.stats.armor / 5.0)
        sq.strength = max(0.0, sq.strength - flak_loss)
        total_losses += flak_loss
        sq.sorties_today += 1
        sq.fuel = max(0.0, sq.fuel - sq.stats.fuel_per_sortie * 0.01)

    accuracy = max(0.3, 0.7 - 0.2 * target_zone.cloud_cover)
    damage = total_ga * accuracy * 0.2 * dt * rng.uniform(0.7, 1.3)
    suppression = damage * 0.15

    return {"ground_damage": damage, "cas_losses": total_losses, "suppression": suppression}


# ═══════════════════════════════════════════════════════════════════════════ #
# Anti-Ship Strike                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def anti_ship_strike(
    strike_squadrons: List[AircraftSquadron],
    target_zone: AirZone,
    target_fleet_aa: float,          # total AA defense of target fleet
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Air attack on naval targets (dive bombers, torpedo bombers, naval bombers).

    Returns:
      - naval_damage: damage to ships
      - aircraft_losses: losses from ship AA
    """
    strikers = [sq for sq in strike_squadrons
                if sq.stats.naval_attack > 1.0 and sq.can_sortie]
    if not strikers:
        return {"naval_damage": 0.0, "aircraft_losses": 0.0}

    total_naval = sum(sq.combat_power * sq.stats.naval_attack for sq in strikers)

    # Ship AA fire
    total_losses = 0.0
    for sq in strikers:
        aa_loss = target_fleet_aa * 0.01 * dt * rng.uniform(0.3, 1.5)
        aa_loss *= max(0.3, 1.0 - sq.stats.armor / 5.0)
        sq.strength = max(0.0, sq.strength - aa_loss)
        total_losses += aa_loss
        sq.sorties_today += 1
        sq.fuel = max(0.0, sq.fuel - sq.stats.fuel_per_sortie * 0.01)

    accuracy = max(0.2, 0.5 - 0.2 * target_zone.cloud_cover)
    damage = total_naval * accuracy * 0.3 * dt * rng.uniform(0.5, 1.5)

    return {"naval_damage": damage, "aircraft_losses": total_losses}


# ═══════════════════════════════════════════════════════════════════════════ #
# Airborne Drop                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def airborne_drop(
    transport_squadrons: List[AircraftSquadron],
    target_zone: AirZone,
    defender_fighters: List[AircraftSquadron],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Paratroop drop behind enemy lines.

    Extremely risky:
      - Transport aircraft are slow and defenseless
      - Interception by fighters = massacre
      - Flak over drop zone
      - Scattered landing reduces effective strength
      - No heavy equipment
    """
    transports = [sq for sq in transport_squadrons
                  if sq.stats.carry_capacity > 0 and sq.can_sortie]
    if not transports:
        return {"troops_dropped": 0, "transport_losses": 0.0, "scattered": 0.0}

    total_capacity = sum(sq.stats.carry_capacity * sq.strength for sq in transports)

    # Fighter interception
    transport_losses = 0.0
    if defender_fighters:
        intercept_power = sum(sq.combat_power * sq.stats.air_attack
                              for sq in defender_fighters if sq.can_sortie)
        for tq in transports:
            loss = intercept_power * 0.02 * dt * rng.uniform(0.3, 1.5)
            tq.strength = max(0.0, tq.strength - loss)
            transport_losses += loss

    # Flak over drop zone
    flak = target_zone.flak_density
    for tq in transports:
        flak_loss = flak * 0.04 * dt * rng.uniform(0.5, 1.5)
        tq.strength = max(0.0, tq.strength - flak_loss)
        transport_losses += flak_loss
        tq.sorties_today += 1

    # Surviving capacity
    surviving = sum(sq.stats.carry_capacity * sq.strength for sq in transports if sq.strength > 0.1)

    # Scatter: wind, cloud, enemy fire disperse paratroopers
    scatter = 0.2 + 0.3 * target_zone.cloud_cover + 0.2 * flak
    effective_troops = int(surviving * (1.0 - scatter))

    return {
        "troops_dropped": max(0, effective_troops),
        "transport_losses": transport_losses,
        "scattered": scatter,
    }
