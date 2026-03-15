"""
air_state.py — State containers for the air force system.

═══════════════════════════════════════════════════════════════════════════════
AIRCRAFT TYPES (10 across 5 roles)

  Fighters — air superiority, interception
    INTERCEPTOR          — fast climber, point defense, short range
    AIR_SUPERIORITY      — dedicated dogfighter, long loiter
    HEAVY_FIGHTER        — twin-engine, escort range, bomber destroyer

  Bombers — strategic destruction
    TACTICAL_BOMBER      — medium range, precision targets
    STRATEGIC_BOMBER     — long range, carpet bombing, heavy payload
    DIVE_BOMBER          — anti-ship + CAS specialist

  Close Air Support
    GROUND_ATTACK        — armored, low-level, tank killer

  Reconnaissance
    RECON_AIRCRAFT       — high altitude photo recon, unarmed

  Transport / Special
    TRANSPORT_AIRCRAFT   — paratroop drops, air supply
    FLYING_BOAT          — maritime patrol, ASW, sea rescue

═══════════════════════════════════════════════════════════════════════════════
AIR ZONES

  Air zones map to clusters/sea zones. Each has:
    - control: SUPERIORITY / CONTESTED / DENIED / UNCONTESTED
    - radar coverage level [0, 1]
    - flak density [0, 1] (ground-based AA)

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION

  1. Fuel per sortie — aircraft must return to base to refuel
  2. Crew fatigue — pilots need rest between missions (sortie limit/day)
  3. Aircraft attrition — losses accumulate, replacements take time
  4. Weather grounds aircraft (sea_state > 0.6 or cloud > 0.7)
  5. Radar detection — bombers are detected, giving defender time to scramble
  6. Escort range — fighters have limited range, deep raids are unescorted
  7. Night operations — reduced accuracy, higher accident rate
  8. Flak attrition — ground AA damages aircraft regardless of air superiority
  9. Crew experience — green pilots die fast, aces are irreplaceable
  10. Aircraft production takes 5-30 steps per squadron
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════ #
# Aircraft Roles & Types                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class AircraftRole(Enum):
    FIGHTER    = 0
    BOMBER     = 1
    CAS        = 2   # close air support
    RECON      = 3
    TRANSPORT  = 4


class AircraftType(Enum):
    # Fighters
    INTERCEPTOR        = 0    # Short range, fast climb, point defense
    AIR_SUPERIORITY    = 1    # Dogfighter, escort capable
    HEAVY_FIGHTER      = 2    # Twin-engine, long range escort, bomber destroyer

    # Bombers
    TACTICAL_BOMBER    = 3    # Medium range, precision
    STRATEGIC_BOMBER   = 4    # Long range, heavy payload, carpet bombing
    DIVE_BOMBER        = 5    # Anti-ship + CAS, accurate but vulnerable

    # CAS
    GROUND_ATTACK      = 6    # Armored, low level, tank killer (Il-2 / A-10 style)

    # Recon
    RECON_AIRCRAFT     = 7    # High altitude photo recon

    # Transport
    TRANSPORT_AIRCRAFT = 8    # Paratroop + cargo
    FLYING_BOAT        = 9    # Maritime patrol, ASW


N_AIRCRAFT_TYPES = len(AircraftType)


# ═══════════════════════════════════════════════════════════════════════════ #
# Aircraft Stats                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass(frozen=True)
class AircraftStats:
    name: str
    role: AircraftRole
    # Combat
    air_attack: float         # dogfight capability [0, 10]
    ground_attack: float      # CAS / bombing power [0, 10]
    naval_attack: float       # anti-ship capability [0, 8]
    bombing: float            # strategic bombing payload [0, 10]
    # Defense
    air_defense: float        # survivability in dogfight [0, 8]
    armor: float              # resistance to flak [0, 5]
    # Performance
    speed: float              # km/h (affects interception + evasion)
    range_km: float           # operational radius
    ceiling_m: float          # max altitude (affects interception)
    # Logistics
    fuel_per_sortie: float    # fuel consumed per mission
    crew_size: int            # affects casualty politics
    sortie_rate: float        # missions per step (limited by maintenance)
    # Production
    build_time: int           # steps to produce one squadron
    steel_cost: float
    fuel_cost: float
    # Special
    can_escort: bool          # can escort bombers
    can_intercept: bool       # can scramble to intercept
    carry_capacity: int       # paratroopers / cargo tons
    asw_capability: float     # anti-submarine warfare from air [0, 5]
    detection_range_km: float # radar/visual detection range


AIRCRAFT_STATS: Dict[AircraftType, AircraftStats] = {
    # ── Fighters ──────────────────────────────────────────────────────── #
    AircraftType.INTERCEPTOR: AircraftStats(
        name="Interceptor", role=AircraftRole.FIGHTER,
        air_attack=8.0, ground_attack=1.0, naval_attack=0.5, bombing=0.0,
        air_defense=6.0, armor=1.0, speed=650, range_km=600, ceiling_m=12000,
        fuel_per_sortie=1.5, crew_size=1, sortie_rate=3.0,
        build_time=8, steel_cost=3.0, fuel_cost=1.5,
        can_escort=False, can_intercept=True, carry_capacity=0,
        asw_capability=0.0, detection_range_km=20,
    ),
    AircraftType.AIR_SUPERIORITY: AircraftStats(
        name="Air Superiority Fighter", role=AircraftRole.FIGHTER,
        air_attack=9.0, ground_attack=2.0, naval_attack=1.0, bombing=0.5,
        air_defense=7.0, armor=1.5, speed=600, range_km=900, ceiling_m=11000,
        fuel_per_sortie=2.0, crew_size=1, sortie_rate=2.5,
        build_time=10, steel_cost=4.0, fuel_cost=2.0,
        can_escort=True, can_intercept=True, carry_capacity=0,
        asw_capability=0.0, detection_range_km=25,
    ),
    AircraftType.HEAVY_FIGHTER: AircraftStats(
        name="Heavy Fighter", role=AircraftRole.FIGHTER,
        air_attack=6.0, ground_attack=3.0, naval_attack=2.0, bombing=1.0,
        air_defense=5.0, armor=2.5, speed=550, range_km=1500, ceiling_m=10000,
        fuel_per_sortie=3.0, crew_size=2, sortie_rate=2.0,
        build_time=12, steel_cost=5.0, fuel_cost=2.5,
        can_escort=True, can_intercept=True, carry_capacity=0,
        asw_capability=0.5, detection_range_km=30,
    ),

    # ── Bombers ───────────────────────────────────────────────────────── #
    AircraftType.TACTICAL_BOMBER: AircraftStats(
        name="Tactical Bomber", role=AircraftRole.BOMBER,
        air_attack=1.0, ground_attack=5.0, naval_attack=3.0, bombing=6.0,
        air_defense=3.0, armor=2.0, speed=450, range_km=1200, ceiling_m=8000,
        fuel_per_sortie=3.5, crew_size=4, sortie_rate=1.5,
        build_time=15, steel_cost=6.0, fuel_cost=3.0,
        can_escort=False, can_intercept=False, carry_capacity=0,
        asw_capability=0.0, detection_range_km=15,
    ),
    AircraftType.STRATEGIC_BOMBER: AircraftStats(
        name="Strategic Bomber", role=AircraftRole.BOMBER,
        air_attack=1.5, ground_attack=3.0, naval_attack=2.0, bombing=10.0,
        air_defense=4.0, armor=3.0, speed=400, range_km=3000, ceiling_m=9000,
        fuel_per_sortie=6.0, crew_size=10, sortie_rate=1.0,
        build_time=25, steel_cost=10.0, fuel_cost=5.0,
        can_escort=False, can_intercept=False, carry_capacity=0,
        asw_capability=0.0, detection_range_km=10,
    ),
    AircraftType.DIVE_BOMBER: AircraftStats(
        name="Dive Bomber", role=AircraftRole.BOMBER,
        air_attack=2.0, ground_attack=7.0, naval_attack=8.0, bombing=4.0,
        air_defense=2.0, armor=1.5, speed=400, range_km=800, ceiling_m=7000,
        fuel_per_sortie=2.0, crew_size=2, sortie_rate=2.0,
        build_time=10, steel_cost=4.0, fuel_cost=2.0,
        can_escort=False, can_intercept=False, carry_capacity=0,
        asw_capability=1.0, detection_range_km=20,
    ),

    # ── CAS ───────────────────────────────────────────────────────────── #
    AircraftType.GROUND_ATTACK: AircraftStats(
        name="Ground Attack", role=AircraftRole.CAS,
        air_attack=3.0, ground_attack=9.0, naval_attack=2.0, bombing=3.0,
        air_defense=4.0, armor=4.0, speed=400, range_km=600, ceiling_m=5000,
        fuel_per_sortie=2.5, crew_size=1, sortie_rate=2.5,
        build_time=10, steel_cost=5.0, fuel_cost=2.0,
        can_escort=False, can_intercept=False, carry_capacity=0,
        asw_capability=0.0, detection_range_km=15,
    ),

    # ── Recon ─────────────────────────────────────────────────────────── #
    AircraftType.RECON_AIRCRAFT: AircraftStats(
        name="Reconnaissance", role=AircraftRole.RECON,
        air_attack=0.0, ground_attack=0.0, naval_attack=0.0, bombing=0.0,
        air_defense=1.0, armor=0.5, speed=700, range_km=2000, ceiling_m=15000,
        fuel_per_sortie=2.0, crew_size=1, sortie_rate=2.0,
        build_time=6, steel_cost=2.0, fuel_cost=1.0,
        can_escort=False, can_intercept=False, carry_capacity=0,
        asw_capability=0.0, detection_range_km=100,
    ),

    # ── Transport ─────────────────────────────────────────────────────── #
    AircraftType.TRANSPORT_AIRCRAFT: AircraftStats(
        name="Transport", role=AircraftRole.TRANSPORT,
        air_attack=0.0, ground_attack=0.0, naval_attack=0.0, bombing=0.0,
        air_defense=1.0, armor=1.0, speed=350, range_km=2500, ceiling_m=7000,
        fuel_per_sortie=4.0, crew_size=5, sortie_rate=1.5,
        build_time=12, steel_cost=4.0, fuel_cost=2.0,
        can_escort=False, can_intercept=False, carry_capacity=30,
        asw_capability=0.0, detection_range_km=10,
    ),
    AircraftType.FLYING_BOAT: AircraftStats(
        name="Flying Boat", role=AircraftRole.TRANSPORT,
        air_attack=1.0, ground_attack=0.5, naval_attack=2.0, bombing=1.0,
        air_defense=2.0, armor=1.5, speed=300, range_km=3000, ceiling_m=6000,
        fuel_per_sortie=3.5, crew_size=8, sortie_rate=1.0,
        build_time=15, steel_cost=5.0, fuel_cost=2.5,
        can_escort=False, can_intercept=False, carry_capacity=10,
        asw_capability=4.0, detection_range_km=60,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Aircraft Squadron                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class AircraftSquadron:
    """A squadron of aircraft (12-24 planes of one type)."""
    squadron_id: int
    aircraft_type: AircraftType
    faction_id: int
    strength: float = 1.0         # [0, 1] — fraction of full strength (24 planes)
    experience: float = 0.0       # [0, 1] — combat experience
    crew_fatigue: float = 0.0     # [0, 1] — pilot fatigue
    fuel: float = 1.0             # [0, 1] — fuel reserves
    ammo: float = 1.0             # [0, 1] — bombs/rockets remaining
    morale: float = 0.8           # [0, 1] — pilot morale
    based_at: int = 0             # cluster ID where airfield is
    mission: str = "STANDBY"      # STANDBY, CAP, ESCORT, INTERCEPT, BOMB, CAS, RECON, TRANSPORT
    sorties_today: int = 0        # anti-exploitation: limited sorties per step

    @property
    def stats(self) -> AircraftStats:
        return AIRCRAFT_STATS[self.aircraft_type]

    @property
    def is_operational(self) -> bool:
        return self.strength > 0.1 and self.fuel > 0.05 and self.crew_fatigue < 0.9

    @property
    def combat_power(self) -> float:
        """Effective combat strength [0, ~1.5]."""
        s = self.strength
        exp_mod = 1.0 + 0.5 * self.experience  # veterans are deadly
        fatigue_mod = 1.0 - 0.4 * self.crew_fatigue
        morale_mod = 0.5 + 0.5 * self.morale
        return s * exp_mod * fatigue_mod * morale_mod

    @property
    def can_sortie(self) -> bool:
        return (self.is_operational
                and self.sorties_today < int(self.stats.sortie_rate + 0.5)
                and self.fuel > self.stats.fuel_per_sortie * 0.01)

    def copy(self) -> "AircraftSquadron":
        return AircraftSquadron(
            squadron_id=self.squadron_id, aircraft_type=self.aircraft_type,
            faction_id=self.faction_id, strength=self.strength,
            experience=self.experience, crew_fatigue=self.crew_fatigue,
            fuel=self.fuel, ammo=self.ammo, morale=self.morale,
            based_at=self.based_at, mission=self.mission,
            sorties_today=self.sorties_today,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Air Wing (group of squadrons at one base)                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class AirWing:
    """Group of squadrons operating from the same airfield."""
    wing_id: int
    faction_id: int
    base_cluster: int                # airfield location
    squadrons: List[AircraftSquadron] = field(default_factory=list)

    @property
    def total_fighters(self) -> float:
        return sum(sq.combat_power * sq.stats.air_attack
                   for sq in self.squadrons if sq.stats.role == AircraftRole.FIGHTER and sq.is_operational)

    @property
    def total_bombers(self) -> float:
        return sum(sq.combat_power * sq.stats.bombing
                   for sq in self.squadrons if sq.stats.role == AircraftRole.BOMBER and sq.is_operational)

    @property
    def total_cas(self) -> float:
        return sum(sq.combat_power * sq.stats.ground_attack
                   for sq in self.squadrons if sq.stats.role == AircraftRole.CAS and sq.is_operational)

    @property
    def total_naval_strike(self) -> float:
        return sum(sq.combat_power * sq.stats.naval_attack
                   for sq in self.squadrons if sq.is_operational)

    def copy(self) -> "AirWing":
        return AirWing(
            wing_id=self.wing_id, faction_id=self.faction_id,
            base_cluster=self.base_cluster,
            squadrons=[sq.copy() for sq in self.squadrons],
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Air Zone Control                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class AirZoneControl(Enum):
    UNCONTESTED   = 0  # No significant air presence
    CONTESTED     = 1  # Both sides present, neither dominant
    SUPERIORITY   = 2  # One side has air advantage
    SUPREMACY     = 3  # Total air dominance, enemy grounded


@dataclass
class RadarStation:
    """Ground-based radar providing early warning."""
    cluster_id: int
    faction_id: int
    detection_range_km: float = 150.0
    effectiveness: float = 0.8     # [0, 1] — damaged radar is less effective
    is_operational: bool = True


@dataclass
class AirZone:
    """Air control state over a cluster or sea zone."""
    zone_id: int                    # matches cluster_id or sea_zone_id
    zone_type: str = "LAND"         # LAND or SEA
    control: AirZoneControl = AirZoneControl.UNCONTESTED
    controlling_faction: Optional[int] = None
    flak_density: float = 0.0       # [0, 1] ground-based AA
    radar_coverage: float = 0.0     # [0, 1] early warning level
    cloud_cover: float = 0.0        # [0, 1] reduces accuracy + detection

    def copy(self) -> "AirZone":
        return AirZone(
            zone_id=self.zone_id, zone_type=self.zone_type,
            control=self.control, controlling_faction=self.controlling_faction,
            flak_density=self.flak_density, radar_coverage=self.radar_coverage,
            cloud_cover=self.cloud_cover,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Air World State                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class AirWorld:
    """Complete air force state."""
    air_zones: List[AirZone]
    air_wings: List[AirWing]
    radar_stations: List[RadarStation] = field(default_factory=list)
    production_queues: Dict[int, List[Tuple[AircraftType, int]]] = field(default_factory=dict)
    next_squadron_id: int = 0
    step: int = 0

    @property
    def n_zones(self) -> int:
        return len(self.air_zones)

    def faction_wings(self, faction_id: int) -> List[AirWing]:
        return [w for w in self.air_wings if w.faction_id == faction_id]

    def faction_squadrons(self, faction_id: int) -> List[AircraftSquadron]:
        sqs = []
        for w in self.air_wings:
            if w.faction_id == faction_id:
                sqs.extend(w.squadrons)
        return sqs

    def zone_fighter_power(self, zone_id: int, faction_id: int) -> float:
        """Total fighter power available to defend/attack a zone."""
        power = 0.0
        for wing in self.air_wings:
            if wing.faction_id != faction_id:
                continue
            # Check if wing's base is in range of the zone
            # Simplified: any wing within 2 clusters of the zone can contribute
            for sq in wing.squadrons:
                if sq.stats.role == AircraftRole.FIGHTER and sq.is_operational:
                    power += sq.combat_power * sq.stats.air_attack
        return power

    def alloc_squadron_id(self) -> int:
        sid = self.next_squadron_id
        self.next_squadron_id += 1
        return sid

    def copy(self) -> "AirWorld":
        return AirWorld(
            air_zones=[z.copy() for z in self.air_zones],
            air_wings=[w.copy() for w in self.air_wings],
            radar_stations=[RadarStation(
                r.cluster_id, r.faction_id, r.detection_range_km,
                r.effectiveness, r.is_operational,
            ) for r in self.radar_stations],
            production_queues={fid: list(q) for fid, q in self.production_queues.items()},
            next_squadron_id=self.next_squadron_id,
            step=self.step,
        )
