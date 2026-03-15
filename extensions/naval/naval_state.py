"""
naval_state.py — State containers for the naval warfare system.

═══════════════════════════════════════════════════════════════════════════════
SHIP CLASSES (14 types in 5 categories)

  Capital Ships — high firepower, slow, expensive, long build time
    BATTLESHIP       — 16-inch guns, massive armor, fleet flagship
    BATTLECRUISER    — fast capital ship, less armor than BB
    HEAVY_CRUISER    — 8-inch guns, versatile, moderate armor
    AIRCRAFT_CARRIER — launches air strikes, vulnerable to torpedoes

  Escorts — anti-submarine, convoy protection, screening
    LIGHT_CRUISER    — 6-inch guns, AA platform, flotilla leader
    DESTROYER        — torpedoes + depth charges, fast, ASW primary
    DESTROYER_ESCORT — cheap ASW specialist, convoy escort
    CORVETTE         — smallest warship, mass-producible, coastal ASW

  Submarines — stealth, commerce raiding, wolf packs
    FLEET_SUBMARINE  — long-range, torpedo attacks, wolfpack capable
    COASTAL_SUBMARINE— short-range, defensive, cheap
    MIDGET_SUBMARINE — harbor attack, one-shot, expendable

  Auxiliary — logistics, support
    TRANSPORT        — carries troops for amphibious ops (unarmed)
    SUPPLY_SHIP      — extends fleet range (fuel + ammo)
    MINELAYER        — deploys mine fields

  Amphibious — beach assault
    LANDING_CRAFT    — delivers troops to shore under fire

═══════════════════════════════════════════════════════════════════════════════
SEA ZONES

  Each sea zone has:
    - control_state: UNCONTESTED / CONTESTED / CONTROLLED_BY / DENIED
    - connected clusters (coastal sectors that border this zone)
    - mine density [0, 1]
    - weather/sea state affecting combat

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION

  1. Ships consume FUEL every step — no fuel = immobilized
  2. Crew fatigue accumulates during sustained ops — reduces effectiveness
  3. Damage requires STEEL + time to repair (no instant healing)
  4. Detection is probabilistic — submarines aren't auto-found
  5. Build times are long (20-80 steps) — can't spam capital ships
  6. Ammo is limited — prolonged combat depletes magazines
  7. Weather affects all naval operations (Channel storms!)
  8. Mine fields damage friend and foe alike if not swept
  9. Amphibious landings have massive attacker penalties
  10. Fleet coordination degrades with fleet size (command span limit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════ #
# Ship Categories & Classes                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class ShipCategory(Enum):
    CAPITAL    = 0   # Battleships, carriers, heavy cruisers
    ESCORT     = 1   # Destroyers, light cruisers, corvettes
    SUBMARINE  = 2   # All submarine types
    AUXILIARY  = 3   # Transports, supply ships, minelayers
    AMPHIBIOUS = 4   # Landing craft


class ShipClass(Enum):
    # Capital Ships
    BATTLESHIP        = 0
    BATTLECRUISER     = 1
    HEAVY_CRUISER     = 2
    AIRCRAFT_CARRIER  = 3
    # Escorts
    LIGHT_CRUISER     = 4
    DESTROYER         = 5
    DESTROYER_ESCORT  = 6
    CORVETTE          = 7
    # Submarines
    FLEET_SUBMARINE   = 8
    COASTAL_SUBMARINE = 9
    MIDGET_SUBMARINE  = 10
    # Auxiliary
    TRANSPORT         = 11
    SUPPLY_SHIP       = 12
    MINELAYER         = 13


N_SHIP_CLASSES = len(ShipClass)


# ═══════════════════════════════════════════════════════════════════════════ #
# Ship Stats — the "stat block" for each class                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass(frozen=True)
class ShipStats:
    """Immutable stats for a ship class."""
    name: str
    category: ShipCategory
    # Combat
    firepower: float         # surface gunnery output [0, 10]
    torpedo: float           # torpedo attack strength [0, 8]
    anti_air: float          # AA defense [0, 6]
    anti_sub: float          # ASW capability [0, 8]
    armor: float             # damage absorption [0, 10]
    # Movement
    speed: float             # knots (affects initiative + evasion)
    range_nm: float          # operational range in nautical miles
    # Stealth
    surface_detection: float # how visible (0=invisible, 10=obvious)
    sub_detection: float     # ability to detect submarines [0, 8]
    # Logistics
    fuel_per_step: float     # fuel consumed per simulation step
    ammo_capacity: float     # combat steps before resupply needed
    crew: int                # crew size (affects casualty politics)
    # Production
    build_time: int          # steps to construct
    steel_cost: float        # steel resource cost
    fuel_cost: float         # fuel to build
    # Special
    carry_capacity: int      # troops that can be carried (transport/landing craft)
    aircraft_capacity: int   # planes carried (carrier only)
    mine_capacity: int       # mines that can be laid (minelayer)
    can_shore_bombard: bool  # can fire at land targets
    is_submersible: bool     # operates underwater (stealth)


SHIP_STATS: Dict[ShipClass, ShipStats] = {
    # ── Capital Ships ─────────────────────────────────────────────────── #
    ShipClass.BATTLESHIP: ShipStats(
        name="Battleship", category=ShipCategory.CAPITAL,
        firepower=10.0, torpedo=0.0, anti_air=4.0, anti_sub=1.0,
        armor=10.0, speed=21.0, range_nm=8000,
        surface_detection=9.0, sub_detection=2.0,
        fuel_per_step=3.0, ammo_capacity=20, crew=1500,
        build_time=80, steel_cost=50.0, fuel_cost=20.0,
        carry_capacity=0, aircraft_capacity=3, mine_capacity=0,
        can_shore_bombard=True, is_submersible=False,
    ),
    ShipClass.BATTLECRUISER: ShipStats(
        name="Battlecruiser", category=ShipCategory.CAPITAL,
        firepower=8.0, torpedo=2.0, anti_air=3.5, anti_sub=1.5,
        armor=6.0, speed=28.0, range_nm=7000,
        surface_detection=8.0, sub_detection=2.5,
        fuel_per_step=2.5, ammo_capacity=18, crew=1200,
        build_time=65, steel_cost=40.0, fuel_cost=18.0,
        carry_capacity=0, aircraft_capacity=2, mine_capacity=0,
        can_shore_bombard=True, is_submersible=False,
    ),
    ShipClass.HEAVY_CRUISER: ShipStats(
        name="Heavy Cruiser", category=ShipCategory.CAPITAL,
        firepower=6.0, torpedo=3.0, anti_air=4.0, anti_sub=2.0,
        armor=5.0, speed=30.0, range_nm=9000,
        surface_detection=7.0, sub_detection=3.0,
        fuel_per_step=2.0, ammo_capacity=22, crew=800,
        build_time=50, steel_cost=30.0, fuel_cost=12.0,
        carry_capacity=0, aircraft_capacity=2, mine_capacity=0,
        can_shore_bombard=True, is_submersible=False,
    ),
    ShipClass.AIRCRAFT_CARRIER: ShipStats(
        name="Aircraft Carrier", category=ShipCategory.CAPITAL,
        firepower=1.0, torpedo=0.0, anti_air=6.0, anti_sub=1.0,
        armor=4.0, speed=30.0, range_nm=10000,
        surface_detection=9.0, sub_detection=4.0,
        fuel_per_step=3.5, ammo_capacity=30, crew=2500,
        build_time=90, steel_cost=60.0, fuel_cost=25.0,
        carry_capacity=0, aircraft_capacity=60, mine_capacity=0,
        can_shore_bombard=False, is_submersible=False,
    ),

    # ── Escorts ───────────────────────────────────────────────────────── #
    ShipClass.LIGHT_CRUISER: ShipStats(
        name="Light Cruiser", category=ShipCategory.ESCORT,
        firepower=4.0, torpedo=3.0, anti_air=5.0, anti_sub=3.0,
        armor=3.0, speed=32.0, range_nm=7000,
        surface_detection=6.0, sub_detection=4.0,
        fuel_per_step=1.5, ammo_capacity=20, crew=500,
        build_time=35, steel_cost=20.0, fuel_cost=8.0,
        carry_capacity=0, aircraft_capacity=1, mine_capacity=0,
        can_shore_bombard=True, is_submersible=False,
    ),
    ShipClass.DESTROYER: ShipStats(
        name="Destroyer", category=ShipCategory.ESCORT,
        firepower=3.0, torpedo=5.0, anti_air=3.0, anti_sub=6.0,
        armor=1.5, speed=35.0, range_nm=5000,
        surface_detection=5.0, sub_detection=6.0,
        fuel_per_step=1.0, ammo_capacity=15, crew=250,
        build_time=20, steel_cost=10.0, fuel_cost=5.0,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=20,
        can_shore_bombard=True, is_submersible=False,
    ),
    ShipClass.DESTROYER_ESCORT: ShipStats(
        name="Destroyer Escort", category=ShipCategory.ESCORT,
        firepower=1.5, torpedo=2.0, anti_air=2.0, anti_sub=7.0,
        armor=1.0, speed=24.0, range_nm=5000,
        surface_detection=4.0, sub_detection=7.0,
        fuel_per_step=0.7, ammo_capacity=12, crew=150,
        build_time=15, steel_cost=6.0, fuel_cost=3.0,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=10,
        can_shore_bombard=False, is_submersible=False,
    ),
    ShipClass.CORVETTE: ShipStats(
        name="Corvette", category=ShipCategory.ESCORT,
        firepower=1.0, torpedo=0.0, anti_air=1.5, anti_sub=5.0,
        armor=0.5, speed=16.0, range_nm=3500,
        surface_detection=3.0, sub_detection=5.0,
        fuel_per_step=0.4, ammo_capacity=10, crew=80,
        build_time=10, steel_cost=3.0, fuel_cost=1.5,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=0,
        can_shore_bombard=False, is_submersible=False,
    ),

    # ── Submarines ────────────────────────────────────────────────────── #
    ShipClass.FLEET_SUBMARINE: ShipStats(
        name="Fleet Submarine", category=ShipCategory.SUBMARINE,
        firepower=0.5, torpedo=7.0, anti_air=0.5, anti_sub=0.0,
        armor=1.0, speed=20.0, range_nm=12000,
        surface_detection=1.5, sub_detection=1.0,
        fuel_per_step=0.8, ammo_capacity=12, crew=60,
        build_time=25, steel_cost=8.0, fuel_cost=4.0,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=12,
        can_shore_bombard=False, is_submersible=True,
    ),
    ShipClass.COASTAL_SUBMARINE: ShipStats(
        name="Coastal Submarine", category=ShipCategory.SUBMARINE,
        firepower=0.0, torpedo=5.0, anti_air=0.0, anti_sub=0.0,
        armor=0.5, speed=14.0, range_nm=3000,
        surface_detection=1.0, sub_detection=0.5,
        fuel_per_step=0.4, ammo_capacity=6, crew=30,
        build_time=15, steel_cost=4.0, fuel_cost=2.0,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=6,
        can_shore_bombard=False, is_submersible=True,
    ),
    ShipClass.MIDGET_SUBMARINE: ShipStats(
        name="Midget Submarine", category=ShipCategory.SUBMARINE,
        firepower=0.0, torpedo=3.0, anti_air=0.0, anti_sub=0.0,
        armor=0.2, speed=8.0, range_nm=500,
        surface_detection=0.5, sub_detection=0.0,
        fuel_per_step=0.1, ammo_capacity=2, crew=4,
        build_time=5, steel_cost=1.0, fuel_cost=0.5,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=2,
        can_shore_bombard=False, is_submersible=True,
    ),

    # ── Auxiliary ─────────────────────────────────────────────────────── #
    ShipClass.TRANSPORT: ShipStats(
        name="Transport", category=ShipCategory.AUXILIARY,
        firepower=0.2, torpedo=0.0, anti_air=1.0, anti_sub=0.0,
        armor=0.5, speed=15.0, range_nm=6000,
        surface_detection=7.0, sub_detection=0.0,
        fuel_per_step=1.0, ammo_capacity=5, crew=200,
        build_time=12, steel_cost=5.0, fuel_cost=3.0,
        carry_capacity=2000, aircraft_capacity=0, mine_capacity=0,
        can_shore_bombard=False, is_submersible=False,
    ),
    ShipClass.SUPPLY_SHIP: ShipStats(
        name="Supply Ship", category=ShipCategory.AUXILIARY,
        firepower=0.1, torpedo=0.0, anti_air=0.5, anti_sub=0.0,
        armor=0.3, speed=14.0, range_nm=8000,
        surface_detection=6.0, sub_detection=0.0,
        fuel_per_step=0.8, ammo_capacity=3, crew=100,
        build_time=10, steel_cost=4.0, fuel_cost=2.0,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=0,
        can_shore_bombard=False, is_submersible=False,
    ),
    ShipClass.MINELAYER: ShipStats(
        name="Minelayer", category=ShipCategory.AUXILIARY,
        firepower=0.5, torpedo=0.0, anti_air=1.0, anti_sub=0.0,
        armor=1.0, speed=18.0, range_nm=4000,
        surface_detection=5.0, sub_detection=1.0,
        fuel_per_step=0.6, ammo_capacity=8, crew=120,
        build_time=12, steel_cost=5.0, fuel_cost=2.5,
        carry_capacity=0, aircraft_capacity=0, mine_capacity=200,
        can_shore_bombard=False, is_submersible=False,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Ship Instance                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ShipInstance:
    """A single ship in a fleet."""
    ship_id: int
    ship_class: ShipClass
    faction_id: int
    hp: float                    # [0, max_hp] — 0 = sunk
    max_hp: float = 100.0
    ammo: float = 1.0           # fraction remaining [0, 1]
    fuel: float = 1.0           # fraction remaining [0, 1]
    crew_fatigue: float = 0.0   # [0, 1] — 1 = exhausted, combat penalty
    damage_level: float = 0.0   # [0, 1] — accumulated damage needing repair
    experience: float = 0.0     # [0, 1] — combat experience bonus
    is_detected: bool = False   # has been spotted by enemy

    @property
    def is_alive(self) -> bool:
        return self.hp > 0.0

    @property
    def is_operational(self) -> bool:
        """Can fight: alive + has fuel + ammo + not critically damaged."""
        return self.hp > 0 and self.fuel > 0.05 and self.damage_level < 0.8

    @property
    def combat_effectiveness(self) -> float:
        """Combined effectiveness modifier [0, 1]."""
        hp_mod = self.hp / self.max_hp
        fatigue_mod = 1.0 - 0.4 * self.crew_fatigue
        damage_mod = 1.0 - 0.6 * self.damage_level
        exp_mod = 1.0 + 0.3 * self.experience
        ammo_mod = 0.3 + 0.7 * self.ammo  # can still fight at low ammo, just weaker
        return max(0.05, hp_mod * fatigue_mod * damage_mod * exp_mod * ammo_mod)

    @property
    def stats(self) -> ShipStats:
        return SHIP_STATS[self.ship_class]

    def copy(self) -> "ShipInstance":
        return ShipInstance(
            ship_id=self.ship_id, ship_class=self.ship_class,
            faction_id=self.faction_id, hp=self.hp, max_hp=self.max_hp,
            ammo=self.ammo, fuel=self.fuel, crew_fatigue=self.crew_fatigue,
            damage_level=self.damage_level, experience=self.experience,
            is_detected=self.is_detected,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Fleet                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class Fleet:
    """A group of ships operating together in a sea zone."""
    fleet_id: int
    faction_id: int
    sea_zone_id: int
    ships: List[ShipInstance] = field(default_factory=list)
    mission: str = "PATROL"    # PATROL, CONVOY_ESCORT, BLOCKADE, RAID, SHORE_BOMBARD, AMPHIBIOUS

    @property
    def alive_ships(self) -> List[ShipInstance]:
        return [s for s in self.ships if s.is_alive]

    @property
    def operational_ships(self) -> List[ShipInstance]:
        return [s for s in self.ships if s.is_operational]

    @property
    def total_firepower(self) -> float:
        return sum(s.stats.firepower * s.combat_effectiveness for s in self.operational_ships)

    @property
    def total_torpedo(self) -> float:
        return sum(s.stats.torpedo * s.combat_effectiveness for s in self.operational_ships)

    @property
    def total_anti_sub(self) -> float:
        return sum(s.stats.anti_sub * s.combat_effectiveness for s in self.operational_ships)

    @property
    def total_anti_air(self) -> float:
        return sum(s.stats.anti_air * s.combat_effectiveness for s in self.operational_ships)

    @property
    def transport_capacity(self) -> int:
        return sum(s.stats.carry_capacity for s in self.operational_ships
                   if s.ship_class in (ShipClass.TRANSPORT,))

    @property
    def fleet_speed(self) -> float:
        """Fleet speed = slowest operational ship."""
        speeds = [s.stats.speed for s in self.operational_ships]
        return min(speeds) if speeds else 0.0

    @property
    def submarine_count(self) -> int:
        return sum(1 for s in self.operational_ships if s.stats.is_submersible)

    @property
    def avg_detection_risk(self) -> float:
        """How detectable this fleet is. Subs are stealthy; big ships are obvious."""
        if not self.operational_ships:
            return 0.0
        return sum(s.stats.surface_detection for s in self.operational_ships) / len(self.operational_ships)

    @property
    def command_span_penalty(self) -> float:
        """Large fleets are harder to coordinate. Anti-exploitation."""
        n = len(self.operational_ships)
        if n <= 6:
            return 0.0
        if n <= 12:
            return 0.05 * (n - 6)
        return 0.30 + 0.03 * (n - 12)  # severe penalty past 12 ships

    def copy(self) -> "Fleet":
        return Fleet(
            fleet_id=self.fleet_id, faction_id=self.faction_id,
            sea_zone_id=self.sea_zone_id,
            ships=[s.copy() for s in self.ships],
            mission=self.mission,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Sea Zone                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class SeaZoneControl(Enum):
    UNCONTESTED  = 0   # No significant naval presence
    CONTESTED    = 1   # Both sides present, active combat likely
    CONTROLLED   = 2   # One side has naval superiority
    DENIED       = 3   # Enemy submarines/mines make transit dangerous


@dataclass
class MineField:
    """Mine density in a sea zone. Damages any ship transiting."""
    density: float = 0.0       # [0, 1] — mines per km²
    faction_id: int = -1       # who laid them (-1 = neutral)
    anti_ship: float = 0.5     # damage to surface ships
    anti_sub: float = 0.3      # damage to submarines


@dataclass
class SeaZone:
    """
    A body of water connecting coastal clusters.

    connected_clusters: which land clusters border this zone
    adjacent_zones: which other sea zones connect to this one
    """
    zone_id: int
    name: str
    connected_clusters: List[int]       # coastal sectors bordering this zone
    adjacent_zones: List[int] = field(default_factory=list)
    control: SeaZoneControl = SeaZoneControl.UNCONTESTED
    controlling_faction: Optional[int] = None
    fleets: List[Fleet] = field(default_factory=list)
    mines: MineField = field(default_factory=MineField)
    sea_state: float = 0.0              # [0, 1] — 0=calm, 1=storm (reduces combat effectiveness)
    width_km: float = 50.0              # crossing distance

    @property
    def faction_fleets(self) -> Dict[int, List[Fleet]]:
        result: Dict[int, List[Fleet]] = {}
        for f in self.fleets:
            result.setdefault(f.faction_id, []).append(f)
        return result

    def copy(self) -> "SeaZone":
        return SeaZone(
            zone_id=self.zone_id, name=self.name,
            connected_clusters=list(self.connected_clusters),
            adjacent_zones=list(self.adjacent_zones),
            control=self.control, controlling_faction=self.controlling_faction,
            fleets=[f.copy() for f in self.fleets],
            mines=MineField(self.mines.density, self.mines.faction_id,
                           self.mines.anti_ship, self.mines.anti_sub),
            sea_state=self.sea_state, width_km=self.width_km,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Convoy Route                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ConvoyRoute:
    """
    A supply convoy route across sea zones.
    Convoys deliver resources between clusters via sea.
    """
    route_id: int
    faction_id: int
    origin_cluster: int          # port of departure
    destination_cluster: int     # port of arrival
    sea_zones: List[int]         # zone IDs the convoy transits
    cargo_capacity: float = 100.0  # resource units per delivery
    escort_fleet_id: Optional[int] = None  # fleet assigned to protect
    is_active: bool = True
    deliveries_completed: int = 0
    losses_suffered: int = 0     # ships sunk on this route

    @property
    def loss_rate(self) -> float:
        total = self.deliveries_completed + self.losses_suffered
        if total == 0:
            return 0.0
        return self.losses_suffered / total


# ═══════════════════════════════════════════════════════════════════════════ #
# Naval World State                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class NavalWorld:
    """Complete naval warfare state."""
    sea_zones: List[SeaZone]
    convoy_routes: List[ConvoyRoute] = field(default_factory=list)
    shipyard_queues: Dict[int, List[Tuple[ShipClass, int]]] = field(default_factory=dict)
    # faction_id → list of (ship_class, steps_remaining)
    next_ship_id: int = 0
    step: int = 0

    @property
    def n_zones(self) -> int:
        return len(self.sea_zones)

    def get_zone(self, zone_id: int) -> SeaZone:
        return self.sea_zones[zone_id]

    def faction_ships(self, faction_id: int) -> List[ShipInstance]:
        """All ships belonging to a faction across all zones."""
        ships = []
        for zone in self.sea_zones:
            for fleet in zone.fleets:
                if fleet.faction_id == faction_id:
                    ships.extend(fleet.alive_ships)
        return ships

    def faction_fleets(self, faction_id: int) -> List[Fleet]:
        fleets = []
        for zone in self.sea_zones:
            for fleet in zone.fleets:
                if fleet.faction_id == faction_id:
                    fleets.append(fleet)
        return fleets

    def alloc_ship_id(self) -> int:
        sid = self.next_ship_id
        self.next_ship_id += 1
        return sid

    def copy(self) -> "NavalWorld":
        return NavalWorld(
            sea_zones=[z.copy() for z in self.sea_zones],
            convoy_routes=[ConvoyRoute(
                route_id=r.route_id, faction_id=r.faction_id,
                origin_cluster=r.origin_cluster,
                destination_cluster=r.destination_cluster,
                sea_zones=list(r.sea_zones),
                cargo_capacity=r.cargo_capacity,
                escort_fleet_id=r.escort_fleet_id,
                is_active=r.is_active,
                deliveries_completed=r.deliveries_completed,
                losses_suffered=r.losses_suffered,
            ) for r in self.convoy_routes],
            shipyard_queues={fid: list(q) for fid, q in self.shipyard_queues.items()},
            next_ship_id=self.next_ship_id,
            step=self.step,
        )
