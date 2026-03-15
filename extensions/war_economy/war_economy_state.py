"""
war_economy_state.py — Sophisticated multi-sector economy with Leontief I/O model.

═══════════════════════════════════════════════════════════════════════════════
ECONOMIC SECTORS (7)

  AGRICULTURE    — Produces raw food. Requires fuel + chemicals (fertilizer).
  MINING         — Extracts ore, coal, raw materials. Requires fuel + steel.
  ENERGY         — Refines crude oil + coal into fuel. Requires steel (infra).
  HEAVY_INDUSTRY — Smelts steel from ore + coal. Requires fuel.
  MANUFACTURING  — Produces consumer goods, military equipment, ammo.
  CONSTRUCTION   — Builds/repairs infrastructure + capacity. Consumes steel.
  SERVICES       — Healthcare, finance, education. Boosts productivity.

═══════════════════════════════════════════════════════════════════════════════
RESOURCES — 3-tier supply chain (12 types)

  Tier 1 — Raw (extracted from terrain):
    CRUDE_OIL, IRON_ORE, COAL, RAW_FOOD, RAW_MATERIALS

  Tier 2 — Processed (from Tier 1 via sector transformation):
    FUEL, STEEL, CHEMICALS, PROCESSED_FOOD

  Tier 3 — Finished (from Tier 2 via Manufacturing):
    CONSUMER_GOODS, MILITARY_EQUIPMENT, AMMUNITION

  Key constraint: you cannot produce Tier 3 without Tier 2, which requires
  Tier 1.  Bottleneck in any tier cascades upward.

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION MECHANISMS

  1. Leontief bottleneck: output = min(input_i / required_i) × capacity
  2. Diminishing returns: output = cap × (1 - exp(-labor / cap))
  3. Capital depreciation: 1-2% per step, needs Construction to maintain
  4. Labor rigidity: max 10% workforce shift per step between sectors
  5. Inventory carrying cost: stockpiles decay, consume resources
  6. Infrastructure dependency: damaged infra reduces ALL sector output
  7. Price discovery: market-clearing from supply/demand ratios
  8. War fatigue: prolonged conflict reduces productivity globally
  9. Transition friction: reallocating capital wastes 20% during transition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════ #
# Economic Sectors                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class EconSector(Enum):
    """Seven interconnected economic sectors."""
    AGRICULTURE    = 0   # Food production — needs fuel, chemicals
    MINING         = 1   # Ore/coal/material extraction — needs fuel, steel
    ENERGY         = 2   # Oil refining, power generation — needs steel
    HEAVY_INDUSTRY = 3   # Steel smelting, heavy machinery — needs fuel, ore, coal
    MANUFACTURING  = 4   # Consumer + military goods — needs steel, chemicals, fuel
    CONSTRUCTION   = 5   # Infrastructure, buildings — needs steel, fuel
    SERVICES       = 6   # Healthcare, finance, education — boosts productivity


N_SECTORS = len(EconSector)
SECTOR_NAMES = [s.name for s in EconSector]


# ═══════════════════════════════════════════════════════════════════════════ #
# Resources — 3-tier supply chain                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class Resource(Enum):
    """12 resources in 3 tiers. Higher tiers require lower tiers to produce."""
    # Tier 1 — Raw (extracted from terrain by Mining/Agriculture/Energy)
    CRUDE_OIL      = 0
    IRON_ORE       = 1
    COAL           = 2
    RAW_FOOD       = 3
    RAW_MATERIALS  = 4    # timber, rubber latex, fibres, etc.
    # Tier 2 — Processed (transformed by Energy/Heavy Industry/Agriculture)
    FUEL           = 5    # from crude_oil + coal
    STEEL          = 6    # from iron_ore + coal
    CHEMICALS      = 7    # from coal + raw_materials (fertiliser, explosives precursor)
    PROCESSED_FOOD = 8    # from raw_food (+ fuel for transport/preservation)
    # Tier 3 — Finished (manufactured from Tier 2)
    CONSUMER_GOODS     = 9    # from steel + chemicals + processed_food
    MILITARY_EQUIPMENT = 10   # from steel + chemicals + fuel
    AMMUNITION         = 11   # from chemicals + steel


N_RESOURCES = len(Resource)
RESOURCE_NAMES = [r.name for r in Resource]

# Tier classification
TIER_1 = [Resource.CRUDE_OIL, Resource.IRON_ORE, Resource.COAL,
          Resource.RAW_FOOD, Resource.RAW_MATERIALS]
TIER_2 = [Resource.FUEL, Resource.STEEL, Resource.CHEMICALS, Resource.PROCESSED_FOOD]
TIER_3 = [Resource.CONSUMER_GOODS, Resource.MILITARY_EQUIPMENT, Resource.AMMUNITION]

TIER_1_IDX = np.array([r.value for r in TIER_1])
TIER_2_IDX = np.array([r.value for r in TIER_2])
TIER_3_IDX = np.array([r.value for r in TIER_3])


# ═══════════════════════════════════════════════════════════════════════════ #
# Leontief Input-Output Matrix                                                #
# ═══════════════════════════════════════════════════════════════════════════ #
#
# SECTOR_INPUTS[sector][resource] = amount of resource consumed per unit output.
# SECTOR_OUTPUTS[sector][resource] = amount of resource produced per unit output.
#
# The Leontief constraint: actual_output = capacity × min_r(available_r / required_r)
# This means a shortage of ANY input bottlenecks the entire sector.

# Input requirements: how much of each resource a sector consumes per unit of output
_SECTOR_INPUTS = np.zeros((N_SECTORS, N_RESOURCES), dtype=np.float64)

# Agriculture: consumes fuel (tractors) + chemicals (fertiliser)
_SECTOR_INPUTS[EconSector.AGRICULTURE.value, Resource.FUEL.value] = 0.3
_SECTOR_INPUTS[EconSector.AGRICULTURE.value, Resource.CHEMICALS.value] = 0.4

# Mining: consumes fuel (machinery) + steel (drill bits, rails)
_SECTOR_INPUTS[EconSector.MINING.value, Resource.FUEL.value] = 0.5
_SECTOR_INPUTS[EconSector.MINING.value, Resource.STEEL.value] = 0.2

# Energy: consumes crude_oil + coal (feedstock) + steel (infrastructure)
_SECTOR_INPUTS[EconSector.ENERGY.value, Resource.CRUDE_OIL.value] = 0.6
_SECTOR_INPUTS[EconSector.ENERGY.value, Resource.COAL.value] = 0.4
_SECTOR_INPUTS[EconSector.ENERGY.value, Resource.STEEL.value] = 0.1

# Heavy Industry: consumes iron_ore + coal (smelting) + fuel (furnaces)
_SECTOR_INPUTS[EconSector.HEAVY_INDUSTRY.value, Resource.IRON_ORE.value] = 0.7
_SECTOR_INPUTS[EconSector.HEAVY_INDUSTRY.value, Resource.COAL.value] = 0.5
_SECTOR_INPUTS[EconSector.HEAVY_INDUSTRY.value, Resource.FUEL.value] = 0.3

# Manufacturing: consumes steel + chemicals + fuel
_SECTOR_INPUTS[EconSector.MANUFACTURING.value, Resource.STEEL.value] = 0.5
_SECTOR_INPUTS[EconSector.MANUFACTURING.value, Resource.CHEMICALS.value] = 0.3
_SECTOR_INPUTS[EconSector.MANUFACTURING.value, Resource.FUEL.value] = 0.2

# Construction: consumes steel + fuel + raw_materials (timber)
_SECTOR_INPUTS[EconSector.CONSTRUCTION.value, Resource.STEEL.value] = 0.6
_SECTOR_INPUTS[EconSector.CONSTRUCTION.value, Resource.FUEL.value] = 0.3
_SECTOR_INPUTS[EconSector.CONSTRUCTION.value, Resource.RAW_MATERIALS.value] = 0.3

# Services: consumes processed_food + consumer_goods (operating inputs)
_SECTOR_INPUTS[EconSector.SERVICES.value, Resource.PROCESSED_FOOD.value] = 0.2
_SECTOR_INPUTS[EconSector.SERVICES.value, Resource.CONSUMER_GOODS.value] = 0.2

SECTOR_INPUTS: NDArray[np.float64] = _SECTOR_INPUTS

# Output production: what each sector produces per unit of output
_SECTOR_OUTPUTS = np.zeros((N_SECTORS, N_RESOURCES), dtype=np.float64)

# Agriculture → raw_food
_SECTOR_OUTPUTS[EconSector.AGRICULTURE.value, Resource.RAW_FOOD.value] = 1.0

# Mining → iron_ore, coal, raw_materials, crude_oil (proportional)
_SECTOR_OUTPUTS[EconSector.MINING.value, Resource.IRON_ORE.value] = 0.35
_SECTOR_OUTPUTS[EconSector.MINING.value, Resource.COAL.value] = 0.30
_SECTOR_OUTPUTS[EconSector.MINING.value, Resource.RAW_MATERIALS.value] = 0.20
_SECTOR_OUTPUTS[EconSector.MINING.value, Resource.CRUDE_OIL.value] = 0.15

# Energy → fuel
_SECTOR_OUTPUTS[EconSector.ENERGY.value, Resource.FUEL.value] = 1.0

# Heavy Industry → steel + chemicals (byproduct)
_SECTOR_OUTPUTS[EconSector.HEAVY_INDUSTRY.value, Resource.STEEL.value] = 0.8
_SECTOR_OUTPUTS[EconSector.HEAVY_INDUSTRY.value, Resource.CHEMICALS.value] = 0.2

# Manufacturing → consumer_goods, military_equipment, ammunition (split by priority)
_SECTOR_OUTPUTS[EconSector.MANUFACTURING.value, Resource.CONSUMER_GOODS.value] = 0.4
_SECTOR_OUTPUTS[EconSector.MANUFACTURING.value, Resource.MILITARY_EQUIPMENT.value] = 0.35
_SECTOR_OUTPUTS[EconSector.MANUFACTURING.value, Resource.AMMUNITION.value] = 0.25

# Construction → infrastructure (not a stockpileable resource, handled separately)
# Services → productivity boost (not a stockpileable resource, handled separately)

# Agriculture also produces processed_food from raw_food (simplified chain)
_SECTOR_OUTPUTS[EconSector.AGRICULTURE.value, Resource.PROCESSED_FOOD.value] = 0.3

SECTOR_OUTPUTS: NDArray[np.float64] = _SECTOR_OUTPUTS


# ═══════════════════════════════════════════════════════════════════════════ #
# Terrain → Resource Endowment                                               #
# ═══════════════════════════════════════════════════════════════════════════ #
# Base extraction rate for Tier 1 resources by terrain type.

TERRAIN_ENDOWMENT: Dict[str, NDArray[np.float64]] = {
    # terrain → [CRUDE_OIL, IRON_ORE, COAL, RAW_FOOD, RAW_MATERIALS]
    "DESERT":    np.array([3.0, 0.0, 0.0, 0.0, 0.0]),   # Oil fields
    "MARSH":     np.array([2.0, 0.0, 0.2, 0.3, 0.5]),   # Oil + some organics
    "MOUNTAINS": np.array([0.0, 2.5, 2.0, 0.0, 0.3]),   # Mining heartland
    "PLAINS":    np.array([0.0, 0.0, 0.0, 3.0, 0.5]),   # Breadbasket
    "FOREST":    np.array([0.0, 0.0, 0.2, 0.5, 2.5]),   # Timber, rubber
    "URBAN":     np.array([0.0, 0.3, 0.5, 0.0, 0.2]),   # Some coal, scrap
    "OPEN":      np.array([0.5, 0.2, 0.3, 1.5, 0.5]),   # Mixed
}
DEFAULT_ENDOWMENT = np.array([0.3, 0.3, 0.3, 1.0, 0.3])


# ═══════════════════════════════════════════════════════════════════════════ #
# Spoilage / Carrying Costs                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #
# Fraction of stockpile lost per step. Perishables decay fast; metals don't.

SPOILAGE = np.array([
    0.002,  # CRUDE_OIL — evaporation
    0.000,  # IRON_ORE — doesn't spoil
    0.000,  # COAL — doesn't spoil
    0.008,  # RAW_FOOD — highly perishable
    0.002,  # RAW_MATERIALS — slow degradation
    0.003,  # FUEL — evaporation + degradation
    0.000,  # STEEL — doesn't spoil
    0.004,  # CHEMICALS — stability decay
    0.005,  # PROCESSED_FOOD — moderate shelf life
    0.002,  # CONSUMER_GOODS — fashion obsolescence
    0.001,  # MILITARY_EQUIPMENT — maintenance
    0.003,  # AMMUNITION — propellant degradation
], dtype=np.float64)

MAX_STOCKPILE = 200.0  # per resource per cluster


# ═══════════════════════════════════════════════════════════════════════════ #
# Anti-Exploitation Constants                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

LABOR_SHIFT_MAX = 0.10         # max 10% of workforce can shift sectors per step
CAPITAL_DEPRECIATION = 0.015   # 1.5% capacity lost per step without maintenance
TRANSITION_FRICTION = 0.20     # 20% waste when reallocating capital between sectors
WAR_FATIGUE_RATE = 0.001       # per-step productivity loss during sustained conflict
INFRASTRUCTURE_DECAY = 0.005   # per-step infra decay without Construction sector
DIMINISHING_RETURNS_K = 2.0    # steepness of diminishing returns curve


# ═══════════════════════════════════════════════════════════════════════════ #
# Trade Agreement                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class TradeAgreement:
    """
    Bilateral trade deal between two factions.

    The exporter sends `amount` of `resource` per step to the importer.
    Payment flows back as GDP. Routes can be blockaded/sanctioned.
    """
    exporter_faction: int
    importer_faction: int
    resource: Resource
    amount_per_step: float
    price_ratio: float = 0.5
    remaining_steps: int = 100
    route_clusters: Tuple[int, ...] = ()
    is_blocked: bool = False

    @property
    def is_active(self) -> bool:
        return self.remaining_steps > 0 and not self.is_blocked


# ═══════════════════════════════════════════════════════════════════════════ #
# Lend-Lease Package                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class LendLeasePackage:
    """
    Asymmetric military/economic aid (historical: US Lend-Lease to UK/USSR).
    One-directional, no immediate payment, tied to alliance threshold.
    """
    donor_faction: int
    recipient_faction: int
    resource_amounts: Dict[Resource, float]
    equipment_bonus: float = 0.0
    remaining_steps: int = 200
    alliance_threshold: float = 0.3
    donor_gdp_cost: float = 0.02
    political_cost: float = 0.01

    @property
    def is_active(self) -> bool:
        return self.remaining_steps > 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Cluster Economy State — Sector-Based                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ClusterEconomy:
    """
    Per-cluster multi-sector economy state.

    All arrays are shape (N_SECTORS,) or (N_RESOURCES,).
    Vectorised for fast NumPy computation.

    sector_capacity     — installed production capacity per sector [0, max]
    sector_labor        — fraction of workforce allocated to each sector (sums ≤ 1)
    sector_output       — actual output last step (for observation)
    resource_stockpile  — current stockpile per resource [0, MAX_STOCKPILE]
    terrain_endowment   — Tier 1 extraction bonus from terrain (fixed at init)
    infrastructure      — overall infrastructure level [0, 1]; affects ALL sectors
    war_damage          — accumulated war damage [0, 1]; reduces capacity
    sanctions_level     — trade sanctions intensity [0, 1]
    blockaded           — trade route blockade flag
    war_bond_active     — production burst from war bonds
    war_bond_remaining  — steps left on war bond
    """
    cluster_id: int
    sector_capacity: NDArray[np.float64]      # (N_SECTORS,)
    sector_labor: NDArray[np.float64]         # (N_SECTORS,) sums ≤ 1.0
    sector_output: NDArray[np.float64]        # (N_SECTORS,) last step's output
    resource_stockpile: NDArray[np.float64]   # (N_RESOURCES,)
    terrain_endowment: NDArray[np.float64]    # (5,) Tier 1 extraction rates
    infrastructure: float = 0.7
    war_damage: float = 0.0
    sanctions_level: float = 0.0
    blockaded: bool = False
    war_bond_active: bool = False
    war_bond_remaining: int = 0

    def copy(self) -> "ClusterEconomy":
        return ClusterEconomy(
            cluster_id=self.cluster_id,
            sector_capacity=self.sector_capacity.copy(),
            sector_labor=self.sector_labor.copy(),
            sector_output=self.sector_output.copy(),
            resource_stockpile=self.resource_stockpile.copy(),
            terrain_endowment=self.terrain_endowment.copy(),
            infrastructure=self.infrastructure,
            war_damage=self.war_damage,
            sanctions_level=self.sanctions_level,
            blockaded=self.blockaded,
            war_bond_active=self.war_bond_active,
            war_bond_remaining=self.war_bond_remaining,
        )

    def stockpile_ratio(self, resource: Resource) -> float:
        """Stockpile fullness [0, 1]."""
        return min(self.resource_stockpile[resource.value] / MAX_STOCKPILE, 1.0)

    def has_shortage(self, resource: Resource, threshold: float = 10.0) -> bool:
        return self.resource_stockpile[resource.value] < threshold

    @property
    def effective_capacity(self) -> NDArray[np.float64]:
        """Capacity after infrastructure and war damage penalties."""
        infra_mult = 0.3 + 0.7 * self.infrastructure
        damage_mult = 1.0 - 0.8 * self.war_damage
        return self.sector_capacity * infra_mult * damage_mult

    @property
    def total_labor_allocated(self) -> float:
        return float(np.sum(self.sector_labor))


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Economy State                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionEconomy:
    """
    Faction-level economic aggregates and policy settings.

    war_mobilization  — [0, 1] fraction of economy shifted to military production
    fiscal_debt       — accumulated government debt [0, 1+]
    inflation         — price level growth [0, 1]; high inflation erodes GDP
    trade_balance     — net resource flow value
    war_fatigue       — accumulated productivity penalty from sustained conflict
    manufacturing_priority — [0,1] bias toward military vs consumer goods in Manufacturing
    sanctions_imposed — which factions are sanctioned and at what level
    """
    faction_id: int
    war_mobilization: float = 0.0
    fiscal_debt: float = 0.1
    inflation: float = 0.02
    trade_balance: float = 0.0
    war_fatigue: float = 0.0
    lend_lease_given: float = 0.0
    lend_lease_received: float = 0.0
    manufacturing_priority: float = 0.5   # 0=consumer, 1=military
    sanctions_imposed: Dict[int, float] = field(default_factory=dict)

    def copy(self) -> "FactionEconomy":
        return FactionEconomy(
            faction_id=self.faction_id,
            war_mobilization=self.war_mobilization,
            fiscal_debt=self.fiscal_debt,
            inflation=self.inflation,
            trade_balance=self.trade_balance,
            war_fatigue=self.war_fatigue,
            lend_lease_given=self.lend_lease_given,
            lend_lease_received=self.lend_lease_received,
            manufacturing_priority=self.manufacturing_priority,
            sanctions_imposed=dict(self.sanctions_imposed),
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# War Economy World State                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class WarEconomyWorld:
    """
    Complete multi-sector war economy state.

    Attaches to CowWorldState or GravitasWorld as an optional extension.
    All per-cluster data is vectorisable for fast simulation.
    """
    cluster_economies: List[ClusterEconomy]
    faction_economies: Dict[int, FactionEconomy]
    trade_agreements: List[TradeAgreement] = field(default_factory=list)
    lend_lease_packages: List[LendLeasePackage] = field(default_factory=list)
    market_prices: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(N_RESOURCES, dtype=np.float64)
    )
    step: int = 0

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_economies)

    def get_cluster(self, cid: int) -> ClusterEconomy:
        return self.cluster_economies[cid]

    def get_faction(self, fid: int) -> FactionEconomy:
        return self.faction_economies[fid]

    def active_trades(self) -> List[TradeAgreement]:
        return [t for t in self.trade_agreements if t.is_active]

    def active_lend_lease(self) -> List[LendLeasePackage]:
        return [ll for ll in self.lend_lease_packages if ll.is_active]

    def faction_total_stockpile(self, faction_id: int,
                                 cluster_owners: Dict[int, int]) -> NDArray[np.float64]:
        """Sum stockpiles across all clusters owned by a faction."""
        total = np.zeros(N_RESOURCES, dtype=np.float64)
        for ce in self.cluster_economies:
            if cluster_owners.get(ce.cluster_id) == faction_id:
                total += ce.resource_stockpile
        return total

    def copy(self) -> "WarEconomyWorld":
        return WarEconomyWorld(
            cluster_economies=[ce.copy() for ce in self.cluster_economies],
            faction_economies={fid: fe.copy() for fid, fe in self.faction_economies.items()},
            trade_agreements=[TradeAgreement(
                exporter_faction=t.exporter_faction, importer_faction=t.importer_faction,
                resource=t.resource, amount_per_step=t.amount_per_step,
                price_ratio=t.price_ratio, remaining_steps=t.remaining_steps,
                route_clusters=t.route_clusters, is_blocked=t.is_blocked,
            ) for t in self.trade_agreements],
            lend_lease_packages=[LendLeasePackage(
                donor_faction=ll.donor_faction, recipient_faction=ll.recipient_faction,
                resource_amounts=dict(ll.resource_amounts),
                equipment_bonus=ll.equipment_bonus, remaining_steps=ll.remaining_steps,
                alliance_threshold=ll.alliance_threshold, donor_gdp_cost=ll.donor_gdp_cost,
                political_cost=ll.political_cost,
            ) for ll in self.lend_lease_packages],
            market_prices=self.market_prices.copy(),
            step=self.step,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Backward-compatible aliases                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #
# Old code may reference these names.

StrategicResource = Resource
ClusterWarEconomy = ClusterEconomy
FactionWarEconomy = FactionEconomy
FactoryState = None  # removed — use sector_capacity instead
FactoryType = None   # removed
