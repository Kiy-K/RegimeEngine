"""
war_economy_dynamics.py — Leontief input-output production engine.

Per-step pipeline:
  1. Leontief production — each sector produces based on min(inputs/required)
  2. Diminishing returns — output = capacity × (1 - exp(-k × labor/capacity))
  3. Tier-ordered processing — Tier 1 first, then Tier 2, then Tier 3
  4. Resource consumption — population + military upkeep
  5. Trade execution — bilateral resource transfers
  6. Lend-lease delivery — asymmetric aid
  7. Capital depreciation — capacity decays without Construction maintenance
  8. Infrastructure decay — decays without Construction investment
  9. War damage effects — combat zones lose capacity + infrastructure
  10. Market price discovery — supply/demand clearing
  11. Cross-system feedback → economy.py, military, population
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .war_economy_state import (
    EconSector,
    Resource,
    ClusterEconomy,
    FactionEconomy,
    WarEconomyWorld,
    N_SECTORS,
    N_RESOURCES,
    SECTOR_INPUTS,
    SECTOR_OUTPUTS,
    TERRAIN_ENDOWMENT,
    DEFAULT_ENDOWMENT,
    SPOILAGE,
    MAX_STOCKPILE,
    CAPITAL_DEPRECIATION,
    INFRASTRUCTURE_DECAY,
    DIMINISHING_RETURNS_K,
    WAR_FATIGUE_RATE,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Sector processing order                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

_PRODUCTION_ORDER = [
    EconSector.MINING.value,           # Tier 1 extractors
    EconSector.AGRICULTURE.value,
    EconSector.ENERGY.value,           # Tier 2 processors
    EconSector.HEAVY_INDUSTRY.value,
    EconSector.MANUFACTURING.value,    # Tier 3 finished goods
    EconSector.CONSTRUCTION.value,     # Infrastructure
    EconSector.SERVICES.value,         # Productivity
]

# Population consumption per resource per step (scaled by population level)
BASE_POP_CONSUMPTION = np.zeros(N_RESOURCES, dtype=np.float64)
BASE_POP_CONSUMPTION[Resource.COAL.value] = 0.1
BASE_POP_CONSUMPTION[Resource.FUEL.value] = 0.3
BASE_POP_CONSUMPTION[Resource.CHEMICALS.value] = 0.05
BASE_POP_CONSUMPTION[Resource.PROCESSED_FOOD.value] = 0.8
BASE_POP_CONSUMPTION[Resource.CONSUMER_GOODS.value] = 0.3

# Military consumption per resource per step (scaled by military presence)
BASE_MIL_CONSUMPTION = np.zeros(N_RESOURCES, dtype=np.float64)
BASE_MIL_CONSUMPTION[Resource.FUEL.value] = 1.2
BASE_MIL_CONSUMPTION[Resource.STEEL.value] = 0.2
BASE_MIL_CONSUMPTION[Resource.CHEMICALS.value] = 0.1
BASE_MIL_CONSUMPTION[Resource.PROCESSED_FOOD.value] = 0.5
BASE_MIL_CONSUMPTION[Resource.MILITARY_EQUIPMENT.value] = 0.3
BASE_MIL_CONSUMPTION[Resource.AMMUNITION.value] = 0.8


# ═══════════════════════════════════════════════════════════════════════════ #
# Core Leontief Production Function                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def leontief_produce(
    sector_idx: int,
    stockpile: NDArray[np.float64],    # (N_RESOURCES,) — mutable, consumed in-place
    capacity: float,                    # effective sector capacity
    labor_frac: float,                  # fraction of workforce in this sector
    sanctions: float,                   # [0, 1] sanctions level
    war_fatigue: float,                 # [0, 1] accumulated fatigue
    war_bond_mult: float,              # 1.0 normally, 1.5 during war bonds
    mfg_priority: float,               # manufacturing military vs civilian split
) -> Tuple[float, NDArray[np.float64]]:
    """
    Leontief production for a single sector in a single cluster.

    Returns:
        (actual_output, produced_resources)

    The function CONSUMES inputs from stockpile in-place and returns
    the produced resources (caller adds to stockpile).

    Anti-exploitation mechanisms:
      - Bottleneck: output = min(available_input / required_input) across ALL inputs
      - Diminishing returns: labor_eff = 1 - exp(-K × labor / capacity)
      - Sanctions reduce output multiplicatively
      - War fatigue reduces output multiplicatively
    """
    if capacity <= 0.01 or labor_frac <= 0.001:
        return 0.0, np.zeros(N_RESOURCES, dtype=np.float64)

    inputs_needed = SECTOR_INPUTS[sector_idx]  # (N_RESOURCES,)

    # ── Leontief bottleneck ──────────────────────────────────────────── #
    # For each required input, compute how many units of output we can make
    bottleneck = 1e6  # effectively unlimited if no inputs needed
    has_requirements = False
    for r in range(N_RESOURCES):
        if inputs_needed[r] > 0.001:
            has_requirements = True
            if stockpile[r] <= 0.0:
                return 0.0, np.zeros(N_RESOURCES, dtype=np.float64)
            ratio = stockpile[r] / inputs_needed[r]
            if ratio < bottleneck:
                bottleneck = ratio

    if not has_requirements:
        bottleneck = 1.0  # no inputs needed (e.g. Services)

    # ── Diminishing returns on labor ─────────────────────────────────── #
    # output = capacity × (1 - exp(-K × labor / capacity))
    # At labor=0: output=0. At labor>>capacity: output→capacity.
    labor_eff = 1.0 - np.exp(-DIMINISHING_RETURNS_K * labor_frac / max(capacity, 0.01))

    # ── Raw output ───────────────────────────────────────────────────── #
    raw_output = capacity * labor_eff * min(bottleneck, capacity)

    # ── Sanctions and war fatigue penalties ───────────────────────────── #
    penalty = (1.0 - 0.6 * sanctions) * (1.0 - war_fatigue) * war_bond_mult
    actual_output = max(0.0, raw_output * penalty)

    # Cap output to prevent runaway
    actual_output = min(actual_output, capacity * 3.0)

    # ── Consume inputs ───────────────────────────────────────────────── #
    for r in range(N_RESOURCES):
        consumed = inputs_needed[r] * actual_output
        stockpile[r] = max(0.0, stockpile[r] - consumed)

    # ── Produce outputs ──────────────────────────────────────────────── #
    outputs = SECTOR_OUTPUTS[sector_idx].copy()

    # Manufacturing: split between consumer and military based on priority
    if sector_idx == EconSector.MANUFACTURING.value:
        mil_p = mfg_priority  # 0=all consumer, 1=all military
        outputs[Resource.CONSUMER_GOODS.value] = 0.4 * (1.0 - mil_p) + 0.1 * mil_p
        outputs[Resource.MILITARY_EQUIPMENT.value] = 0.1 * (1.0 - mil_p) + 0.5 * mil_p
        outputs[Resource.AMMUNITION.value] = 0.1 * (1.0 - mil_p) + 0.4 * mil_p

    produced = outputs * actual_output
    return actual_output, produced


# ═══════════════════════════════════════════════════════════════════════════ #
# Consumption                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_consumption(
    stockpile: NDArray[np.float64],
    population: float,
    military: float,
    is_combat: bool,
    dt: float,
) -> NDArray[np.float64]:
    """
    Compute and apply resource consumption. Modifies stockpile in-place.
    Returns the actual consumption vector.
    """
    pop_demand = BASE_POP_CONSUMPTION * max(population, 0.1) * dt
    mil_demand = BASE_MIL_CONSUMPTION * military * dt
    if is_combat:
        mil_demand[Resource.AMMUNITION.value] *= 2.0
        mil_demand[Resource.PROCESSED_FOOD.value] *= 1.3

    total_demand = pop_demand + mil_demand
    actual = np.minimum(total_demand, stockpile)
    stockpile -= actual
    np.clip(stockpile, 0.0, MAX_STOCKPILE, out=stockpile)
    return actual


# ═══════════════════════════════════════════════════════════════════════════ #
# Capital Depreciation & Infrastructure                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_depreciation(
    ce: ClusterEconomy,
    construction_output: float,
    services_output: float,
    hazard: float,
    dt: float,
) -> None:
    """
    Apply capital depreciation and infrastructure decay. Modifies ce in-place.

    Construction sector output counteracts depreciation.
    Services sector boosts overall productivity (stored as infrastructure bonus).
    Hazard accelerates decay (war damage).
    """
    # Capacity depreciation: all sectors lose capacity each step
    depreciation = CAPITAL_DEPRECIATION * dt * (1.0 + 2.0 * hazard)
    maintenance = 0.02 * construction_output * dt  # Construction repairs capacity
    net_depreciation = max(0.0, depreciation - maintenance)
    ce.sector_capacity *= (1.0 - net_depreciation)
    ce.sector_capacity = np.clip(ce.sector_capacity, 0.01, 20.0)

    # Infrastructure decay: without Construction, infrastructure crumbles
    infra_decay = INFRASTRUCTURE_DECAY * dt * (1.0 + 3.0 * hazard)
    infra_repair = 0.03 * construction_output * dt
    ce.infrastructure = max(0.05, min(1.0,
        ce.infrastructure - infra_decay + infra_repair))

    # Services boost: healthcare + education improve productivity ceiling
    services_boost = 0.005 * services_output * dt
    ce.infrastructure = min(1.0, ce.infrastructure + services_boost)

    # War damage from hazard
    if hazard > 0.3:
        ce.war_damage = min(1.0, ce.war_damage + 0.005 * (hazard - 0.3) * dt)
    else:
        ce.war_damage = max(0.0, ce.war_damage - 0.002 * dt)  # slow repair


# ═══════════════════════════════════════════════════════════════════════════ #
# Trade Execution                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def execute_trades(
    ww: WarEconomyWorld,
    cluster_owners: Dict[int, int],
) -> None:
    """Execute active trade agreements. Modifies ww in-place."""
    for trade in ww.trade_agreements:
        if not trade.is_active:
            continue
        trade.remaining_steps -= 1

        # Check blockade on route
        for cid in trade.route_clusters:
            if cid < len(ww.cluster_economies) and ww.cluster_economies[cid].blockaded:
                trade.is_blocked = True
                break
        if trade.is_blocked:
            continue

        r_idx = trade.resource.value
        amount = trade.amount_per_step

        # Find exporter and importer clusters
        exp_ces = [ce for ce in ww.cluster_economies
                   if cluster_owners.get(ce.cluster_id) == trade.exporter_faction]
        imp_ces = [ce for ce in ww.cluster_economies
                   if cluster_owners.get(ce.cluster_id) == trade.importer_faction]
        if not exp_ces or not imp_ces:
            continue

        # Withdraw from richest exporter cluster (never drain below 20%)
        src = max(exp_ces, key=lambda c: c.resource_stockpile[r_idx])
        available = src.resource_stockpile[r_idx]
        actual = min(amount, available * 0.8)
        if actual < 0.1:
            continue

        src.resource_stockpile[r_idx] -= actual

        # Deliver to neediest importer cluster
        dst = min(imp_ces, key=lambda c: c.resource_stockpile[r_idx])
        dst.resource_stockpile[r_idx] = min(
            dst.resource_stockpile[r_idx] + actual, MAX_STOCKPILE)

        # GDP payment via trade balance
        price = actual * trade.price_ratio * ww.market_prices[r_idx]
        if trade.exporter_faction in ww.faction_economies:
            ww.faction_economies[trade.exporter_faction].trade_balance += price
        if trade.importer_faction in ww.faction_economies:
            ww.faction_economies[trade.importer_faction].trade_balance -= price


# ═══════════════════════════════════════════════════════════════════════════ #
# Lend-Lease Execution                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def execute_lend_lease(
    ww: WarEconomyWorld,
    cluster_owners: Dict[int, int],
    alliance: Optional[NDArray[np.float64]],
) -> None:
    """Execute active lend-lease packages. Modifies ww in-place."""
    for ll in ww.lend_lease_packages:
        if not ll.is_active:
            continue
        ll.remaining_steps -= 1

        recip_ces = [ce for ce in ww.cluster_economies
                     if cluster_owners.get(ce.cluster_id) == ll.recipient_faction]
        if not recip_ces:
            continue

        for resource, amount in ll.resource_amounts.items():
            r_idx = resource.value
            per_cluster = amount / max(len(recip_ces), 1)
            for ce in recip_ces:
                ce.resource_stockpile[r_idx] = min(
                    ce.resource_stockpile[r_idx] + per_cluster, MAX_STOCKPILE)

        if ll.donor_faction in ww.faction_economies:
            ww.faction_economies[ll.donor_faction].lend_lease_given += sum(
                ll.resource_amounts.values())
        if ll.recipient_faction in ww.faction_economies:
            ww.faction_economies[ll.recipient_faction].lend_lease_received += sum(
                ll.resource_amounts.values())


# ═══════════════════════════════════════════════════════════════════════════ #
# Market Prices                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def update_market_prices(ww: WarEconomyWorld) -> None:
    """Update market prices based on global supply/demand. Modifies ww in-place."""
    supply = np.zeros(N_RESOURCES, dtype=np.float64)
    demand = np.zeros(N_RESOURCES, dtype=np.float64)

    for ce in ww.cluster_economies:
        supply += ce.resource_stockpile
        demand += BASE_POP_CONSUMPTION + BASE_MIL_CONSUMPTION * 0.5

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(demand > 0.01, supply / (demand + 1e-8), 10.0)

    scarcity = np.clip(1.0 - ratio / 10.0, -0.5, 3.0)
    target_prices = 1.0 + scarcity
    # Smooth EMA
    ww.market_prices = np.clip(
        0.1 * target_prices + 0.9 * ww.market_prices, 0.1, 5.0)


# ═══════════════════════════════════════════════════════════════════════════ #
# Cross-System Feedback                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_feedback(ce: ClusterEconomy) -> Dict[str, float]:
    """
    Compute feedback signals from war economy to other systems.
    """
    fuel = ce.stockpile_ratio(Resource.FUEL)
    food = ce.stockpile_ratio(Resource.PROCESSED_FOOD)
    ammo = ce.stockpile_ratio(Resource.AMMUNITION)
    steel = ce.stockpile_ratio(Resource.STEEL)
    cg = ce.stockpile_ratio(Resource.CONSUMER_GOODS)
    mil_eq = ce.stockpile_ratio(Resource.MILITARY_EQUIPMENT)

    return {
        "supply_refill_mult": max(0.1, 0.3 + 0.7 * min(fuel, food, ammo)),
        "combat_effectiveness": max(0.2, 0.5 + 0.3 * ammo + 0.2 * fuel),
        "production_efficiency": max(0.1, 0.4 + 0.6 * min(steel, fuel)),
        "gdp_modifier": 0.02 * (food - 0.5) + 0.01 * (steel - 0.5),
        "morale_modifier": 0.05 * (food - 0.3) + 0.03 * (cg - 0.3),
        "military_production": max(0.0, mil_eq * 0.5 + ammo * 0.3),
    }


# ═══════════════════════════════════════════════════════════════════════════ #
# Main Step Function                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_war_economy(
    war_world: WarEconomyWorld,
    cluster_data: NDArray[np.float64],       # (N, 6): [σ, h, r, m, τ, p]
    population: Optional[NDArray[np.float64]],
    cluster_owners: Dict[int, int],
    alliance: Optional[NDArray[np.float64]],
    terrain_types: List[str],
    dt: float = 0.01,
) -> WarEconomyWorld:
    """
    Advance the multi-sector war economy by one step.

    Modifies war_world in-place for performance (call .copy() before if needed).
    """
    ww = war_world
    N = ww.n_clusters

    for i in range(N):
        ce = ww.cluster_economies[i]
        pop = population[i] if population is not None and i < len(population) else 0.5
        mil = cluster_data[i, 3] if i < cluster_data.shape[0] else 0.0
        hazard = cluster_data[i, 1] if i < cluster_data.shape[0] else 0.0
        is_combat = hazard > 0.5

        # Get faction for this cluster
        owner = cluster_owners.get(ce.cluster_id)
        fe = ww.faction_economies.get(owner) if owner is not None else None
        war_fatigue = fe.war_fatigue if fe is not None else 0.0
        mfg_priority = fe.manufacturing_priority if fe is not None else 0.5
        war_bond_mult = 1.5 if ce.war_bond_active else 1.0

        # ── Tier 1: Mining extracts raw materials based on terrain ──── #
        # Add terrain endowment to stockpile (natural extraction)
        terrain = terrain_types[i] if i < len(terrain_types) else "OPEN"
        endowment = TERRAIN_ENDOWMENT.get(terrain, DEFAULT_ENDOWMENT)
        mining_labor = ce.sector_labor[EconSector.MINING.value]
        mining_cap = ce.effective_capacity[EconSector.MINING.value]
        mining_eff = 1.0 - np.exp(-DIMINISHING_RETURNS_K * mining_labor / max(mining_cap, 0.01))
        for r_idx in range(5):  # Tier 1 resources only
            extraction = endowment[r_idx] * mining_eff * mining_cap * dt
            ce.resource_stockpile[r_idx] = min(
                ce.resource_stockpile[r_idx] + extraction, MAX_STOCKPILE)

        # ── Sector production in tier order ──────────────────────────── #
        construction_output = 0.0
        services_output = 0.0

        for s_idx in _PRODUCTION_ORDER:
            cap = ce.effective_capacity[s_idx]
            labor = ce.sector_labor[s_idx]

            output, produced = leontief_produce(
                s_idx, ce.resource_stockpile, cap, labor,
                ce.sanctions_level, war_fatigue, war_bond_mult, mfg_priority,
            )

            # Add produced resources to stockpile
            ce.resource_stockpile += produced * dt
            np.clip(ce.resource_stockpile, 0.0, MAX_STOCKPILE, out=ce.resource_stockpile)

            ce.sector_output[s_idx] = output

            if s_idx == EconSector.CONSTRUCTION.value:
                construction_output = output
            elif s_idx == EconSector.SERVICES.value:
                services_output = output

        # ── Resource consumption ─────────────────────────────────────── #
        compute_consumption(ce.resource_stockpile, pop, mil, is_combat, dt)

        # ── Spoilage ─────────────────────────────────────────────────── #
        ce.resource_stockpile -= SPOILAGE * ce.resource_stockpile * dt
        np.clip(ce.resource_stockpile, 0.0, MAX_STOCKPILE, out=ce.resource_stockpile)

        # ── Capital depreciation + infrastructure ────────────────────── #
        apply_depreciation(ce, construction_output, services_output, hazard, dt)

        # ── War bond countdown ───────────────────────────────────────── #
        if ce.war_bond_active:
            ce.war_bond_remaining -= 1
            if ce.war_bond_remaining <= 0:
                ce.war_bond_active = False
            if fe is not None:
                fe.fiscal_debt = min(2.0, fe.fiscal_debt + 0.005 * dt)

    # ── Faction-level updates ────────────────────────────────────────── #
    for fid, fe in ww.faction_economies.items():
        # War fatigue accumulates during conflict
        faction_clusters = [ce for ce in ww.cluster_economies
                           if cluster_owners.get(ce.cluster_id) == fid]
        if faction_clusters:
            avg_hazard = np.mean([cluster_data[ce.cluster_id, 1]
                                  for ce in faction_clusters
                                  if ce.cluster_id < cluster_data.shape[0]])
            if avg_hazard > 0.3:
                fe.war_fatigue = min(0.5, fe.war_fatigue + WAR_FATIGUE_RATE * dt)
            else:
                fe.war_fatigue = max(0.0, fe.war_fatigue - WAR_FATIGUE_RATE * 0.5 * dt)

        # Inflation from debt + money printing
        fe.inflation = max(0.0, min(0.5,
            fe.inflation + 0.001 * fe.fiscal_debt * dt - 0.0005 * dt))

    # ── Trade and lend-lease ─────────────────────────────────────────── #
    execute_trades(ww, cluster_owners)
    execute_lend_lease(ww, cluster_owners, alliance)

    # ── Market prices ────────────────────────────────────────────────── #
    update_market_prices(ww)

    # ── Cleanup expired agreements ───────────────────────────────────── #
    ww.trade_agreements = [t for t in ww.trade_agreements if t.remaining_steps > 0]
    ww.lend_lease_packages = [ll for ll in ww.lend_lease_packages if ll.remaining_steps > 0]

    ww.step += 1
    return ww


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_war_economy(
    n_clusters: int,
    faction_ids: List[int],
    cluster_owners: Dict[int, int],
    terrain_types: List[str],
    rng: np.random.Generator,
    initial_stockpile: float = 80.0,
) -> WarEconomyWorld:
    """
    Initialize the multi-sector war economy.

    Sector capacities and resource endowments are terrain-derived.
    Labor starts balanced across sectors with slight terrain bias.
    """
    cluster_economies = []

    for i in range(n_clusters):
        terrain = terrain_types[i] if i < len(terrain_types) else "OPEN"
        endowment = TERRAIN_ENDOWMENT.get(terrain, DEFAULT_ENDOWMENT)

        # Sector capacity: terrain-derived + noise
        base_cap = np.full(N_SECTORS, 1.0, dtype=np.float64)
        if terrain in ("URBAN", "FORTIFIED"):
            base_cap[EconSector.HEAVY_INDUSTRY.value] = 2.5
            base_cap[EconSector.MANUFACTURING.value] = 2.0
            base_cap[EconSector.SERVICES.value] = 2.0
            base_cap[EconSector.CONSTRUCTION.value] = 1.5
        elif terrain in ("PLAINS", "OPEN"):
            base_cap[EconSector.AGRICULTURE.value] = 2.5
            base_cap[EconSector.CONSTRUCTION.value] = 1.2
        elif terrain == "MOUNTAINS":
            base_cap[EconSector.MINING.value] = 3.0
            base_cap[EconSector.HEAVY_INDUSTRY.value] = 1.5
        elif terrain in ("FOREST", "JUNGLE"):
            base_cap[EconSector.AGRICULTURE.value] = 1.5
            base_cap[EconSector.MINING.value] = 1.2
        elif terrain in ("DESERT", "MARSH"):
            base_cap[EconSector.ENERGY.value] = 2.5
            base_cap[EconSector.MINING.value] = 1.5

        sector_cap = np.clip(base_cap + rng.uniform(-0.2, 0.2, N_SECTORS), 0.3, 5.0)

        # Labor: start balanced, slight bias toward terrain strength
        labor = np.full(N_SECTORS, 1.0 / N_SECTORS, dtype=np.float64)
        strongest = int(np.argmax(sector_cap))
        labor[strongest] += 0.05
        labor /= labor.sum()  # re-normalize

        # Stockpile: Tier 1 from endowment, Tier 2-3 moderate
        stockpile = np.clip(
            initial_stockpile * np.ones(N_RESOURCES) + rng.uniform(-20, 20, N_RESOURCES),
            5.0, MAX_STOCKPILE)
        # Tier 1 resources boosted by terrain
        for r_idx in range(5):
            stockpile[r_idx] += endowment[r_idx] * 20.0

        stockpile = np.clip(stockpile, 5.0, MAX_STOCKPILE)

        cluster_economies.append(ClusterEconomy(
            cluster_id=i,
            sector_capacity=sector_cap,
            sector_labor=labor,
            sector_output=np.zeros(N_SECTORS, dtype=np.float64),
            resource_stockpile=stockpile,
            terrain_endowment=endowment,
            infrastructure=0.6 + rng.uniform(0.0, 0.2),
        ))

    faction_economies = {
        fid: FactionEconomy(faction_id=fid)
        for fid in faction_ids
    }

    return WarEconomyWorld(
        cluster_economies=cluster_economies,
        faction_economies=faction_economies,
        market_prices=np.ones(N_RESOURCES, dtype=np.float64),
    )
