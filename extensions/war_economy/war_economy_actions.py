"""
war_economy_actions.py — Agent actions for the multi-sector war economy.

Actions:
  NOOP                — No economic action
  REALLOCATE_LABOR    — Shift workforce between sectors (constrained by LABOR_SHIFT_MAX)
  TRADE_PROPOSE       — Bilateral trade agreement
  LEND_LEASE          — Asymmetric military aid
  IMPOSE_SANCTIONS    — Block enemy trade
  LIFT_SANCTIONS      — Diplomatic gesture
  ISSUE_WAR_BONDS     — Short-term production burst + debt
  BLOCKADE            — Block trade routes through enemy cluster
  SET_MFG_PRIORITY    — Shift manufacturing between consumer vs military goods
  MOBILIZE_ECONOMY    — Faction-wide war mobilization level

Observation helper builds a flat vector from the multi-sector state.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .war_economy_state import (
    EconSector,
    Resource,
    TradeAgreement,
    LendLeasePackage,
    WarEconomyWorld,
    N_SECTORS,
    N_RESOURCES,
    LABOR_SHIFT_MAX,
    MAX_STOCKPILE,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Action Types                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class WarEconomyAction(Enum):
    NOOP              = 0
    REALLOCATE_LABOR  = 1   # shift workers between sectors
    TRADE_PROPOSE     = 2
    LEND_LEASE        = 3
    IMPOSE_SANCTIONS  = 4
    LIFT_SANCTIONS    = 5
    ISSUE_WAR_BONDS   = 6
    BLOCKADE          = 7
    SET_MFG_PRIORITY  = 8   # consumer vs military manufacturing split
    MOBILIZE_ECONOMY  = 9


N_ECON_ACTIONS = len(WarEconomyAction)


# ═══════════════════════════════════════════════════════════════════════════ #
# Action Application                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_war_economy_action(
    war_world: WarEconomyWorld,
    faction_id: int,
    action_type: int,
    target_faction: int,
    cluster_id: int,
    resource_idx: int,
    intensity: float,
    sector_from: int,
    sector_to: int,
    cluster_owners: Dict[int, int],
    alliance: Optional[NDArray[np.float64]] = None,
) -> Tuple[WarEconomyWorld, float]:
    """
    Apply a war economy action. Returns (updated_world, reward_modifier).
    """
    try:
        action = WarEconomyAction(action_type)
    except ValueError:
        return war_world, 0.0

    if action == WarEconomyAction.NOOP:
        return war_world, 0.0

    elif action == WarEconomyAction.REALLOCATE_LABOR:
        return _reallocate_labor(war_world, faction_id, cluster_id, sector_from,
                                  sector_to, intensity, cluster_owners)

    elif action == WarEconomyAction.TRADE_PROPOSE:
        return _trade_propose(war_world, faction_id, target_faction, resource_idx,
                               intensity, cluster_owners, alliance)

    elif action == WarEconomyAction.LEND_LEASE:
        return _lend_lease(war_world, faction_id, target_faction, resource_idx,
                            intensity, cluster_owners, alliance)

    elif action == WarEconomyAction.IMPOSE_SANCTIONS:
        return _impose_sanctions(war_world, faction_id, target_faction, intensity)

    elif action == WarEconomyAction.LIFT_SANCTIONS:
        return _lift_sanctions(war_world, faction_id, target_faction)

    elif action == WarEconomyAction.ISSUE_WAR_BONDS:
        return _issue_war_bonds(war_world, faction_id, cluster_id, cluster_owners)

    elif action == WarEconomyAction.BLOCKADE:
        return _blockade(war_world, faction_id, cluster_id, cluster_owners)

    elif action == WarEconomyAction.SET_MFG_PRIORITY:
        return _set_mfg_priority(war_world, faction_id, intensity)

    elif action == WarEconomyAction.MOBILIZE_ECONOMY:
        return _mobilize(war_world, faction_id, intensity)

    return war_world, 0.0


# ═══════════════════════════════════════════════════════════════════════════ #
# Individual Actions                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def _reallocate_labor(
    ww: WarEconomyWorld, faction_id: int, cluster_id: int,
    sector_from: int, sector_to: int, intensity: float,
    cluster_owners: Dict[int, int],
) -> Tuple[WarEconomyWorld, float]:
    """Shift labor between sectors. Constrained by LABOR_SHIFT_MAX per step."""
    if cluster_owners.get(cluster_id) != faction_id:
        return ww, -0.1
    if cluster_id >= len(ww.cluster_economies):
        return ww, -0.1
    s_from = min(sector_from, N_SECTORS - 1)
    s_to = min(sector_to, N_SECTORS - 1)
    if s_from == s_to:
        return ww, -0.05

    ce = ww.cluster_economies[cluster_id]
    shift = LABOR_SHIFT_MAX * intensity
    actual_shift = min(shift, ce.sector_labor[s_from])

    ce.sector_labor[s_from] -= actual_shift
    ce.sector_labor[s_to] += actual_shift
    # Re-normalize to sum ≤ 1.0
    total = ce.sector_labor.sum()
    if total > 1.0:
        ce.sector_labor /= total

    return ww, 0.1 * actual_shift


def _trade_propose(
    ww: WarEconomyWorld, faction_id: int, target_faction: int,
    resource_idx: int, intensity: float,
    cluster_owners: Dict[int, int],
    alliance: Optional[NDArray[np.float64]],
) -> Tuple[WarEconomyWorld, float]:
    if faction_id == target_faction:
        return ww, -0.1

    # Check existing trade for same resource
    for t in ww.trade_agreements:
        if (t.is_active and t.exporter_faction == faction_id
                and t.importer_faction == target_faction
                and t.resource.value == resource_idx):
            return ww, -0.05

    r_idx = min(resource_idx, N_RESOURCES - 1)
    resource = Resource(r_idx)
    amount = 1.0 + 4.0 * intensity

    f_clusters = [cid for cid, fid in cluster_owners.items() if fid == faction_id]
    t_clusters = [cid for cid, fid in cluster_owners.items() if fid == target_faction]
    route = tuple(f_clusters[:1] + t_clusters[:1]) if f_clusters and t_clusters else ()

    ww.trade_agreements.append(TradeAgreement(
        exporter_faction=faction_id,
        importer_faction=target_faction,
        resource=resource,
        amount_per_step=amount,
        price_ratio=0.3 + 0.4 * intensity,
        remaining_steps=int(50 + 100 * intensity),
        route_clusters=route,
    ))
    return ww, 0.5


def _lend_lease(
    ww: WarEconomyWorld, faction_id: int, target_faction: int,
    resource_idx: int, intensity: float,
    cluster_owners: Dict[int, int],
    alliance: Optional[NDArray[np.float64]],
) -> Tuple[WarEconomyWorld, float]:
    if faction_id == target_faction:
        return ww, -0.1

    r_idx = min(resource_idx, N_RESOURCES - 1)
    resource = Resource(r_idx)

    ww.lend_lease_packages.append(LendLeasePackage(
        donor_faction=faction_id,
        recipient_faction=target_faction,
        resource_amounts={resource: 2.0 + 3.0 * intensity},
        equipment_bonus=0.05 * intensity,
        remaining_steps=int(80 + 120 * intensity),
        donor_gdp_cost=0.01 + 0.02 * intensity,
        political_cost=0.005 + 0.01 * intensity,
    ))
    return ww, 0.3


def _impose_sanctions(
    ww: WarEconomyWorld, faction_id: int, target_faction: int, intensity: float,
) -> Tuple[WarEconomyWorld, float]:
    if faction_id == target_faction:
        return ww, -0.1
    if faction_id in ww.faction_economies:
        ww.faction_economies[faction_id].sanctions_imposed[target_faction] = (
            0.3 + 0.7 * intensity)
    for trade in ww.trade_agreements:
        if ({trade.exporter_faction, trade.importer_faction} ==
                {faction_id, target_faction}):
            trade.is_blocked = True
    return ww, 0.2


def _lift_sanctions(
    ww: WarEconomyWorld, faction_id: int, target_faction: int,
) -> Tuple[WarEconomyWorld, float]:
    if faction_id in ww.faction_economies:
        ww.faction_economies[faction_id].sanctions_imposed.pop(target_faction, None)
    for trade in ww.trade_agreements:
        if ({trade.exporter_faction, trade.importer_faction} ==
                {faction_id, target_faction}):
            trade.is_blocked = False
    return ww, 0.1


def _issue_war_bonds(
    ww: WarEconomyWorld, faction_id: int, cluster_id: int,
    cluster_owners: Dict[int, int],
) -> Tuple[WarEconomyWorld, float]:
    if cluster_owners.get(cluster_id) != faction_id:
        return ww, -0.2
    if cluster_id >= len(ww.cluster_economies):
        return ww, -0.1
    ce = ww.cluster_economies[cluster_id]
    if ce.war_bond_active:
        return ww, -0.1
    if faction_id in ww.faction_economies and ww.faction_economies[faction_id].fiscal_debt > 0.8:
        return ww, -0.5
    ce.war_bond_active = True
    ce.war_bond_remaining = 30
    return ww, 0.4


def _blockade(
    ww: WarEconomyWorld, faction_id: int, cluster_id: int,
    cluster_owners: Dict[int, int],
) -> Tuple[WarEconomyWorld, float]:
    if cluster_owners.get(cluster_id) == faction_id:
        return ww, -0.2
    if cluster_id >= len(ww.cluster_economies):
        return ww, -0.1
    ww.cluster_economies[cluster_id].blockaded = True
    for trade in ww.trade_agreements:
        if cluster_id in trade.route_clusters:
            trade.is_blocked = True
    return ww, 0.3


def _set_mfg_priority(
    ww: WarEconomyWorld, faction_id: int, intensity: float,
) -> Tuple[WarEconomyWorld, float]:
    """Set manufacturing priority: 0=consumer, 1=military."""
    if faction_id not in ww.faction_economies:
        return ww, 0.0
    ww.faction_economies[faction_id].manufacturing_priority = max(0.0, min(1.0, intensity))
    return ww, 0.1


def _mobilize(
    ww: WarEconomyWorld, faction_id: int, intensity: float,
) -> Tuple[WarEconomyWorld, float]:
    if faction_id not in ww.faction_economies:
        return ww, 0.0
    ww.faction_economies[faction_id].war_mobilization = max(0.0, min(1.0, intensity))
    return ww, 0.1 * abs(intensity - ww.faction_economies[faction_id].war_mobilization)


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation Helper                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def war_economy_obs(
    ww: WarEconomyWorld,
    faction_id: int,
    cluster_owners: Dict[int, int],
    max_clusters: int = 12,
) -> NDArray[np.float32]:
    """
    Build a flat observation vector for the war economy.

    Per cluster (N_PER_CLUSTER floats × max_clusters):
      - 12 resource stockpile ratios [0, 1]
      - 7 sector output levels (normalized)
      - 7 sector labor fractions [0, 1]
      - infrastructure [0, 1]
      - war_damage [0, 1]
      - sanctions [0, 1]
      - blockade {0, 1}
      = 30 floats per cluster

    Faction-level (N_FACTION floats):
      - war_mobilization, fiscal_debt, inflation, trade_balance,
        war_fatigue, manufacturing_priority
      - 12 market prices
      = 18 floats

    Total: 30 × max_clusters + 18
    """
    n_per = 30
    n_fac = 18
    obs = np.zeros(n_per * max_clusters + n_fac, dtype=np.float32)

    for i, ce in enumerate(ww.cluster_economies):
        if i >= max_clusters:
            break
        o = i * n_per
        # Resource stockpiles (12)
        obs[o:o + N_RESOURCES] = np.clip(ce.resource_stockpile / MAX_STOCKPILE, 0.0, 1.0)
        # Sector outputs (7, normalized by capacity)
        eff_cap = ce.effective_capacity
        safe_cap = np.where(eff_cap > 0.01, eff_cap, 1.0)
        obs[o + 12:o + 12 + N_SECTORS] = np.clip(ce.sector_output / safe_cap, 0.0, 2.0) / 2.0
        # Sector labor (7)
        obs[o + 19:o + 19 + N_SECTORS] = ce.sector_labor
        # Infrastructure, war_damage, sanctions, blockade
        obs[o + 26] = ce.infrastructure
        obs[o + 27] = ce.war_damage
        obs[o + 28] = ce.sanctions_level
        obs[o + 29] = 1.0 if ce.blockaded else 0.0

    # Faction-level
    fo = n_per * max_clusters
    if faction_id in ww.faction_economies:
        fe = ww.faction_economies[faction_id]
        obs[fo + 0] = fe.war_mobilization
        obs[fo + 1] = min(fe.fiscal_debt, 2.0) / 2.0
        obs[fo + 2] = fe.inflation
        obs[fo + 3] = np.clip(fe.trade_balance / 200.0, -1.0, 1.0)
        obs[fo + 4] = fe.war_fatigue
        obs[fo + 5] = fe.manufacturing_priority
    # Market prices (12)
    obs[fo + 6:fo + 6 + N_RESOURCES] = np.clip(ww.market_prices / 5.0, 0.0, 1.0)

    return obs


def war_economy_obs_size(max_clusters: int = 12) -> int:
    return 30 * max_clusters + 18
