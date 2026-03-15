"""
budget_system.py — Budget Distribution + Corruption system.

═══════════════════════════════════════════════════════════════════════════════
BUDGET DISTRIBUTION (% of GDP across 7 categories)

  Every faction must distribute 100% of their GDP revenue across 7 buckets.
  The allocation vector B[7] sums to 1.0:

    0  MILITARY       — Army, navy, air force operations + equipment maintenance
    1  PRODUCTION     — Factory output boost, industrial investment
    2  RESEARCH       — Tech research speed multiplier
    3  WELFARE        — Civilian goods, food, housing → prole satisfaction + morale
    4  POLICE         — Thought Police, internal security → suppresses resistance
    5  INFRASTRUCTURE — Roads, rail, ports → logistics capacity + trade
    6  DEBT_SERVICE   — Pay down war debt → reduces interest + inflation

  ANTI-EXPLOITATION MECHANICS:
    - Minimum 5% per category (can't zero anything out)
    - Military < 60% (above that, civilian economy collapses)
    - Welfare < 10% triggers starvation + unrest
    - Research < 5% means NO research progress
    - Police < 5% and resistance grows unchecked
    - Diminishing returns: each % above 30% in any category gives less benefit
    - Budget changes limited to ±15% per turn (can't wildly swing)

═══════════════════════════════════════════════════════════════════════════════
CORRUPTION SYSTEM

  Corruption = bureaucratic leakage. A fraction of every budget category
  is lost to embezzlement, waste, and incompetence.

  Base corruption rate: 8% (Oceania) / 12% (Eurasia — larger bureaucracy)

  Corruption GROWS when:
    - High military spending (procurement fraud)
    - High police spending (Thought Police extortion)
    - Low welfare (desperate officials skim more)
    - War duration (longer war = more entrenched corruption)

  Corruption SHRINKS when:
    - ANTI_CORRUPTION action taken (-3% per action, but costs political capital)
    - High welfare (satisfied population reports fraud)
    - Research in ELECTRONICS (better auditing/tracking)

  Effects of corruption:
    - Effective budget = allocated × (1 - corruption_rate)
    - GDP growth slowed by corruption²
    - Military equipment quality reduced
    - Factory efficiency reduced
    - Research speed reduced
    - Popular trust in government drops

  Anti-exploitation: you CANNOT just spam ANTI_CORRUPTION because:
    - Each action costs 2% GDP (investigation expenses)
    - Diminishing returns (harder to find remaining corruption)
    - Political cost (purges destabilize the bureaucracy)
    - Corruption regrows naturally (+1% per turn from war profiteering)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Budget Categories                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class BudgetCategory(Enum):
    MILITARY       = 0
    PRODUCTION     = 1
    RESEARCH       = 2
    WELFARE        = 3
    POLICE         = 4
    INFRASTRUCTURE = 5
    DEBT_SERVICE   = 6

N_BUDGET = 7

BUDGET_NAMES = {
    BudgetCategory.MILITARY:       "Military",
    BudgetCategory.PRODUCTION:     "Production",
    BudgetCategory.RESEARCH:       "Research",
    BudgetCategory.WELFARE:        "Welfare",
    BudgetCategory.POLICE:         "Police/Security",
    BudgetCategory.INFRASTRUCTURE: "Infrastructure",
    BudgetCategory.DEBT_SERVICE:   "Debt Service",
}

# Minimum allocation per category (anti-exploitation)
MIN_ALLOCATION = {
    BudgetCategory.MILITARY:       0.05,
    BudgetCategory.PRODUCTION:     0.05,
    BudgetCategory.RESEARCH:       0.03,
    BudgetCategory.WELFARE:        0.05,
    BudgetCategory.POLICE:         0.03,
    BudgetCategory.INFRASTRUCTURE: 0.03,
    BudgetCategory.DEBT_SERVICE:   0.01,
}

MAX_MILITARY = 0.60  # above this, civilian economy collapses
MAX_CHANGE_PER_TURN = 0.15  # max ±15% shift per category per turn


# ═══════════════════════════════════════════════════════════════════════════ #
# Diminishing Returns                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def _diminishing_return(allocation: float, threshold: float = 0.30) -> float:
    """Apply diminishing returns above threshold.
    Below threshold: linear (1:1).
    Above threshold: sqrt curve (increasingly less effective).
    """
    if allocation <= threshold:
        return allocation
    excess = allocation - threshold
    return threshold + np.sqrt(excess) * 0.3


# ═══════════════════════════════════════════════════════════════════════════ #
# Corruption State                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class CorruptionState:
    """Corruption tracking for one faction.

    ASYMPTOTIC DESIGN: corruption can NEVER reach 0%. It always regrows.
    Like entropy — you can reduce it, but the universe trends toward disorder.
    Minimum floor = base_rate × 0.4 (even the most efficient state has ~3-5% waste).
    """
    base_rate: float = 0.08           # 8% base (Oceania) / 12% (Eurasia)
    current_rate: float = 0.08        # current effective rate
    total_stolen: float = 0.0         # cumulative GDP lost to corruption
    anti_corruption_fatigue: float = 0.0  # diminishing returns on purges
    purges_this_game: int = 0
    purges_this_turn: int = 0         # cap: max 1 purge per turn

    @property
    def floor(self) -> float:
        """Minimum corruption — efficient bureaucracy never reaches 0%."""
        return self.base_rate * 0.4  # Oceania floor = 3.2%, Eurasia floor = 4.8%

    @property
    def effective_rate(self) -> float:
        return min(0.50, max(self.floor, self.current_rate))

    @property
    def efficiency_mult(self) -> float:
        """How much of the budget actually reaches its target."""
        return 1.0 - self.effective_rate


# ═══════════════════════════════════════════════════════════════════════════ #
# Bureaucracy — Paperwork Delay System                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class PendingOrder:
    """An action stuck in the bureaucratic pipeline."""
    action_type: str
    params: Dict[str, Any]
    faction_id: int
    turns_remaining: int          # turns until it processes
    original_delay: int           # how long the delay was set to
    filed_turn: int = 0           # when it was submitted

    @property
    def is_ready(self) -> bool:
        return self.turns_remaining <= 0


@dataclass
class BureaucracyState:
    """Bureaucratic delay tracking for one faction.

    CORE PRINCIPLE: "Efficient bureaucracy simply never exists.
    Change is the only thing that is unchangeable."

    Every major action goes through a paperwork pipeline:
      - Base delay depends on action category
      - Corruption adds random extra delays (lost paperwork)
      - Changing your mind (canceling/reissuing) costs MORE time
      - INFRASTRUCTURE spending reduces delays
      - Totalitarian regimes (Oceania) = faster but more rigid
      - Communist regimes (Eurasia) = more bureaucratic layers

    Quick actions (1 turn): NOOP, NAVAL_MISSION, CAS_SUPPORT
    Medium actions (2-3 turns): BUILD_SHIP, BUILD_SQUADRON, CONSCRIPT
    Slow actions (3-5 turns): BUILD_FACTORY, PLAN_INVASION, SET_BUDGET
    Very slow (4-6 turns): RESEARCH (bureaucratic approval for funding)
    """
    pending_orders: List[PendingOrder] = field(default_factory=list)
    regime_speed: float = 1.0        # Oceania=1.1 (totalitarian efficiency), Eurasia=0.9 (layers)
    infra_speed: float = 1.0         # from INFRASTRUCTURE budget allocation
    changes_this_game: int = 0       # budget changes penalize further changes
    last_budget_change_turn: int = -10  # when budget was last changed

    @property
    def active_orders(self) -> List[PendingOrder]:
        return [o for o in self.pending_orders if not o.is_ready]

    @property
    def ready_orders(self) -> List[PendingOrder]:
        return [o for o in self.pending_orders if o.is_ready]

    @property
    def queue_depth(self) -> int:
        return len(self.active_orders)


# Base delays per action type (turns of bureaucratic processing)
ACTION_DELAYS = {
    # Instant (no paperwork — field commanders act)
    "noop": 0, "naval_mission": 0, "cas_support": 0, "anti_ship_strike": 0,
    "shore_bombard": 0, "lay_mines": 0, "sweep_mines": 0,
    # Fast (1 turn — standing orders, routine)
    "strategic_bomb": 1, "train_military": 1, "plant_spy": 1,
    "code_break": 1, "counter_intel": 1, "deception": 1, "change_codes": 1,
    "mobilize_reserves": 1, "repair_factory": 1, "inspire": 0,
    # Medium (2 turns — procurement, logistics)
    "build_ship": 2, "build_squadron": 2, "conscript": 2,
    "set_manufacturing": 1, "war_bonds": 2, "sanctions": 2,
    "support_blf": 2, "declare_war_blf": 2,
    # Slow (3 turns — major policy decisions)
    "build_factory": 3, "set_budget": 3, "anti_corruption": 3,
    "plan_invasion": 2, "contact_eurasia": 2,
    # Very slow (4 turns — requires committee approval)
    "research": 4,
}

# Max pending orders per faction (bureaucratic capacity)
MAX_PENDING = 8


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Budget                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionBudget:
    """Complete budget state for one faction."""
    faction_id: int

    # Budget allocation vector (sums to 1.0)
    allocation: np.ndarray = field(
        default_factory=lambda: np.array([0.30, 0.20, 0.10, 0.15, 0.10, 0.10, 0.05], dtype=np.float64))

    # Effective spending (after corruption)
    effective: np.ndarray = field(
        default_factory=lambda: np.zeros(N_BUDGET, dtype=np.float64))

    # Corruption
    corruption: CorruptionState = field(default_factory=CorruptionState)

    # Bureaucracy
    bureaucracy: BureaucracyState = field(default_factory=BureaucracyState)

    # GDP input (set each turn from economy)
    gdp_revenue: float = 100.0

    # Effects cache (recalculated each turn)
    military_mult: float = 1.0
    production_mult: float = 1.0
    research_mult: float = 1.0
    welfare_satisfaction: float = 0.5
    police_suppression: float = 0.3
    infra_logistics: float = 0.5
    debt_reduction: float = 0.0

    def get_allocation(self, cat: BudgetCategory) -> float:
        return float(self.allocation[cat.value])

    def set_allocation(self, cat: BudgetCategory, value: float):
        self.allocation[cat.value] = value


# ═══════════════════════════════════════════════════════════════════════════ #
# Governance World                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class GovernanceWorld:
    """Budget + corruption state for all factions."""
    factions: Dict[int, FactionBudget] = field(default_factory=dict)
    turn: int = 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Budget Step                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_governance(
    world: GovernanceWorld,
    faction_gdps: Dict[int, float],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[GovernanceWorld, Dict[str, Any]]:
    """Advance budget + corruption one turn."""
    feedback: Dict[str, Any] = {}

    for fid, fb in world.factions.items():
        gdp = faction_gdps.get(fid, 100.0)
        fb.gdp_revenue = gdp
        corr = fb.corruption

        # ── 1. Corruption evolution (unpredictable, multi-factor) ────── #
        # Reset per-turn purge counter
        corr.purges_this_turn = 0

        # Natural growth DIMINISHES as corruption rises (saturation)
        # At 10% corruption: grows +0.8%/turn. At 40%: grows only +0.2%/turn
        saturation = 1.0 - (corr.current_rate / 0.50)  # 1.0 at 0%, 0.0 at 50%
        base_growth = 0.008 * max(0.2, saturation) * dt

        # Random factor: corruption growth is UNPREDICTABLE
        # Economy health, bureaucracy efficiency, random events
        random_factor = rng.uniform(0.3, 1.8)  # wild swings
        base_growth *= random_factor

        corr.current_rate = min(0.50, corr.current_rate + base_growth)

        # High military spending breeds procurement fraud
        mil_alloc = fb.get_allocation(BudgetCategory.MILITARY)
        if mil_alloc > 0.35:
            corr.current_rate += (mil_alloc - 0.35) * 0.03 * rng.uniform(0.5, 1.5) * dt

        # High police spending breeds extortion
        pol_alloc = fb.get_allocation(BudgetCategory.POLICE)
        if pol_alloc > 0.15:
            corr.current_rate += (pol_alloc - 0.15) * 0.02 * dt

        # Low welfare = desperate officials skim more
        wel_alloc = fb.get_allocation(BudgetCategory.WELFARE)
        if wel_alloc < 0.10:
            corr.current_rate += (0.10 - wel_alloc) * 0.03 * dt

        # Good welfare = population reports fraud (stronger effect)
        if wel_alloc > 0.20:
            corr.current_rate = max(corr.floor, corr.current_rate - 0.005 * dt)

        # Infrastructure investment reduces bureaucratic waste
        infra_alloc = fb.get_allocation(BudgetCategory.INFRASTRUCTURE)
        if infra_alloc > 0.10:
            corr.current_rate = max(corr.floor, corr.current_rate - (infra_alloc - 0.10) * 0.02 * dt)

        # Enforce asymptotic floor — corruption NEVER reaches 0%
        corr.current_rate = max(corr.floor, min(0.50, corr.current_rate))

        # Anti-corruption fatigue decays slowly
        corr.anti_corruption_fatigue = max(0, corr.anti_corruption_fatigue - 0.02 * dt)

        # Track stolen GDP
        stolen = gdp * corr.effective_rate
        corr.total_stolen += stolen

        # ── 1b. Bureaucracy pipeline ──────────────────────────────────── #
        bur = fb.bureaucracy
        # Infrastructure budget → bureaucracy speed
        infra_alloc = fb.get_allocation(BudgetCategory.INFRASTRUCTURE)
        bur.infra_speed = 0.7 + infra_alloc * 2.0  # 0.7 at 0%, 1.3 at 30%

        # Process pending orders
        ready_actions = []
        for order in bur.pending_orders:
            if order.is_ready:
                ready_actions.append(order)
            else:
                # Tick down delay
                speed = bur.regime_speed * bur.infra_speed
                # Corruption adds random stalls (lost paperwork)
                if rng.random() < corr.effective_rate * 0.3:
                    speed *= 0.5  # paperwork "lost", half speed this turn
                order.turns_remaining -= max(1, int(speed))

        # Remove processed orders
        bur.pending_orders = [o for o in bur.pending_orders if not o.is_ready]

        # ── 2. Calculate effective spending ───────────────────────────── #
        eff_mult = corr.efficiency_mult
        for i in range(N_BUDGET):
            raw = fb.allocation[i] * gdp
            diminished = _diminishing_return(fb.allocation[i]) * gdp
            fb.effective[i] = diminished * eff_mult

        # ── 3. Compute effect multipliers ─────────────────────────────── #
        total_eff = max(fb.effective.sum(), 1.0)

        # Military: combat effectiveness, equipment quality
        fb.military_mult = 0.5 + fb.effective[BudgetCategory.MILITARY.value] / total_eff * 1.5

        # Production: factory output boost
        fb.production_mult = 0.6 + fb.effective[BudgetCategory.PRODUCTION.value] / total_eff * 1.2

        # Research: tech speed multiplier
        fb.research_mult = 0.3 + fb.effective[BudgetCategory.RESEARCH.value] / total_eff * 2.0

        # Welfare: civilian satisfaction
        fb.welfare_satisfaction = min(1.0, fb.effective[BudgetCategory.WELFARE.value] / max(gdp * 0.15, 1.0))

        # Police: resistance suppression
        fb.police_suppression = min(1.0, fb.effective[BudgetCategory.POLICE.value] / max(gdp * 0.10, 1.0))

        # Infrastructure: logistics multiplier
        fb.infra_logistics = 0.5 + fb.effective[BudgetCategory.INFRASTRUCTURE.value] / total_eff * 1.0

        # Debt service: reduces faction debt
        fb.debt_reduction = fb.effective[BudgetCategory.DEBT_SERVICE.value] * 0.5

    world.turn += 1
    return world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Budget Actions                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_budget_action(
    world: GovernanceWorld,
    faction_id: int,
    action: str,
    params: Dict[str, float],
) -> Tuple[GovernanceWorld, str]:
    """Apply a budget action. Returns (world, message)."""
    fb = world.factions.get(faction_id)
    if fb is None:
        return world, "No budget system."

    if action == "SET_BUDGET":
        # params: {"MILITARY": 0.30, "PRODUCTION": 0.20, ...}
        new_alloc = np.copy(fb.allocation)
        for cat_name, value in params.items():
            cat_map = {c.name: c for c in BudgetCategory}
            cat = cat_map.get(cat_name.upper())
            if cat is not None:
                old = new_alloc[cat.value]
                # Enforce max change per turn
                clamped = max(old - MAX_CHANGE_PER_TURN, min(old + MAX_CHANGE_PER_TURN, value))
                # Enforce minimums
                clamped = max(MIN_ALLOCATION[cat], clamped)
                # Enforce military cap
                if cat == BudgetCategory.MILITARY:
                    clamped = min(MAX_MILITARY, clamped)
                new_alloc[cat.value] = clamped

        # Normalize to sum to 1.0
        total = new_alloc.sum()
        if total > 0:
            new_alloc /= total
        else:
            new_alloc = fb.allocation

        # Enforce minimums again after normalization
        for cat in BudgetCategory:
            if new_alloc[cat.value] < MIN_ALLOCATION[cat]:
                new_alloc[cat.value] = MIN_ALLOCATION[cat]
        new_alloc /= new_alloc.sum()

        fb.allocation = new_alloc

        parts = [f"{BUDGET_NAMES[BudgetCategory(i)]}:{new_alloc[i]:.0%}" for i in range(N_BUDGET)]
        return world, f"Budget reallocated: {', '.join(parts)}"

    elif action == "ANTI_CORRUPTION":
        corr = fb.corruption

        # Anti-exploit: MAX 1 PURGE PER TURN
        if corr.purges_this_turn >= 1:
            return world, "Anti-corruption already done this turn. DO NOT repeat — focus on other actions."

        corr.purges_this_turn += 1

        # Cost: 2% of GDP
        cost = fb.gdp_revenue * 0.02
        # Diminishing returns: each purge is less effective
        effectiveness = max(0.003, 0.03 * (1.0 / (1.0 + corr.anti_corruption_fatigue)))
        corr.current_rate = max(corr.floor, corr.current_rate - effectiveness)
        corr.anti_corruption_fatigue += 0.3
        corr.purges_this_game += 1

        # Show futility warning when purges are useless
        futility = ""
        if effectiveness < 0.008:
            futility = " ⚠ DIMINISHING RETURNS — invest in WELFARE or INFRASTRUCTURE instead."

        return world, (
            f"Anti-corruption purge. Corruption: {corr.current_rate:.1%} "
            f"(-{effectiveness:.1%}). Cost: {cost:.0f} GDP. "
            f"Purge #{corr.purges_this_game}.{futility}")

    return world, f"Unknown budget action: {action}"


def queue_action(
    world: GovernanceWorld,
    faction_id: int,
    action_type: str,
    params: Dict[str, Any],
    turn: int,
    rng: np.random.Generator,
) -> Tuple[bool, str, int]:
    """
    Queue an action through the bureaucratic pipeline.

    Returns (is_instant, message, delay_turns).
    If is_instant=True, execute immediately. Otherwise it's queued.
    """
    fb = world.factions.get(faction_id)
    if fb is None:
        return True, "", 0

    base_delay = ACTION_DELAYS.get(action_type, 1)

    if base_delay == 0:
        return True, "", 0  # instant — field command, no paperwork

    bur = fb.bureaucracy
    corr = fb.corruption

    # Calculate actual delay
    delay = base_delay

    # Corruption adds random extra delay (lost paperwork, kickback negotiations)
    if rng.random() < corr.effective_rate:
        delay += rng.integers(1, 3)  # 1-2 extra turns

    # Infrastructure budget reduces delay
    delay = max(1, int(delay / max(bur.infra_speed, 0.5)))

    # Regime speed
    delay = max(1, int(delay / max(bur.regime_speed, 0.5)))

    # Frequent budget changes → bureaucratic confusion penalty
    if action_type == "set_budget":
        turns_since_last = turn - bur.last_budget_change_turn
        if turns_since_last < 5:
            delay += 2  # changing budget too often clogs the system
        bur.last_budget_change_turn = turn
        bur.changes_this_game += 1

    # Queue overflow — too many pending = slower everything
    if bur.queue_depth >= MAX_PENDING:
        return False, "Bureaucratic backlog! Too many pending orders. Wait for some to process.", delay

    # Queue it
    order = PendingOrder(
        action_type=action_type,
        params=params,
        faction_id=faction_id,
        turns_remaining=delay,
        original_delay=delay,
        filed_turn=turn,
    )
    bur.pending_orders.append(order)

    return False, f"Order filed. Processing in {delay} turn(s) (bureaucratic delay).", delay


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_governance(faction_ids: List[int]) -> GovernanceWorld:
    """Initialize budget + corruption + bureaucracy for all factions."""
    factions = {}
    for fid in faction_ids:
        if fid == 0:
            # Oceania (Ingsoc): totalitarian → fast but rigid bureaucracy, lower corruption
            alloc = np.array([0.35, 0.18, 0.08, 0.12, 0.12, 0.10, 0.05], dtype=np.float64)
            corr = CorruptionState(base_rate=0.08, current_rate=0.08)
            bur = BureaucracyState(regime_speed=1.15)  # totalitarian = fewer approvals needed
        else:
            # Eurasia (Neo-Bolshevism): communist → more layers, higher corruption
            alloc = np.array([0.32, 0.22, 0.10, 0.14, 0.08, 0.09, 0.05], dtype=np.float64)
            corr = CorruptionState(base_rate=0.12, current_rate=0.12)
            bur = BureaucracyState(regime_speed=0.85)  # committee approvals slow things down

        factions[fid] = FactionBudget(
            faction_id=fid,
            allocation=alloc,
            corruption=corr,
            bureaucracy=bur,
        )
    return GovernanceWorld(factions=factions)


# ═══════════════════════════════════════════════════════════════════════════ #
# Summary                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def governance_summary(world: GovernanceWorld, faction_id: int) -> str:
    """Generate budget + corruption summary for turn reports."""
    fb = world.factions.get(faction_id)
    if fb is None:
        return "No budget data."

    parts = []

    # Budget allocation (top 3)
    indices = np.argsort(fb.allocation)[::-1]
    top3 = [f"{BUDGET_NAMES[BudgetCategory(i)]}:{fb.allocation[i]:.0%}" for i in indices[:3]]
    parts.append("Budget: " + ", ".join(top3))

    # Corruption
    corr = fb.corruption
    corr_label = "low" if corr.effective_rate < 0.10 else "moderate" if corr.effective_rate < 0.20 else "HIGH" if corr.effective_rate < 0.35 else "CRITICAL"
    parts.append(f"Corruption: {corr.effective_rate:.0%} ({corr_label})")

    # Bureaucracy pipeline
    bur = fb.bureaucracy
    if bur.queue_depth > 0:
        pending_types = [o.action_type for o in bur.active_orders]
        parts.append(f"Pending orders: {bur.queue_depth} ({', '.join(pending_types[:3])})")
    else:
        parts.append("Bureaucracy: clear")

    # Key warnings
    if fb.welfare_satisfaction < 0.3:
        parts.append("⚠ WELFARE CRISIS")
    if fb.research_mult < 0.5:
        parts.append("⚠ Research underfunded")
    if corr.effective_rate > 0.25:
        parts.append("⚠ Corruption HIGH — use ANTI_CORRUPTION")

    # Efficiency
    parts.append(f"Efficiency: {corr.efficiency_mult:.0%}")

    return " | ".join(parts)
