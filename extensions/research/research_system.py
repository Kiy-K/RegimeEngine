"""
research_system.py — HOI4-style Research & Technology system.

═══════════════════════════════════════════════════════════════════════════════
6 TECH BRANCHES

  INDUSTRY    — Factory efficiency, construction speed, resource extraction
  ELECTRONICS — Radar, code-breaking, communications, computing
  NAVAL       — Ship design, submarine tech, torpedoes, sonar
  AIR         — Aircraft design, engines, bombing accuracy, radar intercept
  LAND        — Infantry equipment, armor, artillery, fortifications
  DOCTRINE    — Military doctrine, combined arms, logistics, special forces

Each branch has 5 tiers (0-4). Higher tiers unlock better bonuses.
Research takes turns proportional to tier. Only 1 project active per branch.

═══════════════════════════════════════════════════════════════════════════════
HOW IT WORKS

  1. Faction chooses: RESEARCH branch (e.g., RESEARCH NAVAL)
  2. System starts the next available tier in that branch
  3. Research progresses each turn based on:
     - Base speed (affected by ELECTRONICS level)
     - Number of researchers (from Outer Party technicians)
     - Factory support (CIVIL_FACTORY output helps)
  4. When complete: permanent bonus applied to faction

═══════════════════════════════════════════════════════════════════════════════
BONUSES PER TIER

  INDUSTRY:
    T1: +10% factory output         T2: +10% construction speed
    T3: +15% resource extraction    T4: +20% factory output (total +30%)

  ELECTRONICS:
    T1: +15% radar range            T2: +20% code-breaking speed
    T3: +10% all production         T4: +25% intel quality

  NAVAL:
    T1: +10% ship combat            T2: +15% submarine stealth
    T3: Unlock HEAVY_CRUISER build  T4: +20% torpedo damage

  AIR:
    T1: +10% fighter effectiveness  T2: +15% bombing accuracy
    T3: +10% aircraft speed         T4: Unlock JET_FIGHTER

  LAND:
    T1: +10% infantry combat        T2: +15% artillery range
    T3: +10% fortification strength T4: +20% armor effectiveness

  DOCTRINE:
    T1: +10% org recovery           T2: +15% supply efficiency
    T3: +10% planning speed         T4: +20% combined arms bonus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Tech Branches                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TechBranch(Enum):
    INDUSTRY    = 0
    ELECTRONICS = 1
    NAVAL       = 2
    AIR         = 3
    LAND        = 4
    DOCTRINE    = 5

N_BRANCHES = 6
MAX_TIER = 5

# Research time per tier (turns) — increasing cost, diminishing returns
TIER_RESEARCH_TIME = [8, 12, 18, 25, 35]  # T1=8, T2=12, T3=18, T4=25, T5=35

BRANCH_NAMES = {
    TechBranch.INDUSTRY:    "Industry",
    TechBranch.ELECTRONICS: "Electronics",
    TechBranch.NAVAL:       "Naval Tech",
    TechBranch.AIR:         "Air Tech",
    TechBranch.LAND:        "Land Tech",
    TechBranch.DOCTRINE:    "Doctrine",
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Tech Tree — bonuses per branch per tier                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class TechBonus:
    """Bonus granted when a tech tier is completed."""
    name: str
    description: str
    # Multiplier bonuses (1.0 = no bonus, 1.1 = +10%)
    factory_output_mult: float = 1.0
    construction_speed_mult: float = 1.0
    resource_extraction_mult: float = 1.0
    radar_range_mult: float = 1.0
    code_break_speed_mult: float = 1.0
    intel_quality_mult: float = 1.0
    ship_combat_mult: float = 1.0
    sub_stealth_mult: float = 1.0
    torpedo_mult: float = 1.0
    fighter_mult: float = 1.0
    bombing_accuracy_mult: float = 1.0
    aircraft_speed_mult: float = 1.0
    infantry_combat_mult: float = 1.0
    artillery_mult: float = 1.0
    fortification_mult: float = 1.0
    armor_mult: float = 1.0
    org_recovery_mult: float = 1.0
    supply_efficiency_mult: float = 1.0
    planning_speed_mult: float = 1.0
    combined_arms_mult: float = 1.0
    production_mult: float = 1.0  # general production bonus


TECH_TREE: Dict[TechBranch, List[TechBonus]] = {
    TechBranch.INDUSTRY: [
        TechBonus("Improved Machine Tools", "Factory output +8%", factory_output_mult=1.08),
        TechBonus("Prefab Construction", "Construction speed +8%, factory +5%",
                  construction_speed_mult=1.08, factory_output_mult=1.05),
        TechBonus("Synthetic Materials", "Resource extraction +10%, production +5%",
                  resource_extraction_mult=1.10, production_mult=1.05),
        TechBonus("Automated Assembly", "Factory +12%, construction +8%",
                  factory_output_mult=1.12, construction_speed_mult=1.08),
        TechBonus("Nuclear Power", "Factory +15%, all production +8% (requires Electronics T3)",
                  factory_output_mult=1.15, production_mult=1.08),
    ],
    TechBranch.ELECTRONICS: [
        TechBonus("Improved Radar", "Radar range +10%", radar_range_mult=1.10),
        TechBonus("Cryptanalysis Machine", "Code-breaking +15%", code_break_speed_mult=1.15),
        TechBonus("Transistor Circuits", "Production +6%, radar +8%",
                  production_mult=1.06, radar_range_mult=1.08),
        TechBonus("Electronic Warfare", "Intel +15%, code-break +10%",
                  intel_quality_mult=1.15, code_break_speed_mult=1.10),
        TechBonus("Computing Machines", "Intel +20%, all research +15%",
                  intel_quality_mult=1.20, production_mult=1.05),
    ],
    TechBranch.NAVAL: [
        TechBonus("Fire Control Mk II", "Ship combat +8%", ship_combat_mult=1.08),
        TechBonus("Snorkel Submarines", "Sub stealth +10%", sub_stealth_mult=1.10),
        TechBonus("Acoustic Torpedoes", "Torpedo +12%, ship combat +5%",
                  torpedo_mult=1.12, ship_combat_mult=1.05),
        TechBonus("Advanced Sonar", "Sub stealth +10%, ship combat +8% (requires Electronics T2)",
                  sub_stealth_mult=1.10, ship_combat_mult=1.08),
        TechBonus("Nuclear Propulsion", "Sub stealth +15%, torpedo +10% (requires Industry T4)",
                  sub_stealth_mult=1.15, torpedo_mult=1.10),
    ],
    TechBranch.AIR: [
        TechBonus("Improved Engines", "Fighter +8%, speed +5%",
                  fighter_mult=1.08, aircraft_speed_mult=1.05),
        TechBonus("Precision Bombsight", "Bombing accuracy +10%", bombing_accuracy_mult=1.10),
        TechBonus("All-Weather Navigation", "Bombing +8%, speed +5%",
                  bombing_accuracy_mult=1.08, aircraft_speed_mult=1.05),
        TechBonus("Swept-Wing Design", "Fighter +10%, speed +10%",
                  fighter_mult=1.10, aircraft_speed_mult=1.10),
        TechBonus("Jet Propulsion", "Fighter +12%, speed +12% (requires Industry T3)",
                  fighter_mult=1.12, aircraft_speed_mult=1.12),
    ],
    TechBranch.LAND: [
        TechBonus("Improved Infantry Kit", "Infantry +8%", infantry_combat_mult=1.08),
        TechBonus("Self-Propelled Artillery", "Artillery +10%", artillery_mult=1.10),
        TechBonus("Fortification Engineering", "Fort +8%, infantry +5%",
                  fortification_mult=1.08, infantry_combat_mult=1.05),
        TechBonus("Improved Armor Plate", "Armor +12%, fort +5%",
                  armor_mult=1.12, fortification_mult=1.05),
        TechBonus("Guided Munitions", "Artillery +15%, armor +8% (requires Electronics T3)",
                  artillery_mult=1.15, armor_mult=1.08),
    ],
    TechBranch.DOCTRINE: [
        TechBonus("Flexible Defense", "Org recovery +8%", org_recovery_mult=1.08),
        TechBonus("Supply Corps Reform", "Supply efficiency +10%", supply_efficiency_mult=1.10),
        TechBonus("Combined Arms Basics", "Combined arms +8%, planning +5%",
                  combined_arms_mult=1.08, planning_speed_mult=1.05),
        TechBonus("Deep Operations", "Planning +10%, combined arms +8%",
                  planning_speed_mult=1.10, combined_arms_mult=1.08),
        TechBonus("Total War Doctrine", "All military +10%, supply +8%",
                  combined_arms_mult=1.10, supply_efficiency_mult=1.08, org_recovery_mult=1.05),
    ],
}

# Prerequisites: {(branch, tier) -> [(required_branch, required_tier), ...]}
TECH_PREREQUISITES: Dict[Tuple[TechBranch, int], List[Tuple[TechBranch, int]]] = {
    (TechBranch.INDUSTRY, 4):    [(TechBranch.ELECTRONICS, 2)],   # Nuclear Power needs Electronics T3
    (TechBranch.NAVAL, 3):       [(TechBranch.ELECTRONICS, 1)],   # Advanced Sonar needs Electronics T2
    (TechBranch.NAVAL, 4):       [(TechBranch.INDUSTRY, 3)],      # Nuclear Sub needs Industry T4
    (TechBranch.AIR, 4):         [(TechBranch.INDUSTRY, 2)],      # Jet Propulsion needs Industry T3
    (TechBranch.LAND, 4):        [(TechBranch.ELECTRONICS, 2)],   # Guided Munitions needs Electronics T3
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Research Project                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class TechProject:
    """An active research project."""
    branch: TechBranch
    tier: int                       # 0-3 (which tier being researched)
    progress: float = 0.0          # 0.0 → 1.0
    turns_spent: int = 0
    is_complete: bool = False

    @property
    def required_turns(self) -> int:
        return TIER_RESEARCH_TIME[self.tier] if self.tier < len(TIER_RESEARCH_TIME) else 30

    @property
    def progress_pct(self) -> float:
        return min(1.0, self.progress)


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Research State                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionResearch:
    """Complete research state for one faction."""
    faction_id: int
    # Current tier level per branch (0 = no research done, 1 = T1 complete, etc.)
    levels: Dict[TechBranch, int] = field(
        default_factory=lambda: {b: 0 for b in TechBranch})
    # Active projects (max 2 simultaneous)
    active_projects: List[TechProject] = field(default_factory=list)
    # Accumulated bonuses (multiplicative)
    bonuses: Dict[str, float] = field(default_factory=lambda: {})
    # Research speed modifier (from ELECTRONICS)
    research_speed: float = 1.0
    # Completed tech names (for display)
    completed_techs: List[str] = field(default_factory=list)
    max_simultaneous: int = 2  # max projects at once

    def current_level(self, branch: TechBranch) -> int:
        return self.levels.get(branch, 0)

    def can_research(self, branch: TechBranch) -> Tuple[bool, str]:
        """Can we start a new project in this branch? Returns (ok, reason)."""
        level = self.current_level(branch)
        if level >= MAX_TIER:
            return False, f"{BRANCH_NAMES[branch]} maxed at T{level}."
        # Check not already researching this branch
        for p in self.active_projects:
            if p.branch == branch and not p.is_complete:
                return False, f"Already researching {BRANCH_NAMES[branch]}."
        if len([p for p in self.active_projects if not p.is_complete]) >= self.max_simultaneous:
            return False, f"All {self.max_simultaneous} research slots full."
        # Check prerequisites
        prereqs = TECH_PREREQUISITES.get((branch, level), [])
        for req_branch, req_tier in prereqs:
            if self.current_level(req_branch) < req_tier + 1:
                return False, f"Requires {BRANCH_NAMES[req_branch]} T{req_tier+1} first."
        return True, "OK"

    def get_bonus(self, key: str) -> float:
        """Get accumulated bonus multiplier for a key. Default 1.0."""
        return self.bonuses.get(key, 1.0)


# ═══════════════════════════════════════════════════════════════════════════ #
# Research World                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ResearchWorld:
    """Research state for all factions."""
    factions: Dict[int, FactionResearch] = field(default_factory=dict)
    turn: int = 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Research Step                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_research(
    world: ResearchWorld,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[ResearchWorld, Dict[str, Any]]:
    """Advance all research projects by one turn."""
    feedback: Dict[str, Any] = {}
    events = []

    for fid, fr in world.factions.items():
        for proj in fr.active_projects:
            if proj.is_complete:
                continue

            # Research speed: base + electronics bonus
            speed = (1.0 / proj.required_turns) * fr.research_speed * dt
            proj.progress += speed
            proj.turns_spent += 1

            if proj.progress >= 1.0:
                proj.is_complete = True
                fr.levels[proj.branch] = proj.tier + 1

                # Apply bonus
                bonus = TECH_TREE[proj.branch][proj.tier]
                for attr in dir(bonus):
                    if attr.endswith('_mult') and not attr.startswith('_'):
                        val = getattr(bonus, attr)
                        if val != 1.0:
                            key = attr
                            old = fr.bonuses.get(key, 1.0)
                            fr.bonuses[key] = old * val

                # Electronics research boosts research speed itself
                if proj.branch == TechBranch.ELECTRONICS:
                    fr.research_speed = min(2.0, fr.research_speed * 1.1)

                fr.completed_techs.append(bonus.name)
                events.append(f"Faction {fid} completed: {bonus.name} ({bonus.description})")

        # Clean completed projects
        fr.active_projects = [p for p in fr.active_projects if not p.is_complete]

    feedback["events"] = events
    world.turn += 1
    return world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Research Actions                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_research_action(
    world: ResearchWorld,
    faction_id: int,
    branch_name: str,
) -> Tuple[ResearchWorld, str]:
    """Start researching a tech branch. Returns (world, message)."""
    fr = world.factions.get(faction_id)
    if fr is None:
        return world, "No research capability."

    # Parse branch name
    branch_map = {
        "INDUSTRY": TechBranch.INDUSTRY,
        "ELECTRONICS": TechBranch.ELECTRONICS,
        "NAVAL": TechBranch.NAVAL,
        "AIR": TechBranch.AIR,
        "LAND": TechBranch.LAND,
        "DOCTRINE": TechBranch.DOCTRINE,
    }
    branch = branch_map.get(branch_name.upper())
    if branch is None:
        return world, f"Unknown research branch: {branch_name}"

    ok, reason = fr.can_research(branch)
    if not ok:
        return world, reason

    tier = fr.current_level(branch)
    proj = TechProject(branch=branch, tier=tier)
    fr.active_projects.append(proj)

    bonus = TECH_TREE[branch][tier]
    return world, f"Research started: {bonus.name} ({BRANCH_NAMES[branch]} T{tier+1}, {proj.required_turns} turns). {bonus.description}"


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_research(faction_ids: List[int]) -> ResearchWorld:
    """Initialize research for all factions. Everyone starts at T0."""
    factions = {}
    for fid in faction_ids:
        # Oceania starts with slight electronics advantage (Colossus heritage)
        fr = FactionResearch(faction_id=fid)
        if fid == 0:  # Oceania
            fr.research_speed = 1.05  # slight bonus from Turing's legacy
        factions[fid] = fr
    return ResearchWorld(factions=factions)


# ═══════════════════════════════════════════════════════════════════════════ #
# Summary                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def research_summary(world: ResearchWorld, faction_id: int) -> str:
    """Generate research summary for turn reports."""
    fr = world.factions.get(faction_id)
    if fr is None:
        return "No research data."

    parts = []

    # Tech levels
    levels = []
    for branch in TechBranch:
        lvl = fr.current_level(branch)
        if lvl > 0:
            levels.append(f"{BRANCH_NAMES[branch]}:T{lvl}")
    if levels:
        parts.append("Tech: " + ", ".join(levels))
    else:
        parts.append("Tech: None researched")

    # Active projects
    active = [p for p in fr.active_projects if not p.is_complete]
    if active:
        projs = []
        for p in active:
            bonus = TECH_TREE[p.branch][p.tier]
            projs.append(f"{bonus.name}({p.progress_pct:.0%})")
        parts.append("Researching: " + ", ".join(projs))
    else:
        slots = fr.max_simultaneous - len(active)
        parts.append(f"Research: {slots} slot(s) FREE — use RESEARCH branch_name")

    return " | ".join(parts)
