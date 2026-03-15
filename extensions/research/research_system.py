"""
research_system.py — HOI4-style Research & Technology system.

═══════════════════════════════════════════════════════════════════════════════
10 TECH BRANCHES — Historically based (1940s–1960s progression)

  INDUSTRY        — War production, construction, synthetic materials, automation
  ELECTRONICS     — Radar, vacuum tubes, transistors, early computing
  NAVAL           — Fire control, submarines, torpedoes, sonar, nuclear propulsion
  AIR             — Engines, bombsights, jet propulsion, swept wings, strategic bombers
  LAND            — Infantry kit, self-propelled artillery, armor, guided weapons
  DOCTRINE        — Flexible defense, combined arms, deception, deep battle, total war
  NUCLEAR         — Atomic theory, chain reaction, fission weapon, H-bomb, warheads
  ROCKETRY        — V-1 flying bomb, V-2 ballistic, SAMs, cruise missiles, IRBMs
  CRYPTOGRAPHY    — SIGINT, Enigma breaking, ECM/chaff, COMINT networks, automation
  INFRASTRUCTURE  — Railways, temporary harbors, civil defense, highways, bunkers

Each branch has 5 tiers (T1–T5). Higher tiers = better bonuses + longer research.
Max 2 simultaneous projects. Cross-branch prerequisites create strategic choices.

═══════════════════════════════════════════════════════════════════════════════
PREREQUISITE WEB (creates strategic dilemmas)

  Nuclear T3 (Fission Weapon)     ← Electronics T2 + Industry T3
  Nuclear T5 (Miniaturized)       ← Rocketry T3
  Rocketry T5 (IRBM)             ← Nuclear T3
  Rocketry T3 (SAM)              ← Electronics T2
  Air T3 (Jet Engine)            ← Industry T2
  Air T5 (Strategic Jet Bomber)  ← Rocketry T1
  Naval T5 (Nuclear Sub)         ← Nuclear T2
  Land T5 (ATGM)                 ← Electronics T3 + Rocketry T2
  Electronics T5 (Integrated Circuits) ← Industry T3
  Infrastructure T4 (Highway)    ← Industry T2

═══════════════════════════════════════════════════════════════════════════════
HISTORICAL NOTES

  Every tech corresponds to a real historical development:
  - Chain Home radar (1940), Colossus computer (1944), Transistor (1947)
  - Type XXI U-boat snorkel (1944), Mk 24 FIDO acoustic torpedo (1943)
  - Me 262 / Gloster Meteor (1944), MiG-15 / F-86 Sabre (1947)
  - Manhattan Project (1942-45), Ivy Mike H-bomb (1952)
  - V-1 pulse-jet (1944), V-2 ballistic (1944), Nike Ajax SAM (1953)
  - Mulberry harbour (1944), Autobahn/M1 motorway (1959)
  - Operation Bodyguard/Fortitude deception (1944)
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
    INDUSTRY        = 0
    ELECTRONICS     = 1
    NAVAL           = 2
    AIR             = 3
    LAND            = 4
    DOCTRINE        = 5
    NUCLEAR         = 6
    ROCKETRY        = 7
    CRYPTOGRAPHY    = 8
    INFRASTRUCTURE  = 9

N_BRANCHES = 10
MAX_TIER = 5

# Research time per tier (turns/weeks) — increasing cost
TIER_RESEARCH_TIME = [8, 12, 18, 25, 35]  # T1=8wk, T2=12wk, T3=18wk, T4=25wk, T5=35wk

BRANCH_NAMES = {
    TechBranch.INDUSTRY:        "Industry",
    TechBranch.ELECTRONICS:     "Electronics",
    TechBranch.NAVAL:           "Naval Tech",
    TechBranch.AIR:             "Air Tech",
    TechBranch.LAND:            "Land Tech",
    TechBranch.DOCTRINE:        "Doctrine",
    TechBranch.NUCLEAR:         "Nuclear",
    TechBranch.ROCKETRY:        "Rocketry",
    TechBranch.CRYPTOGRAPHY:    "Cryptography",
    TechBranch.INFRASTRUCTURE:  "Infrastructure",
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Tech Tree — bonuses per branch per tier                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class TechBonus:
    """Bonus granted when a tech tier is completed."""
    name: str
    description: str
    year: int = 1944                    # historical year of development
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
    production_mult: float = 1.0        # general production bonus
    # New bonus types for expanded branches
    air_defense_mult: float = 1.0       # SAM/AAA/interception effectiveness
    strategic_strike_mult: float = 1.0  # V-weapons, missiles, strategic bombing
    naval_strike_mult: float = 1.0      # anti-ship capability (air/missile)
    invasion_support_mult: float = 1.0  # amphibious operations bonus
    population_resilience_mult: float = 1.0  # civilian survival, morale
    counter_intel_mult: float = 1.0     # counter-espionage effectiveness
    power_generation_mult: float = 1.0  # power plant output
    research_speed_mult: float = 1.0    # bonus to all research speed
    military_mobility_mult: float = 1.0 # ground unit movement speed
    nuclear_damage_mult: float = 1.0    # nuclear weapon damage
    deterrence_mult: float = 1.0        # strategic deterrence value


# ═══════════════════════════════════════════════════════════════════════════ #
# INDUSTRY — War production to nuclear age                                     #
# Historical: US War Production Board (1942) → Liberty ships → synthetic       #
# rubber → assembly lines → Calder Hall nuclear power (1956)                   #
# ═══════════════════════════════════════════════════════════════════════════ #
_INDUSTRY_TREE = [
    TechBonus("War Production Board", "Factory output +8%. Standardized parts.",
             year=1942, factory_output_mult=1.08),
    TechBonus("Prefabricated Construction", "Construction +10%, factory +5%. Liberty ship methods.",
             year=1943, construction_speed_mult=1.10, factory_output_mult=1.05),
    TechBonus("Synthetic Materials", "Resources +12%, production +5%. Buna-S rubber, Fischer-Tropsch.",
             year=1944, resource_extraction_mult=1.12, production_mult=1.05),
    TechBonus("Assembly Line Automation", "Factory +12%, construction +8%. Semi-automated.",
             year=1952, factory_output_mult=1.12, construction_speed_mult=1.08),
    TechBonus("Nuclear Power Plants", "Power +20%, production +10%. Calder Hall / Obninsk type.",
             year=1956, power_generation_mult=1.20, production_mult=1.10, factory_output_mult=1.08),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# ELECTRONICS — Radar to computing                                             #
# Historical: Chain Home (1940) → Colossus (1944) → transistor (1947) →        #
# SAGE (1958) → integrated circuits (1961)                                     #
# ═══════════════════════════════════════════════════════════════════════════ #
_ELECTRONICS_TREE = [
    TechBonus("Chain Home Radar", "Radar +12%. Early warning network like Battle of Britain.",
             year=1940, radar_range_mult=1.12),
    TechBonus("Colossus Computer", "Code-breaking +15%. Bletchley Park electronic code-breaker.",
             year=1944, code_break_speed_mult=1.15, research_speed_mult=1.05),
    TechBonus("Transistor Circuits", "Production +8%, radar +10%. Bell Labs 1947 breakthrough.",
             year=1947, production_mult=1.08, radar_range_mult=1.10),
    TechBonus("SAGE Air Defense", "Air defense +15%, intel +12%. Semi-Automatic Ground Environment.",
             year=1958, air_defense_mult=1.15, intel_quality_mult=1.12, radar_range_mult=1.08),
    TechBonus("Integrated Circuits", "Intel +15%, research +12%. Miniaturized electronics revolution.",
             year=1961, intel_quality_mult=1.15, research_speed_mult=1.12, production_mult=1.05),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# NAVAL — Ship warfare to nuclear submarines                                   #
# Historical: Mk 37 fire control → Type XXI snorkel → Mk 24 FIDO →            #
# improved sonar → USS Nautilus (1955)                                         #
# ═══════════════════════════════════════════════════════════════════════════ #
_NAVAL_TREE = [
    TechBonus("Mk 37 Fire Control", "Ship combat +8%. Electromechanical gun director.",
             year=1941, ship_combat_mult=1.08),
    TechBonus("Type XXI Snorkel", "Sub stealth +12%. Submarine snorkel for submerged diesel.",
             year=1944, sub_stealth_mult=1.12),
    TechBonus("Mk 24 FIDO Torpedo", "Torpedo +12%, combat +5%. Acoustic homing 'mine'.",
             year=1943, torpedo_mult=1.12, ship_combat_mult=1.05),
    TechBonus("Active/Passive Sonar", "Combat +10%, sub stealth +8%. Dual-mode sonar array.",
             year=1952, ship_combat_mult=1.10, sub_stealth_mult=1.08),
    TechBonus("Nuclear Submarine", "Sub stealth +18%, torpedo +12%. USS Nautilus — unlimited range.",
             year=1955, sub_stealth_mult=1.18, torpedo_mult=1.12, naval_strike_mult=1.10),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# AIR — Propeller to jet age                                                   #
# Historical: Merlin supercharger → Norden bombsight → Me 262 jet →            #
# F-86 Sabre swept wing → V-bomber/B-52 strategic jet                          #
# ═══════════════════════════════════════════════════════════════════════════ #
_AIR_TREE = [
    TechBonus("Rolls-Royce Merlin 61", "Fighter +8%, speed +5%. Two-stage supercharger.",
             year=1942, fighter_mult=1.08, aircraft_speed_mult=1.05),
    TechBonus("Norden Bombsight", "Bombing +12%. Precision level bombing from altitude.",
             year=1943, bombing_accuracy_mult=1.12),
    TechBonus("Jet Engine", "Speed +12%, fighter +10%. Me 262 / Gloster Meteor type.",
             year=1944, aircraft_speed_mult=1.12, fighter_mult=1.10),
    TechBonus("Swept-Wing Design", "Fighter +12%, speed +10%. MiG-15 / F-86 Sabre era.",
             year=1947, fighter_mult=1.12, aircraft_speed_mult=1.10),
    TechBonus("Strategic Jet Bomber", "Bombing +15%, strike +12%. V-bomber / B-52 class.",
             year=1952, bombing_accuracy_mult=1.15, strategic_strike_mult=1.12, aircraft_speed_mult=1.08),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# LAND — Infantry to guided weapons                                            #
# Historical: Improved kit (1942) → M7 Priest SP gun → mine warfare →          #
# composite armor (1950s) → SS.11 ATGM (1956)                                  #
# ═══════════════════════════════════════════════════════════════════════════ #
_LAND_TREE = [
    TechBonus("Improved Infantry Kit", "Infantry +8%. Better rifle, helmet, webbing.",
             year=1942, infantry_combat_mult=1.08),
    TechBonus("Self-Propelled Artillery", "Artillery +12%. M7 Priest / Sexton type.",
             year=1943, artillery_mult=1.12),
    TechBonus("Mine Warfare Doctrine", "Fort +10%, infantry +5%. Anti-tank/anti-personnel mines.",
             year=1944, fortification_mult=1.10, infantry_combat_mult=1.05),
    TechBonus("Composite Armor Plate", "Armor +12%, fort +5%. Spaced/sloped armor design.",
             year=1952, armor_mult=1.12, fortification_mult=1.05),
    TechBonus("Anti-Tank Guided Missile", "Artillery +15%, armor +10%. SS.11 wire-guided ATGM.",
             year=1956, artillery_mult=1.15, armor_mult=1.10),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# DOCTRINE — Military organization and tactics                                 #
# Historical: Elastic defense (1940) → combined arms (1943) →                  #
# Bodyguard deception (1944) → Soviet deep battle → total war mobilization     #
# ═══════════════════════════════════════════════════════════════════════════ #
_DOCTRINE_TREE = [
    TechBonus("Flexible Defense", "Org recovery +8%. Defense in depth, elastic retreat.",
             year=1940, org_recovery_mult=1.08),
    TechBonus("Combined Arms Tactics", "Combined arms +10%, supply +5%. Infantry-tank-air coordination.",
             year=1943, combined_arms_mult=1.10, supply_efficiency_mult=1.05),
    TechBonus("Strategic Deception", "Planning +12%, intel +8%. Operation Bodyguard/Fortitude.",
             year=1944, planning_speed_mult=1.12, intel_quality_mult=1.08),
    TechBonus("Deep Battle Operations", "Combined arms +10%, planning +10%. Soviet operational art.",
             year=1945, combined_arms_mult=1.10, planning_speed_mult=1.10, supply_efficiency_mult=1.05),
    TechBonus("Total War Mobilization", "All military +8%, supply +10%, mobility +8%.",
             year=1950, combined_arms_mult=1.08, supply_efficiency_mult=1.10,
             military_mobility_mult=1.08, org_recovery_mult=1.05),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# NUCLEAR — Atomic theory to deliverable warheads                              #
# Historical: Manhattan Project (1942) → Chicago Pile-1 (1942) →               #
# Trinity/Fat Man (1945) → Ivy Mike H-bomb (1952) → Mk 7 miniaturized (1956)  #
# ═══════════════════════════════════════════════════════════════════════════ #
_NUCLEAR_TREE = [
    TechBonus("Atomic Theory", "Research +5%. Theoretical physics program established.",
             year=1942, research_speed_mult=1.05),
    TechBonus("Chain Reaction", "Power +10%, research +5%. Chicago Pile-1 type reactor.",
             year=1943, power_generation_mult=1.10, research_speed_mult=1.05),
    TechBonus("Fission Weapon", "Nuclear damage +50%, deterrence +30%. Fat Man / Little Boy type.",
             year=1945, nuclear_damage_mult=1.50, deterrence_mult=1.30, strategic_strike_mult=1.15),
    TechBonus("Thermonuclear Device", "Nuclear +100%, deterrence +50%. Ivy Mike — city-killer.",
             year=1952, nuclear_damage_mult=2.00, deterrence_mult=1.50, strategic_strike_mult=1.20),
    TechBonus("Miniaturized Warhead", "Nuclear +30%, deterrence +40%. Deliverable by missile/aircraft.",
             year=1956, nuclear_damage_mult=1.30, deterrence_mult=1.40, strategic_strike_mult=1.25),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# ROCKETRY — V-weapons to ballistic missiles                                   #
# Historical: V-1 (1944) → V-2 (1944) → Nike Ajax SAM (1953) →                #
# SSM-N-8 Regulus cruise missile (1955) → Thor IRBM (1957)                     #
# ═══════════════════════════════════════════════════════════════════════════ #
_ROCKETRY_TREE = [
    TechBonus("V-1 Flying Bomb", "Strategic strike +10%. Pulse-jet cruise missile, terror weapon.",
             year=1944, strategic_strike_mult=1.10, bombing_accuracy_mult=1.05),
    TechBonus("V-2 Ballistic Missile", "Strike +15%. Liquid-fuel rocket, unstoppable.",
             year=1944, strategic_strike_mult=1.15),
    TechBonus("Nike Ajax SAM", "Air defense +15%, fighter +5%. First operational surface-to-air missile.",
             year=1953, air_defense_mult=1.15, fighter_mult=1.05),
    TechBonus("Regulus Cruise Missile", "Naval strike +15%, strike +10%. Ship/sub-launched cruise missile.",
             year=1955, naval_strike_mult=1.15, strategic_strike_mult=1.10),
    TechBonus("Thor IRBM", "Strike +25%, deterrence +30%. Intermediate-range ballistic missile.",
             year=1957, strategic_strike_mult=1.25, deterrence_mult=1.30),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# CRYPTOGRAPHY — Signals intelligence to automation                            #
# Historical: Y-stations (1940) → Bombe/Enigma (1941) → Window/chaff (1943) → #
# GCHQ/NSA COMINT (1952) → automated SIGINT (1960)                            #
# ═══════════════════════════════════════════════════════════════════════════ #
_CRYPTOGRAPHY_TREE = [
    TechBonus("Signals Intelligence", "Intel +10%. Y-station intercept network, traffic analysis.",
             year=1940, intel_quality_mult=1.10),
    TechBonus("Enigma Breaking", "Code-break +18%. Bombe machine — read enemy communications.",
             year=1941, code_break_speed_mult=1.18, intel_quality_mult=1.05),
    TechBonus("Electronic Countermeasures", "Air defense +10%, naval +8%. Window/chaff, jammers.",
             year=1943, air_defense_mult=1.10, ship_combat_mult=1.08, counter_intel_mult=1.10),
    TechBonus("COMINT Network", "Intel +15%, counter-intel +12%. GCHQ/NSA-type signals network.",
             year=1952, intel_quality_mult=1.15, counter_intel_mult=1.12),
    TechBonus("Automated SIGINT", "Intel +15%, code-break +15%, research +5%. Machine-aided analysis.",
             year=1960, intel_quality_mult=1.15, code_break_speed_mult=1.15, research_speed_mult=1.05),
]

# ═══════════════════════════════════════════════════════════════════════════ #
# INFRASTRUCTURE — Logistics, transport, civil defense                         #
# Historical: Railway expansion → Mulberry harbour (1944) → civil defense →    #
# M1 Motorway (1959) → underground command bunkers                             #
# ═══════════════════════════════════════════════════════════════════════════ #
_INFRASTRUCTURE_TREE = [
    TechBonus("Railway Expansion", "Supply +10%. Increased rail capacity, marshalling yards.",
             year=1942, supply_efficiency_mult=1.10),
    TechBonus("Mulberry Harbour", "Invasion support +15%, supply +5%. Temporary port construction.",
             year=1944, invasion_support_mult=1.15, supply_efficiency_mult=1.05),
    TechBonus("Civil Defense Program", "Population resilience +15%. Bunkers, shelters, evacuation drills.",
             year=1950, population_resilience_mult=1.15, fortification_mult=1.05),
    TechBonus("Highway Network", "Supply +12%, mobility +10%. Motorway/Autobahn construction.",
             year=1955, supply_efficiency_mult=1.12, military_mobility_mult=1.10),
    TechBonus("Underground Command", "Resilience +15%, fort +10%. Hardened bunkers, continuity of govt.",
             year=1960, population_resilience_mult=1.15, fortification_mult=1.10,
             counter_intel_mult=1.08),
]


TECH_TREE: Dict[TechBranch, List[TechBonus]] = {
    TechBranch.INDUSTRY:        _INDUSTRY_TREE,
    TechBranch.ELECTRONICS:     _ELECTRONICS_TREE,
    TechBranch.NAVAL:           _NAVAL_TREE,
    TechBranch.AIR:             _AIR_TREE,
    TechBranch.LAND:            _LAND_TREE,
    TechBranch.DOCTRINE:        _DOCTRINE_TREE,
    TechBranch.NUCLEAR:         _NUCLEAR_TREE,
    TechBranch.ROCKETRY:        _ROCKETRY_TREE,
    TechBranch.CRYPTOGRAPHY:    _CRYPTOGRAPHY_TREE,
    TechBranch.INFRASTRUCTURE:  _INFRASTRUCTURE_TREE,
}

# ═══════════════════════════════════════════════════════════════════════════ #
# Prerequisites — cross-branch requirements create strategic dilemmas          #
# {(branch, tier_index) -> [(required_branch, required_tier_index), ...]}     #
# tier_index is 0-based: tier 0 = T1, tier 2 = T3, etc.                      #
# ═══════════════════════════════════════════════════════════════════════════ #
TECH_PREREQUISITES: Dict[Tuple[TechBranch, int], List[Tuple[TechBranch, int]]] = {
    # Nuclear T3 (Fission Weapon) needs Electronics T2 + Industry T3
    (TechBranch.NUCLEAR, 2):        [(TechBranch.ELECTRONICS, 1), (TechBranch.INDUSTRY, 2)],
    # Nuclear T5 (Miniaturized Warhead) needs Rocketry T3
    (TechBranch.NUCLEAR, 4):        [(TechBranch.ROCKETRY, 2)],
    # Rocketry T3 (Nike Ajax SAM) needs Electronics T2
    (TechBranch.ROCKETRY, 2):       [(TechBranch.ELECTRONICS, 1)],
    # Rocketry T5 (Thor IRBM) needs Nuclear T3
    (TechBranch.ROCKETRY, 4):       [(TechBranch.NUCLEAR, 2)],
    # Air T3 (Jet Engine) needs Industry T2
    (TechBranch.AIR, 2):            [(TechBranch.INDUSTRY, 1)],
    # Air T5 (Strategic Jet Bomber) needs Rocketry T1
    (TechBranch.AIR, 4):            [(TechBranch.ROCKETRY, 0)],
    # Naval T4 (Sonar) needs Electronics T2
    (TechBranch.NAVAL, 3):          [(TechBranch.ELECTRONICS, 1)],
    # Naval T5 (Nuclear Sub) needs Nuclear T2
    (TechBranch.NAVAL, 4):          [(TechBranch.NUCLEAR, 1)],
    # Land T5 (ATGM) needs Electronics T3 + Rocketry T2
    (TechBranch.LAND, 4):           [(TechBranch.ELECTRONICS, 2), (TechBranch.ROCKETRY, 1)],
    # Electronics T5 (Integrated Circuits) needs Industry T3
    (TechBranch.ELECTRONICS, 4):    [(TechBranch.INDUSTRY, 2)],
    # Industry T5 (Nuclear Power) needs Nuclear T2
    (TechBranch.INDUSTRY, 4):       [(TechBranch.NUCLEAR, 1)],
    # Infrastructure T4 (Highway) needs Industry T2
    (TechBranch.INFRASTRUCTURE, 3): [(TechBranch.INDUSTRY, 1)],
    # Doctrine T3 (Strategic Deception) needs Cryptography T2
    (TechBranch.DOCTRINE, 2):       [(TechBranch.CRYPTOGRAPHY, 1)],
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
        "NUCLEAR": TechBranch.NUCLEAR,
        "ROCKETRY": TechBranch.ROCKETRY,
        "CRYPTOGRAPHY": TechBranch.CRYPTOGRAPHY,
        "INFRASTRUCTURE": TechBranch.INFRASTRUCTURE,
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

    # Completed tech levels
    levels = []
    for branch in TechBranch:
        lvl = fr.current_level(branch)
        if lvl > 0:
            levels.append(f"{BRANCH_NAMES[branch]}:T{lvl}")
    if levels:
        parts.append("Completed: " + ", ".join(levels))
    else:
        parts.append("Completed: None")

    # Active projects — very explicit so LLMs don't re-research
    active = [p for p in fr.active_projects if not p.is_complete]
    if active:
        projs = []
        for p in active:
            bonus = TECH_TREE[p.branch][p.tier]
            turns_left = p.required_turns - int(p.progress_pct * p.required_turns)
            projs.append(f"{bonus.name} ({p.branch.name} T{p.tier+1}, {p.progress_pct:.0%}, ~{turns_left} turns left)")
        parts.append("IN PROGRESS (DO NOT re-research these): " + ", ".join(projs))
    else:
        parts.append("IN PROGRESS: Nothing — research slot FREE")

    # Free slots
    used = len(active)
    free = fr.max_simultaneous - used
    if free > 0 and active:
        # Show what branches are available
        available = []
        for branch in TechBranch:
            ok, _ = fr.can_research(branch)
            if ok:
                lvl = fr.current_level(branch)
                available.append(f"{branch.name}(→T{lvl+1})")
        if available:
            parts.append(f"{free} slot(s) FREE — available: {', '.join(available)}")
    elif free > 0:
        available = []
        for branch in TechBranch:
            ok, _ = fr.can_research(branch)
            if ok:
                available.append(branch.name)
        parts.append(f"{free} slot(s) FREE — use RESEARCH {'/'.join(available) if available else '(none available)'}")

    return " | ".join(parts)
