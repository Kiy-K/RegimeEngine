"""
national_spirit.py — National Spirit system for GRAVITAS Engine.

National Spirits are persistent faction-level modifiers that represent
the deep cultural, ideological, and historical character of a nation.
Unlike government types (which are structural), spirits are narrative —
they tell the story of who these people ARE.

Each faction can have multiple spirits. Spirits can be:
  - Permanent (core identity, never removed)
  - Conditional (activated/deactivated by game events)
  - Timed (expire after N turns)

1984 Spirits:
  Oceania (Ingsoc):
    - "War is Peace"          — perpetual war justifies the Party
    - "Thought Police"        — total surveillance, crushing dissent
    - "Doublethink"           — contradictions accepted, morale unshakeable
    - "Newspeak"              — language control, reduced innovation
    - "Prole Indifference"    — 85% ignored, potential uprising vector

  Eurasia (Neo-Bolshevism):
    - "Death of Capitalism"   — ideological fervor, production bonuses
    - "Committee Layers"      — bureaucratic delays, slow decisions
    - "People's Army"         — mass conscription, quantity over quality
    - "Continental Doctrine"  — land warfare focus, weak navy
    - "Revolutionary Zeal"    — high morale but brittle under defeat

  BLF (British Liberation Front):
    - "The Ghost of London"   — Winston's legend inspires
    - "Prole Rage"            — desperation makes fierce fighters
    - "No Retreat"            — fight to the death, no surrender
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

# Lazy import to avoid circular dependency
_CowExternalModifiers = None


def _get_cow_modifiers_class():
    global _CowExternalModifiers
    if _CowExternalModifiers is None:
        from extensions.military.military_state import CowExternalModifiers
        _CowExternalModifiers = CowExternalModifiers
    return _CowExternalModifiers


# ═══════════════════════════════════════════════════════════════════════════ #
# Spirit Category                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class SpiritCategory(Enum):
    MILITARY    = auto()
    ECONOMY     = auto()
    SOCIETY     = auto()
    IDEOLOGY    = auto()
    ESPIONAGE   = auto()
    SPECIAL     = auto()


# ═══════════════════════════════════════════════════════════════════════════ #
# National Spirit                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class NationalSpirit:
    """A single national spirit with modifiers."""
    spirit_id: str                       # unique key e.g. "ingsoc_war_is_peace"
    name: str                            # display name e.g. "War is Peace"
    description: str                     # flavor text
    category: SpiritCategory = SpiritCategory.IDEOLOGY
    is_permanent: bool = True            # permanent spirits can't be removed
    turns_remaining: int = -1            # -1 = permanent, >0 = expires

    # ── Military modifiers ──
    combat_effectiveness: float = 1.0    # multiplier
    morale_mod: float = 0.0              # additive
    production_speed_mult: float = 1.0
    production_cost_mult: float = 1.0
    conscription_bonus: float = 0.0      # additive to conscription capacity

    # ── Economy modifiers ──
    resource_production_mod: float = 1.0
    factory_output_mod: float = 1.0
    corruption_mod: float = 0.0          # additive to corruption floor

    # ── Society modifiers ──
    propaganda_mod: float = 0.0          # additive to propaganda effectiveness
    trust_generation_mod: float = 0.0    # additive to trust building rate
    unrest_suppression: float = 0.0      # additive — how much unrest is suppressed

    # ── Research modifiers ──
    research_speed_mod: float = 0.0      # additive to research speed
    innovation_mod: float = 0.0          # additive to innovation capacity

    # ── Espionage modifiers ──
    counter_intel_mod: float = 0.0       # additive
    internal_security_mod: float = 0.0   # additive

    # ── Special flags ──
    winter_adapted: bool = False
    partisan_bonus: bool = False
    scorched_earth: bool = False

    # ── Exhaustion ──
    exhaustion_rate_mod: float = 0.0     # additive (positive = faster exhaustion)
    exhaustion_recovery_mod: float = 0.0 # additive (positive = faster recovery)

    # ── Cohesion ──
    cohesion_mod: float = 0.0            # additive
    hazard_resistance: float = 0.0       # additive


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Spirit State                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionSpirits:
    """All national spirits for one faction."""
    faction_id: int
    spirits: List[NationalSpirit] = field(default_factory=list)

    @property
    def active_spirits(self) -> List[NationalSpirit]:
        return [s for s in self.spirits if s.turns_remaining != 0]

    def add_spirit(self, spirit: NationalSpirit) -> None:
        # Don't add duplicates
        if any(s.spirit_id == spirit.spirit_id for s in self.spirits):
            return
        self.spirits.append(spirit)

    def remove_spirit(self, spirit_id: str) -> None:
        self.spirits = [s for s in self.spirits if s.spirit_id != spirit_id or s.is_permanent]

    def has_spirit(self, spirit_id: str) -> bool:
        return any(s.spirit_id == spirit_id for s in self.active_spirits)

    def step(self) -> List[str]:
        """Advance timed spirits. Returns list of expired spirit names."""
        expired = []
        for s in self.spirits:
            if s.turns_remaining > 0:
                s.turns_remaining -= 1
                if s.turns_remaining == 0:
                    expired.append(s.name)
        self.spirits = [s for s in self.spirits if s.turns_remaining != 0]
        return expired


# ═══════════════════════════════════════════════════════════════════════════ #
# Spirit World                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class SpiritWorld:
    """National spirits for all factions."""
    factions: Dict[int, FactionSpirits] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════ #
# 1984 National Spirits — Oceania (Ingsoc)                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

OCEANIA_SPIRITS = [
    NationalSpirit(
        spirit_id="ingsoc_war_is_peace",
        name="War is Peace",
        description="The Party teaches that perpetual war is necessary for perpetual peace. "
                    "War justifies rationing, surveillance, and the destruction of surplus production. "
                    "The population accepts deprivation as patriotic duty.",
        category=SpiritCategory.IDEOLOGY,
        morale_mod=0.05,
        exhaustion_recovery_mod=0.03,     # war is normal, people adapt
        propaganda_mod=0.15,              # war propaganda is incredibly effective
        factory_output_mod=0.95,          # surplus destruction reduces real output
        trust_generation_mod=-0.1,        # perpetual war erodes real trust
    ),
    NationalSpirit(
        spirit_id="ingsoc_thought_police",
        name="Thought Police",
        description="The Thought Police are everywhere. Telescreens in every room. Children spy on parents. "
                    "Thoughtcrime is death. No cell can form, no conspiracy can survive — "
                    "unless it hides among the proles, where the Party never looks.",
        category=SpiritCategory.ESPIONAGE,
        internal_security_mod=0.35,       # massive internal security
        counter_intel_mod=0.25,           # excellent counter-intelligence
        unrest_suppression=0.2,           # unrest is crushed
        trust_generation_mod=-0.15,       # fear ≠ trust
        innovation_mod=-0.1,             # fear kills creativity
        cohesion_mod=0.1,                # forced cohesion through terror
    ),
    NationalSpirit(
        spirit_id="ingsoc_doublethink",
        name="Doublethink",
        description="The power of holding two contradictory beliefs simultaneously. "
                    "WAR IS PEACE. FREEDOM IS SLAVERY. IGNORANCE IS STRENGTH. "
                    "The population cannot be demoralized because they cannot perceive contradiction.",
        category=SpiritCategory.IDEOLOGY,
        morale_mod=0.08,                  # contradictions don't hurt morale
        propaganda_mod=0.2,               # propaganda works even when obviously false
        research_speed_mod=-0.1,          # 2+2=5 is not good for science
        innovation_mod=-0.15,            # deliberate ignorance
        hazard_resistance=0.05,           # contradictions are absorbed
    ),
    NationalSpirit(
        spirit_id="ingsoc_newspeak",
        name="Newspeak",
        description="The language is being systematically reduced. By 2050, thoughtcrime will be "
                    "literally impossible because there will be no words to express it. "
                    "Meanwhile, communication becomes crude and imprecise.",
        category=SpiritCategory.SOCIETY,
        propaganda_mod=0.1,               # simpler language = simpler control
        research_speed_mod=-0.1,          # can't think complex thoughts
        innovation_mod=-0.2,             # language constrains thought
        counter_intel_mod=0.1,            # enemy can't decode Newspeak intercepts
        cohesion_mod=0.05,               # everyone speaks the same way
    ),
    NationalSpirit(
        spirit_id="ingsoc_prole_indifference",
        name="Prole Indifference",
        description="The proles are free. The Party doesn't bother with them because "
                    "'proles and animals are free.' 85% of the population lives in squalor "
                    "but is left alone — a vast untapped reservoir of potential resistance.",
        category=SpiritCategory.SOCIETY,
        conscription_bonus=-0.1,          # can't conscript proles effectively
        resource_production_mod=0.9,      # prole labor is underutilized
        unrest_suppression=-0.1,          # proles are ungoverned — potential uprising
        exhaustion_rate_mod=-0.02,        # proles absorb the burden silently
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 1984 National Spirits — Eurasia (Neo-Bolshevism)                           #
# ═══════════════════════════════════════════════════════════════════════════ #

EURASIA_SPIRITS = [
    NationalSpirit(
        spirit_id="eurasia_death_of_capitalism",
        name="Death of Capitalism",
        description="The Revolution has abolished private property, class distinction, and individual ambition. "
                    "The Collective owns everything. Workers toil for the State with ideological fervor — "
                    "but without market incentives, efficiency suffers.",
        category=SpiritCategory.IDEOLOGY,
        morale_mod=0.05,
        production_speed_mult=1.05,       # ideological drive
        factory_output_mod=0.90,          # central planning inefficiency
        resource_production_mod=0.95,
        propaganda_mod=0.1,
        trust_generation_mod=-0.05,
    ),
    NationalSpirit(
        spirit_id="eurasia_committee_layers",
        name="Committee Layers",
        description="Every decision passes through seven levels of committee approval. "
                    "The Politburo approves the Supreme Soviet which directs the People's Commissariat "
                    "which instructs the Regional Committee which... nothing happens quickly.",
        category=SpiritCategory.SOCIETY,
        production_speed_mult=0.85,       # everything is slow
        research_speed_mod=-0.05,         # committees delay innovation
        corruption_mod=0.05,              # bureaucracy breeds corruption
        exhaustion_rate_mod=-0.03,        # but also: nothing gets done fast enough to exhaust
        cohesion_mod=-0.05,              # committees cause infighting
    ),
    NationalSpirit(
        spirit_id="eurasia_peoples_army",
        name="People's Army",
        description="Every citizen is a soldier of the Revolution. Mass conscription is not just accepted — "
                    "it is celebrated. The army is vast but poorly equipped, its officers politically appointed, "
                    "its tactics favoring human wave over finesse.",
        category=SpiritCategory.MILITARY,
        combat_effectiveness=0.95,        # quantity over quality
        conscription_bonus=0.15,          # mass conscription
        morale_mod=0.03,
        production_cost_mult=0.85,        # cheap mass-produced equipment
        exhaustion_rate_mod=0.02,         # human cost of mass warfare
    ),
    NationalSpirit(
        spirit_id="eurasia_continental_doctrine",
        name="Continental Doctrine",
        description="Eurasia's military doctrine is land-focused. The vast steppes shaped a tradition of "
                    "deep operations, mass armor thrusts, and artillery barrages. The navy is an afterthought — "
                    "a fleet of submarines and coastal craft, never a blue-water force.",
        category=SpiritCategory.MILITARY,
        combat_effectiveness=1.05,        # land combat bonus
        production_cost_mult=0.95,        # efficient land unit production
        # Navy penalty is implicit — fewer starting ships, doctrine doesn't help at sea
    ),
    NationalSpirit(
        spirit_id="eurasia_revolutionary_zeal",
        name="Revolutionary Zeal",
        description="The fires of Revolution still burn. Soldiers fight for the Cause, not for pay. "
                    "Commissars ensure discipline through inspiration — or firing squads. "
                    "Morale is high but brittle: a major defeat can shatter the illusion.",
        category=SpiritCategory.IDEOLOGY,
        morale_mod=0.08,
        propaganda_mod=0.15,
        exhaustion_recovery_mod=0.02,
        hazard_resistance=0.03,
        trust_generation_mod=-0.08,       # zeal ≠ trust in institutions
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# 1984 National Spirits — BLF (British Liberation Front)                      #
# ═══════════════════════════════════════════════════════════════════════════ #

BLF_SPIRITS = [
    NationalSpirit(
        spirit_id="blf_ghost_of_london",
        name="The Ghost of London",
        description="Winston Smith is more than a man — he is a legend. Every prole in London whispers "
                    "his name. Every piece of graffiti bears his mark. He is the proof that the Party can be defied. "
                    "His survival is the revolution's greatest weapon.",
        category=SpiritCategory.SPECIAL,
        morale_mod=0.15,
        propaganda_mod=0.2,
        combat_effectiveness=0.9,         # inspiring but untrained
        hazard_resistance=0.05,
        partisan_bonus=True,
    ),
    NationalSpirit(
        spirit_id="blf_prole_rage",
        name="Prole Rage",
        description="Decades of ignored hunger, squalor, and contempt have forged the proles into something "
                    "the Party never expected: an angry, desperate, fearless mass. They have nothing to lose. "
                    "Their fury makes them dangerous fighters — but undisciplined and wasteful.",
        category=SpiritCategory.MILITARY,
        combat_effectiveness=1.1,         # rage makes fierce fighters
        morale_mod=0.1,                   # nothing to lose
        exhaustion_rate_mod=0.05,         # rage burns fast
        production_cost_mult=1.3,         # no logistics, wasteful
        conscription_bonus=0.2,           # everyone joins the fight
    ),
    NationalSpirit(
        spirit_id="blf_no_retreat",
        name="No Retreat",
        description="If the revolution fails, everyone dies. There is no surrender, no negotiation, "
                    "no going back to the Chestnut Tree Café. This is a fight to the death. "
                    "Every prole knows it. They fight like cornered animals.",
        category=SpiritCategory.MILITARY,
        combat_effectiveness=1.15,        # fight to the death
        morale_mod=0.12,
        exhaustion_rate_mod=0.08,         # fighting to death is exhausting
        exhaustion_recovery_mod=-0.05,    # no rest for the desperate
    ),
]


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_spirits(faction_ids: List[int]) -> SpiritWorld:
    """Initialize national spirits for Air Strip One factions."""
    world = SpiritWorld()
    for fid in faction_ids:
        fs = FactionSpirits(faction_id=fid)
        if fid == 0:  # Oceania
            for s in OCEANIA_SPIRITS:
                fs.add_spirit(s)
        elif fid == 1:  # Eurasia
            for s in EURASIA_SPIRITS:
                fs.add_spirit(s)
        elif fid == 2:  # BLF
            for s in BLF_SPIRITS:
                fs.add_spirit(s)
        world.factions[fid] = fs
    return world


def step_spirits(world: SpiritWorld) -> Dict[int, List[str]]:
    """Advance all timed spirits. Returns {faction_id: [expired spirit names]}."""
    expired = {}
    for fid, fs in world.factions.items():
        exp = fs.step()
        if exp:
            expired[fid] = exp
    return expired


# ═══════════════════════════════════════════════════════════════════════════ #
# Aggregate Modifiers                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def aggregate_spirit_modifiers(fs: FactionSpirits) -> Dict[str, float]:
    """Aggregate all active spirit modifiers into a single dict."""
    agg = {
        "combat_effectiveness": 1.0,
        "morale_mod": 0.0,
        "production_speed_mult": 1.0,
        "production_cost_mult": 1.0,
        "resource_production_mod": 1.0,
        "factory_output_mod": 1.0,
        "corruption_mod": 0.0,
        "propaganda_mod": 0.0,
        "trust_generation_mod": 0.0,
        "unrest_suppression": 0.0,
        "research_speed_mod": 0.0,
        "innovation_mod": 0.0,
        "counter_intel_mod": 0.0,
        "internal_security_mod": 0.0,
        "conscription_bonus": 0.0,
        "exhaustion_rate_mod": 0.0,
        "exhaustion_recovery_mod": 0.0,
        "cohesion_mod": 0.0,
        "hazard_resistance": 0.0,
        "winter_adapted": False,
        "partisan_bonus": False,
        "scorched_earth": False,
    }
    for s in fs.active_spirits:
        agg["combat_effectiveness"] *= s.combat_effectiveness
        agg["morale_mod"] += s.morale_mod
        agg["production_speed_mult"] *= s.production_speed_mult
        agg["production_cost_mult"] *= s.production_cost_mult
        agg["resource_production_mod"] *= s.resource_production_mod
        agg["factory_output_mod"] *= s.factory_output_mod
        agg["corruption_mod"] += s.corruption_mod
        agg["propaganda_mod"] += s.propaganda_mod
        agg["trust_generation_mod"] += s.trust_generation_mod
        agg["unrest_suppression"] += s.unrest_suppression
        agg["research_speed_mod"] += s.research_speed_mod
        agg["innovation_mod"] += s.innovation_mod
        agg["counter_intel_mod"] += s.counter_intel_mod
        agg["internal_security_mod"] += s.internal_security_mod
        agg["conscription_bonus"] += s.conscription_bonus
        agg["exhaustion_rate_mod"] += s.exhaustion_rate_mod
        agg["exhaustion_recovery_mod"] += s.exhaustion_recovery_mod
        agg["cohesion_mod"] += s.cohesion_mod
        agg["hazard_resistance"] += s.hazard_resistance
        if s.winter_adapted:
            agg["winter_adapted"] = True
        if s.partisan_bonus:
            agg["partisan_bonus"] = True
        if s.scorched_earth:
            agg["scorched_earth"] = True
    return agg


def spirit_to_cow_modifiers(fs: FactionSpirits) -> "CowExternalModifiers":
    """Convert aggregated spirit modifiers to CowExternalModifiers."""
    CowMods = _get_cow_modifiers_class()
    agg = aggregate_spirit_modifiers(fs)

    return CowMods(
        production_cost_mult=agg["production_cost_mult"],
        production_speed_mult=agg["production_speed_mult"],
        combat_effectiveness=agg["combat_effectiveness"],
        morale_mod=agg["morale_mod"],
        resource_income_mult=agg["resource_production_mod"],
        winter_adapted=agg["winter_adapted"],
        partisan_bonus=agg["partisan_bonus"],
        scorched_earth=agg["scorched_earth"],
        hazard_resistance=agg["hazard_resistance"],
        exhaustion_rate_mod=agg["exhaustion_rate_mod"],
        exhaustion_recovery_mod=agg["exhaustion_recovery_mod"],
        cohesion_mod=agg["cohesion_mod"],
        propaganda_mod=agg["propaganda_mod"],
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# Summary for turn reports                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def spirits_summary(fs: FactionSpirits) -> str:
    """One-line summary of active spirits for turn reports."""
    active = fs.active_spirits
    if not active:
        return "None"
    names = [s.name for s in active]
    return ", ".join(names)


def load_spirit_from_yaml(cfg: dict) -> NationalSpirit:
    """Load a single spirit from YAML configuration dict."""
    return NationalSpirit(
        spirit_id=cfg.get("id", "custom"),
        name=cfg.get("name", "Custom Spirit"),
        description=cfg.get("description", ""),
        category=SpiritCategory[cfg.get("category", "IDEOLOGY").upper()],
        is_permanent=cfg.get("permanent", True),
        turns_remaining=cfg.get("turns", -1),
        combat_effectiveness=cfg.get("combat_effectiveness", 1.0),
        morale_mod=cfg.get("morale_mod", 0.0),
        production_speed_mult=cfg.get("production_speed_mult", 1.0),
        production_cost_mult=cfg.get("production_cost_mult", 1.0),
        resource_production_mod=cfg.get("resource_production_mod", 1.0),
        factory_output_mod=cfg.get("factory_output_mod", 1.0),
        propaganda_mod=cfg.get("propaganda_mod", 0.0),
        research_speed_mod=cfg.get("research_speed_mod", 0.0),
        counter_intel_mod=cfg.get("counter_intel_mod", 0.0),
        winter_adapted=cfg.get("winter_adapted", False),
        partisan_bonus=cfg.get("partisan_bonus", False),
        scorched_earth=cfg.get("scorched_earth", False),
    )
