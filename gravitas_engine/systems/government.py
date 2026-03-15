"""
government.py — Government type system for GRAVITAS Engine.

Defines 12 government types with detailed modifiers that affect:
  - Military: combat effectiveness, production cost, conscription capacity
  - Economy: efficiency, corruption, resource extraction
  - Society: unrest tolerance, propaganda effectiveness, trust generation
  - Research: speed modifier, innovation capacity
  - Espionage: counter-intelligence, internal security

Each government type maps to CowExternalModifiers for integration with
the military/land combat system, and provides standalone modifiers for
the economy, research, governance, and intelligence systems.

Historical basis:
  - TOTALITARIAN (Oceania/Ingsoc): total control, fear-based, innovation dies
  - COMMUNIST (Eurasia/Neo-Bolshevism): central planning, committee layers
  - Each type is balanced: strengths come with real weaknesses
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

# Lazy import to avoid circular dependency
_CowExternalModifiers = None


def _get_cow_modifiers_class():
    global _CowExternalModifiers
    if _CowExternalModifiers is None:
        from extensions.military.military_state import CowExternalModifiers
        _CowExternalModifiers = CowExternalModifiers
    return _CowExternalModifiers


# ═══════════════════════════════════════════════════════════════════════════ #
# Government Types                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class GovernmentType(Enum):
    """12 government types — mirrors RegimeType in manpower.py."""
    # Democratic spectrum
    LIBERAL_DEMOCRACY     = 0    # US/UK pre-1984 — civil liberties, free markets
    SOCIAL_DEMOCRACY      = 1    # Nordics — welfare state, strong institutions
    REPUBLIC              = 2    # Generic republic with elections
    MANAGED_DEMOCRACY     = 3    # Controlled elections, state media

    # Authoritarian spectrum
    OLIGARCHY             = 4    # Rule by elites / business interests
    MILITARY_JUNTA        = 5    # Military officers in power
    ONE_PARTY_STATE       = 6    # Single ruling party (China model)
    AUTHORITARIAN         = 7    # Strongman / personalist rule

    # Ideological extremes
    THEOCRACY             = 8    # Religious law governs
    FASCIST               = 9    # Ultranationalist corporatism
    COMMUNIST             = 10   # Marxist-Leninist state control
    TOTALITARIAN          = 11   # Complete state control — Ingsoc, the Party


# ═══════════════════════════════════════════════════════════════════════════ #
# Government Modifiers — per-type detailed stat blocks                       #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass(frozen=True)
class GovernmentModifiers:
    """Complete modifier set for a government type."""
    name: str
    description: str

    # Military
    combat_effectiveness: float = 1.0    # multiplier on unit combat power
    production_cost_mult: float = 1.0    # military production cost (lower = cheaper)
    production_speed_mult: float = 1.0   # military production speed
    morale_base: float = 0.0             # additive morale modifier
    conscription_capacity: float = 1.0   # how much of population can be drafted

    # Economy
    economy_efficiency: float = 1.0      # GDP multiplier
    corruption_floor: float = 0.05       # minimum corruption (0-1)
    resource_income_mult: float = 1.0    # resource extraction rate
    factory_output_mult: float = 1.0     # factory production multiplier
    trade_efficiency: float = 1.0        # international trade bonus

    # Society
    unrest_tolerance: float = 1.0        # how much unrest before instability (higher = more tolerant)
    propaganda_effectiveness: float = 1.0  # how well propaganda works
    trust_generation: float = 1.0        # rate of institutional trust building
    civil_liberties: float = 0.5         # 0=none, 1=full — affects innovation + unrest
    political_stability: float = 0.5     # 0=unstable, 1=rock-solid

    # Research
    research_speed_mult: float = 1.0     # research speed modifier
    innovation_capacity: float = 1.0     # breakthrough chance modifier

    # Espionage
    counter_intel_mult: float = 1.0      # counter-intelligence effectiveness
    internal_security: float = 1.0       # domestic security / Thought Police
    espionage_offense: float = 1.0       # offensive spy operations

    # Special flags
    censorship: bool = False             # blocks enemy propaganda
    forced_labor: bool = False           # can use forced labor for production
    secret_police: bool = False          # enhanced internal security
    state_planning: bool = False         # centralized economic planning


# ═══════════════════════════════════════════════════════════════════════════ #
# Government Type Definitions                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

GOVERNMENT_MODIFIERS: Dict[GovernmentType, GovernmentModifiers] = {

    GovernmentType.LIBERAL_DEMOCRACY: GovernmentModifiers(
        name="Liberal Democracy",
        description="Free markets, civil liberties, independent judiciary. Strong economy but slow to mobilize.",
        combat_effectiveness=1.0, production_cost_mult=1.0, production_speed_mult=0.9,
        morale_base=0.05, conscription_capacity=0.6,
        economy_efficiency=1.15, corruption_floor=0.03, resource_income_mult=1.1,
        factory_output_mult=1.1, trade_efficiency=1.2,
        unrest_tolerance=0.6, propaganda_effectiveness=0.7, trust_generation=1.3,
        civil_liberties=0.9, political_stability=0.7,
        research_speed_mult=1.15, innovation_capacity=1.2,
        counter_intel_mult=0.8, internal_security=0.7, espionage_offense=0.9,
    ),

    GovernmentType.SOCIAL_DEMOCRACY: GovernmentModifiers(
        name="Social Democracy",
        description="Welfare state with strong institutions. Balanced but lacks military edge.",
        combat_effectiveness=1.0, production_cost_mult=1.05, production_speed_mult=0.95,
        morale_base=0.08, conscription_capacity=0.65,
        economy_efficiency=1.10, corruption_floor=0.02, resource_income_mult=1.05,
        factory_output_mult=1.05, trade_efficiency=1.15,
        unrest_tolerance=0.7, propaganda_effectiveness=0.75, trust_generation=1.4,
        civil_liberties=0.85, political_stability=0.8,
        research_speed_mult=1.10, innovation_capacity=1.15,
        counter_intel_mult=0.85, internal_security=0.75, espionage_offense=0.85,
    ),

    GovernmentType.REPUBLIC: GovernmentModifiers(
        name="Republic",
        description="Elected government with checks and balances. Jack of all trades.",
        combat_effectiveness=1.0, production_cost_mult=1.0, production_speed_mult=1.0,
        morale_base=0.03, conscription_capacity=0.7,
        economy_efficiency=1.05, corruption_floor=0.05, resource_income_mult=1.0,
        factory_output_mult=1.0, trade_efficiency=1.05,
        unrest_tolerance=0.65, propaganda_effectiveness=0.8, trust_generation=1.1,
        civil_liberties=0.7, political_stability=0.65,
        research_speed_mult=1.05, innovation_capacity=1.05,
        counter_intel_mult=0.9, internal_security=0.8, espionage_offense=0.95,
    ),

    GovernmentType.MANAGED_DEMOCRACY: GovernmentModifiers(
        name="Managed Democracy",
        description="Elections exist but outcomes are controlled. State media, oligarch networks.",
        combat_effectiveness=1.05, production_cost_mult=0.95, production_speed_mult=1.0,
        morale_base=0.0, conscription_capacity=0.8,
        economy_efficiency=0.95, corruption_floor=0.10, resource_income_mult=0.95,
        factory_output_mult=0.95, trade_efficiency=0.9,
        unrest_tolerance=0.8, propaganda_effectiveness=1.1, trust_generation=0.8,
        civil_liberties=0.4, political_stability=0.7,
        research_speed_mult=0.95, innovation_capacity=0.9,
        counter_intel_mult=1.1, internal_security=1.1, espionage_offense=1.1,
        censorship=True,
    ),

    GovernmentType.OLIGARCHY: GovernmentModifiers(
        name="Oligarchy",
        description="Rule by wealthy elites. Efficient extraction but rampant corruption.",
        combat_effectiveness=0.9, production_cost_mult=0.9, production_speed_mult=0.95,
        morale_base=-0.05, conscription_capacity=0.75,
        economy_efficiency=0.90, corruption_floor=0.15, resource_income_mult=1.1,
        factory_output_mult=0.9, trade_efficiency=1.1,
        unrest_tolerance=0.7, propaganda_effectiveness=0.85, trust_generation=0.6,
        civil_liberties=0.35, political_stability=0.5,
        research_speed_mult=0.90, innovation_capacity=0.85,
        counter_intel_mult=0.95, internal_security=0.9, espionage_offense=1.0,
    ),

    GovernmentType.MILITARY_JUNTA: GovernmentModifiers(
        name="Military Junta",
        description="Officers rule. Military is priority. Economy suffers. Loyalty through discipline.",
        combat_effectiveness=1.15, production_cost_mult=0.85, production_speed_mult=1.1,
        morale_base=0.05, conscription_capacity=0.9,
        economy_efficiency=0.80, corruption_floor=0.12, resource_income_mult=0.85,
        factory_output_mult=0.85, trade_efficiency=0.7,
        unrest_tolerance=0.85, propaganda_effectiveness=0.9, trust_generation=0.5,
        civil_liberties=0.15, political_stability=0.6,
        research_speed_mult=0.85, innovation_capacity=0.75,
        counter_intel_mult=1.15, internal_security=1.2, espionage_offense=1.05,
        secret_police=True,
    ),

    GovernmentType.ONE_PARTY_STATE: GovernmentModifiers(
        name="One-Party State",
        description="Single ruling party controls all institutions. Efficient mobilization, stifled dissent.",
        combat_effectiveness=1.05, production_cost_mult=0.9, production_speed_mult=1.05,
        morale_base=0.0, conscription_capacity=0.85,
        economy_efficiency=0.85, corruption_floor=0.10, resource_income_mult=0.9,
        factory_output_mult=0.95, trade_efficiency=0.8,
        unrest_tolerance=0.9, propaganda_effectiveness=1.15, trust_generation=0.6,
        civil_liberties=0.2, political_stability=0.75,
        research_speed_mult=0.90, innovation_capacity=0.8,
        counter_intel_mult=1.1, internal_security=1.15, espionage_offense=1.1,
        censorship=True, state_planning=True,
    ),

    GovernmentType.AUTHORITARIAN: GovernmentModifiers(
        name="Authoritarian",
        description="Strongman rule. Personal loyalty networks. Efficient but brittle.",
        combat_effectiveness=1.10, production_cost_mult=0.9, production_speed_mult=1.0,
        morale_base=-0.02, conscription_capacity=0.85,
        economy_efficiency=0.85, corruption_floor=0.12, resource_income_mult=0.9,
        factory_output_mult=0.9, trade_efficiency=0.75,
        unrest_tolerance=0.85, propaganda_effectiveness=1.0, trust_generation=0.5,
        civil_liberties=0.15, political_stability=0.55,
        research_speed_mult=0.85, innovation_capacity=0.8,
        counter_intel_mult=1.1, internal_security=1.15, espionage_offense=1.0,
        secret_police=True,
    ),

    GovernmentType.THEOCRACY: GovernmentModifiers(
        name="Theocracy",
        description="Religious law governs. High morale from zealotry but poor economic management.",
        combat_effectiveness=1.0, production_cost_mult=1.0, production_speed_mult=0.9,
        morale_base=0.12, conscription_capacity=0.75,
        economy_efficiency=0.75, corruption_floor=0.08, resource_income_mult=0.85,
        factory_output_mult=0.8, trade_efficiency=0.65,
        unrest_tolerance=0.75, propaganda_effectiveness=1.2, trust_generation=0.7,
        civil_liberties=0.1, political_stability=0.7,
        research_speed_mult=0.70, innovation_capacity=0.6,
        counter_intel_mult=1.0, internal_security=1.1, espionage_offense=0.85,
        censorship=True,
    ),

    GovernmentType.FASCIST: GovernmentModifiers(
        name="Fascist",
        description="Ultranationalist corporatism. Aggressive military, nationalized industry, total war.",
        combat_effectiveness=1.15, production_cost_mult=0.85, production_speed_mult=1.15,
        morale_base=0.08, conscription_capacity=0.95,
        economy_efficiency=0.90, corruption_floor=0.10, resource_income_mult=0.95,
        factory_output_mult=1.05, trade_efficiency=0.6,
        unrest_tolerance=0.9, propaganda_effectiveness=1.25, trust_generation=0.4,
        civil_liberties=0.05, political_stability=0.65,
        research_speed_mult=0.95, innovation_capacity=0.9,
        counter_intel_mult=1.2, internal_security=1.3, espionage_offense=1.15,
        censorship=True, secret_police=True, forced_labor=True,
    ),

    # ═══════════════════════════════════════════════════════════════════ #
    # EURASIA — Neo-Bolshevist Communist State                           #
    # Central planning, committee bureaucracy, mass warfare doctrine      #
    # Strong military numbers but slow decision-making                    #
    # ═══════════════════════════════════════════════════════════════════ #
    GovernmentType.COMMUNIST: GovernmentModifiers(
        name="Neo-Bolshevist State",
        description="Marxist-Leninist state control. Central planning, mass production, committee decision-making. "
                    "Eurasia's government: quantity over quality, ideological fervor, endemic corruption.",
        combat_effectiveness=1.10, production_cost_mult=0.80, production_speed_mult=1.10,
        morale_base=0.05, conscription_capacity=0.95,
        economy_efficiency=0.80, corruption_floor=0.12, resource_income_mult=0.85,
        factory_output_mult=1.0, trade_efficiency=0.5,
        unrest_tolerance=0.9, propaganda_effectiveness=1.15, trust_generation=0.45,
        civil_liberties=0.05, political_stability=0.7,
        research_speed_mult=0.85, innovation_capacity=0.75,
        counter_intel_mult=1.15, internal_security=1.2, espionage_offense=1.2,
        censorship=True, secret_police=True, state_planning=True, forced_labor=True,
    ),

    # ═══════════════════════════════════════════════════════════════════ #
    # OCEANIA — Ingsoc Totalitarian State                                #
    # Complete state control, Thought Police, doublethink, Newspeak       #
    # Maximum internal control but innovation and economy suffer          #
    # The Party sees everything, controls everything, owns everything     #
    # ═══════════════════════════════════════════════════════════════════ #
    GovernmentType.TOTALITARIAN: GovernmentModifiers(
        name="Ingsoc Totalitarian State",
        description="Complete state control over all aspects of life. Thought Police, doublethink, Newspeak. "
                    "Oceania's government: absolute loyalty through fear, total surveillance, perpetual war. "
                    "The economy exists to serve the Party. Innovation is thoughtcrime.",
        combat_effectiveness=1.10, production_cost_mult=0.80, production_speed_mult=1.15,
        morale_base=0.0, conscription_capacity=1.0,
        economy_efficiency=0.70, corruption_floor=0.08, resource_income_mult=0.80,
        factory_output_mult=0.90, trade_efficiency=0.3,
        unrest_tolerance=0.95, propaganda_effectiveness=1.35, trust_generation=0.3,
        civil_liberties=0.0, political_stability=0.85,
        research_speed_mult=0.80, innovation_capacity=0.65,
        counter_intel_mult=1.30, internal_security=1.40, espionage_offense=1.25,
        censorship=True, secret_police=True, state_planning=True, forced_labor=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Conversion to CowExternalModifiers                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def government_to_cow_modifiers(gov_type: GovernmentType) -> "CowExternalModifiers":
    """Convert government type to CowExternalModifiers for military integration."""
    CowMods = _get_cow_modifiers_class()
    gm = GOVERNMENT_MODIFIERS[gov_type]

    return CowMods(
        production_cost_mult=gm.production_cost_mult,
        production_speed_mult=gm.production_speed_mult,
        combat_effectiveness=gm.combat_effectiveness,
        morale_mod=gm.morale_base,
        resource_income_mult=gm.resource_income_mult,
        org_factor=gm.political_stability,
        hazard_resistance=0.1 if gm.secret_police else 0.0,
        exhaustion_rate_mod=-0.05 if gm.forced_labor else 0.0,
        cohesion_mod=0.1 if gm.censorship else 0.0,
        propaganda_mod=gm.propaganda_effectiveness - 1.0,
    )


def get_government_modifiers(gov_type: GovernmentType) -> GovernmentModifiers:
    """Get the full modifier set for a government type."""
    return GOVERNMENT_MODIFIERS[gov_type]


def government_summary(gov_type: GovernmentType) -> str:
    """One-line summary for turn reports."""
    gm = GOVERNMENT_MODIFIERS[gov_type]
    parts = [gm.name]
    if gm.combat_effectiveness > 1.05:
        parts.append(f"combat +{(gm.combat_effectiveness-1)*100:.0f}%")
    if gm.economy_efficiency < 0.85:
        parts.append(f"economy {(gm.economy_efficiency-1)*100:.0f}%")
    if gm.research_speed_mult < 0.9:
        parts.append(f"research {(gm.research_speed_mult-1)*100:.0f}%")
    if gm.propaganda_effectiveness > 1.1:
        parts.append(f"propaganda +{(gm.propaganda_effectiveness-1)*100:.0f}%")
    if gm.internal_security > 1.2:
        parts.append("secret police")
    return " | ".join(parts)
