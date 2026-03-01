"""
Military Extension for GravitasEngine

This extension adds tactical military operations to the GravitasEngine simulation,
building on the existing military presence system with:

1. Unit System: Separate military units with hit points and capabilities
2. Movement: Tactical movement between clusters
3. Strategy/Tactics: Basic AI for unit behavior
4. Winning Conditions: Objective-based victory system
5. Reward Functions: Military objective completion rewards

The military extension works alongside the existing military presence system,
adding tactical depth while maintaining compatibility with core dynamics.
"""

from .military_dynamics import (
    step_military_units, apply_military_action,
    compute_military_reward, check_victory_conditions
)
from .military_wrapper import MilitaryWrapper
from .advanced_wrapper import AdvancedMilitaryWrapper, AdvancedMilitaryActionSpace
from .advanced_tactics import (
    AdvancedTacticsEngine, FormationType, CommandRank,
    IntelligenceType, CombatTactic, SupplyType,
    UnitFormation, CommandStructure, IntelligenceReport,
    TacticalOperation, ElectronicWarfareState, SupplyChain,
    AdvancedClusterMilitaryState, AdvancedWorldMilitaryState,
    EWCapability
)
from .unit_types import (
    DamageType, MilitaryUnitType, UnitRole, get_unit_role,
    CombatMatrix, ExpandedUnitParams, SupportCompanyType,
    SupportCompany, EnhancedMilitaryUnit, calculate_damage,
    resolve_enhanced_combat
)
from .military_state import (
    StandardizedUnitParams, StandardizedMilitaryUnit,
    StandardizedClusterMilitaryState, StandardizedWorldMilitaryState,
    StandardizedMilitaryObjective, calculate_standardized_damage,
    resolve_standardized_combat, initialize_standardized_military_state,
    create_standardized_unit, create_infantry_division, create_armored_division
)
from .military_extensions import (
    # Combat systems
    FactionCombatSystem, CombatResult, type_advantage,
    # Morale system
    MoraleSystem,
    ROUTE_THRESHOLD, RALLY_THRESHOLD, MORALE_CASCADE_FACTOR, MORALE_CASCADE_MAX,
    # Fog of war
    FogOfWarSystem,
    # Cluster control
    ClusterControlSystem,
    CONTROL_DOMINANCE_THRESHOLD,
    # Electronic warfare execution
    EWExecutionEngine,
    # Supply interdiction
    SupplyInterdictionSystem,
    # All-in-one step
    step_military_extensions,
)

__all__ = [
    "step_military_units", "apply_military_action",
    "compute_military_reward", "check_victory_conditions",
    "MilitaryWrapper",
    "AdvancedMilitaryWrapper", "AdvancedMilitaryActionSpace",

    # Advanced tactics
    "AdvancedTacticsEngine", "FormationType", "CommandRank",
    "IntelligenceType", "CombatTactic", "SupplyType",
    "UnitFormation", "CommandStructure", "IntelligenceReport",
    "TacticalOperation", "ElectronicWarfareState", "SupplyChain",
    "AdvancedClusterMilitaryState", "AdvancedWorldMilitaryState",
    "EWCapability",

    # Unit types and combat
    "DamageType", "MilitaryUnitType", "UnitRole", "get_unit_role",
    "CombatMatrix", "ExpandedUnitParams", "SupportCompanyType",
    "SupportCompany", "EnhancedMilitaryUnit", "calculate_damage",
    "resolve_enhanced_combat",

    # Standardized system
    "StandardizedUnitParams", "StandardizedMilitaryUnit",
    "StandardizedClusterMilitaryState", "StandardizedWorldMilitaryState",
    "StandardizedMilitaryObjective", "calculate_standardized_damage",
    "resolve_standardized_combat", "initialize_standardized_military_state",
    "create_standardized_unit", "create_infantry_division", "create_armored_division",

    # Extension systems
    "FactionCombatSystem", "CombatResult", "type_advantage",
    "MoraleSystem",
    "ROUTE_THRESHOLD", "RALLY_THRESHOLD", "MORALE_CASCADE_FACTOR",
    "FogOfWarSystem",
    "ClusterControlSystem", "CONTROL_DOMINANCE_THRESHOLD",
    "EWExecutionEngine",
    "SupplyInterdictionSystem",
    "step_military_extensions",
]
