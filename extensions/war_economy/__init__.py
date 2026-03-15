"""
War Economy & Diplomacy Extension for GRAVITAS Engine.

Adds strategic resource management, trade agreements, lend-lease aid,
factory conversion pipelines, war bonds, and economic sanctions to the
existing economy and diplomacy systems.

Integrates with:
  - gravitas_engine.systems.economy   (extends GDP/Unemployment/Debt/Industry)
  - gravitas_engine.systems.diplomacy (alliance matrix drives trade routes)
  - extensions.military               (factory conversion feeds unit production)

Usage:
    from extensions.war_economy import WarEconomyState, step_war_economy
"""

from .war_economy_state import (
    EconSector,
    Resource,
    TradeAgreement,
    LendLeasePackage,
    ClusterEconomy,
    FactionEconomy,
    WarEconomyWorld,
    N_SECTORS,
    N_RESOURCES,
    SECTOR_INPUTS,
    SECTOR_OUTPUTS,
    # Backward-compatible aliases
    StrategicResource,
    ClusterWarEconomy,
    FactionWarEconomy,
)
from .war_economy_dynamics import (
    step_war_economy,
    initialize_war_economy,
    leontief_produce,
    compute_feedback,
)
from .war_economy_actions import (
    WarEconomyAction,
    apply_war_economy_action,
    war_economy_obs,
    war_economy_obs_size,
)
from .manpower import (
    ConscriptionLaw,
    RegimeType,
    SkillLevel,
    MilitaryTraining,
    TrainingBatch,
    ClusterManpower,
    FactionManpowerPolicy,
    ManpowerAction,
    step_manpower,
    apply_manpower_action,
    initialize_manpower,
    manpower_obs,
    manpower_obs_size,
)

__all__ = [
    # State
    "EconSector", "Resource", "TradeAgreement", "LendLeasePackage",
    "ClusterEconomy", "FactionEconomy", "WarEconomyWorld",
    "N_SECTORS", "N_RESOURCES", "SECTOR_INPUTS", "SECTOR_OUTPUTS",
    # Backward compat
    "StrategicResource", "ClusterWarEconomy", "FactionWarEconomy",
    # Dynamics
    "step_war_economy", "initialize_war_economy", "leontief_produce", "compute_feedback",
    # Actions
    "WarEconomyAction", "apply_war_economy_action", "war_economy_obs", "war_economy_obs_size",
    # Manpower
    "ConscriptionLaw", "RegimeType", "SkillLevel", "MilitaryTraining",
    "TrainingBatch", "ClusterManpower", "FactionManpowerPolicy",
    "ManpowerAction", "step_manpower", "apply_manpower_action",
    "initialize_manpower", "manpower_obs", "manpower_obs_size",
]
