"""
Economy V2 — Realistic GDP-based economy with factory vectors, population pools,
and military logistics.

Replaces the simpler war_economy resource model with:
  - 10 factory types per cluster (power, civil, military, dockyard, airfield, etc.)
  - GDP calculation from factory output
  - Population pools (working age, employed, unemployed, military, wounded, POW)
  - Conscription drawing from population with economic penalties
  - Military units with equipment, supply consumption, organization
  - Logistics: factory → depot → front line, infrastructure-dependent
"""

from .economy_core import (
    FactoryType, PopulationPool, ClusterEconomy, FactionEconomy,
    EconomyWorld, initialize_economy_v2, step_economy_v2,
    economy_obs, economy_summary,
)

__all__ = [
    "FactoryType", "PopulationPool", "ClusterEconomy", "FactionEconomy",
    "EconomyWorld", "initialize_economy_v2", "step_economy_v2",
    "economy_obs", "economy_summary",
]
