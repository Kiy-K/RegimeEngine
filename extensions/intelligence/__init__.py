"""
Intelligence, Fog of War & Espionage Extension for GRAVITAS Engine.

Complete intelligence system with:
  - Fog of War: factions only see what they've scouted/detected
  - Espionage: spy networks, infiltration, code-breaking, counter-intelligence
  - Signals Intelligence (SIGINT): radio intercepts, code-breaking
  - Human Intelligence (HUMINT): spy rings, double agents, defectors
  - Reconnaissance: aerial recon, naval patrols, radar
  - Deception: fake armies, radio deception, double agents feeding false intel
  - Counter-intelligence: mole hunting, security sweeps, compartmentalization
"""

from .intel_system import (
    IntelSource, IntelReliability, IntelClassification,
    IntelReport, SpyRing, CodeBreaking, RadarNetwork,
    FactionIntelState, IntelWorld,
    step_intelligence, initialize_intelligence,
    apply_intel_action, IntelAction,
    get_faction_visibility, intel_obs, intel_obs_size,
)

__all__ = [
    "IntelSource", "IntelReliability", "IntelClassification",
    "IntelReport", "SpyRing", "CodeBreaking", "RadarNetwork",
    "FactionIntelState", "IntelWorld",
    "step_intelligence", "initialize_intelligence",
    "apply_intel_action", "IntelAction",
    "get_faction_visibility", "intel_obs", "intel_obs_size",
]
