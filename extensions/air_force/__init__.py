"""
Air Force Extension for GRAVITAS Engine.

Strategic air warfare system with:
  - 10 aircraft types across 5 roles (fighter, bomber, CAS, recon, transport)
  - Air zone control (superiority / contested / denied)
  - Air combat (BVR + dogfight phases)
  - Strategic bombing campaigns (factories, infrastructure, population)
  - Close air support for ground combat
  - Naval air operations (carrier strikes, anti-ship, convoy cover)
  - Airborne operations (paratroop drops)
  - Air supply (Berlin Airlift style)
  - Radar and early warning systems
  - Anti-exploitation: fuel, crew fatigue, sortie limits, weather, radar detection
"""

from .air_state import (
    AircraftType, AircraftRole, AirZoneControl,
    AircraftSquadron, AirWing, AirZone, RadarStation,
    AirWorld, N_AIRCRAFT_TYPES, AIRCRAFT_STATS,
)
from .air_combat import (
    resolve_air_battle, compute_air_detection,
    strategic_bombing, close_air_support,
    anti_ship_strike, airborne_drop,
)
from .air_operations import (
    AirAction, apply_air_action,
    step_air, initialize_air,
    air_obs, air_obs_size,
)

__all__ = [
    "AircraftType", "AircraftRole", "AirZoneControl",
    "AircraftSquadron", "AirWing", "AirZone", "RadarStation",
    "AirWorld", "N_AIRCRAFT_TYPES", "AIRCRAFT_STATS",
    "resolve_air_battle", "compute_air_detection",
    "strategic_bombing", "close_air_support",
    "anti_ship_strike", "airborne_drop",
    "AirAction", "apply_air_action",
    "step_air", "initialize_air",
    "air_obs", "air_obs_size",
]
