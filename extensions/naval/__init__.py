"""
Naval Warfare Extension for GRAVITAS Engine.

Realistic strategic naval system with:
  - 14 ship classes across 5 categories (capital, escort, submarine, auxiliary, amphibious)
  - Sea zone control with contested/controlled/denied states
  - Lanchester-law naval combat with range brackets and initiative
  - Convoy escort system with wolf pack mechanics
  - Amphibious operations with beach assault penalties
  - Mine warfare (offensive + defensive)
  - Shore bombardment
  - Blockade and sea denial
  - Anti-exploitation: fuel consumption, crew fatigue, repair time, detection mechanics
"""

from .naval_state import (
    ShipClass, ShipCategory, SeaZoneControl,
    ShipInstance, Fleet, SeaZone, ConvoyRoute,
    MineField, NavalWorld,
    N_SHIP_CLASSES, SHIP_STATS,
)
from .naval_combat import (
    resolve_naval_battle, compute_detection,
    shore_bombardment, anti_submarine_warfare,
)
from .naval_operations import (
    NavalAction, apply_naval_action,
    step_naval, initialize_naval,
    run_convoys, attempt_amphibious_landing,
    enforce_blockade, lay_mines, sweep_mines,
    naval_obs, naval_obs_size,
)

__all__ = [
    "ShipClass", "ShipCategory", "SeaZoneControl",
    "ShipInstance", "Fleet", "SeaZone", "ConvoyRoute",
    "MineField", "NavalWorld",
    "N_SHIP_CLASSES", "SHIP_STATS",
    "resolve_naval_battle", "compute_detection",
    "shore_bombardment", "anti_submarine_warfare",
    "NavalAction", "apply_naval_action",
    "step_naval", "initialize_naval",
    "run_convoys", "attempt_amphibious_landing",
    "enforce_blockade", "lay_mines", "sweep_mines",
    "naval_obs", "naval_obs_size",
]
