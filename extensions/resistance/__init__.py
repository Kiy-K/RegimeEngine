"""
Resistance Extension — British Liberation Front (BLF).

Winston Smith escaped Room 101. The Thought Police think he's dead.
He is the Ghost of London, leading the real resistance from the sewers
beneath Victory Mansions.

The BLF operates as underground cells across Oceania's sectors.
When prole unrest reaches critical levels, cells activate and the
resistance escalates from graffiti to sabotage to open revolt.
"""

from .resistance import (
    CellStatus, EscalationLevel, ResistanceCell, WinstonState, BLFState,
    step_resistance, initialize_resistance,
    resistance_event_text, resistance_obs,
    ESCALATION_NAMES,
)

__all__ = [
    "CellStatus", "ResistanceCell", "BLFState",
    "step_resistance", "initialize_resistance",
    "resistance_event_text", "resistance_obs",
]
