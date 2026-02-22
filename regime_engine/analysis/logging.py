"""
Structured state logger.

Provides StateLogger — a lightweight observer that records RegimeState objects
into an in-memory list during simulation.  Supports serialisation to
list-of-dicts for downstream persistence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.state import RegimeState


class StateLogger:
    """Records RegimeState objects produced during a simulation run.

    Intended for use as a post-step hook with the Scheduler:

        logger = StateLogger()
        scheduler.register_post_hook(lambda before, after: logger.record(after))

    Attributes:
        max_records: Maximum number of states to retain (None = unlimited).
    """

    def __init__(self, max_records: Optional[int] = None) -> None:
        """Initialise an empty logger.

        Args:
            max_records: If set, older records are discarded when the buffer
                         exceeds this limit (FIFO).
        """
        if max_records is not None and max_records <= 0:
            raise ValueError(
                f"max_records must be > 0 or None, got {max_records}"
            )
        self.max_records: Optional[int] = max_records
        self._records: List[RegimeState] = []

    def record(self, state: RegimeState) -> None:
        """Append a state to the log.

        Args:
            state: RegimeState to record.
        """
        self._records.append(state)
        if self.max_records is not None and len(self._records) > self.max_records:
            self._records.pop(0)  # FIFO eviction

    def records(self) -> List[RegimeState]:
        """Return all recorded states (read-only view via copy).

        Returns:
            List of RegimeState objects in chronological order.
        """
        return list(self._records)

    def clear(self) -> None:
        """Empty the log."""
        self._records = []

    def __len__(self) -> int:
        """Number of recorded states."""
        return len(self._records)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Serialise all recorded states to a list of plain dictionaries.

        Returns:
            List of dicts as produced by RegimeState.to_dict().
        """
        return [s.to_dict() for s in self._records]

    def macro_series(self) -> Dict[str, List[float]]:
        """Return time-series of each macro variable as lists.

        Returns:
            Dictionary mapping macro variable name to list of values.
        """
        if not self._records:
            return {
                "legitimacy": [],
                "cohesion": [],
                "fragmentation": [],
                "instability": [],
                "mobilization": [],
                "repression": [],
                "elite_alignment": [],
                "volatility": [],
                "exhaustion": [],
            }
        return {
            "legitimacy": [s.system.legitimacy for s in self._records],
            "cohesion": [s.system.cohesion for s in self._records],
            "fragmentation": [s.system.fragmentation for s in self._records],
            "instability": [s.system.instability for s in self._records],
            "mobilization": [s.system.mobilization for s in self._records],
            "repression": [s.system.repression for s in self._records],
            "elite_alignment": [s.system.elite_alignment for s in self._records],
            "volatility": [s.system.volatility for s in self._records],
            "exhaustion": [s.system.exhaustion for s in self._records],
        }

    def faction_series(self, faction_idx: int) -> Dict[str, List[float]]:
        """Return time-series of micro-state for a specific faction.

        Args:
            faction_idx: 0-indexed faction identifier.

        Returns:
            Dictionary mapping variable name to list of values.
        """
        if not self._records:
            return {"power": [], "radicalization": [], "cohesion": [], "memory": []}
        if faction_idx >= self._records[0].n_factions:
            raise IndexError(
                f"faction_idx {faction_idx} out of range "
                f"(n_factions={self._records[0].n_factions})"
            )
        return {
            "power": [s.factions[faction_idx].power for s in self._records],
            "radicalization": [
                s.factions[faction_idx].radicalization for s in self._records
            ],
            "cohesion": [s.factions[faction_idx].cohesion for s in self._records],
            "memory": [s.factions[faction_idx].memory for s in self._records],
        }
