"""
State containers for the Adaptive Memory-Driven Regime Engine.

Two tiers of immutable state:
  - FactionState  : micro-level, one per faction (P, Rad, Coh, Mem, Wealth)
  - SystemState   : macro-level, 10 scalars + pillars array
  - RegimeState   : top-level container binding both tiers + step counter + affinity matrix

All fields are strictly clamped at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
# Micro state                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FactionState:
    """Immutable state for a single political faction.

    Attributes:
        power:          Normalised power share P_i ∈ [0, 1].
        radicalization: Ideological extremism Rad_i ∈ [0, 1].
        cohesion:       Internal organisational coherence Coh_i ∈ [0, 1].
        memory:         Accumulated grievance memory Mem_i ∈ [0, 1].
        wealth:         Accumulated capital/resource Wealth_i ∈ [0, 1].
    """

    power: float
    radicalization: float
    cohesion: float
    memory: float
    wealth: float = 0.5

    def __post_init__(self) -> None:
        """Reject any out-of-range values at construction time."""
        fields = {
            "power": self.power,
            "radicalization": self.radicalization,
            "cohesion": self.cohesion,
            "memory": self.memory,
            "wealth": self.wealth,
        }
        for name, value in fields.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"FactionState.{name} must be in [0, 1], got {value}"
                )

    def to_array(self) -> NDArray[np.float64]:
        """Return variables as float64 array of shape (5,)."""
        return np.array(
            [self.power, self.radicalization, self.cohesion, self.memory, self.wealth],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "FactionState":
        """Construct from a length-5 float64 array, clamping to [0, 1]."""
        if arr.shape != (5,):
            raise ValueError(f"FactionState.from_array expects shape (5,), got {arr.shape}")
        arr_c = np.clip(arr, 0.0, 1.0)
        return cls(
            power=float(arr_c[0]),
            radicalization=float(arr_c[1]),
            cohesion=float(arr_c[2]),
            memory=float(arr_c[3]),
            wealth=float(arr_c[4]),
        )

    def copy_with(self, **kwargs: float) -> "FactionState":
        """Return a new FactionState with selected fields overridden."""
        current = {
            "power": self.power,
            "radicalization": self.radicalization,
            "cohesion": self.cohesion,
            "memory": self.memory,
            "wealth": self.wealth,
        }
        current.update(kwargs)
        return FactionState(**current)

    def to_dict(self) -> Dict[str, float]:
        """Serialise to plain dictionary."""
        return {
            "power": self.power,
            "radicalization": self.radicalization,
            "cohesion": self.cohesion,
            "memory": self.memory,
            "wealth": self.wealth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "FactionState":
        """Deserialise from plain dictionary."""
        return cls(
            power=data["power"],
            radicalization=data["radicalization"],
            cohesion=data["cohesion"],
            memory=data["memory"],
            wealth=data.get("wealth", 0.5),
        )


# --------------------------------------------------------------------------- #
# Macro state                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SystemState:
    """Immutable system-level macro variables.

    Attributes:
        legitimacy:      L   — power-weighted average cohesion.
        cohesion:        C   — power-weighted squared cohesion (≠ L).
        fragmentation:   F   — Gini-based power dispersion ∈ [0, 1).
        instability:     I   — power-weighted radicalization × (1−cohesion).
        mobilization:    M   — power-weighted memory × radicalization.
        repression:      R   — 1 − L (inverse legitimacy proxy).
        elite_alignment: E   — L × (1 − F).
        volatility:      V   — tanh-bounded cascade measure.
        exhaustion:      Exh — time-integrated societal fatigue.
        state_gdp:       GDP — health of national economy ∈ [0, 1].
        pillars:         Control level of N functional institutions.
    """

    legitimacy: float
    cohesion: float
    fragmentation: float
    instability: float
    mobilization: float
    repression: float
    elite_alignment: float
    volatility: float
    exhaustion: float
    state_gdp: float = 0.5
    pillars: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Reject any out-of-range values at construction time."""
        fields = {
            "legitimacy": self.legitimacy,
            "cohesion": self.cohesion,
            "fragmentation": self.fragmentation,
            "instability": self.instability,
            "mobilization": self.mobilization,
            "repression": self.repression,
            "elite_alignment": self.elite_alignment,
            "volatility": self.volatility,
            "exhaustion": self.exhaustion,
            "state_gdp": self.state_gdp,
        }
        for name, value in fields.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"SystemState.{name} must be in [0, 1], got {value}"
                )
        for i, val in enumerate(self.pillars):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"SystemState.pillars[{i}] must be in [0, 1], got {val}")

    def to_array(self) -> NDArray[np.float64]:
        """Return all macro scalars followed by pillars as float64 array."""
        base = [
            self.legitimacy,
            self.cohesion,
            self.fragmentation,
            self.instability,
            self.mobilization,
            self.repression,
            self.elite_alignment,
            self.volatility,
            self.exhaustion,
            self.state_gdp,
        ]
        return np.array(base + list(self.pillars), dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "SystemState":
        """Construct from a float64 array, clamping to [0, 1]."""
        arr_c = np.clip(arr, 0.0, 1.0)
        return cls(
            legitimacy=float(arr_c[0]),
            cohesion=float(arr_c[1]),
            fragmentation=float(arr_c[2]),
            instability=float(arr_c[3]),
            mobilization=float(arr_c[4]),
            repression=float(arr_c[5]),
            elite_alignment=float(arr_c[6]),
            volatility=float(arr_c[7]),
            exhaustion=float(arr_c[8]),
            state_gdp=float(arr_c[9]),
            pillars=tuple(float(x) for x in arr_c[10:]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "legitimacy": self.legitimacy,
            "cohesion": self.cohesion,
            "fragmentation": self.fragmentation,
            "instability": self.instability,
            "mobilization": self.mobilization,
            "repression": self.repression,
            "elite_alignment": self.elite_alignment,
            "volatility": self.volatility,
            "exhaustion": self.exhaustion,
            "state_gdp": self.state_gdp,
            "pillars": list(self.pillars),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemState":
        """Deserialise from plain dictionary."""
        return cls(
            legitimacy=data["legitimacy"],
            cohesion=data["cohesion"],
            fragmentation=data["fragmentation"],
            instability=data["instability"],
            mobilization=data["mobilization"],
            repression=data["repression"],
            elite_alignment=data["elite_alignment"],
            volatility=data["volatility"],
            exhaustion=data["exhaustion"],
            state_gdp=data.get("state_gdp", 0.5),
            pillars=tuple(data.get("pillars", [])),
        )

    @classmethod
    def neutral(cls, n_pillars: int = 3) -> "SystemState":
        """Return a low-energy starting macro state."""
        return cls(
            legitimacy=0.60,
            cohesion=0.40,
            fragmentation=0.10,
            instability=0.05,
            mobilization=0.05,
            repression=0.40,
            elite_alignment=0.54,
            volatility=0.05,
            exhaustion=0.00,
            state_gdp=0.50,
            pillars=tuple([0.5] * n_pillars),
        )


# --------------------------------------------------------------------------- #
# Top-level regime state                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegimeState:
    """Complete, immutable state of the regime engine at one timestep.

    Attributes:
        factions:         Ordered list of FactionState objects (2–6 items).
        system:           Derived macro SystemState.
        affinity_matrix:  N x N tuple tracking inter-faction affinity [-1, 1].
        step:             Non-negative integer timestep counter.
        hierarchical:     Optional Phase 3 district/province state (when use_hierarchy).
    """

    factions: List[FactionState] = field(default_factory=list)
    system: SystemState = field(default_factory=SystemState.neutral)
    affinity_matrix: Tuple[Tuple[float, ...], ...] = field(default_factory=tuple)
    step: int = 0
    hierarchical: Optional[Any] = None  # HierarchicalState when Phase 3 enabled

    def __post_init__(self) -> None:
        """Validate structural invariants."""
        if not isinstance(self.factions, list):
            raise TypeError("factions must be a list")
        n = len(self.factions)
        if not 2 <= n <= 6:
            raise ValueError(f"Number of factions must be in [2, 6], got {n}")
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
            
        if self.affinity_matrix:
            if len(self.affinity_matrix) != n:
                raise ValueError(f"Affinity matrix must be {n}x{n}")
            for row in self.affinity_matrix:
                if len(row) != n:
                    raise ValueError(f"Affinity matrix must be {n}x{n}")
                for val in row:
                    if not (-1.0 <= val <= 1.0):
                        raise ValueError(f"Affinity values must be in [-1, 1], got {val}")

    @property
    def n_factions(self) -> int:
        """Number of active factions."""
        return len(self.factions)

    def get_faction_powers(self) -> NDArray[np.float64]:
        return np.array([f.power for f in self.factions], dtype=np.float64)

    def get_faction_radicalizations(self) -> NDArray[np.float64]:
        return np.array([f.radicalization for f in self.factions], dtype=np.float64)

    def get_faction_cohesions(self) -> NDArray[np.float64]:
        return np.array([f.cohesion for f in self.factions], dtype=np.float64)

    def get_faction_memories(self) -> NDArray[np.float64]:
        return np.array([f.memory for f in self.factions], dtype=np.float64)
        
    def get_faction_wealths(self) -> NDArray[np.float64]:
        return np.array([f.wealth for f in self.factions], dtype=np.float64)

    def copy_with_factions(self, factions: List[FactionState]) -> "RegimeState":
        return RegimeState(
            factions=factions,
            system=self.system,
            affinity_matrix=self.affinity_matrix,
            step=self.step,
            hierarchical=self.hierarchical,
        )

    def copy_with_system(self, system: SystemState) -> "RegimeState":
        return RegimeState(
            factions=self.factions,
            system=system,
            affinity_matrix=self.affinity_matrix,
            step=self.step,
            hierarchical=self.hierarchical,
        )

    def copy_with_hierarchical(self, hierarchical: Optional[Any]) -> "RegimeState":
        return RegimeState(
            factions=self.factions,
            system=self.system,
            affinity_matrix=self.affinity_matrix,
            step=self.step,
            hierarchical=hierarchical,
        )

    def advance_step(self) -> "RegimeState":
        return RegimeState(
            factions=self.factions,
            system=self.system,
            affinity_matrix=self.affinity_matrix,
            step=self.step + 1,
            hierarchical=self.hierarchical,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "factions": [f.to_dict() for f in self.factions],
            "system": self.system.to_dict(),
            "affinity_matrix": [list(row) for row in self.affinity_matrix],
            "step": self.step,
        }
        if self.hierarchical is not None:
            out["hierarchical"] = {
                "district_array": self.hierarchical.get_district_array().tolist(),
                "pipeline_buffers": self.hierarchical.pipeline_buffers.tolist(),
            }
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeState":
        factions = [FactionState.from_dict(fd) for fd in data["factions"]]
        system = SystemState.from_dict(data["system"])
        aff_mat = data.get("affinity_matrix", [])
        aff_tuple = tuple(tuple(float(x) for x in row) for row in aff_mat)
        # hierarchical not fully restored from dict (would need topology); omit for now
        return cls(
            factions=factions,
            system=system,
            affinity_matrix=aff_tuple,
            step=data.get("step", 0),
            hierarchical=None,
        )