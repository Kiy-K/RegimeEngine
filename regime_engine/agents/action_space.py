"""
Agent action space definition.

Agents interact with the regime engine by submitting Action objects that
perturb faction-level micro-state variables.  Actions are additive impulses
applied each step before the integrator runs.

Action taxonomy:

  STABILITY_OPERATION   — increase cohesion of a target faction
  SUPPRESSION           — decrease radicalization of a target faction (by force)
  LEGITIMACY_CAMPAIGN   — increase cohesion + decrease radicalization (soft)
  POWER_REDISTRIBUTION  — transfer power share from target to neutral pool
  RADICALIZE            — increase radicalization of a target faction
  COOPT                 — increase power of a target faction
  MEMORY_RELIEF         — decrease grievance memory of a target faction
  PROVOKE               — increase memory of a target faction

All actions specify:
  - action_type: Which kind of operation to apply.
  - actor_idx:   The faction paying for the action.
  - target_idx:  The faction receiving the action.
  - intensity:   Magnitude of the impulse ∈ [0, 0.2].
  - agent_id:    The agent issuing the action (string identifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.state import FactionState, RegimeState


@unique
class ActionType(IntEnum):
    """Enumeration of all valid action types."""

    STABILITY_OPERATION = 0
    SUPPRESSION = 1
    LEGITIMACY_CAMPAIGN = 2
    POWER_REDISTRIBUTION = 3
    RADICALIZE = 4
    COOPT = 5
    MEMORY_RELIEF = 6
    PROVOKE = 7


# Maximum impulse magnitude
MAX_INTENSITY: float = 0.20


@dataclass(frozen=True)
class Action:
    """A single agent action targeting one faction and paid by another.

    Attributes:
        action_type: Which kind of operation to apply.
        actor_idx:   0-indexed acting faction (pays cost).
        target_idx:  0-indexed target faction.
        intensity:   Impulse magnitude ∈ [0, MAX_INTENSITY].
        agent_id:    String identifier of the issuing agent.
    """

    action_type: ActionType
    actor_idx: int
    target_idx: int
    intensity: float
    agent_id: str

    def __post_init__(self) -> None:
        if self.actor_idx < 0:
            raise ValueError(f"actor_idx must be >= 0, got {self.actor_idx}")
        if self.target_idx < 0:
            raise ValueError(f"target_idx must be >= 0, got {self.target_idx}")
        if not 0.0 <= self.intensity <= MAX_INTENSITY:
            raise ValueError(f"intensity must be in [0, {MAX_INTENSITY}]")
        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")


def apply_action(state: RegimeState, action: Action) -> RegimeState:
    """Apply a single action impulse, deducting wealth from the actor.

    Action effects scale down if wealth is insufficient. Cost = intensity * 0.5.
    """
    if action.actor_idx >= state.n_factions or action.target_idx >= state.n_factions:
        raise ValueError("Faction indices out of range.")

    factions = list(state.factions)
    actor = factions[action.actor_idx]
    
    cost = action.intensity * 0.5
    if actor.wealth < cost and cost > 0:
        effective_mag = action.intensity * (actor.wealth / cost)
        actor = actor.copy_with(wealth=0.0)
    else:
        effective_mag = action.intensity
        actor = actor.copy_with(wealth=actor.wealth - cost)
        
    factions[action.actor_idx] = actor
    
    # If targeting self, grab the updated actor
    target = factions[action.target_idx]
    mag = effective_mag

    if action.action_type == ActionType.STABILITY_OPERATION:
        target = target.copy_with(cohesion=min(1.0, target.cohesion + mag))
    elif action.action_type == ActionType.SUPPRESSION:
        target = target.copy_with(radicalization=max(0.0, target.radicalization - mag))
    elif action.action_type == ActionType.LEGITIMACY_CAMPAIGN:
        half = mag * 0.5
        target = target.copy_with(
            cohesion=min(1.0, target.cohesion + half),
            radicalization=max(0.0, target.radicalization - half),
        )
    elif action.action_type == ActionType.POWER_REDISTRIBUTION:
        factions[action.target_idx] = target
        factions = _redistribute_power(factions, action.target_idx, -mag)
        return state.copy_with_factions(factions)
    elif action.action_type == ActionType.RADICALIZE:
        target = target.copy_with(radicalization=min(1.0, target.radicalization + mag))
    elif action.action_type == ActionType.COOPT:
        factions[action.target_idx] = target
        factions = _redistribute_power(factions, action.target_idx, +mag)
        return state.copy_with_factions(factions)
    elif action.action_type == ActionType.MEMORY_RELIEF:
        target = target.copy_with(memory=max(0.0, target.memory - mag))
    elif action.action_type == ActionType.PROVOKE:
        target = target.copy_with(memory=min(1.0, target.memory + mag))

    factions[action.target_idx] = target
    return state.copy_with_factions(factions)


def apply_actions(state: RegimeState, actions: List[Action]) -> RegimeState:
    current = state
    for action in actions:
        current = apply_action(current, action)
    return current


def _redistribute_power(
    factions: List[FactionState],
    target_idx: int,
    delta: float,
) -> List[FactionState]:
    n = len(factions)
    powers = np.array([f.power for f in factions], dtype=np.float64)
    compensation = -delta / max(1, n - 1)

    for i in range(n):
        if i == target_idx:
            powers[i] = float(np.clip(powers[i] + delta, 0.0, 1.0))
        else:
            powers[i] = float(np.clip(powers[i] + compensation, 0.0, 1.0))

    total = float(np.sum(powers))
    if total > 0.0:
        powers = powers / total
    else:
        powers = np.full(n, 1.0 / n)

    return [f.copy_with(power=float(powers[i])) for i, f in enumerate(factions)]


def null_action(actor_idx: int, target_idx: int, agent_id: str) -> Action:
    return Action(
        action_type=ActionType.STABILITY_OPERATION,
        actor_idx=actor_idx,
        target_idx=target_idx,
        intensity=0.0,
        agent_id=agent_id,
    )


def action_space_size() -> int:
    return len(ActionType)