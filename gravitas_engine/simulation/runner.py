"""
Simulation runner.

Provides SimulationRunner — a high-level orchestrator that initialises the
scheduler, drives the tick loop for N steps, and collects the full trajectory.

Usage:
    from regime_engine.core.parameters import SystemParameters
    from regime_engine.core.factions import create_balanced_factions
    from regime_engine.core.state import RegimeState, SystemState
    from regime_engine.simulation.runner import SimulationRunner

    params = SystemParameters(n_factions=3, max_steps=500)
    factions = create_balanced_factions(3)
    initial_state = RegimeState(factions=factions,
                                system=SystemState.neutral(), step=0)

    runner = SimulationRunner(params=params)
    trajectory = runner.run(initial_state)
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..agents.action_space import Action
from ..core.factions import create_balanced_factions, recompute_system_state
from ..core.parameters import SystemParameters
from ..core.state import RegimeState, SystemState
from ..systems.crisis_classifier import ClassifierThresholds, CrisisLevel, classify
from .scheduler import Scheduler

# Type alias for an agent policy:  policy(state) -> list of actions (possibly empty)
AgentPolicy = Callable[[RegimeState], List[Action]]


class SimulationRunner:
    """Runs a regime simulation from an initial state for N steps.

    Agents are represented as policy functions: callables that receive the
    current RegimeState and return a list of Actions for the next tick.

    Attributes:
        params:           System parameters.
        thresholds:       Crisis classifier thresholds for early stopping.
        stop_on_collapse: Whether to halt when CrisisLevel.COLLAPSE is reached.
    """

    def __init__(
        self,
        params: Optional[SystemParameters] = None,
        thresholds: Optional[ClassifierThresholds] = None,
        stop_on_collapse: bool = True,
    ) -> None:
        """Initialise the runner.

        Args:
            params:           System parameters (defaults to SystemParameters()).
            thresholds:       Crisis thresholds (defaults to ClassifierThresholds()).
            stop_on_collapse: If True, halt simulation when COLLAPSE is detected.
        """
        self.params: SystemParameters = (
            params if params is not None else SystemParameters()
        )
        self.thresholds: ClassifierThresholds = (
            thresholds if thresholds is not None else ClassifierThresholds()
        )
        self.stop_on_collapse: bool = stop_on_collapse

    def run(
        self,
        initial_state: Optional[RegimeState] = None,
        n_steps: Optional[int] = None,
        agent_policies: Optional[List[AgentPolicy]] = None,
    ) -> List[RegimeState]:
        """Run the simulation and return the full state trajectory.

        Args:
            initial_state:   Starting RegimeState.  If None, a balanced
                             n_factions-faction state is created.
            n_steps:         Number of steps to run.  Defaults to params.max_steps.
            agent_policies:  List of policy functions (one per conceptual agent).
                             Each policy is called with the current state and may
                             return an empty list (no-op).

        Returns:
            Ordered list of RegimeState objects including the initial state.
            Length is min(n_steps + 1, collapse_step + 1).
        """
        n_steps = n_steps if n_steps is not None else self.params.max_steps
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")

        # Build initial state
        if initial_state is None:
            factions = create_balanced_factions(self.params.n_factions)
            n = self.params.n_factions
            aff_mat = tuple(tuple(1.0 if i==j else 0.0 for j in range(n)) for i in range(n))
            initial_state = RegimeState(
                factions=factions,
                system=SystemState.neutral(self.params.n_pillars),
                affinity_matrix=aff_mat,
                step=0,
            )
        state = recompute_system_state(initial_state, self.params)

        policies: List[AgentPolicy] = agent_policies or []

        scheduler = Scheduler(params=self.params)
        scheduler.initialise(state)

        trajectory: List[RegimeState] = [scheduler.state]

        for _ in range(n_steps):
            # Collect actions from all policies
            current = scheduler.state
            for policy in policies:
                for action in policy(current):
                    scheduler.collect_action(action)

            new_state = scheduler.tick()
            trajectory.append(new_state)

            # Early termination on collapse
            if self.stop_on_collapse:
                crisis = classify(new_state, self.thresholds)
                if crisis == CrisisLevel.COLLAPSE:
                    break

        return trajectory

    def run_headless(
        self,
        initial_state: Optional[RegimeState] = None,
        n_steps: Optional[int] = None,
        agent_policies: Optional[List[AgentPolicy]] = None,
    ) -> RegimeState:
        """Run and return only the final state (no trajectory stored).

        Memory-efficient alternative to run() for long simulations.

        Args:
            initial_state:  Starting state (None → balanced).
            n_steps:        Number of steps.
            agent_policies: Agent policy functions.

        Returns:
            Final RegimeState.
        """
        n_steps = n_steps if n_steps is not None else self.params.max_steps
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")

        if initial_state is None:
            factions = create_balanced_factions(self.params.n_factions)
            n = self.params.n_factions
            aff_mat = tuple(tuple(1.0 if i==j else 0.0 for j in range(n)) for i in range(n))
            initial_state = RegimeState(
                factions=factions,
                system=SystemState.neutral(self.params.n_pillars),
                affinity_matrix=aff_mat,
                step=0,
            )
        state = recompute_system_state(initial_state, self.params)

        policies: List[AgentPolicy] = agent_policies or []
        scheduler = Scheduler(params=self.params)
        scheduler.initialise(state)

        for _ in range(n_steps):
            current = scheduler.state
            for policy in policies:
                for action in policy(current):
                    scheduler.collect_action(action)
            new_state = scheduler.tick()

            if self.stop_on_collapse:
                crisis = classify(new_state, self.thresholds)
                if crisis == CrisisLevel.COLLAPSE:
                    break

        return scheduler.state
