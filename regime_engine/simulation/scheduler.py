"""
Event-driven simulation scheduler.

The scheduler manages:
  1. Collection of agent actions for the current step.
  2. Application of those actions to the regime state.
  3. Invocation of the RK4 integrator.
  4. Optional event hooks (callbacks) at pre/post integration.

Data flow:
    Agent → Action → Scheduler.collect_action()
    Scheduler.tick() → apply_actions → rk4_step → new RegimeState

The scheduler is stateful across ticks but holds no shared mutable arrays —
all state is carried in the immutable RegimeState chain.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..agents.action_space import Action, apply_actions
from ..core.factions import recompute_system_state
from ..core.integrator import step as rk4_step
from ..core.parameters import SystemParameters
from ..core.state import RegimeState
from ..systems.events import check_and_apply_events

# Type alias for pre/post-step hooks:  hook(state_before, state_after) -> None
StepHook = Callable[[RegimeState, RegimeState], None]


class Scheduler:
    """Collects agent actions and applies them synchronously each tick.

    The scheduler follows a strict per-tick protocol:
      1. Zero or more agents call collect_action() to register their intention.
      2. tick() is called once — it applies all collected actions, integrates,
         clears the action queue, and fires registered hooks.

    Attributes:
        params: System parameters used for integration.
    """

    def __init__(
        self,
        params: SystemParameters,
        pre_step_hooks: Optional[List[StepHook]] = None,
        post_step_hooks: Optional[List[StepHook]] = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            params:           System parameters.
            pre_step_hooks:   Callables invoked with (state_before, state_before)
                              before integration.
            post_step_hooks:  Callables invoked with (state_before, state_after)
                              after integration.
        """
        self.params: SystemParameters = params
        self._pre_hooks: List[StepHook] = list(pre_step_hooks or [])
        self._post_hooks: List[StepHook] = list(post_step_hooks or [])
        self._pending_actions: List[Action] = []
        self._current_state: Optional[RegimeState] = None

    # ------------------------------------------------------------------ #
    # Initialisation                                                        #
    # ------------------------------------------------------------------ #

    def initialise(self, state: RegimeState) -> None:
        """Set the starting regime state.

        Args:
            state: Initial regime state (macro variables must be current).
        """
        self._current_state = recompute_system_state(state, self.params)

    # ------------------------------------------------------------------ #
    # Action collection                                                    #
    # ------------------------------------------------------------------ #

    def collect_action(self, action: Action) -> None:
        """Register an action to be applied on the next tick.

        Multiple actions from different agents for the same step are allowed.
        They are applied in collection order.

        Args:
            action: The action to enqueue.
        """
        if self._current_state is not None:
            if action.faction_idx >= self._current_state.n_factions:
                raise ValueError(
                    f"faction_idx {action.faction_idx} out of range "
                    f"(n_factions={self._current_state.n_factions})"
                )
        self._pending_actions.append(action)

    # ------------------------------------------------------------------ #
    # Tick                                                                  #
    # ------------------------------------------------------------------ #

    def tick(self) -> RegimeState:
        """Apply all pending actions and advance the simulation by one step.

        sequence:
          1. Fire pre-step hooks.
          2. Apply pending actions.
          3. Recompute macro variables.
          4. RK4 integration.
          5. Clear action queue.
          6. Fire post-step hooks.
          7. Return new state.

        Returns:
            The new RegimeState after integration.

        Raises:
            RuntimeError: If the scheduler has not been initialised.
        """
        if self._current_state is None:
            raise RuntimeError(
                "Scheduler not initialised. Call initialise(state) first."
            )

        state_before = self._current_state

        # Pre-step hooks
        for hook in self._pre_hooks:
            hook(state_before, state_before)

        # Apply actions
        state_after_actions = apply_actions(state_before, self._pending_actions)
        state_after_actions = recompute_system_state(state_after_actions, self.params)

        # Integrate and apply exogenous shocks
        state_after = rk4_step(state_after_actions, self.params)
        state_after = check_and_apply_events(state_after, self.params)

        # Clear queue
        self._pending_actions = []

        # Post-step hooks
        for hook in self._post_hooks:
            hook(state_before, state_after)

        self._current_state = state_after
        return self._current_state

    # ------------------------------------------------------------------ #
    # State access                                                         #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> RegimeState:
        """Current regime state.

        Raises:
            RuntimeError: If the scheduler has not been initialised.
        """
        if self._current_state is None:
            raise RuntimeError(
                "Scheduler not initialised. Call initialise(state) first."
            )
        return self._current_state

    @property
    def pending_action_count(self) -> int:
        """Number of actions queued for the next tick."""
        return len(self._pending_actions)

    def register_pre_hook(self, hook: StepHook) -> None:
        """Add a pre-step callback.

        Args:
            hook: Callable(state_before, state_before) → None.
        """
        self._pre_hooks.append(hook)

    def register_post_hook(self, hook: StepHook) -> None:
        """Add a post-step callback.

        Args:
            hook: Callable(state_before, state_after) → None.
        """
        self._post_hooks.append(hook)
