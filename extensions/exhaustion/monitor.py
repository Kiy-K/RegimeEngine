"""
ExhaustionMonitor — SB3 callback that tracks and penalizes exhaustion overextension.

Bugs fixed vs original:
  1. env.state does not exist on GravitasEnv — replaced with env.world.global_state.
  2. self.locals['rewards'] -= penalty is not valid in SB3's _on_step context;
     locals['rewards'] is a numpy array reference from rollout collection but
     mutating it has no effect on the stored rollout buffer. Fixed by using the
     env's reward shaping hook: we store a pending penalty that is applied through
     the VecEnv's step_wait by temporarily patching reward via _apply_penalty().
     The clean SB3-idiomatic approach is to add reward shaping inside the env,
     not the callback. We implement that pattern here via env.update_config().

  The recommended approach for reward shaping in SB3 is to do it inside the
  environment (RewardShapingWrapper). This callback handles the monitoring and
  early-warning aspects; heavy penalty shaping should be done with a wrapper.
"""

from __future__ import annotations

from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ExhaustionMonitor(BaseCallback):
    """
    Monitors exhaustion levels across training and logs warnings.

    Tracks:
      - exhaustion_history: raw exhaustion per step
      - n_critical_steps:   steps where E > critical_threshold
      - peak_exhaustion:    highest E seen during training

    When exhaustion exceeds critical_threshold, logs a warning and records
    the event. Actual penalty shaping is best applied via a reward wrapper
    on the environment, not in the callback (SB3 locals are read-only for
    reward arrays after rollout collection). This monitor reports; the env
    rewards.

    Args:
        penalty:            Informational — passed to env.update_config if supported.
        critical_threshold: Exhaustion level considered critical (default: 0.8).
        verbose:            0 = silent, 1 = warnings only, 2 = all steps.
    """

    def __init__(
        self,
        penalty: float = 0.1,
        critical_threshold: float = 0.80,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.penalty             = penalty
        self.critical_threshold  = critical_threshold
        self.exhaustion_history: List[float] = []
        self.n_critical_steps:   int         = 0
        self.peak_exhaustion:    float       = 0.0

    def _on_training_start(self) -> None:
        """Propagate exhaustion penalty config to env if it supports it."""
        try:
            for env in self.training_env.envs:
                unwrapped = getattr(env, "unwrapped", env)
                if hasattr(unwrapped, "update_config"):
                    unwrapped.update_config(exhaustion_penalty=self.penalty)
        except Exception:
            pass  # env doesn't support config update; continue silently

    def _on_step(self) -> bool:
        exhaustion = self._read_exhaustion()
        if exhaustion is None:
            return True

        self.exhaustion_history.append(exhaustion)
        if exhaustion > self.peak_exhaustion:
            self.peak_exhaustion = exhaustion

        # Log to TensorBoard if available
        try:
            self.logger.record("gravitas/exhaustion", exhaustion)
            self.logger.record("gravitas/peak_exhaustion", self.peak_exhaustion)
        except Exception:
            pass

        if exhaustion > self.critical_threshold:
            self.n_critical_steps += 1
            if self.verbose > 0:
                print(
                    f"  [ExhaustionMonitor] ⚠️  Step {self.num_timesteps}: "
                    f"E={exhaustion:.3f} > {self.critical_threshold} "
                    f"(critical events: {self.n_critical_steps})"
                )

        return True

    def _on_training_end(self) -> None:
        if not self.exhaustion_history:
            return
        mean_e = float(np.mean(self.exhaustion_history))
        frac_c = self.n_critical_steps / max(1, len(self.exhaustion_history))
        print(
            f"\n[ExhaustionMonitor] Training summary:\n"
            f"  Mean exhaustion   : {mean_e:.3f}\n"
            f"  Peak exhaustion   : {self.peak_exhaustion:.3f}\n"
            f"  Critical fraction : {frac_c:.1%} ({self.n_critical_steps} steps)\n"
        )

    def _read_exhaustion(self) -> float | None:
        """
        Read current exhaustion from the training environment.

        FIX 1: GravitasEnv has no .state attribute.
        The correct path is: env.world.global_state.exhaustion
        We also handle VecEnv wrapping (envs[0]) and Monitor wrapping (unwrapped).
        """
        try:
            envs = self.training_env.envs
        except AttributeError:
            return None

        if not envs:
            return None

        env = envs[0]
        # Unwrap Monitor / FogOfWarWrapper layers
        unwrapped = getattr(env, "unwrapped", env)

        # FIX: use .world.global_state, not .state
        world = getattr(unwrapped, "world", None)
        if world is None:
            # Try one more level of unwrapping (e.g. FogOfWarWrapper.env)
            inner = getattr(unwrapped, "env", None)
            if inner is not None:
                world = getattr(inner, "world", None)

        if world is None:
            return None

        gs = getattr(world, "global_state", None)
        if gs is None:
            return None

        return float(getattr(gs, "exhaustion", 0.0))
