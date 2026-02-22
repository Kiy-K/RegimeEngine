"""
Multi-Agent Survival RL Training — Hierarchical Political Simulation.

Objective: systemic survival under spatial instability (not fairness).
- SurvivalRegimeEnv: hierarchy, topology randomization per episode, shock injection
- Early warning index, hazard clustering amplification, exhaustion growth tracking
- Reward: R = α*survival_time - β*peak_hazard - γ*exhaustion_acceleration
          - δ*cluster_intensity - ε*volatility_spike_rate

Evaluation: mean survival time, hazard peak reduction, exhaustion slope,
cluster containment, robustness under unseen adjacency graphs.
"""

import os
import time
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from regime_engine.agents.survival_env import SurvivalRegimeEnv, SurvivalConfig


# --------------------------------------------------------------------------- #
# Environment factory                                                          #
# --------------------------------------------------------------------------- #

def make_survival_env(
    config: Optional[SurvivalConfig] = None,
    seed: Optional[int] = None,
):
    """Environment factory for vectorized training. Topology randomized each reset."""
    def _init():
        return SurvivalRegimeEnv(config=config, seed=seed)
    return _init


# --------------------------------------------------------------------------- #
# Evaluation metrics (IX. Evaluation Metrics)                                 #
# --------------------------------------------------------------------------- #

def evaluate_survival_policy(
    model: PPO,
    env: SurvivalRegimeEnv,
    n_episodes: int = 20,
    deterministic: bool = True,
    seeds: Optional[List[int]] = None,
) -> dict:
    """
    Evaluate trained policy on randomized topologies.
    Success: agent survives long-term across diverse spatial environments
    without total authoritarian freeze or chaotic entropy.
    """
    if seeds is None:
        seeds = list(range(n_episodes))
    survival_steps: List[int] = []
    peak_hazards: List[float] = []
    final_exhaustions: List[float] = []
    cluster_intensities: List[float] = []

    for i, seed in enumerate(seeds[:n_episodes]):
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        peak_h = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(int(action))
            done = term or trunc
            steps += 1
            peak_h = max(peak_h, info.get("peak_hazard", 0.0))
        survival_steps.append(steps)
        peak_hazards.append(peak_h)
        final_exhaustions.append(info.get("exhaustion", 0.0))
        cluster_intensities.append(info.get("clustering_index", 0.0))

    return {
        "mean_survival_time": float(np.mean(survival_steps)),
        "std_survival_time": float(np.std(survival_steps)),
        "mean_peak_hazard": float(np.mean(peak_hazards)),
        "mean_final_exhaustion": float(np.mean(final_exhaustions)),
        "mean_cluster_intensity": float(np.mean(cluster_intensities)),
        "min_survival": int(np.min(survival_steps)),
        "max_survival": int(np.max(survival_steps)),
    }


class SurvivalEvalCallback(BaseCallback):
    """Log survival metrics to tensorboard and run multi-topology evaluation."""

    def __init__(
        self,
        eval_env: SurvivalRegimeEnv,
        n_eval_episodes: int = 10,
        eval_freq: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0 or self.n_calls == 0:
            return True
        metrics = evaluate_survival_policy(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        if self.logger is not None:
            for k, v in metrics.items():
                self.logger.record(f"eval_survival/{k}", v)
        if self.verbose:
            print(f"[{self.n_calls}] eval: mean_survival={metrics['mean_survival_time']:.1f} peak_hazard={metrics['mean_peak_hazard']:.4f}")
        return True


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Survival RL: Hierarchical political simulation (Phase 3)")

    config = SurvivalConfig(
        max_steps=500,
        alpha_survival=1.0,
        beta_peak_hazard=2.0,
        gamma_exh_acceleration=1.5,
        delta_cluster_intensity=1.2,
        epsilon_volatility_spike=0.8,
        shock_prob_per_step=0.02,
    )

    n_envs = 4
    try:
        env = SubprocVecEnv([make_survival_env(config=config) for _ in range(n_envs)])
        eval_env = SurvivalRegimeEnv(config=config, seed=999)
    except Exception as e:
        print(f"SubprocVecEnv failed: {e}. Using DummyVecEnv.")
        env = DummyVecEnv([make_survival_env(config=config) for _ in range(n_envs)])
        eval_env = SurvivalRegimeEnv(config=config, seed=999)

    log_dir = "./logs/ppo_survival"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 25_000 // n_envs),
        save_path=log_dir,
        name_prefix="ppo_survival",
    )
    survival_eval_callback = SurvivalEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=max(1, 10_000 // n_envs),
        verbose=1,
    )

    # Policy: 48-dim compressed spatial obs -> discrete faction actions
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )

    total_timesteps = 150_000
    print(f"Training for {total_timesteps} timesteps (topology randomized per episode)...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, survival_eval_callback],
    )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} s.")

    model.save(f"{log_dir}/ppo_survival_final")

    # Final evaluation across diverse topologies
    print("Final evaluation (20 episodes, random topologies):")
    metrics = evaluate_survival_policy(model, eval_env, n_episodes=20, deterministic=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
