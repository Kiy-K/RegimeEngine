#!/usr/bin/env python3
"""
train_stalingrad_selfplay.py — Self-play training for Stalingrad multi-agent.

Trains two RecurrentPPO agents (Axis vs Soviet) via iterative self-play:
  1. Train Axis agent for K steps vs frozen Soviet opponent
  2. Freeze Axis → new Soviet opponent
  3. Train Soviet agent for K steps vs frozen Axis opponent
  4. Freeze Soviet → new Axis opponent
  5. Repeat

Both agents share the same architecture (RecurrentPPO + LSTM) and
observation/action space shape for compatibility with pretrained models.

Usage:
    python tests/train_stalingrad_selfplay.py [options]

    # Start from scratch
    python tests/train_stalingrad_selfplay.py --total-rounds 20

    # Fine-tune from pre-trained single-agent model
    python tests/train_stalingrad_selfplay.py \\
        --pretrained logs/rppo_gravitas/gravitas_rppo_final.zip \\
        --total-rounds 20

    # Resume from checkpoint
    python tests/train_stalingrad_selfplay.py \\
        --resume-from logs/stalingrad_selfplay/round_005
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ── Project root on path ─────────────────────────────────────────────────── #
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gravitas_engine.core.gravitas_params import GravitasParams
from gravitas_engine.agents.stalingrad_ma import (
    AXIS, SOVIET, SIDE_NAMES,
    SelfPlayEnv,
    DEFAULT_AXIS_CLUSTERS,
    DEFAULT_SOVIET_CLUSTERS,
    CONTESTED_CLUSTERS,
)
from regime_loader import load_standalone_regime, build_initial_states

# ── Optional imports ──────────────────────────────────────────────────────── #
try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factory                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def load_regime_config(regime_path: str, seed: int = 0) -> Dict[str, Any]:
    """Load Stalingrad YAML and extract initial states."""
    regime_data = load_standalone_regime(regime_path, seed=seed)
    params = regime_data["params"]

    # Override n_clusters_max to 12 for obs compatibility with pretrained models
    params = GravitasParams(**{
        **{k: getattr(params, k) for k in GravitasParams.__dataclass_fields__},
        "n_clusters_max": 12,
    })

    init_data = build_initial_states(regime_data, max_N=12)

    # Parse agent cluster assignments
    agents = regime_data.get("agents", [])
    axis_clusters = DEFAULT_AXIS_CLUSTERS
    soviet_clusters = DEFAULT_SOVIET_CLUSTERS
    for ag in agents:
        if ag.get("side") == "Axis":
            axis_clusters = ag.get("controlled_clusters", axis_clusters)
        elif ag.get("side") == "Soviet":
            soviet_clusters = ag.get("controlled_clusters", soviet_clusters)

    return {
        "params": params,
        "initial_clusters": init_data.get("initial_clusters"),
        "initial_alliances": init_data.get("initial_alliances"),
        "axis_clusters": axis_clusters,
        "soviet_clusters": soviet_clusters,
    }


def make_selfplay_env(
    regime_config: Dict[str, Any],
    trainable_side: int,
    opponent_policy: Optional[Any] = None,
    seed: int = 42,
) -> SelfPlayEnv:
    """Create a SelfPlayEnv for one side."""
    return SelfPlayEnv(
        params=regime_config["params"],
        trainable_side=trainable_side,
        opponent_policy=opponent_policy,
        axis_clusters=regime_config["axis_clusters"],
        soviet_clusters=regime_config["soviet_clusters"],
        initial_clusters=regime_config["initial_clusters"],
        initial_alliances=regime_config["initial_alliances"],
        seed=seed,
    )


def make_vec_env(
    regime_config: Dict[str, Any],
    trainable_side: int,
    opponent_policy: Optional[Any] = None,
    n_envs: int = 4,
    seed: int = 42,
) -> VecNormalize:
    """Create a vectorized + normalized env for SB3 training."""
    def _make(i: int):
        def _thunk():
            env = make_selfplay_env(
                regime_config, trainable_side, opponent_policy, seed=seed + i * 1000,
            )
            return Monitor(env)
        return _thunk

    vec = DummyVecEnv([_make(i) for i in range(n_envs)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


# ─────────────────────────────────────────────────────────────────────────── #
# Self-play training loop                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def train_selfplay(args: argparse.Namespace) -> None:
    """Main self-play training loop."""
    assert HAS_RPPO, "sb3-contrib required: pip install sb3-contrib"
    assert HAS_SB3,  "stable-baselines3 required"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    regime_path = args.regime_file or str(_ROOT / "training" / "regimes" / "stalingrad.yaml")
    regime_config = load_regime_config(regime_path, seed=args.seed)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          STALINGRAD SELF-PLAY TRAINING                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Axis clusters:   {regime_config['axis_clusters']}")
    print(f"║  Soviet clusters: {regime_config['soviet_clusters']}")
    print(f"║  Rounds:          {args.total_rounds}")
    print(f"║  Steps/round:     {args.steps_per_round}")
    print(f"║  N envs:          {args.n_envs}")
    print(f"║  Log dir:         {log_dir}")
    print("╚══════════════════════════════════════════════════════════╝")

    # WandB
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="gravitas-stalingrad-selfplay",
            config=vars(args),
            name=f"selfplay_r{args.total_rounds}_s{args.steps_per_round}",
        )

    # ── Initialize models ─────────────────────────────────────────────── #
    axis_model: Optional[RecurrentPPO] = None
    soviet_model: Optional[RecurrentPPO] = None

    if args.pretrained:
        print(f"\n  Loading pretrained model: {args.pretrained}")
        base_model = RecurrentPPO.load(args.pretrained, device=args.device)
        # Both sides start from the same pretrained weights
        axis_model = base_model
        soviet_model = RecurrentPPO.load(args.pretrained, device=args.device)
        print("  ✓ Both sides initialized from pretrained single-agent model")
    elif args.resume_from:
        rdir = Path(args.resume_from)
        axis_path = rdir / "axis_model.zip"
        soviet_path = rdir / "soviet_model.zip"
        if axis_path.exists():
            axis_model = RecurrentPPO.load(str(axis_path), device=args.device)
            print(f"  ✓ Resumed Axis model from {axis_path}")
        if soviet_path.exists():
            soviet_model = RecurrentPPO.load(str(soviet_path), device=args.device)
            print(f"  ✓ Resumed Soviet model from {soviet_path}")

    # ── Self-play rounds ──────────────────────────────────────────────── #
    for rnd in range(1, args.total_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {rnd}/{args.total_rounds}")
        print(f"{'='*60}")

        # ── Phase 1: Train Axis vs frozen Soviet ──────────────────────── #
        print(f"\n  Phase 1: Training AXIS (vs {'Soviet model' if soviet_model else 'random'})")
        t0 = time.time()

        axis_env = make_vec_env(
            regime_config,
            trainable_side=AXIS,
            opponent_policy=soviet_model,
            n_envs=args.n_envs,
            seed=args.seed + rnd * 100,
        )

        if axis_model is None:
            axis_model = RecurrentPPO(
                "MlpLstmPolicy",
                axis_env,
                verbose=1,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                clip_range=0.2,
                device=args.device,
                policy_kwargs=dict(
                    lstm_hidden_size=128,
                    n_lstm_layers=1,
                    shared_lstm=True,
                    enable_critic_lstm=True,
                    net_arch=dict(pi=[128, 64], vf=[128, 64]),
                ),
            )
        else:
            axis_model.set_env(axis_env)

        axis_model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        axis_time = time.time() - t0
        print(f"  ✓ Axis training done ({axis_time:.1f}s)")
        axis_env.close()

        # ── Phase 2: Train Soviet vs frozen Axis ──────────────────────── #
        print(f"\n  Phase 2: Training SOVIET (vs Axis model)")
        t0 = time.time()

        soviet_env = make_vec_env(
            regime_config,
            trainable_side=SOVIET,
            opponent_policy=axis_model,
            n_envs=args.n_envs,
            seed=args.seed + rnd * 100 + 50,
        )

        if soviet_model is None:
            soviet_model = RecurrentPPO(
                "MlpLstmPolicy",
                soviet_env,
                verbose=1,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                clip_range=0.2,
                device=args.device,
                policy_kwargs=dict(
                    lstm_hidden_size=128,
                    n_lstm_layers=1,
                    shared_lstm=True,
                    enable_critic_lstm=True,
                    net_arch=dict(pi=[128, 64], vf=[128, 64]),
                ),
            )
        else:
            soviet_model.set_env(soviet_env)

        soviet_model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        soviet_time = time.time() - t0
        print(f"  ✓ Soviet training done ({soviet_time:.1f}s)")
        soviet_env.close()

        # ── Save checkpoint ───────────────────────────────────────────── #
        if rnd % args.save_every == 0 or rnd == args.total_rounds:
            ckpt_dir = log_dir / f"round_{rnd:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            axis_model.save(str(ckpt_dir / "axis_model"))
            soviet_model.save(str(ckpt_dir / "soviet_model"))
            print(f"  💾 Saved checkpoint → {ckpt_dir}")

        # ── Quick evaluation ──────────────────────────────────────────── #
        if rnd % args.eval_every == 0:
            eval_results = quick_eval(
                regime_config, axis_model, soviet_model, n_episodes=5, seed=args.seed + rnd,
            )
            print(f"\n  📊 Quick Eval (5 eps):")
            print(f"     Axis  mean reward: {eval_results['axis_reward_mean']:.2f}")
            print(f"     Soviet mean reward: {eval_results['soviet_reward_mean']:.2f}")
            print(f"     Axis  mean σ: {eval_results['axis_sigma_mean']:.3f}")
            print(f"     Soviet mean σ: {eval_results['soviet_sigma_mean']:.3f}")
            print(f"     Survival rate: {eval_results['survival_rate']:.0%}")
            print(f"     Avg ep length: {eval_results['avg_length']:.0f}")

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "round": rnd,
                    **{f"eval/{k}": v for k, v in eval_results.items()},
                })

    # ── Final save ────────────────────────────────────────────────────── #
    axis_model.save(str(log_dir / "axis_final"))
    soviet_model.save(str(log_dir / "soviet_final"))
    print(f"\n✅ Training complete. Models saved to {log_dir}/")

    if args.wandb and HAS_WANDB:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────── #
# Quick evaluation                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def quick_eval(
    regime_config: Dict[str, Any],
    axis_model: Any,
    soviet_model: Any,
    n_episodes: int = 5,
    seed: int = 99999,
) -> Dict[str, float]:
    """Run a few episodes of Axis vs Soviet and return summary stats."""
    import torch

    env = make_selfplay_env(regime_config, trainable_side=AXIS, opponent_policy=None, seed=seed)

    axis_rewards = []
    soviet_rewards = []
    axis_sigmas = []
    soviet_sigmas = []
    ep_lengths = []
    survivals = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            obs_dict = env._ma_env.reset(seed=seed + ep)

            axis_obs    = obs_dict[AXIS]
            soviet_obs  = obs_dict[SOVIET]
            ax_lstm     = None
            sv_lstm     = None
            ax_starts   = np.ones((1,), dtype=bool)
            sv_starts   = np.ones((1,), dtype=bool)

            ax_total_r  = 0.0
            sv_total_r  = 0.0
            done = False
            trunc = False
            steps = 0

            while not (done or trunc):
                # Axis action
                ax_act, ax_lstm = axis_model.predict(
                    axis_obs.reshape(1, -1), state=ax_lstm,
                    episode_start=ax_starts, deterministic=True,
                )
                # Soviet action
                sv_act, sv_lstm = soviet_model.predict(
                    soviet_obs.reshape(1, -1), state=sv_lstm,
                    episode_start=sv_starts, deterministic=True,
                )

                actions = {AXIS: int(ax_act[0]), SOVIET: int(sv_act[0])}
                obs_dict, rewards, done, trunc, info = env._ma_env.step(actions)

                axis_obs   = obs_dict[AXIS]
                soviet_obs = obs_dict[SOVIET]
                ax_starts  = np.zeros((1,), dtype=bool)
                sv_starts  = np.zeros((1,), dtype=bool)

                ax_total_r += rewards[AXIS]
                sv_total_r += rewards[SOVIET]
                steps += 1

            axis_rewards.append(ax_total_r)
            soviet_rewards.append(sv_total_r)
            axis_sigmas.append(info.get("axis_mean_sigma", 0.0))
            soviet_sigmas.append(info.get("soviet_mean_sigma", 0.0))
            ep_lengths.append(steps)
            if not done:
                survivals += 1

    return {
        "axis_reward_mean": float(np.mean(axis_rewards)),
        "soviet_reward_mean": float(np.mean(soviet_rewards)),
        "axis_sigma_mean": float(np.mean(axis_sigmas)),
        "soviet_sigma_mean": float(np.mean(soviet_sigmas)),
        "survival_rate": survivals / max(n_episodes, 1),
        "avg_length": float(np.mean(ep_lengths)),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stalingrad Self-Play Training")
    p.add_argument("--regime-file", type=str, default=None,
                   help="Path to stalingrad.yaml (auto-detected if None)")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Pretrained model to initialize both agents from")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Directory with round checkpoint to resume from")
    p.add_argument("--log-dir", type=str, default="logs/stalingrad_selfplay",
                   help="Output directory for models and logs")
    p.add_argument("--total-rounds", type=int, default=10,
                   help="Number of self-play rounds")
    p.add_argument("--steps-per-round", type=int, default=50000,
                   help="Training steps per side per round")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Parallel envs for training")
    p.add_argument("--n-steps", type=int, default=256,
                   help="Rollout steps per env per update")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Minibatch size")
    p.add_argument("--n-epochs", type=int, default=4,
                   help="PPO epochs per update")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="Entropy coefficient")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=2,
                   help="Save checkpoint every N rounds")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run quick eval every N rounds")
    p.add_argument("--wandb", action="store_true",
                   help="Log to Weights & Biases")
    return p.parse_args()


if __name__ == "__main__":
    train_selfplay(parse_args())
