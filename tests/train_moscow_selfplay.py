#!/usr/bin/env python3
"""
train_moscow_selfplay.py — Self-play training for Battle of Moscow.

Trains two RecurrentPPO agents (Axis vs Soviet) via iterative self-play
with nonlinear combat, logistics network, and partisan warfare plugins
active during every training step.

Plugin hooks fire inside the env step, creating:
  - Nonlinear Lanchester combat dynamics
  - Graph-based logistics with distance decay & sabotage
  - Autonomous partisan units (uncontrolled by either agent)

Usage:
    # From scratch (optimized with Nuitka .so if available)
    python tests/train_moscow_selfplay.py --total-rounds 20

    # Resume from checkpoint
    python tests/train_moscow_selfplay.py \\
        --resume-from logs/moscow_selfplay/round_005

    # With Nuitka-compiled modules (2-3× faster)
    PYTHONPATH=build:$PYTHONPATH python tests/train_moscow_selfplay.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Project root on path ─────────────────────────────────────────────────── #
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gravitas_engine.core.gravitas_params import GravitasParams
from gravitas_engine.agents.stalingrad_ma import (
    AXIS, SOVIET, SIDE_NAMES,
    StalingradMultiAgentEnv,
    SelfPlayEnv,
)
from regime_loader import load_standalone_regime, build_initial_states
from gravitas.plugins import load_plugins_from_config

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
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gymnasium_shim as gym          # type: ignore[no-redef]
    from gymnasium_shim import spaces     # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────── #
# Moscow defaults                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

MOSCOW_AXIS_CLUSTERS   = [5, 6, 7, 8]
MOSCOW_SOVIET_CLUSTERS = [0, 1, 2, 3]
MOSCOW_CONTESTED       = [4]

MOSCOW_PLUGINS = ["nonlinear_combat", "logistics_network", "partisan_warfare"]


# ─────────────────────────────────────────────────────────────────────────── #
# Plugin-aware self-play env                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class MoscowSelfPlayEnv(gym.Env):
    """
    Wraps SelfPlayEnv and applies Moscow plugins after each env step.

    Plugins modify world state through on_step hooks, adding:
    - Nonlinear combat (Lanchester, fatigue, breakthrough)
    - Logistics network (production, flow, sabotage)
    - Partisan warfare (autonomous units)
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        regime_config: Dict[str, Any],
        trainable_side: int = AXIS,
        opponent_policy: Optional[Any] = None,
        plugins: Optional[List[Any]] = None,
        plugin_configs: Optional[Dict[str, Dict]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._inner = SelfPlayEnv(
            params=regime_config["params"],
            trainable_side=trainable_side,
            opponent_policy=opponent_policy,
            axis_clusters=regime_config["axis_clusters"],
            soviet_clusters=regime_config["soviet_clusters"],
            contested_clusters=regime_config.get("contested_clusters"),
            initial_clusters=regime_config["initial_clusters"],
            initial_alliances=regime_config["initial_alliances"],
            seed=seed,
        )

        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

        # Initialize plugins
        self._plugins = plugins or []
        self._plugin_configs = plugin_configs or {}
        self._step_count = 0

        # Build scenario metadata for plugins
        self._scenario_meta = {
            "axis_clusters": regime_config["axis_clusters"],
            "soviet_clusters": regime_config["soviet_clusters"],
            "contested_clusters": regime_config.get("contested_clusters", MOSCOW_CONTESTED),
            "logistics_links": regime_config.get("logistics_links", []),
            "terrain": regime_config.get("terrain", {}),
        }

    def set_opponent(self, policy: Any) -> None:
        self._inner.set_opponent(policy)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        obs, info = self._inner.reset(seed=seed, options=options)
        self._step_count = 0

        # Reset plugins with world state and scenario metadata
        world = self._inner._ma_env._world
        for plugin in self._plugins:
            cfg = self._plugin_configs.get(plugin.name, {})
            if cfg:
                for k, v in cfg.items():
                    if hasattr(plugin, k):
                        setattr(plugin, k, v)
            try:
                plugin.on_reset(world, engine=self, env=self._inner)
            except TypeError:
                plugin.on_reset(world)

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._inner.step(action)
        self._step_count += 1

        # Apply plugin on_step hooks to world state
        ma_env = self._inner._ma_env
        world = ma_env._world
        rewards_dict = {
            AXIS: info.get("own_reward", reward) if self._inner.trainable_side == AXIS else info.get("opp_reward", 0),
            SOVIET: info.get("own_reward", reward) if self._inner.trainable_side == SOVIET else info.get("opp_reward", 0),
        }

        for plugin in self._plugins:
            if not plugin.enabled:
                continue
            try:
                new_world = plugin.on_step(
                    world, self._step_count,
                    actions={}, observations={},
                    rewards=rewards_dict,
                )
                if new_world is not None:
                    world = new_world
            except Exception:
                pass  # Plugin errors don't crash training

        # Write modified world back
        ma_env._world = world

        return obs, reward, terminated, truncated, info

    @property
    def _scenario_meta(self):
        return self.__scenario_meta

    @_scenario_meta.setter
    def _scenario_meta(self, val):
        self.__scenario_meta = val

    def render(self) -> Optional[str]:
        return self._inner.render()


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factory                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def load_moscow_config(regime_path: str, seed: int = 0) -> Dict[str, Any]:
    """Load Moscow YAML and extract initial states + logistics metadata."""
    regime_data = load_standalone_regime(regime_path, seed=seed)
    params = regime_data["params"]

    # Override n_clusters_max to 12 for obs compatibility
    params = GravitasParams(**{
        **{k: getattr(params, k) for k in GravitasParams.__dataclass_fields__},
        "n_clusters_max": 12,
    })

    init_data = build_initial_states(regime_data, max_N=12)

    # Parse agent cluster assignments from YAML
    agents = regime_data.get("agents", [])
    axis_clusters = MOSCOW_AXIS_CLUSTERS
    soviet_clusters = MOSCOW_SOVIET_CLUSTERS
    contested = MOSCOW_CONTESTED
    for ag in agents:
        if ag.get("side") == "Axis":
            axis_clusters = ag.get("controlled_clusters", axis_clusters)
        elif ag.get("side") == "Soviet":
            soviet_clusters = ag.get("controlled_clusters", soviet_clusters)

    # Extract logistics and terrain metadata from raw YAML
    raw = regime_data.get("raw", {})
    logistics_links = raw.get("logistics_links", [])
    terrain = raw.get("terrain", {})

    return {
        "params": params,
        "initial_clusters": init_data.get("initial_clusters"),
        "initial_alliances": init_data.get("initial_alliances"),
        "axis_clusters": axis_clusters,
        "soviet_clusters": soviet_clusters,
        "contested_clusters": contested,
        "logistics_links": logistics_links,
        "terrain": terrain,
    }


def make_moscow_env(
    regime_config: Dict[str, Any],
    trainable_side: int,
    opponent_policy: Optional[Any] = None,
    plugins: Optional[List[Any]] = None,
    plugin_configs: Optional[Dict[str, Dict]] = None,
    seed: int = 42,
) -> MoscowSelfPlayEnv:
    """Create a MoscowSelfPlayEnv with plugins for one side."""
    return MoscowSelfPlayEnv(
        regime_config=regime_config,
        trainable_side=trainable_side,
        opponent_policy=opponent_policy,
        plugins=plugins,
        plugin_configs=plugin_configs,
        seed=seed,
    )


def make_vec_env(
    regime_config: Dict[str, Any],
    trainable_side: int,
    opponent_policy: Optional[Any] = None,
    plugins_factory=None,
    plugin_configs: Optional[Dict[str, Dict]] = None,
    n_envs: int = 4,
    seed: int = 42,
) -> VecNormalize:
    """Create a vectorized + normalized env for SB3 training."""
    def _make(i: int):
        def _thunk():
            # Each env gets its own plugin instances (independent RNG state)
            plugins = plugins_factory() if plugins_factory else []
            env = make_moscow_env(
                regime_config,
                trainable_side=trainable_side,
                opponent_policy=opponent_policy,
                plugins=plugins,
                plugin_configs=plugin_configs,
                seed=seed + i * 1000,
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
    """Main self-play training loop for Moscow."""
    assert HAS_RPPO, "sb3-contrib required: pip install sb3-contrib"
    assert HAS_SB3,  "stable-baselines3 required"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load Moscow regime
    regime_path = args.regime_file or str(_ROOT / "gravitas" / "scenarios" / "moscow.yaml")
    regime_config = load_moscow_config(regime_path, seed=args.seed)

    # Plugin configuration
    plugin_configs = {
        "nonlinear_combat": {
            "diminishing_returns_alpha": 0.7,
            "breakthrough_threshold": 0.58,
            "fatigue_midpoint": 120,
        },
        "logistics_network": {
            "winter_axis_penalty": 0.35,
            "distance_decay_rate": 0.3,
        },
        "partisan_warfare": {
            "max_partisans": 6,
            "ambush_military_damage": 0.06,
        },
    }

    # Plugin factory — each env gets fresh plugin instances
    def make_plugins():
        return load_plugins_from_config(args.plugins)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          BATTLE OF MOSCOW — SELF-PLAY TRAINING         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Axis clusters:   {regime_config['axis_clusters']}")
    print(f"║  Soviet clusters: {regime_config['soviet_clusters']}")
    print(f"║  Plugins:         {args.plugins}")
    print(f"║  Rounds:          {args.total_rounds}")
    print(f"║  Steps/round:     {args.steps_per_round}")
    print(f"║  N envs:          {args.n_envs}")
    print(f"║  LSTM hidden:     {args.lstm_hidden}")
    print(f"║  Device:          {args.device}")
    print(f"║  Log dir:         {log_dir}")

    # Check for Nuitka-compiled modules
    nuitka_active = any(
        hasattr(sys.modules.get(m, None), "__compiled__")
        for m in ["gravitas", "gravitas_engine"]
    )
    if nuitka_active:
        print("║  ⚡ Nuitka compiled modules detected — FAST MODE")
    else:
        print("║  ℹ  Interpreted mode (run build_nuitka.sh for 2-3× speedup)")
    print("╚══════════════════════════════════════════════════════════╝")

    # WandB
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="gravitas-moscow-selfplay",
            config=vars(args),
            name=f"moscow_r{args.total_rounds}_s{args.steps_per_round}",
        )

    # ── Initialize models ─────────────────────────────────────────────── #
    axis_model: Optional[RecurrentPPO] = None
    soviet_model: Optional[RecurrentPPO] = None

    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    if args.pretrained:
        print(f"\n  Loading pretrained model: {args.pretrained}")
        axis_model = RecurrentPPO.load(args.pretrained, device=args.device)
        soviet_model = RecurrentPPO.load(args.pretrained, device=args.device)
        print("  ✓ Both sides initialized from pretrained model")
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
            plugins_factory=make_plugins,
            plugin_configs=plugin_configs,
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
                max_grad_norm=0.5,
                device=args.device,
                policy_kwargs=policy_kwargs,
            )
        else:
            axis_model.set_env(axis_env)

        axis_model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=args.progress_bar,
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
            plugins_factory=make_plugins,
            plugin_configs=plugin_configs,
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
                max_grad_norm=0.5,
                device=args.device,
                policy_kwargs=policy_kwargs,
            )
        else:
            soviet_model.set_env(soviet_env)

        soviet_model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=args.progress_bar,
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
                regime_config, axis_model, soviet_model,
                plugins_factory=make_plugins,
                plugin_configs=plugin_configs,
                n_episodes=5, seed=args.seed + rnd,
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
                    "axis_train_time": axis_time,
                    "soviet_train_time": soviet_time,
                })

    # ── Final save ────────────────────────────────────────────────────── #
    axis_model.save(str(log_dir / "axis_final"))
    soviet_model.save(str(log_dir / "soviet_final"))
    print(f"\n✅ Moscow training complete. Models saved to {log_dir}/")

    if args.wandb and HAS_WANDB:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────── #
# Quick evaluation                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def quick_eval(
    regime_config: Dict[str, Any],
    axis_model: Any,
    soviet_model: Any,
    plugins_factory=None,
    plugin_configs: Optional[Dict[str, Dict]] = None,
    n_episodes: int = 5,
    seed: int = 99999,
) -> Dict[str, float]:
    """Run evaluation episodes with plugins active."""
    import torch

    plugins = plugins_factory() if plugins_factory else []
    env = make_moscow_env(
        regime_config,
        trainable_side=AXIS,
        opponent_policy=None,
        plugins=plugins,
        plugin_configs=plugin_configs,
        seed=seed,
    )

    axis_rewards, soviet_rewards = [], []
    axis_sigmas, soviet_sigmas = [], []
    ep_lengths = []
    survivals = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            obs_dict = env._inner._ma_env.reset(seed=seed + ep)

            axis_obs   = obs_dict[AXIS]
            soviet_obs = obs_dict[SOVIET]
            ax_lstm, sv_lstm = None, None
            ax_starts = np.ones((1,), dtype=bool)
            sv_starts = np.ones((1,), dtype=bool)

            ax_total_r, sv_total_r = 0.0, 0.0
            done, trunc = False, False
            steps = 0

            # Reset plugins
            world = env._inner._ma_env._world
            for plugin in plugins:
                try:
                    plugin.on_reset(world, engine=env, env=env._inner)
                except TypeError:
                    plugin.on_reset(world)

            while not (done or trunc):
                ax_act, ax_lstm = axis_model.predict(
                    axis_obs.reshape(1, -1), state=ax_lstm,
                    episode_start=ax_starts, deterministic=True,
                )
                sv_act, sv_lstm = soviet_model.predict(
                    soviet_obs.reshape(1, -1), state=sv_lstm,
                    episode_start=sv_starts, deterministic=True,
                )

                actions = {AXIS: int(ax_act[0]), SOVIET: int(sv_act[0])}
                obs_dict, rewards, done, trunc, info = env._inner._ma_env.step(actions)

                # Apply plugins
                ma_env = env._inner._ma_env
                world = ma_env._world
                for plugin in plugins:
                    if plugin.enabled:
                        try:
                            new_w = plugin.on_step(world, steps, actions={}, observations={}, rewards=rewards)
                            if new_w is not None:
                                world = new_w
                        except Exception:
                            pass
                ma_env._world = world

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
    p = argparse.ArgumentParser(description="Battle of Moscow Self-Play Training")
    p.add_argument("--regime-file", type=str, default=None,
                   help="Path to moscow.yaml (auto-detected if None)")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Pretrained model to initialize both agents from")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Directory with round checkpoint to resume from")
    p.add_argument("--log-dir", type=str, default="logs/moscow_selfplay",
                   help="Output directory for models and logs")
    p.add_argument("--plugins", nargs="+", default=MOSCOW_PLUGINS,
                   help="Plugin names to activate during training")
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
    p.add_argument("--lstm-hidden", type=int, default=128,
                   help="LSTM hidden size")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device (auto/cpu/cuda)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=2,
                   help="Save checkpoint every N rounds")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run quick eval every N rounds")
    p.add_argument("--wandb", action="store_true",
                   help="Log to Weights & Biases")
    p.add_argument("--progress-bar", action="store_true",
                   help="Show training progress bar")
    return p.parse_args()


if __name__ == "__main__":
    train_selfplay(parse_args())
