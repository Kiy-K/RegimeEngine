"""
GRAVITAS — PPO Training Pipeline (v2)

Designed for the new governance environment with:
  - Hierarchical action space (stance + allocation)
  - Partial observability via media bias
  - Military cost accumulation
  - Hawkes-process shock arrival
  - Long-horizon, non-convex reward landscape

Usage:
    python train_ppo.py [--timesteps 600000] [--mode flat|hrl] [--device cpu]
                        [--n-envs 4] [--seed 0] [--resume path/to/model.zip]

Modes:
    flat    — single PPO with flattened hierarchical action (fast baseline)
    hrl     — feudal manager/worker split (recommended for mature training)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# ── Attempt to import the new GRAVITAS env; fall back to SurvivalRegimeEnv ──
try:
    from regime_engine.agents.gravitas_env import GravitasEnv, GravitasConfig
    _ENV_CLASS = GravitasEnv
    _CFG_CLASS = GravitasConfig
    _ENV_NAME  = "GravitasEnv"
except ImportError:
    from regime_engine.agents.survival_env import SurvivalRegimeEnv, SurvivalConfig
    _ENV_CLASS = SurvivalRegimeEnv
    _CFG_CLASS = SurvivalConfig
    _ENV_NAME  = "SurvivalRegimeEnv (fallback)"

# ─────────────────────────────────────────────────────────────────────────── #
# Paths & constants                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

LOG_DIR        = "./logs/ppo_gravitas"
TB_DIR         = "./logs/tensorboard_gravitas"
CHECKPOINT_FREQ = 50_000
EVAL_FREQ       = 50_000
EVAL_EPISODES   = 30

# Curriculum phases: (min_step, config_kwargs)
CURRICULUM = [
    (0,       dict(shock_prob_per_step=0.005, max_steps=500)),
    (100_000, dict(shock_prob_per_step=0.010, max_steps=500)),
    (300_000, dict(shock_prob_per_step=0.020, max_steps=600)),
    (600_000, dict(shock_prob_per_step=0.030, max_steps=700)),
]


# ─────────────────────────────────────────────────────────────────────────── #
# Custom feature extractor  (shared trunk for policy + value networks)        #
# ─────────────────────────────────────────────────────────────────────────── #

class GovernanceFeatureExtractor(BaseFeaturesExtractor):
    """
    Three-stream encoder:
      1. Cluster stream   — per-cluster variables (N × cluster_dim)
      2. Global stream    — global dynamic + power variables
      3. Bias stream      — estimated media bias vector

    All three are projected, concatenated, then passed through a shared
    residual trunk. Tanh activations keep features in bounded range,
    appropriate for the [0,1] state domain.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))

        # We split the flat observation into three logical regions.
        # Adjust these if your env produces a different layout.
        # Default layout (SurvivalRegimeEnv fallback):
        #   [0:10]   regime macro
        #   [10:40]  province summary (10 provinces × 3 vars)
        #   [40:43]  EWI, hazard_amp, exh_growth
        #   [43:48]  top-k unstable

        self.global_dim  = 10
        self.spatial_dim = obs_dim - self.global_dim  # remaining

        hidden = 128

        # Global encoder
        self.global_enc = nn.Sequential(
            nn.Linear(self.global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Spatial encoder
        self.spatial_enc = nn.Sequential(
            nn.Linear(self.spatial_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Shared trunk with residual connection
        trunk_in = hidden * 2
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh(),
        )

        # Residual projection (if trunk_in != features_dim)
        self.residual_proj = nn.Linear(trunk_in, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        global_feat  = self.global_enc(obs[:, :self.global_dim])
        spatial_feat = self.spatial_enc(obs[:, self.global_dim:])
        combined     = torch.cat([global_feat, spatial_feat], dim=-1)
        trunk_out    = self.trunk(combined)
        residual     = self.residual_proj(combined)
        return torch.tanh(trunk_out + residual)


# ─────────────────────────────────────────────────────────────────────────── #
# Callbacks                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class CurriculumCallback(BaseCallback):
    """
    Ramps difficulty in four phases defined by CURRICULUM.
    Updates all envs in the vec env in-place.
    Also tracks stance distribution entropy to detect policy collapse
    (agent spamming a single action type — a key failure mode).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._applied_phase = -1
        self._action_counts: Dict[int, int] = {}

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # Apply curriculum phase transition
        active_phase = 0
        active_kwargs: Dict[str, Any] = {}
        for phase_idx, (threshold, kwargs) in enumerate(CURRICULUM):
            if t >= threshold:
                active_phase = phase_idx
                active_kwargs = kwargs

        if active_phase != self._applied_phase:
            self._applied_phase = active_phase
            self._apply_config(active_kwargs, active_phase)

        # Track action distribution for entropy logging
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).flat:
                self._action_counts[int(a)] = self._action_counts.get(int(a), 0) + 1

        return True

    def _apply_config(self, kwargs: Dict[str, Any], phase: int) -> None:
        # Use env_method to call update_config on each remote environment process
        self.training_env.env_method("update_config", **kwargs)
        if self.verbose:
            print(f"\n[Curriculum] → Phase {phase + 1}: {kwargs}")

    def _on_training_end(self) -> None:
        if not self._action_counts:
            return
        total = sum(self._action_counts.values())
        probs = np.array(list(self._action_counts.values())) / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(self._action_counts) + 1e-10)
        print(f"\n[Curriculum] Action entropy: {entropy:.3f} / {max_entropy:.3f} "
              f"({'HEALTHY' if entropy > 0.5 * max_entropy else 'WARNING: policy collapse'})")


class MetricsCallback(BaseCallback):
    """
    Logs rich per-step metrics to JSON and TensorBoard.
    Tracks the new GRAVITAS-specific signals:
      - stance distribution
      - polarization trajectory
      - trust compounding rate
      - military restraint index
    """

    def __init__(self, output_path: str, log_freq: int = CHECKPOINT_FREQ, verbose: int = 0):
        super().__init__(verbose)
        self.output_path = output_path
        self.log_freq    = log_freq
        self._records:   List[Dict[str, Any]] = []
        self._last_log   = 0

        # Rolling buffers for governance-specific signals
        self._military_loads: List[float] = []
        self._polarizations:  List[float] = []
        self._trust_deltas:   List[float] = []
        self._ewi_vals:       List[float] = []

    def _on_step(self) -> bool:
        # Scrape per-step info dicts from the vec env
        infos = self.locals.get("infos", [])
        for info in infos:
            if "polarization" in info:
                self._polarizations.append(float(info["polarization"]))
            if "military_load" in info:
                self._military_loads.append(float(info["military_load"]))
            if "trust_delta" in info:
                self._trust_deltas.append(float(info["trust_delta"]))
            if "early_warning" in info:
                self._ewi_vals.append(float(info["early_warning"]))

        if self.num_timesteps - self._last_log >= self.log_freq:
            self._snapshot()
            self._last_log = self.num_timesteps

        return True

    def _snapshot(self) -> None:
        ep_buf = self.model.ep_info_buffer
        mean_rew = float(np.mean([x["r"] for x in ep_buf])) if ep_buf else float("nan")
        mean_len = float(np.mean([x["l"] for x in ep_buf])) if ep_buf else float("nan")

        record: Dict[str, Any] = {
            "timestep":         self.num_timesteps,
            "mean_ep_reward":   mean_rew,
            "mean_ep_length":   mean_len,
            "mean_polarization": float(np.mean(self._polarizations)) if self._polarizations else None,
            "mean_military_load": float(np.mean(self._military_loads)) if self._military_loads else None,
            "mean_trust_delta":  float(np.mean(self._trust_deltas)) if self._trust_deltas else None,
            "mean_ewi":          float(np.mean(self._ewi_vals)) if self._ewi_vals else None,
        }
        self._records.append(record)

        # Log to TensorBoard if logger available
        try:
            for k, v in record.items():
                if k != "timestep" and v is not None:
                    self.logger.record(f"gravitas/{k}", v)
        except Exception:
            pass

        if self.verbose:
            pol = f"{record['mean_polarization']:.3f}" if record["mean_polarization"] is not None else "n/a"
            mil = f"{record['mean_military_load']:.3f}" if record["mean_military_load"] is not None else "n/a"
            print(f"  [metrics] t={self.num_timesteps:>8,d}  "
                  f"rew={mean_rew:.2f}  len={mean_len:.0f}  "
                  f"pol={pol}  mil={mil}")

        # Reset rolling buffers
        self._military_loads.clear()
        self._polarizations.clear()
        self._trust_deltas.clear()
        self._ewi_vals.clear()

    def _on_training_end(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"\nMetrics log → {self.output_path}")


class EarlyStopCallback(BaseCallback):
    """
    Halt training if the agent falls into a known degenerate pattern:
      - Policy collapse: same action > 90% of the time for 20k steps
      - Reward collapse: mean episode reward < threshold for 3 consecutive snapshots
    """

    def __init__(self, min_action_entropy: float = 0.3,
                 min_reward: float = -500.0,
                 check_freq: int = 20_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.min_action_entropy = min_action_entropy
        self.min_reward         = min_reward
        self.check_freq         = check_freq
        self._last_check        = 0
        self._bad_reward_count  = 0
        self._action_buf:       List[int] = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._action_buf.extend(np.asarray(actions).flat)

        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        # Entropy check
        if self._action_buf:
            counts = np.bincount(self._action_buf)
            probs  = counts / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            if entropy < self.min_action_entropy:
                if self.verbose:
                    print(f"\n[EarlyStop] Policy collapse detected "
                          f"(entropy={entropy:.3f} < {self.min_action_entropy}). "
                          f"Halting training.")
                return False
            self._action_buf.clear()

        # Reward check
        ep_buf = self.model.ep_info_buffer
        if ep_buf:
            mean_rew = float(np.mean([x["r"] for x in ep_buf]))
            if mean_rew < self.min_reward:
                self._bad_reward_count += 1
                if self._bad_reward_count >= 3:
                    if self.verbose:
                        print(f"\n[EarlyStop] Reward below threshold "
                              f"({mean_rew:.1f} < {self.min_reward}) for 3 checks.")
                    return False
            else:
                self._bad_reward_count = 0

        return True


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factories                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_env(seed: int, is_eval: bool = False, max_steps: int = 500):
    """Factory returning a Monitor-wrapped environment for a given seed."""
    def _init():
        shock = 0.005 if not is_eval else 0.020
        # Determine the appropriate constructor keywords based on the environment name
        if _ENV_NAME == "GravitasEnv":
            cfg = _CFG_CLASS(hawkes_base_rate=shock, max_steps=max_steps)
            env = _ENV_CLASS(params=cfg, seed=seed)
        else:
            cfg = _CFG_CLASS(shock_prob_per_step=shock, max_steps=max_steps)
            env = _ENV_CLASS(config=cfg, seed=seed)
        return Monitor(env)
    return _init


def build_train_env(
    n_envs: int,
    seed: int,
    use_subproc: bool = False,
) -> VecNormalize:
    """Build vectorised, normalised training env."""
    fns = [_make_env(seed + i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if (use_subproc and n_envs > 1) else DummyVecEnv(fns)
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.998)


def build_eval_env(seed: int = 9_000) -> VecNormalize:
    """Build eval env on held-out seeds (never seen during training)."""
    vec = DummyVecEnv([_make_env(seed, is_eval=True)])
    return VecNormalize(vec, norm_obs=True, norm_reward=False,
                        training=False, clip_obs=10.0)


# ─────────────────────────────────────────────────────────────────────────── #
# PPO configuration                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def build_flat_ppo(
    train_env: VecNormalize,
    device: str,
    seed: int,
    learning_rate: float = 1e-4,
) -> PPO:
    """
    Flat PPO with the custom governance feature extractor.
    Suitable as a strong baseline; also the fallback if HRL isn't available.

    Key changes vs v1:
      - Lower LR (1e-4 vs 3e-4): reward landscape is non-convex
      - Longer rollouts (4096 vs 2048): delayed consequences need longer credit
      - Higher gamma (0.998 vs 0.995): governance is truly long-horizon
      - Higher ent_coef (0.02 vs 0.01): explore stance diversity aggressively
      - Tighter clip_range (0.15): non-convex loss → conservative updates
      - Tanh activation: bounded state domain
      - Wider net (512-512-256-128): more complex dynamics
    """
    policy_kwargs = dict(
        features_extractor_class=GovernanceFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[512, 512, 256, 128],
            vf=[512, 512, 256, 128],
        ),
        activation_fn=nn.Tanh,
        normalize_images=False,
    )

    return PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=4096,
        batch_size=128,
        n_epochs=8,
        gamma=0.998,
        gae_lambda=0.97,
        clip_range=0.15,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.3,
        policy_kwargs=policy_kwargs,
        tensorboard_log=TB_DIR,
        seed=seed,
        device=device,
        verbose=1,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Main training routine                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def train(
    total_timesteps: int = 600_000,
    seed: int = 0,
    device: str = "cpu",
    n_envs: int = 1,
    mode: str = "flat",
    resume: Optional[str] = None,
    learning_rate: float = 1e-4,
    use_subproc: bool = False,
) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    print("=" * 65)
    print(f"GRAVITAS — PPO Training  [{mode.upper()} mode]")
    print(f"  Environment      : {_ENV_NAME}")
    print(f"  total_timesteps  : {total_timesteps:,}")
    print(f"  n_envs           : {n_envs}")
    print(f"  seed             : {seed}")
    print(f"  device           : {device}")
    print(f"  learning_rate    : {learning_rate}")
    print(f"  log_dir          : {LOG_DIR}")
    print("=" * 65)

    train_env = build_train_env(n_envs, seed, use_subproc=use_subproc)
    eval_env  = build_eval_env(seed=9_000)

    if mode == "flat":
        if resume and os.path.exists(resume):
            print(f"Resuming from {resume} ...")
            model = PPO.load(resume, env=train_env, device=device)
            model.learning_rate = learning_rate
        else:
            model = build_flat_ppo(train_env, device, seed, learning_rate)

    elif mode == "hrl":
        # HRL placeholder — requires feudal_networks or option-critic library.
        # Falls back to flat PPO with a note.
        print("[HRL] Full HRL requires a feudal/option-critic implementation.")
        print("[HRL] Falling back to flat PPO with manager-biased n_steps.")
        print("[HRL] See GRAVITAS_DESIGN.md §XI for the full HRL spec.")
        model = build_flat_ppo(train_env, device, seed, learning_rate)
        # Manager-biased: longer rollouts, higher gamma
        model.n_steps = 8192
        model.gamma   = 0.999

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'flat' or 'hrl'.")

    # ── Callbacks ──────────────────────────────────────────────────────── #
    checkpoint_cb = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
        save_path=LOG_DIR,
        name_prefix="gravitas_ppo",
        verbose=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        eval_freq=max(EVAL_FREQ // n_envs, 1),
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        verbose=1,
    )
    curriculum_cb = CurriculumCallback(verbose=1)
    metrics_cb = MetricsCallback(
        output_path=os.path.join(LOG_DIR, "training_metrics.json"),
        log_freq=CHECKPOINT_FREQ,
        verbose=1,
    )
    early_stop_cb = EarlyStopCallback(
        min_action_entropy=0.3,    # stop if agent collapses to one action
        min_reward=-500.0,
        check_freq=20_000,
        verbose=1,
    )

    callbacks = CallbackList([
        checkpoint_cb,
        eval_cb,
        curriculum_cb,
        metrics_cb,
        early_stop_cb,
    ])

    # ── Progress bar ───────────────────────────────────────────────────── #
    try:
        import tqdm  # noqa
        import rich  # noqa
        use_pb = True
    except ImportError:
        use_pb = False
        print("(pip install tqdm rich for progress bar)")

    # ── Learn ──────────────────────────────────────────────────────────── #
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
        progress_bar=use_pb,
    )
    elapsed = time.time() - t0

    # ── Save ───────────────────────────────────────────────────────────── #
    final_path = os.path.join(LOG_DIR, "gravitas_ppo_final")
    model.save(final_path)
    train_env.save(os.path.join(LOG_DIR, "vec_normalize.pkl"))

    print(f"\nFinal model → {final_path}.zip")
    print(f"VecNormalize → {LOG_DIR}/vec_normalize.pkl")
    print(f"Training done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Training summary ───────────────────────────────────────────────── #
    ep_buf = model.ep_info_buffer
    if ep_buf:
        rewards = [x["r"] for x in ep_buf]
        lengths = [x["l"] for x in ep_buf]
        print(f"\n  Final mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Final mean length : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRAVITAS PPO agent")
    parser.add_argument("--timesteps", type=int, default=600_000,
                        help="Total training timesteps (default: 600,000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Parallel environments (default: 1)")
    parser.add_argument("--mode", type=str, default="flat",
                        choices=["flat", "hrl"],
                        help="Training mode: flat PPO or HRL scaffold (default: flat)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .zip model to resume from")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--subproc", action="store_true",
                        help="Use SubprocVecEnv instead of DummyVecEnv (faster for n_envs>1)")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        n_envs=args.n_envs,
        mode=args.mode,
        resume=args.resume,
        learning_rate=args.lr,
        use_subproc=args.subproc,
    )