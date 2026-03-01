"""
GRAVITAS — RecurrentPPO Training Pipeline

WHY NOT PPO:
  PPO assumes the Markov property: the optimal action depends only on the
  current observation. GRAVITAS breaks this fundamentally.

  The agent observes  obs = f(true_state, media_bias)  — a distorted view
  whose distortion level depends on past actions (PROPAGANDA raises bias,
  DECENTRALIZE lowers it) and on past shocks. From a single observation the
  agent cannot distinguish:
    - σ = 0.85 because the system is genuinely stable
    - σ = 0.85 because the agent ran PROPAGANDA 8 steps ago and bias = +0.20

  military_load at t=0 looks fine; military_load × steps_sustained is what
  predicts collapse at t=80. PPO has no mechanism to track this.

WHY RECURRENT PPO (LSTM):
  RecurrentPPO maintains a hidden state h_t across each episode. The LSTM
  learns to track bias drift, military accumulation history, Hawkes clustering,
  and trust/coherence trends — all the temporally extended signals PPO misses.
  Installation:  pip install sb3-contrib

BUGS FIXED vs train_ppo.py:
  1. --regime now forwarded from argparse all the way to _make_env().
  2. CURRICULUM used shock_prob_per_step (doesn't exist). Fixed to hawkes_base_rate.
  3. GovernanceFeatureExtractor hardcoded global_dim=10. Now accepts explicit max_N
     so the obs split is always correct — with or without PopWrapper.
  4. HRL mode was a dead stub. Replaced with 'single' and 'curriculum' modes.

POP SYSTEM (--pop flag):
  Wraps GravitasEnv with PopWrapper, adding vectorized demographic state:
    - 8 job/class archetypes: SUBSISTENCE → ELITE
    - Per-archetype income, satisfaction, radicalization
    - Ethnic share simplex + cultural distance matrix per cluster
    - 5 aggregate obs dims per cluster appended to obs vector
  The feature extractor routes these through a dedicated fourth stream.
  Cost: ~4ms/step at N=12.  Tracked in metrics: gini, satisfaction,
  radical_mass, ethnic_tension, and pop_unrest_index (rad × gini composite).

Usage:
    python train_rppo.py
    python train_rppo.py --regime PredatorPrey --timesteps 800000 --n-envs 4
    python train_rppo.py --regime PredatorPrey --pop
    python train_rppo.py --regime CollaborativeBuild --pop --device cuda
    python train_rppo.py --resume logs/rppo_gravitas/gravitas_rppo_final.zip
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from regime_loader import (
    load_regime_config,
    get_regime_by_name,
    build_gravitas_params,
    get_training_config,
)

from sb3_contrib import RecurrentPPO

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gravitas_engine.agents.gravitas_env import GravitasEnv
from gravitas_engine.core.gravitas_params import GravitasParams

# Optional pop system
try:
    from extensions.pop import PopWrapper, PopParams
    _HAS_POP = True
except ImportError:
    _HAS_POP  = False
    PopWrapper = None  # type: ignore[assignment,misc]
    PopParams  = None  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────── #
# Paths & constants                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

LOG_DIR         = "./logs/rppo_gravitas"
TB_DIR          = "./logs/tensorboard_gravitas"
CHECKPOINT_FREQ = 50_000
EVAL_FREQ       = 50_000
EVAL_EPISODES   = 30

# FIX 2: hawkes_base_rate, not shock_prob_per_step
CURRICULUM = [
    (0,       dict(hawkes_base_rate=0.005, max_steps=400)),
    (100_000, dict(hawkes_base_rate=0.010, max_steps=500)),
    (300_000, dict(hawkes_base_rate=0.020, max_steps=600)),
    (600_000, dict(hawkes_base_rate=0.030, max_steps=700)),
]


# ─────────────────────────────────────────────────────────────────────────── #
# Feature extractor                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class GovernanceFeatureExtractor(BaseFeaturesExtractor):
    """
    Multi-stream encoder feeding into the RecurrentPPO LSTM.

    Observation layout (10*max_N + 8 + action_dim):
      Stream 1 — Cluster   obs[:3*max_N]                        σ̂, ĥ, r̂ per cluster
      Stream 2 — Global    obs[3*max_N : 3*max_N+7]             E, Φ, Π, Ψ, M, T, shock_rate
      Stream 3 — Diplomacy obs[3*max_N+7 : 5*max_N+7]           bias(N), alliance(N)
      Stream 4 — Demog     obs[5*max_N+7 : 10*max_N+7]          pop(N), economy(4*N)
      Stream 5 — Context   obs[10*max_N+7 : 10*max_N+7+act+1]   prev_action, step_frac

    WITH PopWrapper — extra pop dims appended after base_obs_dim.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        max_N: int = 12,
        use_pop: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))
        self._use_pop = use_pop

        # Obs layout: 3N cluster + 7 global + 2N diplo + 5N demog + (act+1) context
        self._cluster_dim = 3 * max_N           # σ̂, ĥ, r̂
        self._global_dim  = 7                    # E, Φ, Π, Ψ, M, T, shock_rate
        self._diplo_dim   = 2 * max_N            # bias + alliance
        self._demog_dim   = 5 * max_N            # pop + economy(4)
        base_obs_dim = self._cluster_dim + self._global_dim + self._diplo_dim + self._demog_dim
        self._context_dim = obs_dim - base_obs_dim  # action_dim + step_frac (+ pop ext)
        self._pop_ext_dim = 0

        # If PopWrapper adds extra dims, split them out
        if use_pop:
            # PopWrapper appends 5*max_N dims; context is the remainder
            self._pop_ext_dim = 5 * max_N
            self._context_dim = obs_dim - base_obs_dim - self._pop_ext_dim

        hidden = 128

        self.cluster_enc = nn.Sequential(
            nn.Linear(self._cluster_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),            nn.Tanh(),
        )
        self.global_enc = nn.Sequential(
            nn.Linear(self._global_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2),      nn.Tanh(),
        )
        self.diplo_enc = nn.Sequential(
            nn.Linear(self._diplo_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2),     nn.Tanh(),
        )
        self.demog_enc = nn.Sequential(
            nn.Linear(self._demog_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2),     nn.Tanh(),
        )
        self.context_enc = nn.Sequential(
            nn.Linear(self._context_dim, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2),       nn.Tanh(),
        )

        # 128 + 64 + 64 + 64 + 64 = 384
        trunk_in = hidden + (hidden // 2) * 4

        if use_pop and self._pop_ext_dim > 0:
            self.pop_enc = nn.Sequential(
                nn.Linear(self._pop_ext_dim, hidden // 2), nn.Tanh(),
                nn.Linear(hidden // 2, hidden // 2),       nn.Tanh(),
            )
            trunk_in += hidden // 2
        else:
            self.pop_enc = None

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.Tanh(),
            nn.Linear(256, features_dim), nn.Tanh(),
        )
        self.residual_proj = nn.Linear(trunk_in, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        i0 = 0
        i1 = self._cluster_dim
        i2 = i1 + self._global_dim
        i3 = i2 + self._diplo_dim
        i4 = i3 + self._demog_dim
        i5 = i4 + self._context_dim

        streams = [
            self.cluster_enc(obs[:, i0:i1]),
            self.global_enc(obs[:, i1:i2]),
            self.diplo_enc(obs[:, i2:i3]),
            self.demog_enc(obs[:, i3:i4]),
            self.context_enc(obs[:, i4:i5]),
        ]
        if self.pop_enc is not None and self._pop_ext_dim > 0:
            streams.append(self.pop_enc(obs[:, i5:]))

        combined  = torch.cat(streams, dim=-1)
        trunk_out = self.trunk(combined)
        residual  = self.residual_proj(combined)
        return torch.tanh(trunk_out + residual)


# ─────────────────────────────────────────────────────────────────────────── #
# Callbacks                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

class CurriculumCallback(BaseCallback):
    """Four-phase difficulty ramp. FIX 2: uses hawkes_base_rate."""

    def __init__(self, curriculum=None, verbose: int = 1):
        super().__init__(verbose)
        self._curriculum    = curriculum or CURRICULUM
        self._applied_phase = -1
        self._action_counts: Dict[int, int] = {}

    def _on_step(self) -> bool:
        t = self.num_timesteps
        active_phase, active_kwargs = 0, {}
        for idx, (threshold, kwargs) in enumerate(self._curriculum):
            if t >= threshold:
                active_phase, active_kwargs = idx, kwargs

        if active_phase != self._applied_phase:
            self._applied_phase = active_phase
            self.training_env.env_method("update_config", **active_kwargs)
            if self.verbose:
                print(f"\n[Curriculum] → Phase {active_phase + 1}: {active_kwargs}")

        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).flat:
                self._action_counts[int(a)] = self._action_counts.get(int(a), 0) + 1
        return True

    def _on_training_end(self) -> None:
        if not self._action_counts:
            return
        total  = sum(self._action_counts.values())
        probs  = np.array(list(self._action_counts.values())) / total
        H      = float(-np.sum(probs * np.log(probs + 1e-10)))
        H_max  = np.log(max(1, len(self._action_counts)))
        status = "HEALTHY" if H > 0.5 * H_max else "⚠ POLICY COLLAPSE"
        dist   = {k: f"{v/total:.1%}" for k, v in sorted(self._action_counts.items())}
        print(f"\n[Curriculum] Entropy: {H:.3f}/{H_max:.3f} [{status}]  dist={dist}")


class MetricsCallback(BaseCallback):
    """
    Logs governance signals + pop demographics to JSON + TensorBoard.

    Core signals: polarization, military_load, exhaustion, fragmentation, media_bias
    Pop signals:  gini, mean_satisfaction, radical_mass, ethnic_tension (when --pop)
    Derived:      propaganda_trap_active — Π high while bias is positive
                  pop_unrest_index       — radical_mass × gini early-warning composite
    """

    def __init__(
        self,
        output_path: str,
        log_freq: int = CHECKPOINT_FREQ,
        use_pop: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_path = output_path
        self.log_freq    = log_freq
        self.use_pop     = use_pop
        self._records:  List[Dict[str, Any]] = []
        self._last_log  = 0
        self._buf: Dict[str, List[float]] = {
            "polarization":  [], "military_load": [], "trust_delta":  [],
            "exhaustion":    [], "fragmentation": [], "media_bias":   [],
            "n_shocks":      [],
        }
        if use_pop:
            self._buf.update({
                "pop_gini":              [],
                "pop_mean_satisfaction": [],
                "pop_radical_mass":      [],
                "pop_ethnic_tension":    [],
                "pop_fractionalization": [],
            })

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in ["polarization", "military_load", "trust_delta",
                        "exhaustion", "fragmentation", "media_bias", "n_shocks"]:
                if key in info:
                    self._buf[key].append(float(info[key]))
            if self.use_pop and "pop" in info:
                pop_info = info["pop"]
                for key in ["pop_gini", "pop_mean_satisfaction",
                            "pop_radical_mass", "pop_ethnic_tension",
                            "pop_fractionalization"]:
                    if key in pop_info:
                        self._buf[key].append(float(pop_info[key]))

        if self.num_timesteps - self._last_log >= self.log_freq:
            self._snapshot()
            self._last_log = self.num_timesteps
        return True

    def _snapshot(self) -> None:
        ep_buf   = self.model.ep_info_buffer
        mean_rew = float(np.mean([x["r"] for x in ep_buf])) if ep_buf else float("nan")
        mean_len = float(np.mean([x["l"] for x in ep_buf])) if ep_buf else float("nan")

        record: Dict[str, Any] = {
            "timestep": self.num_timesteps,
            "mean_ep_reward": mean_rew,
            "mean_ep_length": mean_len,
        }
        for key, vals in self._buf.items():
            record[f"mean_{key}"] = float(np.mean(vals)) if vals else None

        pol  = record.get("mean_polarization")
        bias = record.get("mean_media_bias")
        if pol is not None and bias is not None:
            record["propaganda_trap_active"] = bool(pol > 0.40 and bias > 0.05)

        if self.use_pop:
            rad = record.get("mean_pop_radical_mass")
            gin = record.get("mean_pop_gini")
            if rad is not None and gin is not None:
                record["pop_unrest_index"] = float(rad * gin)

        self._records.append(record)

        try:
            for k, v in record.items():
                if k != "timestep" and isinstance(v, (int, float)) and np.isfinite(float(v)):
                    self.logger.record(f"gravitas/{k}", v)
        except Exception:
            pass

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({k: v for k, v in record.items() if v is not None})
        except Exception:
            pass

        if self.verbose:
            pol_s = f"{record.get('mean_polarization', float('nan')):.3f}"
            mil_s = f"{record.get('mean_military_load', float('nan')):.3f}"
            exh_s = f"{record.get('mean_exhaustion', float('nan')):.3f}"
            extra = ""
            if self.use_pop:
                rad_s = f"{record.get('mean_pop_radical_mass', float('nan')):.3f}"
                gin_s = f"{record.get('mean_pop_gini', float('nan')):.3f}"
                extra = f"  rad={rad_s}  gini={gin_s}"
            print(f"  [metrics] t={self.num_timesteps:>8,d}  "
                  f"rew={mean_rew:>7.2f}  len={mean_len:>5.0f}  "
                  f"pol={pol_s}  mil={mil_s}  exh={exh_s}{extra}")

        for buf in self._buf.values():
            buf.clear()

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"\nMetrics log → {self.output_path}")


class EarlyStopCallback(BaseCallback):
    """Halt on policy collapse (low entropy) or reward collapse."""

    def __init__(self, min_action_entropy=0.3, min_reward=-500.0,
                 check_freq=20_000, verbose=1):
        super().__init__(verbose)
        self.min_action_entropy = min_action_entropy
        self.min_reward         = min_reward
        self.check_freq         = check_freq
        self._last_check        = 0
        self._bad_reward_count  = 0
        self._action_buf: List[int] = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self._action_buf.extend(np.asarray(actions).flat)

        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        if self._action_buf:
            counts  = np.bincount(self._action_buf)
            probs   = counts / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            if entropy < self.min_action_entropy:
                if self.verbose:
                    print(f"\n[EarlyStop] Policy collapse (H={entropy:.3f}). Halting.")
                return False
            self._action_buf.clear()

        ep_buf = self.model.ep_info_buffer
        if ep_buf:
            mean_rew = float(np.mean([x["r"] for x in ep_buf]))
            if mean_rew < self.min_reward:
                self._bad_reward_count += 1
                if self._bad_reward_count >= 3:
                    if self.verbose:
                        print(f"\n[EarlyStop] Reward collapsed ({mean_rew:.1f}). Halting.")
                    return False
            else:
                self._bad_reward_count = 0
        return True


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factories                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_env(
    seed: int,
    params: GravitasParams,
    pop_params: Optional["PopParams"] = None,
    is_eval: bool = False,
    custom_shocks: Optional[List[Dict[str, Any]]] = None,
    nation_to_cluster: Optional[Dict[str, int]] = None,
) -> callable:
    """
    Factory returning a Monitor-wrapped GravitasEnv.
    When pop_params is provided and _HAS_POP, inserts PopWrapper before Monitor.
    """
    def _init():
        env = GravitasEnv(params=params, seed=seed)
        if pop_params is not None and _HAS_POP:
            env = PopWrapper(
                env,
                pop_params=pop_params,
                seed=seed,
                custom_shocks=custom_shocks,
                nation_to_cluster=nation_to_cluster,
            )
        return Monitor(env)
    return _init


def build_train_env(
    n_envs: int,
    seed: int,
    params: GravitasParams,
    pop_params: Optional["PopParams"] = None,
    use_subproc: bool = False,
    custom_shocks: Optional[List[Dict[str, Any]]] = None,
    nation_to_cluster: Optional[Dict[str, int]] = None,
) -> VecNormalize:
    fns = [_make_env(seed + i, params, pop_params, custom_shocks=custom_shocks, nation_to_cluster=nation_to_cluster) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if (use_subproc and n_envs > 1) else DummyVecEnv(fns)
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.998)


def build_eval_env(
    seed: int,
    params: GravitasParams,
    pop_params: Optional["PopParams"] = None,
    custom_shocks: Optional[List[Dict[str, Any]]] = None,
    nation_to_cluster: Optional[Dict[str, int]] = None,
) -> VecNormalize:
    vec = DummyVecEnv([_make_env(seed, params, pop_params, is_eval=True, custom_shocks=custom_shocks, nation_to_cluster=nation_to_cluster)])
    return VecNormalize(vec, norm_obs=True, norm_reward=False,
                        training=False, clip_obs=10.0)


# ─────────────────────────────────────────────────────────────────────────── #
# RecurrentPPO configuration                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def build_recurrent_ppo(
    train_env: VecNormalize,
    device: str,
    seed: int,
    learning_rate: float = 1e-4,
    max_N: int = 12,
    use_pop: bool = False,
) -> RecurrentPPO:
    """
    RecurrentPPO with LSTM + GovernanceFeatureExtractor.

    max_N and use_pop flow from train() → here → policy_kwargs so the
    feature extractor always partitions the obs correctly.
    """
    extractor_kwargs = dict(features_dim=256, max_N=max_N, use_pop=use_pop)

    return RecurrentPPO(
        "MlpLstmPolicy", train_env, learning_rate=learning_rate,
        n_steps=2048, batch_size=128, n_epochs=8,
        gamma=0.998, gae_lambda=0.95,
        clip_range=0.15, clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.02, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(
            features_extractor_class=GovernanceFeatureExtractor,
            features_extractor_kwargs=extractor_kwargs,
            lstm_hidden_size=256,
            n_lstm_layers=2,
            shared_lstm=True,
            enable_critic_lstm=False,
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
            activation_fn=nn.Tanh,
            normalize_images=False,
        ),
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
    mode: str = "single",
    regime: Optional[str] = None,
    resume: Optional[str] = None,
    learning_rate: float = 1e-4,
    use_subproc: bool = False,
    use_pop: bool = False,
    config_path: str = "regime_config.yaml",
) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    if use_pop and not _HAS_POP:
        print("⚠  --pop requested but gravitas_engine.extensions.pop not importable.")
        print("   Ensure gravitas_engine is on PYTHONPATH. Continuing without pop.")
        use_pop = False

    # ── Regime → GravitasParams ───────────────────────────────────────────── #
    params      = GravitasParams(seed=seed)
    train_cfg   = {}
    regime_data = None

    if regime:
        try:
            config      = load_regime_config(config_path)
            regime_data = get_regime_by_name(config, regime)
            params      = build_gravitas_params(regime_data, seed=seed)
            train_cfg   = get_training_config(regime_data, config.get("defaults", {}))
            if total_timesteps == 600_000 and "total_timesteps" in train_cfg:
                total_timesteps = train_cfg["total_timesteps"]
            print(f"🎯 Regime: {regime_data['name']}")
            print(f"   {regime_data.get('description', '').strip()[:100]}")
        except Exception as e:
            print(f"⚠  Failed to load regime '{regime}': {e}. Using defaults.")

    pop_params = PopParams() if (use_pop and _HAS_POP) else None
    custom_shocks     = regime_data.get("custom_shocks", []) if regime_data else []
    nation_to_cluster = regime_data.get("nation_to_cluster", {}) if regime_data else {}
    max_N      = params.n_clusters_max

    eval_params = GravitasParams(**{
        **{f: getattr(params, f) for f in params.__dataclass_fields__},
        "seed": seed + 9000,
    })

    # ── Summary ───────────────────────────────────────────────────────────── #
    print("=" * 65)
    print(f"GRAVITAS — RecurrentPPO  [{mode.upper()}]")
    print(f"  algo             : RecurrentPPO+LSTM")
    print(f"  total_timesteps  : {total_timesteps:,}")
    print(f"  n_envs           : {n_envs}  |  seed: {seed}  |  device: {device}")
    print(f"  learning_rate    : {learning_rate}")
    print(f"  regime           : {regime or 'default'}")
    print(f"  pop system       : {'ON (8 archetypes, 3 ethnic groups)' if use_pop else 'OFF'}")
    print(f"  n_clusters range : [{params.n_clusters_min}, {params.n_clusters_max}]")
    print(f"  hawkes_base_rate : {params.hawkes_base_rate}  |  max_steps: {params.max_steps}")
    print("=" * 65)

    # ── W&B ───────────────────────────────────────────────────────────────── #
    try:
        import wandb
        if not wandb.api.api_key:
            raise RuntimeError("No W&B API key configured")
        wandb.init(
            project="gravitas-rppo",
            name=f"rppo-{regime or 'default'}-{'pop' if use_pop else 'base'}-seed{seed}",
            config={
                "algo": "RecurrentPPO", "use_pop": use_pop,
                "total_timesteps": total_timesteps, "n_envs": n_envs,
                "seed": seed, "learning_rate": learning_rate, "regime": regime,
                "lstm_hidden_size": 256, "sequence_length": 64,
                "n_clusters_range": [params.n_clusters_min, params.n_clusters_max],
            },
            sync_tensorboard=True,
        )
        print(f"📊 W&B: {wandb.run.url}")
    except ImportError:
        print("⚠  W&B not installed. pip install wandb")
    except Exception as e:
        print(f"⚠  W&B init failed: {e}")

    # ── Environments ──────────────────────────────────────────────────────── #
    train_env = build_train_env(n_envs, seed, params, pop_params, use_subproc, custom_shocks, nation_to_cluster)
    eval_env  = build_eval_env(seed + 9000, eval_params, pop_params)

    # ── Model ─────────────────────────────────────────────────────────────── #
    resume_path = resume
    if resume_path and not resume_path.endswith(".zip"):
        resume_path += ".zip"

    if resume_path and os.path.exists(resume_path):
        print(f"▶  Resuming from {resume_path}")
        model = RecurrentPPO.load(resume_path, env=train_env, device=device)
        model.learning_rate = learning_rate
    else:
        model = build_recurrent_ppo(
            train_env, device, seed, learning_rate,
            max_N=max_N, use_pop=use_pop,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────── #
    callbacks_list = [
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n_envs, 1),
            save_path=LOG_DIR,
            name_prefix="gravitas_rppo",
        ),
        EvalCallback(
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            eval_freq=max(EVAL_FREQ // n_envs, 1),
            best_model_save_path=LOG_DIR,
            log_path=LOG_DIR,
            deterministic=True,
            verbose=1,
        ),
        MetricsCallback(
            output_path=os.path.join(LOG_DIR, "training_metrics.json"),
            log_freq=CHECKPOINT_FREQ,
            use_pop=use_pop,
            verbose=1,
        ),
        EarlyStopCallback(
            min_action_entropy=0.3, min_reward=-500.0,
            check_freq=20_000, verbose=1,
        ),
    ]

    if mode == "curriculum":
        callbacks_list.insert(2, CurriculumCallback(verbose=1))
        print("[Curriculum] 4-phase difficulty ramp enabled.")

    callbacks = CallbackList(callbacks_list)

    # ── Train ─────────────────────────────────────────────────────────────── #
    try:
        import tqdm  # noqa
        use_pb = True
    except ImportError:
        use_pb = False

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume is None),
        progress_bar=use_pb,
    )
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────── #
    final_path = os.path.join(LOG_DIR, "gravitas_rppo_final")
    model.save(final_path)
    train_env.save(os.path.join(LOG_DIR, "vec_normalize.pkl"))
    print(f"\nFinal model → {final_path}.zip")
    print(f"Training done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    ep_buf = model.ep_info_buffer
    if ep_buf:
        rewards = [x["r"] for x in ep_buf]
        lengths = [x["l"] for x in ep_buf]
        print(f"  Final mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Final mean length : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"final/mean_reward": np.mean(rewards),
                           "final/std_reward": np.std(rewards),
                           "final/mean_length": np.mean(lengths)})
                wandb.finish()
        except ImportError:
            pass


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRAVITAS RecurrentPPO agent")
    parser.add_argument("--timesteps", type=int,   default=600_000)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--device",    type=str,   default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--n-envs",    type=int,   default=1)
    parser.add_argument("--mode",      type=str,   default="single",
                        choices=["single", "curriculum"])
    parser.add_argument("--regime",    type=str,   default=None)
    parser.add_argument("--pop",       action="store_true",
                        help="Enable population system (PopWrapper)")
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--subproc",   action="store_true")
    parser.add_argument("--config",    type=str,   default="regime_config.yaml")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        n_envs=args.n_envs,
        mode=args.mode,
        regime=args.regime,
        resume=args.resume,
        learning_rate=args.lr,
        use_subproc=args.subproc,
        use_pop=args.pop,
        config_path=args.config,
    )
