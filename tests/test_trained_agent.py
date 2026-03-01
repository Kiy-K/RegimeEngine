"""
Test / Evaluate trained RecurrentPPO agent on GravitasEnv.

Loads the final model + VecNormalize stats and runs episodes, collecting:
  - Reward statistics (mean, std, min, max, CI)
  - Episode length distribution
  - Collapse cause breakdown
  - Per-step state trajectories (sigma, hazard, trust, exhaustion, GDP, etc.)
  - Action distribution (policy entropy proxy)
  - Curriculum difficulty sweep (easy → hard)

Usage:
    python tests/test_trained_agent.py [--model PATH] [--episodes 50] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gravitas_engine.agents.gravitas_env import GravitasEnv
from gravitas_engine.core.gravitas_params import GravitasParams
from stable_baselines3.common.monitor import Monitor
from regime_loader import load_standalone_regime, build_initial_states


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

ACTION_NAMES = [
    "STABILIZE", "AID", "REFORM", "SUPPRESS",
    "MILITARY_SURGE", "MILITARY_WITHDRAW", "PROPAGANDA",
    "DIPLOMACY",
]


def bootstrap_ci(values: List[float], n_boot: int = 5000, alpha: float = 0.05) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    arr = np.array(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def make_env(seed: int, params: GravitasParams, reset_options: Optional[Dict] = None):
    def _init():
        env = GravitasEnv(params=params, seed=seed)
        if reset_options:
            env = _RegimeEnvWrapper(env, reset_options)
        return Monitor(env)
    return _init


class _RegimeEnvWrapper(gymnasium.Wrapper):
    """Injects reset options (initial states, alliances) on every reset."""
    def __init__(self, env: GravitasEnv, reset_options: Dict[str, Any]):
        super().__init__(env)
        self._reset_options = reset_options

    def reset(self, seed=None, options=None):
        merged = {**(options or {}), **self._reset_options}
        return self.env.reset(seed=seed, options=merged)


# ─────────────────────────────────────────────────────────────────────────── #
# Single episode rollout (with LSTM state tracking)                            #
# ─────────────────────────────────────────────────────────────────────────── #

def rollout_episode(
    model: RecurrentPPO,
    env: VecNormalize,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Run one full episode, returning detailed trajectory data."""
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    total_reward = 0.0
    step_count = 0
    actions_taken = []

    # Per-step tracking
    trajectory = defaultdict(list)

    done = False
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, reward, dones, infos = env.step(action)
        episode_starts = dones

        total_reward += float(reward[0])
        step_count += 1
        actions_taken.append(int(action[0]))

        info = infos[0]
        # Track key state variables from info dict
        for key in ["exhaustion", "fragmentation", "polarization", "coherence",
                     "military_str", "trust", "n_clusters"]:
            if key in info:
                trajectory[key].append(float(info[key]))

        # Economy vars
        if "mean_gdp" in info:
            trajectory["mean_gdp"].append(float(info["mean_gdp"]))
        if "mean_unemployment" in info:
            trajectory["mean_unemployment"].append(float(info["mean_unemployment"]))

        done = bool(dones[0])

    # Collapse cause from last info
    collapse_cause = infos[0].get("collapse_cause", None)
    survived = collapse_cause is None

    return {
        "reward": total_reward,
        "length": step_count,
        "survived": survived,
        "collapse_cause": collapse_cause,
        "actions": actions_taken,
        "trajectory": dict(trajectory),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main evaluation                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def evaluate(
    model_path: str,
    vec_norm_path: Optional[str] = None,
    n_episodes: int = 50,
    deterministic: bool = True,
    verbose: bool = False,
    difficulty_sweep: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation suite."""

    print("=" * 65)
    print("GRAVITAS — Agent Evaluation")
    print("=" * 65)
    print(f"  Model        : {model_path}")
    print(f"  Episodes     : {n_episodes}")
    print(f"  Deterministic: {deterministic}")

    # ── Load model ────────────────────────────────────────────────────── #
    model = RecurrentPPO.load(model_path)
    print(f"  Policy       : {type(model.policy).__name__}")

    # ── Build eval env ────────────────────────────────────────────────── #
    params = GravitasParams(seed=9999)
    vec_env = DummyVecEnv([make_env(9999, params)])
    eval_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)

    # Load normalization stats if available
    if vec_norm_path and os.path.exists(vec_norm_path):
        eval_env = VecNormalize.load(vec_norm_path, vec_env)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"  VecNormalize : loaded from {vec_norm_path}")
    else:
        print(f"  VecNormalize : fresh (no stats loaded)")

    print("=" * 65)
    print()

    # ── Run episodes ──────────────────────────────────────────────────── #
    results = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        ep_data = rollout_episode(model, eval_env, deterministic=deterministic)
        results.append(ep_data)

        if verbose:
            status = "SURVIVED" if ep_data["survived"] else f"COLLAPSED ({ep_data['collapse_cause']})"
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"rew={ep_data['reward']:8.1f}  len={ep_data['length']:4d}  {status}")

    elapsed = time.perf_counter() - t0

    # ── Aggregate stats ───────────────────────────────────────────────── #
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    survived = [r["survived"] for r in results]
    causes = [r["collapse_cause"] for r in results if r["collapse_cause"]]

    reward_stats = bootstrap_ci(rewards)
    length_stats = bootstrap_ci(lengths)
    survival_rate = sum(survived) / len(survived)
    cause_counts = dict(Counter(causes))

    # Action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r["actions"])
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    action_dist = {}
    for i in range(8):
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"ACTION_{i}"
        count = action_counts.get(i, 0)
        action_dist[name] = f"{count/total_actions*100:.1f}%"

    # Terminal state analysis (from trajectories)
    final_exhaustion = [r["trajectory"]["exhaustion"][-1] for r in results if "exhaustion" in r["trajectory"] and r["trajectory"]["exhaustion"]]
    final_polarization = [r["trajectory"]["polarization"][-1] for r in results if "polarization" in r["trajectory"] and r["trajectory"]["polarization"]]

    # ── Print report ──────────────────────────────────────────────────── #
    print()
    print("━" * 65)
    print("  EVALUATION REPORT")
    print("━" * 65)

    print(f"\n  📊 Reward")
    print(f"     Mean   : {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
    print(f"     95% CI : [{reward_stats['ci_low']:.2f}, {reward_stats['ci_high']:.2f}]")
    print(f"     Range  : [{reward_stats['min']:.1f}, {reward_stats['max']:.1f}]")

    print(f"\n  📏 Episode Length")
    print(f"     Mean   : {length_stats['mean']:.1f} ± {length_stats['std']:.1f}")
    print(f"     Range  : [{length_stats['min']:.0f}, {length_stats['max']:.0f}]")

    print(f"\n  🛡️  Survival Rate: {survival_rate*100:.1f}% ({sum(survived)}/{len(survived)})")
    if cause_counts:
        print(f"     Collapse causes:")
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            print(f"       {cause:25s} : {count:3d} ({count/len(results)*100:.1f}%)")

    print(f"\n  🎮 Action Distribution")
    for name, pct in sorted(action_dist.items(), key=lambda x: -float(x[1].rstrip('%'))):
        print(f"     {name:20s} : {pct}")

    if final_exhaustion:
        print(f"\n  📈 Terminal State (mean over episodes)")
        print(f"     Exhaustion   : {np.mean(final_exhaustion):.4f}")
    if final_polarization:
        print(f"     Polarization : {np.mean(final_polarization):.4f}")

    print(f"\n  ⏱️  Evaluation time: {elapsed:.1f}s ({elapsed/n_episodes:.2f}s/episode)")
    print("━" * 65)

    # ── Difficulty sweep ──────────────────────────────────────────────── #
    sweep_results = {}
    if difficulty_sweep:
        print("\n  🔬 Difficulty Sweep")
        print("  " + "-" * 55)

        difficulties = [
            ("Easy",   {"hawkes_base_rate": 0.005, "max_steps": 400}),
            ("Medium", {"hawkes_base_rate": 0.015, "max_steps": 500}),
            ("Hard",   {"hawkes_base_rate": 0.03,  "max_steps": 700}),
            ("Extreme",{"hawkes_base_rate": 0.05,  "max_steps": 800}),
        ]
        n_sweep = min(20, n_episodes)

        for diff_name, overrides in difficulties:
            sweep_params = GravitasParams(seed=7777, **overrides)
            sweep_vec = DummyVecEnv([make_env(7777, sweep_params)])
            if vec_norm_path and os.path.exists(vec_norm_path):
                sweep_env = VecNormalize.load(vec_norm_path, sweep_vec)
                sweep_env.training = False
                sweep_env.norm_reward = False
            else:
                sweep_env = VecNormalize(sweep_vec, norm_obs=True, norm_reward=False,
                                         training=False, clip_obs=10.0)

            rews, survs = [], []
            for _ in range(n_sweep):
                ep = rollout_episode(model, sweep_env, deterministic=deterministic)
                rews.append(ep["reward"])
                survs.append(ep["survived"])
            sweep_env.close()

            sr = sum(survs) / len(survs) * 100
            sweep_results[diff_name] = {
                "mean_reward": float(np.mean(rews)),
                "survival_rate": sr,
            }
            print(f"     {diff_name:8s} | reward={np.mean(rews):8.1f} ± {np.std(rews):6.1f} | survival={sr:5.1f}%")

        print("  " + "-" * 55)

    eval_env.close()

    # ── Save results ──────────────────────────────────────────────────── #
    output = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "reward": reward_stats,
        "length": length_stats,
        "survival_rate": survival_rate,
        "collapse_causes": cause_counts,
        "action_distribution": action_dist,
        "difficulty_sweep": sweep_results,
    }
    return output


# ─────────────────────────────────────────────────────────────────────────── #
# Regime-specific evaluation (e.g. Stalingrad)                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def evaluate_regime(
    model_path: str,
    regime_path: str,
    vec_norm_path: Optional[str] = None,
    n_episodes: int = 30,
    deterministic: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate agent on a specific regime scenario (standalone YAML)."""

    regime_data = load_standalone_regime(regime_path, seed=7777)
    params = regime_data["params"]
    sectors = regime_data["sectors"]

    # The trained model expects max_N=12 obs space. Override n_clusters_max
    # to match while injecting actual N via reset options.
    TRAINED_MAX_N = 12
    if params.n_clusters_max != TRAINED_MAX_N:
        from dataclasses import replace
        params = replace(params, n_clusters_max=TRAINED_MAX_N)

    init = build_initial_states(regime_data, TRAINED_MAX_N)
    reset_opts = {
        "n_clusters": init["n_clusters"],
        "initial_clusters": init["initial_clusters"],
        "initial_alliances": init["initial_alliances"],
    }

    sector_names = {s["id"]: s["name"] for s in sectors}
    sector_sides = {s["id"]: s.get("side", "?") for s in sectors}
    N = len(sectors)

    print("=" * 65)
    print(f"GRAVITAS — Regime Evaluation: {os.path.basename(regime_path)}")
    print("=" * 65)
    print(f"  Model        : {model_path}")
    print(f"  Clusters     : {N}")
    print(f"  Max steps    : {params.max_steps}")
    print(f"  Hawkes rate  : {params.hawkes_base_rate}")
    print(f"  Sectors:")
    for s in sectors:
        st = s.get("initial_state", {})
        print(f"    [{s['id']}] {s['name']:25s} ({s.get('side','?'):10s}) "
              f"σ={st.get('sigma',0):.2f} h={st.get('hazard',0):.2f} "
              f"r={st.get('resource',0):.2f} m={st.get('military',0):.2f}")
    if regime_data["custom_shocks"]:
        print(f"  Custom shocks: {len(regime_data['custom_shocks'])}")
        for sh in regime_data["custom_shocks"]:
            print(f"    {sh['name']:25s} (p={sh['probability']:.3f}) — {sh.get('description', '')[:50]}")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────── #
    model = RecurrentPPO.load(model_path)

    # ── Build regime env ──────────────────────────────────────────────── #
    vec_env = DummyVecEnv([make_env(7777, params, reset_options=reset_opts)])
    if vec_norm_path and os.path.exists(vec_norm_path):
        eval_env = VecNormalize.load(vec_norm_path, vec_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                                training=False, clip_obs=10.0)

    # ── Run episodes ──────────────────────────────────────────────────── #
    results = []
    t0 = time.perf_counter()

    for ep in range(n_episodes):
        ep_data = rollout_episode(model, eval_env, deterministic=deterministic)
        results.append(ep_data)
        if verbose:
            status = "SURVIVED" if ep_data["survived"] else f"COLLAPSED ({ep_data['collapse_cause']})"
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"rew={ep_data['reward']:8.1f}  len={ep_data['length']:4d}  {status}")

    elapsed = time.perf_counter() - t0
    eval_env.close()

    # ── Aggregate ─────────────────────────────────────────────────────── #
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    survived_list = [r["survived"] for r in results]
    causes = [r["collapse_cause"] for r in results if r["collapse_cause"]]
    survival_rate = sum(survived_list) / len(survived_list)
    cause_counts = dict(Counter(causes))

    all_actions = []
    for r in results:
        all_actions.extend(r["actions"])
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    action_dist = {}
    for i in range(max(action_counts.keys()) + 1 if action_counts else 8):
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"ACTION_{i}"
        count = action_counts.get(i, 0)
        action_dist[name] = f"{count/total_actions*100:.1f}%"

    # Terminal state metrics
    final_exh = [r["trajectory"]["exhaustion"][-1] for r in results
                 if "exhaustion" in r["trajectory"] and r["trajectory"]["exhaustion"]]
    final_pol = [r["trajectory"]["polarization"][-1] for r in results
                 if "polarization" in r["trajectory"] and r["trajectory"]["polarization"]]

    # ── Print regime report ───────────────────────────────────────────── #
    print()
    print("━" * 65)
    regime_name = os.path.splitext(os.path.basename(regime_path))[0].upper()
    print(f"  {regime_name} — EVALUATION REPORT")
    print("━" * 65)

    rstat = bootstrap_ci(rewards)
    print(f"\n  📊 Reward")
    print(f"     Mean   : {rstat['mean']:.2f} ± {rstat['std']:.2f}")
    print(f"     95% CI : [{rstat['ci_low']:.2f}, {rstat['ci_high']:.2f}]")
    print(f"     Range  : [{rstat['min']:.1f}, {rstat['max']:.1f}]")

    lstat = bootstrap_ci(lengths)
    print(f"\n  📏 Episode Length")
    print(f"     Mean   : {lstat['mean']:.1f} ± {lstat['std']:.1f}")
    print(f"     Range  : [{lstat['min']:.0f}, {lstat['max']:.0f}]")

    print(f"\n  🛡️  Survival Rate: {survival_rate*100:.1f}% ({sum(survived_list)}/{len(survived_list)})")
    if cause_counts:
        print(f"     Collapse causes:")
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            print(f"       {cause:25s} : {count:3d} ({count/len(results)*100:.1f}%)")

    print(f"\n  🎮 Action Distribution")
    for name, pct in sorted(action_dist.items(), key=lambda x: -float(x[1].rstrip('%'))):
        print(f"     {name:20s} : {pct}")

    if final_exh:
        print(f"\n  📈 Terminal State (mean)")
        print(f"     Exhaustion   : {np.mean(final_exh):.4f}")
    if final_pol:
        print(f"     Polarization : {np.mean(final_pol):.4f}")

    print(f"\n  ⏱️  {elapsed:.1f}s ({elapsed/n_episodes:.2f}s/ep)")
    print("━" * 65)

    return {
        "regime": regime_path,
        "n_episodes": n_episodes,
        "reward": rstat,
        "length": lstat,
        "survival_rate": survival_rate,
        "collapse_causes": cause_counts,
        "action_distribution": action_dist,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Stochastic vs Deterministic comparison                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def compare_det_vs_stochastic(
    model_path: str,
    vec_norm_path: Optional[str],
    n_episodes: int = 20,
) -> None:
    """Quick comparison: deterministic vs stochastic policy."""
    print("\n  🎲 Deterministic vs Stochastic Comparison")
    print("  " + "-" * 55)

    for mode, det in [("Deterministic", True), ("Stochastic", False)]:
        model = RecurrentPPO.load(model_path)
        params = GravitasParams(seed=8888)
        vec_env = DummyVecEnv([make_env(8888, params)])
        if vec_norm_path and os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, vec_env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               training=False, clip_obs=10.0)

        rews = []
        for _ in range(n_episodes):
            ep = rollout_episode(model, env, deterministic=det)
            rews.append(ep["reward"])
        env.close()

        print(f"     {mode:15s} | reward={np.mean(rews):8.1f} ± {np.std(rews):6.1f}")
    print("  " + "-" * 55)


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained GRAVITAS RecurrentPPO agent")
    parser.add_argument("--model", type=str,
                        default="logs/rppo_gravitas/gravitas_rppo_final.zip",
                        help="Path to trained model .zip")
    parser.add_argument("--vec-norm", type=str,
                        default="logs/rppo_gravitas/vec_normalize.pkl",
                        help="Path to VecNormalize stats")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-sweep", action="store_true", help="Skip difficulty sweep")
    parser.add_argument("--compare", action="store_true", help="Compare det vs stochastic")
    parser.add_argument("--regime-file", type=str, default=None,
                        help="Standalone regime YAML (e.g. training/regimes/stalingrad.yaml)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    if args.regime_file:
        output = evaluate_regime(
            model_path=args.model,
            regime_path=args.regime_file,
            vec_norm_path=args.vec_norm,
            n_episodes=args.episodes,
            verbose=args.verbose,
        )
    else:
        output = evaluate(
            model_path=args.model,
            vec_norm_path=args.vec_norm,
            n_episodes=args.episodes,
            verbose=args.verbose,
            difficulty_sweep=not args.no_sweep,
        )

        if args.compare:
            compare_det_vs_stochastic(args.model, args.vec_norm)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {args.output}")
