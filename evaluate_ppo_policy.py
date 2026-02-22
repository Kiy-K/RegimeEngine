"""
Evaluation Mode for trained PPO policy.

Improvements over original:
  - Bootstrap 95% confidence intervals on all key metrics
  - Collapse cause breakdown (exhaustion vs hazard vs province cascade vs EWI)
  - Topology difficulty buckets: easy / medium / hard by n_provinces
  - Early warning predictive accuracy
  - Per-episode CSV export for downstream analysis

Usage:
    python evaluate_ppo_policy.py [--model path] [--episodes 200] [--output results.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

from regime_engine.agents.survival_env import SurvivalConfig, SurvivalRegimeEnv


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _bootstrap_ci(
    values: List[float],
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Return mean + 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"mean": float(np.mean(arr)), "ci_low": lo, "ci_high": hi, "std": float(np.std(arr))}


def _infer_collapse_cause(info: Dict[str, Any]) -> str:
    """Classify terminal state into a collapse cause."""
    exh = info.get("exhaustion", 0.0)
    n_crit = info.get("n_critical_provinces", 0)
    ph = info.get("peak_hazard", 0.0)
    if exh > 0.9:
        return "exhaustion"
    if n_crit >= 3:
        return "province_cascade"
    if ph > 1.2:
        return "hazard_spike"
    return "ewi_sustained"


# ------------------------------------------------------------------ #
# Main evaluation loop                                                #
# ------------------------------------------------------------------ #

def run_evaluation(
    model_path: str,
    n_episodes: int = 200,
    output_file: str = "evaluation_results.json",
    csv_output: Optional[str] = None,
) -> None:
    print(f"Loading trained policy from {model_path} ...")
    model = PPO.load(model_path)

    config = SurvivalConfig(max_steps=500, shock_prob_per_step=0.02)
    env = SurvivalRegimeEnv(config=config)

    raw_metrics: List[Dict[str, Any]] = []
    print(f"Evaluating {n_episodes} episodes on unseen random topologies ...")

    with torch.no_grad():
        for ep in range(n_episodes):
            # Evaluation seeds start at 5000 — distinct from train (0–4999) and audit (20000+)
            obs, info = env.reset(seed=ep + 5_000)
            n_provinces = info.get("n_provinces", -1)

            terminated = truncated = False
            steps = 0
            peak_hazard = 0.0
            clustering_indices: List[float] = []
            ewi_values: List[float] = []

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                steps += 1
                peak_hazard = max(peak_hazard, info.get("peak_hazard", 0.0))
                clustering_indices.append(info.get("clustering_index", 0.0))
                ewi_values.append(info.get("early_warning", 0.0))

            collapse = bool(terminated)
            cause = _infer_collapse_cause(info) if collapse else "survived"

            record = {
                "episode": ep + 1,
                "seed": ep + 5_000,
                "n_provinces": n_provinces,
                "survival_time": steps,
                "peak_hazard": float(peak_hazard),
                "final_exhaustion": float(info.get("exhaustion", 0.0)),
                "mean_cluster_intensity": float(np.mean(clustering_indices)) if clustering_indices else 0.0,
                "mean_ewi": float(np.mean(ewi_values)) if ewi_values else 0.0,
                "max_ewi": float(np.max(ewi_values)) if ewi_values else 0.0,
                "collapse_flag": collapse,
                "collapse_cause": cause,
            }
            raw_metrics.append(record)

            if (ep + 1) % 25 == 0:
                cr_so_far = np.mean([m["collapse_flag"] for m in raw_metrics])
                print(f"  ep {ep + 1:>4d}/{n_episodes}  "
                      f"collapse_rate={cr_so_far * 100:.1f}%  "
                      f"last_survival={steps}")

    # ---------------------------------------------------------------- #
    # Aggregate statistics with bootstrap CIs                          #
    # ---------------------------------------------------------------- #
    survival_times = [m["survival_time"] for m in raw_metrics]
    peak_hazards = [m["peak_hazard"] for m in raw_metrics]
    exhaustions = [m["final_exhaustion"] for m in raw_metrics]
    collapse_flags = [float(m["collapse_flag"]) for m in raw_metrics]
    ewi_means = [m["mean_ewi"] for m in raw_metrics]
    clusterings = [m["mean_cluster_intensity"] for m in raw_metrics]

    # Collapse cause breakdown
    causes = [m["collapse_cause"] for m in raw_metrics if m["collapse_flag"]]
    cause_dist: Dict[str, float] = {}
    n_collapsed = len(causes)
    for c in causes:
        cause_dist[c] = cause_dist.get(c, 0) + 1
    cause_dist = {k: v / max(1, n_collapsed) for k, v in cause_dist.items()}

    # Difficulty buckets by province count
    prov_buckets: Dict[str, Dict[str, Any]] = {}
    for m in raw_metrics:
        np_ = m["n_provinces"]
        bucket = "easy" if np_ <= 6 else ("hard" if np_ >= 9 else "medium")
        prov_buckets.setdefault(bucket, {"survival": [], "collapses": []})
        prov_buckets[bucket]["survival"].append(m["survival_time"])
        prov_buckets[bucket]["collapses"].append(float(m["collapse_flag"]))

    difficulty_summary = {
        bucket: {
            "mean_survival": float(np.mean(d["survival"])),
            "collapse_rate": float(np.mean(d["collapses"])),
            "n_episodes": len(d["survival"]),
        }
        for bucket, d in prov_buckets.items()
    }

    # EWI predictive accuracy
    ewi_for_collapsed = [m["mean_ewi"] for m in raw_metrics if m["collapse_flag"]]
    ewi_for_survived = [m["mean_ewi"] for m in raw_metrics if not m["collapse_flag"]]

    summary = {
        "n_episodes": n_episodes,
        "survival_time": _bootstrap_ci(survival_times),
        "peak_hazard": _bootstrap_ci(peak_hazards),
        "final_exhaustion": _bootstrap_ci(exhaustions),
        "collapse_rate": _bootstrap_ci(collapse_flags),
        "mean_cluster_intensity": _bootstrap_ci(clusterings),
        "collapse_causes": cause_dist,
        "difficulty_by_topology": difficulty_summary,
        "ewi_analysis": {
            "mean_ewi_collapsed": float(np.mean(ewi_for_collapsed)) if ewi_for_collapsed else None,
            "mean_ewi_survived": float(np.mean(ewi_for_survived)) if ewi_for_survived else None,
            "separation": (
                float(np.mean(ewi_for_collapsed)) - float(np.mean(ewi_for_survived))
                if ewi_for_collapsed and ewi_for_survived else None
            ),
        },
    }

    # ---------------------------------------------------------------- #
    # Print summary                                                     #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    st = summary["survival_time"]
    cr = summary["collapse_rate"]
    ph = summary["peak_hazard"]
    fe = summary["final_exhaustion"]
    print(f"  Survival time  : {st['mean']:.1f} ± {st['std']:.1f}  "
          f"[95% CI: {st['ci_low']:.1f} – {st['ci_high']:.1f}]")
    print(f"  Peak hazard    : {ph['mean']:.4f} ± {ph['std']:.4f}  "
          f"[CI: {ph['ci_low']:.4f} – {ph['ci_high']:.4f}]")
    print(f"  Final exh.     : {fe['mean']:.4f} ± {fe['std']:.4f}  "
          f"[CI: {fe['ci_low']:.4f} – {fe['ci_high']:.4f}]")
    print(f"  Collapse rate  : {cr['mean'] * 100:.1f}%  "
          f"[CI: {cr['ci_low'] * 100:.1f}% – {cr['ci_high'] * 100:.1f}%]")
    print()
    print("  Collapse causes (% of collapses):")
    for cause, frac in sorted(cause_dist.items(), key=lambda x: -x[1]):
        print(f"    {cause:<22s}: {frac * 100:.1f}%")
    print()
    print("  Topology difficulty:")
    for bucket in ("easy", "medium", "hard"):
        if bucket in difficulty_summary:
            d = difficulty_summary[bucket]
            print(f"    {bucket:<8s}: survival={d['mean_survival']:.1f}  "
                  f"collapse={d['collapse_rate'] * 100:.1f}%  (n={d['n_episodes']})")
    ewi = summary["ewi_analysis"]
    if ewi["separation"] is not None:
        print(f"\n  EWI separation (collapsed − survived): {ewi['separation']:+.4f}")
    print("=" * 60)

    # ---------------------------------------------------------------- #
    # Persist results                                                   #
    # ---------------------------------------------------------------- #
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "episodes": raw_metrics}, f, indent=2)
    print(f"\nResults saved → {output_file}")

    if csv_output:
        fieldnames = list(raw_metrics[0].keys())
        with open(csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_metrics)
        print(f"Per-episode CSV  → {csv_output}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--model", type=str,
                        default="./logs/ppo_survival/ppo_survival_final.zip",
                        help="Path to trained model .zip")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of evaluation episodes (default: 200)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON path")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional per-episode CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found at {args.model}")
    else:
        run_evaluation(
            model_path=args.model,
            n_episodes=args.episodes,
            output_file=args.output,
            csv_output=args.csv,
        )