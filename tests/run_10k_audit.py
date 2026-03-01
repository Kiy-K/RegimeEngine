"""
Large-scale stability audit: 10,000 episodes across random topologies.

Reports:
  - Survival time distribution (mean, std, percentiles, CDF)
  - Collapse rate and collapse cause breakdown
  - Early-warning vs actual-collapse correlation
  - Hazard concentration analysis (which topologies are hardest)
  - Province count sensitivity: survival vs n_provinces

Usage:
    python run_10k_audit.py [--model path/to/model.zip] [--episodes 10000]
                            [--output audit_report.json] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from gravitas_engine.agents.gravitas_env import GravitasEnv
    from gravitas_engine.core.gravitas_params import GravitasParams as GravitasConfig
    _ENV_CLASS = GravitasEnv
    _CFG_CLASS = GravitasConfig
    _ENV_NAME = "GravitasEnv"
except ImportError:
    from gravitas_engine.agents.survival_env import SurvivalRegimeEnv, SurvivalConfig
    _ENV_CLASS = SurvivalRegimeEnv
    _CFG_CLASS = SurvivalConfig
    _ENV_NAME = "SurvivalRegimeEnv"


# ------------------------------------------------------------------ #
# Single-episode runner (picklable for multiprocessing)              #
# ------------------------------------------------------------------ #

def _run_episode(args: Tuple[int, Optional[str]]) -> Dict[str, Any]:
    """Run one episode; returns a result dict. Designed for process pool."""
    seed, model_path = args

    # Import torch lazily so workers don't conflict
    if _ENV_NAME == "GravitasEnv":
        config = _CFG_CLASS(hawkes_base_rate=0.02, max_steps=500)
        env = _ENV_CLASS(params=config, seed=seed)
    else:
        config = _CFG_CLASS(shock_prob_per_step=0.02, max_steps=500)
        env = _ENV_CLASS(config=config, seed=seed)

    obs, info = env.reset(seed=seed)
    n_provinces = info.get("n_provinces", -1)

    if model_path is not None:
        import torch
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
    else:
        model = None

    terminated = False
    truncated = False
    steps = 0
    peak_hazard = 0.0
    clustering_indices: List[float] = []
    ewi_values: List[float] = []
    collapse_cause = "none"

    while not (terminated or truncated):
        if model is not None:
            import torch
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(int(action))
        steps += 1
        peak_hazard = max(peak_hazard, info.get("peak_hazard", info.get("polarization", 0.0)))
        clustering_indices.append(info.get("clustering_index", info.get("military_load", 0.0)))
        ewi_values.append(info.get("early_warning", 0.0))

    # Infer collapse cause from final info
    if terminated:
        exh = info.get("exhaustion", info.get("system", {}).get("exhaustion", 0.0))
        if isinstance(exh, dict): exh = 0.0 # fallback
        n_crit = info.get("n_critical_provinces", 0)
        ph = info.get("peak_hazard", info.get("polarization", 0.0))
        if exh > 0.9:
            collapse_cause = "exhaustion"
        elif n_crit >= 3:
            collapse_cause = "province_cascade"
        elif ph > 0.9: # Lower threshold for Gravitas polarization-based "hazard"
            collapse_cause = "systemic_collapse"
        else:
            collapse_cause = "ewi_sustained"

    return {
        "seed": seed,
        "n_provinces": n_provinces,
        "survival_time": steps,
        "peak_hazard": float(peak_hazard),
        "final_exhaustion": float(info.get("exhaustion", 0.0)),
        "mean_cluster_intensity": float(np.mean(clustering_indices)) if clustering_indices else 0.0,
        "mean_ewi": float(np.mean(ewi_values)) if ewi_values else 0.0,
        "max_ewi": float(np.max(ewi_values)) if ewi_values else 0.0,
        "collapse_flag": bool(terminated),
        "collapse_cause": collapse_cause,
    }


# ------------------------------------------------------------------ #
# Aggregate statistics                                                #
# ------------------------------------------------------------------ #

def _compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    survival_times = [r["survival_time"] for r in results]
    peak_hazards = [r["peak_hazard"] for r in results]
    exhaustions = [r["final_exhaustion"] for r in results]
    collapse_flags = [r["collapse_flag"] for r in results]
    ewi_means = [r["mean_ewi"] for r in results]
    clustering = [r["mean_cluster_intensity"] for r in results]

    # Collapse cause breakdown
    causes = [r["collapse_cause"] for r in results if r["collapse_flag"]]
    cause_counts: Dict[str, int] = {}
    for c in causes:
        cause_counts[c] = cause_counts.get(c, 0) + 1
    n_collapsed = len(causes)
    cause_fractions = {
        k: v / max(1, n_collapsed) for k, v in cause_counts.items()
    }

    # Survival by province count
    prov_survival: Dict[int, List[int]] = {}
    for r in results:
        np_ = r["n_provinces"]
        prov_survival.setdefault(np_, []).append(r["survival_time"])
    prov_summary = {
        str(np_): {
            "mean_survival": float(np.mean(times)),
            "collapse_rate": float(np.mean([
                r["collapse_flag"] for r in results if r["n_provinces"] == np_
            ])),
            "n_episodes": len(times),
        }
        for np_, times in sorted(prov_survival.items())
    }

    # EWI predictive correlation
    ewi_for_collapsed = [r["mean_ewi"] for r in results if r["collapse_flag"]]
    ewi_for_survived = [r["mean_ewi"] for r in results if not r["collapse_flag"]]

    # CDF of survival times
    sorted_times = sorted(survival_times)
    n = len(sorted_times)
    cdf_percentiles = {
        "p10": float(np.percentile(sorted_times, 10)),
        "p25": float(np.percentile(sorted_times, 25)),
        "p50": float(np.percentile(sorted_times, 50)),
        "p75": float(np.percentile(sorted_times, 75)),
        "p90": float(np.percentile(sorted_times, 90)),
        "p99": float(np.percentile(sorted_times, 99)),
    }

    # Bootstrap 95% CI on collapse rate
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot_rates = [
        float(np.mean(rng.choice(collapse_flags, size=n, replace=True)))
        for _ in range(n_boot)
    ]
    ci_low, ci_high = float(np.percentile(boot_rates, 2.5)), float(np.percentile(boot_rates, 97.5))

    return {
        "n_episodes": len(results),
        "survival_time": {
            "mean": float(np.mean(survival_times)),
            "std": float(np.std(survival_times)),
            **cdf_percentiles,
        },
        "peak_hazard": {
            "mean": float(np.mean(peak_hazards)),
            "std": float(np.std(peak_hazards)),
            "max": float(np.max(peak_hazards)),
        },
        "final_exhaustion": {
            "mean": float(np.mean(exhaustions)),
            "std": float(np.std(exhaustions)),
        },
        "collapse_rate": float(np.mean(collapse_flags)),
        "collapse_rate_95ci": [ci_low, ci_high],
        "collapse_causes": cause_fractions,
        "mean_cluster_intensity": float(np.mean(clustering)),
        "ewi_analysis": {
            "mean_ewi_collapsed": float(np.mean(ewi_for_collapsed)) if ewi_for_collapsed else None,
            "mean_ewi_survived": float(np.mean(ewi_for_survived)) if ewi_for_survived else None,
        },
        "survival_by_n_provinces": prov_summary,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("10K AUDIT REPORT")
    print("=" * 65)
    st = summary["survival_time"]
    cr = summary["collapse_rate"]
    ci = summary["collapse_rate_95ci"]
    print(f"Episodes evaluated  : {summary['n_episodes']:,}")
    print(f"Survival time       : {st['mean']:.1f} ± {st['std']:.1f} steps")
    print(f"  Percentiles       : p10={st['p10']:.0f}  p25={st['p25']:.0f}  "
          f"p50={st['p50']:.0f}  p75={st['p75']:.0f}  p90={st['p90']:.0f}")
    print(f"Collapse rate       : {cr * 100:.2f}%  "
          f"(95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%])")
    print(f"Peak hazard (mean)  : {summary['peak_hazard']['mean']:.4f}")
    print(f"Final exhaustion    : {summary['final_exhaustion']['mean']:.4f}")
    print(f"Mean cluster index  : {summary['mean_cluster_intensity']:.4f}")

    print("\nCollapse causes:")
    for cause, frac in sorted(summary["collapse_causes"].items(), key=lambda x: -x[1]):
        print(f"  {cause:<22s}: {frac * 100:.1f}%")

    print("\nSurvival by province count:")
    for n_p, stats in summary["survival_by_n_provinces"].items():
        print(f"  provinces={n_p:>2s}  mean_survival={stats['mean_survival']:.1f}  "
              f"collapse={stats['collapse_rate'] * 100:.1f}%  "
              f"(n={stats['n_episodes']})")

    ewi = summary.get("ewi_analysis", {})
    if ewi.get("mean_ewi_collapsed") is not None:
        print(f"\nEWI (mean, collapsed)  : {ewi['mean_ewi_collapsed']:.4f}")
        print(f"EWI (mean, survived)   : {ewi['mean_ewi_survived']:.4f}")
    print("=" * 65)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def run_audit(
    model_path: Optional[str] = None,
    n_episodes: int = 10_000,
    output_file: str = "audit_report.json",
    workers: int = 1,
    seed_offset: int = 0,
) -> None:
    print(f"Starting {n_episodes:,}-episode audit "
          f"({'random policy' if model_path is None else model_path}) ...")

    seeds = list(range(seed_offset, seed_offset + n_episodes))
    args = [(s, model_path) for s in seeds]

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_episode, a): a for a in args}
            completed = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                completed += 1
                if completed % 500 == 0:
                    print(f"  {completed:>6,} / {n_episodes:,} episodes done ...")
    else:
        for i, a in enumerate(args):
            results.append(_run_episode(a))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1:>6,} / {n_episodes:,} episodes done ...")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f} min).")

    summary = _compute_summary(results)
    _print_summary(summary)

    output = {"summary": summary, "episodes": results}
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create dirs if needed
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull audit saved → {output_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10k stability audit for RegimeEngine")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PPO model .zip (default: random policy)")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Number of episodes to run (default: 10,000)")
    parser.add_argument("--output", type=str, default="audit_report.json",
                        help="Output JSON path (default: audit_report.json)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1; set >1 for speed)")
    parser.add_argument("--seed-offset", type=int, default=20_000,
                        help="Starting seed (default: 20,000 — avoids train/eval overlap)")
    args = parser.parse_args()

    model_path = args.model
    if model_path and not os.path.exists(model_path):
        print(f"Warning: model not found at {model_path}. Falling back to random policy.")
        model_path = None

    run_audit(
        model_path=model_path,
        n_episodes=args.episodes,
        output_file=args.output,
        workers=args.workers,
        seed_offset=args.seed_offset,
    )