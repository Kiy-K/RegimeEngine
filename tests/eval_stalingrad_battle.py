#!/usr/bin/env python3
"""
eval_stalingrad_battle.py — Pit Axis vs Soviet agents in Stalingrad battles.

Runs N episodes of the multi-agent Stalingrad environment with two trained
RecurrentPPO agents (or one trained + one random) and reports detailed
per-sector, per-side, and aggregate metrics.

Usage:
    # Both agents trained
    python tests/eval_stalingrad_battle.py \
        --axis-model logs/stalingrad_selfplay/axis_final.zip \
        --soviet-model logs/stalingrad_selfplay/soviet_final.zip

    # One trained agent vs random
    python tests/eval_stalingrad_battle.py \
        --axis-model logs/stalingrad_selfplay/axis_final.zip \
        --soviet-model random

    # Use pretrained single-agent for both sides
    python tests/eval_stalingrad_battle.py \
        --axis-model logs/rppo_gravitas/gravitas_rppo_final.zip \
        --soviet-model logs/rppo_gravitas/gravitas_rppo_final.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from gravitas_engine.core.gravitas_params import GravitasParams
from gravitas_engine.agents.stalingrad_ma import (
    AXIS, SOVIET, SIDE_NAMES,
    StalingradMultiAgentEnv,
    DEFAULT_AXIS_CLUSTERS,
    DEFAULT_SOVIET_CLUSTERS,
    CONTESTED_CLUSTERS,
)
from regime_loader import load_standalone_regime, build_initial_states

try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────────── #
# Sector names from Stalingrad YAML                                            #
# ─────────────────────────────────────────────────────────────────────────── #

SECTOR_NAMES = {
    0: "StalingradCityCenter",
    1: "TractorFactoryDistrict",
    2: "MamayevKurgan",
    3: "VolgaCrossing",
    4: "NorthernDonRiverLine",
    5: "AxisSupplyCorridor",
    6: "SovietStrategicReserve",
    7: "RomanianItalianSector",
    8: "WintergewitterCorridor",
}

SECTOR_SIDES = {
    0: "Axis", 1: "Axis", 2: "Contested",
    3: "Soviet", 4: "Axis", 5: "Axis",
    6: "Soviet", 7: "Axis", 8: "Axis",
}


# ─────────────────────────────────────────────────────────────────────────── #
# Statistics                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def bootstrap_ci(data: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    rng = np.random.default_rng(12345)
    means = np.array([rng.choice(data, len(data), replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, 100 * (1 - ci) / 2)
    hi = np.percentile(means, 100 * (1 + ci) / 2)
    return float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────── #
# Load model helper                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def load_agent(model_path: str, device: str = "cpu") -> Optional[Any]:
    """Load a RecurrentPPO model, or return None for 'random'."""
    if model_path.lower() == "random":
        return None
    assert HAS_RPPO, "sb3-contrib required"
    return RecurrentPPO.load(model_path, device=device)


# ─────────────────────────────────────────────────────────────────────────── #
# Battle evaluation                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def run_battle(
    env: StalingradMultiAgentEnv,
    axis_model: Optional[Any],
    soviet_model: Optional[Any],
    n_episodes: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full Axis vs Soviet battles and collect metrics."""

    all_results = []

    for ep in range(n_episodes):
        obs_dict = env.reset(seed=seed + ep)

        axis_obs   = obs_dict[AXIS]
        soviet_obs = obs_dict[SOVIET]
        ax_lstm    = None
        sv_lstm    = None
        ax_starts  = np.ones((1,), dtype=bool)
        sv_starts  = np.ones((1,), dtype=bool)

        ax_total_r = 0.0
        sv_total_r = 0.0
        done = False
        trunc = False
        steps = 0

        # Per-step tracking
        ax_sigma_history = []
        sv_sigma_history = []
        action_counts = {AXIS: {}, SOVIET: {}}

        while not (done or trunc):
            # Axis action
            if axis_model is not None:
                ax_act, ax_lstm = axis_model.predict(
                    axis_obs.reshape(1, -1), state=ax_lstm,
                    episode_start=ax_starts, deterministic=True,
                )
                ax_action = int(ax_act[0])
            else:
                ax_action = int(np.random.randint(0, env.action_space.n))

            # Soviet action
            if soviet_model is not None:
                sv_act, sv_lstm = soviet_model.predict(
                    soviet_obs.reshape(1, -1), state=sv_lstm,
                    episode_start=sv_starts, deterministic=True,
                )
                sv_action = int(sv_act[0])
            else:
                sv_action = int(np.random.randint(0, env.action_space.n))

            actions = {AXIS: ax_action, SOVIET: sv_action}
            obs_dict, rewards, done, trunc, info = env.step(actions)

            axis_obs   = obs_dict[AXIS]
            soviet_obs = obs_dict[SOVIET]
            ax_starts  = np.zeros((1,), dtype=bool)
            sv_starts  = np.zeros((1,), dtype=bool)

            ax_total_r += rewards[AXIS]
            sv_total_r += rewards[SOVIET]
            steps += 1

            ax_sigma_history.append(info.get("axis_mean_sigma", 0.0))
            sv_sigma_history.append(info.get("soviet_mean_sigma", 0.0))

            action_counts[AXIS][ax_action] = action_counts[AXIS].get(ax_action, 0) + 1
            action_counts[SOVIET][sv_action] = action_counts[SOVIET].get(sv_action, 0) + 1

        # Final state
        w = env.world
        c_arr = w.cluster_array() if w else np.zeros((8, 6))
        N = w.n_clusters if w else 8

        # Per-sector final state
        sector_final = {}
        for i in range(min(N, 9)):
            sector_final[i] = {
                "name": SECTOR_NAMES.get(i, f"Cluster{i}"),
                "side": SECTOR_SIDES.get(i, "Unknown"),
                "sigma": float(c_arr[i, 0]),
                "hazard": float(c_arr[i, 1]),
                "resource": float(c_arr[i, 2]),
                "military": float(c_arr[i, 3]),
                "trust": float(c_arr[i, 4]),
                "polar": float(c_arr[i, 5]),
            }

        # Determine winner
        ax_final_sigma = float(np.mean(c_arr[env.axis_clusters, 0]))
        sv_final_sigma = float(np.mean(c_arr[env.soviet_clusters, 0]))
        if done:
            winner = "collapse"
        elif ax_final_sigma > sv_final_sigma + 0.05:
            winner = "Axis"
        elif sv_final_sigma > ax_final_sigma + 0.05:
            winner = "Soviet"
        else:
            winner = "draw"

        ep_result = {
            "episode": ep,
            "steps": steps,
            "axis_reward": ax_total_r,
            "soviet_reward": sv_total_r,
            "axis_final_sigma": ax_final_sigma,
            "soviet_final_sigma": sv_final_sigma,
            "exhaustion": float(w.global_state.exhaustion) if w else 0.0,
            "collapsed": done,
            "collapse_cause": info.get("collapse_cause"),
            "winner": winner,
            "sector_final": sector_final,
            "axis_action_dist": action_counts[AXIS],
            "soviet_action_dist": action_counts[SOVIET],
        }
        all_results.append(ep_result)

        if verbose:
            sym = {"Axis": "🔴", "Soviet": "🔵", "draw": "⚪", "collapse": "💀"}.get(winner, "❓")
            print(f"  Ep {ep+1:>3}/{n_episodes}  "
                  f"steps={steps:>4}  "
                  f"Ax σ̄={ax_final_sigma:.3f}  Sv σ̄={sv_final_sigma:.3f}  "
                  f"Ax R={ax_total_r:>8.1f}  Sv R={sv_total_r:>8.1f}  "
                  f"{sym} {winner}")

    return _aggregate_results(all_results, env)


def _aggregate_results(
    results: List[Dict[str, Any]],
    env: StalingradMultiAgentEnv,
) -> Dict[str, Any]:
    """Compute aggregate statistics from episode results."""
    n = len(results)
    ax_rewards = np.array([r["axis_reward"] for r in results])
    sv_rewards = np.array([r["soviet_reward"] for r in results])
    ax_sigmas  = np.array([r["axis_final_sigma"] for r in results])
    sv_sigmas  = np.array([r["soviet_final_sigma"] for r in results])
    lengths    = np.array([r["steps"] for r in results])

    winners = [r["winner"] for r in results]
    collapse_count = sum(1 for r in results if r["collapsed"])
    survival_rate  = 1.0 - collapse_count / max(n, 1)

    ax_wins  = sum(1 for w in winners if w == "Axis")
    sv_wins  = sum(1 for w in winners if w == "Soviet")
    draws    = sum(1 for w in winners if w == "draw")

    # Per-sector averages
    sector_avgs = {}
    for i in range(9):
        sigmas = [r["sector_final"][i]["sigma"] for r in results if i in r["sector_final"]]
        hazards = [r["sector_final"][i]["hazard"] for r in results if i in r["sector_final"]]
        if sigmas:
            sector_avgs[i] = {
                "name": SECTOR_NAMES.get(i, f"Cluster{i}"),
                "side": SECTOR_SIDES.get(i, "?"),
                "sigma_mean": float(np.mean(sigmas)),
                "sigma_std": float(np.std(sigmas)),
                "hazard_mean": float(np.mean(hazards)),
            }

    return {
        "n_episodes": n,
        "axis_reward_mean": float(np.mean(ax_rewards)),
        "axis_reward_std": float(np.std(ax_rewards)),
        "axis_reward_ci": bootstrap_ci(ax_rewards) if n >= 5 else None,
        "soviet_reward_mean": float(np.mean(sv_rewards)),
        "soviet_reward_std": float(np.std(sv_rewards)),
        "soviet_reward_ci": bootstrap_ci(sv_rewards) if n >= 5 else None,
        "axis_sigma_mean": float(np.mean(ax_sigmas)),
        "soviet_sigma_mean": float(np.mean(sv_sigmas)),
        "avg_length": float(np.mean(lengths)),
        "survival_rate": survival_rate,
        "axis_wins": ax_wins,
        "soviet_wins": sv_wins,
        "draws": draws,
        "collapses": collapse_count,
        "sector_averages": sector_avgs,
        "episodes": results,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Print report                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def print_report(agg: Dict[str, Any]) -> None:
    """Pretty-print battle results."""
    print("\n" + "=" * 70)
    print("       ⚔️  STALINGRAD BATTLE REPORT  ⚔️")
    print("=" * 70)

    n = agg["n_episodes"]
    print(f"\n  Episodes:       {n}")
    print(f"  Survival rate:  {agg['survival_rate']:.0%}")
    print(f"  Avg length:     {agg['avg_length']:.0f} steps")

    print(f"\n  ── Win/Loss ──────────────────────────────────")
    print(f"  🔴 Axis wins:    {agg['axis_wins']:>3} ({agg['axis_wins']/n:.0%})")
    print(f"  🔵 Soviet wins:  {agg['soviet_wins']:>3} ({agg['soviet_wins']/n:.0%})")
    print(f"  ⚪ Draws:        {agg['draws']:>3} ({agg['draws']/n:.0%})")
    print(f"  💀 Collapses:    {agg['collapses']:>3} ({agg['collapses']/n:.0%})")

    print(f"\n  ── Rewards ───────────────────────────────────")
    print(f"  Axis   mean: {agg['axis_reward_mean']:>9.2f} ± {agg['axis_reward_std']:.2f}")
    if agg.get("axis_reward_ci"):
        lo, hi = agg["axis_reward_ci"]
        print(f"         95% CI: [{lo:.2f}, {hi:.2f}]")
    print(f"  Soviet mean: {agg['soviet_reward_mean']:>9.2f} ± {agg['soviet_reward_std']:.2f}")
    if agg.get("soviet_reward_ci"):
        lo, hi = agg["soviet_reward_ci"]
        print(f"         95% CI: [{lo:.2f}, {hi:.2f}]")

    print(f"\n  ── Final Stability (σ̄) ───────────────────────")
    print(f"  Axis   σ̄: {agg['axis_sigma_mean']:.4f}")
    print(f"  Soviet σ̄: {agg['soviet_sigma_mean']:.4f}")

    print(f"\n  ── Per-Sector Averages ────────────────────────")
    for idx in sorted(agg.get("sector_averages", {}).keys()):
        s = agg["sector_averages"][idx]
        side_tag = {"Axis": "AX", "Soviet": "SV", "Contested": "CT"}.get(s["side"], "??")
        print(f"  [{idx}] {s['name']:<28s} [{side_tag}]  "
              f"σ={s['sigma_mean']:.3f}±{s['sigma_std']:.3f}  "
              f"h={s['hazard_mean']:.3f}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Stalingrad Battle Evaluation")
    parser.add_argument("--axis-model", type=str, required=True,
                        help="Path to Axis model .zip or 'random'")
    parser.add_argument("--soviet-model", type=str, required=True,
                        help="Path to Soviet model .zip or 'random'")
    parser.add_argument("--regime-file", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load regime
    regime_path = args.regime_file or str(_ROOT / "training" / "regimes" / "stalingrad.yaml")
    regime_data = load_standalone_regime(regime_path, seed=args.seed)
    params = regime_data["params"]
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

    # Build env
    env = StalingradMultiAgentEnv(
        params=params,
        axis_clusters=axis_clusters,
        soviet_clusters=soviet_clusters,
        initial_clusters=init_data.get("initial_clusters"),
        initial_alliances=init_data.get("initial_alliances"),
        seed=args.seed,
    )

    # Load models
    print(f"  Loading Axis model:   {args.axis_model}")
    axis_model = load_agent(args.axis_model, args.device)
    print(f"  Loading Soviet model: {args.soviet_model}")
    soviet_model = load_agent(args.soviet_model, args.device)

    # Run battles
    print(f"\n  Running {args.n_episodes} battles...\n")
    t0 = time.time()

    with torch.no_grad() if HAS_TORCH else nullcontext():
        agg = run_battle(
            env, axis_model, soviet_model,
            n_episodes=args.n_episodes, seed=args.seed,
            verbose=not args.quiet,
        )

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Report
    print_report(agg)

    # Save
    if args.output:
        # Remove non-serializable episode data for JSON
        save_data = {k: v for k, v in agg.items() if k != "episodes"}
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")


class nullcontext:
    """Minimal no-op context manager for Python < 3.7 compat."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


if __name__ == "__main__":
    main()
