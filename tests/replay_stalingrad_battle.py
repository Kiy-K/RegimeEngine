#!/usr/bin/env python3
"""
replay_stalingrad_battle.py — Detailed turn-by-turn Stalingrad battle replay.

Plays a single full game between trained Axis and Soviet agents with rich
sector-by-sector commentary, showing which sectors are falling, holding,
being reinforced, and key turning points in the battle.

Usage:
    python tests/replay_stalingrad_battle.py \
        --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
        --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from gravitas_engine.agents.gravitas_actions import Stance, N_STANCES
from regime_loader import load_standalone_regime, build_initial_states

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERROR: sb3-contrib required"); sys.exit(1)

import torch

# ─────────────────────────────────────────────────────────────────────────── #
# Sector metadata                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

SECTOR_NAMES = {
    0: "Stalingrad City Center",
    1: "Tractor Factory District",
    2: "Mamayev Kurgan",
    3: "Volga Crossing",
    4: "Northern Don River Line",
    5: "Axis Supply Corridor",
    6: "Soviet Strategic Reserve",
    7: "Romanian/Italian Sector",
    8: "Wintergewitter Corridor",
}

SECTOR_SIDES = {
    0: "Axis", 1: "Axis", 2: "Contested",
    3: "Soviet", 4: "Axis", 5: "Axis",
    6: "Soviet", 7: "Axis", 8: "Axis",
}

SECTOR_ICONS = {
    0: "🏙️", 1: "🏭", 2: "⛰️", 3: "🚢", 4: "🌊",
    5: "📦", 6: "⭐", 7: "🇷🇴", 8: "🛡️",
}

STANCE_NAMES = {
    0: "MILITARIZE", 1: "REFORM", 2: "INVEST", 3: "STABILIZE",
    4: "PROPAGANDA", 5: "DECENTRALIZE", 6: "DIPLOMACY", 7: "REDEPLOY",
}

STANCE_ICONS = {
    0: "⚔️", 1: "📜", 2: "💰", 3: "🏗️",
    4: "📢", 5: "🔀", 6: "🤝", 7: "🔄",
}


# ─────────────────────────────────────────────────────────────────────────── #
# Commentary engine                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def sigma_status(sigma: float) -> str:
    if sigma >= 0.95:   return "FORTIFIED"
    if sigma >= 0.85:   return "HOLDING"
    if sigma >= 0.70:   return "CONTESTED"
    if sigma >= 0.50:   return "WAVERING"
    if sigma >= 0.30:   return "CRUMBLING"
    if sigma >= 0.10:   return "COLLAPSING"
    return "FALLEN"

def sigma_bar(sigma: float, width: int = 20) -> str:
    filled = int(sigma * width)
    return "█" * filled + "░" * (width - filled)

def hazard_desc(h: float) -> str:
    if h >= 0.7: return "INFERNO"
    if h >= 0.5: return "HEAVY FIRE"
    if h >= 0.3: return "UNDER FIRE"
    if h >= 0.1: return "SKIRMISHES"
    return "QUIET"

def delta_arrow(delta: float) -> str:
    if delta > 0.03:  return "⬆️ SURGING"
    if delta > 0.01:  return "↗️ improving"
    if delta > -0.01: return "➡️ stable"
    if delta > -0.03: return "↘️ declining"
    return "⬇️ FALLING"

def military_desc(m: float) -> str:
    if m >= 0.8: return "Full Strength"
    if m >= 0.6: return "Combat Ready"
    if m >= 0.4: return "Depleted"
    if m >= 0.2: return "Weakened"
    return "Remnants"


def print_sector_status(
    idx: int, c_arr: np.ndarray, prev_arr: Optional[np.ndarray] = None
) -> None:
    """Print detailed status for one sector."""
    s = c_arr[idx]
    sigma, hazard, resource, military, trust, polar = s[0], s[1], s[2], s[3], s[4], s[5]
    icon = SECTOR_ICONS.get(idx, "📍")
    name = SECTOR_NAMES.get(idx, f"Cluster {idx}")
    side = SECTOR_SIDES.get(idx, "?")
    side_tag = "🔴AX" if side == "Axis" else ("🔵SV" if side == "Soviet" else "⚪CT")

    delta = 0.0
    if prev_arr is not None:
        delta = sigma - prev_arr[idx, 0]

    arrow = delta_arrow(delta)
    bar = sigma_bar(sigma)
    status = sigma_status(sigma)

    print(f"    {icon} [{idx}] {name:<28s} [{side_tag}]")
    print(f"       Stability: {bar} {sigma:.3f} ({status}) {arrow}")
    print(f"       Military: {military:.2f} ({military_desc(military)})  "
          f"Hazard: {hazard:.3f} ({hazard_desc(hazard)})  "
          f"Resources: {resource:.2f}  Trust: {trust:.2f}  Polar: {polar:.2f}")


def print_battle_header() -> None:
    print()
    print("╔" + "═" * 72 + "╗")
    print("║" + "  ⚔️  BATTLE OF STALINGRAD — FULL REPLAY  ⚔️  ".center(72) + "║")
    print("║" + "  Axis (Wehrmacht) vs Soviet (Red Army)  ".center(72) + "║")
    print("╚" + "═" * 72 + "╝")
    print()


def print_phase_banner(step: int, max_steps: int) -> None:
    pct = step / max_steps * 100
    if step == 0:
        phase = "🌅 OPENING PHASE — Initial Deployments"
    elif pct <= 20:
        phase = "⚔️ EARLY BATTLE — First Contact"
    elif pct <= 40:
        phase = "🔥 ESCALATION — Main Assault"
    elif pct <= 60:
        phase = "💀 CRISIS POINT — Peak Intensity"
    elif pct <= 80:
        phase = "🏳️ LATE WAR — Exhaustion & Attrition"
    else:
        phase = "🏁 ENDGAME — Final Struggle"

    print(f"\n{'─' * 72}")
    print(f"  {phase}")
    print(f"  Turn {step}/{max_steps} ({pct:.0f}%)")
    print(f"{'─' * 72}")


def generate_commentary(
    step: int,
    c_arr: np.ndarray,
    prev_arr: np.ndarray,
    info: Dict[str, Any],
    ax_action: int,
    sv_action: int,
    ax_reward: float,
    sv_reward: float,
    events: List[str],
) -> List[str]:
    """Generate narrative commentary for the current turn."""
    lines = []
    N = min(c_arr.shape[0], 9)

    # Check for significant sector changes
    for i in range(N):
        delta = c_arr[i, 0] - prev_arr[i, 0]
        name = SECTOR_NAMES.get(i, f"Cluster{i}")
        side = SECTOR_SIDES.get(i, "?")

        if delta < -0.08:
            lines.append(f"  💥 {name} is UNDER HEAVY ASSAULT! (σ dropped {delta:+.3f})")
        elif delta < -0.04:
            lines.append(f"  ⚠️  {name} is taking damage (σ {delta:+.3f})")
        elif delta > 0.05:
            lines.append(f"  🛡️  {name} reinforced and stabilizing (σ {delta:+.3f})")

        # Sector falling below critical thresholds
        if c_arr[i, 0] < 0.30 and prev_arr[i, 0] >= 0.30:
            lines.append(f"  🚨 CRITICAL: {name} is CRUMBLING! σ={c_arr[i,0]:.3f}")
        if c_arr[i, 0] < 0.10 and prev_arr[i, 0] >= 0.10:
            lines.append(f"  💀 {name} has FALLEN! σ={c_arr[i,0]:.3f}")

        # Hazard spikes
        h_delta = c_arr[i, 1] - prev_arr[i, 1]
        if h_delta > 0.1:
            lines.append(f"  🔥 Intense fighting erupts at {name}! (hazard +{h_delta:.3f})")

    # Shock events
    if info.get("shock_occurred"):
        lines.append(f"  ⚡ SHOCK EVENT triggered this turn!")

    # Volga reinforcements
    reinf = info.get("volga_reinforcements", 0)
    if reinf > 0 and step % 50 == 0:
        volga_sigma = c_arr[3, 0] if c_arr.shape[0] > 3 else 0
        if volga_sigma > 0.5:
            lines.append(f"  🚢 VOLGA REINFORCEMENTS ARRIVE! Barges cross under fire. "
                        f"Soviet reserves boosted! (Reinforcement #{reinf})")

    return lines


# ─────────────────────────────────────────────────────────────────────────── #
# Main replay                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def replay_battle(
    env: StalingradMultiAgentEnv,
    axis_model: Any,
    soviet_model: Any,
    seed: int = 42,
    report_interval: int = 25,
) -> Dict[str, Any]:
    """Run a single full battle with detailed turn-by-turn commentary."""

    print_battle_header()

    obs_dict = env.reset(seed=seed)
    axis_obs = obs_dict[AXIS]
    soviet_obs = obs_dict[SOVIET]
    ax_lstm, sv_lstm = None, None
    ax_starts = np.ones((1,), dtype=bool)
    sv_starts = np.ones((1,), dtype=bool)

    ax_total_r = 0.0
    sv_total_r = 0.0

    max_steps = env.params.max_steps
    N = env._cur_N

    # Get initial state
    c_arr = env.world.cluster_array()
    prev_arr = c_arr.copy()

    # Print initial deployment
    print_phase_banner(0, max_steps)
    print("\n  📋 INITIAL DEPLOYMENT:")
    for i in range(min(N, 9)):
        print_sector_status(i, c_arr)
    print()

    ax_sigma_hist = [float(np.mean(c_arr[env.axis_clusters, 0]))]
    sv_sigma_hist = [float(np.mean(c_arr[env.soviet_clusters, 0]))]

    # Track key events for the after-action report
    key_events: List[Tuple[int, str]] = []
    sector_min_sigma = {i: float(c_arr[i, 0]) for i in range(min(N, 9))}
    sector_max_sigma = {i: float(c_arr[i, 0]) for i in range(min(N, 9))}

    done = False
    trunc = False
    step = 0
    last_phase = ""

    with torch.no_grad():
        while not (done or trunc):
            # Get actions
            ax_act, ax_lstm = axis_model.predict(
                axis_obs.reshape(1, -1), state=ax_lstm,
                episode_start=ax_starts, deterministic=True,
            )
            sv_act, sv_lstm = soviet_model.predict(
                soviet_obs.reshape(1, -1), state=sv_lstm,
                episode_start=sv_starts, deterministic=True,
            )
            ax_action = int(ax_act[0])
            sv_action = int(sv_act[0])

            actions = {AXIS: ax_action, SOVIET: sv_action}
            obs_dict, rewards, done, trunc, info = env.step(actions)

            axis_obs = obs_dict[AXIS]
            soviet_obs = obs_dict[SOVIET]
            ax_starts = np.zeros((1,), dtype=bool)
            sv_starts = np.zeros((1,), dtype=bool)

            ax_total_r += rewards[AXIS]
            sv_total_r += rewards[SOVIET]
            step += 1

            c_arr = env.world.cluster_array()

            # Track min/max sigma per sector
            for i in range(min(N, 9)):
                sector_min_sigma[i] = min(sector_min_sigma[i], float(c_arr[i, 0]))
                sector_max_sigma[i] = max(sector_max_sigma[i], float(c_arr[i, 0]))

            ax_sigma = float(np.mean(c_arr[env.axis_clusters, 0]))
            sv_sigma = float(np.mean(c_arr[env.soviet_clusters, 0]))
            ax_sigma_hist.append(ax_sigma)
            sv_sigma_hist.append(sv_sigma)

            # Generate commentary events
            commentary = generate_commentary(
                step, c_arr, prev_arr, info, ax_action, sv_action,
                rewards[AXIS], rewards[SOVIET], [],
            )

            for line in commentary:
                key_events.append((step, line.strip()))

            # Detailed report at intervals
            if step % report_interval == 0 or done or trunc:
                # Phase banner (only when phase changes)
                pct = step / max_steps * 100
                if pct <= 20:   cur_phase = "EARLY"
                elif pct <= 40: cur_phase = "ESCALATION"
                elif pct <= 60: cur_phase = "CRISIS"
                elif pct <= 80: cur_phase = "LATE"
                else:           cur_phase = "ENDGAME"

                if cur_phase != last_phase:
                    print_phase_banner(step, max_steps)
                    last_phase = cur_phase

                # Turn header
                ax_stance = STANCE_NAMES.get(ax_action, "?")
                sv_stance = STANCE_NAMES.get(sv_action, "?")
                ax_icon = STANCE_ICONS.get(ax_action, "?")
                sv_icon = STANCE_ICONS.get(sv_action, "?")

                print(f"\n  ┌─ Turn {step:>3} {'─' * 55}")
                print(f"  │ 🔴 Axis orders:  {ax_icon} {ax_stance:<16s}  "
                      f"│ 🔵 Soviet orders: {sv_icon} {sv_stance}")
                print(f"  │ Axis reward: {rewards[AXIS]:>+7.2f} (total: {ax_total_r:>+8.1f})  "
                      f"│ Soviet reward: {rewards[SOVIET]:>+7.2f} (total: {sv_total_r:>+8.1f})")
                print(f"  │ Axis σ̄: {ax_sigma:.3f}  │  Soviet σ̄: {sv_sigma:.3f}  "
                      f"│  Exhaustion: {info.get('exhaustion', 0):.3f}")

                # Print commentary
                if commentary:
                    print(f"  │")
                    for line in commentary:
                        print(f"  │ {line}")

                # Sector status table
                print(f"  │")
                print(f"  │  {'Sector':<30s} {'Side':>4s}  {'σ':>6s}  {'Δσ':>7s}  "
                      f"{'Status':<12s}  {'Mil':>5s}  {'Haz':>5s}  {'Res':>5s}")
                print(f"  │  {'─'*30} {'─'*4}  {'─'*6}  {'─'*7}  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*5}")

                for i in range(min(N, 9)):
                    name = SECTOR_NAMES.get(i, f"Cluster{i}")
                    icon = SECTOR_ICONS.get(i, "📍")
                    side = SECTOR_SIDES.get(i, "?")
                    stag = "AX" if side == "Axis" else ("SV" if side == "Soviet" else "CT")
                    sigma = c_arr[i, 0]
                    delta = sigma - prev_arr[i, 0]
                    status = sigma_status(sigma)
                    mil = c_arr[i, 3]
                    haz = c_arr[i, 1]
                    res = c_arr[i, 2]

                    # Color-code delta
                    if delta > 0.01:
                        d_str = f"  +{delta:.3f}"
                    elif delta < -0.01:
                        d_str = f"  {delta:.3f}"
                    else:
                        d_str = f"   {delta:+.3f}"

                    print(f"  │  {icon} {name:<27s} {stag:>4s}  {sigma:>6.3f} {d_str}  "
                          f"{status:<12s}  {mil:>5.2f}  {haz:>5.3f}  {res:>5.2f}")

                # Overall momentum
                ax_momentum = ax_sigma - ax_sigma_hist[max(0, step - report_interval)]
                sv_momentum = sv_sigma - sv_sigma_hist[max(0, step - report_interval)]
                print(f"  │")
                if ax_momentum > sv_momentum + 0.02:
                    print(f"  │  📊 Momentum: 🔴 AXIS ADVANCING (Δ={ax_momentum:+.3f} vs {sv_momentum:+.3f})")
                elif sv_momentum > ax_momentum + 0.02:
                    print(f"  │  📊 Momentum: 🔵 SOVIET PUSHING (Δ={sv_momentum:+.3f} vs {ax_momentum:+.3f})")
                else:
                    print(f"  │  📊 Momentum: ⚖️  STALEMATE (Ax Δ={ax_momentum:+.3f}, Sv Δ={sv_momentum:+.3f})")

                print(f"  └{'─' * 68}")

            prev_arr = c_arr.copy()

    # ── Final Report ────────────────────────────────────────────────────── #
    c_final = env.world.cluster_array()
    ax_final = float(np.mean(c_final[env.axis_clusters, 0]))
    sv_final = float(np.mean(c_final[env.soviet_clusters, 0]))

    if done:
        winner = "COLLAPSE"
    elif ax_final > sv_final + 0.05:
        winner = "AXIS VICTORY"
    elif sv_final > ax_final + 0.05:
        winner = "SOVIET VICTORY"
    else:
        winner = "DRAW"

    win_icon = {"AXIS VICTORY": "🔴", "SOVIET VICTORY": "🔵", "DRAW": "⚪", "COLLAPSE": "💀"}.get(winner, "❓")

    print("\n")
    print("╔" + "═" * 72 + "╗")
    print("║" + f"  {win_icon}  {winner}  {win_icon}  ".center(72) + "║")
    print("╚" + "═" * 72 + "╝")

    print(f"\n  📊 FINAL SCOREBOARD")
    print(f"  {'─' * 50}")
    print(f"  🔴 Axis:   Total Reward = {ax_total_r:>+10.1f}  │  Final σ̄ = {ax_final:.3f}")
    print(f"  🔵 Soviet: Total Reward = {sv_total_r:>+10.1f}  │  Final σ̄ = {sv_final:.3f}")
    print(f"  Exhaustion: {env.world.global_state.exhaustion:.3f}")
    print(f"  Battle Duration: {step} turns")
    reinf = getattr(env, '_volga_reinforced', 0)
    print(f"  Volga Reinforcements: {reinf}x delivered")

    # Final sector status
    print(f"\n  📋 FINAL SECTOR STATUS")
    print(f"  {'─' * 50}")
    for i in range(min(N, 9)):
        print_sector_status(i, c_final)
    print()

    # Sector highlights
    print(f"  🏆 SECTOR HIGHLIGHTS")
    print(f"  {'─' * 50}")

    # Most contested (highest sigma swing)
    max_swing_idx = max(range(min(N, 9)),
                        key=lambda i: sector_max_sigma[i] - sector_min_sigma[i])
    swing = sector_max_sigma[max_swing_idx] - sector_min_sigma[max_swing_idx]
    print(f"  ⚔️  Most Contested: {SECTOR_NAMES[max_swing_idx]} "
          f"(σ swing: {sector_min_sigma[max_swing_idx]:.3f} → {sector_max_sigma[max_swing_idx]:.3f}, Δ={swing:.3f})")

    # Most stable
    min_swing_idx = min(range(min(N, 9)),
                        key=lambda i: sector_max_sigma[i] - sector_min_sigma[i])
    print(f"  🏗️  Most Stable: {SECTOR_NAMES[min_swing_idx]} "
          f"(σ range: {sector_min_sigma[min_swing_idx]:.3f}–{sector_max_sigma[min_swing_idx]:.3f})")

    # Weakest sector at end
    weakest = min(range(min(N, 9)), key=lambda i: c_final[i, 0])
    print(f"  💀 Weakest Sector: {SECTOR_NAMES[weakest]} (σ={c_final[weakest, 0]:.3f})")

    # Strongest
    strongest = max(range(min(N, 9)), key=lambda i: c_final[i, 0])
    print(f"  🛡️  Strongest Sector: {SECTOR_NAMES[strongest]} (σ={c_final[strongest, 0]:.3f})")

    # Key events log
    if key_events:
        print(f"\n  📜 KEY EVENTS LOG ({len(key_events)} events)")
        print(f"  {'─' * 50}")
        # Show up to 30 most important events
        shown = 0
        for turn, event in key_events:
            if shown >= 40:
                print(f"  ... and {len(key_events) - shown} more events")
                break
            print(f"  Turn {turn:>3}: {event}")
            shown += 1

    # Sigma trajectory summary
    print(f"\n  📈 STABILITY TRAJECTORY (σ̄ by phase)")
    print(f"  {'─' * 50}")
    checkpoints = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for cp in checkpoints:
        if cp < len(ax_sigma_hist):
            ax_s = ax_sigma_hist[cp]
            sv_s = sv_sigma_hist[cp]
            leader = "🔴 Axis" if ax_s > sv_s + 0.02 else ("🔵 Soviet" if sv_s > ax_s + 0.02 else "⚖️ Even")
            print(f"  Turn {cp:>3}: Axis σ̄={ax_s:.3f}  Soviet σ̄={sv_s:.3f}  │ {leader}")

    print(f"\n{'═' * 72}")
    print(f"  End of Battle Report")
    print(f"{'═' * 72}\n")

    return {
        "winner": winner,
        "steps": step,
        "axis_reward": ax_total_r,
        "soviet_reward": sv_total_r,
        "axis_final_sigma": ax_final,
        "soviet_final_sigma": sv_final,
        "volga_reinforcements": reinf,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main                                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser(description="Stalingrad Battle Replay — Detailed Commentary")
    p.add_argument("--axis-model", type=str, required=True)
    p.add_argument("--soviet-model", type=str, required=True)
    p.add_argument("--regime-file", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interval", type=int, default=25,
                   help="Report every N turns (default 25)")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    # Load regime
    regime_path = args.regime_file
    if regime_path is None:
        for candidate in [
            _ROOT / "training" / "regimes" / "stalingrad.yaml",
            _ROOT / "regimes" / "stalingrad.yaml",
        ]:
            if candidate.exists():
                regime_path = str(candidate)
                break
    assert regime_path, "Cannot find stalingrad.yaml"

    regime = load_standalone_regime(regime_path, seed=args.seed)
    params = regime["params"]
    params = GravitasParams(**{
        **{k: getattr(params, k) for k in GravitasParams.__dataclass_fields__},
        "n_clusters_max": 12,
    })
    init = build_initial_states(regime, max_N=12)

    env = StalingradMultiAgentEnv(
        params=params,
        axis_clusters=DEFAULT_AXIS_CLUSTERS,
        soviet_clusters=DEFAULT_SOVIET_CLUSTERS,
        contested_clusters=CONTESTED_CLUSTERS,
        initial_clusters=init["initial_clusters"],
        initial_alliances=init["initial_alliances"],
        seed=args.seed,
    )

    # Load models
    print(f"  Loading Axis model:   {args.axis_model}")
    axis_model = RecurrentPPO.load(args.axis_model, device=args.device)
    print(f"  Loading Soviet model: {args.soviet_model}")
    soviet_model = RecurrentPPO.load(args.soviet_model, device=args.device)

    # Run replay
    t0 = time.time()
    result = replay_battle(
        env, axis_model, soviet_model,
        seed=args.seed,
        report_interval=args.interval,
    )
    elapsed = time.time() - t0
    print(f"  Replay completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
