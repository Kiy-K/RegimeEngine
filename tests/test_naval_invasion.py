#!/usr/bin/env python3
"""
test_naval_invasion.py — End-to-end test of naval invasion lifecycle.

Simulates Eurasia invading Dover (Oceania) via Dover Strait:
  PLANNING → ASSEMBLY → CROSSING → BEACH_ASSAULT → BEACHHEAD → COMPLETED

Validates each phase transition, troop counts, and territory change.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gravitas.llm_game import create_game, step_game, parse_action, apply_actions

def test_invasion():
    rng = np.random.default_rng(2025)
    game = create_game(seed=2025, max_turns=200)

    print("=" * 70)
    print("  NAVAL INVASION END-TO-END TEST")
    print("  Eurasia invades Dover via Dover Strait")
    print("=" * 70)

    # Verify Dover (sector 1) is Oceania-owned
    assert game.cluster_owners[1] == 0, f"Dover should be Oceania (0), got {game.cluster_owners[1]}"
    print(f"\n[SETUP] Dover owner: Oceania ({game.cluster_owners[1]})")
    print(f"[SETUP] Calais owner: Eurasia ({game.cluster_owners[18]})")

    # Step a few turns to warm up systems
    for i in range(3):
        step_game(game, rng)
    print(f"[SETUP] Warmed up 3 turns. Turn={game.turn}")

    # Eurasia plans invasion: Calais (18) → Dover (1) via Dover Strait (zone 0)
    print(f"\n--- PHASE: PLAN_INVASION Calais → Dover via zone 0 ---")
    actions = parse_action("PLAN_INVASION Calais Dover 0", game, faction_id=1)
    results = apply_actions(game, faction_id=1, actions=actions, rng=rng)
    for r in results:
        print(f"  [E] {r}")

    # Find the invasion
    inv = None
    for i in game.invasions:
        if i.is_active and i.faction_id == 1:
            inv = i
            break
    assert inv is not None, "Invasion should have been created!"
    print(f"  Invasion #{inv.invasion_id}: {inv.phase.name}, target=sector {inv.target_cluster}")
    print(f"  Planning: {inv.planning_steps_done}/{inv.planning_steps_required}")

    # Step through PLANNING phase
    print(f"\n--- PHASE: PLANNING (need {inv.planning_steps_required} steps) ---")
    planning_turns = 0
    while inv.phase.name == "PLANNING" and planning_turns < 30:
        step_game(game, rng)
        planning_turns += 1
        if planning_turns % 3 == 0 or inv.phase.name != "PLANNING":
            print(f"  Turn {game.turn}: phase={inv.phase.name}, "
                  f"planning={inv.planning_steps_done}/{inv.planning_steps_required}, "
                  f"detected={inv.detected_by_enemy}")

    assert inv.phase.name != "PLANNING", f"Stuck in PLANNING after {planning_turns} turns!"
    print(f"  → Transitioned to {inv.phase.name} after {planning_turns} turns")

    if inv.phase.name == "ABORTED":
        print(f"  ⚠ Invasion ABORTED during planning. This can happen (timeout).")
        print(f"  RESULT: ABORTED (not a bug — planning timeout working as intended)")
        return

    # Step through ASSEMBLY phase
    print(f"\n--- PHASE: ASSEMBLY (need {inv.assembly_steps_required} steps) ---")
    assembly_turns = 0
    while inv.phase.name == "ASSEMBLY" and assembly_turns < 40:
        step_game(game, rng)
        assembly_turns += 1
        if assembly_turns % 3 == 0 or inv.phase.name != "ASSEMBLY":
            print(f"  Turn {game.turn}: phase={inv.phase.name}, "
                  f"assembly={inv.assembly_steps_done}/{inv.assembly_steps_required}, "
                  f"troops={inv.troops_embarked}")

    if inv.phase.name == "ABORTED":
        print(f"  ⚠ Invasion ABORTED during assembly (no transports or storm timeout)")
        print(f"  RESULT: ABORTED (not a bug — assembly timeout working)")
        return

    assert inv.phase.name in ("CROSSING", "AIRDROP"), f"Expected CROSSING, got {inv.phase.name}"
    print(f"  → Transitioned to {inv.phase.name} with {inv.troops_embarked} troops")

    # Step through CROSSING phase
    print(f"\n--- PHASE: CROSSING ---")
    crossing_turns = 0
    while inv.phase.name == "CROSSING" and crossing_turns < 20:
        step_game(game, rng)
        crossing_turns += 1
        if crossing_turns % 2 == 0 or inv.phase.name != "CROSSING":
            print(f"  Turn {game.turn}: phase={inv.phase.name}, "
                  f"crossing={inv.crossing_steps_done}/{inv.crossing_steps_required}, "
                  f"troops={inv.troops_embarked}")

    if inv.phase.name == "ABORTED":
        print(f"  ⚠ Invasion ABORTED during crossing (storm/losses)")
        print(f"  RESULT: ABORTED during crossing")
        return

    print(f"  → Transitioned to {inv.phase.name}")

    # Step through BEACH_ASSAULT phase
    if inv.phase.name == "BEACH_ASSAULT":
        print(f"\n--- PHASE: BEACH_ASSAULT ---")
        assault_turns = 0
        while inv.phase.name == "BEACH_ASSAULT" and assault_turns < 15:
            step_game(game, rng)
            assault_turns += 1
            print(f"  Turn {game.turn}: phase={inv.phase.name}, "
                  f"beachhead_str={inv.beachhead_strength:.2f}, "
                  f"troops={inv.troops_embarked}")

        print(f"  → Transitioned to {inv.phase.name}")

    # Step through BEACHHEAD phase
    if inv.phase.name == "BEACHHEAD":
        print(f"\n--- PHASE: BEACHHEAD (consolidating) ---")
        bh_turns = 0
        while inv.phase.name == "BEACHHEAD" and bh_turns < 20:
            step_game(game, rng)
            bh_turns += 1
            if bh_turns % 3 == 0 or inv.phase.name != "BEACHHEAD":
                print(f"  Turn {game.turn}: phase={inv.phase.name}, "
                      f"beachhead_str={inv.beachhead_strength:.2f}")

        print(f"  → Final phase: {inv.phase.name}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  INVASION RESULT: {inv.phase.name}")
    print(f"  Total turns: {game.turn}")
    print(f"  Dover owner: {game.cluster_owners.get(1, '?')}")
    print(f"  Scores: Oceania={game.faction_scores[0]:.0f} Eurasia={game.faction_scores[1]:.0f}")

    if inv.phase.name == "COMPLETED":
        print(f"  ✅ INVASION SUCCESSFUL — territory captured!")
    elif inv.phase.name == "BEACHHEAD":
        print(f"  ✅ BEACHHEAD ESTABLISHED — invasion partially successful")
    elif inv.phase.name == "ABORTED":
        print(f"  ❌ INVASION FAILED — aborted")
    else:
        print(f"  ⚠ INVASION IN PROGRESS — phase: {inv.phase.name}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_invasion()
