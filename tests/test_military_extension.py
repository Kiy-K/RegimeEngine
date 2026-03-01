#!/usr/bin/env python3
"""
Test script for the military extension.

This script tests the basic functionality of the military extension
without requiring a full GravitasEngine environment.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_military_extension():
    """Test the military extension components."""
    print("🔍 Testing Military Extension")
    print("=" * 50)

    # Test 1: Import military modules
    print("1️⃣ Testing imports...")
    try:
        from gravitas_engine.extensions.military.military_state import (
            MilitaryUnit, MilitaryUnitType, MilitaryUnitParams,
            ClusterMilitaryState, WorldMilitaryState, initialize_military_state
        )
        from gravitas_engine.extensions.military.military_dynamics import (
            step_military_units, apply_military_action,
            compute_military_reward, check_victory_conditions
        )
        print("   ✅ All military modules imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test 2: Create military parameters
    print("\n2️⃣ Testing military parameters...")
    try:
        params = MilitaryUnitParams()
        print(f"   ✅ MilitaryUnitParams created")
        print(f"   📊 Infantry combat power: {params.infantry_combat}")
        print(f"   📊 Armor speed: {params.armor_speed}")
        print(f"   📊 Objective hold duration: {params.objective_hold_duration}")
    except Exception as e:
        print(f"   ❌ Parameter creation failed: {e}")
        return False

    # Test 3: Create military units
    print("\n3️⃣ Testing military unit creation...")
    try:
        # Create infantry unit
        infantry = MilitaryUnit(
            unit_id=1,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=params.infantry_hp,
            combat_effectiveness=1.0,
            supply_level=0.8,
            experience=0.0,
            morale=0.9,
            objective_id=1
        )

        # Create armor unit
        armor = MilitaryUnit(
            unit_id=2,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=1,
            hit_points=params.armor_hp,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=2
        )

        print(f"   ✅ Created {infantry.unit_type.name} unit (ID: {infantry.unit_id})")
        print(f"   ✅ Created {armor.unit_type.name} unit (ID: {armor.unit_id})")
        print(f"   📊 Infantry combat power: {infantry.combat_power:.1f}")
        print(f"   📊 Armor combat power: {armor.combat_power:.1f}")
    except Exception as e:
        print(f"   ❌ Unit creation failed: {e}")
        return False

    # Test 4: Create cluster military state
    print("\n4️⃣ Testing cluster military state...")
    try:
        # Create cluster with units
        cluster = ClusterMilitaryState(
            cluster_id=0,
            units=(infantry, armor),
            supply_depot=15.0,
            is_controlled=False,
            controlling_faction=None,
            reinforcement_timer=0.0
        )

        print(f"   ✅ Cluster military state created")
        print(f"   📊 Units in cluster: {cluster.unit_count}")
        print(f"   📊 Total combat power: {cluster.total_combat_power:.1f}")
        print(f"   📊 Supply depot: {cluster.supply_depot:.1f}")
    except Exception as e:
        print(f"   ❌ Cluster state creation failed: {e}")
        return False

    # Test 5: Initialize world military state
    print("\n5️⃣ Testing world military state initialization...")
    try:
        rng = np.random.default_rng(42)
        world_state = initialize_military_state(
            n_clusters=3,
            params=params,
            rng=rng,
            initial_global_supply=100.0,
            initial_reinforcement_pool=50.0
        )

        print(f"   ✅ World military state initialized")
        print(f"   📊 Clusters: {len(world_state.clusters)}")
        print(f"   📊 Objectives: {len(world_state.objectives)}")
        print(f"   📊 Global supply: {world_state.global_supply:.1f}")
        print(f"   📊 Reinforcement pool: {world_state.global_reinforcement_pool:.1f}")

        # Add our test units to cluster 0
        cluster_with_units = world_state.clusters[0].add_unit(infantry).add_unit(armor)
        new_clusters = (cluster_with_units,) + world_state.clusters[1:]
        world_state = world_state.copy_with(clusters=new_clusters, next_unit_id=3)

        print(f"   📊 Total units after adding test units: {world_state.total_unit_count}")
    except Exception as e:
        print(f"   ❌ World state initialization failed: {e}")
        return False

    # Test 6: Test military actions
    print("\n6️⃣ Testing military actions...")
    try:
        # Test deploy action
        deploy_action = {
            'action_type': 'deploy',
            'target_cluster': 1,
            'unit_type': MilitaryUnitType.ARTILLERY,
            'intensity': 0.5
        }

        new_state = apply_military_action(
            world_state,
            deploy_action['action_type'],
            deploy_action['target_cluster'],
            deploy_action['unit_type'],
            deploy_action['intensity'],
            params,
            rng
        )

        print(f"   ✅ Deploy action applied")
        print(f"   📊 Units before: {world_state.total_unit_count}")
        print(f"   📊 Units after: {new_state.total_unit_count}")
        print(f"   📊 Reinforcement pool before: {world_state.global_reinforcement_pool:.1f}")
        print(f"   📊 Reinforcement pool after: {new_state.global_reinforcement_pool:.1f}")

        world_state = new_state
    except Exception as e:
        print(f"   ❌ Military action failed: {e}")
        return False

    # Test 7: Test combat resolution
    print("\n7️⃣ Testing combat resolution...")
    try:
        from gravitas_engine.extensions.military.military_dynamics import resolve_combat

        # Create two units for combat test
        attacker = MilitaryUnit(
            unit_id=10,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=100.0,
            combat_effectiveness=1.0,
            supply_level=1.0,
            experience=1.0,
            morale=0.9,
            objective_id=1
        )

        defender = MilitaryUnit(
            unit_id=11,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=90.0,
            combat_effectiveness=0.9,
            supply_level=0.8,
            experience=0.8,
            morale=0.8,
            objective_id=2
        )

        print(f"   🎯 Before combat:")
        print(f"      Attacker HP: {attacker.hit_points:.1f}, Effectiveness: {attacker.combat_effectiveness:.1f}")
        print(f"      Defender HP: {defender.hit_points:.1f}, Effectiveness: {defender.combat_effectiveness:.1f}")

        updated_attacker, updated_defender = resolve_combat(attacker, defender, params)

        print(f"   ✅ Combat resolved")
        print(f"   🎯 After combat:")
        print(f"      Attacker HP: {updated_attacker.hit_points:.1f}, Effectiveness: {updated_attacker.combat_effectiveness:.1f}")
        print(f"      Defender HP: {updated_defender.hit_points:.1f}, Effectiveness: {updated_defender.combat_effectiveness:.1f}")
        print(f"      Attacker alive: {updated_attacker.is_alive}")
        print(f"      Defender alive: {updated_defender.is_alive}")
    except Exception as e:
        print(f"   ❌ Combat resolution failed: {e}")
        return False

    # Test 8: Test victory conditions
    print("\n8️⃣ Testing victory conditions...")
    try:
        # Complete some objectives manually for testing
        completed_objectives = []
        for i, obj in enumerate(world_state.objectives):
            if i == 0:  # Complete first objective
                completed_obj = obj.update_progress(1.0, world_state.step)
                completed_objectives.append(completed_obj)
            else:
                completed_objectives.append(obj)

        test_state = world_state.copy_with(objectives=tuple(completed_objectives))

        victory_status = check_victory_conditions(test_state, victory_threshold=0.5)

        print(f"   ✅ Victory conditions checked")
        print(f"   🏆 Victory achieved: {victory_status['victory_achieved']}")
        print(f"   📊 Completion: {victory_status['completion_percentage']:.1%}")
        print(f"   📊 Message: {victory_status['message']}")
    except Exception as e:
        print(f"   ❌ Victory condition test failed: {e}")
        return False

    # Test 9: Test reward calculation
    print("\n9️⃣ Testing military reward calculation...")
    try:
        # Create a simple scenario with progress
        military_reward = compute_military_reward(
            world_state,  # current
            world_state,  # previous (same for this test)
            action_type="deploy"
        )

        print(f"   ✅ Military reward calculated: {military_reward:.2f}")
        print(f"   💰 Reward includes unit survival, combat power, and action bonuses")
    except Exception as e:
        print(f"   ❌ Reward calculation failed: {e}")
        return False

    print("\n🎉 All military extension tests passed!")
    print("✅ Military extension is working correctly")
    return True

def main():
    """Main test entry point."""
    try:
        success = test_military_extension()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Military extension test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())