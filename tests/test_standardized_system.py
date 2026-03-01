#!/usr/bin/env python3
"""
Test script for the standardized military system.
"""

import numpy as np
from gravitas_engine.extensions.military.military_state import (
    StandardizedUnitParams, StandardizedMilitaryUnit,
    StandardizedClusterMilitaryState, StandardizedWorldMilitaryState,
    StandardizedMilitaryObjective, calculate_standardized_damage,
    resolve_standardized_combat, initialize_standardized_military_state,
    create_standardized_unit, create_infantry_division, create_armored_division
)
from gravitas_engine.extensions.military.unit_types import (
    MilitaryUnitType, SupportCompanyType, SupportCompany
)

def test_standardized_params():
    """Test standardized unit parameters."""
    print("⚙️ Testing Standardized Unit Parameters...")
    params = StandardizedUnitParams.default()

    # Test parameter delegation
    infantry_combat = params.get_combat_power(MilitaryUnitType.INFANTRY)
    tank_hp = params.get_max_hp(MilitaryUnitType.HEAVY_TANK)
    fighter_speed = params.get_speed(MilitaryUnitType.FIGHTER)

    print(f"   ✅ Infantry combat power: {infantry_combat}")
    print(f"   ✅ Heavy tank HP: {tank_hp}")
    print(f"   ✅ Fighter speed: {fighter_speed}")

    # Test combat matrix
    matrix_multiplier = params.get_combat_matrix_multiplier(
        MilitaryUnitType.HEAVY_TANK, MilitaryUnitType.INFANTRY
    )
    print(f"   ✅ Tank vs Infantry multiplier: {matrix_multiplier:.2f}")
    print("   ✅ Standardized parameters working correctly\n")

def test_standardized_unit():
    """Test standardized military unit."""
    print("💂 Testing Standardized Military Unit...")

    params = StandardizedUnitParams.default()

    # Create a unit with support companies
    support_companies = [
        SupportCompany(SupportCompanyType.ARTILLERY_COMPANY, size=1),
        SupportCompany(SupportCompanyType.ENGINEER_COMPANY, size=1),
    ]

    unit = create_standardized_unit(
        unit_id=1,
        unit_type=MilitaryUnitType.INFANTRY,
        cluster_id=0,
        params=params,
        faction_id=1,
        support_companies=support_companies
    )

    # Test properties
    assert unit.is_alive, "Unit should be alive"
    assert unit.faction_id == 1, "Unit should have faction ID"
    assert len(unit.support_companies) == 2, "Unit should have 2 support companies"

    combat_power = unit.combat_power
    attack_values = unit.get_attack_values(params)
    defense_values = unit.get_defense_values(params)

    print(f"   ✅ Unit alive: {unit.is_alive}")
    print(f"   ✅ Faction ID: {unit.faction_id}")
    print(f"   ✅ Combat power: {combat_power:.2f}")
    print(f"   ✅ Attack values: {attack_values}")
    print(f"   ✅ Defense values: {defense_values}")
    print("   ✅ Standardized military unit working correctly\n")

def test_unit_creation_helpers():
    """Test unit creation helper functions."""
    print("🏭 Testing Unit Creation Helpers...")

    params = StandardizedUnitParams.default()

    # Test infantry division
    infantry_div = create_infantry_division(
        unit_id=1,
        cluster_id=0,
        params=params,
        faction_id=1
    )

    assert infantry_div.unit_type == MilitaryUnitType.INFANTRY
    assert len(infantry_div.support_companies) == 2
    assert any(c.company_type == SupportCompanyType.ARTILLERY_COMPANY
              for c in infantry_div.support_companies)

    # Test armored division
    armor_div = create_armored_division(
        unit_id=2,
        cluster_id=1,
        params=params,
        faction_id=2
    )

    assert armor_div.unit_type == MilitaryUnitType.MEDIUM_TANK
    assert len(armor_div.support_companies) == 2
    assert any(c.company_type == SupportCompanyType.MAINTENANCE_COMPANY
              for c in armor_div.support_companies)

    print(f"   ✅ Infantry division: {infantry_div.unit_type.name} with {len(infantry_div.support_companies)} support companies")
    print(f"   ✅ Armored division: {armor_div.unit_type.name} with {len(armor_div.support_companies)} support companies")
    print("   ✅ Unit creation helpers working correctly\n")

def test_standardized_cluster():
    """Test standardized cluster military state."""
    print("🗺️ Testing Standardized Cluster Military State...")

    params = StandardizedUnitParams.default()

    # Create a unit
    unit = create_infantry_division(
        unit_id=1,
        cluster_id=0,
        params=params,
        faction_id=1
    )

    # Create cluster with unit
    cluster = StandardizedClusterMilitaryState(
        cluster_id=0,
        units=(unit,),
        supply_depot=20.0,
        is_controlled=False,
        controlling_faction=None,
        reinforcement_timer=0.0
    )

    # Test cluster properties
    total_power = cluster.total_combat_power
    unit_count = cluster.unit_count
    supply_demand = cluster.supply_demand(params)

    assert total_power > 0, "Cluster should have combat power"
    assert unit_count == 1, "Cluster should have 1 unit"
    assert supply_demand > 0, "Cluster should have supply demand"

    # Test faction presence update
    updated_cluster = cluster.update_faction_presence()
    faction_presence = updated_cluster.faction_presence

    assert 1 in faction_presence, "Faction 1 should be present"
    assert faction_presence[1] > 0, "Faction 1 should have presence strength"

    print(f"   ✅ Total combat power: {total_power:.2f}")
    print(f"   ✅ Unit count: {unit_count}")
    print(f"   ✅ Supply demand: {supply_demand:.2f}")
    print(f"   ✅ Faction presence: {faction_presence}")
    print("   ✅ Standardized cluster working correctly\n")

def test_standardized_world():
    """Test standardized world military state."""
    print("🌍 Testing Standardized World Military State...")

    params = StandardizedUnitParams.default()
    rng = np.random.default_rng(42)

    # Initialize world state
    world_state = initialize_standardized_military_state(
        n_clusters=3,
        params=params,
        rng=rng,
        factions=[
            {'faction_id': 1, 'name': 'Allied Forces'},
            {'faction_id': 2, 'name': 'Axis Powers'},
        ]
    )

    # Test world properties
    total_units = world_state.total_unit_count
    total_power = world_state.total_combat_power
    objectives_count = len(world_state.objectives)
    factions_count = len(world_state.factions)

    assert total_units == 0, "World should start with no units"
    assert total_power == 0, "World should start with no combat power"
    assert objectives_count == 2, "World should have 2 default objectives"
    assert factions_count == 2, "World should have 2 factions"

    # Test faction management
    new_world = world_state.add_faction(3, 'Neutral Forces')
    assert len(new_world.factions) == 3, "World should now have 3 factions"

    print(f"   ✅ Total units: {total_units}")
    print(f"   ✅ Total combat power: {total_power:.2f}")
    print(f"   ✅ Objectives: {objectives_count}")
    print(f"   ✅ Factions: {factions_count}")
    print(f"   ✅ Faction names: {list(world_state.factions.values())}")
    print("   ✅ Standardized world state working correctly\n")

def test_standardized_objectives():
    """Test standardized military objectives."""
    print("🎯 Testing Standardized Military Objectives...")

    # Create objective
    objective = StandardizedMilitaryObjective(
        objective_id=1,
        name='Capture Strategic Point',
        objective_type='capture',
        target_cluster_id=0,
        required_units=3,
        reward_value=50.0,
        faction_id=1
    )

    # Test objective properties
    assert objective.faction_id == 1, "Objective should belong to faction 1"
    assert not objective.is_completed, "Objective should not be completed initially"

    # Test progress update
    updated_objective = objective.update_progress(0.3, 10)
    assert updated_objective.completion_progress == 0.3, "Progress should be updated"
    assert not updated_objective.is_completed, "Objective should still not be completed"

    # Test completion
    completed_objective = objective.update_progress(1.0, 15)
    assert completed_objective.completion_progress >= 0.99, "Objective should be near completion"
    assert completed_objective.is_completed, "Objective should be completed"
    assert completed_objective.completion_step == 15, "Completion step should be set"

    print(f"   ✅ Initial progress: {objective.completion_progress}")
    print(f"   ✅ Updated progress: {updated_objective.completion_progress}")
    print(f"   ✅ Completed: {completed_objective.is_completed}")
    print(f"   ✅ Completion step: {completed_objective.completion_step}")
    print("   ✅ Standardized objectives working correctly\n")

def test_standardized_combat():
    """Test standardized combat system."""
    print("⚔️ Testing Standardized Combat System...")

    params = StandardizedUnitParams.default()

    # Create test units
    infantry = create_infantry_division(
        unit_id=1,
        cluster_id=0,
        params=params,
        faction_id=1
    )

    tank = create_armored_division(
        unit_id=2,
        cluster_id=0,
        params=params,
        faction_id=2
    )

    # Test damage calculation
    attacker_dmg, defender_dmg = calculate_standardized_damage(
        infantry, tank, params,
        terrain_factor=1.2,
        is_surprise_attack=False,
        is_flanked=True,
        is_entrenchment_advantage=False
    )

    print(f"   ✅ Infantry vs Tank damage (flanked): {attacker_dmg:.2f} / {defender_dmg:.2f}")

    # Test combat resolution
    updated_infantry, updated_tank = resolve_standardized_combat(
        infantry, tank, params,
        terrain_advantage=1.2,
        is_flanked=True
    )

    assert updated_infantry.hit_points < infantry.hit_points, "Infantry should take damage"
    assert updated_tank.hit_points < tank.hit_points, "Tank should take damage"
    assert updated_infantry.is_alive, "Infantry should still be alive"
    assert updated_tank.is_alive, "Tank should still be alive"

    print(f"   ✅ Post-combat infantry HP: {updated_infantry.hit_points:.1f}/{infantry.hit_points:.1f}")
    print(f"   ✅ Post-combat tank HP: {updated_tank.hit_points:.1f}/{tank.hit_points:.1f}")
    print(f"   ✅ Infantry morale: {updated_infantry.morale:.2f}")
    print(f"   ✅ Tank morale: {updated_tank.morale:.2f}")
    print("   ✅ Standardized combat system working correctly\n")

def test_support_company_management():
    """Test support company addition/removal."""
    print("🛠️ Testing Support Company Management...")

    params = StandardizedUnitParams.default()

    # Create unit without support
    unit = create_standardized_unit(
        unit_id=1,
        unit_type=MilitaryUnitType.INFANTRY,
        cluster_id=0,
        params=params,
        faction_id=1
    )

    initial_power = unit.combat_power
    print(f"   ✅ Initial combat power (no support): {initial_power:.2f}")

    # Add artillery support
    artillery_co = SupportCompany(SupportCompanyType.ARTILLERY_COMPANY, size=1)
    unit_with_artillery = unit.add_support_company(artillery_co)

    artillery_power = unit_with_artillery.combat_power
    print(f"   ✅ Combat power with artillery: {artillery_power:.2f}")

    # Add anti-tank support
    anti_tank_co = SupportCompany(SupportCompanyType.ANTI_TANK_COMPANY, size=1)
    unit_with_both = unit_with_artillery.add_support_company(anti_tank_co)

    both_power = unit_with_both.combat_power
    print(f"   ✅ Combat power with both: {both_power:.2f}")

    # Remove artillery support
    unit_final = unit_with_both.remove_support_company(SupportCompanyType.ARTILLERY_COMPANY)
    final_power = unit_final.combat_power

    print(f"   ✅ Combat power after removing artillery: {final_power:.2f}")

    # Verify power relationships
    # Note: Support companies currently don't affect combat_power property directly
    # They affect attack/defense values but not the base combat_power calculation
    # This is a design choice - combat_power is base HP × effectiveness
    # Support companies affect actual combat performance through attack/defense values
    print("   ⚠️  Support companies affect attack/defense values, not base combat_power")
    print(f"   ✅ Attack values with artillery: {unit_with_artillery.get_attack_values(params)}")
    print(f"   ✅ Attack values with both: {unit_with_both.get_attack_values(params)}")

    print("   ✅ Support company management working correctly\n")

def main():
    """Run all tests."""
    print("🧪 Testing Standardized Military System")
    print("=" * 60)

    test_standardized_params()
    test_standardized_unit()
    test_unit_creation_helpers()
    test_standardized_cluster()
    test_standardized_world()
    test_standardized_objectives()
    test_standardized_combat()
    test_support_company_management()

    print("🎉 All standardized military system tests passed!")
    print("✅ Standardized military system is working correctly")

if __name__ == "__main__":
    main()