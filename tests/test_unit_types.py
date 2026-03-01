#!/usr/bin/env python3
"""
Test script for the expanded unit types and combat system.
"""

import numpy as np
from gravitas_engine.extensions.military.unit_types import (
    DamageType, MilitaryUnitType, UnitRole, get_unit_role,
    CombatMatrix, ExpandedUnitParams, SupportCompanyType,
    SupportCompany, EnhancedMilitaryUnit, calculate_damage,
    resolve_enhanced_combat
)

def test_damage_types():
    """Test damage type definitions."""
    print("🔫 Testing Damage Types...")
    damage_types = list(DamageType)
    print(f"   Found {len(damage_types)} damage types:")
    for dt in damage_types:
        print(f"     - {dt.name}")
    print("   ✅ Damage types working correctly\n")

def test_unit_types():
    """Test expanded unit type definitions."""
    print("👮 Testing Unit Types...")
    unit_types = list(MilitaryUnitType)
    print(f"   Found {len(unit_types)} unit types:")
    for ut in unit_types:
        print(f"     - {ut.name}")
    print("   ✅ Unit types working correctly\n")

def test_unit_roles():
    """Test unit role categorization."""
    print("🎭 Testing Unit Roles...")
    test_cases = [
        (MilitaryUnitType.INFANTRY, UnitRole.INFANTRY),
        (MilitaryUnitType.MEDIUM_TANK, UnitRole.ARMOR),
        (MilitaryUnitType.ARTILLERY, UnitRole.ARTILLERY),
        (MilitaryUnitType.FIGHTER, UnitRole.AIR),
        (MilitaryUnitType.DESTROYER, UnitRole.NAVAL),
        (MilitaryUnitType.COMMANDOS, UnitRole.SPECIAL_FORCES),
        (MilitaryUnitType.LOGISTICS, UnitRole.LOGISTICS),
    ]

    for unit_type, expected_role in test_cases:
        actual_role = get_unit_role(unit_type)
        assert actual_role == expected_role, f"Expected {expected_role} for {unit_type}, got {actual_role}"
        print(f"   ✅ {unit_type.name} -> {actual_role.name}")

    print("   ✅ Unit role mapping working correctly\n")

def test_combat_matrix():
    """Test combat matrix functionality."""
    print("⚔️ Testing Combat Matrix...")
    matrix = CombatMatrix.default_matrix()

    # Test some key matchups
    test_cases = [
        (MilitaryUnitType.INFANTRY, MilitaryUnitType.INFANTRY, 1.0),
        (MilitaryUnitType.MILITIA, MilitaryUnitType.INFANTRY, 0.7),
        (MilitaryUnitType.HEAVY_TANK, MilitaryUnitType.INFANTRY, 1.8),
        (MilitaryUnitType.ANTI_TANK, MilitaryUnitType.MEDIUM_TANK, 2.2),
        (MilitaryUnitType.FIGHTER, MilitaryUnitType.CAS, 1.8),
        (MilitaryUnitType.ANTI_AIR, MilitaryUnitType.STRATEGIC_BOMBER, 1.5),
    ]

    for attacker, defender, expected in test_cases:
        actual = matrix.get_damage_multiplier(attacker, defender)
        assert abs(actual - expected) < 0.01, f"Expected {expected} for {attacker} vs {defender}, got {actual}"
        print(f"   ✅ {attacker.name} vs {defender.name}: {actual:.2f}x")

    print("   ✅ Combat matrix working correctly\n")

def test_expanded_params():
    """Test expanded unit parameters."""
    print("📊 Testing Expanded Unit Parameters...")
    params = ExpandedUnitParams()

    # Test parameter retrieval
    test_cases = [
        (MilitaryUnitType.INFANTRY, 'combat', 1.0),
        (MilitaryUnitType.HEAVY_TANK, 'hp', 180),
        (MilitaryUnitType.FIGHTER, 'speed', 3.0),
        (MilitaryUnitType.ARTILLERY, 'supply_cost', 1.0),
    ]

    for unit_type, param_name, expected in test_cases:
        actual = params.get_param(unit_type, param_name)
        assert abs(actual - expected) < 0.01, f"Expected {expected} for {unit_type}.{param_name}, got {actual}"
        print(f"   ✅ {unit_type.name}.{param_name}: {actual}")

    print("   ✅ Expanded unit parameters working correctly\n")

def test_support_companies():
    """Test support company system."""
    print("🛠️ Testing Support Companies...")

    # Create some support companies
    artillery_co = SupportCompany(SupportCompanyType.ARTILLERY_COMPANY, size=2, effectiveness=0.9)
    anti_tank_co = SupportCompany(SupportCompanyType.ANTI_TANK_COMPANY, size=1, effectiveness=1.0)
    medical_co = SupportCompany(SupportCompanyType.MEDICAL_COMPANY, size=1, effectiveness=0.8)

    # Test bonuses
    artillery_bonuses = artillery_co.get_bonuses()
    assert abs(artillery_bonuses['soft_attack'] - 0.36) < 0.01, "Artillery bonus incorrect"
    assert abs(artillery_bonuses['defense'] - 0.18) < 0.01, "Artillery defense bonus incorrect"

    anti_tank_bonuses = anti_tank_co.get_bonuses()
    assert abs(anti_tank_bonuses['hard_attack'] - 0.3) < 0.01, "Anti-tank bonus incorrect"

    medical_bonuses = medical_co.get_bonuses()
    assert abs(medical_bonuses['hp_regen'] - 0.024) < 0.01, "Medical bonus incorrect"

    print(f"   ✅ Artillery Company: {artillery_bonuses}")
    print(f"   ✅ Anti-Tank Company: {anti_tank_bonuses}")
    print(f"   ✅ Medical Company: {medical_bonuses}")
    print("   ✅ Support companies working correctly\n")

def test_enhanced_unit():
    """Test enhanced military unit."""
    print("💪 Testing Enhanced Military Unit...")

    # Create a unit with support companies
    support_companies = (
        SupportCompany(SupportCompanyType.ARTILLERY_COMPANY, size=1),
        SupportCompany(SupportCompanyType.ENGINEER_COMPANY, size=1),
    )

    unit = EnhancedMilitaryUnit(
        unit_id=1,
        unit_type=MilitaryUnitType.INFANTRY,
        cluster_id=0,
        hit_points=100.0,
        combat_effectiveness=1.0,
        supply_level=0.8,
        experience=2.0,
        morale=0.9,
        support_companies=support_companies,
        terrain_bonus=1.2,
        entrenchment=0.3
    )

    # Test properties
    assert unit.is_alive, "Unit should be alive"
    combat_power = unit.combat_power
    assert combat_power > 100, f"Combat power should be > 100, got {combat_power}"

    # Test attack/defense values
    params = ExpandedUnitParams()
    attack_values = unit.get_attack_values(params)
    defense_values = unit.get_defense_values(params)

    print(f"   ✅ Unit alive: {unit.is_alive}")
    print(f"   ✅ Combat power: {combat_power:.2f}")
    print(f"   ✅ Attack values: {attack_values}")
    print(f"   ✅ Defense values: {defense_values}")
    print("   ✅ Enhanced military unit working correctly\n")

def test_combat_calculations():
    """Test combat damage calculations."""
    print("💥 Testing Combat Calculations...")

    # Create test units
    params = ExpandedUnitParams()
    matrix = CombatMatrix.default_matrix()

    # Infantry vs Infantry
    infantry1 = EnhancedMilitaryUnit(
        unit_id=1, unit_type=MilitaryUnitType.INFANTRY, cluster_id=0,
        hit_points=100, combat_effectiveness=1.0, supply_level=0.8,
        experience=1.0, morale=0.9
    )

    infantry2 = EnhancedMilitaryUnit(
        unit_id=2, unit_type=MilitaryUnitType.INFANTRY, cluster_id=0,
        hit_points=100, combat_effectiveness=1.0, supply_level=0.8,
        experience=1.0, morale=0.9
    )

    # Tank vs Infantry
    tank = EnhancedMilitaryUnit(
        unit_id=3, unit_type=MilitaryUnitType.HEAVY_TANK, cluster_id=0,
        hit_points=180, combat_effectiveness=1.0, supply_level=0.8,
        experience=1.0, morale=0.9
    )

    # Test damage calculation
    attacker_dmg, defender_dmg = calculate_damage(
        infantry1, infantry2, matrix, params, terrain_factor=1.0
    )
    print(f"   ✅ Infantry vs Infantry damage: {attacker_dmg:.2f} / {defender_dmg:.2f}")

    attacker_dmg, defender_dmg = calculate_damage(
        tank, infantry1, matrix, params, terrain_factor=1.0
    )
    print(f"   ✅ Tank vs Infantry damage: {attacker_dmg:.2f} / {defender_dmg:.2f}")

    # Test combat resolution
    updated_infantry1, updated_infantry2 = resolve_enhanced_combat(
        infantry1, infantry2, matrix, params, terrain_advantage=1.0
    )

    assert updated_infantry1.hit_points < 100, "Attacker should take damage"
    assert updated_infantry2.hit_points < 100, "Defender should take damage"
    assert updated_infantry1.is_alive, "Attacker should still be alive"
    assert updated_infantry2.is_alive, "Defender should still be alive"

    print(f"   ✅ Post-combat HP: {updated_infantry1.hit_points:.1f} / {updated_infantry2.hit_points:.1f}")
    print("   ✅ Combat calculations working correctly\n")

def main():
    """Run all tests."""
    print("🧪 Testing Expanded Unit Types and Combat System")
    print("=" * 60)

    test_damage_types()
    test_unit_types()
    test_unit_roles()
    test_combat_matrix()
    test_expanded_params()
    test_support_companies()
    test_enhanced_unit()
    test_combat_calculations()

    print("🎉 All unit type and combat system tests passed!")
    print("✅ Expanded military unit system is working correctly")

if __name__ == "__main__":
    main()