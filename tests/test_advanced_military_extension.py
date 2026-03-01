"""
test_advanced_military_extension.py — Comprehensive test suite for advanced military extension.

This test suite validates all advanced military features:
  1. Unit Formations and Group Tactics
  2. Command Hierarchy and Chain of Command
  3. Intelligence and Reconnaissance Systems
  4. Advanced Combat Tactics
  5. Electronic Warfare and Communications
  6. Logistics and Supply Chain Management
  7. Tactical AI for Autonomous Unit Behavior
"""

import numpy as np
from gravitas_engine.extensions.military import (
    MilitaryUnit, MilitaryUnitType, MilitaryUnitParams,
    AdvancedTacticsEngine, FormationType, CommandRank,
    IntelligenceType, CombatTactic, SupplyType,
    UnitFormation, CommandStructure, IntelligenceReport,
    TacticalOperation, ElectronicWarfareState, SupplyChain,
    AdvancedClusterMilitaryState, AdvancedWorldMilitaryState,
    EWCapability, MilitaryObjective
)

def test_advanced_tactics_imports():
    """Test that all advanced tactics classes can be imported."""
    print("1️⃣ Testing advanced tactics imports...")

    # Test all imports
    assert MilitaryUnit is not None
    assert MilitaryUnitType is not None
    assert MilitaryUnitParams is not None
    assert AdvancedTacticsEngine is not None
    assert FormationType is not None
    assert CommandRank is not None
    assert IntelligenceType is not None
    assert CombatTactic is not None
    assert SupplyType is not None
    assert UnitFormation is not None
    assert CommandStructure is not None
    assert IntelligenceReport is not None
    assert TacticalOperation is not None
    assert ElectronicWarfareState is not None
    assert SupplyChain is not None
    assert AdvancedClusterMilitaryState is not None
    assert AdvancedWorldMilitaryState is not None

    print("   ✅ All advanced tactics classes imported successfully")

def test_military_parameters():
    """Test military parameters creation."""
    print("2️⃣ Testing military parameters...")

    params = MilitaryUnitParams()
    assert params is not None
    assert params.infantry_hp > 0
    assert params.armor_hp > 0
    assert params.objective_hold_duration > 0

    print(f"   📊 Infantry HP: {params.infantry_hp}")
    print(f"   📊 Armor HP: {params.armor_hp}")
    print(f"   📊 Objective hold duration: {params.objective_hold_duration}")
    print("   ✅ MilitaryUnitParams created successfully")

def test_unit_formation():
    """Test unit formation creation."""
    print("3️⃣ Testing unit formation creation...")

    # Create a formation
    formation = UnitFormation(
        formation_id=1,
        formation_type=FormationType.LINE,
        leader_unit_id=1,
        member_unit_ids=(1, 2, 3),
        cohesion=0.8,
        morale_boost=0.1,
        combat_bonus=0.15,
        movement_penalty=0.1
    )

    assert formation is not None
    assert formation.formation_id == 1
    assert formation.formation_type == FormationType.LINE
    assert formation.size == 3
    assert formation.cohesion == 0.8

    print(f"   📊 Formation size: {formation.size}")
    print(f"   📊 Cohesion: {formation.cohesion}")
    print(f"   📊 Combat bonus: {formation.combat_bonus}")
    print("   ✅ Unit formation created successfully")

def test_command_structure():
    """Test command structure creation."""
    print("4️⃣ Testing command structure creation...")

    # Create a command structure
    command = CommandStructure(
        commander_unit_id=1,
        subordinate_units=(2, 3, 4),
        rank=CommandRank.CAPTAIN,
        command_radius=4,
        communication_efficiency=0.85,
        initiative=0.8
    )

    assert command is not None
    assert command.commander_unit_id == 1
    assert command.rank == CommandRank.CAPTAIN
    assert command.can_command(2) == True
    assert command.can_command(5) == False

    print(f"   📊 Commander rank: {command.rank}")
    print(f"   📊 Subordinates: {len(command.subordinate_units)}")
    print(f"   📊 Communication efficiency: {command.communication_efficiency}")
    print("   ✅ Command structure created successfully")

def test_intelligence_report():
    """Test intelligence report creation."""
    print("5️⃣ Testing intelligence report creation...")

    # Create an intelligence report
    report = IntelligenceReport(
        report_id=1,
        intelligence_type=IntelligenceType.RECON,
        target_cluster_id=0,
        enemy_units_detected=5,
        enemy_combat_power=250.0,
        terrain_info="open",
        last_updated=0,
        confidence=0.8
    )

    assert report is not None
    assert report.report_id == 1
    assert report.intelligence_type == IntelligenceType.RECON
    assert report.enemy_units_detected == 5
    assert report.confidence == 0.8

    print(f"   📊 Intelligence type: {report.intelligence_type}")
    print(f"   📊 Enemy units detected: {report.enemy_units_detected}")
    print(f"   📊 Confidence: {report.confidence}")
    print("   ✅ Intelligence report created successfully")

def test_tactical_operation():
    """Test tactical operation creation."""
    print("6️⃣ Testing tactical operation creation...")

    # Create a tactical operation
    operation = TacticalOperation(
        operation_id=1,
        name="Flanking Maneuver",
        tactic=CombatTactic.FLANKING,
        primary_target=1,
        secondary_targets=(),
        participating_units=(1, 2, 3),
        success_probability=0.7,
        estimated_duration=3,
        required_intel=0.6
    )

    assert operation is not None
    assert operation.operation_id == 1
    assert operation.tactic == CombatTactic.FLANKING
    assert operation.success_probability == 0.7

    print(f"   📊 Operation name: {operation.name}")
    print(f"   📊 Tactic: {operation.tactic}")
    print(f"   📊 Success probability: {operation.success_probability}")
    print("   ✅ Tactical operation created successfully")

def test_electronic_warfare():
    """Test electronic warfare state creation."""
    print("7️⃣ Testing electronic warfare state creation...")

    # Create EW state
    ew_state = ElectronicWarfareState(
        ew_unit_id=1,
        capabilities=(EWCapability.JAMMING, EWCapability.INTERCEPT),
        jamming_range=5.0,
        interception_range=8.0,
        active_jamming=True,
        intelligence_gathered=0.5,
        enemy_communications_disrupted=0.3
    )

    assert ew_state is not None
    assert ew_state.ew_unit_id == 1
    assert ew_state.active_jamming == True
    assert len(ew_state.capabilities) == 2

    print(f"   📊 EW capabilities: {len(ew_state.capabilities)}")
    print(f"   📊 Jamming range: {ew_state.jamming_range}")
    print(f"   📊 Active jamming: {ew_state.active_jamming}")
    print("   ✅ Electronic warfare state created successfully")

def test_supply_chain():
    """Test supply chain creation."""
    print("8️⃣ Testing supply chain creation...")

    # Create supply chain
    supply_chain = SupplyChain(
        supply_depot_id=1001,
        cluster_id=0,
        supply_types={
            SupplyType.AMMUNITION: 20.0,
            SupplyType.FUEL: 15.0,
            SupplyType.FOOD: 10.0,
            SupplyType.MEDICAL: 5.0,
            SupplyType.REPAIR: 8.0,
            SupplyType.SPECIALIZED: 3.0
        },
        capacity=100.0,
        throughput=10.0,
        vulnerability=0.2,
        connected_depots=(1, 2)
    )

    assert supply_chain is not None
    assert supply_chain.supply_depot_id == 1001
    assert supply_chain.total_supply() > 0
    assert len(supply_chain.connected_depots) == 2

    print(f"   📊 Total supply: {supply_chain.total_supply()}")
    print(f"   📊 Capacity: {supply_chain.capacity}")
    print(f"   📊 Connected depots: {len(supply_chain.connected_depots)}")
    print("   ✅ Supply chain created successfully")

def test_advanced_cluster_state():
    """Test advanced cluster military state creation."""
    print("9️⃣ Testing advanced cluster military state creation...")

    # Create some units
    units = (
        MilitaryUnit(
            unit_id=1,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=100.0,
            combat_effectiveness=1.0,
            supply_level=0.8,
            experience=0.0,
            morale=0.9,
            objective_id=1
        ),
        MilitaryUnit(
            unit_id=2,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=0,
            hit_points=150.0,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=1
        )
    )

    # Create advanced cluster state
    cluster = AdvancedClusterMilitaryState(
        cluster_id=0,
        units=units,
        supply_depot=15.0,
        is_controlled=True,
        controlling_faction=1,
        reinforcement_timer=0.0,
        formations=(),
        command_structure=None,
        intelligence_reports=(),
        active_operations=(),
        ew_state=None,
        supply_chain=None,
        terrain_advantage=1.2,
        fog_of_war=0.3
    )

    assert cluster is not None
    assert cluster.cluster_id == 0
    assert cluster.unit_count == 2
    assert cluster.total_combat_power > 0
    assert cluster.supply_depot == 15.0
    assert cluster.terrain_advantage == 1.2

    print(f"   📊 Units in cluster: {cluster.unit_count}")
    print(f"   📊 Total combat power: {cluster.total_combat_power}")
    print(f"   📊 Supply depot: {cluster.supply_depot}")
    print(f"   📊 Terrain advantage: {cluster.terrain_advantage}")
    print("   ✅ Advanced cluster military state created successfully")

def test_advanced_world_state():
    """Test advanced world military state creation."""
    print("🔟 Testing advanced world military state creation...")

    # Create clusters
    clusters = (
        AdvancedClusterMilitaryState(
            cluster_id=0,
            units=(),
            supply_depot=10.0,
            is_controlled=False,
            terrain_advantage=1.0,
            fog_of_war=0.5
        ),
        AdvancedClusterMilitaryState(
            cluster_id=1,
            units=(),
            supply_depot=12.0,
            is_controlled=False,
            terrain_advantage=1.1,
            fog_of_war=0.4
        )
    )

    # Create objectives
    objectives = (
        MilitaryObjective(
            objective_id=1,
            name="Capture Cluster 0",
            objective_type="capture",
            target_cluster_id=0,
            required_units=3,
            reward_value=25.0,
        ),
        MilitaryObjective(
            objective_id=2,
            name="Hold Cluster 1",
            objective_type="hold",
            target_cluster_id=1,
            required_units=2,
            reward_value=15.0,
        )
    )

    # Create advanced world state
    world_state = AdvancedWorldMilitaryState(
        clusters=clusters,
        objectives=objectives,
        global_supply=100.0,
        global_reinforcement_pool=50.0,
        step=0,
        next_unit_id=1,
        next_formation_id=1,
        next_operation_id=1,
        next_report_id=1,
        global_intelligence={},
        electronic_warfare_active=False,
        communication_network_integrity=1.0
    )

    assert world_state is not None
    assert len(world_state.clusters) == 2
    assert len(world_state.objectives) == 2
    assert world_state.global_supply == 100.0
    assert world_state.global_reinforcement_pool == 50.0

    print(f"   📊 Clusters: {len(world_state.clusters)}")
    print(f"   📊 Objectives: {len(world_state.objectives)}")
    print(f"   📊 Global supply: {world_state.global_supply}")
    print(f"   📊 Reinforcement pool: {world_state.global_reinforcement_pool}")
    print("   ✅ Advanced world military state created successfully")

def test_tactics_engine():
    """Test advanced tactics engine functionality."""
    print("🔥 Testing advanced tactics engine...")

    params = MilitaryUnitParams()
    engine = AdvancedTacticsEngine(params)

    # Test battlefield analysis
    world_state = AdvancedWorldMilitaryState(
        clusters=(
            AdvancedClusterMilitaryState(
                cluster_id=0,
                units=(),
                supply_depot=10.0,
                is_controlled=False,
                terrain_advantage=1.2,
                fog_of_war=0.3
            ),
        ),
        objectives=(),
        global_supply=100.0,
        global_reinforcement_pool=50.0,
        step=0
    )

    analysis = engine.analyze_battlefield(world_state, 0)
    assert analysis is not None
    assert 'cluster_id' in analysis
    assert 'recommended_tactic' in analysis

    print(f"   📊 Recommended tactic: {analysis['recommended_tactic']}")
    print(f"   📊 Force ratio: {analysis['force_ratio']:.2f}")
    print(f"   📊 Terrain factor: {analysis['terrain_factor']:.2f}")
    print("   ✅ Tactics engine analysis working correctly")

def test_formation_creation():
    """Test formation creation through tactics engine."""
    print("🛡️ Testing formation creation...")

    params = MilitaryUnitParams()
    engine = AdvancedTacticsEngine(params)

    # Create units
    units = (
        MilitaryUnit(
            unit_id=1,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=100.0,
            combat_effectiveness=1.0,
            supply_level=0.8,
            experience=0.0,
            morale=0.9,
            objective_id=1
        ),
        MilitaryUnit(
            unit_id=2,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=100.0,
            combat_effectiveness=1.0,
            supply_level=0.8,
            experience=0.0,
            morale=0.9,
            objective_id=1
        ),
        MilitaryUnit(
            unit_id=3,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=0,
            hit_points=150.0,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=1
        )
    )

    # Create world state with units
    world_state = AdvancedWorldMilitaryState(
        clusters=(
            AdvancedClusterMilitaryState(
                cluster_id=0,
                units=units,
                supply_depot=15.0,
                is_controlled=False,
                terrain_advantage=1.0,
                fog_of_war=0.5
            ),
        ),
        objectives=(),
        global_supply=100.0,
        global_reinforcement_pool=50.0,
        step=0,
        next_formation_id=1
    )

    # Create formation
    new_world, formation = engine.create_formation(
        world_state,
        0,
        [1, 2, 3],
        FormationType.WEDGE,
        params
    )

    assert new_world is not None
    assert formation is not None
    assert formation.formation_type == FormationType.WEDGE
    assert formation.size == 3

    print(f"   📊 Formation created: {formation.formation_type}")
    print(f"   📊 Formation size: {formation.size}")
    print(f"   📊 Combat bonus: {formation.combat_bonus}")
    print("   ✅ Formation creation working correctly")

def test_tactical_operation_planning():
    """Test tactical operation planning."""
    print("⚔️ Testing tactical operation planning...")

    params = MilitaryUnitParams()
    engine = AdvancedTacticsEngine(params)

    # Create units
    units = (
        MilitaryUnit(
            unit_id=1,
            unit_type=MilitaryUnitType.INFANTRY,
            cluster_id=0,
            hit_points=100.0,
            combat_effectiveness=1.0,
            supply_level=0.8,
            experience=0.0,
            morale=0.9,
            objective_id=1
        ),
        MilitaryUnit(
            unit_id=2,
            unit_type=MilitaryUnitType.ARMOR,
            cluster_id=0,
            hit_points=150.0,
            combat_effectiveness=1.0,
            supply_level=0.7,
            experience=0.0,
            morale=0.8,
            objective_id=1
        )
    )

    # Create world state with units
    world_state = AdvancedWorldMilitaryState(
        clusters=(
            AdvancedClusterMilitaryState(
                cluster_id=0,
                units=units,
                supply_depot=15.0,
                is_controlled=False,
                terrain_advantage=1.0,
                fog_of_war=0.5
            ),
            AdvancedClusterMilitaryState(
                cluster_id=1,
                units=(),
                supply_depot=10.0,
                is_controlled=False,
                terrain_advantage=1.1,
                fog_of_war=0.4
            ),
        ),
        objectives=(),
        global_supply=100.0,
        global_reinforcement_pool=50.0,
        step=0,
        next_operation_id=1
    )

    # Plan operation
    new_world, operation = engine.plan_tactical_operation(
        world_state,
        CombatTactic.FLANKING,
        1,
        [1, 2],
        params
    )

    assert new_world is not None
    assert operation is not None
    assert operation.tactic == CombatTactic.FLANKING
    assert operation.primary_target == 1

    print(f"   📊 Operation planned: {operation.tactic}")
    print(f"   📊 Target cluster: {operation.primary_target}")
    print(f"   📊 Success probability: {operation.success_probability:.2f}")
    print("   ✅ Tactical operation planning working correctly")

def run_all_tests():
    """Run all advanced military extension tests."""
    print("Testing Advanced Military Extension")
    print("==================================================")

    try:
        test_advanced_tactics_imports()
        test_military_parameters()
        test_unit_formation()
        test_command_structure()
        test_intelligence_report()
        test_tactical_operation()
        test_electronic_warfare()
        test_supply_chain()
        test_advanced_cluster_state()
        test_advanced_world_state()
        test_tactics_engine()
        test_formation_creation()
        test_tactical_operation_planning()

        print("\n🎉 All advanced military extension tests passed!")
        print("✅ Advanced military extension is working correctly")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()