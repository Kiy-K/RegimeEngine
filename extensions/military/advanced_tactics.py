"""
advanced_tactics.py — Advanced military tactics and AI for the military extension.

This module adds sophisticated tactical capabilities:
  1. Unit Formations and Group Tactics
  2. Command Hierarchy and Chain of Command
  3. Intelligence and Reconnaissance Systems
  4. Advanced Combat Tactics (flanking, ambush, suppression)
  5. Electronic Warfare and Communications
  6. Logistics and Supply Chain Management
  7. Tactical AI for Autonomous Unit Behavior

These advanced features build on the basic military system to create
a more realistic and sophisticated military simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any, Set

import numpy as np
from numpy.typing import NDArray

from .military_state import (
    StandardizedUnitParams as MilitaryUnitParams,
    StandardizedMilitaryUnit as MilitaryUnit,
    StandardizedClusterMilitaryState as ClusterMilitaryState,
    StandardizedWorldMilitaryState as WorldMilitaryState,
    StandardizedMilitaryObjective as MilitaryObjective
)

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Unit Formations                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class FormationType(Enum):
    """Types of military formations with different tactical properties."""
    LINE = auto()          # Traditional line formation (good for defense)
    WEDGE = auto()         # Wedge formation (good for penetration)
    COLUMN = auto()        # Column formation (good for movement)
    SQUARE = auto()        # Square formation (anti-cavalry/armor defense)
    SKIRMISH = auto()      # Skirmish formation (flexible, dispersed)
    AMBUSH = auto()        # Ambush setup (hidden, surprise attack)
    PHALANX = auto()       # Tight defensive formation (high protection)

@dataclass(frozen=True)
class UnitFormation:
    """
    A formation of multiple units working together tactically.

    Formations provide bonuses and penalties based on unit composition
    and tactical situation.
    """
    formation_id: int
    formation_type: FormationType
    leader_unit_id: int
    member_unit_ids: Tuple[int, ...]
    cohesion: float
    morale_boost: float
    combat_bonus: float
    movement_penalty: float
    is_concealed: bool = False

    @property
    def size(self) -> int:
        """Number of units in formation."""
        return len(self.member_unit_ids)

    def copy_with(
        self,
        cohesion: Optional[float] = None,
        morale_boost: Optional[float] = None,
        combat_bonus: Optional[float] = None,
        movement_penalty: Optional[float] = None,
        is_concealed: Optional[bool] = None,
    ) -> 'UnitFormation':
        """Create a modified copy of this formation."""
        return UnitFormation(
            formation_id=self.formation_id,
            formation_type=self.formation_type,
            leader_unit_id=self.leader_unit_id,
            member_unit_ids=self.member_unit_ids,
            cohesion=cohesion if cohesion is not None else self.cohesion,
            morale_boost=morale_boost if morale_boost is not None else self.morale_boost,
            combat_bonus=combat_bonus if combat_bonus is not None else self.combat_bonus,
            movement_penalty=movement_penalty if movement_penalty is not None else self.movement_penalty,
            is_concealed=is_concealed if is_concealed is not None else self.is_concealed,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Command Hierarchy                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class CommandRank(Enum):
    """Military command ranks."""
    PRIVATE = auto()       # Basic unit
    SERGEANT = auto()      # Squad leader
    LIEUTENANT = auto()    # Platoon leader
    CAPTAIN = auto()       # Company commander
    MAJOR = auto()         # Battalion commander
    COLONEL = auto()       # Regiment commander
    GENERAL = auto()       # Overall commander

@dataclass(frozen=True)
class CommandStructure:
    """
    Military command hierarchy for coordinated operations.

    Implements chain of command with different authority levels.
    """
    commander_unit_id: int
    subordinate_units: Tuple[int, ...]
    rank: CommandRank
    command_radius: float
    communication_efficiency: float
    initiative: float

    def can_command(self, unit_id: int) -> bool:
        """Check if this commander can issue orders to a unit."""
        return unit_id in self.subordinate_units

    def copy_with(
        self,
        subordinate_units: Optional[Tuple[int, ...]] = None,
        communication_efficiency: Optional[float] = None,
        initiative: Optional[float] = None,
    ) -> 'CommandStructure':
        """Create a modified copy of this command structure."""
        return CommandStructure(
            commander_unit_id=self.commander_unit_id,
            subordinate_units=subordinate_units if subordinate_units is not None else self.subordinate_units,
            rank=self.rank,
            command_radius=self.command_radius,
            communication_efficiency=communication_efficiency if communication_efficiency is not None else self.communication_efficiency,
            initiative=initiative if initiative is not None else self.initiative,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Intelligence and Reconnaissance                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class IntelligenceType(Enum):
    """Types of military intelligence."""
    RECON = auto()         # Basic reconnaissance
    SURVEILLANCE = auto()  # Continuous monitoring
    SIGINT = auto()        # Signals intelligence
    HUMINT = auto()        # Human intelligence
    IMINT = auto()         # Imagery intelligence
    ELINT = auto()         # Electronic intelligence

@dataclass(frozen=True)
class IntelligenceReport:
    """
    Intelligence information about enemy forces and terrain.
    """
    report_id: int
    intelligence_type: IntelligenceType
    target_cluster_id: int
    enemy_units_detected: int
    enemy_combat_power: float
    terrain_info: str
    last_updated: int
    confidence: float
    is_stale: bool = False

    def copy_with(
        self,
        enemy_units_detected: Optional[int] = None,
        enemy_combat_power: Optional[float] = None,
        confidence: Optional[float] = None,
        is_stale: Optional[bool] = None,
    ) -> 'IntelligenceReport':
        """Create a modified copy of this intelligence report."""
        return IntelligenceReport(
            report_id=self.report_id,
            intelligence_type=self.intelligence_type,
            target_cluster_id=self.target_cluster_id,
            enemy_units_detected=enemy_units_detected if enemy_units_detected is not None else self.enemy_units_detected,
            enemy_combat_power=enemy_combat_power if enemy_combat_power is not None else self.enemy_combat_power,
            terrain_info=self.terrain_info,
            last_updated=self.last_updated,
            confidence=confidence if confidence is not None else self.confidence,
            is_stale=is_stale if is_stale is not None else self.is_stale,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Combat Tactics                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class CombatTactic(Enum):
    """Advanced combat tactics."""
    FRONTAL_ASSAULT = auto()     # Direct attack
    FLANKING = auto()            # Attack from the side
    AMBUSH = auto()              # Surprise attack from concealment
    ENVELOPMENT = auto()         # Surround and encircle
    SUPPRESSION = auto()         # Fire superiority to pin down enemy
    WITHDRAWAL = auto()          # Tactical retreat
    FEINT = auto()               # Deceptive attack to draw attention
    BREAKTHROUGH = auto()        # Concentrated attack to penetrate lines
    # ── Politically-relevant tactics ─────────────────────────────────────── #
    COUNTERINSURGENCY = auto()   # Anti-guerrilla; hazard ↓, polarization ↑ (risk)
    PEACEKEEPING_OP = auto()     # Neutral stabilization; trust ↑, hazard ↓, no territorial gain
    BLOCKADE = auto()            # Cuts supply between clusters; resource ↓, hazard ↑ in target
    OCCUPATION = auto()          # Long-term control; stability ↓ (short), trust ↓, resource extraction
    HEARTS_AND_MINDS = auto()    # Civil-mil cooperation; trust ↑↑, polar ↓, slow hazard ↓
    SCORCHED_EARTH = auto()      # Denial tactic; resource ↓↓ in target, hazard ↑↑, pop ↓

@dataclass(frozen=True)
class TacticalOperation:
    """
    A planned tactical operation with specific objectives and methods.
    """
    operation_id: int
    name: str
    tactic: CombatTactic
    primary_target: int
    secondary_targets: Tuple[int, ...]
    participating_units: Tuple[int, ...]
    success_probability: float
    estimated_duration: int
    required_intel: float
    is_active: bool = False
    is_completed: bool = False

    def copy_with(
        self,
        success_probability: Optional[float] = None,
        is_active: Optional[bool] = None,
        is_completed: Optional[bool] = None,
    ) -> 'TacticalOperation':
        """Create a modified copy of this tactical operation."""
        return TacticalOperation(
            operation_id=self.operation_id,
            name=self.name,
            tactic=self.tactic,
            primary_target=self.primary_target,
            secondary_targets=self.secondary_targets,
            participating_units=self.participating_units,
            success_probability=success_probability if success_probability is not None else self.success_probability,
            estimated_duration=self.estimated_duration,
            required_intel=self.required_intel,
            is_active=is_active if is_active is not None else self.is_active,
            is_completed=is_completed if is_completed is not None else self.is_completed,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Electronic Warfare                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class EWCapability(Enum):
    """Electronic warfare capabilities."""
    JAMMING = auto()        # Disrupt enemy communications
    DECEPTION = auto()      # Create false signals
    INTERCEPT = auto()      # Eavesdrop on communications
    DISRUPTION = auto()     # Disable enemy systems
    SPOOFING = auto()        # Impersonate friendly signals

@dataclass(frozen=True)
class ElectronicWarfareState:
    """
    Electronic warfare capabilities and status.
    """
    ew_unit_id: int
    capabilities: Tuple[EWCapability, ...]
    jamming_range: float
    interception_range: float
    active_jamming: bool = False
    intelligence_gathered: float = 0.0
    enemy_communications_disrupted: float = 0.0

    def copy_with(
        self,
        active_jamming: Optional[bool] = None,
        intelligence_gathered: Optional[float] = None,
        enemy_communications_disrupted: Optional[float] = None,
    ) -> 'ElectronicWarfareState':
        """Create a modified copy of this EW state."""
        return ElectronicWarfareState(
            ew_unit_id=self.ew_unit_id,
            capabilities=self.capabilities,
            jamming_range=self.jamming_range,
            interception_range=self.interception_range,
            active_jamming=active_jamming if active_jamming is not None else self.active_jamming,
            intelligence_gathered=intelligence_gathered if intelligence_gathered is not None else self.intelligence_gathered,
            enemy_communications_disrupted=enemy_communications_disrupted if enemy_communications_disrupted is not None else self.enemy_communications_disrupted,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Logistics                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class SupplyType(Enum):
    """Types of military supplies."""
    AMMUNITION = auto()    # Bullets, shells, missiles
    FUEL = auto()           # Fuel for vehicles and aircraft
    FOOD = auto()           # Rations for personnel
    MEDICAL = auto()        # Medical supplies and equipment
    REPAIR = auto()         # Spare parts and repair kits
    SPECIALIZED = auto()    # Specialized equipment

@dataclass(frozen=True)
class SupplyChain:
    """
    Advanced logistics and supply chain management.
    """
    supply_depot_id: int
    cluster_id: int
    supply_types: Dict[SupplyType, float]
    capacity: float
    throughput: float
    vulnerability: float
    connected_depots: Tuple[int, ...]

    def total_supply(self) -> float:
        """Total supply across all types."""
        return sum(self.supply_types.values())

    def copy_with(
        self,
        supply_types: Optional[Dict[SupplyType, float]] = None,
        vulnerability: Optional[float] = None,
    ) -> 'SupplyChain':
        """Create a modified copy of this supply chain."""
        return SupplyChain(
            supply_depot_id=self.supply_depot_id,
            cluster_id=self.cluster_id,
            supply_types=supply_types if supply_types is not None else self.supply_types,
            capacity=self.capacity,
            throughput=self.throughput,
            vulnerability=vulnerability if vulnerability is not None else self.vulnerability,
            connected_depots=self.connected_depots,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Cluster State                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class AdvancedClusterMilitaryState:
    """
    Extended cluster military state with advanced tactical features.
    """
    cluster_id: int
    units: Tuple[MilitaryUnit, ...]
    supply_depot: float
    is_controlled: bool
    controlling_faction: Optional[int] = None
    reinforcement_timer: float = 0.0

    # Advanced features
    formations: Tuple[UnitFormation, ...] = ()
    command_structure: Optional[CommandStructure] = None
    intelligence_reports: Tuple[IntelligenceReport, ...] = ()
    active_operations: Tuple[TacticalOperation, ...] = ()
    ew_state: Optional[ElectronicWarfareState] = None
    supply_chain: Optional[SupplyChain] = None
    terrain_advantage: float = 1.0
    fog_of_war: float = 0.5

    @property
    def total_combat_power(self) -> float:
        """Total combat power including formation bonuses."""
        base_power = sum(unit.combat_power for unit in self.units if unit.is_alive)

        # Add formation bonuses
        formation_bonus = sum(
            formation.combat_bonus * sum(
                unit.combat_power for unit in self.units
                if unit.unit_id in formation.member_unit_ids and unit.is_alive
            )
            for formation in self.formations
        )

        return base_power + formation_bonus

    @property
    def unit_count(self) -> int:
        """Number of alive units in this cluster."""
        return sum(1 for unit in self.units if unit.is_alive)

    def supply_demand(self, params: MilitaryUnitParams) -> float:
        """Total supply demand from all units."""
        return sum(unit.supply_level * params.get_supply_cost(unit.unit_type)
                  for unit in self.units if unit.is_alive)

    def add_unit(self, unit: MilitaryUnit) -> 'AdvancedClusterMilitaryState':
        """Add a unit to this cluster."""
        return self.copy_with(units=(*self.units, unit))

    def remove_unit(self, unit_id: int) -> 'AdvancedClusterMilitaryState':
        """Remove a unit from this cluster."""
        new_units = tuple(u for u in self.units if u.unit_id != unit_id)
        return self.copy_with(units=new_units)

    def update_unit(self, unit: MilitaryUnit) -> 'AdvancedClusterMilitaryState':
        """Update a unit in this cluster."""
        new_units = tuple(
            u if u.unit_id != unit.unit_id else unit
            for u in self.units
        )
        return self.copy_with(units=new_units)

    def copy_with(
        self,
        units: Optional[Tuple[MilitaryUnit, ...]] = None,
        supply_depot: Optional[float] = None,
        is_controlled: Optional[bool] = None,
        controlling_faction: Optional[int] = None,
        reinforcement_timer: Optional[float] = None,
        formations: Optional[Tuple[UnitFormation, ...]] = None,
        command_structure: Optional[CommandStructure] = None,
        intelligence_reports: Optional[Tuple[IntelligenceReport, ...]] = None,
        active_operations: Optional[Tuple[TacticalOperation, ...]] = None,
        ew_state: Optional[ElectronicWarfareState] = None,
        supply_chain: Optional[SupplyChain] = None,
        terrain_advantage: Optional[float] = None,
        fog_of_war: Optional[float] = None,
    ) -> 'AdvancedClusterMilitaryState':
        """Create a modified copy of this cluster state."""
        return AdvancedClusterMilitaryState(
            cluster_id=self.cluster_id,
            units=units if units is not None else self.units,
            supply_depot=supply_depot if supply_depot is not None else self.supply_depot,
            is_controlled=is_controlled if is_controlled is not None else self.is_controlled,
            controlling_faction=controlling_faction if controlling_faction is not None else self.controlling_faction,
            reinforcement_timer=reinforcement_timer if reinforcement_timer is not None else self.reinforcement_timer,
            formations=formations if formations is not None else self.formations,
            command_structure=command_structure if command_structure is not None else self.command_structure,
            intelligence_reports=intelligence_reports if intelligence_reports is not None else self.intelligence_reports,
            active_operations=active_operations if active_operations is not None else self.active_operations,
            ew_state=ew_state if ew_state is not None else self.ew_state,
            supply_chain=supply_chain if supply_chain is not None else self.supply_chain,
            terrain_advantage=terrain_advantage if terrain_advantage is not None else self.terrain_advantage,
            fog_of_war=fog_of_war if fog_of_war is not None else self.fog_of_war,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced World State                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class AdvancedWorldMilitaryState:
    """
    Extended world military state with advanced tactical features.
    """
    clusters: Tuple[AdvancedClusterMilitaryState, ...]
    objectives: Tuple[MilitaryObjective, ...]
    global_supply: float
    global_reinforcement_pool: float
    step: int
    next_unit_id: int = 1
    next_formation_id: int = 1
    next_operation_id: int = 1
    next_report_id: int = 1

    # Global intelligence
    global_intelligence: Optional[Dict[int, IntelligenceReport]] = None  # cluster_id -> report
    electronic_warfare_active: bool = False
    communication_network_integrity: float = 1.0

    def __post_init__(self):
        # Ensure global_intelligence is always a dict (not None) so EW and
        # tactics systems can safely do dict operations on it.
        if self.global_intelligence is None:
            object.__setattr__(self, 'global_intelligence', {})

    @property
    def total_combat_power(self) -> float:
        """Total combat power across all clusters."""
        return sum(c.total_combat_power for c in self.clusters)

    @property
    def total_unit_count(self) -> int:
        """Total number of alive units."""
        return sum(c.unit_count for c in self.clusters)

    def get_cluster(self, cluster_id: int) -> Optional[AdvancedClusterMilitaryState]:
        """Get military state for a specific cluster."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def copy_with_clusters(self, new_clusters: Tuple[AdvancedClusterMilitaryState, ...]) -> 'AdvancedWorldMilitaryState':
        """Create a copy with updated clusters."""
        return AdvancedWorldMilitaryState(
            clusters=new_clusters,
            objectives=self.objectives,
            global_supply=self.global_supply,
            global_reinforcement_pool=self.global_reinforcement_pool,
            step=self.step,
            next_unit_id=self.next_unit_id,
            next_formation_id=self.next_formation_id,
            next_operation_id=self.next_operation_id,
            next_report_id=self.next_report_id,
            global_intelligence=self.global_intelligence,
            electronic_warfare_active=self.electronic_warfare_active,
            communication_network_integrity=self.communication_network_integrity,
        )

    def advance_step(self) -> 'AdvancedWorldMilitaryState':
        """Advance to next step."""
        return self.copy_with(step=self.step + 1)

    def copy_with(
        self,
        clusters: Optional[Tuple[AdvancedClusterMilitaryState, ...]] = None,
        objectives: Optional[Tuple[MilitaryObjective, ...]] = None,
        global_supply: Optional[float] = None,
        global_reinforcement_pool: Optional[float] = None,
        step: Optional[int] = None,
        next_unit_id: Optional[int] = None,
        next_formation_id: Optional[int] = None,
        next_operation_id: Optional[int] = None,
        next_report_id: Optional[int] = None,
        global_intelligence: Optional[Dict[int, IntelligenceReport]] = None,
        electronic_warfare_active: Optional[bool] = None,
        communication_network_integrity: Optional[float] = None,
    ) -> 'AdvancedWorldMilitaryState':
        """Create a modified copy of this world state."""
        return AdvancedWorldMilitaryState(
            clusters=clusters if clusters is not None else self.clusters,
            objectives=objectives if objectives is not None else self.objectives,
            global_supply=global_supply if global_supply is not None else self.global_supply,
            global_reinforcement_pool=global_reinforcement_pool if global_reinforcement_pool is not None else self.global_reinforcement_pool,
            step=step if step is not None else self.step,
            next_unit_id=next_unit_id if next_unit_id is not None else self.next_unit_id,
            next_formation_id=next_formation_id if next_formation_id is not None else self.next_formation_id,
            next_operation_id=next_operation_id if next_operation_id is not None else self.next_operation_id,
            next_report_id=next_report_id if next_report_id is not None else self.next_report_id,
            global_intelligence=global_intelligence if global_intelligence is not None else self.global_intelligence,
            electronic_warfare_active=electronic_warfare_active if electronic_warfare_active is not None else self.electronic_warfare_active,
            communication_network_integrity=communication_network_integrity if communication_network_integrity is not None else self.communication_network_integrity,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Advanced Tactics Engine                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class AdvancedTacticsEngine:
    """
    Engine for advanced tactical operations and AI decision making.
    """

    def __init__(self, params: MilitaryUnitParams):
        self.params = params
        self.tactical_ai_enabled = True

    def analyze_battlefield(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int
    ) -> Dict[str, Any]:
        """
        Analyze the tactical situation in a cluster.

        Returns comprehensive battlefield analysis including:
          - Force ratios
          - Terrain advantages
          - Supply situation
          - Intelligence quality
          - Command effectiveness
        """
        cluster = world_state.get_cluster(cluster_id)
        if cluster is None:
            return {"error": "Cluster not found"}

        # Force analysis
        friendly_units = sum(1 for u in cluster.units if u.is_alive)
        friendly_power = cluster.total_combat_power

        # For simplicity, assume some enemy presence
        enemy_units = max(0, int(friendly_units * 0.8 * (1 + np.random.normal(0, 0.2))))
        enemy_power = friendly_power * 0.9 * (1 + np.random.normal(0, 0.1))

        # Terrain analysis
        terrain_factor = cluster.terrain_advantage

        # Supply analysis
        supply_situation = "good" if cluster.supply_depot > 5 else "poor"

        # Intelligence analysis
        intel_quality = "high" if cluster.fog_of_war < 0.3 else "low"

        # Command analysis
        command_effective = cluster.command_structure is not None and cluster.command_structure.communication_efficiency > 0.7

        return {
            "cluster_id": cluster_id,
            "friendly_units": friendly_units,
            "enemy_units": enemy_units,
            "friendly_power": friendly_power,
            "enemy_power": enemy_power,
            "force_ratio": friendly_power / max(enemy_power, 1),
            "terrain_factor": terrain_factor,
            "supply_situation": supply_situation,
            "intel_quality": intel_quality,
            "command_effective": command_effective,
            "recommended_tactic": self.suggest_tactic(friendly_power, enemy_power, terrain_factor)
        }

    def suggest_tactic(
        self,
        friendly_power: float,
        enemy_power: float,
        terrain_factor: float
    ) -> CombatTactic:
        """
        Suggest an appropriate combat tactic based on current situation.
        """
        power_ratio = friendly_power / max(enemy_power, 1)

        if power_ratio > 1.5 and terrain_factor > 1.2:
            return CombatTactic.FRONTAL_ASSAULT
        elif power_ratio > 1.2:
            return CombatTactic.FLANKING
        elif power_ratio > 0.8 and terrain_factor > 1.1:
            return CombatTactic.ENVELOPMENT
        elif power_ratio < 0.7:
            return CombatTactic.WITHDRAWAL
        elif terrain_factor > 1.3:
            return CombatTactic.SUPPRESSION
        else:
            return CombatTactic.FEINT

    def create_formation(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int,
        unit_ids: List[int],
        formation_type: FormationType,
        params: MilitaryUnitParams
    ) -> Tuple[AdvancedWorldMilitaryState, Optional[UnitFormation]]:
        """
        Create a new unit formation in a cluster.
        """
        cluster = world_state.get_cluster(cluster_id)
        if cluster is None:
            return world_state, None

        # Validate units are in cluster and alive
        valid_units = [uid for uid in unit_ids
                      if any(u.unit_id == uid and u.is_alive for u in cluster.units)]

        if len(valid_units) < 2:
            return world_state, None  # Need at least 2 units for formation

        # Determine formation properties based on type
        formation_properties = self._get_formation_properties(formation_type)

        # Create formation
        formation = UnitFormation(
            formation_id=world_state.next_formation_id,
            formation_type=formation_type,
            leader_unit_id=valid_units[0],  # First unit is leader
            member_unit_ids=tuple(valid_units),
            **formation_properties
        )

        # Add formation to cluster
        new_formations = (*cluster.formations, formation)
        new_cluster = cluster.copy_with(formations=new_formations)

        # Update world state
        new_clusters = tuple(
            new_cluster if c.cluster_id == cluster_id else c
            for c in world_state.clusters
        )

        new_world = world_state.copy_with(
            clusters=new_clusters,
            next_formation_id=world_state.next_formation_id + 1
        )

        return new_world, formation

    def _get_formation_properties(self, formation_type: FormationType) -> Dict[str, float]:
        """Get properties for formation type."""
        properties = {
            FormationType.LINE: {
                "cohesion": 0.8, "morale_boost": 0.1, "combat_bonus": 0.15, "movement_penalty": 0.1
            },
            FormationType.WEDGE: {
                "cohesion": 0.7, "morale_boost": 0.2, "combat_bonus": 0.25, "movement_penalty": 0.05
            },
            FormationType.COLUMN: {
                "cohesion": 0.9, "morale_boost": 0.05, "combat_bonus": 0.1, "movement_penalty": 0.0
            },
            FormationType.SQUARE: {
                "cohesion": 0.9, "morale_boost": 0.15, "combat_bonus": 0.3, "movement_penalty": 0.2
            },
            FormationType.SKIRMISH: {
                "cohesion": 0.6, "morale_boost": 0.05, "combat_bonus": 0.1, "movement_penalty": -0.1
            },
            FormationType.AMBUSH: {
                "cohesion": 0.7, "morale_boost": 0.0, "combat_bonus": 0.5, "movement_penalty": 0.1, "is_concealed": True
            },
            FormationType.PHALANX: {
                "cohesion": 0.95, "morale_boost": 0.2, "combat_bonus": 0.4, "movement_penalty": 0.3
            },
        }

        return properties.get(formation_type, properties[FormationType.LINE])

    def plan_tactical_operation(
        self,
        world_state: AdvancedWorldMilitaryState,
        operation_type: CombatTactic,
        primary_target: int,
        participating_units: List[int],
        params: MilitaryUnitParams
    ) -> Tuple[AdvancedWorldMilitaryState, Optional[TacticalOperation]]:
        """
        Plan a new tactical operation.
        """
        # Validate target cluster exists
        target_cluster = world_state.get_cluster(primary_target)
        if target_cluster is None:
            return world_state, None

        # Validate participating units
        valid_units = []
        for unit_id in participating_units:
            for cluster in world_state.clusters:
                if any(u.unit_id == unit_id and u.is_alive for u in cluster.units):
                    valid_units.append(unit_id)
                    break

        if len(valid_units) < 1:
            return world_state, None

        # Determine operation properties
        op_properties = self._get_operation_properties(operation_type, len(valid_units))

        # Create operation
        operation = TacticalOperation(
            operation_id=world_state.next_operation_id,
            name=f"{operation_type.name.title()} Operation",
            tactic=operation_type,
            primary_target=primary_target,
            secondary_targets=(),
            participating_units=tuple(valid_units),
            **op_properties
        )

        # Add operation to primary target cluster
        new_operations = (*target_cluster.active_operations, operation)
        new_cluster = target_cluster.copy_with(active_operations=new_operations)

        # Update world state
        new_clusters = tuple(
            new_cluster if c.cluster_id == primary_target else c
            for c in world_state.clusters
        )

        new_world = world_state.copy_with(
            clusters=new_clusters,
            next_operation_id=world_state.next_operation_id + 1
        )

        return new_world, operation

    def _get_operation_properties(
        self,
        operation_type: CombatTactic,
        unit_count: int
    ) -> Dict[str, Any]:
        """Get properties for operation type."""
        base_success = 0.5 + 0.1 * min(unit_count, 5) * 0.1
        base_duration = max(2, 5 - unit_count // 2)
        base_intel = 0.3 + 0.1 * unit_count * 0.1

        properties = {
            CombatTactic.FRONTAL_ASSAULT: {
                "success_probability": base_success * 1.1,
                "estimated_duration": base_duration * 1.2,
                "required_intel": base_intel * 0.8
            },
            CombatTactic.FLANKING: {
                "success_probability": base_success * 1.3,
                "estimated_duration": base_duration * 1.5,
                "required_intel": base_intel * 1.2
            },
            CombatTactic.AMBUSH: {
                "success_probability": base_success * 1.8,
                "estimated_duration": base_duration * 0.8,
                "required_intel": base_intel * 1.5
            },
            CombatTactic.ENVELOPMENT: {
                "success_probability": base_success * 1.5,
                "estimated_duration": base_duration * 2.0,
                "required_intel": base_intel * 1.3
            },
            CombatTactic.SUPPRESSION: {
                "success_probability": base_success * 1.2,
                "estimated_duration": base_duration * 1.0,
                "required_intel": base_intel * 0.9
            },
            CombatTactic.WITHDRAWAL: {
                "success_probability": base_success * 1.6,
                "estimated_duration": base_duration * 0.7,
                "required_intel": base_intel * 0.7
            },
            CombatTactic.FEINT: {
                "success_probability": base_success * 1.4,
                "estimated_duration": base_duration * 1.0,
                "required_intel": base_intel * 1.1
            },
            CombatTactic.BREAKTHROUGH: {
                "success_probability": base_success * 1.2,
                "estimated_duration": base_duration * 1.8,
                "required_intel": base_intel * 1.4
            },
            # ── Politically-relevant tactics ────────────────────────────── #
            CombatTactic.COUNTERINSURGENCY: {
                "success_probability": base_success * 0.9,   # hard to win hearts
                "estimated_duration": base_duration * 3.0,   # long campaigns
                "required_intel": base_intel * 2.0,          # needs deep intel
            },
            CombatTactic.PEACEKEEPING_OP: {
                "success_probability": base_success * 1.1,
                "estimated_duration": base_duration * 2.5,
                "required_intel": base_intel * 0.8,
            },
            CombatTactic.BLOCKADE: {
                "success_probability": base_success * 1.4,   # easy to enforce
                "estimated_duration": base_duration * 4.0,   # sustained pressure
                "required_intel": base_intel * 0.6,
            },
            CombatTactic.OCCUPATION: {
                "success_probability": base_success * 1.3,
                "estimated_duration": base_duration * 5.0,   # very long
                "required_intel": base_intel * 0.7,
            },
            CombatTactic.HEARTS_AND_MINDS: {
                "success_probability": base_success * 0.8,   # hard to win
                "estimated_duration": base_duration * 4.0,
                "required_intel": base_intel * 1.5,
            },
            CombatTactic.SCORCHED_EARTH: {
                "success_probability": base_success * 1.8,   # easy to destroy
                "estimated_duration": base_duration * 0.5,   # fast and brutal
                "required_intel": base_intel * 0.3,
            },
        }

        return properties.get(operation_type, properties[CombatTactic.FRONTAL_ASSAULT])

    def execute_tactical_operation(
        self,
        world_state: AdvancedWorldMilitaryState,
        operation_id: int,
        params: MilitaryUnitParams,
        rng: np.random.Generator
    ) -> AdvancedWorldMilitaryState:
        """
        Execute a planned tactical operation.
        """
        # Find the operation
        operation = None
        target_cluster = None

        for cluster in world_state.clusters:
            for op in cluster.active_operations:
                if op.operation_id == operation_id:
                    operation = op
                    target_cluster = cluster
                    break
            if operation is not None:
                break

        if operation is None or target_cluster is None:
            return world_state

        # Check if operation can be executed
        if operation.is_completed or operation.is_active:
            return world_state

        # Activate the operation
        new_operation = operation.copy_with(is_active=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation_id else op
            for op in target_cluster.active_operations
        )
        new_cluster = target_cluster.copy_with(active_operations=new_operations)

        # Apply operation effects based on tactic
        new_world = self._apply_tactical_effects(
            world_state.copy_with_clusters(
                tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                     for c in world_state.clusters)
            ),
            new_operation,
            params,
            rng
        )

        return new_world

    def _apply_tactical_effects(
        self,
        world_state: AdvancedWorldMilitaryState,
        operation: TacticalOperation,
        params: MilitaryUnitParams,
        rng: np.random.Generator
    ) -> AdvancedWorldMilitaryState:
        """
        Apply the effects of a tactical operation.
        """
        target_cluster = world_state.get_cluster(operation.primary_target)
        if target_cluster is None:
            return world_state

        # Determine success based on probability and randomness
        success_roll = rng.random()
        success = success_roll < operation.success_probability

        # Apply tactic-specific effects
        if operation.tactic == CombatTactic.FRONTAL_ASSAULT:
            return self._apply_frontal_assault(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.FLANKING:
            return self._apply_flanking(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.AMBUSH:
            return self._apply_ambush(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.ENVELOPMENT:
            return self._apply_envelopment(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.SUPPRESSION:
            return self._apply_suppression(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.WITHDRAWAL:
            return self._apply_withdrawal(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.FEINT:
            return self._apply_feint(world_state, target_cluster, operation, success, params)
        elif operation.tactic == CombatTactic.BREAKTHROUGH:
            return self._apply_breakthrough(world_state, target_cluster, operation, success, params)
        else:
            return world_state

    def _apply_frontal_assault(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply frontal assault effects."""
        # Boost morale and combat effectiveness of participating units
        new_units = []
        for unit in target_cluster.units:
            if unit.unit_id in operation.participating_units and unit.is_alive:
                morale_boost = 0.2 if success else 0.1
                combat_boost = 0.15 if success else 0.05

                new_unit = unit.copy_with(
                    morale=min(1.0, unit.morale + morale_boost),
                    combat_effectiveness=min(1.0, unit.combat_effectiveness + combat_boost)
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_flanking(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply flanking maneuver effects."""
        # Flanking provides combat advantage
        new_units = []
        for unit in target_cluster.units:
            if unit.unit_id in operation.participating_units and unit.is_alive:
                combat_boost = 0.3 if success else 0.15

                new_unit = unit.copy_with(
                    combat_effectiveness=min(1.0, unit.combat_effectiveness + combat_boost)
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_envelopment(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """
        Apply encirclement / envelopment effects.

        A successful envelopment surrounds the enemy, cutting off retreat and
        supply. Effects:
          - Significant morale drop on ALL enemy units in the target cluster
            (they realise they are encircled).
          - Moderate combat power boost for participating units.
          - Supply depot of the target cluster is reduced (lines cut).
        Failed envelopment: minor morale drain on own units (over-extension).
        """
        if success:
            new_units = []
            for unit in target_cluster.units:
                if unit.is_alive:
                    if unit.unit_id in operation.participating_units:
                        # Our units get a combat effectiveness boost
                        new_units.append(unit.copy_with(
                            combat_effectiveness=min(1.0, unit.combat_effectiveness + 0.20),
                            morale=min(1.0, unit.morale + 0.10),
                        ))
                    else:
                        # Enemy units suffer morale collapse from encirclement
                        new_units.append(unit.copy_with(
                            morale=max(0.0, unit.morale - 0.30),
                            combat_effectiveness=max(0.1, unit.combat_effectiveness - 0.15),
                        ))
                else:
                    new_units.append(unit)
            # Supply interdiction: encirclement cuts supply lines to cluster
            new_depot = max(0.0, target_cluster.supply_depot * 0.50)
        else:
            # Failed envelopment: our units are over-extended
            new_units = []
            for unit in target_cluster.units:
                if unit.is_alive and unit.unit_id in operation.participating_units:
                    new_units.append(unit.copy_with(
                        morale=max(0.0, unit.morale - 0.10),
                        combat_effectiveness=max(0.1, unit.combat_effectiveness - 0.05),
                    ))
                else:
                    new_units.append(unit)
            new_depot = target_cluster.supply_depot

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            supply_depot=new_depot,
            active_operations=new_operations,
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_ambush(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply ambush effects."""
        # Ambush provides significant combat bonus if successful
        new_units = []
        for unit in target_cluster.units:
            if unit.unit_id in operation.participating_units and unit.is_alive:
                combat_boost = 0.5 if success else 0.0  # Ambush either works or fails completely

                # Note: is_concealed lives on UnitFormation, not MilitaryUnit.
                # The formation's concealment is lifted when the ambush fires;
                # that update is handled at the formation level separately.
                new_unit = unit.copy_with(
                    combat_effectiveness=min(1.0, unit.combat_effectiveness + combat_boost),
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_suppression(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply suppression fire effects."""
        # Suppression reduces enemy effectiveness in the cluster
        # For now, we'll model this as a morale boost to our units
        new_units = []
        for unit in target_cluster.units:
            if unit.unit_id in operation.participating_units and unit.is_alive:
                morale_boost = 0.1 if success else 0.05

                new_unit = unit.copy_with(
                    morale=min(1.0, unit.morale + morale_boost)
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_withdrawal(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply tactical withdrawal effects."""
        # Withdrawal moves units to adjacent clusters
        # For simplicity, we'll just mark the operation as completed
        # In a full implementation, this would move units

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def _apply_feint(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply feint attack effects."""
        # Feint creates confusion and may reveal enemy positions
        # For now, we'll model this as an intelligence gain

        # Create or update intelligence report
        new_report = IntelligenceReport(
            report_id=world_state.next_report_id,
            intelligence_type=IntelligenceType.RECON,
            target_cluster_id=target_cluster.cluster_id,
            enemy_units_detected=max(1, int(operation.participating_units.__len__() * 0.8)),
            enemy_combat_power=float(operation.participating_units.__len__()) * 50.0,
            terrain_info="open",
            last_updated=world_state.step,
            confidence=0.7 if success else 0.4
        )

        new_reports = (*target_cluster.intelligence_reports, new_report)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            intelligence_reports=new_reports,
            active_operations=new_operations
        )

        return world_state.copy_with(
            clusters=tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                         for c in world_state.clusters),
            next_report_id=world_state.next_report_id + 1
        )

    def _apply_breakthrough(
        self,
        world_state: AdvancedWorldMilitaryState,
        target_cluster: AdvancedClusterMilitaryState,
        operation: TacticalOperation,
        success: bool,
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """Apply breakthrough attack effects."""
        # Breakthrough provides significant combat bonus for concentrated attack
        new_units = []
        for unit in target_cluster.units:
            if unit.unit_id in operation.participating_units and unit.is_alive:
                combat_boost = 0.4 if success else 0.1

                new_unit = unit.copy_with(
                    combat_effectiveness=min(1.0, unit.combat_effectiveness + combat_boost),
                    hit_points=min(params.get_max_hp(unit.unit_type), unit.hit_points * 1.1)  # Minor HP boost from breakthrough
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Complete the operation
        new_operation = operation.copy_with(is_active=False, is_completed=True)
        new_operations = tuple(
            new_operation if op.operation_id == operation.operation_id else op
            for op in target_cluster.active_operations
        )

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            active_operations=new_operations
        )

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == target_cluster.cluster_id else c
                 for c in world_state.clusters)
        )

    def update_intelligence(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int,
        intel_type: IntelligenceType,
        params: MilitaryUnitParams,
        rng: np.random.Generator
    ) -> AdvancedWorldMilitaryState:
        """
        Update intelligence information for a cluster.
        """
        target_cluster = world_state.get_cluster(cluster_id)
        if target_cluster is None:
            return world_state

        # Generate intelligence report
        enemy_units = max(1, int(target_cluster.unit_count * 0.8 * (1 + rng.normal(0, 0.2))))
        enemy_power = target_cluster.total_combat_power * 0.9 * (1 + rng.normal(0, 0.1))

        new_report = IntelligenceReport(
            report_id=world_state.next_report_id,
            intelligence_type=intel_type,
            target_cluster_id=cluster_id,
            enemy_units_detected=enemy_units,
            enemy_combat_power=enemy_power,
            terrain_info=self._generate_terrain_info(rng),
            last_updated=world_state.step,
            confidence=0.6 + 0.3 * rng.random()
        )

        new_reports = (*target_cluster.intelligence_reports, new_report)

        new_cluster = target_cluster.copy_with(intelligence_reports=new_reports)

        return world_state.copy_with(
            clusters=tuple(new_cluster if c.cluster_id == cluster_id else c
                         for c in world_state.clusters),
            next_report_id=world_state.next_report_id + 1
        )

    def _generate_terrain_info(self, rng: np.random.Generator) -> str:
        """Generate random terrain information."""
        terrain_types = ["open", "urban", "forest", "mountain", "desert", "swamp"]
        return rng.choice(terrain_types)

    def establish_command_structure(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int,
        commander_unit_id: int,
        subordinate_unit_ids: List[int],
        rank: CommandRank
    ) -> AdvancedWorldMilitaryState:
        """
        Establish a command structure in a cluster.
        """
        target_cluster = world_state.get_cluster(cluster_id)
        if target_cluster is None:
            return world_state

        # Validate commander and subordinates
        all_units = [u.unit_id for u in target_cluster.units if u.is_alive]
        if commander_unit_id not in all_units:
            return world_state

        valid_subordinates = [uid for uid in subordinate_unit_ids if uid in all_units and uid != commander_unit_id]

        # Determine command properties based on rank
        command_properties = self._get_command_properties(rank)

        # Create command structure
        command_structure = CommandStructure(
            commander_unit_id=commander_unit_id,
            subordinate_units=tuple(valid_subordinates),
            rank=rank,
            **command_properties
        )

        new_cluster = target_cluster.copy_with(command_structure=command_structure)

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == cluster_id else c
                 for c in world_state.clusters)
        )

    def _get_command_properties(self, rank: CommandRank) -> Dict[str, float]:
        """Get properties for command rank."""
        properties = {
            CommandRank.PRIVATE: {"command_radius": 1, "communication_efficiency": 0.5, "initiative": 0.3},
            CommandRank.SERGEANT: {"command_radius": 2, "communication_efficiency": 0.7, "initiative": 0.5},
            CommandRank.LIEUTENANT: {"command_radius": 3, "communication_efficiency": 0.8, "initiative": 0.7},
            CommandRank.CAPTAIN: {"command_radius": 4, "communication_efficiency": 0.85, "initiative": 0.8},
            CommandRank.MAJOR: {"command_radius": 5, "communication_efficiency": 0.9, "initiative": 0.85},
            CommandRank.COLONEL: {"command_radius": 6, "communication_efficiency": 0.92, "initiative": 0.9},
            CommandRank.GENERAL: {"command_radius": 8, "communication_efficiency": 0.95, "initiative": 0.95},
        }

        return properties.get(rank, properties[CommandRank.SERGEANT])

    def setup_supply_chain(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int,
        connected_depots: List[int]
    ) -> AdvancedWorldMilitaryState:
        """
        Setup a supply chain for a cluster.
        """
        target_cluster = world_state.get_cluster(cluster_id)
        if target_cluster is None:
            return world_state

        # Create supply chain with initial supplies
        supply_types = {
            SupplyType.AMMUNITION: 20.0,
            SupplyType.FUEL: 15.0,
            SupplyType.FOOD: 10.0,
            SupplyType.MEDICAL: 5.0,
            SupplyType.REPAIR: 8.0,
            SupplyType.SPECIALIZED: 3.0
        }

        supply_chain = SupplyChain(
            supply_depot_id=world_state.next_unit_id + 1000,  # Use high ID for depots
            cluster_id=cluster_id,
            supply_types=supply_types,
            capacity=100.0,
            throughput=10.0,
            vulnerability=0.2,
            connected_depots=tuple(connected_depots)
        )

        new_cluster = target_cluster.copy_with(supply_chain=supply_chain)

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == cluster_id else c
                 for c in world_state.clusters)
        )

    def update_supply_chain(
        self,
        world_state: AdvancedWorldMilitaryState,
        cluster_id: int,
        supply_consumption: Dict[SupplyType, float],
        params: MilitaryUnitParams
    ) -> AdvancedWorldMilitaryState:
        """
        Update supply chain based on consumption.
        """
        target_cluster = world_state.get_cluster(cluster_id)
        if target_cluster is None or target_cluster.supply_chain is None:
            return world_state

        # Update supply levels
        new_supply_types = {}
        for supply_type, amount in target_cluster.supply_chain.supply_types.items():
            consumed = supply_consumption.get(supply_type, 0.0)
            new_amount = max(0.0, amount - consumed)
            new_supply_types[supply_type] = new_amount

        # Add some regeneration based on throughput
        total_supply = sum(new_supply_types.values())
        capacity = target_cluster.supply_chain.capacity
        if total_supply < capacity:
            regen_amount = min(
                capacity - total_supply,
                target_cluster.supply_chain.throughput
            )
            # Distribute regeneration proportionally
            if total_supply > 0:
                for supply_type in new_supply_types:
                    new_supply_types[supply_type] += regen_amount * (new_supply_types[supply_type] / total_supply)

        new_supply_chain = target_cluster.supply_chain.copy_with(
            supply_types=new_supply_types
        )

        new_cluster = target_cluster.copy_with(supply_chain=new_supply_chain)

        return world_state.copy_with_clusters(
            tuple(new_cluster if c.cluster_id == cluster_id else c
                 for c in world_state.clusters)
        )