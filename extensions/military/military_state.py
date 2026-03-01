"""
standardized_state.py — Standardized military state definitions using enhanced units.

This module replaces the old military_state.py with a clean, standardized implementation
that uses only the new EnhancedMilitaryUnit system and removes all backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray

from .unit_types import (
    MilitaryUnitType, UnitRole, get_unit_role,
    ExpandedUnitParams, EnhancedMilitaryUnit,
    CombatMatrix, SupportCompany, SupportCompanyType
)

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Military Unit Parameters                                         #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class StandardizedUnitParams:
    """
    Standardized unit parameters using ExpandedUnitParams as the single source of truth.

    This replaces the old MilitaryUnitParams with its 6 hardcoded types and
    uses the comprehensive 40-type system from unit_types.py.
    """
    # Core parameters (delegated to ExpandedUnitParams)
    expanded_params: ExpandedUnitParams
    combat_matrix: CombatMatrix

    # Combat parameters (moved from old params)
    combat_effectiveness_decay: float = 0.05
    supply_consumption_rate: float = 0.01
    reinforcement_rate: float = 0.02
    attrition_rate: float = 0.005

    # Movement parameters
    movement_cost_road: float = 1.0
    movement_cost_rough: float = 1.5
    movement_cost_mountain: float = 2.0

    # Objective parameters
    objective_capture_threshold: float = 0.7
    objective_hold_duration: int = 5

    def get_combat_power(self, unit_type: MilitaryUnitType) -> float:
        """Delegate to expanded params."""
        return self.expanded_params.get_combat_power(unit_type)

    def get_max_hp(self, unit_type: MilitaryUnitType) -> float:
        """Delegate to expanded params."""
        return self.expanded_params.get_max_hp(unit_type)

    def get_speed(self, unit_type: MilitaryUnitType) -> float:
        """Delegate to expanded params."""
        return self.expanded_params.get_speed(unit_type)

    def get_supply_cost(self, unit_type: MilitaryUnitType) -> float:
        """Delegate to expanded params."""
        return self.expanded_params.get_supply_cost(unit_type)

    def get_combat_matrix_multiplier(
        self,
        attacker_type: MilitaryUnitType,
        defender_type: MilitaryUnitType
    ) -> float:
        """Get combat effectiveness multiplier from matrix."""
        return self.combat_matrix.get_damage_multiplier(attacker_type, defender_type)

    @classmethod
    def default(cls) -> 'StandardizedUnitParams':
        """Create default standardized parameters."""
        return cls(
            expanded_params=ExpandedUnitParams(),
            combat_matrix=CombatMatrix.default_matrix()
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Military Unit                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class StandardizedMilitaryUnit:
    """
    Standardized military unit using EnhancedMilitaryUnit as the base.

    This replaces the old MilitaryUnit class and adds standardization features.
    """
    unit_id: int
    unit_type: MilitaryUnitType
    cluster_id: int
    hit_points: float
    combat_effectiveness: float
    supply_level: float
    experience: float
    morale: float
    objective_id: Optional[int] = None
    support_companies: Tuple[SupportCompany, ...] = ()
    terrain_bonus: float = 1.0
    entrenchment: float = 0.0
    faction_id: Optional[int] = None  # Added for faction tracking

    @property
    def is_alive(self) -> bool:
        """Check if unit has any hit points remaining."""
        return self.hit_points > 0.01

    @property
    def combat_power(self) -> float:
        """Current combat power considering all factors."""
        base_power = self.hit_points * self.combat_effectiveness

        # Apply support company bonuses
        support_bonuses = self._calculate_support_bonuses()
        combat_multiplier = 1.0 + support_bonuses.get('combat_effectiveness', 0.0)

        return base_power * combat_multiplier * self.terrain_bonus * (1.0 + self.entrenchment * 0.5)

    def _calculate_support_bonuses(self) -> Dict[str, float]:
        """Calculate cumulative bonuses from support companies."""
        bonuses = {}

        for company in self.support_companies:
            company_bonuses = company.get_bonuses()
            for bonus_type, value in company_bonuses.items():
                bonuses[bonus_type] = bonuses.get(bonus_type, 0.0) + value

        return bonuses

    def get_attack_values(self, params: StandardizedUnitParams) -> Dict[str, float]:
        """Get attack values for different damage types."""
        base_stats = params.expanded_params.unit_params.get(self.unit_type, {})
        support_bonuses = self._calculate_support_bonuses()

        attack_values = {
            'soft_attack': base_stats.get('soft_attack', 1.0) + support_bonuses.get('soft_attack', 0.0),
            'hard_attack': base_stats.get('hard_attack', 0.5) + support_bonuses.get('hard_attack', 0.0),
            'air_attack': base_stats.get('air_attack', 0.0) + support_bonuses.get('air_attack', 0.0),
        }

        return attack_values

    def get_defense_values(self, params: StandardizedUnitParams) -> Dict[str, float]:
        """Get defense values for different damage types."""
        base_stats = params.expanded_params.unit_params.get(self.unit_type, {})
        support_bonuses = self._calculate_support_bonuses()

        defense_values = {
            'defense': base_stats.get('defense', 1.0) + support_bonuses.get('defense', 0.0),
            'breakthrough': base_stats.get('breakthrough', 0.3) + support_bonuses.get('breakthrough', 0.0),
            'air_defense': base_stats.get('air_defense', 0.0) + support_bonuses.get('air_defense', 0.0),
            'armor': base_stats.get('armor', 0.0) + support_bonuses.get('armor', 0.0),
        }

        return defense_values

    def copy_with(
        self,
        cluster_id: Optional[int] = None,
        hit_points: Optional[float] = None,
        combat_effectiveness: Optional[float] = None,
        supply_level: Optional[float] = None,
        experience: Optional[float] = None,
        morale: Optional[float] = None,
        objective_id: Optional[int] = None,
        support_companies: Optional[Tuple[SupportCompany, ...]] = None,
        terrain_bonus: Optional[float] = None,
        entrenchment: Optional[float] = None,
        faction_id: Optional[int] = None,
    ) -> 'StandardizedMilitaryUnit':
        """Create a modified copy of this unit."""
        return StandardizedMilitaryUnit(
            unit_id=self.unit_id,
            unit_type=self.unit_type,
            cluster_id=cluster_id if cluster_id is not None else self.cluster_id,
            hit_points=hit_points if hit_points is not None else self.hit_points,
            combat_effectiveness=combat_effectiveness if combat_effectiveness is not None else self.combat_effectiveness,
            supply_level=supply_level if supply_level is not None else self.supply_level,
            experience=experience if experience is not None else self.experience,
            morale=morale if morale is not None else self.morale,
            objective_id=objective_id if objective_id is not None else self.objective_id,
            support_companies=support_companies if support_companies is not None else self.support_companies,
            terrain_bonus=terrain_bonus if terrain_bonus is not None else self.terrain_bonus,
            entrenchment=entrenchment if entrenchment is not None else self.entrenchment,
            faction_id=faction_id if faction_id is not None else self.faction_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert unit to dictionary for serialization."""
        return {
            'unit_id': self.unit_id,
            'unit_type': self.unit_type.name,
            'cluster_id': self.cluster_id,
            'hit_points': self.hit_points,
            'combat_effectiveness': self.combat_effectiveness,
            'supply_level': self.supply_level,
            'experience': self.experience,
            'morale': self.morale,
            'objective_id': self.objective_id,
            'support_companies': [c.company_type.name for c in self.support_companies],
            'terrain_bonus': self.terrain_bonus,
            'entrenchment': self.entrenchment,
            'faction_id': self.faction_id,
        }

    def add_support_company(self, company: SupportCompany) -> 'StandardizedMilitaryUnit':
        """Add a support company to this unit."""
        new_companies = (*self.support_companies, company)
        return self.copy_with(support_companies=new_companies)

    def remove_support_company(self, company_type: SupportCompanyType) -> 'StandardizedMilitaryUnit':
        """Remove a support company from this unit."""
        new_companies = tuple(c for c in self.support_companies if c.company_type != company_type)
        return self.copy_with(support_companies=new_companies)

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Cluster Military State                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class StandardizedClusterMilitaryState:
    """
    Standardized military state for a single cluster.

    Tracks units present, supply levels, control status, and faction presence.
    """
    cluster_id: int
    units: Tuple[StandardizedMilitaryUnit, ...]
    supply_depot: float
    is_controlled: bool
    controlling_faction: Optional[int] = None
    reinforcement_timer: float = 0.0
    faction_presence: Dict[int, float] = None  # faction_id -> presence strength

    def __post_init__(self):
        if self.faction_presence is None:
            object.__setattr__(self, 'faction_presence', {})

    @property
    def total_combat_power(self) -> float:
        """Total combat power of all units in this cluster."""
        return sum(unit.combat_power for unit in self.units if unit.is_alive)

    @property
    def unit_count(self) -> int:
        """Number of alive units in this cluster."""
        return sum(1 for unit in self.units if unit.is_alive)

    def supply_demand(self, params: StandardizedUnitParams) -> float:
        """Total supply demand from all units."""
        return sum(
            unit.supply_level * params.get_supply_cost(unit.unit_type)
            for unit in self.units if unit.is_alive
        )

    def add_unit(self, unit: StandardizedMilitaryUnit) -> 'StandardizedClusterMilitaryState':
        """Add a unit to this cluster."""
        return self.copy_with(units=(*self.units, unit))

    def remove_unit(self, unit_id: int) -> 'StandardizedClusterMilitaryState':
        """Remove a unit from this cluster."""
        new_units = tuple(u for u in self.units if u.unit_id != unit_id)
        return self.copy_with(units=new_units)

    def update_unit(self, unit: StandardizedMilitaryUnit) -> 'StandardizedClusterMilitaryState':
        """Update a unit in this cluster."""
        new_units = tuple(
            u if u.unit_id != unit.unit_id else unit
            for u in self.units
        )
        return self.copy_with(units=new_units)

    def copy_with(
        self,
        units: Optional[Tuple[StandardizedMilitaryUnit, ...]] = None,
        supply_depot: Optional[float] = None,
        is_controlled: Optional[bool] = None,
        controlling_faction: Optional[int] = None,
        reinforcement_timer: Optional[float] = None,
        faction_presence: Optional[Dict[int, float]] = None,
    ) -> 'StandardizedClusterMilitaryState':
        """Create a modified copy of this cluster state."""
        return StandardizedClusterMilitaryState(
            cluster_id=self.cluster_id,
            units=units if units is not None else self.units,
            supply_depot=supply_depot if supply_depot is not None else self.supply_depot,
            is_controlled=is_controlled if is_controlled is not None else self.is_controlled,
            controlling_faction=controlling_faction if controlling_faction is not None else self.controlling_faction,
            reinforcement_timer=reinforcement_timer if reinforcement_timer is not None else self.reinforcement_timer,
            faction_presence=faction_presence if faction_presence is not None else self.faction_presence,
        )

    def update_faction_presence(self) -> 'StandardizedClusterMilitaryState':
        """Update faction presence based on current units."""
        new_presence = {}

        for unit in self.units:
            if unit.is_alive and unit.faction_id is not None:
                current = new_presence.get(unit.faction_id, 0.0)
                new_presence[unit.faction_id] = current + unit.combat_power

        return self.copy_with(faction_presence=new_presence)

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized World Military State                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class StandardizedWorldMilitaryState:
    """
    Standardized global military state containing all clusters and objectives.
    """
    clusters: Tuple[StandardizedClusterMilitaryState, ...]
    objectives: Tuple['StandardizedMilitaryObjective', ...]
    global_supply: float
    global_reinforcement_pool: float
    step: int
    next_unit_id: int = 1
    factions: Dict[int, Dict[str, Any]] = None  # faction_id -> faction data

    def __post_init__(self):
        if self.factions is None:
            object.__setattr__(self, 'factions', {})

    @property
    def total_combat_power(self) -> float:
        """Total combat power across all clusters."""
        return sum(c.total_combat_power for c in self.clusters)

    @property
    def total_unit_count(self) -> int:
        """Total number of alive units."""
        return sum(c.unit_count for c in self.clusters)

    def get_cluster(self, cluster_id: int) -> Optional[StandardizedClusterMilitaryState]:
        """Get military state for a specific cluster."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def copy_with_clusters(self, new_clusters: Tuple[StandardizedClusterMilitaryState, ...]) -> 'StandardizedWorldMilitaryState':
        """Create a copy with updated clusters."""
        return StandardizedWorldMilitaryState(
            clusters=new_clusters,
            objectives=self.objectives,
            global_supply=self.global_supply,
            global_reinforcement_pool=self.global_reinforcement_pool,
            step=self.step,
            next_unit_id=self.next_unit_id,
            factions=self.factions,
        )

    def advance_step(self) -> 'StandardizedWorldMilitaryState':
        """Advance to next step."""
        return self.copy_with(step=self.step + 1)

    def copy_with(
        self,
        clusters: Optional[Tuple[StandardizedClusterMilitaryState, ...]] = None,
        objectives: Optional[Tuple['StandardizedMilitaryObjective', ...]] = None,
        global_supply: Optional[float] = None,
        global_reinforcement_pool: Optional[float] = None,
        step: Optional[int] = None,
        next_unit_id: Optional[int] = None,
        factions: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> 'StandardizedWorldMilitaryState':
        """Create a modified copy of this world state."""
        return StandardizedWorldMilitaryState(
            clusters=clusters if clusters is not None else self.clusters,
            objectives=objectives if objectives is not None else self.objectives,
            global_supply=global_supply if global_supply is not None else self.global_supply,
            global_reinforcement_pool=global_reinforcement_pool if global_reinforcement_pool is not None else self.global_reinforcement_pool,
            step=step if step is not None else self.step,
            next_unit_id=next_unit_id if next_unit_id is not None else self.next_unit_id,
            factions=factions if factions is not None else self.factions,
        )

    def add_faction(self, faction_id: int, name: str, **kwargs) -> 'StandardizedWorldMilitaryState':
        """Add a new faction to the world state."""
        new_factions = dict(self.factions)
        new_factions[faction_id] = {'name': name, **kwargs}
        return self.copy_with(factions=new_factions)

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Military Objectives                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class StandardizedMilitaryObjective:
    """
    Standardized military objective with faction tracking.

    Objectives provide structure to military operations and contribute to
    victory conditions.
    """
    objective_id: int
    name: str
    objective_type: str  # "capture", "hold", "destroy", "escort", etc.
    target_cluster_id: int
    required_units: int
    reward_value: float
    faction_id: Optional[int] = None  # Which faction owns this objective
    completion_progress: float = 0.0
    is_completed: bool = False
    completion_step: Optional[int] = None

    def update_progress(self, progress_delta: float, current_step: int) -> 'StandardizedMilitaryObjective':
        """Update objective progress."""
        new_progress = min(1.0, self.completion_progress + progress_delta)
        new_completed = new_progress >= 0.99
        new_completion_step = current_step if new_completed and not self.is_completed else self.completion_step

        return self.copy_with(
            completion_progress=new_progress,
            is_completed=new_completed,
            completion_step=new_completion_step,
        )

    def copy_with(
        self,
        completion_progress: Optional[float] = None,
        is_completed: Optional[bool] = None,
        completion_step: Optional[int] = None,
        faction_id: Optional[int] = None,
    ) -> 'StandardizedMilitaryObjective':
        """Create a modified copy of this objective."""
        return StandardizedMilitaryObjective(
            objective_id=self.objective_id,
            name=self.name,
            objective_type=self.objective_type,
            target_cluster_id=self.target_cluster_id,
            required_units=self.required_units,
            reward_value=self.reward_value,
            faction_id=faction_id if faction_id is not None else self.faction_id,
            completion_progress=completion_progress if completion_progress is not None else self.completion_progress,
            is_completed=is_completed if is_completed is not None else self.is_completed,
            completion_step=completion_step if completion_step is not None else self.completion_step,
        )

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Combat Functions                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def calculate_standardized_damage(
    attacker: StandardizedMilitaryUnit,
    defender: StandardizedMilitaryUnit,
    params: StandardizedUnitParams,
    terrain_factor: float = 1.0,
    is_surprise_attack: bool = False,
    is_flanked: bool = False,
    is_entrenchment_advantage: bool = False,
) -> Tuple[float, float]:
    """
    Calculate damage between two units using standardized combat system.

    This replaces the old calculate_damage function with enhanced features.
    """
    # Get attack and defense values
    attacker_stats = attacker.get_attack_values(params)
    defender_stats = defender.get_defense_values(params)

    # Get combat matrix multiplier
    matrix_multiplier = params.get_combat_matrix_multiplier(attacker.unit_type, defender.unit_type)

    # Calculate base combat power
    attacker_power = attacker.combat_power
    defender_power = defender.combat_power * terrain_factor

    # Apply tactical modifiers
    if is_surprise_attack:
        attacker_power *= 1.5
    if is_flanked:
        defender_power *= 0.7
    if is_entrenchment_advantage:
        defender_power *= 1.3

    # Calculate base damage using attacker/defender power ratio
    total_power = attacker_power + defender_power
    if total_power < 0.01:  # Avoid division by zero
        return 0.0, 0.0

    attacker_damage_ratio = defender_power / total_power
    defender_damage_ratio = attacker_power / total_power

    base_attacker_damage = attacker_damage_ratio * attacker_power * 0.2
    base_defender_damage = defender_damage_ratio * defender_power * 0.2

    # Apply combat matrix
    base_attacker_damage *= matrix_multiplier
    base_defender_damage *= params.get_combat_matrix_multiplier(defender.unit_type, attacker.unit_type)

    # Apply random variation
    attacker_damage = base_attacker_damage * (0.9 + 0.2 * np.random.random())
    defender_damage = base_defender_damage * (0.9 + 0.2 * np.random.random())

    return attacker_damage, defender_damage

def resolve_standardized_combat(
    attacker: StandardizedMilitaryUnit,
    defender: StandardizedMilitaryUnit,
    params: StandardizedUnitParams,
    terrain_advantage: float = 1.0,
    is_surprise_attack: bool = False,
    is_flanked: bool = False,
    is_entrenchment_advantage: bool = False,
) -> Tuple[StandardizedMilitaryUnit, StandardizedMilitaryUnit]:
    """
    Resolve combat between two standardized units with enhanced combat features.

    This replaces the old resolve_combat and resolve_enhanced_combat functions.
    """
    # Calculate damage
    attacker_damage, defender_damage = calculate_standardized_damage(
        attacker, defender, params, terrain_advantage,
        is_surprise_attack, is_flanked, is_entrenchment_advantage
    )

    # Apply damage with some randomness
    attacker_hp_loss = min(attacker.hit_points * 0.9, attacker_damage)
    defender_hp_loss = min(defender.hit_points * 0.9, defender_damage)

    # Update combat effectiveness (degrades with combat)
    effectiveness_decay = params.combat_effectiveness_decay
    attacker_effectiveness = max(0.1, attacker.combat_effectiveness - effectiveness_decay)
    defender_effectiveness = max(0.1, defender.combat_effectiveness - effectiveness_decay)

    # Update morale based on combat outcome
    attacker_morale_delta = -0.05 + (defender_damage - attacker_damage) * 0.001
    defender_morale_delta = -0.05 + (attacker_damage - defender_damage) * 0.001

    # Update units
    updated_attacker = attacker.copy_with(
        hit_points=attacker.hit_points - attacker_hp_loss,
        combat_effectiveness=attacker_effectiveness,
        experience=min(10.0, attacker.experience + 0.1),
        morale=max(0.1, attacker.morale + attacker_morale_delta)
    )

    updated_defender = defender.copy_with(
        hit_points=defender.hit_points - defender_hp_loss,
        combat_effectiveness=defender_effectiveness,
        experience=min(10.0, defender.experience + 0.1),
        morale=max(0.1, defender.morale + defender_morale_delta)
    )

    return updated_attacker, updated_defender

# ─────────────────────────────────────────────────────────────────────────── #
# Standardized Initialization                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def initialize_standardized_military_state(
    n_clusters: int,
    params: StandardizedUnitParams,
    rng: np.random.Generator,
    initial_global_supply: float = 100.0,
    initial_reinforcement_pool: float = 50.0,
    objectives: Optional[List[Dict[str, Any]]] = None,
    factions: Optional[List[Dict[str, Any]]] = None,
) -> StandardizedWorldMilitaryState:
    """
    Initialize a standardized military state with empty clusters and optional objectives.

    This replaces the old initialize_military_state function.
    """
    # Create empty clusters
    clusters = tuple(
        StandardizedClusterMilitaryState(
            cluster_id=i,
            units=(),
            supply_depot=10.0,  # Initial supply per cluster
            is_controlled=False,
            controlling_faction=None,
            reinforcement_timer=0.0,
            faction_presence={},
        )
        for i in range(n_clusters)
    )

    # Create objectives if none provided
    if objectives is None:
        objectives = [
            {
                'objective_id': 1,
                'name': 'Capture Central Cluster',
                'objective_type': 'capture',
                'target_cluster_id': 0,
                'required_units': 3,
                'reward_value': 25.0,
                'faction_id': 1,
            },
            {
                'objective_id': 2,
                'name': 'Hold Supply Route',
                'objective_type': 'hold',
                'target_cluster_id': 1,
                'required_units': 2,
                'reward_value': 15.0,
                'faction_id': 1,
            }
        ]

    # Convert objectives to StandardizedMilitaryObjective objects
    military_objectives = tuple(
        StandardizedMilitaryObjective(
            objective_id=obj['objective_id'],
            name=obj['name'],
            objective_type=obj['objective_type'],
            target_cluster_id=obj['target_cluster_id'],
            required_units=obj['required_units'],
            reward_value=obj['reward_value'],
            faction_id=obj.get('faction_id'),
        )
        for obj in objectives
    )

    # Create factions if provided
    factions_dict = {}
    if factions:
        for faction in factions:
            factions_dict[faction['faction_id']] = {
                'name': faction['name'],
                **{k: v for k, v in faction.items() if k != 'faction_id'}
            }

    return StandardizedWorldMilitaryState(
        clusters=clusters,
        objectives=military_objectives,
        global_supply=initial_global_supply,
        global_reinforcement_pool=initial_reinforcement_pool,
        step=0,
        next_unit_id=1,
        factions=factions_dict,
    )

# ─────────────────────────────────────────────────────────────────────────── #
# Unit Creation Helpers                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def create_standardized_unit(
    unit_id: int,
    unit_type: MilitaryUnitType,
    cluster_id: int,
    params: StandardizedUnitParams,
    faction_id: Optional[int] = None,
    support_companies: Optional[List[SupportCompany]] = None,
    objective_id: Optional[int] = None,
) -> StandardizedMilitaryUnit:
    """
    Create a standardized military unit with proper initialization.
    """
    if support_companies is None:
        support_companies = []

    return StandardizedMilitaryUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        cluster_id=cluster_id,
        hit_points=params.get_max_hp(unit_type),
        combat_effectiveness=1.0,
        supply_level=0.8,
        experience=0.0,
        morale=0.9,
        objective_id=objective_id,
        support_companies=tuple(support_companies),
        terrain_bonus=1.0,
        entrenchment=0.0,
        faction_id=faction_id,
    )

def create_infantry_division(
    unit_id: int,
    cluster_id: int,
    params: StandardizedUnitParams,
    faction_id: Optional[int] = None,
    objective_id: Optional[int] = None,
) -> StandardizedMilitaryUnit:
    """
    Create a standardized infantry division with artillery support.
    """
    support_companies = [
        SupportCompany(SupportCompanyType.ARTILLERY_COMPANY, size=1),
        SupportCompany(SupportCompanyType.ENGINEER_COMPANY, size=1),
    ]

    return create_standardized_unit(
        unit_id=unit_id,
        unit_type=MilitaryUnitType.INFANTRY,
        cluster_id=cluster_id,
        params=params,
        faction_id=faction_id,
        support_companies=support_companies,
        objective_id=objective_id,
    )

def create_armored_division(
    unit_id: int,
    cluster_id: int,
    params: StandardizedUnitParams,
    faction_id: Optional[int] = None,
    objective_id: Optional[int] = None,
) -> StandardizedMilitaryUnit:
    """
    Create a standardized armored division with support companies.
    """
    support_companies = [
        SupportCompany(SupportCompanyType.MAINTENANCE_COMPANY, size=1),
        SupportCompany(SupportCompanyType.RECON_COMPANY, size=1),
    ]

    return create_standardized_unit(
        unit_id=unit_id,
        unit_type=MilitaryUnitType.MEDIUM_TANK,
        cluster_id=cluster_id,
        params=params,
        faction_id=faction_id,
        support_companies=support_companies,
        objective_id=objective_id,
    )