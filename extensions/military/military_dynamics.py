"""
military_dynamics.py — Core dynamics for the military extension.

This module implements:
  1. Unit movement between clusters based on topology
  2. Combat resolution between opposing units
  3. Supply and logistics management
  4. Objective progress tracking
  5. Victory condition checking
  6. Military reward calculation

The dynamics integrate with GravitasEngine's existing systems:
  - Cluster military presence is enhanced by unit combat power
  - Units consume cluster resources (supply)
  - Unit movement affects cluster stability
  - Objectives contribute to global victory conditions
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray

from .unit_types import MilitaryUnitType
from .military_state import (
    StandardizedUnitParams as MilitaryUnitParams,
    StandardizedMilitaryUnit as MilitaryUnit,
    StandardizedClusterMilitaryState as ClusterMilitaryState,
    StandardizedWorldMilitaryState as WorldMilitaryState,
    StandardizedMilitaryObjective as MilitaryObjective,
    calculate_standardized_damage as calculate_damage,
    resolve_standardized_combat as resolve_combat,
    initialize_standardized_military_state as initialize_military_state
)
# build_adjacency_matrix removed (unused and caused slow imports)

# ─────────────────────────────────────────────────────────────────────────── #
# Movement and Pathfinding                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_movement_cost(
    unit_type: MilitaryUnitType,
    params: MilitaryUnitParams,
    terrain_type: str = "road"
) -> float:
    """
    Compute movement cost for a unit type on given terrain.

    Args:
        unit_type: Type of military unit
        params: Military unit parameters
        terrain_type: Type of terrain ("road", "rough", "mountain")

    Returns:
        Movement cost multiplier
    """
    base_speed = params.get_speed(unit_type)

    terrain_costs = {
        "road": params.movement_cost_road,
        "rough": params.movement_cost_rough,
        "mountain": params.movement_cost_mountain,
    }

    terrain_cost = terrain_costs.get(terrain_type, params.movement_cost_rough)
    return terrain_cost / base_speed

def can_move_between_clusters(
    from_cluster_id: int,
    to_cluster_id: int,
    adjacency_matrix: NDArray[np.bool_],
) -> bool:
    """
    Check if units can move directly between two clusters.

    Uses the topology adjacency matrix from GravitasEngine.
    """
    return adjacency_matrix[from_cluster_id, to_cluster_id]

def plan_unit_movement(
    unit: MilitaryUnit,
    target_cluster_id: int,
    world_state: WorldMilitaryState,
    adjacency_matrix: NDArray[np.bool_],
    params: MilitaryUnitParams,
) -> Optional[List[int]]:
    """
    Plan a movement path for a unit to reach target cluster.

    Uses BFS to find shortest path through cluster topology.

    Args:
        unit: Military unit to move
        target_cluster_id: Destination cluster
        world_state: Current military world state
        adjacency_matrix: Cluster adjacency from topology
        params: Military unit parameters

    Returns:
        List of cluster IDs representing path, or None if no path exists
    """
    if unit.cluster_id == target_cluster_id:
        return [unit.cluster_id]

    # BFS pathfinding
    from collections import deque
    visited = set()
    queue = deque()
    queue.append((unit.cluster_id, [unit.cluster_id]))
    visited.add(unit.cluster_id)

    while queue:
        current_cluster, path = queue.popleft()

        for neighbor in range(adjacency_matrix.shape[1]):
            if (adjacency_matrix[current_cluster, neighbor] and
                neighbor not in visited):
                visited.add(neighbor)
                new_path = path + [neighbor]

                if neighbor == target_cluster_id:
                    return new_path

                queue.append((neighbor, new_path))

    return None  # No path found

# ─────────────────────────────────────────────────────────────────────────── #
# Combat Resolution                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def resolve_combat(
    attacker: MilitaryUnit,
    defender: MilitaryUnit,
    params: MilitaryUnitParams,
    terrain_advantage: float = 1.0,
) -> Tuple[MilitaryUnit, MilitaryUnit]:
    """
    Resolve combat between two units.

    Uses a simplified combat model based on unit types and current states.

    Args:
        attacker: Attacking unit
        defender: Defending unit
        params: Military unit parameters
        terrain_advantage: Multiplier for defender (1.0 = no advantage)

    Returns:
        Tuple of (updated_attacker, updated_defender) after combat
    """
    # Compute combat power with terrain effects
    attacker_power = attacker.combat_power * (1.0 + np.random.normal(0, 0.1))
    defender_power = defender.combat_power * terrain_advantage * (1.0 + np.random.normal(0, 0.1))

    # Damage calculation
    total_damage = attacker_power + defender_power
    attacker_damage = defender_power / total_damage * attacker.combat_power * 0.3
    defender_damage = attacker_power / total_damage * defender.combat_power * 0.3

    # Apply damage with some randomness
    attacker_hp_loss = min(attacker.hit_points * 0.9, attacker_damage * (0.9 + 0.2 * np.random.random()))
    defender_hp_loss = min(defender.hit_points * 0.9, defender_damage * (0.9 + 0.2 * np.random.random()))

    # Update combat effectiveness (degrades with combat)
    attacker_effectiveness = max(0.1, attacker.combat_effectiveness - params.combat_effectiveness_decay)
    defender_effectiveness = max(0.1, defender.combat_effectiveness - params.combat_effectiveness_decay)

    # Update units
    updated_attacker = attacker.copy_with(
        hit_points=attacker.hit_points - attacker_hp_loss,
        combat_effectiveness=attacker_effectiveness,
        experience=min(10.0, attacker.experience + 0.1),
        morale=max(0.1, attacker.morale - 0.05)
    )

    updated_defender = defender.copy_with(
        hit_points=defender.hit_points - defender_hp_loss,
        combat_effectiveness=defender_effectiveness,
        experience=min(10.0, defender.experience + 0.1),
        morale=max(0.1, defender.morale - 0.05)
    )

    return updated_attacker, updated_defender

def resolve_cluster_combat(
    cluster_state: ClusterMilitaryState,
    params: MilitaryUnitParams,
) -> ClusterMilitaryState:
    """
    Resolve combat between all opposing units in a cluster.

    Groups units by faction/objective and resolves pairwise combat.

    Args:
        cluster_state: Current cluster military state
        params: Military unit parameters

    Returns:
        Updated cluster state after combat resolution
    """
    # For now, simple implementation: find two strongest opposing units and fight
    # In a full implementation, this would handle multiple factions and complex combat

    alive_units = [u for u in cluster_state.units if u.is_alive]
    if len(alive_units) < 2:
        return cluster_state  # No combat possible

    # Sort by combat power and take top two
    sorted_units = sorted(alive_units, key=lambda u: u.combat_power, reverse=True)
    attacker, defender = sorted_units[0], sorted_units[1]

    # Resolve combat
    updated_attacker, updated_defender = resolve_combat(attacker, defender, params)

    # Update units in cluster
    new_cluster = cluster_state
    new_cluster = new_cluster.update_unit(updated_attacker)
    new_cluster = new_cluster.update_unit(updated_defender)

    return new_cluster

# ─────────────────────────────────────────────────────────────────────────── #
# Supply and Logistics                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def update_supply(
    cluster_state: ClusterMilitaryState,
    global_supply: float,
    params: MilitaryUnitParams,
    cluster_resource_level: float,
) -> Tuple[ClusterMilitaryState, float]:
    """
    Update supply levels for units in a cluster.

    Units consume supplies based on their type and current state.
    Supply depots are replenished from global supply.

    Args:
        cluster_state: Current cluster military state
        global_supply: Current global supply level
        params: Military unit parameters
        cluster_resource_level: Resource level of the cluster (0-1)

    Returns:
        Tuple of (updated_cluster_state, supply_consumed)
    """
    # Calculate total supply demand
    supply_demand = cluster_state.supply_demand(params)

    # Supply available from depot + global pool
    available_supply = cluster_state.supply_depot + min(
        global_supply * 0.1,  # Limit global contribution
        supply_demand * 0.3   # Max 30% from global
    )

    # Resource-based supply bonus
    resource_bonus = cluster_resource_level * 2.0

    # Total supply available
    total_available = available_supply + resource_bonus

    # Distribute supplies to units
    new_units = []
    supply_used = 0.0

    for unit in cluster_state.units:
        if not unit.is_alive:
            new_units.append(unit)
            continue

        unit_supply_cost = params.get_supply_cost(unit.unit_type) * params.supply_consumption_rate
        supply_allocated = min(unit_supply_cost, total_available * (unit_supply_cost / max(supply_demand, 1e-6)))

        supply_used += supply_allocated
        new_supply_level = min(1.0, unit.supply_level + supply_allocated * 0.1)

        # HP regeneration based on supply
        hp_regen = 0.0
        if new_supply_level > 0.5:  # Only regen with good supply
            hp_regen = params.reinforcement_rate * unit.hit_points * new_supply_level

        new_hp = min(
            params.get_max_hp(unit.unit_type),
            unit.hit_points + hp_regen - params.attrition_rate * unit.hit_points
        )

        new_units.append(unit.copy_with(
            supply_level=new_supply_level,
            hit_points=new_hp
        ))

    # Update cluster supply depot
    depot_consumed = min(cluster_state.supply_depot, supply_demand * 0.5)
    global_consumed = supply_used - depot_consumed

    new_supply_depot = max(0.0, cluster_state.supply_depot - depot_consumed)
    new_supply_depot += cluster_resource_level * 1.0  # Resource regeneration

    return (
        cluster_state.copy_with(
            units=tuple(new_units),
            supply_depot=new_supply_depot
        ),
        global_consumed
    )

# ─────────────────────────────────────────────────────────────────────────── #
# Objective Management                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def update_objective_progress(
    world_state: WorldMilitaryState,
    params: MilitaryUnitParams,
) -> WorldMilitaryState:
    """
    Update progress on all military objectives.

    Checks each objective and updates progress based on current unit deployment.

    Args:
        world_state: Current military world state
        params: Military unit parameters

    Returns:
        Updated world state with objective progress
    """
    new_objectives = []

    for objective in world_state.objectives:
        if objective.is_completed:
            new_objectives.append(objective)
            continue

        target_cluster = world_state.get_cluster(objective.target_cluster_id)
        if target_cluster is None:
            new_objectives.append(objective)
            continue

        # Calculate progress based on objective type
        if objective.objective_type == "capture":
            # Progress based on combat power in target cluster
            our_units = sum(1 for u in target_cluster.units
                           if u.is_alive and u.objective_id == objective.objective_id)
            progress = min(1.0, our_units / objective.required_units)
            progress_delta = progress - objective.completion_progress

        elif objective.objective_type == "hold":
            # Progress based on maintaining presence over time
            our_units = sum(1 for u in target_cluster.units
                           if u.is_alive and u.objective_id == objective.objective_id)
            if our_units >= objective.required_units:
                progress_delta = 1.0 / params.objective_hold_duration
            else:
                progress_delta = -0.1  # Lose progress if not holding

        elif objective.objective_type == "destroy":
            # Progress based on eliminating enemy units
            enemy_units = sum(1 for u in target_cluster.units
                             if u.is_alive and (u.objective_id != objective.objective_id or u.objective_id is None))
            progress = 1.0 - min(1.0, enemy_units / (objective.required_units * 2))
            progress_delta = progress - objective.completion_progress

        else:  # Default: presence-based progress
            our_units = sum(1 for u in target_cluster.units
                           if u.is_alive and u.objective_id == objective.objective_id)
            progress_delta = min(0.1, our_units * 0.05)

        # Update objective
        updated_objective = objective.update_progress(progress_delta, world_state.step)
        new_objectives.append(updated_objective)

    return world_state.copy_with(objectives=tuple(new_objectives))

def check_victory_conditions(
    world_state: WorldMilitaryState,
    victory_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Check if victory conditions are met.

    Args:
        world_state: Current military world state
        victory_threshold: Fraction of objectives needed to win (0-1)

    Returns:
        Dictionary with victory status and details
    """
    completed_objectives = sum(1 for obj in world_state.objectives if obj.is_completed)
    total_objectives = len(world_state.objectives)

    if total_objectives == 0:
        return {
            'victory_achieved': False,
            'completion_percentage': 0.0,
            'completed_objectives': 0,
            'total_objectives': 0,
            'message': "No objectives defined"
        }

    completion_percentage = completed_objectives / total_objectives
    victory_achieved = completion_percentage >= victory_threshold

    return {
        'victory_achieved': victory_achieved,
        'completion_percentage': completion_percentage,
        'completed_objectives': completed_objectives,
        'total_objectives': total_objectives,
        'message': f"Victory {'achieved' if victory_achieved else 'not achieved'}: "
                  f"{completed_objectives}/{total_objectives} objectives completed "
                  f"({completion_percentage:.1%})"
    }

# ─────────────────────────────────────────────────────────────────────────── #
# Military Actions                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_military_action(
    world_state: WorldMilitaryState,
    action_type: str,
    target_cluster_id: int,
    unit_type: MilitaryUnitType,
    intensity: float,
    params: MilitaryUnitParams,
    rng: np.random.Generator,
) -> WorldMilitaryState:
    """
    Apply a military action to the world state.

    Supported actions: "deploy", "move", "attack", "reinforce", "retreat"

    Args:
        world_state: Current military world state
        action_type: Type of military action
        target_cluster_id: Target cluster for action
        unit_type: Type of unit to create (for deploy)
        intensity: Action intensity (0-1)
        params: Military unit parameters
        rng: Random number generator

    Returns:
        Updated world state after action
    """
    new_world = world_state
    target_cluster = world_state.get_cluster(target_cluster_id)

    if target_cluster is None:
        return world_state

    if action_type == "deploy":
        # Deploy new units to target cluster
        if world_state.global_reinforcement_pool >= 1.0 * intensity:
            # Create new unit
            new_unit = MilitaryUnit(
                unit_id=world_state.next_unit_id,
                unit_type=unit_type,
                cluster_id=target_cluster_id,
                hit_points=params.get_max_hp(unit_type),
                combat_effectiveness=1.0,
                supply_level=0.8,
                experience=0.0,
                morale=0.9,
                objective_id=None  # Will be assigned later
            )

            # Add to cluster and update world state
            new_cluster = target_cluster.add_unit(new_unit)
            new_clusters = tuple(
                new_cluster if c.cluster_id == target_cluster_id else c
                for c in world_state.clusters
            )

            new_world = world_state.copy_with(
                clusters=new_clusters,
                global_reinforcement_pool=world_state.global_reinforcement_pool - 1.0 * intensity,
                next_unit_id=world_state.next_unit_id + 1
            )

    elif action_type == "move":
        # Move units toward target cluster (simplified for now)
        # In full implementation, this would use pathfinding
        for cluster in world_state.clusters:
            if cluster.cluster_id == target_cluster_id:
                continue

            # Find units that could move toward target
            for unit in cluster.units:
                if unit.is_alive and rng.random() < intensity * 0.3:
                    # Simple movement: 30% chance to move toward target per step
                    new_unit = unit.copy_with(cluster_id=target_cluster_id)
                    new_cluster = cluster.update_unit(new_unit)
                    new_target = target_cluster.add_unit(new_unit)

                    new_clusters = tuple(
                        new_target if c.cluster_id == target_cluster_id else
                        new_cluster if c.cluster_id == cluster.cluster_id else
                        c
                        for c in world_state.clusters
                    )

                    new_world = world_state.copy_with(clusters=new_clusters)
                    break

    elif action_type == "attack":
        # Boost combat effectiveness of units in target cluster
        new_units = []
        for unit in target_cluster.units:
            if unit.is_alive:
                new_morale = min(1.0, unit.morale + 0.1 * intensity)
                new_effectiveness = min(1.0, unit.combat_effectiveness + 0.05 * intensity)
                new_unit = unit.copy_with(
                    morale=new_morale,
                    combat_effectiveness=new_effectiveness
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        new_cluster = target_cluster.copy_with(units=tuple(new_units))
        new_clusters = tuple(
            new_cluster if c.cluster_id == target_cluster_id else c
            for c in world_state.clusters
        )
        new_world = world_state.copy_with(clusters=new_clusters)

    elif action_type == "reinforce":
        # Reinforce units in target cluster
        new_units = []
        supply_used = 0.0

        for unit in target_cluster.units:
            if unit.is_alive:
                # HP reinforcement
                hp_gain = params.reinforcement_rate * params.get_max_hp(unit.unit_type) * intensity
                new_hp = min(params.get_max_hp(unit.unit_type), unit.hit_points + hp_gain)

                # Supply replenishment
                supply_gain = 0.1 * intensity
                new_supply = min(1.0, unit.supply_level + supply_gain)
                supply_used += params.get_supply_cost(unit.unit_type) * supply_gain

                new_unit = unit.copy_with(
                    hit_points=new_hp,
                    supply_level=new_supply,
                    morale=min(1.0, unit.morale + 0.05 * intensity)
                )
                new_units.append(new_unit)
            else:
                new_units.append(unit)

        # Update supply depot
        depot_consumed = min(target_cluster.supply_depot, supply_used)
        new_supply_depot = target_cluster.supply_depot - depot_consumed

        new_cluster = target_cluster.copy_with(
            units=tuple(new_units),
            supply_depot=new_supply_depot
        )
        new_clusters = tuple(
            new_cluster if c.cluster_id == target_cluster_id else c
            for c in world_state.clusters
        )
        new_world = world_state.copy_with(clusters=new_clusters)

    elif action_type == "retreat":
        # Retreat units from target cluster (move to adjacent clusters)
        adjacent_clusters = [c for c in world_state.clusters
                            if c.cluster_id != target_cluster_id]

        if adjacent_clusters:
            retreat_target = rng.choice(adjacent_clusters)

            new_units = []
            for unit in target_cluster.units:
                if unit.is_alive and rng.random() < intensity * 0.5:
                    # 50% chance to retreat per unit
                    new_unit = unit.copy_with(cluster_id=retreat_target.cluster_id)
                    new_units.append(new_unit)
                else:
                    new_units.append(unit)

            # Update both clusters
            new_target_cluster = target_cluster.copy_with(units=tuple(new_units))
            new_retreat_cluster = retreat_target.copy_with(
                units=retreat_target.units + tuple(
                    u for u in new_units if u.cluster_id == retreat_target.cluster_id
                )
            )

            new_clusters = tuple(
                new_target_cluster if c.cluster_id == target_cluster_id else
                new_retreat_cluster if c.cluster_id == retreat_target.cluster_id else
                c
                for c in world_state.clusters
            )

            new_world = world_state.copy_with(clusters=new_clusters)

    return new_world

# ─────────────────────────────────────────────────────────────────────────── #
# Reward Calculation                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_military_reward(
    world_state: WorldMilitaryState,
    prev_state: WorldMilitaryState,
    action_type: Optional[str] = None,
) -> float:
    """
    Compute military reward based on state changes and objective progress.

    Args:
        world_state: Current military world state
        prev_state: Previous military world state
        action_type: Type of action taken (if any)

    Returns:
        Total military reward for this step
    """
    reward = 0.0

    # 1. Objective completion rewards
    for objective in world_state.objectives:
        if objective.is_completed and not any(
            prev_obj.objective_id == objective.objective_id and prev_obj.is_completed
            for prev_obj in prev_state.objectives
        ):
            reward += objective.reward_value
            print(f"🎯 Objective completed: {objective.name} (+{objective.reward_value} reward)")

    # 2. Objective progress rewards
    for obj, prev_obj in zip(world_state.objectives, prev_state.objectives):
        if obj.objective_id == prev_obj.objective_id:
            progress_reward = (obj.completion_progress - prev_obj.completion_progress) * 5.0
            reward += progress_reward

    # 3. Unit survival rewards
    prev_units = sum(c.unit_count for c in prev_state.clusters)
    current_units = sum(c.unit_count for c in world_state.clusters)
    survival_reward = (current_units - prev_units) * 2.0
    reward += survival_reward

    # 4. Combat power rewards
    prev_power = sum(c.total_combat_power for c in prev_state.clusters)
    current_power = sum(c.total_combat_power for c in world_state.clusters)
    power_reward = (current_power - prev_power) * 0.1
    reward += power_reward

    # 5. Action-specific rewards
    if action_type:
        if action_type == "deploy":
            reward += 1.0 * world_state.total_unit_count * 0.5
        elif action_type == "attack":
            reward += 0.5 * world_state.total_combat_power * 0.1
        elif action_type == "reinforce":
            reward += 0.3 * world_state.total_unit_count

    # 6. Penalties
    # Supply shortage penalty (can't compute without params, so skip in this context)
    # total_demand = sum(c.supply_demand(params) for c in world_state.clusters)
    # if world_state.global_supply < total_demand * 0.5:
    #     supply_penalty = -0.5 * (total_demand * 0.5 - world_state.global_supply)
    #     reward += supply_penalty

    return float(reward)

# ─────────────────────────────────────────────────────────────────────────── #
# Main Military Step Function                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def step_military_units(
    military_state: WorldMilitaryState,
    world_state: object,  # GravitasWorld
    adjacency_matrix: NDArray[np.bool_],
    params: MilitaryUnitParams,
    rng: np.random.Generator,
    military_action: Optional[Dict[str, Any]] = None,
) -> Tuple[WorldMilitaryState, Dict[str, Any]]:
    """
    Main function to advance military state by one step.

    This function orchestrates all military dynamics:
      1. Apply military actions
      2. Resolve unit movement
      3. Handle combat resolution
      4. Update supply and logistics
      5. Track objective progress
      6. Compute rewards

    Args:
        military_state: Current military world state
        world_state: GravitasWorld for cluster resource levels
        adjacency_matrix: Cluster adjacency from topology
        params: Military unit parameters
        rng: Random number generator
        military_action: Optional military action to apply

    Returns:
        Tuple of (updated_military_state, metrics_dict)
    """
    # Get cluster resource levels from GravitasWorld
    cluster_resources = np.array([
        c.resource for c in world_state.clusters
    ])

    # 1. Apply military action if provided
    new_state = military_state
    action_type = None

    if military_action:
        action_type = military_action.get('action_type', 'deploy')
        new_state = apply_military_action(
            military_state,
            action_type,
            military_action.get('target_cluster', 0),
            military_action.get('unit_type', MilitaryUnitType.INFANTRY),
            military_action.get('intensity', 0.5),
            params,
            rng
        )

    # 2. Resolve combat in each cluster
    new_clusters = []
    for cluster in new_state.clusters:
        updated_cluster = resolve_cluster_combat(cluster, params)
        new_clusters.append(updated_cluster)

    new_state = new_state.copy_with(clusters=tuple(new_clusters))

    # 3. Update supply and logistics
    total_supply_consumed = 0.0
    updated_clusters = []

    for cluster in new_state.clusters:
        cluster_idx = cluster.cluster_id
        resource_level = cluster_resources[cluster_idx]

        updated_cluster, supply_consumed = update_supply(
            cluster,
            new_state.global_supply,
            params,
            resource_level
        )

        updated_clusters.append(updated_cluster)
        total_supply_consumed += supply_consumed

    new_state = new_state.copy_with(
        clusters=tuple(updated_clusters),
        global_supply=max(0.0, new_state.global_supply - total_supply_consumed)
    )

    # 4. Update objective progress
    new_state = update_objective_progress(new_state, params)

    # 5. Check victory conditions
    victory_status = check_victory_conditions(new_state)

    # 6. Compute military reward
    military_reward = compute_military_reward(
        new_state,
        military_state,
        action_type
    )

    # 7. Advance step
    new_state = new_state.advance_step()

    # 8. Prepare metrics
    metrics = {
        'total_units': new_state.total_unit_count,
        'total_combat_power': new_state.total_combat_power,
        'global_supply': new_state.global_supply,
        'global_reinforcement_pool': new_state.global_reinforcement_pool,
        'objectives_completed': sum(1 for obj in new_state.objectives if obj.is_completed),
        'objectives_total': len(new_state.objectives),
        'military_reward': military_reward,
        **victory_status
    }

    return new_state, metrics