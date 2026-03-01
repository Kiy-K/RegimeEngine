"""
military_extensions.py — High-fidelity extensions to the military layer.

This module adds the systems that were referenced but unimplemented, and
introduces several new tactical subsystems:

  1. FactionCombatSystem    — proper multi-faction combat with faction IDs,
                               type-advantage matrix, and casualty tracking
  2. MoraleSystem           — routing triggers, rally mechanics, morale cascade
                               propagation across adjacent clusters
  3. FogOfWarSystem         — dynamic per-cluster fog updated by recon actions,
                               EW jamming, and temporal decay
  4. ClusterControlSystem   — faction control transitions driven by combat
                               outcomes, with contested-zone logic
  5. EWExecutionEngine      — concrete execution of all EWCapability effects:
                               jamming → comms degradation, deception → false
                               intel injection, intercept → intel gain,
                               disruption → unit effectiveness reduction,
                               spoofing → friendly-fire risk
  6. SupplyInterdictionSystem — supply line attacks that degrade depots and
                                 cut throughput for targeted clusters

All systems are stateless functions + frozen dataclasses; state lives in the
existing AdvancedWorldMilitaryState / AdvancedClusterMilitaryState structures,
extended via new fields exposed through helper copy_with calls.

Integration contract
--------------------
These systems are invoked from AdvancedMilitaryWrapper._step_advanced_military_units
in the following order (replacing the simple stubs):

    world = FactionCombatSystem.resolve_all_clusters(world, params, rng)
    world = MoraleSystem.step(world, params, adjacency_matrix)
    world = ClusterControlSystem.step(world)
    world = FogOfWarSystem.step(world, active_ew_units)
    world = EWExecutionEngine.step(world, params, rng)
    # supply chain already handled in _update_advanced_supply
    world = SupplyInterdictionSystem.step(world, adjacency_matrix, rng)

Each system is also callable independently for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from numpy.typing import NDArray

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
from .unit_types import MilitaryUnitType
from .advanced_tactics import (
    AdvancedClusterMilitaryState, AdvancedWorldMilitaryState,
    EWCapability, ElectronicWarfareState, IntelligenceReport,
    IntelligenceType, SupplyChain, SupplyType,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Type-Advantage Matrix                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

# ADVANTAGE[attacker_type][defender_type] = damage multiplier
# Values > 1.0 mean the attacker has an edge vs. that defender type.
_UNIT_ADVANTAGE: Dict[MilitaryUnitType, Dict[MilitaryUnitType, float]] = {
    MilitaryUnitType.INFANTRY: {
        MilitaryUnitType.INFANTRY:      1.00,
        MilitaryUnitType.ARMOR:         0.60,  # infantry weak vs armour
        MilitaryUnitType.ARTILLERY:     1.30,  # infantry good vs arty in CQB
        MilitaryUnitType.AIR:           0.40,  # infantry can't fight air
        MilitaryUnitType.SPECIAL_FORCES: 0.85,
        MilitaryUnitType.LOGISTICS:     1.50,
    },
    MilitaryUnitType.ARMOR: {
        MilitaryUnitType.INFANTRY:      1.40,
        MilitaryUnitType.ARMOR:         1.00,
        MilitaryUnitType.ARTILLERY:     1.20,
        MilitaryUnitType.AIR:           0.50,
        MilitaryUnitType.SPECIAL_FORCES: 0.90,
        MilitaryUnitType.LOGISTICS:     1.60,
    },
    MilitaryUnitType.ARTILLERY: {
        MilitaryUnitType.INFANTRY:      1.50,
        MilitaryUnitType.ARMOR:         1.30,
        MilitaryUnitType.ARTILLERY:     1.00,
        MilitaryUnitType.AIR:           0.30,
        MilitaryUnitType.SPECIAL_FORCES: 0.80,
        MilitaryUnitType.LOGISTICS:     1.40,
    },
    MilitaryUnitType.AIR: {
        MilitaryUnitType.INFANTRY:      1.60,
        MilitaryUnitType.ARMOR:         1.50,
        MilitaryUnitType.ARTILLERY:     1.80,
        MilitaryUnitType.AIR:           1.00,
        MilitaryUnitType.SPECIAL_FORCES: 1.20,
        MilitaryUnitType.LOGISTICS:     1.70,
    },
    MilitaryUnitType.SPECIAL_FORCES: {
        MilitaryUnitType.INFANTRY:      1.30,
        MilitaryUnitType.ARMOR:         0.80,
        MilitaryUnitType.ARTILLERY:     1.50,
        MilitaryUnitType.AIR:           0.60,
        MilitaryUnitType.SPECIAL_FORCES: 1.00,
        MilitaryUnitType.LOGISTICS:     1.80,
    },
    MilitaryUnitType.LOGISTICS: {
        MilitaryUnitType.INFANTRY:      0.20,
        MilitaryUnitType.ARMOR:         0.10,
        MilitaryUnitType.ARTILLERY:     0.15,
        MilitaryUnitType.AIR:           0.05,
        MilitaryUnitType.SPECIAL_FORCES: 0.20,
        MilitaryUnitType.LOGISTICS:     0.30,
    },
}


def type_advantage(attacker: MilitaryUnitType, defender: MilitaryUnitType) -> float:
    """Return damage multiplier for attacker type vs defender type."""
    return _UNIT_ADVANTAGE.get(attacker, {}).get(defender, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# CombatResult                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class CombatResult:
    """
    Outcome of a multi-faction engagement in one cluster.

    Attributes:
        winning_faction:    faction_id of winner, or None if contested
        casualties:         dict mapping faction_id -> total HP lost
        survivors:          dict mapping faction_id -> number of alive units
        combat_occurred:    whether any fighting happened
    """
    winning_faction: Optional[int]
    casualties: Dict[int, float]
    survivors: Dict[int, int]
    combat_occurred: bool


# ─────────────────────────────────────────────────────────────────────────── #
# 1. FactionCombatSystem                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class FactionCombatSystem:
    """
    Multi-faction combat resolution using faction IDs stored in
    MilitaryUnit.objective_id as a faction proxy (until a dedicated
    faction_id field is added; zero / None = neutral).

    Algorithm per cluster:
      1. Group alive units by faction.
      2. If only one faction present → no combat.
      3. For each opposing pair of factions, run a lance exchange:
         - Sum each side's effective combat power (HP × effectiveness × type_advantage).
         - Distribute proportional damage across each unit on the losing side.
         - Experience and morale update for all participants.
      4. Track cumulative casualties and mark dead units.
    """

    @staticmethod
    def resolve_cluster(
        cluster: AdvancedClusterMilitaryState,
        params: MilitaryUnitParams,
        rng: np.random.Generator,
    ) -> Tuple[AdvancedClusterMilitaryState, CombatResult]:
        """Resolve one step of multi-faction combat in a cluster."""
        alive = [u for u in cluster.units if u.is_alive]
        if len(alive) < 2:
            return cluster, CombatResult(
                winning_faction=None,
                casualties={},
                survivors={},
                combat_occurred=False,
            )

        # Group by faction (using objective_id as faction proxy)
        factions: Dict[int, List[MilitaryUnit]] = {}
        for unit in alive:
            fid = unit.objective_id if unit.objective_id is not None else 0
            factions.setdefault(fid, []).append(unit)

        if len(factions) < 2:
            # All same faction — no combat
            return cluster, CombatResult(
                winning_faction=list(factions.keys())[0],
                casualties={},
                survivors={fid: len(units) for fid, units in factions.items()},
                combat_occurred=False,
            )

        faction_ids = list(factions.keys())
        updated_units: Dict[int, MilitaryUnit] = {u.unit_id: u for u in cluster.units}
        casualties: Dict[int, float] = {fid: 0.0 for fid in faction_ids}

        # Each pair of opposing factions engages
        for i, fid_a in enumerate(faction_ids):
            for fid_b in faction_ids[i + 1:]:
                side_a = [updated_units[u.unit_id] for u in factions[fid_a] if updated_units[u.unit_id].is_alive]
                side_b = [updated_units[u.unit_id] for u in factions[fid_b] if updated_units[u.unit_id].is_alive]
                if not side_a or not side_b:
                    continue

                # Compute effective combat powers with type advantages
                def effective_power(units: List[MilitaryUnit], enemy_units: List[MilitaryUnit]) -> float:
                    """Weighted power considering type advantages vs average enemy composition."""
                    if not enemy_units:
                        return 0.0
                    total = 0.0
                    for u in units:
                        avg_adv = np.mean([type_advantage(u.unit_type, e.unit_type) for e in enemy_units])
                        total += u.combat_power * avg_adv
                    return total

                power_a = effective_power(side_a, side_b) * (1.0 + rng.normal(0, 0.08))
                power_b = effective_power(side_b, side_a) * (1.0 + rng.normal(0, 0.08))
                terrain_a = cluster.terrain_advantage  # defender bonus for side with more presence
                power_b *= terrain_a  # slight defender advantage

                total_power = max(power_a + power_b, 1e-9)
                # Damage fraction distributed proportionally to opposing power
                damage_to_a = (power_b / total_power) * 0.25  # fraction of total HP pool
                damage_to_b = (power_a / total_power) * 0.25

                # Distribute damage across each side's units proportionally to their HP
                def apply_damage(units: List[MilitaryUnit], damage_fraction: float, faction_id: int) -> None:
                    total_hp = sum(u.hit_points for u in units)
                    if total_hp < 1e-6:
                        return
                    for u in units:
                        share = (u.hit_points / total_hp) * damage_fraction * total_hp
                        share *= (0.85 + 0.30 * rng.random())
                        new_hp = max(0.0, u.hit_points - share)
                        new_eff = max(0.1, u.combat_effectiveness - params.combat_effectiveness_decay)
                        new_exp = min(10.0, u.experience + 0.15)
                        new_morale = max(0.0, u.morale - 0.04)
                        casualties[faction_id] = casualties.get(faction_id, 0.0) + share
                        updated_units[u.unit_id] = u.copy_with(
                            hit_points=new_hp,
                            combat_effectiveness=new_eff,
                            experience=new_exp,
                            morale=new_morale,
                        )

                apply_damage(side_a, damage_to_a, fid_a)
                apply_damage(side_b, damage_to_b, fid_b)

        # Rebuild units tuple preserving dead units
        new_units = tuple(updated_units[u.unit_id] if u.unit_id in updated_units else u
                          for u in cluster.units)

        # Determine winner
        survivors: Dict[int, int] = {}
        for fid in faction_ids:
            survivors[fid] = sum(1 for u in new_units if u.is_alive and (u.objective_id or 0) == fid)

        winning_faction: Optional[int] = None
        alive_factions = {fid for fid, cnt in survivors.items() if cnt > 0}
        if len(alive_factions) == 1:
            winning_faction = list(alive_factions)[0]

        result = CombatResult(
            winning_faction=winning_faction,
            casualties=casualties,
            survivors=survivors,
            combat_occurred=True,
        )

        new_cluster = cluster.copy_with(units=new_units)
        return new_cluster, result

    @classmethod
    def resolve_all_clusters(
        cls,
        world: AdvancedWorldMilitaryState,
        params: MilitaryUnitParams,
        rng: np.random.Generator,
    ) -> Tuple[AdvancedWorldMilitaryState, List[CombatResult]]:
        """Resolve multi-faction combat across all clusters. Returns (new_world, results)."""
        new_clusters = []
        results = []
        for cluster in world.clusters:
            new_cluster, result = cls.resolve_cluster(cluster, params, rng)
            new_clusters.append(new_cluster)
            results.append(result)
        return world.copy_with(clusters=tuple(new_clusters)), results


# ─────────────────────────────────────────────────────────────────────────── #
# 2. MoraleSystem                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

# Morale thresholds
ROUTE_THRESHOLD = 0.15       # unit routes if morale drops below this
RALLY_THRESHOLD = 0.45       # routed unit can rally if morale recovers above this
MORALE_CASCADE_FACTOR = 0.3  # fraction of routed-unit morale penalty spread to neighbors
MORALE_REGEN_RATE = 0.03     # passive morale recovery per step when not in combat
ROUT_HP_DRAIN = 0.08         # routed units lose HP each step (retreat attrition)
MORALE_CASCADE_MAX = 0.08    # hard cap on cascade morale loss per step


class MoraleSystem:
    """
    Morale routing, rallying, and inter-cluster cascade.

    A unit with morale < ROUTE_THRESHOLD is marked as routed (morale set to
    near-zero, combat_effectiveness clamped). Routed units in adjacent clusters
    spread a morale penalty to their faction-mates (panic propagation).
    Units that are well-supplied and not in combat regain morale gradually.
    """

    @staticmethod
    def _is_in_combat(cluster: AdvancedClusterMilitaryState) -> bool:
        """Check if a cluster has multi-faction presence (proxy for ongoing combat)."""
        factions = set()
        for u in cluster.units:
            if u.is_alive:
                factions.add(u.objective_id or 0)
        return len(factions) > 1

    @classmethod
    def step(
        cls,
        world: AdvancedWorldMilitaryState,
        params: MilitaryUnitParams,
        adjacency_matrix: NDArray[np.bool_],
    ) -> AdvancedWorldMilitaryState:
        """Apply one step of morale dynamics to all clusters."""
        # First pass: identify routed units and collect per-cluster state
        routed_by_cluster: Dict[int, List[MilitaryUnit]] = {}
        new_clusters = list(world.clusters)

        for idx, cluster in enumerate(world.clusters):
            in_combat = cls._is_in_combat(cluster)
            new_units = []
            routed_units = []

            for unit in cluster.units:
                if not unit.is_alive:
                    new_units.append(unit)
                    continue

                morale = unit.morale
                supply_ok = unit.supply_level > 0.4

                # Routing trigger
                if morale < ROUTE_THRESHOLD:
                    # Unit routes: clamped effectiveness, HP attrition
                    new_hp = max(0.0, unit.hit_points - ROUT_HP_DRAIN * params.get_max_hp(unit.unit_type))
                    updated = unit.copy_with(
                        morale=max(0.0, morale),
                        combat_effectiveness=max(0.0, unit.combat_effectiveness * 0.3),
                        hit_points=new_hp,
                    )
                    routed_units.append(updated)
                    new_units.append(updated)
                    continue

                # Morale regen (only when not in active combat and well-supplied)
                if not in_combat and supply_ok:
                    morale = min(1.0, morale + MORALE_REGEN_RATE)

                # Rally: restore effectiveness if morale recovered
                effectiveness = unit.combat_effectiveness
                if morale > RALLY_THRESHOLD and effectiveness < 0.5:
                    effectiveness = min(1.0, effectiveness + 0.05)

                new_units.append(unit.copy_with(morale=morale, combat_effectiveness=effectiveness))

            routed_by_cluster[cluster.cluster_id] = routed_units
            new_clusters[idx] = cluster.copy_with(units=tuple(new_units))

        # Second pass: cascade morale penalty to adjacent clusters
        if adjacency_matrix is not None:
            for src_cid, routed in routed_by_cluster.items():
                if not routed:
                    continue
                penalty = sum(1 - u.morale for u in routed) * MORALE_CASCADE_FACTOR / max(1, len(routed))
                penalty = min(MORALE_CASCADE_MAX, penalty * 0.5)  # hard cap
                if penalty < 0.005:
                    continue
                n = adjacency_matrix.shape[0]
                for tgt_cid in range(n):
                    if tgt_cid >= len(new_clusters):
                        break
                    if src_cid < n and adjacency_matrix[src_cid, tgt_cid]:
                        tgt_cluster = new_clusters[tgt_cid]
                        cascaded = []
                        for unit in tgt_cluster.units:
                            if unit.is_alive:
                                cascaded.append(unit.copy_with(
                                    morale=max(0.0, unit.morale - penalty)
                                ))
                            else:
                                cascaded.append(unit)
                        new_clusters[tgt_cid] = tgt_cluster.copy_with(units=tuple(cascaded))

        return world.copy_with(clusters=tuple(new_clusters))


# ─────────────────────────────────────────────────────────────────────────── #
# 3. FogOfWarSystem                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

FOG_DECAY_RATE = 0.04         # fog drifts back toward 0.5 each step (uncertainty grows)
FOG_INTEL_REDUCTION = 0.15    # each fresh intel report reduces fog by this much
FOG_REGEN_RATE = 0.02         # fog increases per step without intel
FOG_EW_JAMMING_BOOST = 0.12   # active jamming raises target cluster's fog


class FogOfWarSystem:
    """
    Dynamic fog-of-war per cluster.

    Fog of war (0 = perfect knowledge, 1 = complete uncertainty).

    Drivers:
      - Intelligence reports reduce fog for the target cluster.
      - Stale reports allow fog to drift back up.
      - EW jamming raises fog in jammed clusters.
      - Fog naturally drifts toward 0.5 (partial uncertainty) each step.
    """

    @classmethod
    def step(
        cls,
        world: AdvancedWorldMilitaryState,
        jamming_active_cluster_ids: Optional[Set[int]] = None,
    ) -> AdvancedWorldMilitaryState:
        """Update fog-of-war for all clusters."""
        if jamming_active_cluster_ids is None:
            jamming_active_cluster_ids = set()

        new_clusters = []
        for cluster in world.clusters:
            fog = cluster.fog_of_war

            # Count fresh (non-stale) intel reports
            fresh_reports = sum(1 for r in cluster.intelligence_reports if not r.is_stale)
            stale_reports = sum(1 for r in cluster.intelligence_reports if r.is_stale)

            # Intel reduces fog
            fog -= fresh_reports * FOG_INTEL_REDUCTION
            # Stale intel allows fog to recover
            fog += stale_reports * FOG_REGEN_RATE
            # Drift toward 0.5
            fog += (0.5 - fog) * FOG_DECAY_RATE
            # EW jamming raises fog
            if cluster.cluster_id in jamming_active_cluster_ids:
                fog += FOG_EW_JAMMING_BOOST

            fog = float(np.clip(fog, 0.0, 1.0))
            new_clusters.append(cluster.copy_with(fog_of_war=fog))

        return world.copy_with(clusters=tuple(new_clusters))


# ─────────────────────────────────────────────────────────────────────────── #
# 4. ClusterControlSystem                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

CONTROL_DOMINANCE_THRESHOLD = 0.70   # fraction of combat power needed to claim control
CONTESTED_STABILITY_PENALTY = 0.15   # supply penalty while contested


class ClusterControlSystem:
    """
    Tracks and updates cluster control (is_controlled / controlling_faction)
    based on current unit composition after combat.

    Rules:
      - A cluster is controlled by faction F if F owns ≥ CONTROL_DOMINANCE_THRESHOLD
        of all alive combat power present.
      - A cluster with multiple factions present but none dominant is 'contested'
        (is_controlled=False, controlling_faction=None).
      - Empty clusters retain their previous controller.
    """

    @classmethod
    def step(
        cls,
        world: AdvancedWorldMilitaryState,
    ) -> AdvancedWorldMilitaryState:
        """Update cluster control status for all clusters."""
        new_clusters = []
        for cluster in world.clusters:
            alive = [u for u in cluster.units if u.is_alive]
            if not alive:
                # No units — keep previous controller
                new_clusters.append(cluster)
                continue

            # Power by faction
            faction_power: Dict[int, float] = {}
            for u in alive:
                fid = u.objective_id if u.objective_id is not None else 0
                faction_power[fid] = faction_power.get(fid, 0.0) + u.combat_power

            total_power = sum(faction_power.values())
            if total_power < 1e-9:
                new_clusters.append(cluster)
                continue

            # Check dominance
            dominant_faction = None
            dominant_share = 0.0
            for fid, power in faction_power.items():
                share = power / total_power
                if share > dominant_share:
                    dominant_share = share
                    dominant_faction = fid

            if dominant_share >= CONTROL_DOMINANCE_THRESHOLD:
                new_clusters.append(cluster.copy_with(
                    is_controlled=True,
                    controlling_faction=dominant_faction,
                ))
            else:
                # Contested
                new_clusters.append(cluster.copy_with(
                    is_controlled=False,
                    controlling_faction=None,
                ))

        return world.copy_with(clusters=tuple(new_clusters))


# ─────────────────────────────────────────────────────────────────────────── #
# 5. EWExecutionEngine                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class EWExecutionEngine:
    """
    Concrete execution of all EWCapability effects each step.

    For each cluster that has an EW unit (ew_state not None):

      JAMMING     → degrade communication_network_integrity globally;
                    raise fog for all clusters within jamming_range.
      DECEPTION   → inject a false IntelligenceReport into a random adjacent
                    cluster with inflated enemy_units_detected.
      INTERCEPT   → generate a genuine IntelligenceReport for one adjacent
                    cluster, reducing its fog.
      DISRUPTION  → reduce combat_effectiveness of all enemy units in the
                    cluster by a small fraction.
      SPOOFING    → small chance that friendly fire damages own units
                    (communication confusion).
    """

    JAMMING_COMMS_PENALTY = 0.04      # per active jammer per step
    DISRUPTION_EFFECTIVENESS_LOSS = 0.05
    SPOOFING_FRIENDLY_FIRE_PROB = 0.08
    SPOOFING_FF_DAMAGE_FRACTION = 0.05
    DECEPTION_INFLATION_FACTOR = 3.0
    INTERCEPT_FOG_GAIN = 0.10

    @classmethod
    def step(
        cls,
        world: AdvancedWorldMilitaryState,
        params: MilitaryUnitParams,
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Apply one step of EW effects across the world."""
        new_world = world
        jamming_cluster_ids: Set[int] = set()
        comms_penalty = 0.0

        for cluster in world.clusters:
            if cluster.ew_state is None:
                continue
            ew = cluster.ew_state

            for capability in ew.capabilities:
                if capability == EWCapability.JAMMING and ew.active_jamming:
                    # Mark clusters within jamming range as jammed
                    # (approximate: mark all clusters, range is logical)
                    jamming_cluster_ids.add(cluster.cluster_id)
                    for other in world.clusters:
                        if other.cluster_id != cluster.cluster_id:
                            jamming_cluster_ids.add(other.cluster_id)
                    comms_penalty += cls.JAMMING_COMMS_PENALTY

                elif capability == EWCapability.DECEPTION:
                    new_world = cls._apply_deception(new_world, cluster, rng)

                elif capability == EWCapability.INTERCEPT:
                    new_world = cls._apply_intercept(new_world, cluster, rng)

                elif capability == EWCapability.DISRUPTION:
                    new_world = cls._apply_disruption(new_world, cluster, params, rng)

                elif capability == EWCapability.SPOOFING:
                    new_world = cls._apply_spoofing(new_world, cluster, params, rng)

        # Apply fog update from jamming
        if jamming_cluster_ids:
            new_world = FogOfWarSystem.step(new_world, jamming_cluster_ids)

        # Degrade comms integrity
        if comms_penalty > 0:
            new_integrity = max(0.0, new_world.communication_network_integrity - comms_penalty)
            new_world = new_world.copy_with(communication_network_integrity=new_integrity)
        else:
            # Comms integrity recovers passively
            new_integrity = min(1.0, new_world.communication_network_integrity + 0.01)
            new_world = new_world.copy_with(communication_network_integrity=new_integrity)

        return new_world

    @classmethod
    def _apply_deception(
        cls,
        world: AdvancedWorldMilitaryState,
        source_cluster: AdvancedClusterMilitaryState,
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Inject inflated false intelligence into a random adjacent cluster."""
        other_clusters = [c for c in world.clusters if c.cluster_id != source_cluster.cluster_id]
        if not other_clusters:
            return world

        target = rng.choice(other_clusters)
        false_units = int(max(1, target.unit_count * cls.DECEPTION_INFLATION_FACTOR))
        false_power = target.total_combat_power * cls.DECEPTION_INFLATION_FACTOR

        false_report = IntelligenceReport(
            report_id=world.next_report_id,
            intelligence_type=IntelligenceType.SIGINT,  # electronic signal intercept
            target_cluster_id=target.cluster_id,
            enemy_units_detected=false_units,
            enemy_combat_power=float(false_power),
            terrain_info="deceptive_signal",
            last_updated=world.step,
            confidence=0.85,  # high apparent confidence
            is_stale=False,
        )

        new_reports = (*target.intelligence_reports, false_report)
        new_cluster = target.copy_with(intelligence_reports=new_reports)
        new_clusters = tuple(
            new_cluster if c.cluster_id == target.cluster_id else c
            for c in world.clusters
        )
        return world.copy_with(
            clusters=new_clusters,
            next_report_id=world.next_report_id + 1,
        )

    @classmethod
    def _apply_intercept(
        cls,
        world: AdvancedWorldMilitaryState,
        source_cluster: AdvancedClusterMilitaryState,
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Generate genuine intel for an adjacent cluster, reducing its fog."""
        other_clusters = [c for c in world.clusters if c.cluster_id != source_cluster.cluster_id]
        if not other_clusters:
            return world

        target = rng.choice(other_clusters)
        real_report = IntelligenceReport(
            report_id=world.next_report_id,
            intelligence_type=IntelligenceType.ELINT,
            target_cluster_id=target.cluster_id,
            enemy_units_detected=target.unit_count,
            enemy_combat_power=float(target.total_combat_power),
            terrain_info="intercepted_signal",
            last_updated=world.step,
            confidence=0.70 + 0.20 * rng.random(),
            is_stale=False,
        )

        # Also directly reduce fog of the target cluster
        new_fog = max(0.0, target.fog_of_war - cls.INTERCEPT_FOG_GAIN)
        new_reports = (*target.intelligence_reports, real_report)
        new_cluster = target.copy_with(
            intelligence_reports=new_reports,
            fog_of_war=new_fog,
        )
        new_clusters = tuple(
            new_cluster if c.cluster_id == target.cluster_id else c
            for c in world.clusters
        )
        return world.copy_with(
            clusters=new_clusters,
            next_report_id=world.next_report_id + 1,
        )

    @classmethod
    def _apply_disruption(
        cls,
        world: AdvancedWorldMilitaryState,
        source_cluster: AdvancedClusterMilitaryState,
        params: MilitaryUnitParams,
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Reduce combat effectiveness of units in the cluster (enemy comms disrupted)."""
        new_units = []
        for unit in source_cluster.units:
            if unit.is_alive and rng.random() < 0.5:
                new_eff = max(0.1, unit.combat_effectiveness - cls.DISRUPTION_EFFECTIVENESS_LOSS)
                new_units.append(unit.copy_with(combat_effectiveness=new_eff))
            else:
                new_units.append(unit)

        new_cluster = source_cluster.copy_with(units=tuple(new_units))
        new_clusters = tuple(
            new_cluster if c.cluster_id == source_cluster.cluster_id else c
            for c in world.clusters
        )
        return world.copy_with(clusters=new_clusters)

    @classmethod
    def _apply_spoofing(
        cls,
        world: AdvancedWorldMilitaryState,
        source_cluster: AdvancedClusterMilitaryState,
        params: MilitaryUnitParams,
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Apply friendly fire risk to units in the cluster (IFF confusion)."""
        new_units = []
        for unit in source_cluster.units:
            if unit.is_alive and rng.random() < cls.SPOOFING_FRIENDLY_FIRE_PROB:
                ff_damage = unit.hit_points * cls.SPOOFING_FF_DAMAGE_FRACTION
                new_hp = max(0.0, unit.hit_points - ff_damage)
                new_morale = max(0.0, unit.morale - 0.05)
                new_units.append(unit.copy_with(hit_points=new_hp, morale=new_morale))
            else:
                new_units.append(unit)

        new_cluster = source_cluster.copy_with(units=tuple(new_units))
        new_clusters = tuple(
            new_cluster if c.cluster_id == source_cluster.cluster_id else c
            for c in world.clusters
        )
        return world.copy_with(clusters=new_clusters)


# ─────────────────────────────────────────────────────────────────────────── #
# 6. SupplyInterdictionSystem                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

INTERDICTION_PROB_BASE = 0.15         # base probability of interdiction per step
INTERDICTION_DEPOT_DAMAGE = 0.20      # fraction of supply depot destroyed
INTERDICTION_THROUGHPUT_PENALTY = 0.3 # fraction of throughput reduced


class SupplyInterdictionSystem:
    """
    Models enemy interdiction of supply lines.

    When a supply chain exists for a cluster and adjacent clusters are
    occupied by opposing factions, there is a probability each step that
    supply lines are cut, damaging the depot and reducing throughput.

    Interdiction probability scales with:
      - Number of adjacent hostile clusters
      - Special Forces presence in adjacent clusters (high disruption capability)
      - Fog of war in the target cluster (high fog = harder to defend routes)
    """

    @classmethod
    def step(
        cls,
        world: AdvancedWorldMilitaryState,
        adjacency_matrix: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> AdvancedWorldMilitaryState:
        """Apply supply interdiction checks for all clusters."""
        new_clusters = list(world.clusters)

        for idx, cluster in enumerate(world.clusters):
            if cluster.supply_chain is None:
                continue

            # Find adjacent hostile clusters
            n = adjacency_matrix.shape[0]
            hostile_adjacent = 0
            sf_threat = 0.0

            for other in world.clusters:
                if other.cluster_id == cluster.cluster_id:
                    continue
                if other.cluster_id >= n or cluster.cluster_id >= n:
                    continue
                if not adjacency_matrix[cluster.cluster_id, other.cluster_id]:
                    continue

                # Check if adjacent cluster is hostile (different controlling faction)
                cluster_faction = cluster.controlling_faction
                other_faction = other.controlling_faction
                if cluster_faction is not None and other_faction is not None and cluster_faction != other_faction:
                    hostile_adjacent += 1
                    # Special forces in adjacent hostile cluster increase threat
                    sf_in_other = sum(1 for u in other.units
                                      if u.is_alive and u.unit_type == MilitaryUnitType.SPECIAL_FORCES)
                    sf_threat += sf_in_other * 0.05

            if hostile_adjacent == 0:
                continue

            # Probability of interdiction this step
            interdiction_prob = INTERDICTION_PROB_BASE * hostile_adjacent + sf_threat
            interdiction_prob *= (0.5 + cluster.fog_of_war * 0.5)  # fog makes routes harder to protect
            interdiction_prob = min(0.80, interdiction_prob)

            if rng.random() < interdiction_prob:
                # Interdiction event: damage depot and reduce throughput
                new_depot = max(0.0, cluster.supply_depot * (1.0 - INTERDICTION_DEPOT_DAMAGE))
                new_chain = cluster.supply_chain.copy_with(
                    vulnerability=min(1.0, cluster.supply_chain.vulnerability + 0.05),
                )
                new_clusters[idx] = cluster.copy_with(
                    supply_depot=new_depot,
                    supply_chain=new_chain,
                )

        return world.copy_with(clusters=tuple(new_clusters))


# ─────────────────────────────────────────────────────────────────────────── #
# Convenience: full extension step                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def step_military_extensions(
    world: AdvancedWorldMilitaryState,
    params: MilitaryUnitParams,
    adjacency_matrix: NDArray[np.bool_],
    rng: np.random.Generator,
) -> Tuple[AdvancedWorldMilitaryState, Dict[str, Any]]:
    """
    Apply all extension systems in the correct order and return
    the updated world plus a metrics dictionary.

    This is the single call-site integration point for AdvancedMilitaryWrapper.

    Order:
      1. Multi-faction combat
      2. Morale dynamics
      3. Cluster control updates
      4. EW effects (including fog update from jamming)
      5. Supply interdiction

    FogOfWarSystem is called internally by EWExecutionEngine for jamming;
    it is also called here for passive fog drift (no active jammer argument).
    """
    # 1. Multi-faction combat
    world, combat_results = FactionCombatSystem.resolve_all_clusters(world, params, rng)

    # 2. Morale dynamics
    world = MoraleSystem.step(world, params, adjacency_matrix)

    # 3. Cluster control updates
    world = ClusterControlSystem.step(world)

    # 4. EW effects (fog handled inside)
    world = EWExecutionEngine.step(world, params, rng)

    # 5. Passive fog drift (no new jamming)
    world = FogOfWarSystem.step(world, jamming_active_cluster_ids=None)

    # 6. Supply interdiction
    world = SupplyInterdictionSystem.step(world, adjacency_matrix, rng)

    # Compile metrics
    total_casualties = {}
    for result in combat_results:
        for fid, cas in result.casualties.items():
            total_casualties[fid] = total_casualties.get(fid, 0.0) + cas

    routed_units = sum(
        1 for c in world.clusters for u in c.units
        if u.is_alive and u.combat_effectiveness < 0.35 and u.morale < ROUTE_THRESHOLD * 2
    )

    contested_clusters = sum(
        1 for c in world.clusters
        if not c.is_controlled and any(u.is_alive for u in c.units)
    )

    jammed_clusters = sum(
        1 for c in world.clusters
        if c.ew_state is not None and c.ew_state.active_jamming
    )

    metrics = {
        "combat_results": len(combat_results),
        "total_casualties_by_faction": total_casualties,
        "routed_units": routed_units,
        "contested_clusters": contested_clusters,
        "jammed_clusters": jammed_clusters,
        "communication_network_integrity": world.communication_network_integrity,
        "mean_fog_of_war": float(np.mean([c.fog_of_war for c in world.clusters])),
    }

    return world, metrics
