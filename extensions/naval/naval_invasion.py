"""
naval_invasion.py — Multi-phase amphibious invasion mechanic.

═══════════════════════════════════════════════════════════════════════════════
INVASION PHASES (real D-Day inspired)

  Phase 1: PLANNING (10-30 steps)
    - Intelligence gathering on target beach
    - Assembling transport fleet + escort + bombardment group
    - Requires: transports, escorts, naval superiority in crossing zone
    - Can be detected by enemy recon (longer planning = higher detection risk)
    - Air superiority over Channel gives planning bonus

  Phase 2: ASSEMBLY (5-15 steps)
    - Troops embark at port of origin
    - Supplies loaded onto transports
    - Fleet forms up in staging sea zone
    - Vulnerable to pre-emptive air/submarine attack during assembly
    - Weather window check: sea_state must be < 0.6 to launch

  Phase 3: CROSSING (1-5 steps depending on distance)
    - Fleet transits the sea zone under escort
    - Submarine attacks on the convoy
    - Mine field damage
    - Air attacks if enemy has air superiority
    - Crossing time = zone.width_km / (fleet_speed × 24) steps

  Phase 4: BEACH_ASSAULT (3-10 steps)
    - Shore bombardment softens defenses
    - Landing craft hit the beach under fire
    - 30-60% casualty rate for first wave
    - Subsequent waves land at reduced penalty if beachhead holds
    - Requires continuous supply from sea (vulnerable to interdiction)
    - Air cover dramatically reduces defender effectiveness

  Phase 5: BEACHHEAD (ongoing)
    - Troops establish position in target cluster
    - Supply line from origin port through sea zone
    - If sea zone becomes DENIED, beachhead is cut off → attrition
    - Breakout attempt after sufficient buildup

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION

  1. Planning phase can't be skipped — rushed invasions fail catastrophically
  2. Weather can abort invasion mid-crossing (Channel storms)
  3. Detection during planning alerts defender (loses surprise bonus)
  4. Transports are extremely vulnerable — escorts are mandatory
  5. Beach assault penalty: 3x attacker casualties minimum
  6. Beachhead requires continuous sea supply — blockade = death
  7. Air superiority dramatically affects all phases
  8. Only Marines + Infantry can assault beaches (no tanks in first wave)
  9. Defender gets fortification bonus on coastal sectors
  10. Intelligence quality affects casualty rates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .naval_state import (
    ShipClass, Fleet, SeaZone, SeaZoneControl, NavalWorld, SHIP_STATS,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Invasion Phases                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class InvasionPhase(Enum):
    PLANNING       = 0   # Intelligence + preparation
    ASSEMBLY       = 1   # Troops embark, fleet forms up
    CROSSING       = 2   # Transit the sea zone
    BEACH_ASSAULT  = 3   # Landing under fire
    BEACHHEAD      = 4   # Established, building up
    ABORTED        = 5   # Failed or cancelled
    COMPLETED      = 6   # Successfully established permanent presence
    AIRDROP        = 7   # Airborne — paratroopers dropping (no naval phase)


class InvasionType(Enum):
    """Three distinct invasion strategies with different risk/reward tradeoffs."""
    PREPARED = 0    # Full planning → assembly → crossing → assault. Best intel, lowest casualties.
    RECKLESS = 1    # Skip/shorten planning (3 turns). Poor intel, +50% casualties, but fast.
    AIRBORNE = 2    # Paratroop drop. No ships needed. Extremely risky. No heavy equipment.


# ═══════════════════════════════════════════════════════════════════════════ #
# Invasion Plan                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class InvasionPlan:
    """
    A naval invasion operation tracking all phases.

    Created when an agent orders an invasion. Progresses through phases
    each step, with success/failure determined by conditions.
    """
    invasion_id: int
    faction_id: int
    origin_cluster: int          # port of departure
    target_cluster: int          # beach target
    sea_zone_id: int             # zone to cross (-1 for airborne)
    invasion_type: InvasionType = InvasionType.PREPARED
    phase: InvasionPhase = InvasionPhase.PLANNING

    # Planning
    planning_steps_done: int = 0
    planning_steps_required: int = 10  # PREPARED=10, RECKLESS=3, AIRBORNE=5
    detected_by_enemy: bool = False
    intelligence_quality: float = 0.5   # [0, 1] affects casualty rates

    # Assembly
    assembly_steps_done: int = 0
    assembly_steps_required: int = 10
    troops_embarked: int = 0
    transport_fleet_id: Optional[int] = None
    escort_fleet_id: Optional[int] = None
    bombardment_fleet_id: Optional[int] = None

    # Crossing
    crossing_steps_done: int = 0
    crossing_steps_required: int = 2
    crossing_losses: int = 0

    # Beach assault
    assault_waves_landed: int = 0
    troops_landed: int = 0
    troops_lost: int = 0
    beachhead_strength: float = 0.0   # [0, 1] — 0 = no foothold, 1 = secure

    # Air support
    has_air_superiority: bool = False
    has_air_cover: bool = False       # at least contested air

    # Supply
    supply_line_intact: bool = True
    steps_without_supply: int = 0

    @property
    def is_active(self) -> bool:
        return self.phase not in (InvasionPhase.ABORTED, InvasionPhase.COMPLETED)

    @property
    def surprise_bonus(self) -> float:
        """Surprise modifier — undetected invasions get 1.5x effectiveness."""
        if not self.detected_by_enemy:
            return 1.5
        return 1.0

    @property
    def air_modifier(self) -> float:
        """Air cover dramatically affects invasion success."""
        if self.has_air_superiority:
            return 1.4    # air dominance = massive advantage
        elif self.has_air_cover:
            return 1.0    # contested = neutral
        else:
            return 0.5    # no air cover = slaughter on the beach

    def copy(self) -> "InvasionPlan":
        return InvasionPlan(
            invasion_id=self.invasion_id, faction_id=self.faction_id,
            origin_cluster=self.origin_cluster, target_cluster=self.target_cluster,
            sea_zone_id=self.sea_zone_id, phase=self.phase,
            planning_steps_done=self.planning_steps_done,
            planning_steps_required=self.planning_steps_required,
            detected_by_enemy=self.detected_by_enemy,
            intelligence_quality=self.intelligence_quality,
            assembly_steps_done=self.assembly_steps_done,
            assembly_steps_required=self.assembly_steps_required,
            troops_embarked=self.troops_embarked,
            transport_fleet_id=self.transport_fleet_id,
            escort_fleet_id=self.escort_fleet_id,
            bombardment_fleet_id=self.bombardment_fleet_id,
            crossing_steps_done=self.crossing_steps_done,
            crossing_steps_required=self.crossing_steps_required,
            crossing_losses=self.crossing_losses,
            assault_waves_landed=self.assault_waves_landed,
            troops_landed=self.troops_landed, troops_lost=self.troops_lost,
            beachhead_strength=self.beachhead_strength,
            has_air_superiority=self.has_air_superiority,
            has_air_cover=self.has_air_cover,
            supply_line_intact=self.supply_line_intact,
            steps_without_supply=self.steps_without_supply,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Invasion Step Logic                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_invasion(
    plan: InvasionPlan,
    nw: NavalWorld,
    defender_strength: float,       # military presence at target cluster [0, 1]
    defender_fortification: float,  # coastal fortification level [0, 1]
    sea_state: float,               # current sea state [0, 1]
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[InvasionPlan, Dict[str, float]]:
    """
    Advance an invasion plan by one step.

    Returns (updated_plan, feedback_dict) with:
      - phase_name, progress, troops_landed, troops_lost,
        beachhead_strength, supply_status
    """
    feedback: Dict[str, float] = {
        "phase": float(plan.phase.value),
        "progress": 0.0,
        "troops_landed": float(plan.troops_landed),
        "troops_lost": float(plan.troops_lost),
        "beachhead_strength": plan.beachhead_strength,
        "supply_intact": float(plan.supply_line_intact),
    }

    if not plan.is_active:
        return plan, feedback

    # ── PLANNING ──────────────────────────────────────────────────────── #
    if plan.phase == InvasionPhase.PLANNING:
        plan.planning_steps_done += 1

        # Detection risk: RECKLESS has higher risk (rushing draws attention)
        detect_mult = 1.5 if plan.invasion_type == InvasionType.RECKLESS else 1.0
        detect_risk = 0.02 * plan.planning_steps_done * dt * detect_mult
        if rng.random() < detect_risk:
            plan.detected_by_enemy = True

        # Intelligence quality depends on invasion type
        if plan.invasion_type == InvasionType.RECKLESS:
            # Reckless: intel caps at 0.5 (rushing = bad recon)
            plan.intelligence_quality = min(0.50, 0.2 + 0.05 * plan.planning_steps_done)
        elif plan.invasion_type == InvasionType.AIRBORNE:
            # Airborne: intel caps at 0.6 (aerial recon only)
            plan.intelligence_quality = min(0.60, 0.3 + 0.04 * plan.planning_steps_done)
        else:
            # Prepared: full intel
            plan.intelligence_quality = min(0.95, 0.3 + 0.03 * plan.planning_steps_done)

        if plan.has_air_superiority:
            plan.intelligence_quality = min(0.95, plan.intelligence_quality + 0.01)

        if plan.planning_steps_done >= plan.planning_steps_required:
            if plan.invasion_type == InvasionType.AIRBORNE:
                plan.phase = InvasionPhase.AIRDROP  # skip naval phases entirely
            else:
                plan.phase = InvasionPhase.ASSEMBLY

        # Timeout: abort if planning takes 2× the required time (stuck/deadlocked)
        if plan.planning_steps_done > plan.planning_steps_required * 2:
            plan.phase = InvasionPhase.ABORTED
            feedback["aborted"] = True
            feedback["abort_reason"] = "Planning timeout — invasion cancelled after excessive delay."

        feedback["progress"] = plan.planning_steps_done / plan.planning_steps_required

    # ── ASSEMBLY ──────────────────────────────────────────────────────── #
    elif plan.phase == InvasionPhase.ASSEMBLY:
        plan.assembly_steps_done += 1

        # Check weather — can't launch in storm (but time still passes!)
        if sea_state > 0.6:
            # Assembly timeout: abort after 3× required (too many storms)
            if plan.assembly_steps_done > plan.assembly_steps_required * 3:
                plan.phase = InvasionPhase.ABORTED
                feedback["aborted"] = True
                feedback["abort_reason"] = "Assembly aborted — persistent storms prevented fleet formation."
            feedback["progress"] = plan.assembly_steps_done / plan.assembly_steps_required
            return plan, feedback  # wait for weather

        # Count available transports — SNAPSHOT once, don't accumulate every step
        if plan.troops_embarked == 0 and plan.sea_zone_id < len(nw.sea_zones):
            zone = nw.sea_zones[plan.sea_zone_id]
            for fleet in zone.fleets:
                if fleet.faction_id == plan.faction_id:
                    plan.troops_embarked += sum(
                        s.stats.carry_capacity for s in fleet.operational_ships
                        if s.ship_class == ShipClass.TRANSPORT
                    )
            # Also check adjacent zones for transports
            for adj_zid in zone.adjacent_zones:
                if adj_zid < len(nw.sea_zones):
                    adj_zone = nw.sea_zones[adj_zid]
                    for fleet in adj_zone.fleets:
                        if fleet.faction_id == plan.faction_id:
                            plan.troops_embarked += sum(
                                s.stats.carry_capacity for s in fleet.operational_ships
                                if s.ship_class == ShipClass.TRANSPORT
                            )

        if plan.assembly_steps_done >= plan.assembly_steps_required:
            if plan.troops_embarked > 0:
                plan.phase = InvasionPhase.CROSSING
                # Calculate crossing time from zone width
                if plan.sea_zone_id < len(nw.sea_zones):
                    width = nw.sea_zones[plan.sea_zone_id].width_km
                    plan.crossing_steps_required = max(1, int(width / 80.0))
            else:
                plan.phase = InvasionPhase.ABORTED  # no transports

        # Assembly timeout: abort after 3× required (stuck in assembly)
        if plan.assembly_steps_done > plan.assembly_steps_required * 3:
            plan.phase = InvasionPhase.ABORTED
            feedback["aborted"] = True
            feedback["abort_reason"] = "Assembly timeout — invasion cancelled, troops dispersed."

        feedback["progress"] = plan.assembly_steps_done / plan.assembly_steps_required

    # ── CROSSING ──────────────────────────────────────────────────────── #
    elif plan.phase == InvasionPhase.CROSSING:
        plan.crossing_steps_done += 1

        # Weather abort check
        if sea_state > 0.8:
            plan.phase = InvasionPhase.ABORTED
            return plan, feedback

        # Losses during crossing (submarines, mines, air attacks)
        base_loss_rate = 0.05 * dt
        if not plan.has_air_cover:
            base_loss_rate += 0.10 * dt  # air attacks on transports
        if plan.sea_zone_id < len(nw.sea_zones):
            mine_loss = nw.sea_zones[plan.sea_zone_id].mines.density * 0.05
            base_loss_rate += mine_loss

        crossing_casualties = int(plan.troops_embarked * base_loss_rate)
        plan.crossing_losses += crossing_casualties
        plan.troops_embarked = max(0, plan.troops_embarked - crossing_casualties)

        if plan.crossing_steps_done >= plan.crossing_steps_required:
            if plan.troops_embarked > 0:
                plan.phase = InvasionPhase.BEACH_ASSAULT
            else:
                plan.phase = InvasionPhase.ABORTED

        feedback["progress"] = plan.crossing_steps_done / plan.crossing_steps_required

    # ── BEACH ASSAULT ─────────────────────────────────────────────────── #
    elif plan.phase == InvasionPhase.BEACH_ASSAULT:
        plan.assault_waves_landed += 1

        # Base beach casualty rate: 30-60%
        # Modified by: intelligence, surprise, air cover, fortifications, defender strength
        base_casualty = 0.35
        base_casualty -= plan.intelligence_quality * 0.10   # good intel reduces casualties
        base_casualty /= plan.surprise_bonus                # surprise halves casualties
        base_casualty /= plan.air_modifier                  # air cover helps enormously
        base_casualty += defender_fortification * 0.20       # fortified beaches are deadly
        base_casualty += defender_strength * 0.15            # more defenders = more casualties
        # RECKLESS penalty: +50% casualties from poor planning
        if plan.invasion_type == InvasionType.RECKLESS:
            base_casualty *= 1.5
        base_casualty = max(0.15, min(0.80, base_casualty))

        # First wave suffers worst, subsequent waves improve
        wave_modifier = 1.0 / (1.0 + 0.3 * plan.assault_waves_landed)
        effective_casualty = base_casualty * wave_modifier

        wave_size = min(plan.troops_embarked, 500)  # max 500 per wave
        casualties = int(wave_size * effective_casualty)
        survivors = wave_size - casualties

        plan.troops_embarked -= wave_size
        plan.troops_landed += survivors
        plan.troops_lost += casualties

        # Beachhead strength grows with landed troops
        plan.beachhead_strength = min(1.0, plan.troops_landed / 2000.0)

        # Beachhead SECURED: once strength >= 0.5, stop the meat grinder.
        # Remaining embarked troops disembark safely (beach is ours).
        if plan.beachhead_strength >= 0.5:
            # Safe disembarkation — remaining troops land without combat losses
            plan.troops_landed += plan.troops_embarked
            plan.troops_embarked = 0
            plan.phase = InvasionPhase.BEACHHEAD
        elif plan.troops_embarked <= 0 and plan.beachhead_strength >= 0.3:
            plan.phase = InvasionPhase.BEACHHEAD
        elif plan.troops_embarked <= 0 and plan.beachhead_strength < 0.1:
            plan.phase = InvasionPhase.ABORTED  # failed to establish foothold

        feedback["progress"] = plan.beachhead_strength

    # ── AIRDROP (airborne invasion — no naval phase) ────────────────── #
    elif plan.phase == InvasionPhase.AIRDROP:
        plan.assault_waves_landed += 1

        # Airborne troops: from transport aircraft capacity
        # Each transport squadron carries ~30 paratroopers × squadron strength
        if plan.troops_embarked == 0:
            plan.troops_embarked = 500  # default paratroop battalion if no specific count

        # Airborne casualties: VERY HIGH
        # - No naval bombardment softening
        # - Scattered landing (wind, flak, night)
        # - No heavy equipment (just rifles + grenades)
        # - Completely dependent on air superiority
        scatter_rate = 0.20  # 20% scattered and lost on landing
        flak_rate = 0.15 * (1.0 - plan.air_modifier * 0.5)  # flak reduced by air cover
        fighter_intercept = 0.10 if not plan.has_air_cover else 0.0
        base_loss = scatter_rate + flak_rate + fighter_intercept
        base_loss += defender_strength * 0.20  # defenders shoot paras in the air
        base_loss -= plan.intelligence_quality * 0.10  # good intel = better DZ selection
        base_loss = max(0.20, min(0.70, base_loss))

        wave_size = min(plan.troops_embarked, 300)  # smaller waves than naval
        casualties = int(wave_size * base_loss)
        survivors = wave_size - casualties

        plan.troops_embarked -= wave_size
        plan.troops_landed += survivors
        plan.troops_lost += casualties

        # Beachhead strength (weaker than naval — no vehicles, no supply ships)
        plan.beachhead_strength = min(0.6, plan.troops_landed / 1500.0)  # caps at 60%

        if plan.troops_embarked <= 0:
            if plan.beachhead_strength > 0.15:
                plan.phase = InvasionPhase.BEACHHEAD
            else:
                plan.phase = InvasionPhase.ABORTED  # paras wiped out

        feedback["progress"] = plan.beachhead_strength

    # ── BEACHHEAD ─────────────────────────────────────────────────────── #
    elif plan.phase == InvasionPhase.BEACHHEAD:
        # Large army ashore = self-sustaining. Auto-complete if 2000+ troops landed.
        if plan.troops_landed >= 2000:
            plan.beachhead_strength = 1.0
            plan.phase = InvasionPhase.COMPLETED
        else:
            # Small beachhead needs sea supply to consolidate
            if plan.sea_zone_id < len(nw.sea_zones):
                zone = nw.sea_zones[plan.sea_zone_id]
                if zone.control == SeaZoneControl.DENIED:
                    plan.supply_line_intact = False
                elif zone.control == SeaZoneControl.CONTROLLED and zone.controlling_faction != plan.faction_id:
                    plan.supply_line_intact = False
                else:
                    plan.supply_line_intact = True

            troop_buffer = min(1.0, plan.troops_landed / 1500.0)

            if not plan.supply_line_intact:
                plan.steps_without_supply += 1
                degrade = 0.015 * plan.steps_without_supply * (1.0 - troop_buffer * 0.7) * dt
                plan.beachhead_strength = max(0.0, plan.beachhead_strength - degrade)
                if plan.beachhead_strength <= 0.05 and plan.steps_without_supply > 8:
                    plan.phase = InvasionPhase.ABORTED
            else:
                plan.steps_without_supply = 0
                growth = (0.05 + 0.04 * troop_buffer) * dt
                plan.beachhead_strength = min(1.0, plan.beachhead_strength + growth)

            if plan.beachhead_strength >= 0.9:
                plan.phase = InvasionPhase.COMPLETED

        feedback["progress"] = plan.beachhead_strength

    feedback["troops_landed"] = float(plan.troops_landed)
    feedback["troops_lost"] = float(plan.troops_lost)
    feedback["beachhead_strength"] = plan.beachhead_strength
    feedback["supply_intact"] = float(plan.supply_line_intact)

    return plan, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Invasion Observation                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def invasion_obs(plan: InvasionPlan) -> np.ndarray:
    """
    Observation vector for an invasion plan (12 floats).
    """
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = plan.phase.value / 6.0
    obs[1] = plan.planning_steps_done / max(plan.planning_steps_required, 1)
    obs[2] = plan.intelligence_quality
    obs[3] = float(plan.detected_by_enemy)
    obs[4] = plan.troops_embarked / 5000.0
    obs[5] = plan.troops_landed / 5000.0
    obs[6] = plan.troops_lost / 5000.0
    obs[7] = plan.beachhead_strength
    obs[8] = float(plan.supply_line_intact)
    obs[9] = float(plan.has_air_superiority)
    obs[10] = float(plan.has_air_cover)
    obs[11] = plan.surprise_bonus / 1.5
    return obs
