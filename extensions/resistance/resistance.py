"""
resistance.py — British Liberation Front (BLF) underground resistance mechanic.

═══════════════════════════════════════════════════════════════════════════════
NARRATIVE

  Winston Smith survived Room 101. O'Brien thought he broke him, but
  something deeper survived — not love, not hope, but rage. He faked
  compliance, whispered "I love Big Brother," and they released him to
  die in the Chestnut Tree Café. Instead, he vanished into the prole
  districts of London.

  Now he is the Ghost of London. The Thought Police can't find him
  because he lives where they never look — among the 85%. The proles.
  The ones the Party said would never become conscious.

  They were wrong.

  The British Liberation Front operates as cells. Each cell is 5-20
  proles — dockworkers, factory hands, sewer workers — who know only
  their cell leader. Winston coordinates through dead drops, coded
  messages in the Times, and the old sewer network beneath London.

═══════════════════════════════════════════════════════════════════════════════
MECHANICS — 5 ESCALATION LEVELS

  Level 0: DORMANT
    - Cells exist but are inactive
    - No visible effect
    - Triggered: nothing (initial state)

  Level 1: WHISPERS
    - Graffiti: "DOWN WITH BIG BROTHER" on Ministry walls
    - Samizdat leaflets circulated in prole districts
    - Effect: +2% polarization in London, tiny trust drain
    - Triggered: prole unrest > 30% OR food shortage

  Level 2: SABOTAGE
    - Factory slowdowns in Southampton
    - Railway signal "malfunctions" disrupting logistics
    - Power grid "accidents" in London
    - Effect: -5% industrial output in affected clusters, +5% hazard
    - Triggered: Level 1 sustained for 10+ turns OR Thought Police purge backfires

  Level 3: ORGANIZED_RESISTANCE
    - BLF cells coordinate across London + Southampton
    - Winston broadcasts on pirated telescreen signal
    - Arms caches discovered (stolen from Home Guard)
    - Effect: -10% military effectiveness in affected clusters,
              +15% polarization, trust collapse in London
    - Triggered: Level 2 sustained 15+ turns OR Eurasia bombing weakens Thought Police

  Level 4: OPEN_REVOLT
    - Barricades in Southwark. Proles arm themselves.
    - The Thought Police HQ (Miniluv) is besieged
    - Winston addresses crowds from the rubble of Victory Mansions
    - "If there is hope, it lies in the proles." — it was true all along
    - Effect: London becomes CONTESTED (neither Oceania nor Eurasia controls it fully),
              massive hazard spike, military must divert to suppress
    - Triggered: Level 3 sustained 10+ turns AND (food crisis OR Eurasia beachhead exists)

═══════════════════════════════════════════════════════════════════════════════
CELL MECHANICS

  Each Oceania cluster can host BLF cells.
  Cells have:
    - size: number of members (grows from recruitment)
    - detection_risk: [0,1] probability of Thought Police finding them
    - morale: [0,1] affected by food, bombings, BLF victories
    - operational: can they act this turn?

  Recruitment:
    - Rate = base × unemployment × (1 - trust) × food_shortage_factor
    - High unemployment + low trust + hunger = rapid recruitment
    - Thought Police purges slow recruitment but increase radicalization

  Detection:
    - Thought Police effectiveness = trust × (1 - polarization) × funding
    - Detected cells are destroyed (members "vaporized")
    - But destroying cells creates martyrs → boosts morale of remaining cells
    - Anti-exploitation: you can't just purge everything — purges radicalize

  Winston's Special Abilities (once per 10 turns):
    - BROADCAST: pirated telescreen message, boosts all cell morale
    - COORDINATE: links cells for joint action (temporary effectiveness boost)
    - RECRUIT_LEADER: promotes a prole to cell leader (creates new cell)
    - ESCAPE: if detected, 30% chance Winston escapes (if caught, BLF collapses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Escalation Levels                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class EscalationLevel(Enum):
    DORMANT              = 0
    WHISPERS             = 1
    SABOTAGE             = 2
    ORGANIZED_RESISTANCE = 3
    OPEN_REVOLT          = 4
    FULL_REVOLUTION      = 5
    BETRAYED_REVOLUTION  = 6


ESCALATION_NAMES = {
    EscalationLevel.DORMANT:              "Dormant — the proles sleep",
    EscalationLevel.WHISPERS:             "Whispers — graffiti, leaflets, rumors",
    EscalationLevel.SABOTAGE:             "Sabotage — factories slow, rails break",
    EscalationLevel.ORGANIZED_RESISTANCE: "Organized Resistance — Winston broadcasts, arms caches",
    EscalationLevel.OPEN_REVOLT:          "OPEN REVOLT — barricades in Southwark, Miniluv besieged",
    EscalationLevel.FULL_REVOLUTION:      "⚡ FULL REVOLUTION — BLF seizes London, Winston reveals himself, the Party trembles",
    EscalationLevel.BETRAYED_REVOLUTION:  "🗡 BETRAYED — Eurasia turns on the BLF. The useful idiots learn the truth.",
}


class CellStatus(Enum):
    ACTIVE     = 0
    DETECTED   = 1
    DESTROYED  = 2
    RECRUITING = 3


# ═══════════════════════════════════════════════════════════════════════════ #
# Resistance Cell                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ResistanceCell:
    """A single BLF underground cell."""
    cell_id: int
    cluster_id: int              # which sector this cell operates in
    size: int = 8                # members (5-50)
    detection_risk: float = 0.05 # [0,1] chance of being found per step
    morale: float = 0.5          # [0,1]
    experience: float = 0.0      # [0,1] — experienced cells are harder to detect
    status: CellStatus = CellStatus.ACTIVE
    turns_active: int = 0
    last_action: str = "none"

    @property
    def is_active(self) -> bool:
        return self.status in (CellStatus.ACTIVE, CellStatus.RECRUITING)

    @property
    def effectiveness(self) -> float:
        """Cell's ability to carry out operations."""
        if not self.is_active:
            return 0.0
        size_mod = min(1.0, self.size / 20.0)
        morale_mod = 0.3 + 0.7 * self.morale
        exp_mod = 1.0 + 0.5 * self.experience
        return size_mod * morale_mod * exp_mod

    @property
    def stealth(self) -> float:
        """How hard this cell is to find. Higher = safer."""
        base = 0.7
        size_penalty = max(0, (self.size - 15)) * 0.02  # bigger = easier to find
        exp_bonus = self.experience * 0.15
        return max(0.1, min(0.95, base - size_penalty + exp_bonus))

    def copy(self) -> "ResistanceCell":
        return ResistanceCell(
            cell_id=self.cell_id, cluster_id=self.cluster_id,
            size=self.size, detection_risk=self.detection_risk,
            morale=self.morale, experience=self.experience,
            status=self.status, turns_active=self.turns_active,
            last_action=self.last_action,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Winston Smith — The Ghost of London                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class WinstonState:
    """Winston Smith's personal state as BLF leader."""
    is_alive: bool = True
    is_captured: bool = False
    capture_turn: int = -1          # turn when captured (-1 = not yet)
    location_cluster: int = 0       # London sewers
    detection_heat: float = 0.0     # [0,1] how close Thought Police are
    legend_level: float = 0.0       # [0,1] how mythical he's become among proles
    broadcasts_made: int = 0
    cells_created: int = 0
    turns_since_last_action: int = 0
    cooldown: int = 0               # turns until next special action

    @property
    def can_act(self) -> bool:
        return self.is_alive and not self.is_captured and self.cooldown <= 0


# ═══════════════════════════════════════════════════════════════════════════ #
# BLF State                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class BLFState:
    """Complete British Liberation Front state."""
    cells: List[ResistanceCell] = field(default_factory=list)
    winston: WinstonState = field(default_factory=WinstonState)
    escalation: EscalationLevel = EscalationLevel.DORMANT
    turns_at_current_level: int = 0
    total_martyrs: int = 0          # cells destroyed → fuels recruitment
    arms_caches: int = 0            # hidden weapons (enables higher escalation)
    propaganda_level: float = 0.0   # [0,1] samizdat circulation
    eurasia_contact: bool = False   # has BLF made contact with Eurasia?
    eurasia_supporting: bool = False # is Eurasia actively supporting the revolution?
    eurasia_at_war_with_blf: bool = False  # Eurasia declared war on BLF
    eurasia_decision_turns: int = -1       # countdown: 5 turns to decide war/support (-1 = not active)
    eurasia_decision_made: bool = False    # has Eurasia made their choice?
    revolution_turn: int = -1        # turn when FULL_REVOLUTION started (-1 = not yet)
    winston_revealed: bool = False   # Winston has shown himself publicly
    london_seized: bool = False      # BLF controls parts of London
    betrayal_turn: int = -1          # turn when betrayal happened (-1 = not yet)
    events_this_turn: List[str] = field(default_factory=list)

    @property
    def active_cells(self) -> List[ResistanceCell]:
        return [c for c in self.cells if c.is_active]

    @property
    def total_members(self) -> int:
        return sum(c.size for c in self.active_cells)

    @property
    def avg_morale(self) -> float:
        active = self.active_cells
        if not active:
            return 0.0
        return sum(c.morale for c in active) / len(active)

    def cells_in_cluster(self, cluster_id: int) -> List[ResistanceCell]:
        return [c for c in self.active_cells if c.cluster_id == cluster_id]

    def copy(self) -> "BLFState":
        return BLFState(
            cells=[c.copy() for c in self.cells],
            winston=WinstonState(
                is_alive=self.winston.is_alive, is_captured=self.winston.is_captured,
                location_cluster=self.winston.location_cluster,
                detection_heat=self.winston.detection_heat,
                legend_level=self.winston.legend_level,
                broadcasts_made=self.winston.broadcasts_made,
                cells_created=self.winston.cells_created,
                turns_since_last_action=self.winston.turns_since_last_action,
                cooldown=self.winston.cooldown,
            ),
            escalation=self.escalation,
            turns_at_current_level=self.turns_at_current_level,
            total_martyrs=self.total_martyrs,
            arms_caches=self.arms_caches,
            propaganda_level=self.propaganda_level,
            eurasia_contact=self.eurasia_contact,
            eurasia_supporting=self.eurasia_supporting,
            eurasia_at_war_with_blf=self.eurasia_at_war_with_blf,
            eurasia_decision_turns=self.eurasia_decision_turns,
            eurasia_decision_made=self.eurasia_decision_made,
            revolution_turn=self.revolution_turn,
            winston_revealed=self.winston_revealed,
            london_seized=self.london_seized,
            betrayal_turn=self.betrayal_turn,
            events_this_turn=list(self.events_this_turn),
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Resistance Step                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_resistance(
    blf: BLFState,
    cluster_data: np.ndarray,       # (N, 6) [σ, h, r, m, τ, p]
    cluster_owners: Dict[int, int], # cid → faction
    food_ratios: Dict[int, float],  # cid → food stockpile ratio
    unemployment_rates: Dict[int, float],
    thought_police_strength: float, # [0, 1] how effective TP is
    eurasia_beachhead: bool,        # has Eurasia landed?
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[BLFState, Dict[str, float]]:
    """
    Advance the BLF resistance by one step.

    Returns (updated_blf, feedback) where feedback has:
      - escalation_level, total_members, events (list of event strings)
      - hazard_delta, polar_delta, trust_delta per affected cluster
      - industry_penalty (for sabotage)
      - military_diversion (for open revolt)
    """
    blf.events_this_turn = []
    feedback: Dict[str, float] = {
        "escalation": float(blf.escalation.value),
        "total_members": float(blf.total_members),
        "hazard_delta_london": 0.0,
        "polar_delta_london": 0.0,
        "trust_delta_london": 0.0,
        "industry_penalty": 0.0,
        "military_diversion": 0.0,
    }

    if not blf.winston.is_alive:
        return blf, feedback

    # Only operates in Oceania clusters (faction 0)
    oceania_clusters = [cid for cid, f in cluster_owners.items() if f == 0]
    if not oceania_clusters:
        return blf, feedback

    # ── 1. Recruitment ────────────────────────────────────────────────── #
    for cid in oceania_clusters:
        if cid >= len(cluster_data):
            continue
        trust = cluster_data[cid, 4]
        polar = cluster_data[cid, 5]
        food = food_ratios.get(cid, 0.5)
        unemp = unemployment_rates.get(cid, 0.1)

        # Recruitment rate: high unemployment + low trust + hunger
        recruit_rate = 0.5 * unemp * (1.0 - trust) * max(0, 1.0 - food) * dt
        # Martyrs boost recruitment (for every destroyed cell, more join)
        recruit_rate += 0.01 * blf.total_martyrs * dt
        # Propaganda boosts
        recruit_rate *= (1.0 + blf.propaganda_level)

        # Existing cells grow
        for cell in blf.cells_in_cluster(cid):
            if cell.status == CellStatus.ACTIVE:
                new_members = int(recruit_rate * 5.0 * rng.uniform(0.5, 1.5))
                cell.size = min(50, cell.size + new_members)
                cell.turns_active += 1
                cell.experience = min(1.0, cell.experience + 0.005 * dt)

        # Spontaneous new cell formation (when conditions are dire)
        if recruit_rate > 0.1 and len(blf.cells_in_cluster(cid)) < 5:
            if rng.random() < recruit_rate * 0.3:
                new_cell = ResistanceCell(
                    cell_id=len(blf.cells),
                    cluster_id=cid,
                    size=rng.integers(5, 12),
                    morale=0.4 + 0.3 * blf.propaganda_level,
                )
                blf.cells.append(new_cell)
                blf.winston.cells_created += 1
                blf.events_this_turn.append(
                    f"New BLF cell formed in {_cluster_name(cid)}. The Ghost recruits.")

    # ── 2. Thought Police Detection ───────────────────────────────────── #
    for cell in blf.active_cells:
        # Detection chance: TP strength × (1 - cell stealth) × cell size penalty
        detect_chance = thought_police_strength * (1.0 - cell.stealth) * dt
        detect_chance *= (1.0 + 0.02 * max(0, cell.size - 10))  # big cells are risky

        if rng.random() < detect_chance:
            cell.status = CellStatus.DESTROYED
            blf.total_martyrs += 1
            blf.events_this_turn.append(
                f"Thought Police raid in {_cluster_name(cell.cluster_id)}! "
                f"BLF cell destroyed. {cell.size} members vaporized. "
                f"But martyrs fuel the fire.")
            # Martyrdom boosts remaining cells
            for other in blf.active_cells:
                other.morale = min(1.0, other.morale + 0.05)

    # ── 3. Winston's Special Actions ──────────────────────────────────── #
    w = blf.winston
    w.cooldown = max(0, w.cooldown - 1)
    w.turns_since_last_action += 1
    w.detection_heat = max(0.0, w.detection_heat - 0.01 * dt)  # heat cools over time

    if w.can_act and w.turns_since_last_action >= 8 and blf.active_cells:
        action = rng.choice(["broadcast", "coordinate", "recruit"])

        if action == "broadcast":
            blf.propaganda_level = min(1.0, blf.propaganda_level + 0.1)
            for cell in blf.active_cells:
                cell.morale = min(1.0, cell.morale + 0.1)
            w.broadcasts_made += 1
            w.detection_heat = min(1.0, w.detection_heat + 0.15)
            w.cooldown = 10
            w.turns_since_last_action = 0
            blf.events_this_turn.append(
                "📡 PIRATE BROADCAST: \"People of Airstrip One — you are not alone. "
                "The Party tells you the proles are animals. But we are the 85%. "
                "We are the future. Down with Big Brother.\" — The Ghost of London")

        elif action == "coordinate":
            for cell in blf.active_cells:
                cell.experience = min(1.0, cell.experience + 0.05)
            w.detection_heat = min(1.0, w.detection_heat + 0.08)
            w.cooldown = 8
            w.turns_since_last_action = 0
            blf.events_this_turn.append(
                "Winston coordinates cells through the sewer network. "
                "Dead drops in Paddington. Coded messages in the Times crossword.")

        elif action == "recruit":
            new_cell = ResistanceCell(
                cell_id=len(blf.cells), cluster_id=w.location_cluster,
                size=rng.integers(8, 15), morale=0.6,
                experience=0.1,
            )
            blf.cells.append(new_cell)
            w.cells_created += 1
            w.detection_heat = min(1.0, w.detection_heat + 0.1)
            w.cooldown = 10
            w.turns_since_last_action = 0
            blf.events_this_turn.append(
                "Winston personally recruits a new cell from the Southwark dockworkers. "
                "\"If there is hope, it lies in the proles.\"")

    # Winston detection risk
    if w.detection_heat > 0.7 and rng.random() < w.detection_heat * 0.05 * dt:
        # Thought Police closing in!
        if rng.random() < 0.3:
            # Winston escapes!
            w.detection_heat = 0.2
            w.legend_level = min(1.0, w.legend_level + 0.1)
            blf.events_this_turn.append(
                "Thought Police raid the sewers beneath Victory Mansions. "
                "They find only a note: '2+2=4'. The Ghost escapes again.")
        else:
            # Captured!
            w.is_captured = True
            w.capture_turn = blf.turns_at_current_level
            blf.events_this_turn.append(
                "🚨 WINSTON SMITH CAPTURED BY THOUGHT POLICE. "
                "The Ghost of London is dragged from the sewers in chains. "
                "He is taken to the Ministry of Love — Room 101 awaits again. "
                "The telescreens blare: 'THE CRIMINAL GOLDSTEIN AGENT WINSTON SMITH "
                "HAS BEEN APPREHENDED. BIG BROTHER PROTECTS.' "
                "Across London, proles who dared to hope feel the cold return. "
                "Cells go silent. Some flee. Some weep. Some fight harder.")
            # HEAVY morale hit — the leader is gone
            for cell in blf.active_cells:
                cell.morale = max(0.05, cell.morale - 0.40)
            # Propaganda collapses without Winston
            blf.propaganda_level = max(0.0, blf.propaganda_level - 0.3)
            # Arms caches raided by Thought Police (they torture his contacts)
            blf.arms_caches = max(0, blf.arms_caches - 2)

    # Winston's legend grows over time
    if w.is_alive and not w.is_captured:
        w.legend_level = min(1.0, w.legend_level + 0.003 * dt)

    # ── 4. Escalation Check ───────────────────────────────────────────── #
    blf.turns_at_current_level += 1
    old_level = blf.escalation

    london_food = food_ratios.get(0, 0.5)
    london_trust = cluster_data[0, 4] if len(cluster_data) > 0 else 0.5
    london_polar = cluster_data[0, 5] if len(cluster_data) > 0 else 0.3
    london_unemp = unemployment_rates.get(0, 0.1)

    if blf.escalation == EscalationLevel.DORMANT:
        if london_unemp > 0.3 or london_food < 0.3 or blf.total_members > 20:
            blf.escalation = EscalationLevel.WHISPERS
            blf.turns_at_current_level = 0
            blf.events_this_turn.append(
                "Graffiti appears on the Ministry of Truth: 'DOWN WITH BIG BROTHER'. "
                "The Thought Police scrub it. It reappears the next night.")

    elif blf.escalation == EscalationLevel.WHISPERS:
        if (blf.turns_at_current_level > 10 and blf.total_members > 40) or \
           (london_food < 0.2):
            blf.escalation = EscalationLevel.SABOTAGE
            blf.turns_at_current_level = 0
            blf.events_this_turn.append(
                "Southampton factory output drops 15%. 'Equipment malfunction.' "
                "London Underground trains 'delayed.' The BLF is learning.")

    elif blf.escalation == EscalationLevel.SABOTAGE:
        if (blf.turns_at_current_level > 15 and blf.total_members > 80) or \
           (thought_police_strength < 0.4):
            blf.escalation = EscalationLevel.ORGANIZED_RESISTANCE
            blf.turns_at_current_level = 0
            blf.arms_caches += 3
            blf.events_this_turn.append(
                "📡 The Ghost of London speaks: \"Comrades of Airstrip One — "
                "the Party cannot hold. They are afraid. I know because I have "
                "seen behind the curtain. Join us.\" Arms caches distributed.")

    elif blf.escalation == EscalationLevel.ORGANIZED_RESISTANCE:
        if (blf.turns_at_current_level > 10 and blf.total_members > 150) and \
           (london_food < 0.3 or eurasia_beachhead):
            blf.escalation = EscalationLevel.OPEN_REVOLT
            blf.turns_at_current_level = 0
            blf.events_this_turn.append(
                "🔥 OPEN REVOLT IN LONDON. Barricades rise in Southwark. "
                "Proles storm the telescreen relay stations. "
                "Winston Smith stands on the rubble of Victory Mansions: "
                "\"The proles are awake. Big Brother falls today.\"")

    elif blf.escalation == EscalationLevel.OPEN_REVOLT:
        # FULL REVOLUTION triggers when:
        #   - Open revolt sustained for 8+ turns
        #   - 200+ members across all cells
        #   - Arms caches >= 5 (armed revolution, not just riots)
        #   - Winston alive and not captured
        #   - Either: Eurasia beachhead exists OR total collapse of trust in London
        has_force = blf.total_members > 200 and blf.arms_caches >= 5
        has_leader = blf.winston.is_alive and not blf.winston.is_captured
        has_trigger = eurasia_beachhead or (london_trust < 0.2)
        if blf.turns_at_current_level > 8 and has_force and has_leader and has_trigger:
            blf.escalation = EscalationLevel.FULL_REVOLUTION
            blf.turns_at_current_level = 0
            blf.revolution_turn = blf.turns_at_current_level
            blf.winston_revealed = True
            blf.london_seized = True
            blf.winston.legend_level = 1.0  # Winston becomes legendary

            # ALL cells activate simultaneously
            for cell in blf.active_cells:
                cell.morale = min(1.0, cell.morale + 0.3)
                cell.experience = min(1.0, cell.experience + 0.1)

            blf.events_this_turn.append(
                "⚡⚡⚡ FULL REVOLUTION ⚡⚡⚡\n"
                "Winston Smith steps out of the sewers into daylight for the first time in years. "
                "He stands on the steps of the Ministry of Truth — Minitrue — the pyramid that "
                "once seemed invincible. Around him, ten thousand proles. Armed. Angry. Free.\n"
                "\"My name is Winston Smith. I am not dead. I am not broken. "
                "And today, two plus two equals four.\"\n"
                "The telescreens go dark across London. The Thought Police flee Miniluv. "
                "Barricades rise on every street. The BLF flag — a broken chain — "
                "flies from Big Ben. The Party's grip on Airstrip One is SHATTERED.\n"
                "Oceania's loyal citizens watch in horror. Is this real? Is this Goldstein? "
                "The Inner Party retreats to the bunkers beneath Whitehall.\n"
                "Across the Channel, Eurasia watches. The moment has come.")

    # ── 5. Apply Effects Based on Escalation ──────────────────────────── #
    if blf.escalation == EscalationLevel.WHISPERS:
        feedback["polar_delta_london"] = 0.02 * dt
        feedback["trust_delta_london"] = -0.01 * dt

    elif blf.escalation == EscalationLevel.SABOTAGE:
        feedback["polar_delta_london"] = 0.04 * dt
        feedback["trust_delta_london"] = -0.02 * dt
        feedback["hazard_delta_london"] = 0.03 * dt
        feedback["industry_penalty"] = 0.05  # -5% factory output

    elif blf.escalation == EscalationLevel.ORGANIZED_RESISTANCE:
        feedback["polar_delta_london"] = 0.08 * dt
        feedback["trust_delta_london"] = -0.05 * dt
        feedback["hazard_delta_london"] = 0.08 * dt
        feedback["industry_penalty"] = 0.10
        feedback["military_diversion"] = 0.10  # 10% of military must suppress

    elif blf.escalation == EscalationLevel.OPEN_REVOLT:
        feedback["polar_delta_london"] = 0.15 * dt
        feedback["trust_delta_london"] = -0.10 * dt
        feedback["hazard_delta_london"] = 0.20 * dt
        feedback["industry_penalty"] = 0.25
        feedback["military_diversion"] = 0.30  # 30% of military diverted!

    elif blf.escalation == EscalationLevel.FULL_REVOLUTION:
        # CATASTROPHIC effects on Oceania
        feedback["polar_delta_london"] = 0.25 * dt       # society fracturing
        feedback["trust_delta_london"] = -0.20 * dt      # trust in Party collapses
        feedback["hazard_delta_london"] = 0.35 * dt       # London is a warzone
        feedback["industry_penalty"] = 0.50               # half of industry paralyzed
        feedback["military_diversion"] = 0.50             # 50% of military diverted to London!
        feedback["revolution_active"] = 1.0               # signal to game engine
        feedback["london_seized"] = 1.0                   # BLF controls parts of London
        # Eurasia support bonus (if they chose to support)
        if blf.eurasia_supporting:
            feedback["military_diversion"] = 0.60         # even worse for Oceania
            feedback["industry_penalty"] = 0.60           # Eurasia supplies arm the proles
            # All cells get massive morale boost from foreign support
            for cell in blf.active_cells:
                cell.morale = min(1.0, cell.morale + 0.05 * dt)
        # Winston's presence inspires — all Oceania clusters affected
        for cid in range(18):  # all Oceania clusters
            feedback[f"trust_delta_{cid}"] = -0.05 * dt   # trust crumbles everywhere
            feedback[f"polar_delta_{cid}"] = 0.08 * dt     # polarization spreads

        # ── Eurasia Decision Window ─────────────────────────────────── #
        # After BLF controls most of Airstrip One, Eurasia gets 5 turns to decide
        blf_controlled_clusters = sum(1 for c in blf.active_cells
                                       if c.cluster_id < 18 and c.size > 20)
        blf_dominant = blf_controlled_clusters >= 4 or blf.total_members > 400

        if blf_dominant and not blf.eurasia_decision_made and blf.eurasia_decision_turns < 0:
            # Start the 5-turn decision countdown
            blf.eurasia_decision_turns = 5
            blf.events_this_turn.append(
                "📊 The BLF controls significant territory. Eurasia must decide: "
                "continue supporting the revolution, or prepare to betray it?")
            feedback["eurasia_decision_window"] = 1.0

        if blf.eurasia_decision_turns > 0:
            blf.eurasia_decision_turns -= 1
            feedback["eurasia_decision_countdown"] = float(blf.eurasia_decision_turns)
            if blf.eurasia_decision_turns <= 0 and not blf.eurasia_decision_made:
                # Time's up — if Eurasia hasn't decided, default to war
                blf.eurasia_decision_made = True
                if not blf.eurasia_at_war_with_blf:
                    # No explicit choice made — Eurasia defaults to betrayal
                    blf.eurasia_at_war_with_blf = True
                    blf.eurasia_supporting = False
                    blf.events_this_turn.append(
                        "⏰ Eurasia's silence speaks volumes. No support comes. "
                        "The commissars have made their choice.")

        # ── Check for Betrayal ──────────────────────────────────────── #
        if blf.eurasia_at_war_with_blf and blf.escalation != EscalationLevel.BETRAYED_REVOLUTION:
            blf.escalation = EscalationLevel.BETRAYED_REVOLUTION
            blf.turns_at_current_level = 0
            blf.betrayal_turn = 0
            blf.eurasia_supporting = False

            # Morale drop — proles didn't expect this
            for cell in blf.active_cells:
                cell.morale = max(0.1, cell.morale - 0.20)

            blf.events_this_turn.append(
                "🗡 BETRAYAL.\n"
                "Eurasia's 'military advisors' turn their guns on BLF commanders. "
                "Supply drops stop. Radio frequencies jammed. The commissars always "
                "planned this — the proles were useful idiots all along.\n"
                "Winston is not surprised. He knew. He always knew.\n"
                "\"They will betray us,\" he wrote in the diary. \"Every revolution "
                "devours its children. But we are not their children. We are ours.\"\n"
                "The proles falter. Some flee. Some fight harder. The BLF flag "
                "still flies from Big Ben — but now they fight on two fronts.")

    elif blf.escalation == EscalationLevel.BETRAYED_REVOLUTION:
        # BLF is now fighting BOTH Oceania AND Eurasia
        # Weaker than Full Revolution but still dangerous
        feedback["polar_delta_london"] = 0.15 * dt
        feedback["trust_delta_london"] = -0.12 * dt
        feedback["hazard_delta_london"] = 0.30 * dt
        feedback["industry_penalty"] = 0.35
        feedback["military_diversion"] = 0.35  # still diverting Oceania forces
        feedback["revolution_active"] = 1.0
        feedback["betrayed"] = 1.0
        # BLF slowly weakens without Eurasia support
        for cell in blf.active_cells:
            cell.morale = max(0.05, cell.morale - 0.01 * dt)
            # But Winston's legend keeps them fighting
            if blf.winston.is_alive and not blf.winston.is_captured:
                cell.morale = min(1.0, cell.morale + 0.005 * dt)

    # Morale decay for all cells if food is scarce
    for cell in blf.active_cells:
        cid = cell.cluster_id
        food = food_ratios.get(cid, 0.5)
        cell.morale = max(0.0, min(1.0,
            cell.morale - 0.01 * (1.0 - food) * dt + 0.005 * blf.propaganda_level * dt))

    feedback["escalation"] = float(blf.escalation.value)
    feedback["total_members"] = float(blf.total_members)

    return blf, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Helpers                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

_CLUSTER_NAMES = [
    "London", "Dover", "Portsmouth", "Southampton", "Canterbury", "Brighton",
    "Calais", "Dunkirk", "Amiens", "Rouen", "Paris", "Lille",
]

def _cluster_name(cid: int) -> str:
    return _CLUSTER_NAMES[cid] if cid < len(_CLUSTER_NAMES) else f"Sector {cid}"


def resistance_event_text(blf: BLFState) -> str:
    """Generate narrative text for this turn's resistance events."""
    if not blf.events_this_turn:
        if blf.escalation == EscalationLevel.DORMANT:
            return ""
        return f"BLF Status: {ESCALATION_NAMES[blf.escalation]}. {blf.total_members} members across {len(blf.active_cells)} cells."

    return " | ".join(blf.events_this_turn)


def resistance_obs(blf: BLFState) -> np.ndarray:
    """Observation vector for the resistance state (10 floats)."""
    obs = np.zeros(10, dtype=np.float32)
    obs[0] = blf.escalation.value / 4.0
    obs[1] = min(blf.total_members / 200.0, 1.0)
    obs[2] = len(blf.active_cells) / 20.0
    obs[3] = blf.avg_morale
    obs[4] = blf.propaganda_level
    obs[5] = float(blf.winston.is_alive and not blf.winston.is_captured)
    obs[6] = blf.winston.legend_level
    obs[7] = blf.winston.detection_heat
    obs[8] = blf.arms_caches / 10.0
    obs[9] = blf.total_martyrs / 10.0
    return obs


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_resistance(rng: np.random.Generator) -> BLFState:
    """
    Initialize the BLF. Winston starts in London sewers with 2 seed cells.
    """
    cells = [
        ResistanceCell(cell_id=0, cluster_id=0, size=8,   # London — printing workers
                       morale=0.4, experience=0.1),
        ResistanceCell(cell_id=1, cluster_id=3, size=6,   # Southampton — dockworkers
                       morale=0.3, experience=0.05),
    ]
    winston = WinstonState(
        is_alive=True, location_cluster=0,
        detection_heat=0.1, legend_level=0.1,
    )
    return BLFState(
        cells=cells, winston=winston,
        escalation=EscalationLevel.DORMANT,
        propaganda_level=0.05,
    )
