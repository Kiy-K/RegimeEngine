"""
intel_system.py — Complete Intelligence, Fog of War & Espionage system.

═══════════════════════════════════════════════════════════════════════════════
FOG OF WAR

  Each faction has a VISIBILITY MAP: per-cluster visibility [0, 1].
    1.0 = full intelligence (own territory, heavily scouted)
    0.5 = partial (adjacent to own territory, some recon)
    0.0 = blind (deep enemy territory, no agents)

  What you CAN'T see when visibility < threshold:
    - Exact enemy troop counts (only "light/moderate/heavy")
    - Enemy resource stockpiles (only "adequate/strained/critical")
    - Enemy production queues
    - Submarine positions (always hidden unless detected)
    - Invasion plans (hidden until CROSSING phase)
    - BLF cell locations (hidden from Oceania unless Thought Police find them)

  What you CAN always see:
    - Your own territory (full detail)
    - Bombing damage (visible aftermath)
    - Naval battles you participated in
    - Propaganda broadcasts (public by nature)

═══════════════════════════════════════════════════════════════════════════════
INTELLIGENCE SOURCES

  SIGINT (Signals Intelligence):
    - Radio intercepts from enemy communications
    - Code-breaking (like Bletchley Park / Enigma)
    - Reveals enemy orders 1-2 turns in advance if codes are broken
    - Counter: changing codes (costs efficiency for a few turns)

  HUMINT (Human Intelligence):
    - Spy rings embedded in enemy territory
    - Provide detailed reports on specific clusters
    - Risk of detection → captured spies → false intel / doubled agents
    - Double agents: feed enemy false information

  RECON:
    - Aerial reconnaissance (photo recon aircraft)
    - Naval patrols (surface + submarine)
    - Radar networks (detect air raids, naval movements)
    - Satellite (future, not in 1984 setting)

  OSINT (Open Source):
    - Propaganda analysis (what the enemy says reveals what they fear)
    - Refugee reports (people fleeing reveal conditions)
    - Trade patterns (supply ship routes reveal priorities)

═══════════════════════════════════════════════════════════════════════════════
ESPIONAGE ACTIONS

  PLANT_SPY: Insert agent into enemy cluster (takes 5-10 turns to establish)
  EXTRACT_SPY: Pull endangered agent out before capture
  CODE_BREAK: Invest in breaking enemy communications
  CHANGE_CODES: Reset own communications (costs 3 turns of reduced coordination)
  COUNTER_INTEL: Hunt for enemy moles in own territory
  DECEPTION: Plant false intelligence (fake armies, radio deception)
  TURN_AGENT: Attempt to double an enemy spy (very risky, very rewarding)

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION

  1. Spies take time to establish (5-10 turns) — no instant intel
  2. Spy rings can be detected and rolled up (losing everything)
  3. Code-breaking degrades if enemy changes codes
  4. False intel from double agents can cause catastrophic decisions
  5. Counter-intel diverts resources from offense
  6. Recon aircraft can be shot down (losing intel + plane)
  7. Over-reliance on SIGINT fails when codes change
  8. Compartmentalization means losing one spy doesn't lose all
  9. Intel reports have RELIABILITY ratings — not all intel is true
  10. Deception operations can be detected, revealing YOUR plans instead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════ #
# Enums                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class IntelSource(Enum):
    SIGINT  = 0   # Signals intelligence (radio intercepts, code-breaking)
    HUMINT  = 1   # Human intelligence (spy rings, agents)
    RECON   = 2   # Aerial/naval reconnaissance
    OSINT   = 3   # Open source (propaganda analysis, refugees)
    DECEPTION = 4  # False intel planted by enemy


class IntelReliability(Enum):
    CONFIRMED   = 0   # Multiple sources agree, high confidence
    PROBABLE    = 1   # Single reliable source
    POSSIBLE    = 2   # Unconfirmed, single source
    DOUBTFUL    = 3   # Conflicting reports
    FALSE       = 4   # Planted by enemy (appears real until verified)


class IntelClassification(Enum):
    TACTICAL    = 0   # Unit positions, immediate threats
    OPERATIONAL = 1   # Enemy plans for next few turns
    STRATEGIC   = 2   # Long-term enemy capabilities, production, morale


# ═══════════════════════════════════════════════════════════════════════════ #
# Intel Report                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class IntelReport:
    """A single intelligence report received by a faction."""
    report_id: int
    turn_received: int
    source: IntelSource
    reliability: IntelReliability
    classification: IntelClassification
    target_cluster: int              # what cluster this intel is about
    content: str                     # human-readable summary
    # Specific intel values (may be inaccurate if reliability is low)
    enemy_military_estimate: Optional[float] = None   # [0,1] estimated military presence
    enemy_resource_estimate: Optional[float] = None    # [0,1] estimated resources
    enemy_production_info: Optional[str] = None        # what they're building
    enemy_plans: Optional[str] = None                  # intended actions
    is_false: bool = False           # TRUE if this is planted deception
    expires_turn: int = 0            # intel goes stale after this turn


# ═══════════════════════════════════════════════════════════════════════════ #
# Spy Ring                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class SpyRing:
    """A network of agents operating in enemy territory."""
    ring_id: int
    faction_id: int                  # who owns this ring
    target_cluster: int              # where agents are operating
    agents: int = 3                  # number of agents in the ring
    establishment_turns: int = 0     # turns since planted (needs 5+ to be effective)
    detection_risk: float = 0.05     # [0,1] chance of being discovered per turn
    effectiveness: float = 0.0       # [0,1] grows over time as agents embed
    is_compromised: bool = False     # enemy knows about this ring
    is_doubled: bool = False         # enemy has turned our agents against us
    cover_strength: float = 0.7      # [0,1] how good is the cover story
    last_report_turn: int = -1

    @property
    def is_active(self) -> bool:
        return not self.is_compromised and self.agents > 0

    @property
    def is_established(self) -> bool:
        return self.establishment_turns >= 5 and self.is_active

    def copy(self) -> "SpyRing":
        return SpyRing(
            ring_id=self.ring_id, faction_id=self.faction_id,
            target_cluster=self.target_cluster, agents=self.agents,
            establishment_turns=self.establishment_turns,
            detection_risk=self.detection_risk,
            effectiveness=self.effectiveness,
            is_compromised=self.is_compromised,
            is_doubled=self.is_doubled,
            cover_strength=self.cover_strength,
            last_report_turn=self.last_report_turn,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Code-Breaking                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class CodeBreaking:
    """SIGINT / cryptanalysis state for a faction."""
    progress: float = 0.0            # [0,1] how close to breaking enemy codes
    is_broken: bool = False          # have we cracked their current codes?
    turns_since_break: int = 0       # codes degrade as enemy adapts
    own_code_strength: float = 0.7   # [0,1] how secure our own codes are
    code_change_cooldown: int = 0    # turns of reduced coordination after code change
    cryptanalysts: int = 50          # personnel dedicated to code-breaking
    # Bletchley Park / Colossus equivalent
    machines_level: int = 1          # 1=manual, 2=basic machine, 3=advanced (Colossus)

    @property
    def intercept_quality(self) -> float:
        """Quality of intercepted enemy communications."""
        if not self.is_broken:
            return 0.1  # can still get some traffic analysis
        freshness = max(0.0, 1.0 - self.turns_since_break * 0.05)
        machine_bonus = self.machines_level * 0.15
        return min(1.0, 0.5 + freshness * 0.3 + machine_bonus)


# ═══════════════════════════════════════════════════════════════════════════ #
# Radar Network                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class RadarNetwork:
    """Ground-based radar + observer corps detection network."""
    stations: Dict[int, float] = field(default_factory=dict)  # cluster_id → effectiveness [0,1]
    overall_coverage: float = 0.5    # [0,1] network completeness
    jamming_resistance: float = 0.6  # [0,1] resistance to enemy ECM

    def detection_at(self, cluster_id: int) -> float:
        """Detection capability at a specific cluster."""
        base = self.stations.get(cluster_id, 0.1)
        return min(1.0, base * self.overall_coverage * self.jamming_resistance)


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Intelligence State                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionIntelState:
    """Complete intelligence state for one faction."""
    faction_id: int
    # Visibility map: how much we can see of each cluster [0, 1]
    visibility: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(12, dtype=np.float64))
    # Intel collection assets
    spy_rings: List[SpyRing] = field(default_factory=list)
    code_breaking: CodeBreaking = field(default_factory=CodeBreaking)
    radar: RadarNetwork = field(default_factory=RadarNetwork)
    # Intel reports (recent, actionable)
    reports: List[IntelReport] = field(default_factory=list)
    # Counter-intelligence
    counter_intel_strength: float = 0.5  # [0,1] ability to find enemy spies
    security_level: float = 0.5          # [0,1] how tight is operational security
    # Deception
    active_deceptions: List[Dict] = field(default_factory=list)  # ongoing deception ops
    # Stats
    next_report_id: int = 0
    spies_lost: int = 0
    codes_broken_count: int = 0

    def copy(self) -> "FactionIntelState":
        return FactionIntelState(
            faction_id=self.faction_id,
            visibility=self.visibility.copy(),
            spy_rings=[s.copy() for s in self.spy_rings],
            code_breaking=CodeBreaking(
                progress=self.code_breaking.progress,
                is_broken=self.code_breaking.is_broken,
                turns_since_break=self.code_breaking.turns_since_break,
                own_code_strength=self.code_breaking.own_code_strength,
                code_change_cooldown=self.code_breaking.code_change_cooldown,
                cryptanalysts=self.code_breaking.cryptanalysts,
                machines_level=self.code_breaking.machines_level,
            ),
            radar=RadarNetwork(
                stations=dict(self.radar.stations),
                overall_coverage=self.radar.overall_coverage,
                jamming_resistance=self.radar.jamming_resistance,
            ),
            reports=list(self.reports[-20:]),  # keep last 20 reports
            counter_intel_strength=self.counter_intel_strength,
            security_level=self.security_level,
            active_deceptions=list(self.active_deceptions),
            next_report_id=self.next_report_id,
            spies_lost=self.spies_lost,
            codes_broken_count=self.codes_broken_count,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Intel World                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class IntelWorld:
    """Complete intelligence state for all factions."""
    factions: Dict[int, FactionIntelState] = field(default_factory=dict)
    step: int = 0

    def get(self, faction_id: int) -> FactionIntelState:
        return self.factions[faction_id]

    def copy(self) -> "IntelWorld":
        return IntelWorld(
            factions={fid: fs.copy() for fid, fs in self.factions.items()},
            step=self.step,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Fog of War — Visibility Calculation                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_visibility(
    faction_intel: FactionIntelState,
    cluster_owners: Dict[int, int],
    adjacency_pairs: List[Tuple[int, int]],
    n_clusters: int,
) -> NDArray[np.float64]:
    """
    Compute what a faction can see. Returns visibility map [0, 1] per cluster.

    Sources of visibility:
      - Own territory: 1.0 (full visibility)
      - Adjacent to own territory: 0.4 (border observation)
      - Established spy ring: +0.3 per ring
      - Broken enemy codes (SIGINT): +0.2 to all enemy clusters
      - Radar coverage: +0.2 per station in range
      - Aerial recon: +0.15 per recon flight
      - Base fog: 0.05 (OSINT — refugees, propaganda analysis)
    """
    vis = np.full(n_clusters, 0.05, dtype=np.float64)  # base OSINT

    fid = faction_intel.faction_id

    # Own territory = full visibility
    for cid, owner in cluster_owners.items():
        if owner == fid and cid < n_clusters:
            vis[cid] = 1.0

    # Adjacent to own territory = border observation
    for a, b in adjacency_pairs:
        if a < n_clusters and b < n_clusters:
            if cluster_owners.get(a) == fid and cluster_owners.get(b) != fid:
                vis[b] = max(vis[b], 0.4)
            if cluster_owners.get(b) == fid and cluster_owners.get(a) != fid:
                vis[a] = max(vis[a], 0.4)

    # Spy rings in enemy territory
    for ring in faction_intel.spy_rings:
        if ring.is_established and ring.target_cluster < n_clusters:
            bonus = 0.3 * ring.effectiveness
            if ring.is_doubled:
                bonus = -0.1  # doubled agents give FALSE visibility (worse than nothing)
            vis[ring.target_cluster] = min(1.0, vis[ring.target_cluster] + bonus)

    # SIGINT (broken codes reveal all enemy clusters somewhat)
    if faction_intel.code_breaking.is_broken:
        sigint_bonus = 0.2 * faction_intel.code_breaking.intercept_quality
        for cid, owner in cluster_owners.items():
            if owner != fid and cid < n_clusters:
                vis[cid] = min(1.0, vis[cid] + sigint_bonus)

    # Radar coverage
    for cid, eff in faction_intel.radar.stations.items():
        if cid < n_clusters:
            vis[cid] = min(1.0, vis[cid] + 0.2 * eff)

    return np.clip(vis, 0.0, 1.0)


def get_faction_visibility(
    intel_world: IntelWorld,
    faction_id: int,
    cluster_owners: Dict[int, int],
    n_clusters: int = 12,
) -> NDArray[np.float64]:
    """Get current visibility map for a faction."""
    if faction_id not in intel_world.factions:
        return np.full(n_clusters, 0.05, dtype=np.float64)

    # Adjacency pairs for 32-sector All British Isles + Central France map
    adjacency = [
        # Oceania — South England
        (0, 1), (0, 4), (0, 3), (0, 5), (1, 4), (1, 5), (2, 3), (2, 5), (3, 5),
        # South↔SW+Wales
        (3, 6), (6, 7), (6, 8),
        # South↔Midlands
        (0, 9), (4, 13), (3, 9),
        # SW/Wales↔Midlands
        (8, 9), (6, 9),
        # Midlands internal + North
        (9, 10), (10, 11), (10, 12), (11, 12),
        # East Anglia
        (0, 13), (13, 12),
        # Midlands/North↔Scotland
        (12, 14), (11, 15),
        # Scotland internal
        (14, 15),
        # Ireland
        (16, 17),
        # Ireland↔mainland (sea)
        (11, 16), (15, 17), (6, 16),
        # Eurasia — Channel front
        (18, 19), (19, 20), (20, 21),
        # Channel↔N France
        (18, 22), (18, 24), (19, 24), (20, 23), (22, 23), (22, 24),
        # N France↔Benelux
        (24, 25), (19, 25), (25, 26),
        # N France↔Central
        (22, 27), (23, 27), (27, 28), (28, 29),
        # Central↔Atlantic
        (21, 30), (28, 31), (29, 31),
        # Atlantic internal
        (30, 31),
        # Cross-Channel (sea)
        (1, 18), (2, 19), (5, 18), (5, 20), (7, 21),
        # North Sea (sea)
        (11, 26), (14, 26),
        # Bay of Biscay (sea)
        (7, 30), (7, 31),
    ]
    return compute_visibility(
        intel_world.factions[faction_id], cluster_owners, adjacency, n_clusters)


# ═══════════════════════════════════════════════════════════════════════════ #
# Intel Step                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_intelligence(
    intel_world: IntelWorld,
    cluster_data: np.ndarray,
    cluster_owners: Dict[int, int],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[IntelWorld, Dict[str, Any]]:
    """Advance intelligence system by one step."""
    feedback: Dict[str, Any] = {}

    for fid, fis in intel_world.factions.items():
        enemy_id = 1 - fid
        events = []

        # ── 1. Spy ring operations ─────────────────────────────────────── #
        for ring in fis.spy_rings:
            if not ring.is_active:
                continue
            ring.establishment_turns += 1
            # Effectiveness grows as agents embed deeper
            if ring.establishment_turns >= 5:
                ring.effectiveness = min(0.9,
                    0.3 + 0.05 * (ring.establishment_turns - 5))

            # Detection check by enemy counter-intel
            enemy_fis = intel_world.factions.get(enemy_id)
            if enemy_fis:
                detect_chance = (enemy_fis.counter_intel_strength * 0.03
                                 * (1.0 - ring.cover_strength) * dt)
                detect_chance *= (1.0 + 0.02 * ring.agents)  # more agents = more risk
                if rng.random() < detect_chance:
                    # Ring compromised!
                    if rng.random() < 0.4:
                        # Enemy doubles the agents instead of arresting
                        ring.is_doubled = True
                        events.append(f"⚠ Spy ring in C{ring.target_cluster} may be compromised.")
                    else:
                        ring.is_compromised = True
                        ring.agents = 0
                        fis.spies_lost += 1
                        events.append(f"Spy ring DESTROYED in C{ring.target_cluster}. Agents captured.")

            # Generate intel report if established
            if ring.is_established and ring.last_report_turn < intel_world.step:
                tc = ring.target_cluster
                if tc < len(cluster_data):
                    reliability = IntelReliability.PROBABLE
                    is_false = False
                    mil_est = float(cluster_data[tc, 3])
                    res_est = float(cluster_data[tc, 2])

                    if ring.is_doubled:
                        # Double agent feeds false intel!
                        reliability = IntelReliability.FALSE
                        is_false = True
                        mil_est = rng.uniform(0.1, 0.9)  # random garbage
                        res_est = rng.uniform(0.1, 0.9)

                    # Add noise based on effectiveness
                    noise = (1.0 - ring.effectiveness) * 0.2
                    mil_est += rng.uniform(-noise, noise)
                    res_est += rng.uniform(-noise, noise)

                    report = IntelReport(
                        report_id=fis.next_report_id,
                        turn_received=intel_world.step,
                        source=IntelSource.HUMINT,
                        reliability=reliability,
                        classification=IntelClassification.TACTICAL,
                        target_cluster=tc,
                        content=f"Agent reports from C{tc}: military {'heavy' if mil_est > 0.6 else 'moderate' if mil_est > 0.3 else 'light'}, "
                                f"supplies {'adequate' if res_est > 0.5 else 'strained' if res_est > 0.2 else 'critical'}",
                        enemy_military_estimate=float(np.clip(mil_est, 0, 1)),
                        enemy_resource_estimate=float(np.clip(res_est, 0, 1)),
                        is_false=is_false,
                        expires_turn=intel_world.step + 5,
                    )
                    fis.reports.append(report)
                    fis.next_report_id += 1
                    ring.last_report_turn = intel_world.step

        # ── 2. Code-breaking progress ──────────────────────────────────── #
        cb = fis.code_breaking
        if not cb.is_broken:
            progress_rate = 0.01 * (cb.cryptanalysts / 100.0) * cb.machines_level * dt
            cb.progress = min(1.0, cb.progress + progress_rate)
            if cb.progress >= 0.8 and rng.random() < 0.1 * dt:
                cb.is_broken = True
                cb.turns_since_break = 0
                fis.codes_broken_count += 1
                events.append("🔓 ENEMY CODES BROKEN! SIGINT now active.")
        else:
            cb.turns_since_break += 1
            # Enemy may change codes (probability increases over time)
            if cb.turns_since_break > 10 and rng.random() < 0.05 * dt:
                cb.is_broken = False
                cb.progress = 0.3  # partial knowledge retained
                events.append("Enemy changed codes. SIGINT degraded.")

        # Code change cooldown
        if cb.code_change_cooldown > 0:
            cb.code_change_cooldown -= 1

        # ── 3. Update visibility ───────────────────────────────────────── #
        fis.visibility = get_faction_visibility(intel_world, fid, cluster_owners)

        # ── 4. Expire old reports ──────────────────────────────────────── #
        fis.reports = [r for r in fis.reports if r.expires_turn > intel_world.step][-20:]

        feedback[f"faction_{fid}_events"] = events

    intel_world.step += 1
    return intel_world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Intel Actions                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class IntelAction(Enum):
    NOOP            = 0
    PLANT_SPY       = 1   # Insert spy ring into enemy cluster
    EXTRACT_SPY     = 2   # Pull endangered ring out
    CODE_BREAK      = 3   # Invest in cryptanalysis
    CHANGE_CODES    = 4   # Reset own codes (3-turn coordination penalty)
    COUNTER_INTEL   = 5   # Hunt for enemy spies
    DECEPTION       = 6   # Plant false intelligence
    TURN_AGENT      = 7   # Try to double an enemy spy
    BOOST_RADAR     = 8   # Improve radar coverage at a cluster


def apply_intel_action(
    intel_world: IntelWorld,
    faction_id: int,
    action_type: int,
    target_cluster: int,
    rng: np.random.Generator,
) -> Tuple[IntelWorld, float]:
    """Apply an intelligence action. Returns (updated_world, reward)."""
    try:
        action = IntelAction(action_type)
    except ValueError:
        return intel_world, 0.0

    fis = intel_world.factions.get(faction_id)
    if fis is None:
        return intel_world, 0.0

    if action == IntelAction.NOOP:
        return intel_world, 0.0

    elif action == IntelAction.PLANT_SPY:
        # Check if we already have a ring there
        existing = [r for r in fis.spy_rings if r.target_cluster == target_cluster and r.is_active]
        if existing:
            return intel_world, -0.1  # already have agents there
        if len([r for r in fis.spy_rings if r.is_active]) >= 5:
            return intel_world, -0.1  # max 5 active rings

        new_ring = SpyRing(
            ring_id=len(fis.spy_rings),
            faction_id=faction_id,
            target_cluster=target_cluster,
            agents=3 + rng.integers(0, 3),
            cover_strength=0.5 + rng.uniform(0, 0.3),
        )
        fis.spy_rings.append(new_ring)
        return intel_world, 0.3

    elif action == IntelAction.EXTRACT_SPY:
        for ring in fis.spy_rings:
            if ring.target_cluster == target_cluster and ring.is_active:
                ring.is_compromised = True  # extracted = deactivated
                ring.agents = 0
                return intel_world, 0.1  # saved the agents' lives
        return intel_world, -0.05

    elif action == IntelAction.CODE_BREAK:
        fis.code_breaking.cryptanalysts = min(200, fis.code_breaking.cryptanalysts + 20)
        fis.code_breaking.progress = min(1.0, fis.code_breaking.progress + 0.05)
        return intel_world, 0.2

    elif action == IntelAction.CHANGE_CODES:
        if fis.code_breaking.code_change_cooldown > 0:
            return intel_world, -0.1
        fis.code_breaking.own_code_strength = min(1.0, fis.code_breaking.own_code_strength + 0.2)
        fis.code_breaking.code_change_cooldown = 3
        # This also breaks enemy's SIGINT on us
        enemy_id = 1 - faction_id
        if enemy_id in intel_world.factions:
            ecb = intel_world.factions[enemy_id].code_breaking
            if ecb.is_broken:
                ecb.is_broken = False
                ecb.progress = 0.2
        return intel_world, 0.2

    elif action == IntelAction.COUNTER_INTEL:
        fis.counter_intel_strength = min(1.0, fis.counter_intel_strength + 0.05)
        # Sweep for enemy spies
        enemy_id = 1 - faction_id
        if enemy_id in intel_world.factions:
            for ring in intel_world.factions[enemy_id].spy_rings:
                if ring.is_active and ring.target_cluster in [
                    cid for cid, o in {} .items() if o == faction_id]:  # our territory
                    # Enhanced detection this turn
                    if rng.random() < fis.counter_intel_strength * 0.1:
                        ring.is_compromised = True
                        ring.agents = 0
                        return intel_world, 0.5  # caught a spy!
        return intel_world, 0.1

    elif action == IntelAction.DECEPTION:
        # Plant false intelligence that enemy will pick up
        enemy_id = 1 - faction_id
        if enemy_id in intel_world.factions:
            false_report = IntelReport(
                report_id=intel_world.factions[enemy_id].next_report_id,
                turn_received=intel_world.step,
                source=IntelSource.DECEPTION,
                reliability=IntelReliability.PROBABLE,  # looks real!
                classification=IntelClassification.OPERATIONAL,
                target_cluster=target_cluster,
                content=f"SIGINT intercept suggests major buildup at C{target_cluster}",
                enemy_military_estimate=rng.uniform(0.6, 0.9),  # fake heavy presence
                is_false=True,
                expires_turn=intel_world.step + 8,
            )
            intel_world.factions[enemy_id].reports.append(false_report)
            intel_world.factions[enemy_id].next_report_id += 1
        return intel_world, 0.3

    elif action == IntelAction.BOOST_RADAR:
        if target_cluster not in fis.radar.stations:
            fis.radar.stations[target_cluster] = 0.3
        else:
            fis.radar.stations[target_cluster] = min(1.0,
                fis.radar.stations[target_cluster] + 0.15)
        fis.radar.overall_coverage = min(1.0, fis.radar.overall_coverage + 0.03)
        return intel_world, 0.15

    return intel_world, 0.0


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_intelligence(
    faction_ids: List[int],
    cluster_owners: Dict[int, int],
    n_clusters: int = 12,
    rng: np.random.Generator = None,
) -> IntelWorld:
    """Initialize intelligence state for all factions."""
    factions = {}

    for fid in faction_ids:
        # Base visibility: own territory = 1.0, everything else = low
        vis = np.full(n_clusters, 0.05, dtype=np.float64)
        for cid, owner in cluster_owners.items():
            if owner == fid and cid < n_clusters:
                vis[cid] = 1.0

        # Radar stations at key positions
        radar_stations = {}
        if fid == 0:  # Oceania — Chain Home radar along south coast
            radar_stations = {1: 0.8, 5: 0.6, 0: 0.5, 2: 0.7}
        elif fid == 1:  # Eurasia — Freya radar at Channel ports
            radar_stations = {6: 0.7, 7: 0.6, 10: 0.5}

        # Starting code-breaking progress
        cb = CodeBreaking(
            progress=0.2 + (rng.uniform(0, 0.2) if rng else 0.1),
            machines_level=2 if fid == 0 else 1,  # Oceania has better machines (Colossus)
            cryptanalysts=80 if fid == 0 else 60,
        )

        factions[fid] = FactionIntelState(
            faction_id=fid,
            visibility=vis,
            code_breaking=cb,
            radar=RadarNetwork(
                stations=radar_stations,
                overall_coverage=0.5 + (0.1 if fid == 0 else 0.0),  # Oceania has Chain Home
                jamming_resistance=0.6,
            ),
            counter_intel_strength=0.5 + (0.15 if fid == 0 else 0.0),  # Thought Police bonus
            security_level=0.6,
        )

    return IntelWorld(factions=factions)


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def intel_obs(
    intel_world: IntelWorld,
    faction_id: int,
    n_clusters: int = 12,
) -> NDArray[np.float32]:
    """
    Observation vector for intelligence state.

    Per cluster (3 floats × n_clusters):
      - visibility level [0, 1]
      - has_spy_ring {0, 1}
      - latest_intel_reliability (0=confirmed, 1=false)

    Global (10 floats):
      - code_breaking_progress, codes_broken, intercept_quality
      - counter_intel_strength, security_level
      - active_spy_rings, spies_lost
      - radar_coverage
      - active_reports_count
      - deception_ops_active

    Total: 3 × n_clusters + 10
    """
    n_per = 3
    n_global = 10
    obs = np.zeros(n_per * n_clusters + n_global, dtype=np.float32)

    fis = intel_world.factions.get(faction_id)
    if fis is None:
        return obs

    for i in range(n_clusters):
        o = i * n_per
        obs[o] = fis.visibility[i] if i < len(fis.visibility) else 0.05
        obs[o + 1] = float(any(r.target_cluster == i and r.is_active for r in fis.spy_rings))
        # Latest intel reliability for this cluster
        cluster_reports = [r for r in fis.reports if r.target_cluster == i]
        if cluster_reports:
            obs[o + 2] = cluster_reports[-1].reliability.value / 4.0
        else:
            obs[o + 2] = 1.0  # no intel = worst reliability

    go = n_per * n_clusters
    obs[go + 0] = fis.code_breaking.progress
    obs[go + 1] = float(fis.code_breaking.is_broken)
    obs[go + 2] = fis.code_breaking.intercept_quality
    obs[go + 3] = fis.counter_intel_strength
    obs[go + 4] = fis.security_level
    obs[go + 5] = len([r for r in fis.spy_rings if r.is_active]) / 5.0
    obs[go + 6] = fis.spies_lost / 10.0
    obs[go + 7] = fis.radar.overall_coverage
    obs[go + 8] = len(fis.reports) / 20.0
    obs[go + 9] = len(fis.active_deceptions) / 3.0

    return np.clip(obs, 0.0, 1.0)


def intel_obs_size(n_clusters: int = 12) -> int:
    return 3 * n_clusters + 10
