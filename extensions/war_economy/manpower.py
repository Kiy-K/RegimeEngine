"""
manpower.py — Sophisticated workforce & conscription system.

═══════════════════════════════════════════════════════════════════════════════
POPULATION POOLS (per cluster)

  Total Population → Working Age (62%) → Available Workforce
                                       → Conscripted (military)
                                       → In Training (pipeline)
                                       → Disabled / Casualties
                   → Children (20%)    → Future workforce (delayed)
                   → Elderly (18%)     → Non-productive (consume food)

  Available Workforce is allocated across:
    - 7 economic sectors (Agriculture, Mining, Energy, Heavy Industry, etc.)
    - Military service (via conscription)
    - Training pipeline (becoming skilled workers or soldiers)
    - Unemployed (idle, generates unrest)

═══════════════════════════════════════════════════════════════════════════════
CONSCRIPTION LAWS (faction-level, regime-dependent)

  Each law sets a max_conscription_rate (fraction of working-age that can serve):

  VOLUNTEER_ONLY       — 2.5%  (peacetime democracies)
  LIMITED_CONSCRIPTION — 5.0%  (early wartime, moderate regimes)
  EXTENSIVE_DRAFT      — 10%   (total war, authoritarian leaning)
  SERVICE_BY_REQUIREMENT — 15% (WW2-level mobilisation)
  ALL_ADULTS_SERVE     — 25%   (desperate last stand)
  SCRAPING_THE_BARREL  — 35%   (children + elderly, destroys economy)

  Constraints:
    - Higher conscription → more soldiers but cripples civilian economy
    - Regime type gates which laws are available
    - Changing law takes time (transition_steps) and costs political capital
    - Each step UP generates unrest proportional to jump size
    - Economic health limits effective conscription (starving people can't fight)

═══════════════════════════════════════════════════════════════════════════════
TRAINING PIPELINE

  Workers don't instantly become skilled. Training takes time:
    - UNTRAINED → BASIC (10 steps) → SKILLED (20 steps) → EXPERT (40 steps)
    - Military: CIVILIAN → RECRUIT (15 steps) → REGULAR (30 steps) → VETERAN (60 steps)

  Each skill level provides productivity multipliers:
    - Untrained: 0.4x sector output
    - Basic: 0.7x
    - Skilled: 1.0x
    - Expert: 1.3x

  Anti-exploitation: can't instantly mass-produce experts.

═══════════════════════════════════════════════════════════════════════════════
ANTI-EXPLOITATION MECHANISMS

  1. Training takes real time — no instant expert workers
  2. Conscription damages economy — soldiers leave production
  3. Higher conscription = more unrest + polarization
  4. Regime type limits available conscription laws
  5. Law changes take transition time + political cost
  6. Disabled/casualties are permanent population loss
  7. Elderly/children consume food but don't produce
  8. Skill degradation — workers lose skills if reassigned to different sector
  9. Brain drain — high conscription causes skilled workers to flee
  10. Morale collapse — overconscription destroys military effectiveness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .war_economy_state import EconSector, N_SECTORS


# ═══════════════════════════════════════════════════════════════════════════ #
# Conscription Laws — 15 tiers based on real-world military service policies  #
# ═══════════════════════════════════════════════════════════════════════════ #
#
# Real-world basis:
#   0-3: Peacetime volunteer systems (US, UK, modern France, Japan)
#   4-6: Selective service / national service (Israel, South Korea, Switzerland)
#   7-9: Wartime mobilisation (WW1/WW2 major powers, USSR, Nazi Germany)
#   10-12: Total war mobilisation (late WW2 Japan, Volkssturm, Soviet 1941)
#   13-14: Civilisation-ending desperation (siege of Berlin, Okinawa civilians)
#
# Each law has 8 dimensions:
#   - max_conscription_rate: fraction of working-age pop in uniform
#   - economy_penalty: factory output reduction
#   - unrest_per_step: political instability generated
#   - training_quality: how well conscripts are trained (0-1)
#   - min_age / max_age: age range eligible
#   - gender_policy: who can be drafted
#   - desertion_rate: fraction of conscripts who desert per step
#   - political_cost: one-time cost to enact this law
#
# Anti-exploitation: higher conscription doesn't just give more soldiers —
# it destroys the economy, causes unrest, produces untrained cannon fodder
# with high desertion, and requires specific regime types to sustain.

class ConscriptionLaw(Enum):
    """15-tier conscription system based on real-world military service policies."""
    # ── Peacetime (tiers 0-3) ─────────────────────────────────────────── #
    DISARMED_NATION        = 0   # Post-war Japan Art. 9 — no military
    VOLUNTEER_ONLY         = 1   # Professional army (US/UK model)
    PROFESSIONAL_ARMY      = 2   # Larger volunteer force + reserves
    TOKEN_NATIONAL_SERVICE = 3   # 3-month symbolic service (some EU nations)

    # ── Selective Service (tiers 4-6) ─────────────────────────────────── #
    SELECTIVE_SERVICE      = 4   # Lottery draft, many exemptions (US Vietnam-era)
    UNIVERSAL_MALE_SERVICE = 5   # All males serve 12-24 months (Israel, SK, Switzerland)
    UNIVERSAL_SERVICE      = 6   # All citizens serve regardless of gender (Israel model)

    # ── Wartime Mobilisation (tiers 7-9) ──────────────────────────────── #
    WARTIME_DRAFT          = 7   # Mass call-up, limited exemptions (US WW2)
    GENERAL_MOBILISATION   = 8   # Full national mobilisation (France 1914, USSR 1941)
    TOTAL_MOBILISATION     = 9   # All able-bodied, minimal exemptions (Germany 1943)

    # ── Total War (tiers 10-12) ───────────────────────────────────────── #
    WAR_ECONOMY_DRAFT      = 10  # Workers conscripted from factories (Germany 1944)
    SERVICE_BY_REQUIREMENT = 11  # Everyone assigned role by state (USSR model)
    ALL_ADULTS_SERVE       = 12  # 16-60 age range, no exemptions (Volkssturm)

    # ── Desperation (tiers 13-14) ─────────────────────────────────────── #
    SCRAPING_THE_BARREL    = 13  # Children + elderly armed (Hitler Youth, Okinawa)
    NATION_IN_ARMS         = 14  # Entire population = military (siege of Berlin)


N_CONSCRIPTION_LAWS = len(ConscriptionLaw)

# ── Conscription Law Properties ──────────────────────────────────────────── #
# Stored as a dict-of-dicts keyed by ConscriptionLaw → property → value.
# This is the "stat block" for each law.

@dataclass(frozen=True)
class ConscriptionLawStats:
    """Immutable properties of a conscription law."""
    max_rate: float           # max fraction of working-age pop in military
    economy_penalty: float    # factory output penalty [0, 1]
    unrest_per_step: float    # political instability generated per step
    training_quality: float   # base training quality [0, 1] (higher = better troops)
    min_draft_age: int        # minimum age for conscription
    max_draft_age: int        # maximum age for conscription
    desertion_rate: float     # fraction of conscripts who desert per step
    political_cost: float     # political capital spent to enact this law
    gender_all: bool          # True = all genders, False = males only
    research_penalty: float   # penalty to tech/research from brain drain
    morale_modifier: float    # modifier to military morale (-1 to +1)
    description: str          # human-readable description


LAW_STATS: Dict[ConscriptionLaw, ConscriptionLawStats] = {
    # ── Peacetime ──────────────────────────────────────────────────────── #
    ConscriptionLaw.DISARMED_NATION: ConscriptionLawStats(
        max_rate=0.005, economy_penalty=0.00, unrest_per_step=0.000,
        training_quality=0.90, min_draft_age=18, max_draft_age=45,
        desertion_rate=0.000, political_cost=0.0, gender_all=False,
        research_penalty=0.00, morale_modifier=0.0,
        description="Demilitarised state. Token self-defense force only.",
    ),
    ConscriptionLaw.VOLUNTEER_ONLY: ConscriptionLawStats(
        max_rate=0.020, economy_penalty=0.00, unrest_per_step=0.000,
        training_quality=0.95, min_draft_age=18, max_draft_age=45,
        desertion_rate=0.001, political_cost=0.0, gender_all=False,
        research_penalty=0.00, morale_modifier=0.10,
        description="Professional all-volunteer force. High quality, low quantity.",
    ),
    ConscriptionLaw.PROFESSIONAL_ARMY: ConscriptionLawStats(
        max_rate=0.035, economy_penalty=0.01, unrest_per_step=0.000,
        training_quality=0.90, min_draft_age=18, max_draft_age=50,
        desertion_rate=0.001, political_cost=5.0, gender_all=False,
        research_penalty=0.00, morale_modifier=0.08,
        description="Expanded professional force with trained reserves.",
    ),
    ConscriptionLaw.TOKEN_NATIONAL_SERVICE: ConscriptionLawStats(
        max_rate=0.050, economy_penalty=0.02, unrest_per_step=0.001,
        training_quality=0.75, min_draft_age=18, max_draft_age=45,
        desertion_rate=0.002, political_cost=10.0, gender_all=False,
        research_penalty=0.01, morale_modifier=0.05,
        description="Short mandatory service. Builds reserves, mild economic impact.",
    ),

    # ── Selective Service ──────────────────────────────────────────────── #
    ConscriptionLaw.SELECTIVE_SERVICE: ConscriptionLawStats(
        max_rate=0.070, economy_penalty=0.04, unrest_per_step=0.003,
        training_quality=0.70, min_draft_age=18, max_draft_age=45,
        desertion_rate=0.005, political_cost=15.0, gender_all=False,
        research_penalty=0.02, morale_modifier=0.00,
        description="Lottery-based draft with deferments. Inequitable, causes protests.",
    ),
    ConscriptionLaw.UNIVERSAL_MALE_SERVICE: ConscriptionLawStats(
        max_rate=0.100, economy_penalty=0.06, unrest_per_step=0.004,
        training_quality=0.75, min_draft_age=18, max_draft_age=50,
        desertion_rate=0.003, political_cost=20.0, gender_all=False,
        research_penalty=0.03, morale_modifier=0.02,
        description="All males serve 12-24 months. Strong reserves, moderate drain.",
    ),
    ConscriptionLaw.UNIVERSAL_SERVICE: ConscriptionLawStats(
        max_rate=0.130, economy_penalty=0.08, unrest_per_step=0.005,
        training_quality=0.75, min_draft_age=18, max_draft_age=50,
        desertion_rate=0.003, political_cost=25.0, gender_all=True,
        research_penalty=0.03, morale_modifier=0.03,
        description="All citizens serve. Maximum peacetime readiness. Israeli model.",
    ),

    # ── Wartime Mobilisation ───────────────────────────────────────────── #
    ConscriptionLaw.WARTIME_DRAFT: ConscriptionLawStats(
        max_rate=0.160, economy_penalty=0.12, unrest_per_step=0.008,
        training_quality=0.60, min_draft_age=18, max_draft_age=55,
        desertion_rate=0.008, political_cost=30.0, gender_all=False,
        research_penalty=0.05, morale_modifier=-0.02,
        description="Mass wartime call-up. Economy strained, training shortened.",
    ),
    ConscriptionLaw.GENERAL_MOBILISATION: ConscriptionLawStats(
        max_rate=0.200, economy_penalty=0.18, unrest_per_step=0.012,
        training_quality=0.50, min_draft_age=17, max_draft_age=55,
        desertion_rate=0.010, political_cost=40.0, gender_all=False,
        research_penalty=0.08, morale_modifier=-0.05,
        description="Full national mobilisation. Factories lose workers en masse.",
    ),
    ConscriptionLaw.TOTAL_MOBILISATION: ConscriptionLawStats(
        max_rate=0.250, economy_penalty=0.25, unrest_per_step=0.018,
        training_quality=0.40, min_draft_age=16, max_draft_age=60,
        desertion_rate=0.015, political_cost=50.0, gender_all=True,
        research_penalty=0.12, morale_modifier=-0.10,
        description="All able-bodied. Minimal training. Economy in freefall.",
    ),

    # ── Total War ──────────────────────────────────────────────────────── #
    ConscriptionLaw.WAR_ECONOMY_DRAFT: ConscriptionLawStats(
        max_rate=0.280, economy_penalty=0.32, unrest_per_step=0.025,
        training_quality=0.35, min_draft_age=16, max_draft_age=60,
        desertion_rate=0.020, political_cost=60.0, gender_all=True,
        research_penalty=0.15, morale_modifier=-0.15,
        description="Workers pulled from factories. Production collapses.",
    ),
    ConscriptionLaw.SERVICE_BY_REQUIREMENT: ConscriptionLawStats(
        max_rate=0.320, economy_penalty=0.38, unrest_per_step=0.030,
        training_quality=0.30, min_draft_age=15, max_draft_age=65,
        desertion_rate=0.025, political_cost=70.0, gender_all=True,
        research_penalty=0.20, morale_modifier=-0.20,
        description="State assigns every citizen a role. No civilian life remains.",
    ),
    ConscriptionLaw.ALL_ADULTS_SERVE: ConscriptionLawStats(
        max_rate=0.380, economy_penalty=0.45, unrest_per_step=0.035,
        training_quality=0.25, min_draft_age=15, max_draft_age=65,
        desertion_rate=0.030, political_cost=80.0, gender_all=True,
        research_penalty=0.25, morale_modifier=-0.25,
        description="16-60 armed. No exemptions. Starvation and collapse imminent.",
    ),

    # ── Desperation ────────────────────────────────────────────────────── #
    ConscriptionLaw.SCRAPING_THE_BARREL: ConscriptionLawStats(
        max_rate=0.450, economy_penalty=0.55, unrest_per_step=0.045,
        training_quality=0.15, min_draft_age=14, max_draft_age=70,
        desertion_rate=0.050, political_cost=90.0, gender_all=True,
        research_penalty=0.35, morale_modifier=-0.35,
        description="Children and elderly armed. Mass desertion. Society disintegrating.",
    ),
    ConscriptionLaw.NATION_IN_ARMS: ConscriptionLawStats(
        max_rate=0.550, economy_penalty=0.70, unrest_per_step=0.060,
        training_quality=0.10, min_draft_age=12, max_draft_age=75,
        desertion_rate=0.080, political_cost=100.0, gender_all=True,
        research_penalty=0.50, morale_modifier=-0.50,
        description="Entire population is the army. No economy. Civilisation ends.",
    ),
}


# ── Convenience accessors ────────────────────────────────────────────────── #
# These dicts are used by the rest of the code for quick property lookup.

CONSCRIPTION_RATES = {law: stats.max_rate for law, stats in LAW_STATS.items()}
CONSCRIPTION_ECONOMY_PENALTY = {law: stats.economy_penalty for law, stats in LAW_STATS.items()}
CONSCRIPTION_UNREST = {law: stats.unrest_per_step for law, stats in LAW_STATS.items()}
CONSCRIPTION_TRAINING_QUALITY = {law: stats.training_quality for law, stats in LAW_STATS.items()}
CONSCRIPTION_DESERTION = {law: stats.desertion_rate for law, stats in LAW_STATS.items()}
CONSCRIPTION_MORALE_MOD = {law: stats.morale_modifier for law, stats in LAW_STATS.items()}

# Steps required to transition per level jumped
LAW_TRANSITION_STEPS = 12


# ═══════════════════════════════════════════════════════════════════════════ #
# Regime Types — 12 real-world governance models                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class RegimeType(Enum):
    """Government types that gate available conscription laws and modify effects."""
    # Democratic spectrum
    LIBERAL_DEMOCRACY     = 0    # US, UK, France — strong civil liberties
    SOCIAL_DEMOCRACY      = 1    # Nordics, Germany — welfare state + rights
    REPUBLIC              = 2    # Generic republic with elections
    MANAGED_DEMOCRACY     = 3    # Russia-style, elections but controlled

    # Authoritarian spectrum
    OLIGARCHY             = 4    # Rule by elites / business interests
    MILITARY_JUNTA        = 5    # Military officers in power
    ONE_PARTY_STATE       = 6    # Single ruling party (China model)
    AUTHORITARIAN         = 7    # Strongman / personalist rule

    # Ideological extremes
    THEOCRACY             = 8    # Religious law governs (Iran model)
    FASCIST               = 9    # Ultranationalist corporatism
    COMMUNIST             = 10   # Marxist-Leninist state control
    TOTALITARIAN          = 11   # Complete state control over all aspects


# Regime → max conscription law reachable
# Democratic regimes are capped early; totalitarian can go to the bitter end.
REGIME_MAX_LAW: Dict[RegimeType, ConscriptionLaw] = {
    RegimeType.LIBERAL_DEMOCRACY:  ConscriptionLaw.WARTIME_DRAFT,
    RegimeType.SOCIAL_DEMOCRACY:   ConscriptionLaw.WARTIME_DRAFT,
    RegimeType.REPUBLIC:           ConscriptionLaw.GENERAL_MOBILISATION,
    RegimeType.MANAGED_DEMOCRACY:  ConscriptionLaw.TOTAL_MOBILISATION,
    RegimeType.OLIGARCHY:          ConscriptionLaw.TOTAL_MOBILISATION,
    RegimeType.MILITARY_JUNTA:     ConscriptionLaw.ALL_ADULTS_SERVE,
    RegimeType.ONE_PARTY_STATE:    ConscriptionLaw.ALL_ADULTS_SERVE,
    RegimeType.AUTHORITARIAN:      ConscriptionLaw.ALL_ADULTS_SERVE,
    RegimeType.THEOCRACY:          ConscriptionLaw.SERVICE_BY_REQUIREMENT,
    RegimeType.FASCIST:            ConscriptionLaw.SCRAPING_THE_BARREL,
    RegimeType.COMMUNIST:          ConscriptionLaw.SCRAPING_THE_BARREL,
    RegimeType.TOTALITARIAN:       ConscriptionLaw.NATION_IN_ARMS,
}

# Regime → base political stability modifier (affects unrest from conscription)
# Democracies generate MORE unrest per conscription tier; authoritarian regimes suppress it.
REGIME_UNREST_MODIFIER: Dict[RegimeType, float] = {
    RegimeType.LIBERAL_DEMOCRACY:  1.50,   # citizens protest loudly
    RegimeType.SOCIAL_DEMOCRACY:   1.30,
    RegimeType.REPUBLIC:           1.10,
    RegimeType.MANAGED_DEMOCRACY:  0.80,   # state controls media
    RegimeType.OLIGARCHY:          0.70,
    RegimeType.MILITARY_JUNTA:     0.50,   # protests suppressed
    RegimeType.ONE_PARTY_STATE:    0.45,
    RegimeType.AUTHORITARIAN:      0.40,
    RegimeType.THEOCRACY:          0.60,   # religious duty reduces dissent
    RegimeType.FASCIST:            0.35,   # nationalism overrides dissent
    RegimeType.COMMUNIST:          0.40,   # state propaganda
    RegimeType.TOTALITARIAN:       0.25,   # total suppression
}

# Regime → economy efficiency modifier (some regimes run economies better)
REGIME_ECONOMY_MODIFIER: Dict[RegimeType, float] = {
    RegimeType.LIBERAL_DEMOCRACY:  1.15,   # free markets, innovation
    RegimeType.SOCIAL_DEMOCRACY:   1.10,
    RegimeType.REPUBLIC:           1.05,
    RegimeType.MANAGED_DEMOCRACY:  0.95,
    RegimeType.OLIGARCHY:          0.90,   # corruption
    RegimeType.MILITARY_JUNTA:     0.80,   # inefficient
    RegimeType.ONE_PARTY_STATE:    0.85,   # state planning
    RegimeType.AUTHORITARIAN:      0.85,
    RegimeType.THEOCRACY:          0.75,   # religious law ≠ economic law
    RegimeType.FASCIST:            0.90,   # corporatist, some efficiency
    RegimeType.COMMUNIST:          0.80,   # central planning overhead
    RegimeType.TOTALITARIAN:       0.70,   # fear-based, innovation dies
}

# Regime → military effectiveness modifier (some regimes produce better soldiers)
REGIME_MILITARY_MODIFIER: Dict[RegimeType, float] = {
    RegimeType.LIBERAL_DEMOCRACY:  1.05,   # well-equipped but cautious
    RegimeType.SOCIAL_DEMOCRACY:   1.00,
    RegimeType.REPUBLIC:           1.00,
    RegimeType.MANAGED_DEMOCRACY:  1.05,
    RegimeType.OLIGARCHY:          0.90,
    RegimeType.MILITARY_JUNTA:     1.15,   # military is priority
    RegimeType.ONE_PARTY_STATE:    1.05,
    RegimeType.AUTHORITARIAN:      1.10,
    RegimeType.THEOCRACY:          1.00,   # zealotry ≠ tactics
    RegimeType.FASCIST:            1.15,   # aggressive doctrine
    RegimeType.COMMUNIST:          1.10,   # mass warfare doctrine
    RegimeType.TOTALITARIAN:       1.10,   # total state resources
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Skill Levels                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class SkillLevel(Enum):
    UNTRAINED = 0
    BASIC     = 1
    SKILLED   = 2
    EXPERT    = 3

# Steps required to train to each level (cumulative from previous)
TRAINING_STEPS = {
    SkillLevel.UNTRAINED: 0,
    SkillLevel.BASIC:     10,
    SkillLevel.SKILLED:   20,
    SkillLevel.EXPERT:    40,
}

# Productivity multiplier per skill level
SKILL_PRODUCTIVITY = {
    SkillLevel.UNTRAINED: 0.40,
    SkillLevel.BASIC:     0.70,
    SkillLevel.SKILLED:   1.00,
    SkillLevel.EXPERT:    1.30,
}

class MilitaryTraining(Enum):
    CIVILIAN = 0
    RECRUIT  = 1
    REGULAR  = 2
    VETERAN  = 3

MILITARY_TRAINING_STEPS = {
    MilitaryTraining.CIVILIAN: 0,
    MilitaryTraining.RECRUIT:  15,
    MilitaryTraining.REGULAR:  30,
    MilitaryTraining.VETERAN:  60,
}

MILITARY_EFFECTIVENESS = {
    MilitaryTraining.CIVILIAN: 0.20,
    MilitaryTraining.RECRUIT:  0.50,
    MilitaryTraining.REGULAR:  1.00,
    MilitaryTraining.VETERAN:  1.40,
}

# Max training slots per cluster (anti-exploitation: can't train unlimited people)
MAX_TRAINING_SLOTS = 500


# ═══════════════════════════════════════════════════════════════════════════ #
# Training Batch                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class TrainingBatch:
    """A group of people being trained for a specific role."""
    count: int                     # number of people in this batch
    target_sector: int             # EconSector.value or -1 for military
    current_level: int             # SkillLevel.value or MilitaryTraining.value
    target_level: int              # what they're training toward
    steps_remaining: int           # steps until graduation
    is_military: bool = False      # True = military training, False = civilian

    @property
    def is_complete(self) -> bool:
        return self.steps_remaining <= 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Cluster Manpower State                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ClusterManpower:
    """
    Per-cluster manpower state.

    total_population     — absolute population count (thousands)
    working_age_frac     — fraction that is working age [0.5, 0.7]
    sector_workers       — (N_SECTORS,) workers allocated per economic sector
    military_personnel   — currently serving in military
    in_training          — list of active training batches
    unemployed           — idle workers (generates unrest)
    disabled             — permanently removed from workforce (casualties)
    skill_distribution   — (N_SECTORS, 4) workers per sector per skill level
    military_skill_dist  — (4,) military personnel per training level
    """
    cluster_id: int
    total_population: float = 100.0       # thousands
    working_age_frac: float = 0.62
    sector_workers: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(N_SECTORS, dtype=np.float64))
    military_personnel: float = 0.0
    in_training: List[TrainingBatch] = field(default_factory=list)
    unemployed: float = 10.0              # thousands
    disabled: float = 0.0                 # thousands
    skill_distribution: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((N_SECTORS, 4), dtype=np.float64))
    military_skill_dist: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64))

    @property
    def working_age_pop(self) -> float:
        """Working-age population in thousands."""
        return self.total_population * self.working_age_frac

    @property
    def total_employed(self) -> float:
        """Total people currently employed or serving."""
        return float(np.sum(self.sector_workers)) + self.military_personnel

    @property
    def total_in_training(self) -> int:
        return sum(b.count for b in self.in_training)

    @property
    def available_workforce(self) -> float:
        """Workers available for assignment (employed + unemployed - training)."""
        return max(0.0, self.working_age_pop - self.total_employed
                   - self.total_in_training - self.disabled)

    @property
    def unemployment_rate(self) -> float:
        wap = self.working_age_pop
        if wap <= 0:
            return 0.0
        return min(1.0, self.unemployed / wap)

    @property
    def conscription_rate(self) -> float:
        """Fraction of working-age pop currently in military."""
        wap = self.working_age_pop
        if wap <= 0:
            return 0.0
        return min(1.0, self.military_personnel / wap)

    @property
    def avg_sector_skill(self) -> NDArray[np.float64]:
        """Average productivity multiplier per sector from skill distribution."""
        mults = np.array([SKILL_PRODUCTIVITY[SkillLevel(i)] for i in range(4)])
        total_per_sector = np.sum(self.skill_distribution, axis=1)  # (N_SECTORS,)
        safe_total = np.where(total_per_sector > 0.1, total_per_sector, 1.0)
        weighted = (self.skill_distribution @ mults) / safe_total
        return np.clip(weighted, 0.4, 1.3)

    @property
    def avg_military_skill(self) -> float:
        """Average military effectiveness from training distribution."""
        mults = np.array([MILITARY_EFFECTIVENESS[MilitaryTraining(i)] for i in range(4)])
        total = float(np.sum(self.military_skill_dist))
        if total < 0.1:
            return 0.5
        return float(np.clip(np.dot(self.military_skill_dist, mults) / total, 0.2, 1.4))

    def copy(self) -> "ClusterManpower":
        return ClusterManpower(
            cluster_id=self.cluster_id,
            total_population=self.total_population,
            working_age_frac=self.working_age_frac,
            sector_workers=self.sector_workers.copy(),
            military_personnel=self.military_personnel,
            in_training=[TrainingBatch(
                count=b.count, target_sector=b.target_sector,
                current_level=b.current_level, target_level=b.target_level,
                steps_remaining=b.steps_remaining, is_military=b.is_military,
            ) for b in self.in_training],
            unemployed=self.unemployed,
            disabled=self.disabled,
            skill_distribution=self.skill_distribution.copy(),
            military_skill_dist=self.military_skill_dist.copy(),
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Manpower Policy                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionManpowerPolicy:
    """Faction-level conscription and manpower policy."""
    faction_id: int
    regime_type: RegimeType = RegimeType.REPUBLIC
    conscription_law: ConscriptionLaw = ConscriptionLaw.VOLUNTEER_ONLY
    target_law: Optional[ConscriptionLaw] = None    # law being transitioned to
    transition_remaining: int = 0                     # steps left in law change
    political_capital: float = 50.0                   # spent on law changes

    @property
    def max_conscription_rate(self) -> float:
        return CONSCRIPTION_RATES[self.conscription_law]

    @property
    def economy_penalty(self) -> float:
        return CONSCRIPTION_ECONOMY_PENALTY[self.conscription_law]

    @property
    def unrest_per_step(self) -> float:
        return CONSCRIPTION_UNREST[self.conscription_law]

    @property
    def max_allowed_law(self) -> ConscriptionLaw:
        return REGIME_MAX_LAW[self.regime_type]

    def can_change_to(self, law: ConscriptionLaw) -> bool:
        """Check if this law is reachable given regime type."""
        return law.value <= self.max_allowed_law.value

    def copy(self) -> "FactionManpowerPolicy":
        return FactionManpowerPolicy(
            faction_id=self.faction_id,
            regime_type=self.regime_type,
            conscription_law=self.conscription_law,
            target_law=self.target_law,
            transition_remaining=self.transition_remaining,
            political_capital=self.political_capital,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Manpower Step Function                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_manpower(
    cluster_mp: ClusterManpower,
    faction_policy: FactionManpowerPolicy,
    hazard: float,
    food_ratio: float,
    gdp_level: float,
    dt: float = 0.01,
) -> Tuple[ClusterManpower, Dict[str, float]]:
    """
    Advance manpower state by one step.

    Returns:
        (updated_cluster_manpower, feedback_dict)

    feedback_dict keys:
        - sector_skill_mult: (N_SECTORS,) productivity multiplier from skills
        - military_effectiveness: float from training quality
        - unrest_delta: float political unrest from conscription
        - unemployment_rate: float for economy feedback
        - conscription_economy_penalty: float output reduction
    """
    mp = cluster_mp

    # ── 1. Population dynamics (births, deaths, aging) ─────────────────── #
    birth_rate = 0.001 * food_ratio * dt  # food enables population growth
    death_rate = 0.0005 * (1.0 + 2.0 * hazard) * dt  # hazard kills people
    mp.total_population = max(1.0, mp.total_population * (1.0 + birth_rate - death_rate))

    # ── 2. Casualties from combat (hazard > 0.5) ──────────────────────── #
    if hazard > 0.5:
        casualties = 0.5 * (hazard - 0.5) * dt  # thousands killed
        mp.disabled += casualties
        # Remove from military first, then civilian sectors
        mil_loss = min(mp.military_personnel, casualties * 0.6)
        mp.military_personnel -= mil_loss
        civ_loss = casualties * 0.4
        # Spread civilian losses across sectors proportionally
        total_workers = float(np.sum(mp.sector_workers))
        if total_workers > 0.1:
            loss_frac = min(1.0, civ_loss / total_workers)
            mp.sector_workers *= (1.0 - loss_frac)

    # ── 3. Training pipeline tick ──────────────────────────────────────── #
    completed = []
    ongoing = []
    for batch in mp.in_training:
        batch.steps_remaining -= 1
        if batch.is_complete:
            completed.append(batch)
        else:
            ongoing.append(batch)
    mp.in_training = ongoing

    # Graduate completed trainees
    for batch in completed:
        if batch.is_military:
            level = min(batch.target_level, 3)
            mp.military_skill_dist[level] += batch.count
            mp.military_personnel += batch.count
        else:
            sector = batch.target_sector
            level = min(batch.target_level, 3)
            if 0 <= sector < N_SECTORS:
                mp.skill_distribution[sector, level] += batch.count
                mp.sector_workers[sector] += batch.count

    # ── 4. Conscription law transition ─────────────────────────────────── #
    if faction_policy.target_law is not None:
        faction_policy.transition_remaining -= 1
        if faction_policy.transition_remaining <= 0:
            faction_policy.conscription_law = faction_policy.target_law
            faction_policy.target_law = None

    # ── 5. Conscription enforcement ────────────────────────────────────── #
    max_rate = faction_policy.max_conscription_rate
    # Economic health limits effective conscription
    effective_max = max_rate * (0.5 + 0.5 * min(food_ratio, gdp_level))
    current_rate = mp.conscription_rate
    # Don't exceed allowed rate
    if current_rate > effective_max * 1.1:
        # Over-conscripted: demobilize some troops
        excess = (current_rate - effective_max) * mp.working_age_pop
        demob = min(mp.military_personnel, excess * 0.1 * dt)
        mp.military_personnel -= demob
        mp.unemployed += demob

    # ── 6. Desertion ─────────────────────────────────────────────────────── #
    # Soldiers desert based on conscription law + food availability + morale.
    # Emergency conscripts (civilian-tier) desert at 3× the base rate.
    law_stats = LAW_STATS[faction_policy.conscription_law]
    base_desertion = law_stats.desertion_rate * dt
    hunger_desertion = max(0.0, (0.5 - food_ratio) * 0.02) * dt  # starving = desert
    total_desertion_rate = base_desertion + hunger_desertion

    if total_desertion_rate > 0.0 and mp.military_personnel > 1.0:
        # Untrained conscripts desert at 3× rate
        for level in range(4):
            level_rate = total_desertion_rate * (3.0 if level == 0 else 1.0)
            deserted = mp.military_skill_dist[level] * level_rate
            mp.military_skill_dist[level] = max(0.0, mp.military_skill_dist[level] - deserted)
            mp.military_personnel = max(0.0, mp.military_personnel - deserted)
            mp.unemployed += deserted  # deserters become unemployed

    # ── 7. Unemployment update ─────────────────────────────────────────── #
    total_employed = mp.total_employed + mp.total_in_training
    wap = mp.working_age_pop
    mp.unemployed = max(0.0, wap - total_employed - mp.disabled)

    # ── 8. Skill degradation for reassigned workers ────────────────────── #
    mp.skill_distribution[:, 3] *= (1.0 - 0.001 * dt)  # experts slowly degrade
    mp.skill_distribution[:, 2] *= (1.0 - 0.0005 * dt)  # skilled degrade slower

    # ── 9. Brain drain from high conscription ──────────────────────────── #
    if faction_policy.conscription_law.value >= ConscriptionLaw.ALL_ADULTS_SERVE.value:
        flight = 0.005 * dt * (1.0 + 0.5 * (faction_policy.conscription_law.value - 12))
        mp.skill_distribution[:, 2:] *= (1.0 - flight)
        mp.total_population -= mp.total_population * flight * 0.1

    # ── 10. Regime modifiers on feedback ───────────────────────────────── #
    regime = faction_policy.regime_type
    unrest_mod = REGIME_UNREST_MODIFIER.get(regime, 1.0)
    econ_mod = REGIME_ECONOMY_MODIFIER.get(regime, 1.0)
    mil_mod = REGIME_MILITARY_MODIFIER.get(regime, 1.0)

    # ── Compute feedback ──────────────────────────────────────────────── #
    feedback = {
        "sector_skill_mult": mp.avg_sector_skill,
        "military_effectiveness": mp.avg_military_skill * mil_mod,
        "unrest_delta": faction_policy.unrest_per_step * dt * unrest_mod,
        "unemployment_rate": mp.unemployment_rate,
        "conscription_economy_penalty": faction_policy.economy_penalty / max(econ_mod, 0.5),
        "regime_economy_modifier": econ_mod,
        "regime_military_modifier": mil_mod,
        "desertion_rate": total_desertion_rate if mp.military_personnel > 1.0 else 0.0,
        "training_quality": law_stats.training_quality,
        "research_penalty": law_stats.research_penalty,
        "morale_modifier": CONSCRIPTION_MORALE_MOD[faction_policy.conscription_law],
    }

    return mp, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Manpower Actions                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class ManpowerAction(Enum):
    """Actions available for manpower management."""
    NOOP                  = 0
    TRAIN_SECTOR_WORKERS  = 1   # Train workers for a specific economic sector
    TRAIN_MILITARY        = 2   # Train military recruits
    CONSCRIPT             = 3   # Draft civilians into military (immediate, untrained)
    DEMOBILIZE            = 4   # Release soldiers back to civilian workforce
    CHANGE_CONSCRIPTION_LAW = 5 # Change the conscription law (takes time)
    REASSIGN_WORKERS      = 6   # Move workers between sectors (with skill penalty)


def apply_manpower_action(
    cluster_mp: ClusterManpower,
    faction_policy: FactionManpowerPolicy,
    action_type: int,
    sector_target: int,
    count: int,
    law_target: int,
    cluster_owners: Dict[int, int],
    faction_id: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """
    Apply a manpower action.

    Returns: (updated_manpower, updated_policy, reward)
    """
    try:
        action = ManpowerAction(action_type)
    except ValueError:
        return cluster_mp, faction_policy, 0.0

    if action == ManpowerAction.NOOP:
        return cluster_mp, faction_policy, 0.0

    elif action == ManpowerAction.TRAIN_SECTOR_WORKERS:
        return _train_sector(cluster_mp, sector_target, count)

    elif action == ManpowerAction.TRAIN_MILITARY:
        return _train_military(cluster_mp, faction_policy, count)

    elif action == ManpowerAction.CONSCRIPT:
        return _conscript(cluster_mp, faction_policy, count)

    elif action == ManpowerAction.DEMOBILIZE:
        return _demobilize(cluster_mp, faction_policy, count)

    elif action == ManpowerAction.CHANGE_CONSCRIPTION_LAW:
        return _change_law(cluster_mp, faction_policy, law_target)

    elif action == ManpowerAction.REASSIGN_WORKERS:
        return _reassign(cluster_mp, sector_target, count)

    return cluster_mp, faction_policy, 0.0


def _train_sector(
    mp: ClusterManpower, sector: int, count: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """Train workers for an economic sector."""
    sector = min(sector, N_SECTORS - 1)
    available = mp.available_workforce
    actual = min(count, int(available), MAX_TRAINING_SLOTS - mp.total_in_training)
    if actual <= 0:
        return mp, FactionManpowerPolicy(faction_id=-1), -0.1

    # Determine target level based on current distribution
    current_dist = mp.skill_distribution[sector]
    if current_dist[1] < current_dist[0]:  # need more basics
        target = SkillLevel.BASIC.value
        steps = TRAINING_STEPS[SkillLevel.BASIC]
    elif current_dist[2] < current_dist[1]:
        target = SkillLevel.SKILLED.value
        steps = TRAINING_STEPS[SkillLevel.SKILLED]
    else:
        target = SkillLevel.EXPERT.value
        steps = TRAINING_STEPS[SkillLevel.EXPERT]

    mp.in_training.append(TrainingBatch(
        count=actual, target_sector=sector,
        current_level=SkillLevel.UNTRAINED.value, target_level=target,
        steps_remaining=steps, is_military=False,
    ))
    mp.unemployed = max(0.0, mp.unemployed - actual)

    return mp, FactionManpowerPolicy(faction_id=-1), 0.2


def _train_military(
    mp: ClusterManpower, policy: FactionManpowerPolicy, count: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """Train military recruits (proper training pipeline, not instant conscription)."""
    available = mp.available_workforce
    max_allowed = policy.max_conscription_rate * mp.working_age_pop - mp.military_personnel
    actual = min(count, int(available), int(max(0, max_allowed)),
                 MAX_TRAINING_SLOTS - mp.total_in_training)
    if actual <= 0:
        return mp, policy, -0.1

    # Determine training level
    current_mil = mp.military_skill_dist
    if current_mil[1] < current_mil[0]:
        target = MilitaryTraining.RECRUIT.value
        steps = MILITARY_TRAINING_STEPS[MilitaryTraining.RECRUIT]
    elif current_mil[2] < current_mil[1]:
        target = MilitaryTraining.REGULAR.value
        steps = MILITARY_TRAINING_STEPS[MilitaryTraining.REGULAR]
    else:
        target = MilitaryTraining.VETERAN.value
        steps = MILITARY_TRAINING_STEPS[MilitaryTraining.VETERAN]

    mp.in_training.append(TrainingBatch(
        count=actual, target_sector=-1,
        current_level=MilitaryTraining.CIVILIAN.value, target_level=target,
        steps_remaining=steps, is_military=True,
    ))
    mp.unemployed = max(0.0, mp.unemployed - actual)

    return mp, policy, 0.3


def _conscript(
    mp: ClusterManpower, policy: FactionManpowerPolicy, count: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """
    Emergency conscription — instant but untrained.
    These become CIVILIAN-level military (0.2x effectiveness).
    Generates significant unrest.
    """
    available = mp.available_workforce
    max_allowed = policy.max_conscription_rate * mp.working_age_pop - mp.military_personnel
    actual = min(count, int(available), int(max(0, max_allowed)))
    if actual <= 0:
        return mp, policy, -0.2

    mp.military_personnel += actual
    mp.military_skill_dist[MilitaryTraining.CIVILIAN.value] += actual
    mp.unemployed = max(0.0, mp.unemployed - actual)

    # Conscription generates unrest proportional to count
    unrest_cost = 0.01 * actual / max(mp.working_age_pop, 1.0)
    return mp, policy, 0.1 - unrest_cost


def _demobilize(
    mp: ClusterManpower, policy: FactionManpowerPolicy, count: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """Release soldiers back to civilian workforce."""
    actual = min(count, int(mp.military_personnel))
    if actual <= 0:
        return mp, policy, -0.05

    mp.military_personnel -= actual
    mp.unemployed += actual
    # Remove from military skill distribution (lowest skill first)
    remaining = actual
    for level in range(4):
        removed = min(remaining, mp.military_skill_dist[level])
        mp.military_skill_dist[level] -= removed
        remaining -= removed
        if remaining <= 0:
            break

    return mp, policy, 0.1


def _change_law(
    mp: ClusterManpower, policy: FactionManpowerPolicy, law_target: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """Change conscription law. Takes LAW_TRANSITION_STEPS and costs political capital."""
    try:
        target = ConscriptionLaw(min(law_target, 5))
    except ValueError:
        return mp, policy, -0.1

    if not policy.can_change_to(target):
        return mp, policy, -0.3  # regime doesn't allow this law

    if policy.target_law is not None:
        return mp, policy, -0.2  # already transitioning

    if target == policy.conscription_law:
        return mp, policy, -0.05  # already at this level

    # Cost: 1 political capital per level jumped
    levels_jumped = abs(target.value - policy.conscription_law.value)
    cost = levels_jumped * 10.0
    if policy.political_capital < cost:
        return mp, policy, -0.2  # not enough political capital

    policy.political_capital -= cost
    policy.target_law = target
    policy.transition_remaining = LAW_TRANSITION_STEPS * levels_jumped

    return mp, policy, 0.2


def _reassign(
    mp: ClusterManpower, sector_to: int, count: int,
) -> Tuple[ClusterManpower, FactionManpowerPolicy, float]:
    """Move workers between sectors. Incurs skill penalty."""
    sector_to = min(sector_to, N_SECTORS - 1)

    # Find sector with most workers to pull from
    sector_from = int(np.argmax(mp.sector_workers))
    if sector_from == sector_to:
        return mp, FactionManpowerPolicy(faction_id=-1), -0.05

    available = mp.sector_workers[sector_from]
    actual = min(count, int(available * 0.15))  # max 15% from one sector
    if actual <= 0:
        return mp, FactionManpowerPolicy(faction_id=-1), -0.1

    mp.sector_workers[sector_from] -= actual
    mp.sector_workers[sector_to] += actual

    # Skill penalty: reassigned workers lose one skill level
    for level in range(3, 0, -1):
        moved = min(actual, mp.skill_distribution[sector_from, level])
        mp.skill_distribution[sector_from, level] -= moved
        mp.skill_distribution[sector_to, max(0, level - 1)] += moved
        actual -= int(moved)
        if actual <= 0:
            break

    return mp, FactionManpowerPolicy(faction_id=-1), 0.05


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_manpower(
    n_clusters: int,
    faction_ids: List[int],
    cluster_owners: Dict[int, int],
    terrain_types: List[str],
    rng: np.random.Generator,
    base_population: float = 100.0,
) -> Tuple[List[ClusterManpower], Dict[int, FactionManpowerPolicy]]:
    """Initialize manpower for all clusters and factions."""
    cluster_manpower = []

    for i in range(n_clusters):
        terrain = terrain_types[i] if i < len(terrain_types) else "OPEN"

        # Population varies by terrain
        pop_mult = {"URBAN": 2.0, "PLAINS": 1.5, "OPEN": 1.2,
                     "FOREST": 0.8, "MOUNTAINS": 0.6, "DESERT": 0.4,
                     "MARSH": 0.5}.get(terrain, 1.0)
        pop = base_population * pop_mult + rng.uniform(-10, 10)

        # Initial worker distribution across sectors
        workers = np.full(N_SECTORS, pop * 0.62 / N_SECTORS * 0.85, dtype=np.float64)
        # Terrain bias
        if terrain in ("PLAINS", "OPEN"):
            workers[EconSector.AGRICULTURE.value] *= 1.5
        elif terrain == "URBAN":
            workers[EconSector.MANUFACTURING.value] *= 1.5
            workers[EconSector.SERVICES.value] *= 1.5
        elif terrain == "MOUNTAINS":
            workers[EconSector.MINING.value] *= 2.0

        # Initial skill distribution: mostly basic, some skilled
        skill_dist = np.zeros((N_SECTORS, 4), dtype=np.float64)
        for s in range(N_SECTORS):
            w = workers[s]
            skill_dist[s, 0] = w * 0.2   # untrained
            skill_dist[s, 1] = w * 0.5   # basic
            skill_dist[s, 2] = w * 0.25  # skilled
            skill_dist[s, 3] = w * 0.05  # expert

        unemployed = pop * 0.62 * 0.10 + rng.uniform(0, 5)

        cluster_manpower.append(ClusterManpower(
            cluster_id=i,
            total_population=max(10.0, pop),
            working_age_frac=0.60 + rng.uniform(0.0, 0.05),
            sector_workers=workers,
            military_personnel=pop * 0.02,  # 2% baseline military
            unemployed=max(0.0, unemployed),
            skill_distribution=skill_dist,
            military_skill_dist=np.array([
                pop * 0.005,   # civilians in uniform
                pop * 0.008,   # recruits
                pop * 0.005,   # regulars
                pop * 0.002,   # veterans
            ]),
        ))

    faction_policies = {
        fid: FactionManpowerPolicy(
            faction_id=fid,
            regime_type=RegimeType.REPUBLIC,
            conscription_law=ConscriptionLaw.VOLUNTEER_ONLY,
            political_capital=50.0 + rng.uniform(0, 20),
        )
        for fid in faction_ids
    }

    return cluster_manpower, faction_policies


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation Helper                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def manpower_obs(
    mp: ClusterManpower,
    policy: FactionManpowerPolicy,
) -> NDArray[np.float32]:
    """
    Build observation vector for manpower state.

    Layout (per cluster, 25 floats):
      - total_population (normalized)
      - working_age_frac
      - 7 sector worker fractions
      - military_rate
      - unemployment_rate
      - training_count (normalized)
      - disabled_rate
      - 7 avg_sector_skill values
      - military_effectiveness

    Plus faction policy (5 floats):
      - conscription_law (normalized 0-5)
      - regime_type (normalized 0-5)
      - economy_penalty
      - political_capital (normalized)
      - transition_progress

    Total: 25 + 5 = 30 floats per cluster
    """
    obs = np.zeros(30, dtype=np.float32)

    wap = mp.working_age_pop
    safe_wap = max(wap, 1.0)

    obs[0] = min(mp.total_population / 500.0, 1.0)
    obs[1] = mp.working_age_frac
    # Sector worker fractions (7)
    obs[2:2 + N_SECTORS] = np.clip(mp.sector_workers / safe_wap, 0.0, 1.0)
    obs[9] = mp.conscription_rate
    obs[10] = mp.unemployment_rate
    obs[11] = min(mp.total_in_training / safe_wap, 1.0)
    obs[12] = min(mp.disabled / safe_wap, 1.0)
    # Sector skill multipliers (7)
    obs[13:13 + N_SECTORS] = (mp.avg_sector_skill - 0.4) / 0.9  # normalize to ~[0,1]
    obs[20] = (mp.avg_military_skill - 0.2) / 1.2

    # Faction policy (5)
    obs[21] = policy.conscription_law.value / 5.0
    obs[22] = policy.regime_type.value / 5.0
    obs[23] = policy.economy_penalty
    obs[24] = min(policy.political_capital / 100.0, 1.0)
    obs[25] = (policy.transition_remaining / (LAW_TRANSITION_STEPS * 5.0)
               if policy.target_law is not None else 0.0)

    return obs


def manpower_obs_size() -> int:
    """Observation vector size for manpower state per cluster."""
    return 30
