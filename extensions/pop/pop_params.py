"""
PopParams — Hyperparameter pack for the population (pop) subsystem.

Design philosophy:
  Pops are demographic vectors, not agents. Each cluster carries a fixed-size
  distribution over (job × class) archetypes and ethnic groups. All operations
  are dot products and weighted sums — no per-agent simulation.

  Pop dynamics are deliberately lightweight:
    - Semi-implicit ODE (one pass per env step, not per RK4 sub-step)
    - Size evolution is slow (updated every pop_mobility_period steps)
    - Aggregates (gini, radical_mass, etc.) are computed once per step
      and injected as drivers into the existing GravitasEnv ODEs

  The pop system does NOT replace the existing σ/h/τ/p/E/Φ dynamics.
  It enriches them with richer societal drivers.

Pop archetypes (P = 8):
  0  SUBSISTENCE   — farmer/lower class. Large in agrarian regimes.
  1  YEOMAN        — farmer/middle class. Smallholders, politically conservative.
  2  URBAN_LABORER — laborer/lower class. High radicalization potential.
  3  ARTISAN       — skilled craftsman/middle class. Sensitive to resource levels.
  4  CLERK         — bureaucrat/middle class. Loyal to institutions, trust-boosting.
  5  MERCHANT      — trader/middle class. Income-sensitive, mobility-oriented.
  6  PROFESSIONAL  — doctor/lawyer/engineer upper-middle. Coherence-building.
  7  ELITE         — landowner/capital upper class. Fragmentation risk if threatened.

Ethnic groups (E = 3 by default, configurable):
  Groups are not named — they are indices 0..E-1 with a cultural distance matrix
  C[i,j] ∈ [0,1]. High C[i,j] means high inter-group tension potential.
  Ethnic fractionalization = 1 - Σ share_i² (Herfindahl index, 0=homogeneous).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────── #
# Archetype metadata                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

N_ARCHETYPES = 9  # 8 original + 1 soldier

ARCHETYPE_NAMES = [
    "SUBSISTENCE",    # 0
    "YEOMAN",         # 1
    "URBAN_LABORER",  # 2
    "ARTISAN",        # 3
    "CLERK",          # 4
    "MERCHANT",       # 5
    "PROFESSIONAL",   # 6
    "ELITE",          # 7
    "SOLDIER",        # 8: Professional military
]

# Base income multiplier per archetype (relative to mean = 1.0)
# Lower class < 1.0, Middle ~ 1.0–1.5, Upper > 1.5
ARCHETYPE_BASE_INCOME = np.array([
    0.35,  # SUBSISTENCE
    0.70,  # YEOMAN
    0.50,  # URBAN_LABORER
    0.95,  # ARTISAN
    1.10,  # CLERK
    1.40,  # MERCHANT
    1.80,  # PROFESSIONAL
    3.50,  # ELITE
    1.20,  # SOLDIER: State-funded, middle income
], dtype=np.float64)

# Base radicalization potential (how quickly each archetype radicalizes when
# unsatisfied). Urban laborers and subsistence farmers have highest potential.
ARCHETYPE_RAD_POTENTIAL = np.array([
    0.80,  # SUBSISTENCE — very high
    0.45,  # YEOMAN
    0.90,  # URBAN_LABORER — highest
    0.55,  # ARTISAN
    0.25,  # CLERK — institutionally loyal
    0.40,  # MERCHANT
    0.30,  # PROFESSIONAL
    0.60,  # ELITE — radical if threatened
    0.35,  # SOLDIER: Disciplined but can mutiny if abandoned
], dtype=np.float64)

# Political weight per archetype (contribution to hazard and trust dynamics)
# Does NOT equal population size — small elites have outsized political weight
ARCHETYPE_POLITICAL_WEIGHT = np.array([
    0.60,  # SUBSISTENCE — large numerically, low per-capita weight
    0.75,  # YEOMAN
    0.70,  # URBAN_LABORER — organized labor
    0.80,  # ARTISAN
    1.20,  # CLERK — state apparatus
    1.10,  # MERCHANT — capital access
    1.30,  # PROFESSIONAL — media/legal influence
    1.80,  # ELITE — highest per-capita weight
    1.40,  # SOLDIER — institutional leverage
], dtype=np.float64)

# Archetype class membership: 0=lower, 1=middle, 2=upper
ARCHETYPE_CLASS = np.array([0, 1, 0, 1, 1, 1, 2, 2, 1], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────── #
# PopParams dataclass                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class PopParams:
    """
    All hyperparameters for the pop subsystem.

    These are separate from GravitasParams to keep the core env clean.
    PopWrapper reads PopParams; GravitasEnv is unmodified.
    """

    # ── Dimensionality ───────────────────────────────────────────────────── #
    n_archetypes: int   = N_ARCHETYPES
    """Number of job/class archetypes. Fixed at 9; do not change unless you
    also update ARCHETYPE_BASE_INCOME / RAD_POTENTIAL / POLITICAL_WEIGHT."""

    n_ethnic_groups: int = 3
    """Number of ethnic groups per cluster. Default 3 (majority + two minorities)."""

    # ── Satisfaction ODE  dS_i/dt ────────────────────────────────────────── #
    alpha_sat: float = 0.08
    """Income-driven satisfaction growth rate."""
    beta_sat: float  = 0.12
    """Polarization × trust-deficit drain on satisfaction."""
    sat_floor: float = 0.05
    """Minimum satisfaction (existential floor — pops never fully pacified)."""

    # ── Radicalization ODE  dR_i/dt ─────────────────────────────────────── #
    alpha_rad: float = 0.10
    """Unsatisfied-pop radicalization rate (scaled by archetype_rad_potential)."""
    beta_rad:  float = 0.15
    """Satisfied-pop de-radicalization rate (scaled by cluster trust)."""
    rad_ceiling: float = 0.95
    """Maximum radicalization (prevents all archetypes going fully radical)."""

    # ── Income ODE  dY_i/dt ──────────────────────────────────────────────── #
    income_growth_base: float = 0.04
    """Base income growth rate when resources are high and hazard is low."""
    income_hazard_drain: float = 0.12
    """Income reduction per unit cluster hazard."""
    income_invest_boost: float = 0.20
    """Income multiplier boost from INVEST action (weighted by middle class share)."""

    # ── Size evolution (slow) ────────────────────────────────────────────── #
    pop_mobility_period: int = 15
    """Number of env steps between size re-distributions (mobility is slow)."""
    mobility_rate: float = 0.02
    """Fraction of population that can change archetype per mobility update."""
    upward_mobility_trust: float = 0.50
    """Cluster trust threshold above which upward mobility is enabled."""

    # ── Ethnic dynamics ──────────────────────────────────────────────────── #
    ethnic_tension_base: float = 0.08
    """Base ethnic tension growth rate when fractionalization is high."""
    ethnic_tension_decay: float = 0.05
    """Natural decay of ethnic tension (integration / contact hypothesis)."""
    decentralize_tension_relief: float = 0.20
    """Ethnic tension reduction from DECENTRALIZE action."""

    # ── Injection into GravitasEnv ODE drivers ───────────────────────────── #
    # These coefficients scale how pop aggregates affect the existing dynamics.

    pop_hazard_coeff: float   = 0.15
    """radical_mass contribution to cluster hazard h_i."""
    pop_tau_coeff: float      = 0.10
    """mean_satisfaction contribution to cluster trust dτ/dt."""
    pop_pol_coeff: float      = 0.12
    """ethnic_fractionalization contribution to systemic polarization dΠ/dt."""
    pop_phi_coeff: float      = 0.08
    """income gini contribution to fragmentation dΦ/dt."""
    pop_stability_coeff: float = 0.06
    """class_tension drag on cluster stability dσ/dt."""

    # ── Action effects on pops ───────────────────────────────────────────── #
    # Each action has a direct immediate effect on pop satisfaction / income.

    reform_sat_boost: float      = 0.08
    """Satisfaction boost to lower/middle classes from REFORM."""
    reform_rad_relief: float     = 0.06
    """Radicalization reduction from REFORM."""
    invest_income_boost: float   = 0.10
    """Income boost to middle-class archetypes from INVEST."""
    propaganda_sat_boost: float  = 0.05
    """Temporary satisfaction boost from PROPAGANDA (surface effect)."""
    propaganda_rad_boost: float  = 0.04
    """Hidden radicalization increase from PROPAGANDA."""
    military_sat_drain: float    = 0.06
    """Satisfaction drain on lower-class pops from MILITARIZE (conscription)."""
    decentral_sat_boost: float   = 0.07
    """Satisfaction boost to minority ethnic groups from DECENTRALIZE."""
    stabilize_rad_relief: float  = 0.05
    """Broad radicalization reduction from STABILIZE action."""

    # ── Reward extension ─────────────────────────────────────────────────── #
    w_pop_satisfaction: float    = 0.40
    """Reward weight for mean population satisfaction."""
    w_pop_inequality: float      = 0.30
    """Reward penalty weight for income Gini coefficient."""
    w_ethnic_tension: float      = 0.25
    """Reward penalty weight for mean ethnic tension."""

    # ── Observation ──────────────────────────────────────────────────────── #
    expose_full_pop_dist: bool   = False
    """If True, expose full P-dim pop distribution per cluster in obs
    (adds n_archetypes * max_N dims). Default False — expose only 5 aggregates."""

    def __post_init__(self) -> None:
        assert self.n_archetypes == N_ARCHETYPES, \
            f"n_archetypes must be {N_ARCHETYPES}; update archetype arrays if changing."
        assert self.n_ethnic_groups >= 1
        assert 0.0 < self.alpha_sat
        assert 0.0 < self.beta_rad
        assert self.hawkes_stationarity_check(), \
            "mobility_rate must be < 1.0"

    def hawkes_stationarity_check(self) -> bool:
        return 0.0 < self.mobility_rate < 1.0
