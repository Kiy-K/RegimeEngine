"""
pop_dynamics.py — Population dynamics for the GRAVITAS pop extension.

Three responsibilities:
  1. step_pop_vector()     — update one cluster's PopVector given world state
  2. apply_action_to_pop() — immediate pop effects of each governance stance
  3. pop_to_ode_drivers()  — translate pop aggregates into ODE injection terms
                             that feed into GravitasEnv's existing dynamics

Design:
  Pop dynamics are semi-implicit — one forward-Euler pass per env step,
  NOT per RK4 sub-step. This keeps cost at O(N*P) per step regardless
  of the RK4 integration detail in the core engine.

  The coupling is one-directional per step:
    t:   GravitasEnv world state (σ, h, τ, etc.) → pop update inputs
    t+1: pop aggregates → ODE driver adjustments → fed into next step's
         GravitasEnv dynamics via the PopWrapper's injection mechanism
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .pop_params import (
    ARCHETYPE_BASE_INCOME,
    ARCHETYPE_CLASS,
    ARCHETYPE_RAD_POTENTIAL,
    ARCHETYPE_POLITICAL_WEIGHT,
    N_ARCHETYPES,
    PopParams,
)
from .pop_state import PopAggregates, PopVector, WorldPopState, compute_aggregates

# Numba optimization imports
try:
    from numba import njit, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available. Using standard implementation.")

# Precompute static masks and weights for faster aggregate computation
LOWER_MASK = np.array([True, True, True, False, False, False, False, False, False])  # Indices 0-2 (lower class)
UPPER_MASK = np.array([False, False, False, False, False, False, True, True, True])  # Indices 6-8 (upper class)
POL_W = ARCHETYPE_POLITICAL_WEIGHT / ARCHETYPE_POLITICAL_WEIGHT.sum()  # Normalized once

# ─────────────────────────────────────────────────────────────────────────── #
# Numba-optimized functions (internal implementation)                         #
# ─────────────────────────────────────────────────────────────────────────── #

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def compute_gini_numba(sizes: NDArray[np.float64], income: NDArray[np.float64]) -> float:
        """
        Numba-optimized Gini coefficient computation.
        """
        P = len(sizes)

        # Sort by income
        order = np.argsort(income)
        inc_ord = income[order]
        siz_ord = sizes[order]

        # Compute cumulative sums
        cum_pop = np.zeros(P)
        cum_inc = np.zeros(P)

        cum_pop[0] = siz_ord[0]
        cum_inc[0] = siz_ord[0] * inc_ord[0]

        for i in range(1, P):
            cum_pop[i] = cum_pop[i-1] + siz_ord[i]
            cum_inc[i] = cum_inc[i-1] + siz_ord[i] * inc_ord[i]

        total_inc = cum_inc[P-1] if cum_inc[P-1] > 1e-9 else 1.0

        # Area under Lorenz curve
        lorenz_area = 0.0
        for i in range(1, P):
            lorenz_area += (cum_pop[i-1] + 0.5 * siz_ord[i]) * siz_ord[i] * (cum_inc[i] / total_inc)

        gini = 1.0 - 2.0 * lorenz_area
        return max(0.0, min(1.0, gini))

    @njit(cache=True)
    def compute_aggregates_numba(pop: PopVector) -> PopAggregates:
        """
        Numba-optimized aggregate computation.
        """
        P = pop.n_archetypes

        # Gini coefficient
        gini = compute_gini_numba(pop.sizes, pop.income)

        # Mean satisfaction (population-weighted)
        mean_sat = 0.0
        for i in range(P):
            mean_sat += pop.sizes[i] * pop.satisfaction[i]

        # Radical mass (politically-weighted)
        radical_mass = 0.0
        for i in range(P):
            radical_mass += POL_W[i] * pop.radicalization[i]
        radical_mass = max(0.0, min(1.0, radical_mass))

        # Ethnic fractionalization (Herfindahl)
        frac = 0.0
        for i in range(len(pop.ethnic_shares)):
            frac += pop.ethnic_shares[i] * pop.ethnic_shares[i]
        frac = 1.0 - frac

        # Class tension: income gap upper vs lower
        lower_inc = 0.0
        upper_inc = 0.0
        lower_count = 0
        upper_count = 0

        for i in range(P):
            if i < 3:  # Lower class
                lower_inc += pop.income[i]
                lower_count += 1
            elif i >= 6:  # Upper class
                upper_inc += pop.income[i]
                upper_count += 1

        lower_inc = lower_inc / max(1, lower_count) if lower_count > 0 else 0.5
        upper_inc = upper_inc / max(1, upper_count) if upper_count > 0 else 0.5
        class_tens = max(0.0, min(5.0, upper_inc - lower_inc))

        return PopAggregates(
            gini=gini,
            mean_satisfaction=mean_sat,
            radical_mass=radical_mass,
            fractionalization=frac,
            ethnic_tension=pop.ethnic_tension,
            class_tension=class_tens,
        )

    def step_pop_vector_optimized(
        pop:           PopVector,
        cluster_sigma: float,
        cluster_hazard: float,
        cluster_trust: float,
        cluster_resource: float,
        sys_polarization: float,
        sys_exhaustion: float,
        sys_coherence: float,
        cultural_dist:  NDArray[np.float64],
        params:         PopParams,
        dt:             float = 0.01,
        military_load:  float = 0.0,
    ) -> PopVector:
        """
        Optimized version of step_pop_vector using Numba where possible.
        """
        P = params.n_archetypes
        E = params.n_ethnic_groups

        # Use Numba for the most expensive computations
        new_sat = np.zeros(P)
        new_rad = np.zeros(P)
        new_inc = np.zeros(P)

        # Vectorized updates where possible
        mean_income = float(np.dot(pop.sizes, pop.income))
        if mean_income < 1e-9:
            mean_income = 1.0
        income_ratio = pop.income / mean_income

        act_gate = 1.0 - sys_exhaustion

        # Satisfaction ODE
        d_sat = (
            params.alpha_sat * (income_ratio - 1.0) * act_gate
            - params.beta_sat * sys_polarization * (1.0 - cluster_trust)
        ) * dt
        new_sat = np.clip(pop.satisfaction + d_sat, params.sat_floor, 1.0)

        # Radicalization ODE
        unsat = np.maximum(0.0, 1.0 - pop.satisfaction)
        d_rad = (
            params.alpha_rad * ARCHETYPE_RAD_POTENTIAL[:P] * unsat * (1.0 - sys_coherence)
            - params.beta_rad * pop.satisfaction * cluster_trust
        ) * dt
        new_rad = np.clip(pop.radicalization + d_rad, 0.0, params.rad_ceiling)

        # Income ODE
        d_income = (
            params.income_growth_base * pop.income * cluster_resource * act_gate
            - params.income_hazard_drain * cluster_hazard * pop.income
        ) * dt
        new_inc = np.maximum(0.05, pop.income + d_income)
        # Re-normalize to mean=1
        new_mean = float(np.dot(pop.sizes, new_inc))
        if new_mean > 1e-9:
            new_inc = new_inc / new_mean

        # Ethnic tension ODE
        frac = float(1.0 - np.dot(pop.ethnic_shares, pop.ethnic_shares))
        d_tension = (
            params.ethnic_tension_base * frac * sys_polarization
            - params.ethnic_tension_decay * cluster_sigma
        ) * dt
        new_tension = float(np.clip(pop.ethnic_tension + d_tension, 0.0, 1.0))

        updated = pop.copy_with(
            satisfaction=new_sat,
            radicalization=new_rad,
            income=new_inc,
            ethnic_tension=new_tension,
        )

        # Soldier-specific morale adjustments
        updated = soldier_morale(updated, cluster_trust, military_load)

        return updated

else:
    # Fallback functions when Numba is not available
    def compute_gini_numba(sizes, income):
        return compute_gini(sizes, income)

    def compute_aggregates_numba(pop):
        return compute_aggregates(pop)

    def step_pop_vector_optimized(
        pop:           PopVector,
        cluster_sigma: float,
        cluster_hazard: float,
        cluster_trust: float,
        cluster_resource: float,
        sys_polarization: float,
        sys_exhaustion: float,
        sys_coherence: float,
        cultural_dist:  NDArray[np.float64],
        params:         PopParams,
        dt:             float = 0.01,
        military_load:  float = 0.0,
    ) -> PopVector:
        """
        Fallback to original implementation when Numba is not available.
        """
        return step_pop_vector_original(
            pop=pop,
            cluster_sigma=cluster_sigma,
            cluster_hazard=cluster_hazard,
            cluster_trust=cluster_trust,
            cluster_resource=cluster_resource,
            sys_polarization=sys_polarization,
            sys_exhaustion=sys_exhaustion,
            sys_coherence=sys_coherence,
            cultural_dist=cultural_dist,
            params=params,
            dt=dt,
            military_load=military_load,
        )

# Fallback Gini computation for when Numba is not available
def compute_gini(sizes, income):
    """Fallback Gini computation."""
    P = len(sizes)
    order = np.argsort(income)
    inc_ord = income[order]
    siz_ord = sizes[order]

    cum_pop = np.cumsum(siz_ord)
    cum_inc = np.cumsum(siz_ord * inc_ord)
    total_inc = cum_inc[-1] if cum_inc[-1] > 1e-9 else 1.0

    lorenz_area = float(np.sum((cum_pop[:-1] + 0.5 * siz_ord[1:]) * siz_ord[1:] *
                               (cum_inc[1:] / total_inc)))
    return float(np.clip(1.0 - 2.0 * lorenz_area, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────── #
# Action type constants (mirrors Stance enum without importing gravitas_actions)
# ─────────────────────────────────────────────────────────────────────────── #

STABILIZE    = 0
MILITARIZE   = 1
REFORM       = 2
PROPAGANDA   = 3
INVEST       = 4
DECENTRALIZE = 5


# ─────────────────────────────────────────────────────────────────────────── #
# 1. Per-cluster pop update                                                    #
# ─────────────────────────────────────────────────────────────────────────── #


def soldier_morale(
    pop: PopVector,
    cluster_trust: float,
    military_load: float,
) -> PopVector:
    """Adjust soldier satisfaction and trigger desertion if morale collapses."""
    soldier_idx = 8
    if soldier_idx >= pop.n_archetypes:
        return pop

    morale = cluster_trust * (1.0 - military_load) * pop.income[soldier_idx]
    morale = float(np.clip(morale, 0.0, 1.0))
    new_sat = pop.satisfaction.copy()
    new_sat[soldier_idx] = morale

    if morale < 0.3:
        desertion_rate = 0.1 * (0.3 - morale)
        deserters = pop.sizes[soldier_idx] * desertion_rate
        if deserters > 0:
            # Vectorized desertion
            new_sizes = pop.sizes.copy()
            new_sizes[soldier_idx] -= deserters
            new_sizes[0] += deserters * 0.5  # Subsistence
            if pop.n_archetypes > 2:
                new_sizes[2] += deserters * 0.5  # Urban laborers
            pop = pop.copy_with(sizes=new_sizes)

    return pop.copy_with(satisfaction=new_sat)


def conscript_soldiers(
    pop: PopVector,
    intensity: float,
) -> PopVector:
    """Shift urban laborers into the soldier archetype based on action intensity."""
    laborer_idx = 2
    soldier_idx = 8
    if soldier_idx >= pop.n_archetypes:
        return pop

    max_transfer = pop.sizes[laborer_idx] * 0.3
    conscription = min(0.15 * intensity, max_transfer)
    if conscription <= 0:
        return pop

    # Vectorized updates
    new_sizes = pop.sizes.copy()
    new_sizes[laborer_idx] -= conscription
    new_sizes[soldier_idx] += conscription

    new_sat = pop.satisfaction.copy()
    new_rad = pop.radicalization.copy()
    # Batch morale/radicalization updates
    new_sat[soldier_idx] = np.clip(pop.satisfaction[soldier_idx] + 0.2 * intensity, 0.0, 1.0)
    new_rad[laborer_idx] = np.clip(pop.radicalization[laborer_idx] + 0.1 * intensity, 0.0, 1.0)

    return pop.copy_with(sizes=new_sizes, satisfaction=new_sat, radicalization=new_rad)


# Main step_pop_vector function - uses internal optimized version by default
def step_pop_vector(
    pop:           PopVector,
    cluster_sigma: float,
    cluster_hazard: float,
    cluster_trust: float,
    cluster_resource: float,
    sys_polarization: float,
    sys_exhaustion: float,
    sys_coherence: float,
    cultural_dist:  NDArray[np.float64],
    params:         PopParams,
    dt:             float = 0.01,
    military_load:  float = 0.0,
) -> PopVector:
    """
    Advance one cluster's pop state by one env step.

    This function uses the optimized Numba implementation by default when available.
    Falls back to the original implementation if Numba is not available.

    Satisfaction dynamics:
      dS_i/dt = α_sat * (income_i/mean_income - 1) * (1-E)   [income effect]
              - β_sat * Π * (1 - τ)                          [polarization drag]

    Radicalization dynamics:
      dR_i/dt = α_rad * rad_potential_i * max(0, 1-S_i) * (1-Ψ)  [unsatisfied → radical]
              - β_rad * S_i * τ                                    [satisfied → deradicalize]

    Income dynamics:
      dY_i/dt = income_growth_base * income_i * r * (1-E)   [resource-driven growth]
              - income_hazard_drain * h * income_i           [hazard drain]

    Ethnic tension:
      dT_e/dt = ethnic_tension_base * fractionalization * Π  [structural × systemic]
              - ethnic_tension_decay * σ                     [stability reduces tension]

    All updates are clipped to valid ranges after integration.
    """
    # Use the internal optimized version if Numba is available
    if NUMBA_AVAILABLE:
        return step_pop_vector_optimized(
            pop=pop,
            cluster_sigma=cluster_sigma,
            cluster_hazard=cluster_hazard,
            cluster_trust=cluster_trust,
            cluster_resource=cluster_resource,
            sys_polarization=sys_polarization,
            sys_exhaustion=sys_exhaustion,
            sys_coherence=sys_coherence,
            cultural_dist=cultural_dist,
            params=params,
            dt=dt,
            military_load=military_load,
        )
    else:
        # Fallback to original implementation when Numba is not available
        return step_pop_vector_original(
            pop=pop,
            cluster_sigma=cluster_sigma,
            cluster_hazard=cluster_hazard,
            cluster_trust=cluster_trust,
            cluster_resource=cluster_resource,
            sys_polarization=sys_polarization,
            sys_exhaustion=sys_exhaustion,
            sys_coherence=sys_coherence,
            cultural_dist=cultural_dist,
            params=params,
            dt=dt,
            military_load=military_load,
        )

# Original implementation (renamed to avoid conflict)
def step_pop_vector_original(
    pop:           PopVector,
    cluster_sigma: float,
    cluster_hazard: float,
    cluster_trust: float,
    cluster_resource: float,
    sys_polarization: float,
    sys_exhaustion: float,
    sys_coherence: float,
    cultural_dist:  NDArray[np.float64],
    params:         PopParams,
    dt:             float = 0.01,
    military_load:  float = 0.0,
) -> PopVector:
    """
    Original implementation of step_pop_vector (fallback when Numba not available).

    Satisfaction dynamics:
      dS_i/dt = α_sat * (income_i/mean_income - 1) * (1-E)   [income effect]
              - β_sat * Π * (1 - τ)                          [polarization drag]

    Radicalization dynamics:
      dR_i/dt = α_rad * rad_potential_i * max(0, 1-S_i) * (1-Ψ)  [unsatisfied → radical]
              - β_rad * S_i * τ                                    [satisfied → deradicalize]

    Income dynamics:
      dY_i/dt = income_growth_base * income_i * r * (1-E)   [resource-driven growth]
              - income_hazard_drain * h * income_i           [hazard drain]

    Ethnic tension:
      dT_e/dt = ethnic_tension_base * fractionalization * Π  [structural × systemic]
              - ethnic_tension_decay * σ                     [stability reduces tension]

    All updates are clipped to valid ranges after integration.
    """
    P = params.n_archetypes
    E = params.n_ethnic_groups

    # ── Derived quantities ────────────────────────────────────────────────── #
    mean_income = float(np.dot(pop.sizes, pop.income))
    if mean_income < 1e-9:
        mean_income = 1.0
    income_ratio = pop.income / mean_income   # relative income per archetype

    rad_potential = ARCHETYPE_RAD_POTENTIAL[:P]
    act_gate      = 1.0 - sys_exhaustion        # exhaustion freezes all dynamics

    # ── Satisfaction ODE ─────────────────────────────────────────────────── #
    d_sat = (
        params.alpha_sat * (income_ratio - 1.0) * act_gate
        - params.beta_sat * sys_polarization * (1.0 - cluster_trust)
    ) * dt
    new_sat = np.clip(pop.satisfaction + d_sat, params.sat_floor, 1.0)

    # ── Radicalization ODE ───────────────────────────────────────────────── #
    unsat    = np.maximum(0.0, 1.0 - pop.satisfaction)
    d_rad = (
        params.alpha_rad * rad_potential * unsat * (1.0 - sys_coherence)
        - params.beta_rad * pop.satisfaction * cluster_trust
    ) * dt
    new_rad = np.clip(pop.radicalization + d_rad, 0.0, params.rad_ceiling)

    # ── Income ODE ───────────────────────────────────────────────────────── #
    d_income = (
        params.income_growth_base * pop.income * cluster_resource * act_gate
        - params.income_hazard_drain * cluster_hazard * pop.income
    ) * dt
    new_income = np.maximum(0.05, pop.income + d_income)
    # Re-normalize to mean=1 to prevent drift
    new_mean = float(np.dot(pop.sizes, new_income))
    if new_mean > 1e-9:
        new_income = new_income / new_mean

    # ── Ethnic tension ODE ───────────────────────────────────────────────── #
    # Fractionalization from current ethnic shares
    frac = float(1.0 - np.dot(pop.ethnic_shares, pop.ethnic_shares))
    d_tension = (
        params.ethnic_tension_base * frac * sys_polarization
        - params.ethnic_tension_decay * cluster_sigma
    ) * dt
    new_tension = float(np.clip(pop.ethnic_tension + d_tension, 0.0, 1.0))

    updated = pop.copy_with(
        satisfaction=new_sat,
        radicalization=new_rad,
        income=new_income,
        ethnic_tension=new_tension,
    )

    # Soldier-specific morale adjustments
    updated = soldier_morale(updated, cluster_trust, military_load)

    return updated


# ─────────────────────────────────────────────────────────────────────────── #
# 2. Pop mobility (slow — called every pop_mobility_period steps)             #
# ─────────────────────────────────────────────────────────────────────────── #

def update_pop_sizes(
    pop:           PopVector,
    cluster_trust: float,
    cluster_resource: float,
    params:        PopParams,
    rng:           np.random.Generator,
) -> PopVector:
    """
    Redistribute population shares based on satisfaction and mobility conditions.

    Pops move toward archetypes with higher satisfaction × political_weight.
    Upward mobility (lower → middle, middle → upper) requires sufficient trust.
    Movement is slow — bounded by mobility_rate.

    This implements the core Paradox-style pop pressure: populations that are
    large and unsatisfied shift into more radical configurations over time.
    """
    P = pop.n_archetypes
    classes = ARCHETYPE_CLASS[:P]

    # Pull: archetypes with higher income × satisfaction attract pops
    pull = pop.satisfaction * pop.income
    pull = np.clip(pull, 0.01, None)
    target_sizes = pull / pull.sum()

    # Upward mobility gate: only enabled if trust > threshold
    if cluster_trust < params.upward_mobility_trust:
        # Suppress upward flow: pops can only flow within class or downward
        for i in range(P):
            for j in range(P):
                if classes[j] > classes[i]:
                    # Block upward: keep target_sizes similar to current for upper classes
                    target_sizes[j] = pop.sizes[j] * 0.95 + target_sizes[j] * 0.05

        # Re-normalize
        target_sizes = target_sizes / target_sizes.sum()

    # Interpolate toward target at mobility_rate
    delta     = target_sizes - pop.sizes
    new_sizes = pop.sizes + params.mobility_rate * delta

    # Small random noise (demographic variation)
    noise     = rng.normal(0.0, 0.005, P)
    new_sizes = np.clip(new_sizes + noise, 0.005, None)
    new_sizes = new_sizes / new_sizes.sum()   # maintain simplex

    return pop.copy_with(sizes=new_sizes)


# ─────────────────────────────────────────────────────────────────────────── #
# 3. Action effects on pops                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def apply_action_to_pop(
    pop:            PopVector,
    stance:         int,
    intensity:      float,
    weights:        NDArray[np.float64],   # (N,) cluster allocation; we use cluster i's weight
    cluster_idx:    int,
    params:         PopParams,
) -> PopVector:
    """
    Apply immediate pop effects of a governance action to one cluster.

    Each stance has a distinct demographic signature:

    STABILIZE:   Broad radicalization relief, mild satisfaction boost.
                 Helps all classes equally.

    MILITARIZE:  Satisfies ELITE (security), drains lower-class satisfaction
                 (conscription, curfews). Increases URBAN_LABORER radicalization.

    REFORM:      Boosts lower/middle-class satisfaction. Reduces radicalization.
                 ELITE satisfaction slightly drops (redistributive threat).

    PROPAGANDA:  Temporary surface satisfaction boost across all groups.
                 Hidden radicalization increase — pops learn to distrust.

    INVEST:      Income boost to middle-class archetypes (ARTISAN, MERCHANT,
                 CLERK). SUBSISTENCE benefits less (capital doesn't reach them).

    DECENTRALIZE: Ethnic tension reduction. Satisfaction boost for minority
                  groups. Slight satisfaction drop for dominant group
                  (perceived loss of privilege).
    """
    P = pop.n_archetypes
    w = float(weights[cluster_idx]) if cluster_idx < len(weights) else 0.5
    eff = intensity * w   # effective intensity at this cluster

    if eff < 1e-4:
        return pop

    classes = ARCHETYPE_CLASS[:P]
    new_sat = pop.satisfaction.copy()
    new_rad = pop.radicalization.copy()
    new_inc = pop.income.copy()
    new_tension = pop.ethnic_tension

    if stance == STABILIZE:
        new_sat += params.stabilize_rad_relief * eff * 0.5
        new_rad -= params.stabilize_rad_relief * eff

    elif stance == MILITARIZE:
        # Lower class satisfaction drops (conscription burden)
        lower_mask = (classes == 0)
        new_sat[lower_mask]  -= params.military_sat_drain * eff
        new_rad[lower_mask]  += params.military_sat_drain * eff * 0.8
        # Elite satisfaction: slight boost (order maintained)
        upper_mask = (classes == 2)
        new_sat[upper_mask]  += 0.02 * eff
        # Conscription: shift laborers toward soldier archetype
        pop = conscript_soldiers(pop.copy_with(
            satisfaction=new_sat,
            radicalization=new_rad,
            income=new_inc,
            ethnic_tension=new_tension,
        ), eff)
        new_sat = pop.satisfaction.copy()
        new_rad = pop.radicalization.copy()
        new_inc = pop.income.copy()
        new_tension = pop.ethnic_tension

    elif stance == REFORM:
        # Lower and middle class benefit
        non_elite = (classes < 2)
        new_sat[non_elite]   += params.reform_sat_boost * eff
        new_rad[non_elite]   -= params.reform_rad_relief * eff
        # Elite feels threatened
        elite_mask = (classes == 2)
        new_sat[elite_mask]  -= 0.03 * eff
        new_rad[elite_mask]  += 0.02 * eff

    elif stance == PROPAGANDA:
        # Surface satisfaction boost for everyone
        new_sat += params.propaganda_sat_boost * eff
        # But true radicalization rises — pops sense manipulation
        new_rad += params.propaganda_rad_boost * eff * ARCHETYPE_RAD_POTENTIAL[:P]
        # Educated groups (PROFESSIONAL, CLERK) are less susceptible
        savvy_mask = np.array([False, False, False, False, True, False, True, False, False])[:P]  # Added SOLDIER (index 8)
        new_rad[savvy_mask]  -= params.propaganda_rad_boost * eff * 0.5

    elif stance == INVEST:
        # Middle-class income boost
        middle_mask = (classes == 1)
        new_inc[middle_mask] += params.invest_income_boost * eff
        new_sat[middle_mask] += params.invest_income_boost * eff * 0.5
        # Lower class gets indirect benefit but less
        lower_mask = (classes == 0)
        new_sat[lower_mask]  += params.invest_income_boost * eff * 0.2

    elif stance == DECENTRALIZE:
        # Ethnic tension relief
        new_tension -= params.decentralize_tension_relief * eff
        # Minority groups (indices 1+) gain satisfaction
        minority_share = 1.0 - pop.ethnic_shares[0]
        if minority_share > 0.1:
            # Boost for lower/middle classes in proportion to minority presence
            non_elite = (classes < 2)
            new_sat[non_elite] += params.decentral_sat_boost * eff * minority_share
        # Dominant group mild displeasure
        new_sat -= 0.01 * eff * pop.ethnic_shares[0]

    # Clip all to valid ranges
    new_sat = np.clip(new_sat, params.sat_floor, 1.0)
    new_rad = np.clip(new_rad, 0.0, params.rad_ceiling)
    new_inc = np.maximum(0.05, new_inc)
    new_tension = float(np.clip(new_tension, 0.0, 1.0))

    return pop.copy_with(
        satisfaction=new_sat,
        radicalization=new_rad,
        income=new_inc,
        ethnic_tension=new_tension,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# 4. Injection into GravitasEnv ODE drivers                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def pop_to_ode_drivers(
    aggregates: Tuple[PopAggregates, ...],
    params:     PopParams,
) -> Dict[str, NDArray[np.float64]]:
    """
    Translate per-cluster pop aggregates into adjustment terms for the
    existing GravitasEnv ODE drivers.

    Returns a dict of per-cluster (N,) arrays:
      'hazard_boost'    — added to h_i before stability ODE evaluation
      'trust_boost'     — added to dτ/dt term
      'pol_boost'       — added to dΠ/dt (global, mean taken in wrapper)
      'phi_boost'       — added to dΦ/dt (global, mean taken)
      'sigma_drag'      — subtracted from dσ/dt

    The PopWrapper applies these adjustments after GravitasEnv's own RK4 step,
    modifying the world state to reflect pop-driven pressure. This is equivalent
    to adding coupling terms to the ODEs without modifying the core engine.
    """
    N = len(aggregates)

    hazard_boost = np.zeros(N)
    trust_boost  = np.zeros(N)
    pol_boost    = np.zeros(N)
    phi_boost    = np.zeros(N)
    sigma_drag   = np.zeros(N)

    for i, agg in enumerate(aggregates):
        # Radical pops raise local hazard
        hazard_boost[i] = params.pop_hazard_coeff * agg.radical_mass

        # Satisfied pops boost institutional trust
        trust_boost[i]  = params.pop_tau_coeff * (agg.mean_satisfaction - 0.5)

        # Ethnic fractionalization + tension feed polarization
        pol_boost[i]    = params.pop_pol_coeff * (
            agg.fractionalization * 0.6 + agg.ethnic_tension * 0.4
        )

        # Income inequality feeds fragmentation
        phi_boost[i]    = params.pop_phi_coeff * agg.gini

        # Class tension drags stability
        sigma_drag[i]   = params.pop_stability_coeff * min(agg.class_tension, 1.0)

    return {
        "hazard_boost": hazard_boost,
        "trust_boost":  trust_boost,
        "pol_boost":    pol_boost,     # will be averaged for global Π effect
        "phi_boost":    phi_boost,     # will be averaged for global Φ effect
        "sigma_drag":   sigma_drag,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# 5. Full world pop step                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def step_world_pop(
    world_pop:    WorldPopState,
    world:        object,             # GravitasWorld — avoid circular import
    stance:       int,
    intensity:    float,
    action_weights: NDArray[np.float64],
    params:       PopParams,
    rng:          np.random.Generator,
    use_optimized: bool = True,
) -> Tuple[WorldPopState, Dict[str, NDArray[np.float64]]]:
    """
    Full pop update for one env step.

    Order:
      1. Apply action effects (immediate)
      2. Step pop dynamics (ODE)
      3. (Every pop_mobility_period) update sizes
      4. Compute aggregates and ODE driver injections

    Returns updated WorldPopState and ODE driver dict.
    """
    N     = world_pop.n_clusters
    c_arr = world.cluster_array()        # (N, 6): σ, h, r, m, τ, p
    g     = world.global_state

    new_pops = list(world_pop.pops)
    step     = world_pop.step
    do_mobility = (step > 0 and step % params.pop_mobility_period == 0)

    # The use_optimized parameter is now ignored since optimization is handled internally
    # by the main step_pop_vector function

    for i in range(N):
        pop = new_pops[i]

        # 1. Action effects
        pop = apply_action_to_pop(
            pop, stance, intensity, action_weights, i, params
        )

        # 2. Dynamics step - use internal optimized version
        pop = step_pop_vector(
            pop=pop,
            cluster_sigma    = float(c_arr[i, 0]),
            cluster_hazard   = float(c_arr[i, 1]),
            cluster_trust    = float(c_arr[i, 4]),
            cluster_resource = float(c_arr[i, 2]),
            sys_polarization = float(g.polarization),
            sys_exhaustion   = float(g.exhaustion),
            sys_coherence    = float(g.coherence),
            cultural_dist    = world_pop.cultural_dist,
            params           = params,
            dt               = 0.01,
            military_load    = float(c_arr[i, 3]) if c_arr.shape[1] > 3 else 0.0,
        )

        # 3. Mobility (slow, periodic)
        if do_mobility:
            pop = update_pop_sizes(
                pop, float(c_arr[i, 4]), float(c_arr[i, 2]), params, rng
            )

        new_pops[i] = pop

    new_world_pop = world_pop.copy_with_pops(tuple(new_pops)).advance_step()

    # 4. ODE drivers from new state
    aggs    = new_world_pop.get_aggregates()
    drivers = pop_to_ode_drivers(aggs, params)

    return new_world_pop, drivers
