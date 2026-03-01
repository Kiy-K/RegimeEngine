"""
pop_state.py — Immutable population state containers for the GRAVITAS pop extension.

Hierarchy:
  PopVector         — demographic state for one cluster (job/class + ethnicity)
  PopAggregates     — derived scalars computed from one PopVector
  WorldPopState     — collection of PopVectors across all clusters

All containers are immutable (frozen dataclasses or tuples). Mutation
produces new objects, identical to the core gravitas_state.py pattern.

Observation contribution (per cluster, 5 scalars):
  [gini, mean_satisfaction, radical_mass, fractionalization, ethnic_tension]
Total obs addition: 5 * max_N (cheap — same cost as adding one ODE variable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .pop_params import (
    N_ARCHETYPES,
    ARCHETYPE_BASE_INCOME,
    ARCHETYPE_POLITICAL_WEIGHT,
    ARCHETYPE_CLASS,
    PopParams,
)

# Precompute static masks and weights for faster aggregate computation
LOWER_MASK = np.array([True, True, True, False, False, False, False, False, False])  # Indices 0-2 (lower class)
UPPER_MASK = np.array([False, False, False, False, False, False, True, True, True])  # Indices 6-8 (upper class)
POL_W = ARCHETYPE_POLITICAL_WEIGHT / ARCHETYPE_POLITICAL_WEIGHT.sum()  # Normalized once


# ─────────────────────────────────────────────────────────────────────────── #
# PopVector — state for one cluster's population                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class PopVector:
    """
    Full demographic state for one cluster.

    sizes[i]          — share of total population in archetype i (simplex)
    satisfaction[i]   — satisfaction level of archetype i in [0, 1]
    radicalization[i] — radicalization of archetype i in [0, 1]
    income[i]         — relative income of archetype i (mean-normalized, > 0)
    ethnic_shares[j]  — share of ethnic group j in this cluster (simplex)
    ethnic_tension    — scalar tension level in [0, 1]

    All arrays have dtype float64. Validation is lazy (done in __post_init__
    of WorldPopState, not here, for performance during dynamics updates).
    """
    sizes:          NDArray[np.float64]   # (P,)
    satisfaction:   NDArray[np.float64]   # (P,)
    radicalization: NDArray[np.float64]   # (P,)
    income:         NDArray[np.float64]   # (P,)  — mean-normalized
    ethnic_shares:  NDArray[np.float64]   # (E,)
    ethnic_tension: float                 # scalar

    def copy_with(self, **kwargs) -> "PopVector":
        """Return a new PopVector with specified fields replaced."""
        return PopVector(
            sizes          = kwargs.get("sizes",          self.sizes),
            satisfaction   = kwargs.get("satisfaction",   self.satisfaction),
            radicalization = kwargs.get("radicalization",  self.radicalization),
            income         = kwargs.get("income",         self.income),
            ethnic_shares  = kwargs.get("ethnic_shares",  self.ethnic_shares),
            ethnic_tension = kwargs.get("ethnic_tension", self.ethnic_tension),
        )

    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array: [sizes | sat | rad | income | ethnic | tension]."""
        return np.concatenate([
            self.sizes,
            self.satisfaction,
            self.radicalization,
            self.income,
            self.ethnic_shares,
            [self.ethnic_tension],
        ])

    @property
    def n_archetypes(self) -> int:
        return len(self.sizes)

    @property
    def n_ethnic(self) -> int:
        return len(self.ethnic_shares)


# ─────────────────────────────────────────────────────────────────────────── #
# PopAggregates — derived signals used as ODE drivers and obs features       #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class PopAggregates:
    """
    Five scalar aggregates computed from one PopVector.

    These are what actually interface with the GravitasEnv ODE system.
    Computing them is O(P) — trivially fast.

    gini              — income Gini coefficient [0, 1). 0 = perfect equality.
                        Drives fragmentation Φ.
    mean_satisfaction — population-weighted mean satisfaction [0, 1].
                        Drives institutional trust τ.
    radical_mass      — politically-weighted radicalization [0, 1].
                        Drives cluster hazard h.
    fractionalization — Herfindahl ethnic diversity index [0, 1).
                        0 = fully homogeneous, drives polarization Π.
    ethnic_tension    — current inter-group tension [0, 1].
                        Additional polarization driver.
    class_tension     — income gap between top and bottom classes [0, ∞).
                        Drives stability drag.
    """
    gini:              float
    mean_satisfaction: float
    radical_mass:      float
    fractionalization: float
    ethnic_tension:    float
    class_tension:     float

    def to_obs_vector(self) -> NDArray[np.float32]:
        """Return 5-element obs contribution [gini, sat, rad, frac, tension]."""
        return np.array([
            self.gini,
            self.mean_satisfaction,
            self.radical_mass,
            self.fractionalization,
            self.ethnic_tension,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "pop_gini":              self.gini,
            "pop_mean_satisfaction": self.mean_satisfaction,
            "pop_radical_mass":      self.radical_mass,
            "pop_fractionalization": self.fractionalization,
            "pop_ethnic_tension":    self.ethnic_tension,
            "pop_class_tension":     self.class_tension,
        }


def compute_aggregates(pop: PopVector) -> PopAggregates:
    """
    Compute all six aggregate signals from a PopVector. O(P + E).

    Gini: standard formula on income-weighted population shares.
    Radical mass: dot(sizes, radicalization) × political_weight, normalized.
    Fractionalization: Herfindahl index (1 - Σ share_i²).
    Class tension: gap between mean upper-class and lower-class income.
    """
    P = pop.n_archetypes

    # ── Gini coefficient ─────────────────────────────────────────────────── #
    # Sort by income
    order   = np.argsort(pop.income)
    inc_ord = pop.income[order]
    siz_ord = pop.sizes[order]
    # Standard Lorenz-curve Gini
    cum_pop = np.cumsum(siz_ord)
    cum_inc = np.cumsum(siz_ord * inc_ord)
    total_inc = cum_inc[-1] if cum_inc[-1] > 1e-9 else 1.0
    # Area under Lorenz curve
    lorenz_area = float(np.sum((cum_pop[:-1] + 0.5 * siz_ord[1:]) * siz_ord[1:] *
                               (cum_inc[1:] / total_inc)))
    gini = float(np.clip(1.0 - 2.0 * lorenz_area, 0.0, 1.0))

    # ── Mean satisfaction (population-weighted) ──────────────────────────── #
    mean_sat = float(np.dot(pop.sizes, pop.satisfaction))

    # ── Radical mass (politically-weighted) ──────────────────────────────── #
    # Political weight is NOT population size — small elites count more
    radical_mass = float(np.clip(np.dot(POL_W[:P], pop.radicalization), 0.0, 1.0))

    # ── Ethnic fractionalization (Herfindahl) ─────────────────────────────── #
    frac = float(np.clip(1.0 - np.dot(pop.ethnic_shares, pop.ethnic_shares), 0.0, 1.0))

    # ── Class tension: income gap upper vs lower ──────────────────────────── #
    lower_mask  = LOWER_MASK[:P]
    upper_mask  = UPPER_MASK[:P]
    lower_inc   = float(np.mean(pop.income[lower_mask])) if lower_mask.any() else 0.5
    upper_inc   = float(np.mean(pop.income[upper_mask])) if upper_mask.any() else 0.5
    class_tens  = float(np.clip(upper_inc - lower_inc, 0.0, 5.0))

    return PopAggregates(
        gini=gini,
        mean_satisfaction=mean_sat,
        radical_mass=radical_mass,
        fractionalization=frac,
        ethnic_tension=pop.ethnic_tension,
        class_tension=class_tens,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# WorldPopState — all clusters' pop vectors                                   #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class WorldPopState:
    """
    Population state for all clusters in one episode step.

    pops            — tuple of PopVector, one per cluster
    cultural_dist   — (E, E) cultural distance matrix, symmetric, 0 diagonal
    step            — current env step (used for mobility period check)
    """
    pops:          Tuple[PopVector, ...]
    cultural_dist: NDArray[np.float64]   # (E, E)
    step:          int = 0

    @property
    def n_clusters(self) -> int:
        return len(self.pops)

    def get_aggregates(self) -> Tuple[PopAggregates, ...]:
        """Compute aggregates for all clusters. Cheap — O(N*P)."""
        return tuple(compute_aggregates(p) for p in self.pops)

    def obs_matrix(self, max_N: int, cached_aggs: Optional[List[PopAggregates]] = None) -> NDArray[np.float32]:
        """
        Return (max_N, 5) observation matrix, zero-padded if N < max_N.
        Row i = [gini, mean_sat, radical_mass, fractionalization, ethnic_tension]
        """
        if cached_aggs is not None:
            aggs = cached_aggs
        else:
            aggs = self.get_aggregates()
        mat  = np.zeros((max_N, 5), dtype=np.float32)
        for i, agg in enumerate(aggs):
            if i >= max_N:
                break
            mat[i] = agg.to_obs_vector()
        return mat

    def obs_flat(self, max_N: int, cached_aggs: Optional[List[PopAggregates]] = None) -> NDArray[np.float32]:
        """Return flattened (max_N * 5,) pop observation vector."""
        return self.obs_matrix(max_N, cached_aggs).flatten()

    def copy_with_pops(self, pops: Tuple[PopVector, ...]) -> "WorldPopState":
        return WorldPopState(
            pops=pops,
            cultural_dist=self.cultural_dist,
            step=self.step,
        )

    def advance_step(self) -> "WorldPopState":
        return WorldPopState(
            pops=self.pops,
            cultural_dist=self.cultural_dist,
            step=self.step + 1,
        )

    def mean_aggregates(self) -> PopAggregates:
        """Mean aggregates across all clusters — useful for global reward."""
        aggs = self.get_aggregates()
        return PopAggregates(
            gini              = float(np.mean([a.gini              for a in aggs])),
            mean_satisfaction = float(np.mean([a.mean_satisfaction for a in aggs])),
            radical_mass      = float(np.mean([a.radical_mass      for a in aggs])),
            fractionalization = float(np.mean([a.fractionalization for a in aggs])),
            ethnic_tension    = float(np.mean([a.ethnic_tension    for a in aggs])),
            class_tension     = float(np.mean([a.class_tension     for a in aggs])),
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Initialization helpers                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def initialize_pop_vector(
    params: PopParams,
    rng: np.random.Generator,
    cluster_resource: float = 0.6,
    cluster_trust: float    = 0.5,
) -> PopVector:
    """
    Initialize a PopVector for one cluster at episode start.

    Sizes are randomized with a bias toward agrarian archetypes
    (SUBSISTENCE + YEOMAN large, ELITE small). This reflects realistic
    starting conditions for a regime facing governance challenges.

    Starting satisfaction is correlated with base income (richer archetypes
    start more satisfied) and cluster resource level.
    """
    P = params.n_archetypes
    E = params.n_ethnic_groups

    # Sizes: Dirichlet with concentration reflecting realistic distribution
    # SUBSISTENCE/LABORER heavy, ELITE light
    concentration = np.array([3.0, 2.0, 2.5, 1.8, 1.2, 1.0, 0.6, 0.3, 0.2])[:P]  # Added SOLDIER (index 8)
    raw  = rng.dirichlet(concentration)
    sizes = raw.astype(np.float64)

    # Income: base income + resource modifier + noise
    income = ARCHETYPE_BASE_INCOME[:P].copy()
    income *= (0.7 + 0.6 * cluster_resource)   # resource scales all incomes
    income += rng.uniform(-0.05, 0.05, P)
    income = np.clip(income, 0.05, None)
    # Mean-normalize
    income = income / (np.dot(sizes, income) + 1e-9)

    # Satisfaction: driven by income relative to mean
    raw_sat = 0.3 + 0.4 * income / (income.max() + 1e-9) + 0.2 * cluster_trust
    satisfaction = np.clip(raw_sat + rng.uniform(-0.05, 0.05, P), 0.0, 1.0)

    # Radicalization: inversely correlated with satisfaction, weighted by potential
    from .pop_params import ARCHETYPE_RAD_POTENTIAL
    rad_potential = ARCHETYPE_RAD_POTENTIAL[:P]
    raw_rad = rad_potential * (1.0 - satisfaction) * rng.uniform(0.5, 1.0, P)
    radicalization = np.clip(raw_rad, 0.0, params.rad_ceiling)

    # Ethnic composition: Dirichlet — first group is majority by default
    ethnic_concentration = np.ones(E) * 0.8
    ethnic_concentration[0] = 3.0   # first group is majority
    ethnic_shares = rng.dirichlet(ethnic_concentration).astype(np.float64)

    # Initial ethnic tension: low at start
    ethnic_tension = float(rng.uniform(0.02, 0.15))

    return PopVector(
        sizes=sizes,
        satisfaction=satisfaction,
        radicalization=radicalization,
        income=income,
        ethnic_shares=ethnic_shares,
        ethnic_tension=ethnic_tension,
    )


def initialize_cultural_distance(
    n_ethnic_groups: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Initialize symmetric cultural distance matrix with zero diagonal.
    Random distances in [0.2, 0.8] — not extreme in either direction.
    """
    E   = n_ethnic_groups
    raw = rng.uniform(0.2, 0.8, (E, E))
    # Symmetrize
    dist = (raw + raw.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def initialize_world_pop(
    n_clusters: int,
    params: PopParams,
    rng: np.random.Generator,
    cluster_resources: Optional[NDArray] = None,
    cluster_trusts: Optional[NDArray]    = None,
) -> WorldPopState:
    """Initialize WorldPopState for all clusters at episode start."""
    pops = []
    for i in range(n_clusters):
        res   = float(cluster_resources[i]) if cluster_resources is not None else 0.6
        trust = float(cluster_trusts[i])    if cluster_trusts    is not None else 0.5
        pops.append(initialize_pop_vector(params, rng, res, trust))

    cultural_dist = initialize_cultural_distance(params.n_ethnic_groups, rng)

    return WorldPopState(
        pops=tuple(pops),
        cultural_dist=cultural_dist,
        step=0,
    )
