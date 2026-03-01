"""
economy.py — Per-cluster economic dynamics for GRAVITAS.

Economic state per cluster — 4 variables ∈ [0, 1]:
  G  — GDP index          (0 = collapsed  │ 0.5 = peacetime neutral │ 1 = thriving)
  U  — Unemployment rate  (0 = full employment │ 1 = total unemployment)
  D  — Debt ratio         (0 = debt-free  │ 1 = fiscal crisis)
  I  — Industrial capacity (0 = no industry │ 1 = full industrial base)

Stored as (max_N, 4) NDArray; column order: G, U, D, I

═══════════════════════════════════════════════════════════════════════════════
ODE SYSTEM

  dG/dt =  eco_gdp_recovery · σ · r · labor_mult         # stable+labour grows GDP
          + eco_trade_gain · adj_G_mean · alliance_bonus   # trade from neighbours
          - eco_gdp_hazard_cost · h · G                    # hazard destroys output
          - eco_gdp_debt_drag · D · G                      # debt service drag

  dU/dt =  eco_unemp_hazard_rate · h · (1 - U)            # hazard creates unemployment
          + eco_unemp_military_draft · m · (1 - U)         # conscription removes workers
          - eco_unemp_gdp_recovery · G · U                 # growth absorbs unemployment

  dD/dt =  eco_debt_war_rate · m · (1 - σ)               # war spending
          + eco_debt_crisis_rate · h · (1 - D)             # crisis borrowing
          - eco_debt_repay_rate · G · D                    # growth enables repayment

  dI/dt =  eco_industry_invest · r · σ · (1 - I)         # investment rebuilds industry
          - eco_industry_war_damage · m · (1 - σ) · I     # conflict damages industry
          - eco_industry_hazard · h · I                    # hazard disrupts production

═══════════════════════════════════════════════════════════════════════════════
CROSS-SYSTEM FEEDBACK (applied as impulse post-integration)

  Economy → Cluster politics (delta applied to cluster state):
    resource  +=  eco_gdp_resource_boost  · G · (1-r)     # GDP drives resource recovery
    hazard    +=  eco_unemp_hazard_boost  · U              # unemployment breeds instability
    trust     -=  eco_unemp_trust_cost   · U               # unemployment erodes trust
    trust     -=  eco_debt_trust_cost    · D               # debt burden erodes institutions
    resource  -=  eco_debt_resource_cost · D               # debt service consumes resources
    sigma     +=  eco_industry_sigma     · I · (1-h/5)    # industrial base stabilises

  Economy → Population (see population.py for labor_force, draft_pool):
    labor_force(i) = P(i) · working_age_frac · (1 - U(i))
    draft_pool(i)  = P(i) · draft_eligible_frac           # ceiling on conscriptable manpower

  Economy → Military (via political_interface):
    supply refill rate   × (1 + eco_industry_supply_bonus · I(i))
    global_reinf_pool    × (1 - eco_debt_reinf_penalty · D_mean)
    unit deployment cap  = draft_pool(i)                  # hard ceiling per cluster
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.gravitas_params import GravitasParams

# ── Column indices ──────────────────────────────────────────────────────── #
I_GDP   = 0
I_UNEMP = 1
I_DEBT  = 2
I_IND   = 3
N_ECO_VARS = 4


# ─────────────────────────────────────────────────────────────────────────── #
# Economy ODE derivatives                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def economy_derivatives(
    economy: NDArray[np.float64],         # (N, 4)
    c_arr:   NDArray[np.float64],         # (N, 6): [σ, h, r, m, τ, p]
    population: Optional[NDArray[np.float64]],  # (N,)
    adjacency: NDArray[np.float64],       # (N, N)
    alliance:  Optional[NDArray[np.float64]],   # (N, N) or None
    params: GravitasParams,
    N: int,
) -> NDArray[np.float64]:
    """
    Compute d(economy)/dt for each cluster.  Returns (N, 4).
    """
    sigma    = c_arr[:N, 0]
    hazard   = c_arr[:N, 1]
    resource = c_arr[:N, 2]
    military = c_arr[:N, 3]
    trust    = c_arr[:N, 4]

    G = economy[:N, I_GDP]
    U = economy[:N, I_UNEMP]
    D = economy[:N, I_DEBT]
    I = economy[:N, I_IND]

    # ── Labor force multiplier ──────────────────────────────────────────── #
    if population is not None:
        P = population[:N]
    else:
        P = np.full(N, 0.65)
    labor_force = P * params.working_age_frac * (1.0 - U)  # ∈ [0, 1]

    # ── Trade bonus from allied neighbours ─────────────────────────────── #
    adj = adjacency[:N, :N]
    if alliance is not None:
        ally = np.clip(alliance[:N, :N], 0.0, 1.0)
        trade_weights = adj * (1.0 + ally)
    else:
        trade_weights = adj

    row_sums = trade_weights.sum(axis=1, keepdims=True) + 1e-8
    trade_weights_norm = trade_weights / row_sums
    gdp_neighbor = trade_weights_norm @ G   # weighted mean neighbour GDP

    # ── GDP ────────────────────────────────────────────────────────────── #
    gdp_growth = (
        params.eco_gdp_recovery * sigma * resource * (0.4 + 0.6 * labor_force)
        + params.eco_trade_gain * gdp_neighbor * G
        + params.eco_trust_gdp  * trust * (1.0 - G)
    )
    gdp_loss = (
        params.eco_gdp_hazard_cost * (hazard / 5.0) * G
        + params.eco_gdp_debt_drag * D * G
    )
    dG = gdp_growth - gdp_loss

    # ── Unemployment ───────────────────────────────────────────────────── #
    unemp_rise = (
        params.eco_unemp_hazard_rate    * (hazard / 5.0) * (1.0 - U)
        + params.eco_unemp_military_draft * military * (1.0 - sigma) * (1.0 - U)
    )
    unemp_fall = params.eco_unemp_gdp_recovery * G * U
    dU = unemp_rise - unemp_fall

    # ── Debt ───────────────────────────────────────────────────────────── #
    debt_rise = (
        params.eco_debt_war_rate    * military * (1.0 - sigma) * (1.0 - D)
        + params.eco_debt_crisis_rate * (hazard / 5.0) * (1.0 - D)
    )
    debt_fall = params.eco_debt_repay_rate * G * D
    dD = debt_rise - debt_fall

    # ── Industrial capacity ────────────────────────────────────────────── #
    ind_growth = (
        params.eco_industry_invest * resource * sigma * (1.0 - I)
    )
    ind_loss = (
        params.eco_industry_war_damage * military * (1.0 - sigma) * I
        + params.eco_industry_hazard   * (hazard / 5.0) * I
    )
    dI = ind_growth - ind_loss

    deriv = np.zeros((N, N_ECO_VARS), dtype=np.float64)
    deriv[:, I_GDP]   = dG
    deriv[:, I_UNEMP] = dU
    deriv[:, I_DEBT]  = dD
    deriv[:, I_IND]   = dI

    return deriv


# ─────────────────────────────────────────────────────────────────────────── #
# Economy step (Euler, after RK4)                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def step_economy(
    economy:    NDArray[np.float64],      # (max_N, 4)
    c_arr:      NDArray[np.float64],      # (N, 6)
    population: Optional[NDArray[np.float64]],
    adjacency:  NDArray[np.float64],
    alliance:   Optional[NDArray[np.float64]],
    params:     GravitasParams,
    N:          int,
    rng:        Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Advance economy by one Euler step. Returns updated (max_N, 4)."""
    deriv = economy_derivatives(economy, c_arr, population, adjacency, alliance, params, N)

    new_eco = economy.copy()
    new_eco[:N] = np.clip(economy[:N] + params.dt * deriv, 0.0, 1.0)

    # Small stochastic noise on GDP and unemployment
    if rng is not None:
        noise_G = 0.003 * np.sqrt(params.dt) * rng.standard_normal(N)
        noise_U = 0.002 * np.sqrt(params.dt) * rng.standard_normal(N)
        new_eco[:N, I_GDP]   = np.clip(new_eco[:N, I_GDP]   + noise_G, 0.0, 1.0)
        new_eco[:N, I_UNEMP] = np.clip(new_eco[:N, I_UNEMP] + noise_U, 0.0, 1.0)

    return new_eco


# ─────────────────────────────────────────────────────────────────────────── #
# Economy → Cluster feedback impulse (post-step)                             #
# ─────────────────────────────────────────────────────────────────────────── #

def economy_cluster_feedback(
    economy:  NDArray[np.float64],   # (N, 4) current values
    c_arr:    NDArray[np.float64],   # (N, 6)
    params:   GravitasParams,
    N:        int,
    dt:       float,
) -> NDArray[np.float64]:
    """
    Per-step impulse on cluster variables from economic state.
    Returns delta (N, 6) to add to cluster array.
    """
    G = economy[:N, I_GDP]
    U = economy[:N, I_UNEMP]
    D = economy[:N, I_DEBT]
    I = economy[:N, I_IND]

    hazard = c_arr[:N, 1]
    resource = c_arr[:N, 2]

    delta = np.zeros((N, 6), dtype=np.float64)

    # sigma [0]: industrial base stabilises, unemployment destabilises
    delta[:, 0] = (params.eco_industry_sigma * I * (1.0 - hazard / 5.0)
                   - params.eco_unemp_sigma_cost * U) * dt

    # hazard [1]: high unemployment breeds unrest
    delta[:, 1] = params.eco_unemp_hazard_boost * U * dt

    # resource [2]: GDP drives recovery; debt service consumes it
    delta[:, 2] = (params.eco_gdp_resource_boost * G * (1.0 - resource)
                   - params.eco_debt_resource_cost * D) * dt

    # trust [4]: unemployment and debt erode trust
    delta[:, 4] = -(params.eco_unemp_trust_cost * U
                    + params.eco_debt_trust_cost * D) * dt

    # polar [5]: high unemployment + debt polarises society
    delta[:, 5] = params.eco_unemp_polar_boost * (U + 0.5 * D) * dt

    return delta


# ─────────────────────────────────────────────────────────────────────────── #
# Population demographics helpers                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_labor_force(
    population: NDArray[np.float64],   # (N,)
    economy:    NDArray[np.float64],   # (N, 4)
    params:     GravitasParams,
    N:          int,
) -> NDArray[np.float64]:
    """
    Labor force per cluster: P_i × working_age_frac × (1 - U_i).
    Returns (N,) in [0, 1].
    """
    P = population[:N]
    U = economy[:N, I_UNEMP]
    return P * params.working_age_frac * (1.0 - U)


def compute_draft_pool(
    population: NDArray[np.float64],   # (N,)
    economy:    NDArray[np.float64],   # (N, 4)
    c_arr:      NDArray[np.float64],   # (N, 6)
    params:     GravitasParams,
    N:          int,
) -> NDArray[np.float64]:
    """
    Conscriptable manpower per cluster.

    draft_pool(i) = P(i) × working_age_frac × draft_eligible_frac
                  × mobilization_factor(sigma, GDP)

    mobilization_factor: crisis states (low sigma, low GDP) actually
    boost draft eligibility (desperation conscription).
    Returns (N,) representing fraction of cluster population draftable.
    """
    P     = population[:N]
    sigma = c_arr[:N, 0]
    G     = economy[:N, I_GDP]

    # Mobilization is higher under threat (low sigma) but capped by GDP viability
    mobilization = np.clip(
        params.draft_eligible_frac * (1.0 + 0.5 * (1.0 - sigma) * (1.0 - G)),
        0.0,
        params.draft_eligible_frac * 2.0,
    )
    return P * params.working_age_frac * mobilization


# ─────────────────────────────────────────────────────────────────────────── #
# Initializer                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def initialize_economy(
    N:     int,
    max_N: int,
    c_arr: NDArray[np.float64],   # (N, 6) initial cluster state
    rng:   np.random.Generator,
) -> NDArray[np.float64]:
    """
    Initialize per-cluster economy.  Returns (max_N, 4) array.

    Starting values are grounded in cluster initial conditions:
      - GDP ~ σ × r with noise
      - Unemployment inversely correlated with GDP
      - Debt starts low (peacetime)
      - Industry ~ σ with noise
    """
    eco = np.zeros((max_N, N_ECO_VARS), dtype=np.float64)

    sigma    = c_arr[:N, 0]
    resource = c_arr[:N, 2]

    # GDP: baseline from stability × resource
    eco[:N, I_GDP] = np.clip(
        sigma * resource + rng.uniform(-0.05, 0.10, N),
        0.10, 0.90,
    )
    # Unemployment: inversely correlated with GDP
    eco[:N, I_UNEMP] = np.clip(
        0.30 * (1.0 - eco[:N, I_GDP]) + rng.uniform(0.02, 0.10, N),
        0.02, 0.50,
    )
    # Debt: low at start, slightly higher for unstable clusters
    eco[:N, I_DEBT] = np.clip(
        0.15 * (1.0 - sigma) + rng.uniform(0.01, 0.08, N),
        0.01, 0.40,
    )
    # Industry: follows sigma
    eco[:N, I_IND] = np.clip(
        sigma * 0.8 + rng.uniform(-0.05, 0.10, N),
        0.05, 0.90,
    )

    return eco
