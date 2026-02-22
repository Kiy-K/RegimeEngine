"""
Survival-optimization RL environment for hierarchical political simulation.

Objective: systemic survival under spatial instability. Not fairness.
- Hierarchical districts, topology randomized per episode
- Spatial policy allocation (optional), shock injection, early warning index
- Hazard clustering amplification, exhaustion growth tracking
- Reward: R = α*survival_time - β*peak_hazard - γ*exhaustion_acceleration
          - δ*cluster_intensity - ε*volatility_spike_rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from ..core.factions import create_balanced_factions, recompute_system_state
from ..core.hierarchical_state import (
    HierarchicalState,
    create_hierarchical_state,
    array_to_districts,
)
from ..core.hierarchical_coupling import get_geography_summary
from ..core.integrator import step as rk4_step
from ..core.parameters import SystemParameters
from ..core.state import RegimeState, SystemState
from ..core.topology import (
    build_province_district_layout,
    district_to_province,
    build_randomized_adjacency,
    diffusion_rate_bound,
)
from ..systems.crisis_classifier import ClassifierThresholds, CrisisLevel, classify
from ..systems.events import check_and_apply_events
from ..systems.hazard import HazardParameters, compute_hazard
from ..systems.early_warning import early_warning_index, exhaustion_growth_rate, volatility_spike_indicator
from ..systems.shocks import maybe_inject_shock
from ..systems.collapse_physics import (
    build_province_adjacency,
    is_bridge_province,
    apply_domino_effects,
    apply_national_shock,
    apply_exhaustion_admin_decay,
    apply_exhaustion_unrest_drift,
    CONSECUTIVE_STEPS_FOR_CRITICAL,
    UNREST_CRITICAL_THRESHOLD,
)
from .action_space import (
    MAX_INTENSITY,
    Action,
    ActionType,
    action_space_size,
    apply_action,
)

# Observation: aggregated regime (10 macro) + province summary (max_provinces*3) + EWI (1) + hazard_amp (1) + exh_growth (1) + top_k (k)
MAX_PROVINCES = 10
TOP_K_UNSTABLE = 5


@dataclass
class SurvivalConfig:
    """Configuration for survival env: reward weights and shock rate."""
    alpha_survival: float = 1.0
    beta_peak_hazard: float = 2.0
    gamma_exh_acceleration: float = 1.5
    delta_cluster_intensity: float = 1.2
    epsilon_volatility_spike: float = 0.8
    shock_prob_per_step: float = 0.02
    early_warning_threshold: float = 0.5
    max_steps: int = 500


def _make_base_params(seed: int, max_steps: int) -> SystemParameters:
    """Base params with hierarchy enabled; nu/tau will be overridden at reset."""
    return SystemParameters(
        use_hierarchy=True,
        n_provinces=7,
        districts_per_province=10,
        max_steps=max_steps,
        seed=seed,
        sigma_noise=0.04,
    )


def _observation_dim(n_provinces: int) -> int:
    """Compressed spatial observation size: regime macro + province*3 + EWI + hazard_amp + exh_growth + top_k."""
    regime = 10  # L, C, F, I, M, R, E, V, Exh, GDP
    province_summary = MAX_PROVINCES * 3  # unrest_mean, gdp_mean, admin_mean per province (padded)
    return regime + province_summary + 1 + 1 + 1 + TOP_K_UNSTABLE


class SurvivalRegimeEnv(gym.Env):
    """
    Survival-optimization regime env: hierarchy, topology randomization, shocks.
    Agents operate on compressed spatial signals; no full district state.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: Optional[SurvivalConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or SurvivalConfig()
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._base_params = _make_base_params(
            seed or 0,
            self.config.max_steps,
        )
        self._current_params: Optional[SystemParameters] = None
        self._state: Optional[RegimeState] = None
        self._n_provinces: int = 7
        self._n_factions = self._base_params.n_factions
        self._n_action_types = action_space_size()
        self._total_actions = self._n_factions * self._n_action_types

        # Tracking for survival reward
        self._peak_hazard: float = 0.0
        self._prev_exhaustion: float = 0.0
        self._prev_volatility: float = 0.0
        self._exh_acceleration_accum: float = 0.0
        self._volatility_spike_count: float = 0.0

        # Phase 4: province tipping and collapse
        self._consecutive_high_unrest: Optional[NDArray[np.intp]] = None  # (n_provinces,)
        self._province_critical: Optional[NDArray[np.uint8]] = None
        self._province_instability_counter: Optional[NDArray[np.intp]] = None
        self._province_adjacency: Optional[NDArray[np.float64]] = None
        self._province_weight: Optional[NDArray[np.float64]] = None
        self._ewi_history: list = []  # last 5 EWI values for collapse condition
        self._diffusion_amplified_until_step: int = 0  # step index until which nu is 1.5x

        obs_dim = _observation_dim(MAX_PROVINCES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._total_actions)

    def _randomize_topology_and_params(self) -> Tuple[SystemParameters, HierarchicalState]:
        """New topology and params each episode; agents must learn spatial principles."""
        n_provinces = int(self._rng.integers(5, 11))
        dpp = int(self._rng.integers(5, 16))
        counts, n_d = build_province_district_layout(n_provinces, dpp)
        province_of = district_to_province(counts)
        A = build_randomized_adjacency(
            counts,
            self._rng,
            connect_within=True,
            between_prob=0.35,
            between_weight=0.5,
        )
        bound = diffusion_rate_bound(A)
        nu_high = min(0.25, bound * 0.9)
        nu_low = max(0.01, nu_high * 0.2)
        if nu_low >= nu_high:
            nu_low, nu_high = bound * 0.1, bound * 0.9
        nu = float(self._rng.uniform(nu_low, nu_high))
        tau = float(self._rng.uniform(1.0, 4.0))

        # New params for this episode (replace hierarchy-related fields)
        from dataclasses import replace
        params = replace(
            self._base_params,
            n_provinces=n_provinces,
            districts_per_province=dpp,
            nu_diffusion=nu,
            tau_delay_base=tau,
            max_steps=self.config.max_steps,
        )

        # Build hierarchical state with this topology (custom A and counts)
        from ..core.hierarchical_state import DistrictState
        districts = [
            DistrictState(
                local_gdp=0.5,
                local_unrest=0.05,
                admin_capacity=0.7,
                factional_dominance=0.1,
                implementation_efficiency=0.7,
                local_memory=0.1,
            )
            for _ in range(n_d)
        ]
        n_pol = params.n_pillars
        pipeline = np.zeros((n_d, n_pol), dtype=np.float64)
        hier = HierarchicalState(
            district_states=tuple(districts),
            pipeline_buffers=pipeline,
            adjacency=A,
            counts_per_province=tuple(counts),
            province_of_district=province_of,
        )
        return params, hier

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_params, hier = self._randomize_topology_and_params()
        self._n_provinces = hier.n_provinces

        factions = create_balanced_factions(self._current_params.n_factions)
        placeholder = SystemState.neutral(self._current_params.n_pillars)
        aff_mat = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(self._n_factions))
            for i in range(self._n_factions)
        )
        state = RegimeState(
            factions=factions,
            system=placeholder,
            affinity_matrix=aff_mat,
            step=0,
            hierarchical=hier,
        )
        self._state = recompute_system_state(state, self._current_params)

        self._peak_hazard = 0.0
        self._prev_exhaustion = self._state.system.exhaustion
        self._prev_volatility = self._state.system.volatility
        self._exh_acceleration_accum = 0.0
        self._volatility_spike_count = 0.0
        self._ewi_history = []
        self._diffusion_amplified_until_step = 0
        self._consecutive_high_unrest = np.zeros(self._n_provinces, dtype=np.intp)
        self._province_critical = np.zeros(self._n_provinces, dtype=np.uint8)
        self._province_instability_counter = np.zeros(self._n_provinces, dtype=np.intp)
        if self._state.hierarchical is not None:
            self._province_adjacency, self._province_weight = build_province_adjacency(self._state.hierarchical)
        else:
            self._province_adjacency = np.zeros((MAX_PROVINCES, MAX_PROVINCES), dtype=np.float64)
            self._province_weight = np.zeros((MAX_PROVINCES, MAX_PROVINCES), dtype=np.float64)

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if self._state is None or self._current_params is None:
            raise RuntimeError("Call reset() first.")
        state = self._state

        # Phase 4: diffusion amplification decay after 5 steps
        if self._diffusion_amplified_until_step > 0 and state.step >= self._diffusion_amplified_until_step:
            self._diffusion_amplified_until_step = 0
        from dataclasses import replace
        from ..core.topology import diffusion_rate_bound
        step_params = self._current_params
        if self._diffusion_amplified_until_step > 0 and state.step < self._diffusion_amplified_until_step and state.hierarchical is not None:
            bound = diffusion_rate_bound(state.hierarchical.adjacency)
            nu_amp = min(bound * 0.9, step_params.nu_diffusion * 1.5)
            step_params = replace(step_params, nu_diffusion=nu_amp)

        # Apply faction action; Phase 4: exhaustion += suppression_intensity**2
        faction_idx = action // self._n_action_types
        action_type = ActionType(action % self._n_action_types)
        intensity = 0.08
        suppression_intensity = intensity if action_type == ActionType.SUPPRESSION else 0.0
        agent_action = Action(
            action_type=action_type,
            actor_idx=0,
            target_idx=faction_idx,
            intensity=intensity,
            agent_id="survival_agent",
        )
        state = apply_action(state, agent_action)
        state = recompute_system_state(state, step_params)
        if suppression_intensity > 0:
            exh = state.system.exhaustion
            new_exh = min(1.0, exh + suppression_intensity ** 2)
            new_sys = state.system.__class__(
                legitimacy=state.system.legitimacy,
                cohesion=state.system.cohesion,
                fragmentation=state.system.fragmentation,
                instability=state.system.instability,
                mobilization=state.system.mobilization,
                repression=state.system.repression,
                elite_alignment=state.system.elite_alignment,
                volatility=state.system.volatility,
                exhaustion=new_exh,
                state_gdp=state.system.state_gdp,
                pillars=state.system.pillars,
            )
            state = state.copy_with_system(new_sys)
        state = rk4_step(state, step_params)
        state = check_and_apply_events(state, step_params)

        # Shock injection (Phase 4: returns shock_info for instability_counter)
        state, shock_applied, shock_info = maybe_inject_shock(
            state, self._current_params, self._rng,
            shock_prob=self.config.shock_prob_per_step,
        )
        if shock_applied:
            state = recompute_system_state(state, step_params)
            prov_bump = shock_info.get("instability_counter_province")
            if prov_bump is not None and self._province_instability_counter is not None and 0 <= prov_bump < len(self._province_instability_counter):
                self._province_instability_counter[prov_bump] += 1

        # Phase 4: province critical tracking and domino
        geo = get_geography_summary(state.hierarchical, self._current_params) if state.hierarchical else {}
        province_unrest_means = geo.get("province_unrest_means", [])
        n_p = len(province_unrest_means)
        if n_p > 0 and self._consecutive_high_unrest is not None and self._province_critical is not None:
            for p in range(min(n_p, len(self._consecutive_high_unrest))):
                if province_unrest_means[p] > UNREST_CRITICAL_THRESHOLD:
                    self._consecutive_high_unrest[p] += 1
                else:
                    self._consecutive_high_unrest[p] = 0
                self._province_critical[p] = 1 if self._consecutive_high_unrest[p] >= CONSECUTIVE_STEPS_FOR_CRITICAL else 0
            state = apply_domino_effects(
                state, self._province_critical,
                self._province_adjacency[:n_p, :n_p],
                self._province_weight[:n_p, :n_p],
                step_params,
            )
            n_critical = int(np.sum(self._province_critical[:n_p]))
            if n_critical >= 2:
                state = apply_national_shock(state)
                state = recompute_system_state(state, step_params)
            for p in range(n_p):
                if not self._province_critical[p]:
                    continue
                if state.hierarchical is not None and is_bridge_province(p, state.hierarchical):
                    self._diffusion_amplified_until_step = state.step + 5
                    vol = min(1.0, state.system.volatility + 0.15)
                    new_sys = state.system.__class__(
                        legitimacy=state.system.legitimacy,
                        cohesion=state.system.cohesion,
                        fragmentation=state.system.fragmentation,
                        instability=state.system.instability,
                        mobilization=state.system.mobilization,
                        repression=state.system.repression,
                        elite_alignment=state.system.elite_alignment,
                        volatility=vol,
                        exhaustion=state.system.exhaustion,
                        state_gdp=state.system.state_gdp,
                        pillars=state.system.pillars,
                    )
                    state = state.copy_with_system(new_sys)
                    break

        # Phase 4: exhaustion nonlinearity (admin decay, unrest drift)
        state = apply_exhaustion_admin_decay(state, step_params)
        state = apply_exhaustion_unrest_drift(state)
        if state.hierarchical is not None:
            state = recompute_system_state(state, step_params)

        self._state = state

        # Survival reward
        exh = state.system.exhaustion
        vol = state.system.volatility
        exh_growth = exhaustion_growth_rate(exh, self._prev_exhaustion, step_params.dt)
        vol_spike = volatility_spike_indicator(vol, self._prev_volatility, threshold=0.08)
        self._prev_exhaustion = exh
        self._prev_volatility = vol

        # Phase 4: hazard with nonlinear unrest and clustering amplification
        geo_final = get_geography_summary(state.hierarchical, self._current_params) if state.hierarchical else {}
        unrest_mean = float(np.mean(geo_final.get("province_unrest_means", [0.0]))) if geo_final.get("province_unrest_means") else 0.0
        clustering = geo_final.get("clustering_index", 0.0)
        hazard = compute_hazard(state, HazardParameters(), unrest_mean=unrest_mean, clustering_index=clustering)
        self._peak_hazard = max(self._peak_hazard, hazard)

        # EWI for collapse (EWI > 0.8 for 5 consecutive steps)
        ewi = early_warning_index(state, unrest_variance=geo_final.get("unrest_variance", 0.0), clustering_index=clustering, exhaustion_growth_rate=exh_growth)
        self._ewi_history.append(ewi)
        if len(self._ewi_history) > 5:
            self._ewi_history.pop(0)
        ewi_critical_5 = (len(self._ewi_history) == 5 and all(x > 0.8 for x in self._ewi_history))

        n_critical = int(np.sum(self._province_critical[:n_p])) if n_p and self._province_critical is not None else 0

        # Phase 4 reward: R += α*exp(-hazard) - 0.5*n_critical - other penalties
        step_reward = self.config.alpha_survival * float(np.exp(-hazard))
        step_reward -= 0.5 * n_critical
        step_reward -= self.config.beta_peak_hazard * (hazard / max(1, state.step))
        step_reward -= self.config.gamma_exh_acceleration * min(1.0, abs(exh_growth) * 10.0)
        step_reward -= self.config.delta_cluster_intensity * clustering * 0.1
        step_reward -= self.config.epsilon_volatility_spike * vol_spike

        # Phase 4 collapse: hazard > 1.2, 3+ critical, exhaustion > 0.9, or EWI > 0.8 for 5 steps
        terminated = (
            hazard > 1.2
            or n_critical >= 3
            or state.system.exhaustion > 0.9
            or ewi_critical_5
        )
        truncated = state.step >= self._current_params.max_steps and not terminated

        info = self._get_info()
        info["peak_hazard"] = self._peak_hazard
        info["shock_applied"] = shock_applied
        info["survival_steps"] = state.step
        info["n_critical_provinces"] = n_critical

        return self._get_obs(), float(step_reward), terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float32]:
        """Compressed spatial observation: no full district state."""
        assert self._state is not None
        sys = self._state.system
        # Regime macro (10)
        regime = np.array([
            sys.legitimacy, sys.cohesion, sys.fragmentation, sys.instability,
            sys.mobilization, sys.repression, sys.elite_alignment,
            sys.volatility, sys.exhaustion, sys.state_gdp,
        ], dtype=np.float32)

        # Province summary (MAX_PROVINCES * 3), padded
        province_unrest = np.zeros(MAX_PROVINCES, dtype=np.float32)
        province_gdp = np.zeros(MAX_PROVINCES, dtype=np.float32)
        province_admin = np.zeros(MAX_PROVINCES, dtype=np.float32)
        if self._state.hierarchical is not None:
            geo = get_geography_summary(self._state.hierarchical, self._current_params)
            n_p = len(geo["province_unrest_means"])
            province_unrest[:n_p] = geo["province_unrest_means"]
            province_gdp[:n_p] = geo["province_gdp_means"]
            province_admin[:n_p] = geo["province_admin_means"]

        # Early warning, hazard amplification, exhaustion growth
        geo = get_geography_summary(self._state.hierarchical, self._current_params) if self._state.hierarchical else {}
        unrest_var = geo.get("unrest_variance", 0.0)
        clustering = geo.get("clustering_index", 0.0)
        exh_growth = exhaustion_growth_rate(
            sys.exhaustion, self._prev_exhaustion,
            self._current_params.dt if self._current_params else 0.01,
        )
        ewi = early_warning_index(
            self._state,
            unrest_variance=unrest_var,
            clustering_index=clustering,
            exhaustion_growth_rate=exh_growth,
        )
        hazard_amp = geo.get("hazard_amplification", 1.0)
        ewi_arr = np.array([ewi], dtype=np.float32)
        hazard_amp_arr = np.array([min(1.0, hazard_amp)], dtype=np.float32)
        exh_growth_arr = np.array([np.clip((exh_growth + 0.05) / 0.1, 0.0, 1.0)], dtype=np.float32)

        # Top-k unstable provinces (values, not indices)
        top_k_vals = np.zeros(TOP_K_UNSTABLE, dtype=np.float32)
        if self._state.hierarchical is not None:
            from ..core.hierarchical_obs import top_k_unstable_districts
            _, vals = top_k_unstable_districts(self._state.hierarchical, k=TOP_K_UNSTABLE)
            # By province: use province unrest means and take top k provinces
            prov_means = geo.get("province_unrest_means", [])
            if prov_means:
                sorted_provinces = np.argsort(prov_means)[::-1][:TOP_K_UNSTABLE]
                for i, p in enumerate(sorted_provinces):
                    top_k_vals[i] = prov_means[p]

        obs = np.concatenate([
            regime,
            province_unrest, province_gdp, province_admin,
            ewi_arr, hazard_amp_arr, exh_growth_arr,
            top_k_vals,
        ], axis=0)
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        assert self._state is not None
        crisis = classify(self._state, ClassifierThresholds())
        geo = get_geography_summary(self._state.hierarchical, self._current_params) if self._state.hierarchical else {}
        return {
            "step": self._state.step,
            "crisis_level": crisis.name,
            "exhaustion": self._state.system.exhaustion,
            "legitimacy": self._state.system.legitimacy,
            "early_warning": early_warning_index(
                self._state,
                unrest_variance=geo.get("unrest_variance", 0.0),
                clustering_index=geo.get("clustering_index", 0.0),
                exhaustion_growth_rate=0.0,
            ),
            "clustering_index": geo.get("clustering_index", 0.0),
            "n_provinces": self._n_provinces,
        }

    def current_state(self) -> RegimeState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state
