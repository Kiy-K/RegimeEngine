"""
Survival-optimization RL environment for hierarchical political simulation.

Objective: systemic survival under spatial instability. Not fairness.
- Hierarchical districts, topology randomized per episode
- Spatial policy allocation (optional), shock injection, early warning index
- Hazard clustering amplification, exhaustion growth tracking
- Reward: R = α*exp(-hazard) - β*hazard/step - γ*exh_accel
          - δ*cluster_intensity - ε*volatility_spike - 0.5*n_critical
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

MAX_PROVINCES = 10
TOP_K_UNSTABLE = 5
_SUPPRESSION_EXH_COST = 0.0064  # intensity**2 = 0.08**2


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
    return SystemParameters(
        use_hierarchy=True,
        n_provinces=7,
        districts_per_province=10,
        max_steps=max_steps,
        seed=seed,
        sigma_noise=0.04,
    )


def _observation_dim(n_provinces: int) -> int:
    regime = 10
    province_summary = MAX_PROVINCES * 3
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
        self._base_params = _make_base_params(seed or 0, self.config.max_steps)
        self._current_params: Optional[SystemParameters] = None
        self._state: Optional[RegimeState] = None
        self._n_provinces: int = 7
        self._n_factions = self._base_params.n_factions
        self._n_action_types = action_space_size()
        self._total_actions = self._n_factions * self._n_action_types

    def update_config(self, **kwargs):
        """Update environment configuration in-place."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # Survival tracking
        self._peak_hazard: float = 0.0
        self._prev_exhaustion: float = 0.0
        self._prev_volatility: float = 0.0
        self._exh_acceleration_accum: float = 0.0
        self._volatility_spike_count: float = 0.0

        # Province tipping / collapse
        self._consecutive_high_unrest: Optional[NDArray[np.intp]] = None
        self._province_critical: Optional[NDArray[np.uint8]] = None
        self._province_instability_counter: Optional[NDArray[np.intp]] = None
        self._province_adjacency: Optional[NDArray[np.float64]] = None
        self._province_weight: Optional[NDArray[np.float64]] = None
        self._ewi_history: list = []
        self._diffusion_amplified_until_step: int = 0

        obs_dim = _observation_dim(MAX_PROVINCES)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._total_actions)

    # ------------------------------------------------------------------ #
    # Topology                                                             #
    # ------------------------------------------------------------------ #

    def _randomize_topology_and_params(self) -> Tuple[SystemParameters, HierarchicalState]:
        """New random topology and params each episode."""
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

        from dataclasses import replace
        params = replace(
            self._base_params,
            n_provinces=n_provinces,
            districts_per_province=dpp,
            nu_diffusion=nu,
            tau_delay_base=tau,
            max_steps=self.config.max_steps,
        )

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

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
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

        # Reset trackers
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
            self._province_adjacency, self._province_weight = build_province_adjacency(
                self._state.hierarchical
            )
        else:
            self._province_adjacency = np.zeros((MAX_PROVINCES, MAX_PROVINCES), dtype=np.float64)
            self._province_weight = np.zeros((MAX_PROVINCES, MAX_PROVINCES), dtype=np.float64)

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------ #
    # Step (orchestrator — calls focused private methods)                 #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: int,
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if self._state is None or self._current_params is None:
            raise RuntimeError("Call reset() first.")

        step_params = self._resolve_step_params()

        state = self._apply_faction_action(self._state, action, step_params)
        state = rk4_step(state, step_params, self._rng)
        state = check_and_apply_events(state, step_params)

        state, n_critical = self._process_spatial_dynamics(state, step_params)
        state = self._apply_exhaustion_effects(state, step_params)

        self._state = state

        geo = self._get_geo(state)
        hazard, ewi, exh_growth, vol_spike = self._compute_survival_signals(
            state, geo
        )

        self._peak_hazard = max(self._peak_hazard, hazard)
        self._update_ewi_history(ewi)

        reward = self._compute_step_reward(
            hazard, exh_growth, vol_spike, n_critical, state.step
        )
        terminated = self._check_termination(hazard, n_critical, state)
        truncated = state.step >= self._current_params.max_steps and not terminated

        info = self._get_info()
        info["peak_hazard"] = self._peak_hazard
        info["survival_steps"] = state.step
        info["n_critical_provinces"] = n_critical
        info["clustering_index"] = geo.get("clustering_index", 0.0)

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Private step helpers                                                 #
    # ------------------------------------------------------------------ #

    def _resolve_step_params(self) -> SystemParameters:
        """Return params, optionally with diffusion amplified."""
        from dataclasses import replace
        params = self._current_params
        state = self._state
        if (
            self._diffusion_amplified_until_step > 0
            and state.step < self._diffusion_amplified_until_step
            and state.hierarchical is not None
        ):
            bound = diffusion_rate_bound(state.hierarchical.adjacency)
            nu_amp = min(bound * 0.9, params.nu_diffusion * 1.5)
            params = replace(params, nu_diffusion=nu_amp)
        elif self._diffusion_amplified_until_step > 0 and state.step >= self._diffusion_amplified_until_step:
            self._diffusion_amplified_until_step = 0
        return params

    def _apply_faction_action(
        self,
        state: RegimeState,
        action: int,
        params: SystemParameters,
    ) -> RegimeState:
        """Decode action, apply it, and handle suppression exhaustion cost."""
        faction_idx = action // self._n_action_types
        action_type = ActionType(action % self._n_action_types)
        intensity = 0.08

        agent_action = Action(
            action_type=action_type,
            actor_idx=0,
            target_idx=faction_idx,
            intensity=intensity,
            agent_id="survival_agent",
        )
        state = apply_action(state, agent_action)
        state = recompute_system_state(state, params)

        if action_type == ActionType.SUPPRESSION:
            new_exh = min(1.0, state.system.exhaustion + _SUPPRESSION_EXH_COST)
            new_sys = state.system.__class__(
                **{**state.system.to_dict(), "exhaustion": new_exh}
            )
            state = state.copy_with_system(new_sys)

        return state

    def _process_spatial_dynamics(
        self,
        state: RegimeState,
        params: SystemParameters,
    ) -> Tuple[RegimeState, int]:
        """Handle shocks, province tipping, domino effects. Returns (state, n_critical)."""
        # Shock injection
        state, shock_applied, shock_info = maybe_inject_shock(
            state, self._current_params, self._rng,
            shock_prob=self.config.shock_prob_per_step,
        )
        if shock_applied:
            state = recompute_system_state(state, params)
            prov_bump = shock_info.get("instability_counter_province")
            if (
                prov_bump is not None
                and self._province_instability_counter is not None
                and 0 <= prov_bump < len(self._province_instability_counter)
            ):
                self._province_instability_counter[prov_bump] += 1

        # Province tipping
        n_critical = self._update_province_critical(state)

        # Domino effects
        n_p = len(state.hierarchical.province_of_district.tolist() and
                   self._province_critical[:self._n_provinces]) if state.hierarchical else 0
        if state.hierarchical is not None:
            n_p = state.hierarchical.n_provinces
            state = apply_domino_effects(
                state,
                self._province_critical,
                self._province_adjacency[:n_p, :n_p],
                self._province_weight[:n_p, :n_p],
                params,
            )
            if n_critical >= 2:
                state = apply_national_shock(state)
                state = recompute_system_state(state, params)
            state = self._handle_bridge_provinces(state, n_p)

        return state, n_critical

    def _update_province_critical(self, state: RegimeState) -> int:
        """Track consecutive high-unrest steps per province; return n_critical."""
        if state.hierarchical is None or self._consecutive_high_unrest is None:
            return 0
        geo = get_geography_summary(state.hierarchical, self._current_params)
        province_unrest_means = geo.get("province_unrest_means", [])
        n_p = len(province_unrest_means)
        for p in range(min(n_p, len(self._consecutive_high_unrest))):
            if province_unrest_means[p] > UNREST_CRITICAL_THRESHOLD:
                self._consecutive_high_unrest[p] += 1
            else:
                self._consecutive_high_unrest[p] = 0
            self._province_critical[p] = (
                1 if self._consecutive_high_unrest[p] >= CONSECUTIVE_STEPS_FOR_CRITICAL else 0
            )
        return int(np.sum(self._province_critical[:n_p]))

    def _handle_bridge_provinces(self, state: RegimeState, n_p: int) -> RegimeState:
        """Amplify diffusion and volatility if a critical bridge province is detected."""
        for p in range(n_p):
            if not self._province_critical[p]:
                continue
            if state.hierarchical is not None and is_bridge_province(p, state.hierarchical):
                self._diffusion_amplified_until_step = state.step + 5
                vol = min(1.0, state.system.volatility + 0.15)
                new_sys = state.system.__class__(
                    **{**state.system.to_dict(), "volatility": vol}
                )
                return state.copy_with_system(new_sys)
        return state

    def _apply_exhaustion_effects(
        self, state: RegimeState, params: SystemParameters
    ) -> RegimeState:
        """Admin decay and unrest drift under high exhaustion."""
        state = apply_exhaustion_admin_decay(state, params)
        state = apply_exhaustion_unrest_drift(state)
        if state.hierarchical is not None:
            state = recompute_system_state(state, params)
        return state

    def _get_geo(self, state: RegimeState) -> Dict[str, Any]:
        """Fetch geography summary once per step."""
        if state.hierarchical is not None:
            return get_geography_summary(state.hierarchical, self._current_params)
        return {}

    def _compute_survival_signals(
        self, state: RegimeState, geo: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Compute hazard, EWI, exhaustion growth, volatility spike."""
        exh = state.system.exhaustion
        vol = state.system.volatility

        exh_growth = exhaustion_growth_rate(exh, self._prev_exhaustion,
                                            self._current_params.dt)
        vol_spike = volatility_spike_indicator(vol, self._prev_volatility, threshold=0.08)
        self._prev_exhaustion = exh
        self._prev_volatility = vol

        unrest_mean = float(np.mean(geo.get("province_unrest_means", [0.0]))) \
            if geo.get("province_unrest_means") else 0.0
        clustering = geo.get("clustering_index", 0.0)

        hazard = compute_hazard(state, HazardParameters(),
                                unrest_mean=unrest_mean, clustering_index=clustering)

        ewi = early_warning_index(
            state,
            unrest_variance=geo.get("unrest_variance", 0.0),
            clustering_index=clustering,
            exhaustion_growth_rate=exh_growth,
        )
        return hazard, ewi, exh_growth, vol_spike

    def _update_ewi_history(self, ewi: float) -> None:
        """Maintain a 5-step rolling EWI history."""
        self._ewi_history.append(ewi)
        if len(self._ewi_history) > 5:
            self._ewi_history.pop(0)

    def _compute_step_reward(
        self,
        hazard: float,
        exh_growth: float,
        vol_spike: float,
        n_critical: int,
        step: int,
    ) -> float:
        """R = α*exp(-hazard) - 0.5*n_critical - β*hazard/step - γ*|exh_growth| - δ*clustering - ε*spike."""
        r = self.config.alpha_survival * float(np.exp(-hazard))
        r -= 0.5 * n_critical
        r -= self.config.beta_peak_hazard * (hazard / max(1, step))
        r -= self.config.gamma_exh_acceleration * min(1.0, abs(exh_growth) * 10.0)
        r -= self.config.epsilon_volatility_spike * vol_spike
        return r

    def _check_termination(
        self, hazard: float, n_critical: int, state: RegimeState
    ) -> bool:
        """Collapse conditions: hazard spike, critical mass, exhaustion, or sustained EWI."""
        ewi_critical_5 = (
            len(self._ewi_history) == 5
            and all(x > 0.8 for x in self._ewi_history)
        )
        return (
            hazard > 1.2
            or n_critical >= 3
            or state.system.exhaustion > 0.9
            or ewi_critical_5
        )

    # ------------------------------------------------------------------ #
    # Observation                                                          #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> NDArray[np.float32]:
        """Compressed spatial observation: regime macro + province summary + signals."""
        assert self._state is not None
        sys = self._state.system

        regime = np.array([
            sys.legitimacy, sys.cohesion, sys.fragmentation, sys.instability,
            sys.mobilization, sys.repression, sys.elite_alignment,
            sys.volatility, sys.exhaustion, sys.state_gdp,
        ], dtype=np.float32)

        province_unrest = np.zeros(MAX_PROVINCES, dtype=np.float32)
        province_gdp = np.zeros(MAX_PROVINCES, dtype=np.float32)
        province_admin = np.zeros(MAX_PROVINCES, dtype=np.float32)
        geo: Dict[str, Any] = {}
        if self._state.hierarchical is not None:
            geo = get_geography_summary(self._state.hierarchical, self._current_params)
            n_p = len(geo["province_unrest_means"])
            province_unrest[:n_p] = geo["province_unrest_means"]
            province_gdp[:n_p] = geo["province_gdp_means"]
            province_admin[:n_p] = geo["province_admin_means"]

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

        top_k_vals = np.zeros(TOP_K_UNSTABLE, dtype=np.float32)
        if self._state.hierarchical is not None:
            prov_means = geo.get("province_unrest_means", [])
            if prov_means:
                sorted_provinces = np.argsort(prov_means)[::-1][:TOP_K_UNSTABLE]
                for i, p in enumerate(sorted_provinces):
                    top_k_vals[i] = prov_means[p]

        obs = np.concatenate([
            regime,
            province_unrest, province_gdp, province_admin,
            np.array([ewi], dtype=np.float32),
            np.array([min(1.0, hazard_amp)], dtype=np.float32),
            np.array([np.clip((exh_growth + 0.05) / 0.1, 0.0, 1.0)], dtype=np.float32),
            top_k_vals,
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        assert self._state is not None
        crisis = classify(self._state, ClassifierThresholds())
        geo = get_geography_summary(
            self._state.hierarchical, self._current_params
        ) if self._state.hierarchical else {}
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