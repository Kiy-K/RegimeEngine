"""
regime_loader.py — Regime Configuration Loader for GRAVITAS.

Loads YAML regime configs and converts them to GravitasParams objects
that GravitasEnv can actually consume.

Bugs fixed vs original:
  1. Added build_gravitas_params() — the original had no way to convert a
     regime config into what GravitasEnv needs. Every call to --regime was
     a no-op because the config could never reach the environment.
  2. Added get_training_config() — extracts training hyperparameters with
     proper fallback to global defaults.
  3. load_regime_config() no longer raises on missing 'agents' — single-agent
     governance envs don't need a multi-agent agents spec.
  4. Added list_regimes() for CLI inspection.

Public API:
    load_regime_config(path)            -> raw config dict
    get_regime_by_name(config, name)    -> regime dict
    build_gravitas_params(regime, seed) -> GravitasParams
    get_training_config(regime, defs)   -> flat training hyperparameter dict
    list_regimes(config)                -> list of (name, description) tuples
    load_standalone_regime(path)        -> (GravitasParams, sectors, alliances, shocks)
"""

from __future__ import annotations

import warnings
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────── #
# YAML loading + validation                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def load_regime_config(config_path: str = "regime_config.yaml") -> Dict[str, Any]:
    """Load and validate a regime configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if "regimes" not in config:
        raise ValueError("Config must include a 'regimes' list.")

    for regime in config["regimes"]:
        if "name" not in regime:
            raise ValueError("Each regime must have a 'name' field.")
        # FIX 3: don't raise on missing 'agents' — governance envs are single-agent
        if "agents" not in regime:
            warnings.warn(f"Regime '{regime['name']}' has no 'agents' section.")

    return config


def get_regime_by_name(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Retrieve a regime configuration dict by name."""
    for regime in config["regimes"]:
        if regime["name"] == name:
            return regime
    available = [r["name"] for r in config["regimes"]]
    raise ValueError(f"Regime '{name}' not found. Available: {available}")


def list_regimes(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return list of (name, description) for all regimes."""
    return [
        (r["name"], r.get("description", "").strip())
        for r in config["regimes"]
    ]


# ─────────────────────────────────────────────────────────────────────────── #
# Regime → GravitasParams conversion                                          #
# FIX 1: this function did not exist in the original loader                  #
# ─────────────────────────────────────────────────────────────────────────── #

# Maps each YAML section → {yaml_key: GravitasParams field name}
_SECTION_MAP: Dict[str, Dict[str, str]] = {
    "topology": {
        "n_clusters_min":            "n_clusters_min",
        "n_clusters_max":            "n_clusters_max",
        "between_link_prob":         "between_link_prob",
        "conflict_link_prob":        "conflict_link_prob",
    },
    "shocks": {
        "hawkes_base_rate":          "hawkes_base_rate",
        "hawkes_alpha":              "hawkes_alpha",
        "hawkes_beta":               "hawkes_beta",
        "shock_pareto_alpha":        "shock_pareto_alpha",
        "shock_pareto_xmin":         "shock_pareto_xmin",
    },
    "military": {
        "military_exh_coeff":        "military_exh_coeff",
        "military_phi_coeff":        "military_phi_coeff",
        "military_tau_cost":         "military_tau_cost",
        "max_military_total":        "max_military_total",
        "military_hazard_reduction": "military_hazard_reduction",
        "military_sigma_boost":      "military_sigma_boost",
        "military_decay":            "military_decay",
    },
    "media": {
        "media_autonomy":            "media_autonomy",
        "rho_bias":                  "rho_bias",
        "beta_max_bias":             "beta_max_bias",
        "phi_bias_prop":             "phi_bias_prop",
        "phi_bias_incoherence":      "phi_bias_incoherence",
        "phi_bias_shock":            "phi_bias_shock",
    },
    "reward": {
        "w_stability":               "w_stability",
        "w_fragmentation":           "w_fragmentation",
        "w_polarization":            "w_polarization",
        "w_exhaustion":              "w_exhaustion",
        "w_resilience":              "w_resilience",
        "w_smoothness":              "w_smoothness",
        "w_unsustainable":           "w_unsustainable",
        "exhaustion_threshold":      "exhaustion_threshold",
        "invest_res_boost":          "invest_res_boost",
    },
    "episode": {
        "max_steps":                 "max_steps",
        "dt":                        "dt",
    },
}


def build_gravitas_params(
    regime: Dict[str, Any],
    seed: int = 0,
) -> "GravitasParams":
    """
    Convert a regime configuration dict to a GravitasParams instance.

    Each section in the regime YAML (topology, shocks, military, media,
    reward, episode) is mapped to the corresponding GravitasParams fields.
    Fields not present in the regime config fall back to GravitasParams defaults.

    Args:
        regime: A regime dict from get_regime_by_name().
        seed:   RNG seed to inject.

    Returns:
        GravitasParams instance configured for this regime.
    """
    from gravitas_engine.core.gravitas_params import GravitasParams

    kwargs: Dict[str, Any] = {"seed": seed}

    for section_name, field_map in _SECTION_MAP.items():
        section = regime.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for yaml_key, param_key in field_map.items():
            if yaml_key in section:
                kwargs[param_key] = section[yaml_key]

    # Only pass fields GravitasParams actually knows about
    valid_fields = set(GravitasParams.__dataclass_fields__.keys())
    unknown = set(kwargs.keys()) - valid_fields
    if unknown:
        warnings.warn(f"Unknown GravitasParams fields ignored: {unknown}")
        for k in unknown:
            del kwargs[k]

    return GravitasParams(**kwargs)


# ─────────────────────────────────────────────────────────────────────────── #
# Training config extraction                                                   #
# FIX 2: this function did not exist in the original loader                  #
# ─────────────────────────────────────────────────────────────────────────── #

def get_training_config(
    regime: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract training hyperparameters from a regime config,
    filling missing keys from the global defaults section.

    Returns a flat dict with keys:
        algo, total_timesteps, exhaustion_penalty,
        log_dir, eval_episodes, device,
        checkpoint_freq, eval_freq, metrics
    """
    g = defaults or {}
    t = regime.get("training", {})

    return {
        "algo":               t.get("algo",               g.get("algo", "rppo")),
        "total_timesteps":    t.get("total_timesteps",    600_000),
        "exhaustion_penalty": t.get("exhaustion_penalty", 0.10),
        "log_dir":            g.get("log_dir",            "logs/"),
        "eval_episodes":      int(g.get("eval_episodes",  30)),
        "device":             g.get("device",             "cpu"),
        "checkpoint_freq":    int(g.get("checkpoint_freq", 50_000)),
        "eval_freq":          int(g.get("eval_freq",       50_000)),
        "metrics":            t.get("metrics",            []),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone regime YAML loader (e.g. training/regimes/stalingrad.yaml)        #
# ─────────────────────────────────────────────────────────────────────────── #

def load_standalone_regime(
    path: str,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Load a standalone regime YAML (like stalingrad.yaml) that uses a flat
    `params:` block instead of nested sections.

    Returns a dict with:
        - 'params': GravitasParams instance
        - 'sectors': list of sector dicts with initial_state
        - 'initial_alliances': list of alliance edge dicts
        - 'custom_shocks': list of shock dicts
        - 'agents': list of agent dicts
        - 'training': training config dict
        - 'raw': the full raw YAML dict
    """
    from gravitas_engine.core.gravitas_params import GravitasParams
    import numpy as np

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Regime file not found: {p.resolve()}")

    with open(p, "r") as f:
        raw = yaml.safe_load(f)

    # ── Build GravitasParams from flat params block ──────────────────── #
    params_block = raw.get("params", {})
    valid_fields = set(GravitasParams.__dataclass_fields__.keys())
    kwargs = {"seed": seed}
    for k, v in params_block.items():
        if k in valid_fields:
            kwargs[k] = v
        else:
            warnings.warn(f"Unknown GravitasParams field ignored: {k}")
    params = GravitasParams(**kwargs)

    # ── Parse sectors → initial cluster states ───────────────────────── #
    sectors = raw.get("sectors", [])

    # ── Parse initial alliances ──────────────────────────────────────── #
    alliances = raw.get("initial_alliances", [])

    # ── Parse custom shocks ──────────────────────────────────────────── #
    shocks = raw.get("custom_shocks", [])

    # ── Parse agents ─────────────────────────────────────────────────── #
    agents = raw.get("agents", [])

    # ── Training config ──────────────────────────────────────────────── #
    training = raw.get("training", {})

    return {
        "params": params,
        "sectors": sectors,
        "initial_alliances": alliances,
        "custom_shocks": shocks,
        "agents": agents,
        "training": training,
        "raw": raw,
    }


def build_initial_states(
    regime_data: Dict[str, Any],
    max_N: int,
) -> Dict[str, Any]:
    """
    Convert regime sectors + alliances into arrays suitable for
    GravitasEnv.reset(options={...}).

    Returns a dict with:
        - 'initial_clusters': np.ndarray (N, 6) or None
        - 'initial_alliances': np.ndarray (max_N, max_N) or None
        - 'n_clusters': int
    """
    import numpy as np

    sectors = regime_data.get("sectors", [])
    alliances = regime_data.get("initial_alliances", [])

    result: Dict[str, Any] = {}

    # ── Cluster initial states ───────────────────────────────────────── #
    if sectors:
        N = len(sectors)
        c_arr = np.zeros((N, 6), dtype=np.float64)
        for sec in sectors:
            idx = sec["id"]
            st = sec.get("initial_state", {})
            c_arr[idx, 0] = st.get("sigma", 0.50)
            c_arr[idx, 1] = st.get("hazard", 0.10)
            c_arr[idx, 2] = st.get("resource", 0.50)
            c_arr[idx, 3] = st.get("military", 0.30)
            c_arr[idx, 4] = st.get("trust", 0.50)
            c_arr[idx, 5] = st.get("polar", 0.20)
        result["initial_clusters"] = c_arr
        result["n_clusters"] = N
    else:
        result["initial_clusters"] = None
        result["n_clusters"] = None

    # ── Alliance matrix ──────────────────────────────────────────────── #
    if alliances and sectors:
        N = len(sectors)
        A = np.zeros((max_N, max_N), dtype=np.float64)
        for edge in alliances:
            i, j = int(edge["from"]), int(edge["to"])
            v = float(edge["value"])
            if i < N and j < N:
                A[i, j] = v
                A[j, i] = v
        result["initial_alliances"] = A
    else:
        result["initial_alliances"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────── #
# CLI inspection                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect GRAVITAS regime configs")
    parser.add_argument("--config", default="regime_config.yaml")
    parser.add_argument("--regime", default=None,
                        help="Show GravitasParams for a specific regime")
    args = parser.parse_args()

    config = load_regime_config(args.config)
    print(f"\nAvailable regimes in '{args.config}':")
    for name, desc in list_regimes(config):
        print(f"  {name:<22} — {desc[:80]}")

    if args.regime:
        regime = get_regime_by_name(config, args.regime)
        params = build_gravitas_params(regime, seed=0)
        tcfg   = get_training_config(regime, config.get("defaults", {}))
        print(f"\nGravitasParams for '{args.regime}':")
        for field, val in params.__dataclass_fields__.items():
            print(f"  {field:<30} = {getattr(params, field)}")
        print(f"\nTraining config for '{args.regime}':")
        for k, v in tcfg.items():
            print(f"  {k:<25} = {v}")
