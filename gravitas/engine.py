"""
gravitas.engine — Core simulation engine with plugin support.

The GravitasEngine class is the high-level orchestrator that:
  1. Loads scenario YAML files (e.g., stalingrad.yaml).
  2. Discovers and loads plugins dynamically from config.
  3. Runs the simulation loop with plugin hooks at each step.
  4. Provides a clean API for training, evaluation, and replay.

Usage:
    from gravitas.engine import GravitasEngine

    engine = GravitasEngine.from_config("configs/custom.yaml")
    engine.run(episodes=30, seed=42)

    # Or manually:
    engine = GravitasEngine(scenario="stalingrad")
    engine.load_plugins(["soviet_reinforcements", "axis_airlift"])
    results = engine.run(episodes=10)
"""

from __future__ import annotations

import logging
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gravitas.plugins import (
    GravitasPlugin,
    load_plugins_from_config,
)

logger = logging.getLogger("gravitas.engine")

_ROOT = Path(__file__).resolve().parent.parent  # GravitasEngine/
_CWD = Path.cwd()  # Fallback for standalone/Nuitka builds


# ─────────────────────────────────────────────────────────────────────────── #
# Scenario discovery                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _scenario_search_paths() -> list:
    """Build scenario search paths, checking both source root and CWD."""
    paths = []
    for base in (_ROOT, _CWD):
        paths.append(base / "gravitas" / "scenarios")
        paths.append(base / "training" / "regimes")
    # Deduplicate while preserving order
    seen = set()
    result = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def find_scenario(name: str) -> Path:
    """Locate a scenario YAML by name (without extension).

    Searches gravitas/scenarios/ first, then training/regimes/.

    Args:
        name: Scenario name (e.g., "stalingrad").

    Returns:
        Path to the YAML file.

    Raises:
        FileNotFoundError: If no matching scenario found.
    """
    candidates = []
    for search_dir in _scenario_search_paths():
        for ext in (".yaml", ".yml"):
            candidate = search_dir / f"{name}{ext}"
            candidates.append(candidate)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Scenario '{name}' not found. Searched:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def list_scenarios() -> List[Dict[str, str]]:
    """List all available scenarios.

    Returns:
        List of dicts with 'name' and 'path' keys.
    """
    scenarios = []
    seen = set()
    for search_dir in _scenario_search_paths():
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.glob("*.yaml")) + sorted(search_dir.glob("*.yml")):
            name = f.stem
            if name not in seen:
                seen.add(name)
                scenarios.append({"name": name, "path": str(f)})
    return scenarios


# ─────────────────────────────────────────────────────────────────────────── #
# GravitasEngine                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class GravitasEngine:
    """High-level simulation engine with plugin support.

    The engine wraps the multi-agent Stalingrad environment (or standard
    GravitasEnv) and provides plugin hooks, config loading, and a clean
    run/eval interface.

    Attributes:
        scenario_name: Name of the loaded scenario.
        plugins: List of active GravitasPlugin instances.
        env: The underlying environment instance.
    """

    def __init__(
        self,
        scenario: str = "stalingrad",
        plugins: Optional[List[GravitasPlugin]] = None,
        seed: int = 42,
        max_steps: int = 500,
        verbose: bool = True,
    ):
        self.scenario_name = scenario
        self.plugins: List[GravitasPlugin] = plugins or []
        self.seed = seed
        self.max_steps = max_steps
        self.verbose = verbose

        self._env = None
        self._regime_data = None
        self._params = None
        self._init_data = None

        # Load scenario
        self._load_scenario(scenario, seed)

    def _load_scenario(self, scenario: str, seed: int) -> None:
        """Load scenario YAML and initialize environment.

        Reads cluster assignments from the YAML ``agents`` section so that
        each scenario can define its own Axis / Soviet / Contested layout.
        Falls back to Stalingrad defaults if the section is missing.
        """
        from gravitas_engine.core.gravitas_params import GravitasParams
        from gravitas_engine.agents.stalingrad_ma import (
            StalingradMultiAgentEnv,
            DEFAULT_AXIS_CLUSTERS,
            DEFAULT_SOVIET_CLUSTERS,
            CONTESTED_CLUSTERS,
        )

        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(_ROOT))
        from regime_loader import load_standalone_regime, build_initial_states

        scenario_path = find_scenario(scenario)
        logger.info(f"Loading scenario: {scenario_path}")

        regime = load_standalone_regime(str(scenario_path), seed=seed)
        self._regime_data = regime

        params = regime["params"]
        params = GravitasParams(**{
            **{k: getattr(params, k) for k in GravitasParams.__dataclass_fields__},
            "n_clusters_max": 12,
        })
        if self.max_steps:
            params = GravitasParams(**{
                **{k: getattr(params, k) for k in GravitasParams.__dataclass_fields__},
                "max_steps": self.max_steps,
            })
        self._params = params

        init = build_initial_states(regime, max_N=12)
        self._init_data = init

        # ── Extract cluster assignments from YAML agents section ─────── #
        axis_clusters = DEFAULT_AXIS_CLUSTERS
        soviet_clusters = DEFAULT_SOVIET_CLUSTERS
        contested_clusters = CONTESTED_CLUSTERS

        raw_yaml = regime.get("raw", {})
        yaml_agents = raw_yaml.get("agents", [])
        yaml_sectors = raw_yaml.get("sectors", [])

        if yaml_agents:
            axis_clusters = []
            soviet_clusters = []
            contested_clusters = []
            for agent_def in yaml_agents:
                side = agent_def.get("side", "").lower()
                clusters = agent_def.get("controlled_clusters", [])
                if side == "axis":
                    axis_clusters.extend(clusters)
                elif side == "soviet":
                    soviet_clusters.extend(clusters)

            # Contested: sectors marked as "Contested" in the sectors list
            for sector in yaml_sectors:
                side = sector.get("side", "").lower()
                if side == "contested":
                    contested_clusters.append(sector["id"])

        # Store scenario metadata for plugins
        self._scenario_meta = {
            "axis_clusters": axis_clusters,
            "soviet_clusters": soviet_clusters,
            "contested_clusters": contested_clusters,
            "sectors": yaml_sectors,
            "logistics_links": raw_yaml.get("logistics_links", {}),
            "terrain": raw_yaml.get("terrain", {}),
            "scenario_name": scenario,
        }

        self._env = StalingradMultiAgentEnv(
            params=params,
            axis_clusters=axis_clusters,
            soviet_clusters=soviet_clusters,
            contested_clusters=contested_clusters,
            initial_clusters=init["initial_clusters"],
            initial_alliances=init["initial_alliances"],
            seed=seed,
        )

    @classmethod
    def from_config(cls, config_path: str, seed: int = 42) -> "GravitasEngine":
        """Create engine from a unified config YAML.

        Args:
            config_path: Path to configs/custom.yaml or similar.
            seed: Random seed.

        Returns:
            Configured GravitasEngine instance.
        """
        path = Path(config_path)
        if not path.exists():
            # Try relative to project root
            path = _ROOT / config_path
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        scenario = config.get("scenario", "stalingrad")
        plugin_names = config.get("plugins", [])
        plugin_configs = config.get("plugin_configs", {})
        max_steps = config.get("max_steps", 500)
        verbose = config.get("verbose", True)

        # Load plugins
        plugins = load_plugins_from_config(plugin_names, plugin_configs)

        engine = cls(
            scenario=scenario,
            plugins=plugins,
            seed=seed,
            max_steps=max_steps,
            verbose=verbose,
        )

        logger.info(
            f"Engine created from config: scenario={scenario}, "
            f"plugins={[p.name for p in plugins]}"
        )
        return engine

    def load_plugins(
        self,
        plugin_names: List[str],
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Load and activate plugins by name.

        Args:
            plugin_names: List of plugin names to load.
            plugin_configs: Optional per-plugin config dicts.
        """
        new_plugins = load_plugins_from_config(plugin_names, plugin_configs)
        self.plugins.extend(new_plugins)
        if self.verbose:
            for p in new_plugins:
                print(f"  Loaded plugin: {p}")

    @property
    def env(self):
        """Access the underlying environment."""
        return self._env

    @property
    def params(self):
        """Access the GravitasParams."""
        return self._params

    # ── Run interface ───────────────────────────────────────────────── #

    def run(
        self,
        episodes: int = 1,
        seed: Optional[int] = None,
        axis_model=None,
        soviet_model=None,
    ) -> List[Dict[str, Any]]:
        """Run simulation episodes with optional trained agents.

        If no models provided, uses random actions.

        Args:
            episodes: Number of episodes to run.
            seed: Base seed (defaults to engine seed).
            axis_model: Trained Axis model (RecurrentPPO or None for random).
            soviet_model: Trained Soviet model (RecurrentPPO or None for random).

        Returns:
            List of per-episode result dicts.
        """
        from gravitas_engine.agents.stalingrad_ma import AXIS, SOVIET

        if seed is None:
            seed = self.seed

        all_results = []
        t0 = time.time()

        for ep in range(episodes):
            result = self._run_episode(
                ep, seed + ep, axis_model, soviet_model,
            )
            all_results.append(result)

            if self.verbose:
                w = result["winner"]
                sym = {"Axis": "🔴", "Soviet": "🔵", "draw": "⚪", "collapse": "💀"}.get(w, "❓")
                print(
                    f"  Ep {ep+1:>3}/{episodes}  "
                    f"steps={result['steps']:>4}  "
                    f"Ax σ̄={result['axis_sigma']:.3f}  "
                    f"Sv σ̄={result['soviet_sigma']:.3f}  "
                    f"Ax R={result['axis_reward']:>+8.1f}  "
                    f"Sv R={result['soviet_reward']:>+8.1f}  "
                    f"{sym} {w}"
                )

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n  Completed {episodes} episodes in {elapsed:.1f}s")
            self._print_summary(all_results)

        return all_results

    def _run_episode(
        self,
        ep_idx: int,
        seed: int,
        axis_model,
        soviet_model,
    ) -> Dict[str, Any]:
        """Run a single episode with plugin hooks."""
        from gravitas_engine.agents.stalingrad_ma import AXIS, SOVIET

        env = self._env
        obs_dict = env.reset(seed=seed)

        # Plugin reset hooks — pass engine ref so plugins can read scenario metadata
        for plugin in self.plugins:
            if plugin.enabled:
                env._world = plugin.on_reset(env._world, engine=self, env=env)

        ax_obs = obs_dict[AXIS]
        sv_obs = obs_dict[SOVIET]
        ax_lstm, sv_lstm = None, None
        ax_starts = np.ones((1,), dtype=bool)
        sv_starts = np.ones((1,), dtype=bool)
        ax_total_r, sv_total_r = 0.0, 0.0
        done, trunc = False, False
        step = 0
        plugin_events = []

        while not (done or trunc):
            # Get actions
            if axis_model is not None:
                ax_act, ax_lstm = axis_model.predict(
                    ax_obs.reshape(1, -1), state=ax_lstm,
                    episode_start=ax_starts, deterministic=True,
                )
                ax_action = int(ax_act[0])
            else:
                ax_action = int(np.random.randint(0, env.action_space.n))

            if soviet_model is not None:
                sv_act, sv_lstm = soviet_model.predict(
                    sv_obs.reshape(1, -1), state=sv_lstm,
                    episode_start=sv_starts, deterministic=True,
                )
                sv_action = int(sv_act[0])
            else:
                sv_action = int(np.random.randint(0, env.action_space.n))

            actions = {AXIS: ax_action, SOVIET: sv_action}
            obs_dict, rewards, done, trunc, info = env.step(actions)

            # Plugin step hooks
            for plugin in self.plugins:
                if plugin.enabled:
                    env._world = plugin.on_step(
                        env._world, step,
                        info=info,
                        actions=actions,
                        rewards=rewards,
                    )

            ax_obs = obs_dict[AXIS]
            sv_obs = obs_dict[SOVIET]
            ax_starts = np.zeros((1,), dtype=bool)
            sv_starts = np.zeros((1,), dtype=bool)
            ax_total_r += rewards[AXIS]
            sv_total_r += rewards[SOVIET]
            step += 1

        # Plugin end hooks + collect events
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_episode_end(env._world, step)
                plugin_events.extend(plugin.events)

        # Compute final stats
        c_arr = env.world.cluster_array()
        ax_sigma = float(np.mean(c_arr[env.axis_clusters, 0]))
        sv_sigma = float(np.mean(c_arr[env.soviet_clusters, 0]))

        if done:
            winner = "collapse"
        elif ax_sigma > sv_sigma + 0.05:
            winner = "Axis"
        elif sv_sigma > ax_sigma + 0.05:
            winner = "Soviet"
        else:
            winner = "draw"

        return {
            "episode": ep_idx,
            "steps": step,
            "axis_reward": ax_total_r,
            "soviet_reward": sv_total_r,
            "axis_sigma": ax_sigma,
            "soviet_sigma": sv_sigma,
            "winner": winner,
            "collapsed": done,
            "plugin_events": plugin_events,
        }

    def _print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print aggregate summary of results."""
        n = len(results)
        if n == 0:
            return

        winners = [r["winner"] for r in results]
        ax_wins = sum(1 for w in winners if w == "Axis")
        sv_wins = sum(1 for w in winners if w == "Soviet")
        draws = sum(1 for w in winners if w == "draw")
        collapses = sum(1 for w in winners if w == "collapse")

        ax_r = np.array([r["axis_reward"] for r in results])
        sv_r = np.array([r["soviet_reward"] for r in results])

        print(f"\n  {'═' * 50}")
        print(f"  🔴 Axis wins:   {ax_wins:>3} ({100*ax_wins/n:.0f}%)")
        print(f"  🔵 Soviet wins: {sv_wins:>3} ({100*sv_wins/n:.0f}%)")
        print(f"  ⚪ Draws:       {draws:>3} ({100*draws/n:.0f}%)")
        print(f"  💀 Collapses:   {collapses:>3} ({100*collapses/n:.0f}%)")
        print(f"  Axis  reward: {ax_r.mean():>+8.1f} ± {ax_r.std():.1f}")
        print(f"  Soviet reward: {sv_r.mean():>+8.1f} ± {sv_r.std():.1f}")

        # Plugin event summary
        all_events = []
        for r in results:
            all_events.extend(r.get("plugin_events", []))
        if all_events:
            plugin_names = set(e["plugin"] for e in all_events)
            print(f"\n  Plugin events: {len(all_events)} total from {plugin_names}")

        print(f"  {'═' * 50}")


# Backward compatibility alias
Engine = GravitasEngine

__all__ = ["GravitasEngine", "Engine", "find_scenario", "list_scenarios"]
