"""
gravitas.plugins — Plugin system for GRAVITAS Engine.

Plugins extend simulation behavior by hooking into the step loop.
Each plugin implements a standardized interface:

    class MyPlugin(GravitasPlugin):
        name = "my_plugin"

        def on_step(self, world, turn, **kwargs):
            # Modify world state, return modified world
            return world

Plugins are loaded automatically from config or discovered in this package.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gravitas_engine.core.gravitas_state import GravitasWorld

logger = logging.getLogger("gravitas.plugins")


class GravitasPlugin(ABC):
    """Base class for all GRAVITAS plugins.

    Attributes:
        name: Unique plugin identifier (e.g., "soviet_reinforcements").
        description: Human-readable description of the plugin.
        version: Semantic version string.
        enabled: Whether the plugin is currently active.
    """

    name: str = "base_plugin"
    description: str = "Base plugin — override this."
    version: str = "1.0.0"
    enabled: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with optional config from custom.yaml."""
        self.config = config or {}
        self._step_count = 0
        self._event_log: List[Dict[str, Any]] = []

    # ── Lifecycle hooks ─────────────────────────────────────────────── #

    def on_init(self, world: "GravitasWorld", **kwargs) -> "GravitasWorld":
        """Called once when the engine initializes. Override to set up state."""
        return world

    def on_reset(self, world: "GravitasWorld", **kwargs) -> "GravitasWorld":
        """Called on each episode reset. Override to reset plugin state."""
        self._step_count = 0
        self._event_log = []
        return world

    @abstractmethod
    def on_step(self, world: "GravitasWorld", turn: int, **kwargs) -> "GravitasWorld":
        """Called after each simulation step. Must return (possibly modified) world.

        Args:
            world: Current GravitasWorld state.
            turn: Current turn number (0-indexed).
            **kwargs: Additional context (e.g., info dict, actions taken).

        Returns:
            Modified GravitasWorld state.
        """
        ...

    def on_episode_end(self, world: "GravitasWorld", turn: int, **kwargs) -> None:
        """Called when an episode ends. Override for cleanup or logging."""
        pass

    # ── Helpers ──────────────────────────────────────────────────────── #

    def log_event(self, turn: int, message: str, data: Optional[Dict] = None) -> None:
        """Record a plugin event for later analysis."""
        event = {"turn": turn, "plugin": self.name, "message": message}
        if data:
            event["data"] = data
        self._event_log.append(event)
        logger.info(f"[{self.name}] Turn {turn}: {message}")

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Return all logged events."""
        return list(self._event_log)

    def __repr__(self) -> str:
        status = "ON" if self.enabled else "OFF"
        return f"<{self.__class__.__name__}({self.name}) [{status}] v{self.version}>"


# ─────────────────────────────────────────────────────────────────────────── #
# Plugin discovery & loading                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

_BUILTIN_PLUGINS: Dict[str, str] = {
    "soviet_reinforcements": "gravitas.plugins.soviet_reinforcements",
    "axis_airlift": "gravitas.plugins.axis_airlift",
    "nonlinear_combat": "gravitas.plugins.nonlinear_combat",
    "logistics_network": "gravitas.plugins.logistics_network",
    "partisan_warfare": "gravitas.plugins.partisan_warfare",
}


def discover_plugins() -> Dict[str, str]:
    """Discover all built-in plugins. Returns {name: module_path}."""
    return dict(_BUILTIN_PLUGINS)


def load_plugin(
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> GravitasPlugin:
    """Load a plugin by name with optional config.

    Args:
        name: Plugin name (e.g., "soviet_reinforcements").
        config: Plugin-specific config dict from custom.yaml.

    Returns:
        Instantiated GravitasPlugin.

    Raises:
        ValueError: If plugin name is not found.
    """
    available = discover_plugins()
    if name not in available:
        raise ValueError(
            f"Unknown plugin '{name}'. Available: {list(available.keys())}"
        )

    module = importlib.import_module(available[name])

    # Convention: plugin module exposes a `Plugin` class
    if not hasattr(module, "Plugin"):
        raise AttributeError(
            f"Plugin module '{available[name]}' must export a 'Plugin' class."
        )

    plugin_cls = module.Plugin
    if not issubclass(plugin_cls, GravitasPlugin):
        raise TypeError(
            f"Plugin class in '{name}' must inherit from GravitasPlugin."
        )

    instance = plugin_cls(config=config)
    logger.info(f"Loaded plugin: {instance}")
    return instance


def load_plugins_from_config(
    plugin_names: List[str],
    plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[GravitasPlugin]:
    """Load multiple plugins from a config specification.

    Args:
        plugin_names: List of plugin names to load.
        plugin_configs: Optional dict mapping plugin name -> config dict.

    Returns:
        List of instantiated GravitasPlugin objects.
    """
    plugin_configs = plugin_configs or {}
    plugins = []
    for name in plugin_names:
        cfg = plugin_configs.get(name, {})
        try:
            plugin = load_plugin(name, config=cfg)
            plugins.append(plugin)
        except Exception as e:
            logger.error(f"Failed to load plugin '{name}': {e}")
            raise
    return plugins


__all__ = [
    "GravitasPlugin",
    "discover_plugins",
    "load_plugin",
    "load_plugins_from_config",
]
