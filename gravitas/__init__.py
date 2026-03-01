"""
gravitas — High-level orchestration package for GRAVITAS Engine.

This package provides:
  - GravitasEngine: Plugin-aware simulation engine with scenario loading.
  - Plugin system: Modular, hot-pluggable simulation extensions.
  - CLI interface: ``python cli.py run stalingrad --episodes 30``

Quick start:
    from gravitas import GravitasEngine

    engine = GravitasEngine(scenario="stalingrad")
    engine.load_plugins(["soviet_reinforcements", "axis_airlift"])
    results = engine.run(episodes=10)

From config:
    engine = GravitasEngine.from_config("configs/custom.yaml")
    results = engine.run(episodes=30)

Backward compatibility:
    The ``Engine`` alias is provided for convenience:
        from gravitas import Engine
"""

from gravitas.engine import GravitasEngine, Engine, find_scenario, list_scenarios
from gravitas.plugins import GravitasPlugin, load_plugin, discover_plugins

__version__ = "3.0.0"

__all__ = [
    "GravitasEngine",
    "Engine",
    "GravitasPlugin",
    "find_scenario",
    "list_scenarios",
    "load_plugin",
    "discover_plugins",
    "__version__",
]
