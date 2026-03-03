# GRAVITAS Engine

> **G**overnance under **R**ecursive **A**nd **V**olatile **I**nstability **T**hrough **A**daptive **S**imulation

High-fidelity simulation and optimization of hierarchical political systems, adversarial multi-agent warfare, and systemic survival under spatial instability.

## Overview

GRAVITAS Engine is a research-grade simulation framework for modeling the dynamics of political regimes, military campaigns, and systemic collapse. It combines non-linear ODE dynamics, Hawkes shock processes, media bias modeling, population demographics, and a full economic subsystem into a unified Gymnasium environment.

The engine supports both **single-agent governance** (PPO stabilization) and **multi-agent adversarial warfare** (RecurrentPPO self-play), with a modular **plugin system** for extending simulation mechanics.

## Key Features

- **CoW-Native Military System**: Full Call of War-style combat with 34 unit types, terrain bonuses, morale dynamics, and physics integration.
- **Physics-Driven Simulation**: Realistic terrain, weather, supply logistics, and line-of-sight modeling integrated with combat mechanics.
- **Advanced Unit Traits**: Elite units, terrain specialization, weather adaptation, engineering capabilities, and combat specials (suppression, breakthrough, ambush).
- **Multi-Agent Warfare**: Adversarial Axis vs. Soviet self-play with per-side observations, actions, and rewards (Battle of Moscow scenario).
- **Plugin Architecture**: Modular, hot-pluggable simulation extensions with standardized `on_step(world, turn)` interface.
- **Historical Scenarios**: YAML-defined scenarios (Moscow 1941 with 9 sectors, detailed terrain, winter attrition, partisan warfare).
- **Hierarchical Modeling**: Districts nested within provinces, with custom adjacency and diffusion rates.
- **Reinforcement Learning**: Trained PPO/RecurrentPPO agents for regime stabilization and adversarial warfare.
- **Economic Subsystem**: Per-cluster GDP, unemployment, debt, and industrial capacity with bidirectional military feedback.
- **Population & Military**: Multi-class demographics, ethnic tension, conscription/desertion dynamics, and 34 military unit types.
- **Spatial Dynamics**: Domino effects, cascade failures, and alliance diplomacy across geographical units.
- **CLI Interface**: `python cli.py run moscow --episodes 30` for quick scenario execution.

## Architecture

```text
GravitasEngine/
├── gravitas/                    # High-level orchestration (NEW)
│   ├── __init__.py              # GravitasEngine + Engine alias
│   ├── engine.py                # Core engine with plugin loader
│   ├── plugins/                 # Plugin system
│   │   ├── __init__.py          # GravitasPlugin base class + discovery
│   │   ├── soviet_reinforcements.py  # Volga barge crossing mechanic
│   │   └── axis_airlift.py      # Luftwaffe airlift mechanic
│   └── scenarios/               # Scenario YAML files
│       ├── moscow.yaml          # 9-sector Battle of Moscow (NEW)
│       └── stalingrad.yaml      # 9-sector Battle of Stalingrad
├── gravitas_engine/             # Core simulation engine
│   ├── core/                    # State, params, integrator
│   ├── agents/                  # Environments (GravitasEnv, StalingradMA)
│   ├── systems/                 # ODE dynamics, shocks, media, economy
│   └── analysis/                # Metrics, logging
├── extensions/                  # Military wrapper, political interface
│   ├── military/                # CoW-native military system (NEW)
│   │   ├── cow_combat.py        # 34 unit types, traits, combat resolution
│   │   ├── military_dynamics.py # Combat, production, research, morale
│   │   ├── military_state.py    # State dataclasses
│   │   ├── military_wrapper.py   # Gymnasium environment
│   │   ├── physics.py           # Physics engine (terrain, weather, supply)
│   │   ├── physics_bridge.py    # Integration layer
│   │   └── unit_types.py        # Legacy unit type mappings
│   └── pop/                     # Population dynamics
├── configs/                     # Unified YAML configuration
│   ├── custom.yaml              # Plugin + scenario config
│   └── moscow.yaml              # Moscow scenario with physics config
├── cli.py                       # CLI entry point
├── tests/                       # Training, evaluation, replay scripts
│   └── train_moscow_selfplay.py # Moscow self-play training (NEW)
└── docs/                        # Documentation
```

## Installation

```bash
git clone https://github.com/Kiy-K/Gravitas-Engine.git
cd Gravitas-Engine
pip install -r requirements.txt
```

## Quick Start

### CLI (Recommended)

```bash
# Run Moscow with physics-enabled military system
python cli.py run moscow --episodes 5

# Run from unified config
python cli.py run --config configs/moscow.yaml --episodes 30

# List available scenarios and plugins
python cli.py list scenarios
python cli.py list plugins

# Run with trained agents
python cli.py run moscow --episodes 30 \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip

# Detailed battle replay with sector-by-sector commentary
python cli.py replay \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip
```

### Python API

```python
from gravitas import GravitasEngine

# From config file
engine = GravitasEngine.from_config("configs/custom.yaml")
results = engine.run(episodes=10)

# Manual setup
engine = GravitasEngine(scenario="stalingrad", seed=42)
engine.load_plugins(["soviet_reinforcements", "axis_airlift"])
results = engine.run(episodes=30)

# Access environment directly
env = engine.env
obs = env.reset(seed=0)
```

### Single-Agent Training

```bash
python train_ppo.py
```

### Multi-Agent Self-Play Training

```bash
python tests/train_moscow_selfplay.py \
    --total-rounds 6 --steps-per-round 25000 --n-envs 4 \
    --log-dir logs/moscow_selfplay
```

### Battle Evaluation

```bash
python tests/eval_moscow_battle.py \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip \
    --n-episodes 30
```

## Plugin System

Plugins extend simulation behavior by hooking into the step loop. Each plugin implements a standardized interface:

```python
from gravitas.plugins import GravitasPlugin

class Plugin(GravitasPlugin):
    name = "my_plugin"
    description = "What this plugin does."

    def on_step(self, world, turn, **kwargs):
        # Inspect and modify world state
        c_arr = world.cluster_array()
        # ... apply mechanic ...
        return world  # Return modified world
```

### Built-in Plugins

| Plugin | Description | Historical Basis |
|--------|-------------|-----------------|
| `soviet_reinforcements` | Boosts Soviet reserves when Volga Crossing is held | Nightly barge crossings across the Volga |
| `axis_airlift` | Diminishing supply drops to encircled Axis sectors | Göring's failed Luftwaffe airlift promise |

### Plugin Lifecycle

| Hook | When Called | Purpose |
|------|-----------|---------|
| `on_init(world)` | Engine init | One-time setup |
| `on_reset(world)` | Episode start | Reset plugin state |
| `on_step(world, turn)` | Each step | Core mechanic (required) |
| `on_episode_end(world, turn)` | Episode end | Cleanup, logging |

### Configuration

Plugins are configured via `configs/custom.yaml`:

```yaml
plugins:
  - soviet_reinforcements
  - axis_airlift

plugin_configs:
  soviet_reinforcements:
    trigger_turn_interval: 50
    military_boost: 0.10
    sigma_threshold: 0.5
  axis_airlift:
    trigger_turn_interval: 40
    base_resource_boost: 0.04
    decay_rate: 0.005
```

## Battle of Moscow Scenario

The flagship scenario simulates the decisive Eastern Front battle (Oct 1941 – Jan 1942) with **9 operational sectors**, **physics-driven terrain/weather**, and **34 unit types**:

| ID | Sector | Controller | Terrain | Role |
|----|--------|-----------|---------|------|
| 0 | Moscow City Center | Soviet | urban | Soviet capital — primary strategic objective |
| 1 | Yaroslavl Rail Hub | Soviet | forest | Logistics hub, supply distribution |
| 2 | Tula Defense Line | Soviet | forest | Fortified defensive line, arms production |
| 3 | Soviet Strategic Reserve | Soviet | forest | Siberian reinforcements, winter troops |
| 4 | Bryansk Forest | Contested | forest | Partisan territory, contested zone |
| 5 | Vyazma Rail Hub | Axis | open | Primary Axis logistics hub |
| 6 | Smolensk Forward Base | Axis | urban | Army Group Center HQ |
| 7 | Kalinin Northern Front | Axis | forest | 3rd Panzer Group advance |
| 8 | Klin-Solnechnogorsk | Axis | open | Closest approach to Moscow |

### Physics Integration

- **Terrain Effects**: Urban (+35% defense for Guards), Forest (+15% defense), Mountains, Marsh
- **Weather Dynamics**: Temperature curve, snow depth, visibility, equipment reliability
- **Supply Logistics**: Rail/road capacity, fuel/ammo consumption, winter attrition
- **Line of Sight**: Recon units, terrain masking, detection ranges

### Unit Types & Traits

- **Infantry**: Guards (elite, urban bonus), Ski Troops (winter hardened), Shock Troops (breakthrough)
- **Specialists**: Engineers (fortifications, mines), Snipers (suppression), Mountain Troops (terrain bonus)
- **Armor**: Light/Medium/Heavy tanks, Tank Destroyers, Flame Tanks (anti-structure)
- **Support**: Artillery, Mortars, Rocket Artillery, Anti-Air, Supply Trucks
- **Recon**: Armored Cars, Recon Infantry (detection)

### Historical Shock Events

- **Rail Sabotage**: Partisans disrupt Axis supply lines
- **Fuel Shortage**: Axis vehicles freeze in extreme cold
- **Winter Blizzard**: -40°C temperatures, equipment failures
- **Siberian Divisions**: Fresh winter-equipped Soviet reinforcements
- **Factory Evacuation**: Soviet production relocation
- **German Panic Retreat**: Unauthorized Axis withdrawals
- **Soviet Counteroffensive**: Zhukov's December 1941 push
- **Partisan Uprising**: Coordinated Bryansk Forest operations

### Training Results (Self-Play)

| Phase | Features | Balance |
|-------|----------|---------|
| Base | 34 unit types, terrain | Soviet defensive advantage |
| + Physics | Weather, supply, attrition | Winter favors Soviets |
| + Traits | Elite units, specialists | Dynamic tactical depth |

## Documentation

- [Moscow Scenario](docs/MOSCOW_SCENARIO.md) — Full scenario documentation with physics integration.
- [Military System Guide](docs/MILITARY_SYSTEM.md) — CoW-native combat, unit types, and traits.
- [Physics Integration](docs/PHYSICS_INTEGRATION.md) — Terrain, weather, and supply modeling.
- [Plugin System Guide](docs/PLUGIN_SYSTEM.md) — Writing and configuring plugins.
- [Stalingrad Scenario](docs/STALINGRAD_SCENARIO.md) — Legacy scenario documentation.
- [Architecture Overview](docs/ARCHITECTURE.md) — Engine internals and design.

## Population + Military Modeling

The `PopWrapper` enables multi-class demographics, ethnic tension, and the Soldier archetype with morale/conscription dynamics:

```python
from gravitas_engine.extensions.pop.pop_wrapper import PopWrapper
from gravitas_engine.agents.gravitas_env import GravitasEnv

env = GravitasEnv()
pop_env = PopWrapper(env)
obs, info = pop_env.reset()
```

See `docs/POPULATION_SYSTEM.md` for full details.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
