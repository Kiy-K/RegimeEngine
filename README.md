# GRAVITAS Engine

> **G**overnance under **R**ecursive **A**nd **V**olatile **I**nstability **T**hrough **A**daptive **S**imulation

High-fidelity simulation and optimization of hierarchical political systems, adversarial multi-agent warfare, and systemic survival under spatial instability.

## Overview

GRAVITAS Engine is a research-grade simulation framework for modeling the dynamics of political regimes, military campaigns, and systemic collapse. It combines non-linear ODE dynamics, Hawkes shock processes, media bias modeling, population demographics, and a full economic subsystem into a unified Gymnasium environment.

The engine supports both **single-agent governance** (PPO stabilization) and **multi-agent adversarial warfare** (RecurrentPPO self-play), with a modular **plugin system** for extending simulation mechanics.

## Key Features

- **Multi-Agent Warfare**: Adversarial Axis vs. Soviet self-play with per-side observations, actions, and rewards (Battle of Stalingrad scenario).
- **Plugin Architecture**: Modular, hot-pluggable simulation extensions with standardized `on_step(world, turn)` interface.
- **Historical Scenarios**: YAML-defined scenarios (Stalingrad with 9 operational sectors, Operation Wintergewitter, Volga reinforcements).
- **Hierarchical Modeling**: Districts nested within provinces, with custom adjacency and diffusion rates.
- **Reinforcement Learning**: Trained PPO/RecurrentPPO agents for regime stabilization and adversarial warfare.
- **Economic Subsystem**: Per-cluster GDP, unemployment, debt, and industrial capacity with bidirectional military feedback.
- **Population & Military**: Multi-class demographics, ethnic tension, conscription/desertion dynamics, and 9 political unit types.
- **Spatial Dynamics**: Domino effects, cascade failures, and alliance diplomacy across geographical units.
- **CLI Interface**: `python cli.py run stalingrad --episodes 30` for quick scenario execution.

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
│       └── stalingrad.yaml      # 9-sector Battle of Stalingrad
├── gravitas_engine/             # Core simulation engine
│   ├── core/                    # State, params, integrator
│   ├── agents/                  # Environments (GravitasEnv, StalingradMA)
│   ├── systems/                 # ODE dynamics, shocks, media, economy
│   └── analysis/                # Metrics, logging
├── extensions/                  # Military wrapper, political interface
├── configs/                     # Unified YAML configuration
│   └── custom.yaml              # Plugin + scenario config
├── cli.py                       # CLI entry point
├── tests/                       # Training, evaluation, replay scripts
└── docs/                        # Documentation
```

## Installation

```bash
git clone https://github.com/Kiy-K/RegimeEngine.git
cd RegimeEngine
pip install -r requirements.txt
```

## Quick Start

### CLI (Recommended)

```bash
# Run Stalingrad with both plugins
python cli.py run stalingrad --episodes 5 --plugins soviet_reinforcements axis_airlift

# Run from unified config
python cli.py run --config configs/custom.yaml --episodes 30

# List available scenarios and plugins
python cli.py list scenarios
python cli.py list plugins

# Run with trained agents
python cli.py run stalingrad --episodes 30 \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip

# Detailed battle replay with sector-by-sector commentary
python cli.py replay \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip
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
python tests/train_stalingrad_selfplay.py \
    --total-rounds 6 --steps-per-round 25000 --n-envs 4 \
    --log-dir logs/stalingrad_selfplay_volga
```

### Battle Evaluation

```bash
python tests/eval_stalingrad_battle.py \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip \
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

## Battle of Stalingrad Scenario

The flagship scenario simulates the decisive Eastern Front battle (Aug 1942 – Feb 1943) with **9 operational sectors**:

| ID | Sector | Controller | Role |
|----|--------|-----------|------|
| 0 | Stalingrad City Center | Axis | Urban combat, extreme hazard |
| 1 | Tractor Factory District | Axis | Industrial zone, main assault axis |
| 2 | Mamayev Kurgan | Contested | Strategic hilltop |
| 3 | Volga Crossing | Soviet | Lifeline for reinforcements |
| 4 | Northern Don River Line | Axis | Defensive arc |
| 5 | Axis Supply Corridor | Axis | Overextended logistics |
| 6 | Soviet Strategic Reserve | Soviet | Operation Uranus buildup |
| 7 | Romanian/Italian Sector | Axis | Weak southern flank |
| 8 | Wintergewitter Corridor | Axis | Hoth's relief attempt |

### Historical Shock Events

- **Operation Wintergewitter** (Turn 200+): German 4th Panzer Army relief push.
- **Operation Little Saturn** (Turn 250+): Soviet counter-offensive cuts off relief.
- **Panzer Spearhead**: Temporary Axis military surge.
- **Soviet Guards Block**: Defensive counter at the relief corridor.

### Training Results (Self-Play)

| Phase | Axis σ̄ | Soviet σ̄ | Soviet Win % |
|-------|--------|----------|-------------|
| Base (8 clusters) | 0.670 | 0.953 | 100% |
| + Wintergewitter (9 clusters) | 0.980 | 0.949 | 3% |
| + Volga Reinforcements | 0.853 | 0.911 | **60%** |

## Documentation

- [Plugin System Guide](docs/PLUGIN_SYSTEM.md) — Writing and configuring plugins.
- [Stalingrad Scenario](docs/STALINGRAD_SCENARIO.md) — Full scenario documentation.
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
