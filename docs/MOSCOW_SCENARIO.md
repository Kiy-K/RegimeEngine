# Battle of Moscow Scenario

## Overview

The Battle of Moscow (October 1941 – January 1942) was the first major Soviet
counteroffensive of World War II. Operation Typhoon — the German drive on Moscow —
stalled in the face of Soviet defensive lines, partisan warfare behind German lines,
and the devastating Russian winter that crippled Axis logistics.

This scenario models the battle with **9 sectors**, **nonlinear combat dynamics**,
a **graph-based logistics network**, and **autonomous partisan warfare** — creating
a simulation that is hard for RL agents to exploit through simple linear strategies.

## Sectors

| ID | Name | Side | Terrain | Description |
|----|------|------|---------|-------------|
| 0 | Moscow City Center | Soviet | urban | Soviet capital — primary strategic objective |
| 1 | Moscow Defense Ring | Soviet | urban | Fortified defensive perimeter |
| 2 | Mozhaisk Line | Soviet | forest | Western defense line, partisan territory |
| 3 | Tula Industrial Zone | Soviet | urban | Critical arms production center |
| 4 | Vyazma Salient | Contested | open | Key contested crossroads |
| 5 | Kalinin Front | Axis | forest | Northern Axis approach |
| 6 | Guderian Southern Axis | Axis | open | Guderian's 2nd Panzer drive on Tula |
| 7 | Smolensk Supply Hub | Axis | open | Primary Axis logistics hub |
| 8 | Rzhev Salient | Axis | forest | Fortified Axis salient |

## Plugins

### 1. Nonlinear Combat (`nonlinear_combat`)

Replaces linear combat with smooth nonlinear functions that resist exploitation:

- **Lanchester Square Law**: Attrition scales with `m²` — concentrating forces gives
  superlinear advantage, but also superlinear losses
- **Diminishing Returns**: Military effectiveness = `m^α` where α=0.7 (sublinear stacking)
- **Breakthrough Sigmoid**: Breakthrough damage only occurs when force ratio exceeds
  threshold (σ=0.58), with smooth sigmoid transition — no sharp threshold to exploit
- **Combat Fatigue**: Prolonged combat (>120 turns) causes exponentially increasing
  attrition via logistic sigmoid
- **Terrain Multipliers**: Forest (×1.3 defense), urban (×1.5 defense), open (×1.0)
- **Fog of War**: Gaussian noise on damage calculations prevents perfect prediction

### 2. Logistics Network (`logistics_network`)

Graph-based supply system with nonlinear flow dynamics:

- **Sigmoid-Gated Production**: Sectors only produce when stability σ > 0.4 (smooth gate)
- **Quadratic Consumption**: Supply burn rate = `base × m × (1 + h²)` — combat zones
  consume supplies quadratically faster
- **Saturating Flow**: Resource transfer along links follows logistic curve that caps
  at link capacity — prevents infinite funneling
- **Exponential Distance Decay**: `exp(-0.3 × hops)` — distant sectors get less supply
- **Winter Penalties**: Axis consumption +35% in winter (turns 150–400), Soviet +10%
- **Sabotage Cascades**: Vulnerable links can be disrupted, cutting downstream supply
- **Congestion**: Multiple flows through same hub get diminishing returns
- **Resource Deprivation**: Sectors below 15% resources suffer military/stability decay

### 3. Partisan Warfare (`partisan_warfare`)

Autonomous asymmetric force operating behind Axis lines:

- **Recruitment**: New partisans spawn in contested/forest sectors every 25 turns,
  costing Soviet resources. Strength scales with local trust
- **Sabotage**: Target Axis high-value sectors — damage resources, boost hazard.
  Success probability = 60% + 30% × experience
- **Ambush**: Hit-and-run attacks on Axis military. Forest terrain gives 30% bonus.
  Counter-damage from garrison
- **Propaganda**: Boost Soviet trust and polarization in contested zones
- **Detection**: Axis military can detect partisans (12% base × local military).
  Forest/urban terrain reduces detection. Detected partisans lose 30% strength
- **Movement**: Hit-and-run doctrine — partisans relocate after actions, preferring
  forest sectors
- **Morale Decay**: Partisans slowly lose morale; recovery near Soviet sectors

Partisans are **not controlled by either RL agent** — they create irreducible
uncertainty that both sides must adapt to.

## Running the Scenario

```bash
# Basic run with all plugins
python cli.py run moscow --episodes 5 \
  --plugins nonlinear_combat logistics_network partisan_warfare

# Using the unified config
python cli.py run --config configs/moscow.yaml --episodes 10

# With trained agents
python cli.py run moscow --episodes 30 \
  --plugins nonlinear_combat logistics_network partisan_warfare \
  --axis-model logs/moscow/axis_final.zip \
  --soviet-model logs/moscow/soviet_final.zip
```

## Configuration

All plugin parameters are tunable via `configs/moscow.yaml`. Key parameters:

| Parameter | Plugin | Default | Effect |
|-----------|--------|---------|--------|
| `diminishing_returns_alpha` | combat | 0.7 | Force stacking sublinearity |
| `breakthrough_threshold` | combat | 0.58 | Min force ratio for breakthrough |
| `fatigue_midpoint` | combat | 120 | Combat turns before fatigue kicks in |
| `winter_axis_penalty` | logistics | 0.35 | Axis winter consumption multiplier |
| `distance_decay_rate` | logistics | 0.3 | Supply decay per network hop |
| `max_partisans` | partisan | 6 | Maximum active partisan units |
| `ambush_military_damage` | partisan | 0.06 | Damage per successful ambush |

## Design Philosophy

The Moscow scenario is designed to be **harder for RL agents to exploit** than
linear combat models:

1. **Nonlinearity**: All key functions (production, combat, flow) use smooth
   nonlinear curves (sigmoids, power laws, exponential decay) — no sharp thresholds
   to game
2. **Stochasticity**: Fog of war, partisan actions, and sabotage events add
   irreducible randomness
3. **Coupled Systems**: Combat affects logistics (hazard → consumption), logistics
   affect combat (resource deprivation → military decay), partisans affect both
4. **Asymmetry**: Winter favors Soviets, partisans are Soviet-aligned, but Axis
   has stronger initial military — mirrors historical asymmetry
5. **Bounded Returns**: Diminishing returns on force concentration, saturating
   supply flow, and congestion all prevent degenerate strategies

## Files

| File | Description |
|------|-------------|
| `gravitas/scenarios/moscow.yaml` | Scenario definition (sectors, agents, shocks, terrain, logistics) |
| `gravitas/plugins/nonlinear_combat.py` | Lanchester combat, fatigue, breakthrough, terrain |
| `gravitas/plugins/logistics_network.py` | Supply graph, production, flow, sabotage |
| `gravitas/plugins/partisan_warfare.py` | Autonomous partisan units |
| `configs/moscow.yaml` | Unified config with all plugin parameters |
