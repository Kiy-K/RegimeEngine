# Battle of Moscow Scenario

## Overview

The Battle of Moscow (October 1941 – January 1942) was the first major Soviet
counteroffensive of World War II. Operation Typhoon — the German drive on Moscow —
stalled in the face of Soviet defensive lines, partisan warfare behind German lines,
and the devastating Russian winter that crippled Axis logistics.

This scenario models the battle with **9 sectors**, **34 unit types**, **physics-driven terrain/weather**, and **advanced unit traits** — creating a realistic simulation that challenges RL agents with complex tactical decisions.

## Key Features

- **CoW-Native Combat**: Full Call of War-style combat with terrain bonuses, morale dynamics, and production systems
- **Physics Integration**: Realistic terrain effects, weather attrition, supply logistics, and line-of-sight modeling
- **Unit Traits System**: Elite units, terrain specialization, weather adaptation, engineering capabilities, combat specials
- **Historical Accuracy**: Detailed unit compositions, winter effects, and partisan operations based on historical records
- **Dynamic Weather**: Temperature curves, snow accumulation, visibility effects on equipment and movement

## Sectors

| ID | Name | Side | Terrain | Description |
|----|------|------|---------|-------------|
| 0 | Moscow City Center | Soviet | urban | Soviet capital — primary strategic objective |
| 1 | Yaroslavl Rail Hub | Soviet | forest | Logistics hub, supply distribution |
| 2 | Tula Defense Line | Soviet | forest | Fortified defensive line, arms production |
| 3 | Soviet Strategic Reserve | Soviet | forest | Siberian reinforcements, winter troops |
| 4 | Bryansk Forest | Contested | forest | Partisan territory, contested zone |
| 5 | Vyazma Rail Hub | Axis | open | Primary Axis logistics hub |
| 6 | Smolensk Forward Base | Axis | urban | Army Group Center HQ |
| 7 | Kalinin Northern Front | Axis | forest | 3rd Panzer Group advance |
| 8 | Klin-Solnechnogorsk | Axis | open | Closest approach to Moscow |

## Military System

### CoW-Native Combat Engine

The scenario uses a full Call of War-style combat system with realistic physics integration:

#### Core Combat Mechanics

- **Lanchester Square Law**: Attrition scales with force concentration squared
- **Terrain Bonuses**: Urban (+35% for Guards), Forest (+15% defense), Mountains, Marsh
- **Morale Dynamics**: Unit losses affect faction morale based on unit traits
- **Fatigue & Attrition**: Prolonged combat causes exponentially increasing losses
- **Experience System**: Units gain XP from combat, elite units learn 50% faster

#### Unit Production & Economy

- **Dynamic Production**: Sectors produce units based on income and buildings
- **Nonlinear Costs**: Higher-tier units have exponential cost scaling
- **Building System**: Barracks, Bunkers, Supply Depots affect production
- **Resource Management**: Fuel, ammo, and supply consumption affect combat effectiveness

## Physics Integration

### Terrain System

- **Realistic Terrain Types**: Urban, Forest, Open, Hills, Mountains, Marsh, Desert
- **Elevation Effects**: Higher elevation provides defensive bonuses
- **Cover Factors**: Terrain affects unit concealment and detection
- **Fortification**: Engineers can build fortifications that provide defensive bonuses

### Weather Dynamics

- **Temperature Curves**: Historical temperature progression from October to January
- **Snow Accumulation**: Gradual snow buildup affects movement and equipment
- **Visibility Effects**: Weather impacts detection ranges and combat effectiveness
- **Equipment Reliability**: Cold weather causes equipment failures without winterization

### Supply Logistics

- **Rail & Road Networks**: Different capacities for supply distribution
- **Fuel & Ammo Consumption**: Units consume resources based on activity and weather
- **Winter Attrition**: Non-winterized units suffer increased losses in extreme cold
- **Supply Depots**: Strategic hubs for resource distribution and resupply

### Line of Sight & Detection

- **Recon Units**: Specialized units with extended detection ranges
- **Terrain Masking**: Hills and forests block line of sight
- **Night Effects**: Reduced visibility at night affects detection and combat
- **Ambush Mechanics**: Units can ambush from concealed positions

## Running the Scenario

```bash
# Basic run with physics-enabled military system
python cli.py run moscow --episodes 5

# Using the unified config
python cli.py run --config configs/moscow.yaml --episodes 10

# With trained agents
python cli.py run moscow --episodes 30 \
  --axis-model logs/moscow_selfplay/axis_final.zip \
  --soviet-model logs/moscow_selfplay/soviet_final.zip

# Self-play training
python tests/train_moscow_selfplay.py \
  --total-rounds 6 --steps-per-round 25000 --n-envs 4 \
  --log-dir logs/moscow_selfplay
```

## Unit Types & Traits

The scenario features **34 distinct unit types** with specialized traits and capabilities:

### Infantry Units

| Unit Type | Traits | Special Abilities |
|-----------|--------|------------------|
| **Guards Infantry** | Elite, Urban +35%, Forest +15% | High morale impact, defensive bonus |
| **Ski Troops** | Winter hardened | No movement penalty in snow |
| **Shock Troops** | Elite, Breakthrough 25%, Suppression 20% | Fortification reduction, post-combat damage |
| **Mountain Troops** | Elite, Mountain bonus | Superior performance in hills |
| **Engineer** | Fortification building, Mine clearing/laying | Can construct defenses, clear obstacles |
| **Sniper Team** | Elite, Suppression 15%, Recon 2 | Long-range suppression, detection |
| **Recon Infantry** | Recon 2 | Extended detection range |

### Armor Units

| Unit Type | Traits | Special Abilities |
|-----------|--------|------------------|
| **Light Tank** | Recon 1, Breakthrough 10% | Fast scouting, limited breakthrough |
| **Medium Tank** | Breakthrough 15% | Main battle tank, balanced |
| **Heavy Tank** | Elite, Breakthrough 25%, Suppression 10% | Powerful but expensive |
| **Tank Destroyer** | Anti-armor specialization | High damage vs armor |
| **Assault Gun** | Suppression 10% | Direct fire support |
| **Flame Tank** | Breakthrough 30%, Suppression 30% | Anti-structure, area denial |

### Support Units

| Unit Type | Traits | Special Abilities |
|-----------|--------|------------------|
| **Artillery** | Suppression 25% | Long-range fire support |
| **Mortar** | Suppression 15% | Indirect fire, mobile |
| **Rocket Artillery** | Suppression 35% | Area saturation |
| **Anti-Air** | Air defense | Protects from air attacks |
| **Anti-Tank** | Anti-armor specialization | Defensive AT role |

### Logistics & Recon

| Unit Type | Traits | Special Abilities |
|-----------|--------|------------------|
| **Supply Truck** | Resupply capability | Can replenish other units |
| **Armored Car** | Recon 2 | Fast reconnaissance |

### Air Units

| Unit Type | Traits | Special Abilities |
|-----------|--------|------------------|
| **Interceptor** | Recon 3, Night fighter | Air superiority |
| **Tactical Bomber** | Suppression 20% | Ground attack |
| **Attack Bomber** | Anti-structure | Precision strikes |
| **Strategic Bomber** | Suppression 25% | Strategic bombing |

## Configuration

Physics and military parameters are configurable via `gravitas/scenarios/moscow.yaml`:

### Physics Configuration

```yaml
map_physics:
  climate:
    temperature_curve: {0: 5.0, 50: -10.0, 100: -20.0, 150: -30.0}
    humidity: 70
    wind_ms: 5.0
  sectors:
    0: # Moscow
      terrain: URBAN
      elevation_m: 156
      cover: 0.7
      fortification: 0.4
```

### Military Configuration

```yaml
cow_military:
  axis_income: [4.0, 3.5, 2.5, 2.0, 1.0]
  soviet_income: [3.5, 4.0, 2.5, 1.5, 1.5]
  objectives:
    - {objective_id: 0, name: "Capture Moscow", target_cluster_id: 0}
```

## Design Philosophy

The Moscow scenario is designed to be **realistic and challenging** for RL agents:

1. **Physics Integration**: Terrain, weather, and supply logistics create realistic constraints
2. **Unit Specialization**: 34 distinct unit types with traits encourage tactical diversity
3. **Seasonal Dynamics**: Winter attrition and equipment failures create temporal challenges
4. **Asymmetric Balance**: Soviet defensive advantages vs. Axis offensive capabilities
5. **Moral Factors**: Elite units and morale effects add psychological depth
6. **Supply Dependencies**: Logistics network creates vulnerability and strategic depth

## Historical Shock Events

The scenario includes historically accurate events that occur during the battle:

- **Rail Sabotage** (Turn 50+): Partisans disrupt Axis supply lines
- **First Snow** (Turn 100): Initial winter weather, equipment reliability drops
- **Fuel Shortage** (Turn 150): Axis vehicles freeze without winterization
- **Winter Blizzard** (Turn 200): -40°C temperatures, massive attrition
- **Siberian Reinforcements** (Turn 220): Fresh winter-equipped Soviet divisions arrive
- **Factory Evacuation** (Turn 250): Soviet production capacity temporarily reduced
- **German Panic Retreat** (Turn 280): Unauthorized Axis withdrawals from forward positions
- **Soviet Counteroffensive** (Turn 300): Zhukov's coordinated December 1941 push
- **Partisan Uprising** (Turn 320): Coordinated Bryansk Forest operations

## Files

| File | Description |
|------|-------------|
| `gravitas/scenarios/moscow.yaml` | Scenario definition with physics config |
| `extensions/military/cow_combat.py` | 34 unit types, traits, combat resolution |
| `extensions/military/military_dynamics.py` | Combat, production, morale dynamics |
| `extensions/military/physics.py` | Terrain, weather, supply physics engine |
| `extensions/military/physics_bridge.py` | Integration layer between CoW and physics |
| `tests/train_moscow_selfplay.py` | Self-play training script |
| `configs/moscow.yaml` | Unified configuration file |

## Quick Reference

### Action Space
- **MultiDiscrete([9, 9, 9, 34, 5])**: Source cluster, target cluster, unit type, building type, special action

### Observation Space
- **(507,)**: Full world state including physics observations

### Victory Conditions
- **Axis**: Capture Moscow (Sector 0) with 30+ strength
- **Soviet**: Hold Moscow for 50+ turns, defend Tula line

### Training Curriculum
1. **Phase 1**: Operation Typhoon (Axis learns advance)
2. **Phase 2**: Mozhaisk Defense (Soviet learns fortification)
3. **Phase 3**: General Winter (Both sides learn winter tactics)
4. **Phase 4**: Partisan Escalation (Contested territory focus)
5. **Phase 5**: Zhukov Counterattack (Soviet reinforcement wave)
6. **Phase 6**: Full Self-Play (Complete dynamics)
