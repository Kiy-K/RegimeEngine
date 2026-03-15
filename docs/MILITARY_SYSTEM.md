# Military System Guide

> **NOTE**: This document describes the legacy Call of War-style ground unit system (`extensions/military/`).
> The **Air Strip One scenario** uses separate naval and air force systems:
> - **Naval**: `extensions/naval/` — 14 ship classes, 6 sea zones, Lanchester combat, 3 invasion types (PREPARED/RECKLESS/AIRBORNE)
> - **Air Force**: `extensions/air_force/` — 10 aircraft types, 7 bases per side, air zone control
> - **Land Combat**: `extensions/military/land_bridge.py` — Per-sector garrisons using CoW units, contested sector combat
> - **Manpower**: `extensions/manpower/` — 15 conscription laws, training pipeline
> - **Population**: `extensions/pop/pop_v2.py` — Real 1958 census numbers, 1984 social classes, 8 job types
>
> See [AIRSTRIP_ONE_SYSTEMS.md](AIRSTRIP_ONE_SYSTEMS.md) for the complete current military documentation.

## Legacy Overview

The GRAVITAS Engine features a comprehensive Call of War-style military system with **34 distinct unit types**, **advanced unit traits**, and **physics-driven combat**. This system provides realistic tactical depth while maintaining compatibility with reinforcement learning training.

## Architecture

```
extensions/military/
├── cow_combat.py          # Core combat engine, unit definitions, traits
├── military_dynamics.py   # Game loop, combat resolution, production
├── military_state.py      # State dataclasses, world representation
├── military_wrapper.py    # Gymnasium environment interface
├── physics.py             # Physics engine (terrain, weather, supply)
├── physics_bridge.py      # Integration layer
└── unit_types.py          # Legacy type mappings
```

## Unit Types

### Infantry (12 types)

| Unit | Role | Traits | Special |
|-------|------|--------|---------|
| **Militia** | Basic defense | Low cost, weak | Emergency defense |
| **Infantry** | Standard | Balanced | Main combat unit |
| **Motorized Infantry** | Mobile | Truck transport | Faster movement |
| **Mechanized Infantry** | Armored | Half-track transport | Breakthrough 10% |
| **Commandos** | Elite | Elite, high morale | Special operations |
| **Paratroopers** | Airborne | Elite, air drop | Behind enemy lines |
| **Guards Infantry** | Elite | Elite, urban +35%, forest +15% | Defensive specialists |
| **Ski Troops** | Winter | Winter hardened | No snow penalty |
| **Cavalry** | Recon | Fast, recon 2 | Scouting |
| **Penal Battalion** | Shock | Low morale, breakthrough 10% | Disposable assault |
| **Recon Infantry** | Scouts | Recon 2 | Extended detection |
| **Mountain Troops** | Mountain | Elite, mountain bonus | High altitude |
| **Shock Troops** | Assault | Elite, breakthrough 25%, suppression 20% | Fortification busters |
| **Engineer** | Support | Fortification building, mines | Construction/demolition |
| **Sniper Team** | Special | Elite, suppression 15%, recon 2 | Long-range support |

### Armor (6 types)

| Unit | Role | Traits | Special |
|-------|------|--------|---------|
| **Armored Car** | Light recon | Recon 2, fast | Scouting |
| **Light Tank** | Light armor | Recon 1, breakthrough 10% | Fast combat |
| **Medium Tank** | Main battle | Breakthrough 15% | Balanced |
| **Heavy Tank** | Heavy armor | Elite, breakthrough 25%, suppression 10% | Powerful but slow |
| **Tank Destroyer** | Anti-armor | High AT damage | Armor specialist |
| **Assault Gun** | Support | Suppression 10% | Direct fire |
| **Flame Tank** | Anti-structure | Breakthrough 30%, suppression 30% | Building destroyer |

### Artillery & Support (7 types)

| Unit | Role | Traits | Special |
|-------|------|--------|---------|
| **Anti-Tank** | AT defense | High AT damage | Defensive |
| **Artillery** | Indirect fire | Suppression 25% | Long range |
| **SP Artillery** | Mobile artillery | Suppression 20% | Self-propelled |
| **Anti-Air** | Air defense | AA protection | Anti-aircraft |
| **SP Anti-Air** | Mobile AA | AA protection | Self-propelled |
| **Mortar** | Light indirect | Suppression 15% | Short range |
| **Rocket Artillery** | Area saturation | Suppression 35% | Area effect |

### Logistics (1 type)

| Unit | Role | Traits | Special |
|-------|------|--------|---------|
| **Supply Truck** | Logistics | Resupply capability | Resource distribution |

### Air Units (4 types)

| Unit | Role | Traits | Special |
|-------|------|--------|---------|
| **Interceptor** | Air superiority | Recon 3, night fighter | Air combat |
| **Tactical Bomber** | Ground attack | Suppression 20% | Close air support |
| **Attack Bomber** | Precision strike | Anti-structure | Targeted bombing |
| **Strategic Bomber** | Strategic | Suppression 25% | Long-range bombing |

## Unit Traits System

### Core Traits

Each unit has a `UnitTraits` dataclass with the following fields:

#### Combat Traits
- **elite**: Boolean - Elite units gain 50% more XP
- **breakthrough**: Float (0-1) - Reduces enemy fortification effectiveness
- **suppression**: Float (0-1) - Reduces enemy HP post-combat
- **recon_range**: Int (0-3) - Detection range in sectors

#### Terrain Specialization
- **urban_bonus**: Float - Additional defense in urban terrain
- **forest_bonus**: Float - Additional defense in forest terrain
- **mountain_bonus**: Float - Additional defense in mountain terrain
- **plain_bonus**: Float - Additional defense in open terrain

#### Weather Adaptation
- **winter_hardened**: Boolean - 50% less winter attrition
- **mud_resistant**: Boolean - Reduced movement penalty in mud
- **night_fighter**: Boolean - No combat penalty at night

#### Engineering Skills
- **entrench**: Float - Fortification building speed
- **mine_clearing**: Float - Mine removal capability
- **mine_laying**: Float - Mine placement capability
- **bridge_building**: Float - Bridge construction speed

#### Logistics
- **supply_consumption**: Float - Base supply consumption multiplier
- **fuel_consumption**: Float - Fuel usage multiplier
- **resupply_capability**: Float - Can resupply other units

#### Morale Effects
- **morale_on_death**: Float - Morale impact when unit is destroyed

### Trait Examples

#### Guards Infantry
```python
UnitTraits(
    elite=True,
    urban_bonus=0.35,      # +35% defense in cities
    forest_bonus=0.15,     # +15% defense in forests
    morale_on_death=0.06,   # Significant morale loss when killed
    breakthrough=0.0,
    suppression=0.10,      # Post-combat suppression
)
```

#### Shock Troops
```python
UnitTraits(
    elite=True,
    breakthrough=0.25,      # Reduces fortifications by 25%
    suppression=0.20,       # 20% post-combat damage
    morale_on_death=0.05,
    urban_bonus=0.10,
)
```

#### Ski Troops
```python
UnitTraits(
    elite=False,
    winter_hardened=True,  # No winter attrition
    mountain_bonus=0.15,    # Good in mountains
    recon_range=1,
)
```

## Combat System

### Resolution Process

1. **Force Comparison**: Calculate total military strength for both sides
2. **Terrain Modifiers**: Apply terrain bonuses based on unit traits
3. **Breakthrough Calculation**: Apply breakthrough trait to reduce fortifications
4. **Damage Calculation**: Lanchester square law with force concentration
5. **Suppression**: Apply suppression trait to reduce post-combat HP
6. **Experience Gain**: Units gain XP, elite units get 50% bonus
7. **Morale Cascade**: Dead units trigger morale loss based on `morale_on_death`

### Terrain Effects

| Terrain | Default Modifier | Trait Bonus |
|---------|------------------|-------------|
| Urban | ×1.3 defense | Guards Infantry +35% |
| Forest | ×1.2 defense | Guards +15%, Ski +15% |
| Mountains | ×1.15 defense | Mountain +20% |
| Open | ×1.0 | Shock +10% |
| Marsh | ×0.9 movement | Engineers +15% |
| Desert | ×0.8 supply | Mountain +10% |

### Weather Effects

| Condition | Effect | Winter Hardened |
|-----------|--------|-----------------|
| Snow | Movement -30%, non-winterized attrition +50% | No penalty |
| Mud | Movement -50%, supply consumption +25% | Mud resistant: -25% |
| Blizzard | Visibility -70%, equipment failure +20% | Night fighter: no penalty |
| Extreme Cold (-30°C) | Attrition +40%, equipment failure +15% | Winter hardened: -50% |

## Production System

### Building Types

| Building | Cost | Effect |
|----------|------|--------|
| Barracks | 100 | Enables unit production |
| Bunker | 150 | +20% defense bonus |
| Supply Depot | 200 | +50% supply capacity |
| Factory | 300 | +25% production speed |
| Airfield | 250 | Enables air unit production |

### Production Costs

Unit costs follow nonlinear scaling:
```
cost = base_cost × (level ^ 1.5)
```

Higher-tier units become exponentially more expensive, encouraging mixed force composition.

### Income System

Each faction receives base income per turn:
- **Axis**: [4.0, 3.5, 2.5, 2.0, 1.0] per cluster tier
- **Soviet**: [3.5, 4.0, 2.5, 1.5, 1.5] per cluster tier

## Integration with Physics

The military system integrates with the physics engine through `physics_bridge.py`:

### Unit Type Mapping
```python
COW_UNIT_TYPE_MAP = {
    CowUnitType.GUARDS_INFANTRY: "INFANTRY",
    CowUnitType.HEAVY_TANK: "HEAVY_TANK",
    CowUnitType.SUPPLY_TRUCK: "MOTORIZED_INFANTRY",
    # ... mappings for all 34 types
}
```

### Combat Modifiers
Physics provides combat modifiers based on:
- **Terrain**: Defense bonuses, movement costs
- **Weather**: Attrition rates, equipment reliability
- **Supply**: Fuel/ammo availability affects combat effectiveness
- **Line of Sight**: Detection ranges, ambush opportunities

### Attrition Integration
Units suffer weather attrition based on:
```python
if not unit.traits.winter_hardened and not faction.winterized:
    attrition *= 2.0  # Double winter attrition
```

## Usage Examples

### Creating Units
```python
from extensions.military.cow_combat import create_unit, CowUnitType

# Create elite Guards Infantry
guards = create_unit(
    unit_type=CowUnitType.GUARDS_INFANTRY,
    level=3,
    faction_id=1,
    hp=26.0,
    xp=0.0,
)
```

### Resolving Combat
```python
from extensions.military.cow_combat import resolve_cow_combat

# Battle between two armies
result = resolve_cow_combat(
    attackers=axis_army,
    defenders=soviet_army,
    terrain=CowTerrain.URBAN,
    fortification=0.4,
)
```

### Accessing Unit Traits
```python
from extensions.military.cow_combat import get_unit_stats

stats = get_unit_stats(CowUnitType.GUARDS_INFANTRY, level=3)
traits = stats.traits

print(f"Urban bonus: {traits.urban_bonus}")
print(f"Elite: {traits.elite}")
print(f"Breakthrough: {traits.breakthrough}")
```

## Configuration

### Scenario YAML
```yaml
cow_military:
  axis_income: [4.0, 3.5, 2.5, 2.0, 1.0]
  soviet_income: [3.5, 4.0, 2.5, 1.5, 1.5]
  cluster_buildings:
    0: {BARRACKS: 2, BUNKER: 1}  # Moscow
    5: {SUPPLY_DEPOT: 1}         # Vyazma
  objectives:
    - {objective_id: 0, name: "Capture Moscow", target_cluster_id: 0}
  initial_units:
    0:
      - {unit_type: GUARDS_INFANTRY, count: 2, hp: 100, faction: 1}
      - {unit_type: ENGINEER, count: 1, hp: 80, faction: 1}
```

### Training Integration
```python
from extensions.military.military_wrapper import MilitaryWrapper

env = MilitaryWrapper(
    scenario_cfg=scenario_config,
    faction_id=0,  # Axis
    opponent_faction_id=1,  # Soviet
    physics_enabled=True,
)
```

## Design Philosophy

1. **Realistic Depth**: 34 unit types with specialized roles encourage tactical diversity
2. **Trait-Based Specialization**: Units have meaningful differences beyond stats
3. **Physics Integration**: Terrain and weather create realistic constraints
4. **Balanced Asymmetry**: Different factions have unique advantages
5. **Learning-Friendly**: Clear mechanics that RL agents can learn to exploit
6. **Historical Accuracy**: Unit capabilities based on WWII realities

## Extension Points

### Adding New Units
1. Add to `CowUnitType` enum in `cow_combat.py`
2. Define stats in `_UNIT_DEFS` dictionary
3. Add physics mapping in `physics_bridge.py`
4. Update legacy mapping if needed

### Custom Traits
Extend `UnitTraits` dataclass:
```python
@dataclass(frozen=True)
class UnitTraits:
    # ... existing fields ...
    camouflage: float = 0.0      # New trait
    radio_communication: bool = False  # New trait
```

### Combat Mechanics
Modify `resolve_cow_combat()` in `cow_combat.py` to implement new mechanics.

## Troubleshooting

### Common Issues

1. **Unit Not Found**: Check `CowUnitType` enum and `_LEGACY_MAP`
2. **Physics Mapping Missing**: Update `COW_UNIT_TYPE_MAP` in `physics_bridge.py`
3. **Trait Not Applied**: Verify trait integration in `military_dynamics.py`
4. **Production Failed**: Check building requirements and income levels

### Debug Tools

```python
# List all unit types
from extensions.military.cow_combat import CowUnitType
for ut in CowUnitType:
    print(ut.name)

# Check unit stats
from extensions.military.cow_combat import get_unit_stats
stats = get_unit_stats(CowUnitType.GUARDS_INFANTRY, 1)
print(stats)

# Verify physics mapping
from extensions.military.physics_bridge import cow_unit_to_physics_key
key = cow_unit_to_physics_key(CowUnitType.GUARDS_INFANTRY)
print(f"Physics key: {key}")
```
