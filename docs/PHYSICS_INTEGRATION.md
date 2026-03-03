# Physics Integration Guide

## Overview

The GRAVITAS Engine includes a comprehensive physics simulation that models terrain, weather, supply logistics, and line-of-sight effects. This physics engine integrates with the CoW-native military system to provide realistic constraints and tactical depth.

## Architecture

```
extensions/military/physics.py           # Core physics engine
extensions/military/physics_bridge.py     # Integration layer
├── Terrain System                        # Elevation, cover, fortification
├── Weather Dynamics                       # Temperature, snow, visibility
├── Supply Logistics                       # Rail/road, fuel/ammo consumption
├── Line of Sight                          # Detection, ambush, recon
└── Attrition Modeling                     # Weather effects, equipment failure
```

## Terrain System

### Terrain Types

| Terrain | Defense Bonus | Movement Cost | Cover Factor | Special Effects |
|---------|---------------|---------------|--------------|-----------------|
| **Urban** | +30% | 1.2× | 0.7 | Guards Infantry +35% |
| **Forest** | +20% | 1.3× | 0.5 | Ski Troops +15%, ambush bonus |
| **Mountains** | +25% | 1.5× | 0.4 | Mountain Troops +20% |
| **Hills** | +15% | 1.2× | 0.3 | Artillery spotting bonus |
| **Open** | +0% | 1.0× | 0.1 | Shock Troops +10% |
| **Marsh** | -10% | 1.8× | 0.3 | Engineers +15% |
| **Desert** | -5% | 1.4× | 0.2 | Mountain Troops +10% |

### Elevation Effects

- **Higher elevation** provides defensive bonus: `+0.1% per meter` above 100m
- **Maximum bonus**: +25% at 350m elevation
- **Line of sight**: Hills and mountains block detection

### Fortification System

```python
fortification_bonus = 1.0 + (fortification_level * 0.2)
# Level 0: 1.0x (no bonus)
# Level 1: 1.2x (+20% defense)
# Level 2: 1.4x (+40% defense)
# Level 3: 1.6x (+60% defense)
```

Engineers can build fortifications:
- **Build rate**: 0.05 per turn per engineer
- **Maximum**: 0.8 (80% bonus)
- **Cost**: 1 supply point per 0.1 fortification

## Weather Dynamics

### Temperature System

Temperature follows historical curves with seasonal progression:

```yaml
temperature_curve:
  0: 5.0    # Turn 0 (October): 5°C
  50: -10.0  # Turn 50 (November): -10°C
  100: -20.0 # Turn 100 (December): -20°C
  150: -30.0 # Turn 150 (January): -30°C
  200: -25.0 # Turn 200 (February): -25°C
```

### Weather Effects

| Condition | Temperature | Visibility | Movement | Equipment |
|-----------|-------------|-------------|----------|----------|
| **Clear** | Normal | 20 km | 1.0× | 100% reliability |
| **Cloudy** | -2°C | 15 km | 1.0× | 95% reliability |
| **Rain** | -5°C | 10 km | 1.2× | 90% reliability |
| **Snow** | -10°C | 8 km | 1.3× | 85% reliability |
| **Blizzard** | -20°C | 2 km | 1.5× | 70% reliability |

### Snow Accumulation

```python
snow_depth = max(0, temperature_below_zero * 0.5)
movement_penalty = 1.0 + (snow_depth * 0.01)
# 10cm snow: +10% movement penalty
# 50cm snow: +50% movement penalty
```

### Equipment Reliability

```python
base_reliability = 0.95
weather_modifier = 1.0 - (extreme_cold * 0.3)
winterization_bonus = 0.2 if faction.winterized else 0.0

final_reliability = base_reliability * weather_modifier + winterization_bonus
```

## Supply Logistics

### Supply Network

Supply flows through rail and road networks with capacity constraints:

```yaml
supply_routes:
  - {from: 0, to: 1, type: rail, capacity_tpd: 50}
  - {from: 1, to: 2, type: road, capacity_tpd: 20}
  - {from: 2, to: 3, type: rail, capacity_tpd: 40}
```

### Resource Types

| Resource | Unit Consumption | Weather Effect | Winter Hardened |
|----------|------------------|---------------|----------------|
| **Fuel** | 0.5-2.0 per turn | +50% in cold | No penalty |
| **Ammo** | 0.3-1.5 per turn | +25% in mud | No penalty |
| **Food** | 0.1 per unit | No effect | No effect |
| **Medical** | 0.05 per unit | No effect | No effect |

### Distance Decay

```python
supply_efficiency = exp(-0.3 * distance_in_hops)
# 1 hop: 74% efficiency
# 2 hops: 55% efficiency  
# 3 hops: 40% efficiency
```

### Winter Attrition

```python
if temperature < -10°C:
    base_attrition = 0.02  # 2% per turn
    
    if not unit.traits.winter_hardened:
        attrition *= 2.0  # Double for non-winterized
        
    if not faction.winterized:
        attrition *= 1.5  # 50% extra for non-winterized faction
```

## Line of Sight & Detection

### Detection Ranges

| Unit Type | Base Range | Terrain Bonus | Weather Penalty |
|-----------|------------|---------------|-----------------|
| **Infantry** | 1 sector | +0 in forest | -1 in blizzard |
| **Recon Infantry** | 2 sectors | +1 in hills | -1 in snow |
| **Armored Car** | 2 sectors | +1 in open | -1 in blizzard |
| **Sniper Team** | 3 sectors | +2 in urban | -2 in blizzard |
| **Interceptor** | 3 sectors | +1 in clear | -1 in clouds |

### Terrain Masking

- **Hills**: Block line of sight beyond 1 sector
- **Mountains**: Block line of sight beyond 2 sectors  
- **Forests**: Reduce detection by 1 sector
- **Urban**: Reduce detection by 1 sector for non-recon units

### Ambush Mechanics

Units can ambush from concealed positions:

```python
ambush_chance = base_concealment * terrain_bonus
# Forest: 40% base chance
# Urban: 30% base chance  
# Hills: 20% base chance

ambush_damage = unit.attack_power * 1.5  # 50% damage bonus
```

### Night Effects

```python
if time_of_day == "NIGHT":
    detection_range *= 0.5  # Halved detection
    
    if not unit.traits.night_fighter:
        combat_effectiveness *= 0.8  # 20% penalty
```

## Integration with Military System

### Physics Bridge

The `physics_bridge.py` module converts between CoW military units and physics objects:

```python
# Unit type mapping
COW_UNIT_TYPE_MAP = {
    CowUnitType.GUARDS_INFANTRY: "INFANTRY",
    CowUnitType.HEAVY_TANK: "HEAVY_TANK", 
    CowUnitType.SUPPLY_TRUCK: "MOTORIZED_INFANTRY",
}

# Combat modifiers
def extract_cluster_modifiers(physics_state, winterized_axis, winterized_soviet):
    return PhysicsModifiers(
        combat_multiplier=terrain_defense_bonus,
        movement_penalty=weather_movement_penalty,
        supply_modifier=supply_availability_multiplier,
        detection_bonus=los_advantage,
    )
```

### Combat Integration

Physics modifiers affect combat resolution:

```python
# Apply terrain bonus
defense_strength *= physics_modifiers.combat_multiplier

# Apply supply penalty  
if supply_level < 0.3:
    attack_strength *= 0.7  # 30% reduction

# Apply weather attrition
if physics_modifiers.weather_attrition > 0:
    unit.hp *= (1.0 - physics_modifiers.weather_attrition)
```

### Unit Trait Integration

Unit traits interact with physics:

```python
# Winter hardening
if unit.traits.winter_hardened:
    weather_attrition *= 0.5  # Half penalty
    
# Engineering
if unit.traits.entrench > 0 and action == "FORTIFY":
    fortification += unit.traits.entrench
    
# Reconnaissance
detection_range = base_range + unit.traits.recon_range
```

## Configuration

### Physics YAML Configuration

```yaml
map_physics:
  name: "Moscow 1941"
  n_sectors: 9
  
  climate:
    type: "continental"
    temperature_curve: {0: 5.0, 50: -10.0, 100: -20.0, 150: -30.0}
    humidity: 70
    wind_ms: 5.0
    steps_per_day: 4
    
  sectors:
    0: # Moscow
      name: "Moscow City Center"
      terrain: "URBAN"
      elevation_m: 156
      cover: 0.7
      fortification: 0.4
      features:
        river: false
        rail: true
        road: true
        airfield: true
      initial_supply:
        axis: {fuel: 5, ammo: 5, food: 5}
        soviet: {fuel: 100, ammo: 100, food: 100}
        
  supply_routes:
    - {from: 0, to: 1, type: rail, capacity_tpd: 50}
    - {from: 1, to: 2, type: road, capacity_tpd: 20}
    
  factions:
    axis:
      winterized: false
      depot_sectors: [5, 6, 7, 8]
    soviet:
      winterized: true  
      depot_sectors: [0, 1, 3]
```

### Runtime Configuration

```python
# Enable physics in military wrapper
env = MilitaryWrapper(
    scenario_cfg=scenario_config,
    physics_enabled=True,
)

# Access physics state
physics_state = world.physics_states[sector_id]
terrain = physics_state.terrain
weather = physics_state.weather
supply = physics_state.supply_axis  # or supply_soviet
```

## Usage Examples

### Checking Terrain Effects

```python
from extensions.military.physics_bridge import extract_cluster_modifiers

modifiers = extract_cluster_modifiers(
    physics_state, 
    winterized_axis=False, 
    winterized_soviet=True
)

print(f"Combat multiplier: {modifiers.combat_multiplier:.2f}")
print(f"Movement penalty: {modifiers.movement_penalty:.2f}")
print(f"Weather attrition: {modifiers.weather_attrition:.3f}")
```

### Supply Route Analysis

```python
from extensions.military.physics import compute_supply_flow

flow = compute_supply_flow(
    from_sector=0,
    to_sector=3,
    route_type="rail",
    capacity=50,
    weather_modifier=0.8,  # Winter reduces capacity
    distance_hops=2
)

print(f"Supply flow: {flow:.1f} tons/day")
```

### Weather Impact Assessment

```python
from extensions.military.physics import WeatherState

weather = WeatherState(
    temperature_c=-25,  # Extreme cold
    snow_depth_cm=40,   # Heavy snow
    visibility_km=3,    # Poor visibility
    wind_ms=10,         # Strong wind
)

attrition_rate = weather.attrition_rate(winterized=False)
equipment_reliability = weather.equipment_reliability(winterized=False)

print(f"Attrition rate: {attrition_rate:.3f}")
print(f"Equipment reliability: {equipment_reliability:.2f}")
```

### Line of Sight Calculation

```python
from extensions.military.physics import calculate_los

los_result = calculate_los(
    from_sector=2,
    to_sector=5,
    terrain_profile=[156, 200, 180, 220, 190],  # Elevations
    weather_visibility=8,  # Snow
    unit_type="RECON_INFANTRY"
)

print(f"Can detect: {los_result.can_detect}")
print(f"Detection range: {los_result.effective_range} sectors")
print(f"Ambush opportunity: {los_result.ambush_possible}")
```

## Performance Considerations

### Computational Complexity

- **Terrain calculations**: O(n) per sector (n = sectors)
- **Weather updates**: O(1) global (simple interpolation)
- **Supply flow**: O(r × s) where r = routes, s = sectors
- **Line of sight**: O(s²) in worst case (all sector pairs)

### Optimization Strategies

1. **Caching**: Pre-compute terrain matrices
2. **Lazy evaluation**: Calculate LOS only when needed
3. **Spatial partitioning**: Group nearby sectors for LOS
4. **Incremental updates**: Update only changed sectors

### Memory Usage

- **Per sector**: ~200 bytes for physics state
- **Total for 9 sectors**: ~2KB (negligible)
- **LOS cache**: ~81 entries for 9×9 sector matrix

## Debugging Tools

### Physics Visualization

```python
from extensions.military.physics import visualize_sector

# Generate terrain visualization
visualize_sector(sector_id=0, show_terrain=True, show_supply=True)
```

### State Inspection

```python
# Dump complete physics state
from extensions.military.physics import dump_physics_state

state_dict = dump_physics_state(world.physics_states)
print(f"Total sectors: {len(state_dict)}")
print(f"Weather: {state_dict['weather']}")
```

### Supply Network Analysis

```python
from extensions.military.physics import analyze_supply_network

network_stats = analyze_supply_network(world.physics_states, world.supply_routes)
print(f"Network efficiency: {network_stats.efficiency:.2f}")
print(f"Bottlenecks: {network_stats.bottlenecks}")
```

## Extension Points

### Custom Terrain Types

Add new terrain in `physics.py`:

```python
class TerrainType(Enum):
    # ... existing types ...
    JUNGLE = "JUNGLE"      # New terrain type
    TUNDRA = "TUNDRA"      # New terrain type
```

### Weather Events

Create custom weather events:

```python
class WeatherEvent:
    def __init__(self, name, duration, effects):
        self.name = name
        self.duration = duration
        self.effects = effects
        
    def apply(self, weather_state):
        for effect in self.effects:
            effect(weather_state)
```

### Supply Constraints

Add new supply constraints:

```python
class SupplyConstraint:
    def __init__(self, name, check_function):
        self.name = name
        self.check = check_function
        
    def validate(self, route, amount):
        return self.check(route, amount)
```

## Troubleshooting

### Common Issues

1. **Physics not loading**: Check `physics_enabled=True` in wrapper
2. **Terrain type errors**: Verify terrain strings match enum values
3. **Supply not flowing**: Check route definitions and depot sectors
4. **Weather not changing**: Verify temperature_curve in config
5. **LOS always blocked**: Check elevation data and terrain types

### Debug Commands

```python
# Check physics initialization
if hasattr(world, 'physics_states'):
    print(f"Physics enabled: {len(world.physics_states)} sectors")
else:
    print("Physics not enabled")

# Verify terrain mapping
from extensions.military.physics import TerrainType
print(f"Available terrains: {[t.name for t in TerrainType]}")

# Check supply routes
print(f"Supply routes: {len(world.supply_routes)}")
for route in world.supply_routes:
    print(f"  {route.from_sector} -> {route.to_sector}: {route.capacity}")
```

## Best Practices

1. **Balance realism vs performance**: Use appropriate update frequencies
2. **Document custom terrain**: Clearly explain effects of new terrain types
3. **Test edge cases**: Verify extreme weather conditions
4. **Validate supply networks**: Check for disconnected sectors
5. **Profile LOS calculations**: Monitor performance in large scenarios

## Future Enhancements

1. **Dynamic weather**: More sophisticated weather patterns
2. **Seasonal campaigns**: Multi-season scenario support
3. **Advanced logistics**: Multi-modal transport (air, sea, rail)
4. **Environmental effects**: Pollution, radiation, disease
5. **Civilian infrastructure**: Cities, roads, bridges with detailed modeling
