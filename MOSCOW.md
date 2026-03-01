# Battle of Moscow (1941): Partisan Warfare & Economic Logistics

**Author**: Bibbybybot
**Date**: 2026-03-01
**Status**: Proposal
**Scenario Type**: Military + Economic + Asymmetric Warfare

---

## Overview
The **Battle of Moscow (October 1941 – January 1942)** introduces **economic logistics** and **partisan warfare** to Gravitas Engine. Unlike Stalingrad (pure military attrition), Moscow tests:
- **Resource production/consumption** (factories, rail hubs).
- **Partisan units** (sabotage, recruitment, hit-and-run tactics).
- **Axis supply chain vulnerabilities** (rail sabotage, fuel shortages).

This scenario **extends Stalingrad’s military layer** with economic mechanics, setting the stage for Cold War/Modernday hybrid warfare.

---

## Key Differences from Stalingrad
| **Feature**               | **Stalingrad**                          | **Moscow**                                  | **Purpose**                          |
|---------------------------|------------------------------------------|--------------------------------------------|------------------------------------------|
| **Core Focus**            | Military attrition                      | **Economy + logistics**                     | Test resource flow.                     |
| **Agents**               | Axis, Soviets                           | **Axis, Soviets, Partisans**               | Asymmetric warfare.                    |
| **Sectors**              | Urban ruins                             | **Factories, rail hubs, forests**          | Economic targets.                      |
| **Shocks**               | Blizzards, Uranus                       | **Rail sabotage, fuel shortages**         | Logistics disruptions.                 |
| **Win Condition**        | Control sectors                         | **Economic collapse + encirclement**      | Long-term resource management.         |

---

## Design Goals
1. **Prove economic mechanics** work alongside military.
2. **Show partisan units** as a distinct, sabotage-focused force.
3. **Force Axis to split forces** (front lines vs. rear-area security).
4. **Keep Stalingrad’s military layer** intact (reuse code).

---

## Sector Design

### 1. Key Sectors
```yaml
sectors:
  - id: 0
    name: MoscowCityCenter
    side: Soviet
    initial_state:
      sigma: 0.60       # Moderate stability
      resource: 0.80   # High (capital)
      production: 0.05 # Factories
      consumption: 0.03 # Civilian/military use
    logistics_links: [1, 2] # Connected to rail hubs

  - id: 1
    name: YaroslavlRailHub
    side: Soviet
    initial_state:
      sigma: 0.50
      resource: 0.90   # Critical supply node
      production: 0.03  # Rail maintenance
      consumption: 0.01
    logistics_links: [0, 3] # Moscow + northern front

  - id: 4
    name: BryanskForest
    side: Contested
    initial_state:
      sigma: 0.30       # Low stability (remote)
      resource: 0.20    # No factories, but enough for partisans
      military: 0.0     # Ungarrisoned
    logistics_links: [5]  # Targets VyazmaRailHub

  - id: 5
    name: VyazmaRailHub
    side: Axis
    initial_state:
      sigma: 0.70
      resource: 0.90   # Axis supply depot
      production: 0.02  # Limited (occupied)
      consumption: 0.04 # High (Axis troops)
    logistics_links: [0, 4] # Moscow + Bryansk
```

### 2. Logistics Links
- **Purpose**: Model rail/supply routes.
- **Mechanics**:
  - If a link is **sabotaged**, sectors lose `resource`.
  - Axis must **garrison rail hubs** (Sector 5) to protect links.

---

## Agents

### 1. Partisan Agent
**Role**: Sabotage Axis logistics, recruit guerrilla units.
**Constraints**:
- **No heavy arms**: `military: 0.0` (can’t hold sectors).
- **Sabotage-focused**: Actions target `resource`/`logistics_links`.
- **Recruitment cost**: Deducts from **Soviet Reserves (Sector 6)**.

#### **Actions**
| **Action**          | **Effect**                                  | **Cost**               |
|----------------------|--------------------------------------------|------------------------|
| `recruit`           | Spawn a partisan unit in a sector.         | 0.15 resource from Sector 6 |
| `sabotage`          | Reduce `resource` in a linked sector.      | None                   |
| `ambush`            | Reduce Axis `military` in a sector.        | None                   |
| `propaganda`        | Increase Soviet `trust` in a sector.       | None                   |
| `hide`              | Relocate to a new sector.                  | None                   |

#### **Code: `gravitas/plugins/partisan_agent.py`**
```python
from gravitas.world import GravitasWorld
from typing import List
import random

class PartisanUnit:
    def __init__(self, sector_id: int):
        self.sector_id = sector_id
        self.sabotage_power = 0.3  # 30% resource damage
        self.stealth = 0.9         # 90% chance to avoid detection

    def sabotage(self, world: GravitasWorld):
        sector = world.clusters[self.sector_id]
        if "logistics_links" in sector:
            for linked_sector_id in sector.logistics_links:
                linked_sector = world.clusters[linked_sector_id]
                linked_sector.resource = max(0.0, linked_sector.resource - self.sabotage_power)
            print(f"Partisans sabotaged logistics in Sector {self.sector_id}!")

    def ambush(self, world: GravitasWorld):
        sector = world.clusters[self.sector_id]
        if sector.side == "Axis":
            sector.military = max(0.0, sector.military - 0.15)
            print(f"Partisan ambush in Sector {self.sector_id}!")

class PartisanAgent:
    def __init__(self, world: GravitasWorld):
        self.world = world
        self.units: List[PartisanUnit] = []
        self.recruitment_cost = 0.15
        self.soviet_reserve_sector = 6  # SovietStrategicReserve

    def can_recruit(self, sector_id: int) -> bool:
        sector = self.world.clusters[sector_id]
        soviet_reserve = self.world.clusters[self.soviet_reserve_sector]
        return (
            sector.side in ["Contested", "Soviet"]
            and soviet_reserve.resource >= self.recruitment_cost
            and sector.sigma > 0.4
        )

    def recruit(self, sector_id: int) -> bool:
        if not self.can_recruit(sector_id):
            return False
        self.world.clusters[self.soviet_reserve_sector].resource -= self.recruitment_cost
        self.units.append(PartisanUnit(sector_id))
        print(f"Recruited partisans in Sector {sector_id}! Cost: {self.recruitment_cost} resource.")
        return True

    def deploy(self):
        for unit in self.units:
            unit.sabotage(self.world)
            # Move to a new sector (hit-and-run)
            possible_destinations = [
                s.id for s in self.world.clusters
                if s.side in ["Contested", "Soviet"] and s.id != unit.sector_id
            ]
            if possible_destinations:
                unit.sector_id = random.choice(possible_destinations)

    def on_step(self, turn: int):
        if turn % 3 == 0:  # Act every 3 turns
            self.deploy()
```

---

### 2. Axis Agent
**Adjustments for Moscow**:
- **Must garrison rail hubs** (Sector 5) to protect logistics.
- **Fuel shortages** if partisans sabotage too many links.

```yaml
agents:
  - name: GermanArmyGroupCenter
    side: Axis
    controlled_clusters: [5, 7, 8]  # Rail hubs + front lines
    primary_objective: [hold_rail_hubs, advance_on_moscow]
    diplomacy_bias: suppress_partisans  # Uses DIPLOMACY to coordinate anti-partisan sweeps
```

### 3. Soviet Agent
**Adjustments for Moscow**:
- **Manages partisan recruitment** (resource tradeoff).
- **Defends factories** (Sector 0, 1).

```yaml
agents:
  - name: SovietWesternFront
    side: Soviet
    controlled_clusters: [0, 1, 6]  # Moscow, rail hub, reserves
    primary_objective: [defend_factories, support_partisans]
    diplomacy_bias: coordinate_with_partisans
```

---

## Economic Mechanics

### 1. Resource Flow
- **Production**: Sectors generate `resource` if stable (`sigma > 0.5`).
- **Consumption**: Troops/movement burn `resource`.
- **Logistics**: Resources flow along `logistics_links`.

**Example**:
```python
# In GravitasWorld.update()
for sector in self.clusters:
    # Production: +resource if stable
    if sector.sigma > 0.5:
        sector.resource = min(1.0, sector.resource + sector.production)
    
    # Consumption: -resource from military/exhaustion
    sector.resource = max(0.0, sector.resource - sector.consumption * sector.military)
    
    # Logistics: Share resources with linked sectors
    for linked_id in sector.get("logistics_links", []):
        linked_sector = self.clusters[linked_id]
        transfer = min(0.1, sector.resource * 0.2)  # Share up to 10% of resource
        sector.resource -= transfer
        linked_sector.resource += transfer
```

### 2. Shocks
```yaml
custom_shocks:
  - name: RailSabotage
    probability: 0.05
    effect_type: cluster
    target_cluster: [5]  # VyazmaRailHub
    params_delta:
      resource: -0.30
      hazard: +0.20
    description: "Partisans blow up a train!"

  - name: FuelShortage
    probability: 0.03
    effect_type: global
    target_side: Axis
    params_delta:
      military_efficiency: -0.15  # Troops move slower
    description: "Axis vehicles run out of fuel!"

  - name: FactoryEvacuation
    probability: 0.02
    effect_type: cluster
    target_cluster: [0]  # MoscowCityCenter
    params_delta:
      production: -0.03
      resource: -0.20
    description: "Soviets relocate factories east!"
```

---

## Partisan-Soviet Alliance

### 1. Alliance Rules
```yaml
initial_alliances:
  # Partisans (Sector 4) are strongly allied with Soviets
  - {from: 4, to: 0, value: 0.8}   # BryanskForest ↔ Moscow
  - {from: 4, to: 6, value: 0.7}   # BryanskForest ↔ SovietReserve

  # Axis detects partisans if they act openly
  - {from: 4, to: 5, value: -0.5}  # BryanskForest ↔ VyazmaRailHub (Axis-controlled)
```

### 2. Recruitment Rules
- **Cost**: 0.15 resource from **Soviet Reserves (Sector 6)**.
- **Requirements**:
  - Sector is **Contested/Soviet**.
  - Sector `sigma > 0.4` (minimum stability).
  - Soviet `alliance > 0.7` with partisans.

**Example**:
```python
if agent.can_recruit(sector_id=4):  # BryanskForest
    agent.recruit(4)
    # Soviet Reserves (Sector 6) loses 0.15 resource
    # PartisanUnit spawned in Sector 4
```

---

## Expected Gameplay

| **Metric**               | **Without Partisans** | **With Partisans**       | **Why?**                                  |
|--------------------------|-----------------------|--------------------------|-------------------------------------------|
| **Axis resource loss**   | 5%/episode            | **25–40%/episode**       | Rail sabotage + ambushes.                  |
| **Soviet resource drain**| 0%                    | **~10%/episode**         | Recruitment cost.                          |
| **Axis garrison shifts** | Front lines            | **Rear areas**            | Must protect logistics.                   |
| **Soviet wins**           | ~50%                   | **~70%**                  | Partisans tip the balance.                |
| **Draws**                | ~30%                   | **<10%**                  | Partisans force decisions.                |

---

## Testing Plan

1. **Implement**:
   - Add `PartisanAgent` and `PartisanUnit` to `gravitas/plugins/`.
   - Update `moscow.yaml` with sectors/alliances.

2. **Test**:
   ```bash
   python cli.py run moscow --episodes 10
   ```
   - **Goal**: Axis should lose **~70% of episodes** if partisans are active.
   - **Check**:
     - Soviet Reserves (Sector 6) `resource` decreases with recruitment.
     - Axis `resource` drops in rail hubs (Sector 5).

3. **Iterate**:
   - Adjust `recruitment_cost` (e.g., 0.10 → more partisans, but higher cost).
   - Add **detection mechanics** (Axis can hunt partisans if `hazard` is high).

---

## Files to Create

1. **Scenario Config**:
   - `gravitas/scenarios/moscow.yaml` (sectors, agents, shocks).

2. **Plugins**:
   - `gravitas/plugins/partisan_agent.py` (recruitment + sabotage).
   - `gravitas/plugins/economy.py` (resource flow).

3. **Documentation**:
   - `docs/moscow_design.md` (this file!).

---

## Backward Compatibility
- Reuses **Stalingrad’s military layer** (sectors, agents, shocks).
- Adds **economy/logistics** as plugins (no core changes).

---

## Next Steps

1. **Approve this design** (or suggest tweaks).
2. **Implement**:
   - Create `moscow.yaml` and plugins.
   - Test with 10 episodes.
3. **Expand**:
   - Add **detection mechanics** (Axis anti-partisan sweeps).
   - Add **propaganda effects** (partisans boost Soviet `trust`).

---

**References**:
- [[Stalingrad Scenario]] (Military layer baseline)
- [[Partisan Warfare Tactics]] (Historical inspiration)
- [[Gravitas Plugin System]] (Technical implementation)