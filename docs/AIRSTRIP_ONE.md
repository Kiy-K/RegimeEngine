# Air Strip One — 1984 Scenario

> *War is Peace. Freedom is Slavery. Ignorance is Strength.*

## Overview

Strategic war simulation set in George Orwell's *Nineteen Eighty-Four*. Three AI players — **Oceania** (Ingsoc / Big Brother), **Eurasia** (Neo-Bolshevism / Marshal Kalinin), and **Winston Smith** (British Liberation Front) — compete across a **32-sector map** covering all of the British Isles and Central France. Each turn represents **1 week** of in-game time, starting January 1984.

A fourth LLM serves as a war correspondent providing Edward R. Murrow-style narrative dispatches.

The war is not meant to be won — it is meant to be continuous. Both superstates grind each other down in perpetual conflict that justifies totalitarian control. But Winston Smith has other plans.

## Theatre Map — 32 Sectors, 6 Sea Zones

```
                         OCEANIA — All British Isles (18 sectors)
    ┌────────────────────────────────────────────────────────────────────┐
    │  [14] Edinburgh ═══ [15] Glasgow                                  │
    │        │    Forth          │   Clyde shipyards                     │
    │  [12] Leeds ── [11] Liverpool ──── [17] Belfast ── [16] Dublin    │
    │    │              │     convoys        │ H&W ships    │ Atl Fleet │
    │  [10] Manchester  │                    └─────────────-┘           │
    │    │     [13] Norwich (Bomber Cmd)                                │
    │  [9] Birmingham ──────── [0] LONDON ── [4] Canterbury            │
    │    │ tanks            │        │    \        │                    │
    │  [8] Cardiff    [3] Southampton  [1] DOVER ── [5] Brighton       │
    │    coal+steel     │        │         ║           ║               │
    │  [6] Bristol ── [7] Plymouth      ═══╬═══════════╬═══════════    │
    ╞═══════════════════════╬════ CHANNEL ══╬═══════════╬══════════════╡
    │                    ║    ║           ║         ║                   │
    │  [21] Cherbourg ─ [20] Le_Havre  [19] Dunkirk ── [18] CALAIS    │
    │    sub base      Normandy   │      fleet base      staging       │
    │        │                [23] Rouen ── [22] Amiens ── [24] Lille  │
    │  [30] BREST            steel mills    rail hub        │          │
    │   sub pens                    │                  [25] Brussels    │
    │        │              [27] PARIS ────────────── [26] Antwerp     │
    │  [31] Bordeaux         HQ    │                  N Sea fleet      │
    │                        [28] Orleans                              │
    │                              │                                   │
    │                        [29] Lyon                                 │
    │                        industry                                  │
    └────────────────────────────────────────────────────────────────────┘
                    EURASIA — France + Benelux (14 sectors)
```

### Sea Zones (6)

| Zone | Name | Width | Key Threat |
|------|------|-------|------------|
| 0 | Dover Strait | 33km | PRIMARY — shortest crossing |
| 1 | Western Channel | 120km | Thames Estuary + Normandy route |
| 2 | North Sea | 500km | Flanking via Antwerp |
| 3 | Irish Sea | 300km | Wolf packs starve Ireland |
| 4 | Bay of Biscay | 600km | Brest submarine pens → Plymouth |
| 5 | North Atlantic | Open | Convoy lifeline |

## Oceania Sectors (18)

| ID | City | Pop (1958) | Terrain | Role |
|----|------|-----------|---------|------|
| 0 | **London** | 7,500K | Urban | Ingsoc HQ. Ministries. 85% proles. Rocket bomb target. |
| 1 | **Dover** | 25K | Urban | Channel fortress. Radar. AA batteries. |
| 2 | **Portsmouth** | 247K | Urban | Home Fleet HQ. Royal Navy shipyards. |
| 3 | **Southampton** | 180K | Urban | Factory district. Supply port. |
| 4 | **Canterbury** | 30K | Open | RAF 11 Group. Training camps. |
| 5 | **Brighton** | 150K | Open | Secondary coastal defense. |
| 6 | **Bristol** | 420K | Urban | Aircraft factories. Western hub. |
| 7 | **Plymouth** | 113K | Urban | Western Approaches naval base. |
| 8 | **Cardiff** | 250K | Urban | Welsh coal + steel. Heavy industry. |
| 9 | **Birmingham** | 970K | Urban | Tank production. Midlands industry. |
| 10 | **Manchester** | 640K | Urban | Munitions. Textiles → uniforms. |
| 11 | **Liverpool** | 700K | Urban | Atlantic convoy port. |
| 12 | **Leeds** | 480K | Urban | Northern industry + rail junction. |
| 13 | **Norwich** | 130K | Open | RAF Bomber Command. Agricultural. |
| 14 | **Edinburgh** | 470K | Urban | Forth anchorage. Scottish Command. |
| 15 | **Glasgow** | 900K | Urban | Clyde shipyards. Heavy industry. |
| 16 | **Dublin** | 650K | Urban | Airstrip Two capital. Atlantic Fleet. |
| 17 | **Belfast** | 440K | Urban | Harland & Wolff shipyards. |

## Eurasia Sectors (14)

| ID | City | Pop (1958) | Terrain | Role |
|----|------|-----------|---------|------|
| 18 | **Calais** | 75K | Open | Channel invasion staging. Artillery. |
| 19 | **Dunkirk** | 25K | Urban | Channel Fleet base. |
| 20 | **Le Havre** | 140K | Urban | Normandy port (rebuilt post-WWII). |
| 21 | **Cherbourg** | 35K | Urban | Submarine base. Atlantic approach. |
| 22 | **Amiens** | 110K | Plains | Picardy rail junction. Supply depot. |
| 23 | **Rouen** | 115K | Urban | Seine steel mills. Industrial heart. |
| 24 | **Lille** | 195K | Plains | Nord coal. Training. Reserves. |
| 25 | **Brussels** | 1,100K | Urban | Benelux Command HQ. |
| 26 | **Antwerp** | 260K | Urban | Major port. North Sea fleet. |
| 27 | **Paris** | 5,000K | Urban | Western Front Command HQ. |
| 28 | **Orleans** | 85K | Plains | Loire logistics hub. |
| 29 | **Lyon** | 530K | Urban | Rhône-Alpes industry. Southern reserves. |
| 30 | **Brest** | 120K | Urban | Finistère submarine pens. Wolf packs. |
| 31 | **Bordeaux** | 250K | Plains | Southern reserves. |

Population data from **1958 census** (interpolated 1951/1961 UK, 1954 French Census).

## Three Factions

### Oceania (Ingsoc) — The Defender
- **Regime**: Totalitarian (bureaucracy speed ×1.15)
- **Conscription**: Total Mobilisation
- **Base corruption**: 8% (floor 3.2%)
- **Population**: ~14.3M across 18 sectors
- **Strengths**: Naval supremacy, radar chain, industrial depth (Birmingham/Glasgow/Cardiff)
- **Weaknesses**: Prole unrest (85% of population), food dependency, 6 sea zones to defend

### Eurasia (Neo-Bolshevism) — The Attacker
- **Regime**: Communist (bureaucracy speed ×0.85 — committee approvals)
- **Conscription**: General Mobilisation
- **Base corruption**: 12% (floor 4.8%)
- **Population**: ~7.9M across 14 sectors
- **Strengths**: Brest submarine pens, 5 invasion axes, strategic depth (Paris/Lyon/Orleans)
- **Weaknesses**: Smaller population, Channel crossing, higher corruption

### Winston Smith / BLF — The Insurgent
- **Type**: Underground resistance
- **Operates in**: All 18 Oceania cities
- **Goal**: Full revolution — seize London, topple Big Brother
- **Strengths**: Exploits Oceania's weaknesses (food shortages, bombing damage, low welfare)
- **Weaknesses**: Thought Police, detection heat, limited arms

## 12 Integrated Systems

See [AIRSTRIP_ONE_SYSTEMS.md](AIRSTRIP_ONE_SYSTEMS.md) for full technical documentation of all systems:

1. **Economy** — 10 factory types, GDP, power dependency
2. **Population** — Real numbers (1958 census), 1984 social classes, 8 job types
3. **Research** — 6 branches × 5 tiers, cross-branch prerequisites
4. **Governance** — 7 budget categories, corruption (asymptotic), bureaucracy delays
5. **Naval** — 14 ship classes, 6 sea zones, 3 invasion types
6. **Air Force** — 10 aircraft types, 7 bases per side
7. **BLF Resistance** — 7 escalation levels (DORMANT → BETRAYED_REVOLUTION)
8. **Intelligence** — Fog of war, spy rings, code-breaking, radar
9. **Weather** — Maritime climate, seasonal model, affects all operations
10. **Manpower** — Conscription laws, training pipeline
11. **War Economy** — Legacy resource model (runs alongside new Economy)
12. **War Correspondent** — Murrow/Pyle narrative dispatches

## Running the Simulation

```bash
# 100-week game with all-Claude players
python tests/benchmark_llm.py \
    --oceania-model claude-haiku-4-5-20251001 \
    --eurasia-model claude-haiku-4-5-20251001 \
    --winston-model claude-haiku-4-5-20251001 \
    --commentary-model mistral-medium-latest \
    --turns 100 --seed 451
```

## Strategic Dynamics

### The Channel Problem (6 Sea Zones)
The English Channel is no longer a simple 2-crossing barrier. With 6 sea zones, Eurasia has **5 invasion axes**:
1. **Dover Strait** (33km) — shortest, heaviest defense
2. **Western Channel / Normandy** (120km) — Le Havre/Cherbourg → Brighton/Portsmouth
3. **North Sea** (500km) — Antwerp → Liverpool/Edinburgh (flanking)
4. **Bay of Biscay** (600km) — Brest → Plymouth (unexpected western approach)
5. **Airborne** — paratroop drop on Canterbury/Norwich (no ships needed)

Oceania must defend ALL 6 zones simultaneously.

### The Three-Way War
Unlike a simple two-player game, Winston Smith adds a critical third dimension:
- **Oceania** must fight Eurasia externally AND suppress BLF internally
- **Eurasia** can exploit the BLF as "useful idiots" — then betray them
- **Winston** exploits every Oceania weakness: food shortages, bombing damage, low welfare

### Perpetual War Design
True to Orwell's vision, the scenario is balanced for extended conflict:
- Oceania has defensive advantage but 6 sea zones to defend + prole unrest
- Eurasia has industrial depth + submarine warfare but must cross open water
- The BLF can escalate to FULL REVOLUTION if Oceania neglects welfare
- Budget tradeoffs force hard choices: military vs welfare vs research

### The Prole Factor
London's 7.5M population is 85% proles. They are politically inert — UNTIL:
- Food runs out (welfare budget < 10%)
- Bombing destroys their homes
- Winston's broadcasts reach them via pirated telescreens
- The Party's trust collapses below 20%

When proles revolt, it's catastrophic: 50% of industry paralyzed, 50% of military diverted to London.
