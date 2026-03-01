# Battle of Stalingrad Scenario

The flagship multi-agent scenario simulating the decisive Eastern Front battle (August 23, 1942 – February 2, 1943). Two RecurrentPPO agents control Axis and Soviet forces across 9 operational sectors, competing for territorial stability while managing resources, military strength, and alliance diplomacy.

## Table of Contents

- [Historical Context](#historical-context)
- [Sector Map](#sector-map)
- [Cluster State Variables](#cluster-state-variables)
- [Action Space](#action-space)
- [Reward Design](#reward-design)
- [Special Mechanics](#special-mechanics)
- [Shock Events](#shock-events)
- [Training Pipeline](#training-pipeline)
- [Training Results](#training-results)
- [Running the Scenario](#running-the-scenario)

## Historical Context

The Battle of Stalingrad was the largest and bloodiest battle in the history of warfare. Key phases:

1. **German Advance** (Aug–Sep 1942): 6th Army (Paulus) pushes into the city.
2. **Urban Combat** (Oct–Nov 1942): Brutal fighting in factories and ruins.
3. **Operation Uranus** (Nov 1942): Soviet pincer encircles the 6th Army.
4. **Operation Wintergewitter** (Dec 1942): Hoth's 4th Panzer Army attempts relief.
5. **Operation Little Saturn** (Dec 1942): Soviet counter-offensive cuts off relief.
6. **Surrender** (Feb 1943): Paulus surrenders the remnants of the 6th Army.

The scenario captures these dynamics through sector stability, shock events, and conditional reinforcement mechanics.

## Sector Map

| ID | Sector Name | Icon | Controller | Initial σ | Description |
| -- | ----------- | ---- | ---------- | --------- | ----------- |
| 0 | Stalingrad City Center | 🏙️ | Axis | 0.15 | Urban ruins, extreme hazard, main combat zone |
| 1 | Tractor Factory District | 🏭 | Axis | 0.20 | Industrial zone, axis of main German attack |
| 2 | Mamayev Kurgan | ⛰️ | Contested | 0.15 | Strategic hilltop dominating the city |
| 3 | Volga Crossing | 🚢 | Soviet | 0.50 | Soviet lifeline — key for reinforcement plugin |
| 4 | Northern Don River Line | 🌊 | Axis | 0.40 | German flank, defensive arc |
| 5 | Axis Supply Corridor | 📦 | Axis | 0.50 | Overextended logistics, exhaustion sink |
| 6 | Soviet Strategic Reserve | ⭐ | Soviet | 0.50 | Operation Uranus buildup zone |
| 7 | Romanian/Italian Sector | 🇷🇴 | Axis | 0.40 | Weak southern flank (historically collapsed first) |
| 8 | Wintergewitter Corridor | 🛡️ | Axis | 0.45 | Hoth's 4th Panzer Army relief push from Kotelnikovo |

### Controller Assignments

- **Axis clusters**: [0, 1, 2, 4, 5, 8] — 6 sectors
- **Soviet clusters**: [3, 6] — 2 sectors (but heavily reinforced)
- **Contested**: [2] — Mamayev Kurgan (both sides can influence)

## Cluster State Variables

Each sector has 6 continuous state variables in [0, 1]:

| Variable | Symbol | Description |
| -------- | ------ | ----------- |
| Stability | σ (sigma) | Overall sector control and order. 0 = collapsed, 1 = fully secure |
| Hazard | h | Active combat/threat level. High hazard degrades stability |
| Resources | r | Supply, ammunition, food. Consumed by military operations |
| Military | m | Troop strength and combat capability |
| Trust | τ (tau) | Civilian/institutional trust in controlling faction |
| Polarization | p | Internal faction tensions. High polarization reduces stability |

### Stability Status Thresholds

| Status | σ Range | Meaning |
| ------ | ------- | ------- |
| FORTIFIED | ≥ 0.80 | Strongly held, low risk |
| HOLDING | 0.70–0.80 | Stable but under pressure |
| CONTESTED | 0.60–0.70 | Active fighting, could go either way |
| WAVERING | 0.40–0.60 | Losing ground, needs reinforcement |
| CRUMBLING | 0.25–0.40 | Near collapse, critical |
| COLLAPSING | < 0.25 | Imminent loss of sector |

## Action Space

Each agent chooses from 7 action stances per controlled cluster:

| ID | Stance | Icon | Effect |
| -- | ------ | ---- | ------ |
| 0 | MILITARIZE | ⚔️ | Increase military, consume resources |
| 1 | REFORM | 📜 | Boost trust, reduce polarization |
| 2 | INVEST | 💰 | Improve resources and economic output |
| 3 | PROPAGANDA | 📢 | Influence media bias, boost trust short-term |
| 4 | DIPLOMACY | 🤝 | Strengthen alliances between clusters |
| 5 | DECENTRALIZE | 🔀 | Reduce central control, improve local stability |
| 6 | NOOP | ⏸️ | Do nothing (conserve resources) |

The action space is `Discrete(7 * N_own_clusters)`, decoded into a target cluster + stance.

## Reward Design

Rewards are computed per-side based on:

- **Own stability**: Mean σ across own sectors (positive).
- **Enemy pressure**: Mean hazard on enemy sectors (positive for attacker).
- **Contested territory**: Bonus for controlling Mamayev Kurgan.
- **Exhaustion penalty**: Accumulated wear from sustained operations.
- **Improvement delta**: Change in σ from previous step.

Soviet agents receive approximately 2–3× higher per-step rewards due to fewer but more stable sectors, incentivizing the defensive strategy historically adopted.

## Special Mechanics

### Volga Reinforcements (Plugin: `soviet_reinforcements`)

Every 50 turns, if Volga Crossing (Sector 3) meets conditions:
- Stability σ > 0.5 (secure crossing)
- Hazard h < 0.7 (not under concentrated fire)

Then Soviet Strategic Reserve (Sector 6) receives:
- +10% military strength (capped at 0.95)
- +5% resources

This simulates the historical nightly barge crossings that sustained Soviet resistance. The Soviet agent learns to prioritize defending the Volga Crossing to maintain its reinforcement pipeline.

### Axis Airlift (Plugin: `axis_airlift`)

Every 40 turns, if Axis Supply Corridor (Sector 5) meets conditions:
- Stability σ > 0.3 (not completely cut off)
- Hazard h < 0.8 (not under extreme fire)

Then encircled Axis sectors [0, 1, 2] receive diminishing resource boosts:
- Starting at +4% resources per sector per delivery
- Decaying by 0.5% per delivery (minimum 0.5%)

This simulates the historically inadequate Luftwaffe airlift that delivered only a fraction of the needed 300 tons/day.

### Alliance Diplomacy

The DIPLOMACY stance allows agents to strengthen inter-cluster alliances. Allied clusters share stability benefits and coordinate defense. The alliance matrix decays over time, requiring active maintenance.

## Shock Events

Four historical shock events are defined in `stalingrad.yaml`:

| Shock | Trigger Turn | Affected Clusters | Effect |
| ----- | ------------ | ----------------- | ------ |
| Operation Wintergewitter | 200+ | 8 (Wintergewitter) | Military surge +30%, hazard spike |
| Operation Little Saturn | 250+ | 7, 8 (Romanian, Wintergewitter) | Soviet counter, stability drop |
| Panzer Spearhead | 150+ | 0, 1 (City, Factory) | Temporary Axis military boost |
| Soviet Guards Block | 200+ | 8 (Wintergewitter) | Defensive counter, hazard increase |

Shocks are stochastic — they sample from a Hawkes process with self-excitation, meaning combat cascades lead to more frequent shocks.

## Training Pipeline

### Self-Play Training

```bash
python tests/train_stalingrad_selfplay.py \
    --total-rounds 6 \
    --steps-per-round 25000 \
    --n-envs 4 \
    --log-dir logs/stalingrad_selfplay_volga
```

Each round:
1. Train Axis agent against current Soviet agent (freeze Soviet).
2. Train Soviet agent against updated Axis agent (freeze Axis).
3. Save checkpoints.

### Evaluation

```bash
python tests/eval_stalingrad_battle.py \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip \
    --n-episodes 30
```

### Detailed Replay

```bash
python tests/replay_stalingrad_battle.py \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip \
    --seed 42 --interval 25
```

Or via the CLI:

```bash
python cli.py replay \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip
```

## Training Results

### Phase 1: Base (8 clusters, no Wintergewitter)

- 3 rounds × 25K steps self-play
- **Soviet dominated**: σ̄=0.953 vs Axis σ̄=0.670
- Soviet won 30/30 battles

### Phase 2: + Operation Wintergewitter (9 clusters)

- 6 rounds × 25K steps self-play
- **Axis recovered**: σ̄=0.980, Soviet σ̄=0.949
- Axis 5W / Soviet 1W / 24 Draws

### Phase 3: + Volga Reinforcements

- 6 more rounds × 25K steps self-play
- **Balanced fight**: Axis σ̄=0.853, Soviet σ̄=0.911
- Axis 3W / Soviet **18W** / 9 Draws (**60% Soviet win rate**)
- Soviet learned to defend Volga Crossing (σ=0.906) for reinforcements
- Soviet Reserve reached military=0.94 (Full Strength) from barge deliveries

### Emergent Behaviors

- **Axis**: Alternates between MILITARIZE and INVEST to recover collapsing sectors.
- **Soviet**: Prioritizes INVEST and REFORM to maintain Volga stability.
- **Romanian Sector**: Consistently the weakest Axis link (σ≈0.665), matching history.
- **Mamayev Kurgan**: Most contested sector with largest σ swings (0.15 → 1.0).

## Running the Scenario

### Quick Run (Random Agents)

```bash
python cli.py run stalingrad --episodes 5 --plugins soviet_reinforcements axis_airlift
```

### With Trained Agents

```bash
python cli.py run stalingrad --episodes 30 \
    --axis-model logs/stalingrad_selfplay_volga/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay_volga/soviet_final.zip \
    --plugins soviet_reinforcements axis_airlift
```

### From Config

```bash
python cli.py run --config configs/custom.yaml --episodes 30
```

## Key Files

| File | Purpose |
| ---- | ------- |
| `gravitas_engine/agents/stalingrad_ma.py` | Multi-agent environment |
| `gravitas/scenarios/stalingrad.yaml` | Scenario definition (9 sectors) |
| `gravitas/plugins/soviet_reinforcements.py` | Volga reinforcement plugin |
| `gravitas/plugins/axis_airlift.py` | Axis airlift plugin |
| `tests/train_stalingrad_selfplay.py` | Self-play training script |
| `tests/eval_stalingrad_battle.py` | Batch evaluation script |
| `tests/replay_stalingrad_battle.py` | Detailed replay with commentary |
| `configs/custom.yaml` | Unified config with both plugins |
