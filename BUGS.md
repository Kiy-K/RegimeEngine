# GRAVITAS Engine — Bug Report

## Overview
Critical bugs affecting gameplay and user experience identified from benchmark `benchmark_1949_1773580507.json` (100-turn Air Strip One 1984 simulation).

---

## 🔴 Critical Gameplay Bugs

### 1. Invasion Planning Deadlock
**Priority**: CRITICAL  
**Description**: Invasion plans get stuck in perpetual "PLANNING" state, never executing  
**Evidence**: 
```
"Invasion to Liverpool already in progress (#6, PLANNING). DO NOT re-plan."
"Invasion to Canterbury already in progress (#3, PLANNING). DO NOT re-plan."
```
**Impact**: 
- Eurasia unable to execute primary victory condition
- Multiple invasion attempts blocked indefinitely
- Game balance heavily favors defender

**Affected Systems**: `extensions/naval/naval_invasion.py`, invasion state machine  
**Fix Required**: Add timeout mechanism, force execution after X turns, or cancel stuck plans

---

### 2. Territory Validation Error
**Priority**: HIGH  
**Description**: System allows invading own territory  
**Evidence**:
```
"Cannot invade Dublin — it is YOUR territory."
```
**Impact**:
- Wasted action slots in critical game phases
- AI makes nonsensical decisions
- Player frustration with invalid actions

**Affected Systems**: Action validation, territory ownership checking  
**Fix Required**: Pre-action validation to prevent invalid territory selection

---

### 3. Corruption Purge Ineffectiveness
**Priority**: HIGH  
**Description**: Anti-corruption purges continue despite zero effectiveness  
**Evidence**:
```
"⚠ DIMINISHING RETURNS — invest in WELFARE or INFRASTRUCTURE instead."
```
**Pattern**: 54 purges performed, corruption still rose from 7% to 26.7%  
**Impact**:
- AI wastes actions on ineffective measures
- No hard limit on futile actions
- Economic system becomes unbalanced

**Affected Systems**: `extensions/governance/budget_system.py`  
**Fix Required**: Hard cap on purge attempts, force alternative actions after threshold

---

### 4. BLF Leadership Disconnect
**Priority**: MEDIUM  
**Description**: BLF movement continues growing after Winston Smith's capture  
**Evidence**: Winston captured ~turn 32, BLF grew to 2,304 members by turn 100  
**Impact**:
- Narrative inconsistency
- Reduced strategic impact of eliminating leaders
- Game balance issues

**Affected Systems**: `extensions/resistance/resistance.py`  
**Fix Required**: Reduce BLF effectiveness/growth without leader, add leadership cascade effects

---

## 🟡 System Integration Bugs

### 5. Factory Type Validation
**Priority**: MEDIUM  
**Description**: System accepts non-existent factory types for construction  
**Evidence**:
```
"Unknown factory type: BUREAUCRACY"
```
**Impact**:
- Failed construction projects
- Wasted resources
- Confusing error messages

**Affected Systems**: `extensions/economy_v2/economy_core.py`  
**Fix Required**: Factory type whitelist, pre-validation of construction orders

---

### 6. Research Queue Duplication
**Priority**: LOW  
**Description**: AI attempts duplicate research actions  
**Evidence**:
```
"Already researching Doctrine."
"All 2 research slots full."
```
**Impact**:
- Inefficient AI decision-making
- Wasted turns in early/mid-game

**Affected Systems**: `extensions/research/research_system.py`  
**Fix Required**: Better AI awareness of current research status

---

### 7. Naval Combat Attrition
**Priority**: LOW  
**Description**: Ship counts don't decrease proportionally to combat activity  
**Evidence**: 100 turns of consistent naval battles, final counts 58 vs 41 ships  
**Impact**:
- Unrealistic combat simulation
- Potential balance issues

**Affected Systems**: `extensions/naval/naval_combat.py`  
**Fix Required**: Review combat resolution mechanics, ensure proper attrition

---

## 🟠 Performance/UX Bugs

### 8. API Rate Limiting
**Priority**: MEDIUM  
**Description**: Commentary model hits rate limits causing delays  
**Evidence**:
```
"API retry 1/5 (mistral): API error occurred: Status 429. Service tier capacity exceeded"
```
**Impact**:
- 15+ second delays on some turns
- Disrupted gameplay flow
- Poor user experience

**Affected Systems**: `tests/benchmark_llm.py`, API retry logic  
**Fix Required**: Rate limiting, model fallback, or local commentary generation

---

### 9. Score Display Precision
**Priority**: LOW  
**Description**: Excessive decimal places in score display  
**Evidence**: `32,345.063693144857` instead of `32,345`  
**Impact**:
- Cluttered output
- Hard to read quickly
- Unprofessional appearance

**Affected Systems**: Score formatting in `gravitas/llm_game.py`  
**Fix Required**: Round scores to nearest integer for display

---

### 10. Turn Counter Inconsistency
**Priority**: LOW  
**Description**: Mixed terminology ("TURN 56/100" vs "Week 56")  
**Impact**:
- Minor confusion in reading logs
- Inconsistent user interface

**Affected Systems**: Turn labeling in game output  
**Fix Required**: Standardize on one terminology (recommend "TURN X/100")

---

## 🔵 Design Issues

### 11. Action Validation Gaps
**Priority**: MEDIUM  
**Description**: Insufficient pre-action validation  
**Evidence**: Multiple invalid actions executed (invading own territory, unknown factories)  
**Impact**:
- Wasted turns
- Player frustration
- AI inefficiency

**Fix Required**: Comprehensive action validation before execution

---

### 12. AI Decision Making
**Priority**: MEDIUM  
**Description**: AI repeats ineffective actions  
**Evidence**: 54 corruption purges despite diminishing returns  
**Impact**:
- Poor AI performance
- Predictable gameplay
- Exploitable patterns

**Fix Required**: Action effectiveness tracking, adaptive AI behavior

---

## 📊 Bug Impact Summary

| Severity | Count | Primary Impact |
|----------|-------|----------------|
| Critical | 1 | Game-breaking (invasion deadlock) |
| High | 2 | Major gameplay issues |
| Medium | 5 | Significant UX/balance problems |
| Low | 4 | Minor annoyances |

**Total Bugs Identified**: 12

---

## 🔧 Recommended Fix Priority

### Phase 1 (Critical - Fix Immediately)
1. **Invasion Planning Deadlock** - Breaks core gameplay
2. **Territory Validation Error** - Prevents basic actions

### Phase 2 (High Priority - Next Sprint)
3. **Corruption Purge Logic** - Economic balance
4. **API Rate Limiting** - User experience

### Phase 3 (Medium Priority - Following Sprint)
5. **BLF Leadership Logic** - Narrative consistency
6. **Action Validation Framework** - Prevent future issues
7. **Factory Type Validation** - System integrity

### Phase 4 (Low Priority - Polish)
8. **Score Display Formatting** - UI polish
9. **Turn Counter Consistency** - UI standardization
10. **Research Queue Logic** - AI optimization
11. **Naval Combat Attrition** - Simulation accuracy
12. **AI Decision Making** - Long-term improvement

---

## 🧪 Testing Recommendations

### Regression Tests
- Invasion state machine timeout scenarios
- Territory ownership validation
- Corruption purge effectiveness limits
- BLF leadership cascade effects

### Integration Tests
- End-to-end invasion execution
- Multi-faction action validation
- API rate limit handling

### Performance Tests
- 100+ turn simulations
- Concurrent API calls
- Memory usage monitoring

---

## 📝 Notes

- All bugs identified from single 100-turn simulation
- System showed good error recovery (0 LLM failures)
- Game completed successfully despite bugs
- Some bugs may be design features rather than errors (verify with team)

**Report Generated**: 2026-03-15  
**Simulation**: benchmark_1949_1773580507.json  
**Game Version**: Air Strip One 1984 (35-sector, 13-system)
