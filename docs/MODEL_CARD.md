# Model Card: BBO Optimisation Approach

## Overview

| Field       | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| **Name**    | Adaptive Bayesian Black-Box Optimisation Strategy                   |
| **Type**    | Sequential model-based optimisation with Gaussian Process surrogate |
| **Version** | 2.0 (Final, Round 13)                                               |
| **Author**  | Imperial College ML & AI Programme — Capstone Project               |
| **Date**    | Modules 12-25 (13-week iteration cycle)                             |

---

## Intended Use

### What tasks is this approach suitable for?

- **Expensive function evaluation**: When each query has significant cost (time, money, resources)
- **Low-dimensional optimisation**: Functions with 2-8 input dimensions
- **Smooth, continuous functions**: Where local structure predicts global behaviour
- **Sequential decision-making**: When observations inform subsequent queries
- **Hyperparameter tuning**: Optimising ML model configurations
- **Experimental design**: Drug discovery, manufacturing parameter tuning

### What use cases should be avoided?

- **High-dimensional spaces (>10D)**: Curse of dimensionality limits effectiveness
- **Discrete/combinatorial inputs**: Strategy assumes continuous search space
- **Highly multimodal functions**: May converge to local optima
- **Noisy functions without replication**: Cannot distinguish signal from noise
- **Real-time optimisation**: Weekly iteration cycle inappropriate for latency-sensitive applications
- **Functions with discontinuities**: Smoothness assumption will fail

---

## Details: Strategy Evolution Across Thirteen Rounds

### Phase 1: Exploration (Rounds 1-3)

**Approach**: Boundary testing, corner sampling, initial exploitation near seed data's best points.

**Key decisions**:
- F1: Tested corners (0,1) and (1,0) — both yielded zero; center (0.5, 0.5) found first signal
- F2, F3: Explored edges based on GP uncertainty estimates
- F4-F8: Exploited promising regions from initial data

**Lessons learned**: Aggressive exploration often failed; exploitation near known good points more reliable.

### Phase 2: Gradient Estimation (Rounds 4-6)

**Approach**: Used output changes between rounds to infer directional gradients.

**Key decisions**:
- F4: Discovered sign change from negative to positive — major breakthrough
- F5: Identified X3, X4 as primary drivers; began systematic boundary pushing
- F6: Identified best region around (0.71, 0.13, 0.71, 0.72, 0.03)

**Techniques**: Implicit gradient descent using round-over-round comparisons; step size tuning per function.

### Phase 3: Refinement (Rounds 7-9)

**Approach**: Refined successful regions; implemented "revert-to-best" when refinements failed.

**Key decisions**:
- F1: Diagonal discovery at (0.55, 0.55), then breakthrough at (0.57, 0.57) — 3000x improvement
- F2, F3, F6: Reverted to earlier successful queries after regressions
- F4: Continued gradient following (0.105 → 0.137 → 0.161)
- F5: Pushed X3, X4 toward upper bounds (2993 → 3299 → 3512)

**Techniques**: Conservative step sizes; function-specific strategies; portfolio approach balancing exploration/exploitation.

### Phase 4: Exploitation (Rounds 10-13)

**Approach**: Aggressive pursuit of discovered gradients; securing gains through revert-to-best.

**Key breakthroughs**:
- F1: Diagonal refinement (0.58, 0.58) yielded 8x improvement to 1.40e-4
- F4: Continued climb to 0.181
- F5: Outstanding trajectory (3610 → 3713 → 3821 → 3934) — final +261%
- F7, F8: Small but consistent improvements through fine-tuning

**Final strategy**: 95% exploitation, 5% exploration; revert to proven queries for noisy functions.

---

## Performance

### Results Summary (After Round 13 — FINAL)

| Function | Initial Best | Final Best | Improvement         | Status          |
| -------- | ------------ | ---------- | ------------------- | --------------- |
| F1       | 0            | 1.40e-4    | ∞ (from zero)       | 🚀 Breakthrough |
| F2       | 0.611        | 0.651      | +6.5%               | ⚠️ Noisy        |
| F3       | -0.035       | -0.0275    | +21.4%              | ⚠️ Noisy        |
| F4       | -4.03        | +0.181     | +104% (sign change) | 🚀 Excellent    |
| F5       | 1,089        | 3,934      | **+261%**           | 🚀 Outstanding  |
| F6       | -0.71        | -0.635     | +10.6%              | ⚠️ Noisy        |
| F7       | 1.365        | 1.397      | +2.3%               | ✅ Steady       |
| F8       | 9.60         | 9.792      | +2.0%               | ✅ Plateau      |

### Metrics Used

- **Primary**: Best output value achieved (maximisation objective)
- **Secondary**: Round-over-round improvement rate
- **Diagnostic**: Reversion frequency (indicator of strategy instability)

### Performance by Function Characteristics

| Characteristic           | Functions | Performance     | Notes                       |
| ------------------------ | --------- | --------------- | --------------------------- |
| Clear gradient           | F4, F5    | Excellent       | Consistent improvement      |
| Sparse/narrow optimum    | F1        | Good (after R8) | Required diagonal discovery |
| Noisy/stochastic         | F2, F3, F6| Moderate        | Same query → different outputs |
| High-dimensional plateau | F7, F8    | Limited         | Diminishing returns         |

---

## Assumptions and Limitations

### Key Assumptions

1. **Local smoothness**: Small input changes produce small output changes
   - _Impact_: Enables gradient-following; fails for discontinuous functions
   - _Evidence_: Validated for F4, F5; questionable for F2, F3, F6

2. **Determinism**: Same input yields same output
   - _Impact_: Single queries sufficient; no replication needed
   - _Reality_: F2, F3, F6 violated this — identical queries gave different outputs

3. **Unimodality**: Single global optimum dominates
   - _Impact_: Exploitation-heavy strategy converges to local region
   - _Risk_: May miss superior global optima

4. **Stationarity**: Function behaviour consistent across search space
   - _Impact_: Strategies learned in one region transfer elsewhere
   - _Limitation_: F1's narrow diagonal suggests non-stationary structure

### Constraints

1. **Query budget**: One query per function per week (13 total)
2. **No parallel evaluation**: Sequential queries only
3. **Fixed precision**: Six decimal places for inputs
4. **No derivative information**: Pure black-box setting

### Failure Modes Observed

| Mode                  | Description                      | Affected Functions   |
| --------------------- | -------------------------------- | -------------------- |
| Stochastic noise      | Same query → different outputs   | F2, F3, F6           |
| Premature convergence | Stopped exploring too early      | F8                   |
| Overshoot             | Step size too large              | F1 (R10, R12)        |
| Plateau trap          | Cannot escape local optimum      | F8                   |

---

## Ethical Considerations

### Transparency and Reproducibility

**Documentation provided**:
- Complete query-output history (queries.csv) — all 13 rounds
- Per-round decision rationale
- Strategy evolution narrative
- Failure acknowledgment (not just successes)

**Reproducibility requirements**:
- Initial seed data (from course portal)
- This model card and datasheet
- Jupyter notebooks with analysis code
- Random seeds where applicable

### Real-World Adaptation Considerations

When applying this approach to real problems:

1. **Test for stochasticity early** — replicate queries to detect noise
2. **Validate smoothness assumption** before committing to gradient-based exploitation
3. **Budget for exploration** even when exploitation seems promising
4. **Document failures** as thoroughly as successes
5. **Consider multi-start** to escape local optima

### Limitations of Transparency

- **Implicit decisions**: Some strategic choices made intuitively, not formally documented
- **Hindsight bias**: Post-hoc rationale may overstate intentionality
- **Single trajectory**: Cannot assess strategy variance without comparison runs

---

## Technical Specifications

### Surrogate Model (Conceptual)

```
Gaussian Process with:
- Kernel: RBF (Radial Basis Function)
- Length scale: Adaptive per function
- Noise: Assumed low (validated for F4, F5; invalid for F2, F3, F6)
```

### Acquisition Strategy

```
Implicit UCB with:
- High β (exploration) in Rounds 1-5
- Low β (exploitation) in Rounds 6-13
- Function-specific adaptation based on observed noise
```

### Decision Rules

```python
if output > previous_best:
    continue_direction(step_size)
elif function_is_noisy:
    revert_to_best_historical_query()
elif output < previous_best and recent_failures > 2:
    revert_to_best_query()
else:
    try_smaller_perturbation()
```

---

## Version History

| Version | Round   | Key Changes                              |
| ------- | ------- | ---------------------------------------- |
| 0.1     | R1-R3   | Initial exploration strategy             |
| 0.2     | R4-R6   | Added gradient estimation                |
| 0.3     | R7-R9   | Implemented revert-to-best; F1 diagonal  |
| 1.0     | R10     | Breakthrough exploitation                |
| 2.0     | R11-R13 | Final: noise-aware strategies; securing gains |

---

## Citation

If referencing this approach:

```
BBO Capstone Optimisation Strategy, v2.0
Imperial College Business School, ML & AI Programme
Modules 12-25, 2025-2026
GitHub: https://github.com/elmurci/imperial-ml-ai-capstone-project
```

---

## Contact

For questions or feedback:
- GitHub repository issues
- Course discussion board
