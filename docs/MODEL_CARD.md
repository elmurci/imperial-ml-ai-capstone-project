# Model Card: BBO Optimisation Approach

## Overview

| Field       | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| **Name**    | Adaptive Bayesian Black-Box Optimisation Strategy                   |
| **Type**    | Sequential model-based optimisation with Gaussian Process surrogate |
| **Version** | 1.0 (Final, Round 10)                                               |
| **Author**  | Imperial College ML & AI Programme — Capstone Project               |
| **Date**    | Modules 12-21 (10-week iteration cycle)                             |

---

## Intended Use

### What tasks is this approach suitable for?

- **Expensive function evaluation**: When each query has significant cost (time,
  money, resources)
- **Low-dimensional optimisation**: Functions with 2-6 input dimensions
- **Smooth, continuous functions**: Where local structure predicts global
  behaviour
- **Sequential decision-making**: When observations inform subsequent queries
- **Hyperparameter tuning**: Optimising ML model configurations
- **Experimental design**: Drug discovery, manufacturing parameter tuning

### What use cases should be avoided?

- **High-dimensional spaces (>10D)**: Curse of dimensionality limits
  effectiveness
- **Discrete/combinatorial inputs**: Strategy assumes continuous search space
- **Highly multimodal functions**: May converge to local optima
- **Noisy functions without replication**: Cannot distinguish signal from noise
- **Real-time optimisation**: Weekly iteration cycle inappropriate for
  latency-sensitive applications
- **Functions with discontinuities**: Smoothness assumption will fail

---

## Details: Strategy Evolution Across Ten Rounds

### Phase 1: Exploration (Rounds 1-2)

**Approach**: Boundary testing, corner sampling, initial exploitation near seed
data's best points.

**Key decisions**:

- F1: Tested corners (0,1) and (1,0) — both yielded zero
- F2, F3: Explored edges based on GP uncertainty estimates
- F4-F8: Exploited promising regions from initial data

**Lessons learned**: Aggressive exploration often failed; exploitation near
known good points more reliable.

### Phase 2: Gradient Estimation (Rounds 3-5)

**Approach**: Used output changes between rounds to infer directional gradients.

**Key decisions**:

- F1: Center (0.5, 0.5) found first non-zero signal (2.68e-9)
- F4: Discovered sign change from negative to positive — major breakthrough
- F5: Identified X3, X4 as primary drivers; began systematic increase

**Techniques**: Implicit gradient descent using round-over-round comparisons;
step size tuning per function.

### Phase 3: Exploitation Focus (Rounds 6-8)

**Approach**: Refined successful regions; implemented "revert-to-best" when
refinements failed.

**Key decisions**:

- F2, F3, F6: Reverted to earlier successful queries after regressions
- F4: Continued gradient following (0.016 → 0.064 → 0.105)
- F5: Pushed X3, X4 toward upper bounds (1808 → 2271 → 2717)
- F7: Small perturbations around plateau

**Techniques**: Conservative step sizes; function-specific strategies; portfolio
approach balancing exploration/exploitation.

### Phase 4: Breakthrough Exploitation (Rounds 9-10)

**Approach**: Aggressive pursuit of discovered gradients; diagonal discovery for
F1.

**Key breakthroughs**:

- F1: Diagonal direction (0.55, 0.55) → (0.57, 0.57) yielded 3000x improvement
- F4: Continued climb to 0.161
- F5: Broke 3500 barrier (3512)
- F7: Achieved new best through continued small steps

---

## Performance

### Results Summary (After Round 10)

| Function | Initial Best | Final Best | Improvement         | Status          |
| -------- | ------------ | ---------- | ------------------- | --------------- |
| F1       | 0            | 1.68e-5    | ∞ (from zero)       | 🚀 Breakthrough |
| F2       | 0.611        | 0.651      | +6.5%               | ⚠️ Plateaued    |
| F3       | -0.035       | -0.0275    | +21.4%              | ✅ Improved     |
| F4       | -4.03        | 0.161      | +104% (sign change) | 🚀 Excellent    |
| F5       | 1088.86      | 3512.03    | +222.5%             | 🚀 Outstanding  |
| F6       | -0.71        | -0.669     | +5.8%               | ⚠️ Unstable     |
| F7       | 1.365        | 1.394      | +2.1%               | ✅ Steady       |
| F8       | 9.60         | 9.787      | +1.9%               | ⚠️ Plateaued    |

### Metrics Used

- **Primary**: Best output value achieved (maximisation objective)
- **Secondary**: Round-over-round improvement rate
- **Diagnostic**: Reversion frequency (indicator of strategy instability)

### Performance by Function Characteristics

| Characteristic           | Functions | Performance     | Notes                       |
| ------------------------ | --------- | --------------- | --------------------------- |
| Clear gradient           | F4, F5    | Excellent       | Consistent improvement      |
| Sparse/narrow optimum    | F1        | Good (after R9) | Required diagonal discovery |
| Noisy/unstable           | F2, F6    | Moderate        | Frequent reversions needed  |
| High-dimensional plateau | F7, F8    | Limited         | Diminishing returns         |

---

## Assumptions and Limitations

### Key Assumptions

1. **Local smoothness**: Small input changes produce small output changes
   - _Impact_: Enables gradient-following; fails for discontinuous functions
   - _Evidence_: Validated for F4, F5; questionable for F2

2. **Unimodality**: Single global optimum dominates
   - _Impact_: Exploitation-heavy strategy converges to local region
   - _Risk_: May miss superior global optima

3. **Stationarity**: Function behaviour consistent across search space
   - _Impact_: Strategies learned in one region transfer elsewhere
   - _Limitation_: F1's narrow diagonal suggests non-stationary structure

4. **Low noise**: Outputs reliably reflect true function values
   - _Impact_: Single queries sufficient; no replication needed
   - _Risk_: F2's inconsistency suggests possible noise

### Constraints

1. **Query budget**: One query per function per week (10 total)
2. **No parallel evaluation**: Sequential queries only
3. **Fixed precision**: Six decimal places for inputs
4. **No derivative information**: Pure black-box setting

### Failure Modes

| Mode                  | Description                 | Affected Functions   |
| --------------------- | --------------------------- | -------------------- |
| Premature convergence | Stopped exploring too early | F2, F8               |
| Overshoot             | Step size too large         | F3 (R8-R9)           |
| Wrong direction       | Gradient estimate incorrect | F6 (multiple rounds) |
| Plateau trap          | Cannot escape local optimum | F8                   |

---

## Ethical Considerations

### Transparency and Reproducibility

**Documentation provided**:

- Complete query-output history (queries.csv)
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

1. **Validate smoothness assumption** before committing to gradient-based
   exploitation
2. **Include replication** for noisy functions
3. **Budget for exploration** even when exploitation seems promising
4. **Document failures** as thoroughly as successes
5. **Consider multi-start** to escape local optima

### Limitations of Transparency

- **Implicit decisions**: Some strategic choices made intuitively, not formally
  documented
- **Hindsight bias**: Post-hoc rationale may overstate intentionality
- **Single trajectory**: Cannot assess strategy variance without comparison runs

---

## Technical Specifications

### Surrogate Model (Conceptual)

```
Gaussian Process with:
- Kernel: RBF (Radial Basis Function)
- Length scale: Adaptive per function
- Noise: Assumed low (no WhiteKernel in practice)
```

### Acquisition Strategy

```
Implicit UCB with:
- High β (exploration) in early rounds
- Low β (exploitation) in later rounds
- Function-specific adaptation
```

### Decision Rules

```python
if output > previous_best:
    continue_direction(step_size)
elif output < previous_best and recent_failures > 2:
    revert_to_best_query()
else:
    try_small_perturbation()
```

---

## Version History

| Version | Round  | Key Changes                      |
| ------- | ------ | -------------------------------- |
| 0.1     | R1-R2  | Initial exploration strategy     |
| 0.2     | R3-R5  | Added gradient estimation        |
| 0.3     | R6-R8  | Implemented revert-to-best       |
| 1.0     | R9-R10 | Final: breakthrough exploitation |

---

## Citation

If referencing this approach:

```
BBO Capstone Optimisation Strategy, v1.0
Imperial College Business School, ML & AI Programme
Modules 12-21, 2025-2026
```

---

## Contact

For questions or feedback:

- GitHub repository issues
- Course discussion board
