# Datasheet: BBO Capstone Project Dataset

## Motivation

### Why was this dataset created?

This dataset was created to support the Bayesian Black-Box Optimisation (BBO) capstone project for the Imperial College Business School Machine Learning & AI programme. It documents the iterative process of finding optimal inputs for eight unknown synthetic functions through sequential querying.

### What task does it support?

The dataset supports black-box optimisation — finding input combinations that maximise unknown objective functions when only input-output pairs are observable. This mirrors real-world scenarios such as hyperparameter tuning, drug discovery, manufacturing optimisation, and experimental design where function evaluations are expensive or limited.

### Who created this dataset?

The dataset was generated through my weekly query submissions to the capstone portal, with outputs returned by the course evaluation system. Initial seed data (10-40 points per function) was provided by Imperial College Business School.

---

## Composition

### What does the dataset contain?

The dataset contains input-output pairs for eight synthetic black-box functions:

| Function | Dimensions | Initial Points | Total Points (R13) | Input Format     | Output Type                 |
| -------- | ---------- | -------------- | ------------------ | ---------------- | --------------------------- |
| F1       | 2D         | 10             | 23                 | (x₁, x₂)         | Float (scientific notation) |
| F2       | 2D         | 10             | 23                 | (x₁, x₂)         | Float                       |
| F3       | 3D         | 15             | 28                 | (x₁, x₂, x₃)     | Float (negative)            |
| F4       | 4D         | 30             | 43                 | (x₁, x₂, x₃, x₄) | Float                       |
| F5       | 4D         | 20             | 33                 | (x₁, x₂, x₃, x₄) | Float (large positive)      |
| F6       | 5D         | 20             | 33                 | (x₁, ..., x₅)    | Float (negative)            |
| F7       | 6D         | 30             | 43                 | (x₁, ..., x₆)    | Float                       |
| F8       | 8D         | 40             | 53                 | (x₁, ..., x₈)    | Float                       |

### What is the size and format?

- **Total observations**: 279 input-output pairs across all functions (initial + 13 rounds × 8 functions)
- **Input format**: NumPy arrays (.npy files), values in range [0, 1)
- **Output format**: Single float values (positive or negative)
- **Query format**: Hyphen-separated strings with six decimal places (e.g., `0.123456-0.654321`)

### Are there any gaps?

Yes, significant gaps exist:

- **Spatial coverage**: Queries cluster around promising regions; vast areas remain unexplored
- **F8 undersampling**: 53 points cannot adequately cover 8-dimensional space
- **Corner regions**: Extreme boundary combinations (all zeros, all ones) largely untested
- **No replication**: Each query point tested only once; noise levels inferred from repeated queries giving different outputs

### Best results achieved (after Round 13 — FINAL):

| Function | Best Output | Best Query                                            | Round Achieved |
| -------- | ----------- | ----------------------------------------------------- | -------------- |
| F1       | 1.40e-4     | (0.58, 0.58)                                          | R11            |
| F2       | 0.651       | (0.70, 0.93)                                          | R3             |
| F3       | -0.0275     | (0.49, 0.61, 0.35)                                    | R7             |
| F4       | 0.181       | (0.386, 0.442, 0.412, 0.436)                          | R11, R13       |
| F5       | 3933.72     | (0.11, 0.96, 0.999, 0.999)                            | R13            |
| F6       | -0.635      | (0.712, 0.132, 0.712, 0.718, 0.032)                   | R11            |
| F7       | 1.397       | (0.051, 0.498, 0.243, 0.192, 0.392, 0.732)            | R12, R13       |
| F8       | 9.792       | (0.063, 0.043, 0.013, 0.013, 0.547, 0.347, 0.018, 0.247) | R13         |

---

## Collection Process

### How were the queries generated?

Queries were generated using an evolving Bayesian optimisation strategy:

1. **Rounds 1-3**: Exploratory phase — corner sampling, boundary testing, initial exploitation of promising regions from seed data
2. **Rounds 4-6**: Gradient estimation — using output changes to infer directional improvements
3. **Rounds 7-9**: Refinement focus — refining successful regions, reverting when exploration failed, F1 diagonal discovery
4. **Rounds 10-13**: Exploitation focus — aggressive pursuit of discovered gradients, securing gains through revert-to-best

### What strategy did you use?

- **Surrogate model**: Gaussian Process with RBF kernel (conceptual; manual implementation)
- **Acquisition approach**: Implicit Upper Confidence Bound balancing exploration/exploitation
- **Per-function adaptation**: Different strategies based on observed behaviour:
  - F1: Diagonal exploration after breakthrough (x=y constraint)
  - F4, F5: Consistent gradient following
  - F2, F3, F6: Revert-to-best when refinements failed (noisy functions)
  - F7, F8: Small perturbations around plateau

### Over what time frame?

- **Duration**: 13 weeks (Modules 12-25)
- **Frequency**: One query per function per week
- **Processing**: Outputs returned at end of each module

### What are the sources of uncertainty?

- **Function noise**: Confirmed for F2, F3, F6 — same query returned different outputs across rounds
- **Measurement precision**: Outputs reported to ~15 decimal places
- **Query precision**: Inputs specified to 6 decimal places

---

## Preprocessing and Uses

### Have you applied any transformations?

Minimal preprocessing:

- **Input normalisation**: Already in [0, 1) range; no additional scaling
- **Output handling**: Scientific notation preserved for F1's small values
- **Data accumulation**: New observations appended to initial seed data each round
- **No outlier removal**: All observations retained regardless of unexpected values

### What are the intended uses?

- **Primary**: Demonstrating Bayesian optimisation methodology
- **Secondary**: Portfolio artifact showing iterative ML problem-solving
- **Educational**: Teaching exploration-exploitation trade-offs
- **Benchmarking**: Comparing different optimisation strategies

### What are inappropriate uses?

- **Production ML**: Synthetic functions don't represent real-world complexity
- **Algorithm benchmarking**: Limited query budget prevents fair comparison
- **Transfer learning**: Function-specific strategies unlikely to generalise
- **Statistical inference**: Insufficient replication for uncertainty quantification

---

## Distribution and Maintenance

### Where is the dataset available?

- **Repository**: GitHub (public) — https://github.com/elmurci/imperial-ml-ai-capstone-project
- **Format**: CSV (queries.csv) and NumPy arrays (.npy)
- **Documentation**: This datasheet, README, MODEL_CARD, and Jupyter notebooks

### What are the terms of use?

- **License**: Educational use as part of Imperial College Executive Education programme
- **Attribution**: Imperial College Business School, ML & AI Programme
- **Restrictions**: Initial seed data provided by course; not for redistribution

### Who maintains it?

- **Creator**: Student (capstone project author)
- **Updates**: Completed after Round 13 (FINAL); no further updates planned
- **Contact**: Via GitHub repository issues

---

## Ethical Considerations

### Privacy

No personal data involved — entirely synthetic functions.

### Bias

- **Sampling bias**: Exploitation-heavy strategy underexplores search space
- **Confirmation bias**: Success with gradient-following may have led to premature convergence
- **Survivorship bias**: Only successful strategies documented in detail

### Transparency

- All queries and outputs recorded with timestamps
- Decision rationale documented for each round
- Failed strategies (F2, F3, F6 regressions) explicitly acknowledged
- Noisy function behaviour documented

---

## Limitations Summary

1. **Query budget**: One query per week severely limits exploration
2. **Dimensionality curse**: High-dimensional functions (F6-F8) inadequately sampled
3. **No ground truth**: True optima unknown; cannot verify convergence
4. **Single trajectory**: No comparison runs to assess strategy robustness
5. **Stochastic functions**: F2, F3, F6 showed noise that was only discovered mid-project
