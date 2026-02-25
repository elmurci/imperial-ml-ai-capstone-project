# Datasheet: BBO Capstone Project Dataset

## Motivation

### Why was this dataset created?

This dataset was created to support the Bayesian Black-Box Optimisation (BBO)
capstone project for the Imperial College Business School Machine Learning & AI
programme. It documents the iterative process of finding optimal inputs for
eight unknown synthetic functions through sequential querying.

### What task does it support?

The dataset supports black-box optimisation — finding input combinations that
maximise unknown objective functions when only input-output pairs are
observable. This mirrors real-world scenarios such as hyperparameter tuning,
drug discovery, manufacturing optimisation, and experimental design where
function evaluations are expensive or limited.

### Who created this dataset?

The dataset was generated through my weekly query submissions to the capstone
portal, with outputs returned by the course evaluation system. Initial seed data
(10-40 points per function) was provided by Imperial College Business School.

---

## Composition

### What does the dataset contain?

The dataset contains input-output pairs for eight synthetic black-box functions:

| Function | Dimensions | Initial Points | Total Points (R10) | Input Format     | Output Type                 |
| -------- | ---------- | -------------- | ------------------ | ---------------- | --------------------------- |
| F1       | 2D         | 10             | 20                 | (x₁, x₂)         | Float (scientific notation) |
| F2       | 2D         | 10             | 20                 | (x₁, x₂)         | Float                       |
| F3       | 3D         | 15             | 25                 | (x₁, x₂, x₃)     | Float (negative)            |
| F4       | 4D         | 30             | 40                 | (x₁, x₂, x₃, x₄) | Float                       |
| F5       | 4D         | 20             | 30                 | (x₁, x₂, x₃, x₄) | Float (large positive)      |
| F6       | 5D         | 20             | 30                 | (x₁, ..., x₅)    | Float (negative)            |
| F7       | 6D         | 30             | 40                 | (x₁, ..., x₆)    | Float                       |
| F8       | 8D         | 40             | 50                 | (x₁, ..., x₈)    | Float                       |

### What is the size and format?

- **Total observations**: 255 input-output pairs across all functions
- **Input format**: NumPy arrays (.npy files), values in range [0, 1)
- **Output format**: Single float values (positive or negative)
- **Query format**: Hyphen-separated strings with six decimal places (e.g.,
  `0.123456-0.654321`)

### Are there any gaps?

Yes, significant gaps exist:

- **Spatial coverage**: Queries cluster around promising regions; vast areas
  remain unexplored
- **F8 undersampling**: 50 points cannot adequately cover 8-dimensional space
- **Corner regions**: Extreme boundary combinations (all zeros, all ones)
  largely untested
- **No replication**: Each query point tested only once; noise levels unknown

### Best results achieved (after Round 10):

| Function | Best Output | Best Query                                 | Round Achieved |
| -------- | ----------- | ------------------------------------------ | -------------- |
| F1       | 1.68e-5     | (0.57, 0.57)                               | R9             |
| F2       | 0.651       | (0.70, 0.93)                               | R3             |
| F3       | -0.0275     | (0.49, 0.61, 0.35)                         | R7             |
| F4       | 0.161       | (0.39, 0.438, 0.408, 0.44)                 | R9             |
| F5       | 3512.03     | (0.15, 0.92, 0.999, 0.999)                 | R9             |
| F6       | -0.669      | (0.71, 0.13, 0.71, 0.72, 0.03)             | R8             |
| F7       | 1.394       | (0.047, 0.506, 0.251, 0.184, 0.384, 0.724) | R9             |
| F8       | 9.787       | Multiple queries                           | R2, R7         |

---

## Collection Process

### How were the queries generated?

Queries were generated using an evolving Bayesian optimisation strategy:

1. **Rounds 1-2**: Exploratory phase — corner sampling, boundary testing,
   initial exploitation of promising regions from seed data
2. **Rounds 3-5**: Gradient estimation — using output changes to infer
   directional improvements
3. **Rounds 6-8**: Exploitation focus — refining successful regions, reverting
   when exploration failed
4. **Rounds 9-10**: Breakthrough exploitation — aggressive pursuit of discovered
   gradients (F1 diagonal, F4/F5 continued climbing)

### What strategy did you use?

- **Surrogate model**: Gaussian Process with RBF kernel (conceptual; manual
  implementation)
- **Acquisition approach**: Implicit Upper Confidence Bound balancing
  exploration/exploitation
- **Per-function adaptation**: Different strategies based on observed behaviour:
  - F1: Diagonal exploration after breakthrough
  - F4, F5: Consistent gradient following
  - F2, F3, F6: Revert-to-best when refinements failed
  - F7, F8: Small perturbations around plateau

### Over what time frame?

- **Duration**: 10 weeks (Modules 12-21)
- **Frequency**: One query per function per week
- **Processing**: Outputs returned at end of each module

### What are the sources of uncertainty?

- **Function noise**: Unknown; F2's inconsistent results suggest possible
  stochasticity
- **Measurement precision**: Outputs reported to ~15 decimal places
- **Query precision**: Inputs specified to 6 decimal places

---

## Preprocessing and Uses

### Have you applied any transformations?

Minimal preprocessing:

- **Input normalisation**: Already in [0, 1) range; no additional scaling
- **Output handling**: Scientific notation preserved for F1's small values
- **Data accumulation**: New observations appended to initial seed data each
  round
- **No outlier removal**: All observations retained regardless of unexpected
  values

### What are the intended uses?

- **Primary**: Demonstrating Bayesian optimisation methodology
- **Secondary**: Portfolio artifact showing iterative ML problem-solving
- **Educational**: Teaching exploration-exploitation trade-offs
- **Benchmarking**: Comparing different optimisation strategies

### What are inappropriate uses?

- **Production ML**: Synthetic functions don't represent real-world complexity
- **Algorithm benchmarking**: Limited query budget prevents fair comparison
- **Transfer learning**: Function-specific strategies unlikely to generalise
- **Statistical inference**: Insufficient replication for uncertainty
  quantification

---

## Distribution and Maintenance

### Where is the dataset available?

- **Repository**: GitHub (public) — [repository link]
- **Format**: CSV (queries.csv) and NumPy arrays (.npy)
- **Documentation**: This datasheet, README, and Jupyter notebooks

### What are the terms of use?

- **License**: Educational use as part of Imperial College Executive Education
  programme
- **Attribution**: Imperial College Business School, ML & AI Programme
- **Restrictions**: Initial seed data provided by course; not for redistribution

### Who maintains it?

- **Creator**: Student (capstone project author)
- **Updates**: Completed after Round 10; no further updates planned
- **Contact**: Via GitHub repository issues

---

## Ethical Considerations

### Privacy

No personal data involved — entirely synthetic functions.

### Bias

- **Sampling bias**: Exploitation-heavy strategy underexplores search space
- **Confirmation bias**: Success with gradient-following may have led to
  premature convergence
- **Survivorship bias**: Only successful strategies documented in detail

### Transparency

- All queries and outputs recorded with timestamps
- Decision rationale documented for each round
- Failed strategies (F2, F3, F6 regressions) explicitly acknowledged

---

## Limitations Summary

1. **Query budget**: One query per week severely limits exploration
2. **Dimensionality curse**: High-dimensional functions (F6-F8) inadequately
   sampled
3. **No ground truth**: True optima unknown; cannot verify convergence
4. **Single trajectory**: No comparison runs to assess strategy robustness
5. **Unknown noise**: Cannot distinguish function noise from optimisation error
