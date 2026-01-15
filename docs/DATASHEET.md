# Datasheet: BBO Capstone Project Data

## Motivation

### For what purpose was the dataset created?
This dataset was created for the Imperial College Business School Machine Learning & AI capstone project. It provides initial observations for eight synthetic black-box functions to simulate real-world Bayesian optimisation scenarios where function evaluations are expensive and limited.

### Who created the dataset?
Imperial College Business School, Executive Education programme.

### Who funded the creation of the dataset?
Imperial College Business School as part of the educational curriculum.

---

## Composition

### What do the instances represent?
Each instance represents an input-output pair: a point in the input space and the corresponding function evaluation.

### How many instances are there?
Initial data points per function:

| Function | Dimensions | Initial Points | Shape |
|----------|------------|----------------|-------|
| F1 | 2D | 10 | (10, 2) |
| F2 | 2D | 10 | (10, 2) |
| F3 | 3D | 15 | (15, 3) |
| F4 | 4D | 30 | (30, 4) |
| F5 | 4D | 20 | (20, 4) |
| F6 | 5D | 20 | (20, 5) |
| F7 | 6D | 30 | (30, 6) |
| F8 | 8D | 40 | (40, 8) |

After 5 rounds of querying, each function has accumulated 5 additional data points.

### What data does each instance consist of?
- **Inputs (X)**: Float values in range [0, 1) for each dimension
- **Outputs (Y)**: Single float value (can be positive or negative)

### Is there a label or target associated with each instance?
Yes, the output value Y serves as the target. The goal is to find inputs that maximise Y.

### Is any information missing from individual instances?
No. All instances have complete input-output pairs.

### Are relationships between instances explicit?
No explicit relationships. However, nearby inputs in the search space often produce correlated outputs (smoothness assumption).

### Are there any errors, sources of noise, or redundancies?
The functions may contain noise by design. Some functions exhibit:
- Multiple local optima
- Flat regions (plateaus)
- Sharp peaks (narrow optima)

### Is the dataset self-contained?
Yes. The .npy files contain all necessary information.

---

## Collection Process

### How was the data acquired?
- **Initial data**: Provided by the course organisers, generated from synthetic black-box functions
- **Subsequent data**: Acquired through iterative querying via the capstone portal

### What mechanisms were used to collect the data?
1. Initial .npy files downloaded from course portal
2. Weekly query submissions via capstone portal
3. Outputs returned after processing (end of each module)

### Who was involved in the data collection process?
- Course organisers: Provided initial data and portal infrastructure
- Student: Submitted queries based on optimisation strategy

### Over what timeframe was the data collected?
- Initial data: Provided at Module 12
- Accumulated data: One query per function per week (Modules 12-25)
- Current dataset: Through Module 16 (Round 5)

### Were any ethical review processes conducted?
Not applicable—synthetic data with no personal information.

---

## Preprocessing/Cleaning/Labeling

### Was any preprocessing/cleaning/labeling applied?
Minimal preprocessing:
1. Loading .npy files using NumPy
2. Appending new observations after each round
3. No normalisation applied (inputs already in [0, 1) range)

### Was the raw data saved in addition to the preprocessed data?
Yes. Original .npy files preserved; accumulated data stored separately.

### Is the software used to preprocess/clean/label available?
Yes. Standard NumPy operations:
```python
import numpy as np
X = np.load("initial_inputs.npy")
Y = np.load("initial_outputs.npy")
X_updated = np.vstack((X, new_point))
Y_updated = np.append(Y, new_output)
```

---

## Uses

### What tasks has the dataset been used for?
- Bayesian optimisation experimentation
- Surrogate model training (Gaussian Processes)
- Acquisition function evaluation
- Exploration-exploitation strategy development

### Is there anything about the composition that might impact future uses?
- Limited sample size constrains complex model training
- High-dimensional functions (F6-F8) suffer from curse of dimensionality
- Some functions may have multiple optima

### Are there tasks for which the dataset should not be used?
- Production ML systems (synthetic, educational data)
- Benchmarking against real-world optimisation problems

---

## Distribution

### How is the dataset distributed?
- Initial data: Via Imperial College course portal (.npy files in ZIP archive)
- Accumulated data: Not redistributed; students maintain their own copies

### When was the dataset released?
Module 12 of the Imperial College ML & AI programme (2024-2025 cohort).

### Is the dataset distributed under a copyright or IP license?
Educational use only, as part of Imperial College Executive Education programme.

---

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
Imperial College Business School, Executive Education.

### How can the owner/curator/manager be contacted?
Through the course support system and discussion boards.

### Will the dataset be updated?
Yes. New observations added weekly through Module 25 (13 total rounds).

### Are older versions available?
Students maintain their own versioned copies via the accumulation process.

---

## Limitations

1. **Synthetic nature**: Functions don't capture full complexity of real-world problems
2. **Limited queries**: One query per function per week constrains learning
3. **No ground truth**: True optima unknown until project completion
4. **No gradient information**: Black-box setting prevents direct gradient computation
5. **Dimensionality variation**: Strategies that work for 2D may fail for 8D
