# Bayesian Black-Box Optimisation (BBO) Capstone Project

## Non-Technical Summary

This project tackles the challenge of finding optimal inputs for eight unknown "black-box" functions—systems where we can observe outputs but cannot see the internal workings. Using Bayesian optimisation, we intelligently balance exploring new regions against exploiting known promising areas, making informed decisions with limited data. This mirrors real-world scenarios like drug discovery, manufacturing tuning, or hyperparameter optimisation where each evaluation is expensive. Over multiple rounds of iterative querying, we progressively improved outputs across functions of varying complexity (2D to 8D), demonstrating practical optimisation skills applicable to industry ML challenges.

---

## Project Overview

| Function | Dimensions | Initial Points | Best Output (Round 5) | Real-World Analogy |
|----------|------------|----------------|----------------------|-------------------|
| F1 | 2D | 10 | ~0 (searching) | Radiation detection |
| F2 | 2D | 10 | 0.651 | Drug efficacy |
| F3 | 3D | 15 | -0.031 | Manufacturing quality |
| F4 | 4D | 30 | -0.0033 | Process optimisation |
| F5 | 4D | 20 | 1808.33 | Resource allocation |
| F6 | 5D | 20 | -0.678 | Side effect minimisation |
| F7 | 6D | 30 | 1.390 | Robot control |
| F8 | 8D | 40 | 9.787 | Complex system tuning |

## Repository Structure

```
bbo-capstone/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Original .npy files (not stored - see Data section)
│   └── processed/            # Accumulated data after each round
├── notebooks/
│   └── bbo_optimization.ipynb    # Main analysis notebook
├── src/
│   ├── surrogate.py          # Gaussian Process surrogate model
│   ├── acquisition.py        # Acquisition function implementations
│   └── utils.py              # Helper functions
├── results/
│   ├── queries.csv           # All submissions and outputs
│   └── figures/              # Visualisations
└── docs/
    ├── DATASHEET.md          # Data documentation
    └── MODEL_CARD.md         # Model documentation
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/[username]/bbo-capstone.git
cd bbo-capstone

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/bbo_optimization.ipynb
```

## Data

The initial `.npy` data files are provided by Imperial College Business School and are not stored in this repository due to size. They can be obtained from the course portal (Mini-lesson 12.8).

To load data:
```python
import numpy as np
X = np.load("data/raw/function_1/initial_inputs.npy")
Y = np.load("data/raw/function_1/initial_outputs.npy")
```

## Methodology

### Why Bayesian Optimisation?

1. **Limited evaluations**: Only one query per function per week
2. **Unknown function structure**: No access to gradients or analytical form
3. **Exploration-exploitation trade-off**: Need to balance searching new regions vs. refining known good areas

### Approach

1. **Surrogate Model**: Gaussian Process (GP) regression to approximate the unknown function
2. **Acquisition Function**: Upper Confidence Bound (UCB) and Expected Improvement (EI) to guide query selection
3. **Iterative Refinement**: Update beliefs after each observation

## Results Summary

### Best Outputs by Round

| Function | Initial Best | R1 | R2 | R3 | R4 | Trend |
|----------|-------------|-----|-----|-----|-----|-------|
| F1 | 0 | 0 | 0 | 2.68e-9 | 1.45e-23 | Searching |
| F2 | 0.611 | -0.069 | 0.334 | 0.651 | 0.639 | ✅ Improved |
| F3 | -0.035 | -0.161 | -0.122 | -0.033 | -0.031 | ✅ Improved |
| F4 | -4.03 | -0.0055 | -0.037 | -0.338 | -0.0033 | ✅ Improved |
| F5 | 1088.86 | 1139.17 | 1264.40 | 1484.13 | 1808.33 | 🚀 Excellent |
| F6 | -0.71 | -1.36 | -0.737 | -0.678 | -0.680 | ✅ Improved |
| F7 | 1.365 | 1.353 | 1.271 | 1.390 | 1.376 | ✅ Improved |
| F8 | 9.60 | 9.78 | 9.787 | 9.782 | 9.777 | ✅ Improved |

### Key Insights

- **F5** showed remarkable unimodal behaviour, climbing from 1088 to 1808 (+66% improvement)
- **F1** remains challenging—signal is extremely weak, suggesting a narrow optimum
- **F4** demonstrated the importance of reverting to successful regions after failed exploration
- Higher-dimensional functions (F6-F8) show slower convergence due to curse of dimensionality

## Challenges & Lessons Learned

1. **Exploration vs. Exploitation**: Early aggressive exploration often backfired; exploitation near known good points proved more reliable
2. **Gradient Information**: Round-by-round output changes provided implicit gradient estimates
3. **Dimensionality Curse**: Functions F6-F8 required more conservative step sizes
4. **Local Optima Risk**: F1 may have multiple optima, making search difficult

## Documentation

- [Datasheet](docs/DATASHEET.md) - Data sources, preprocessing, and limitations
- [Model Card](docs/MODEL_CARD.md) - Model behaviour, assumptions, and interpretability

## Technologies Used

- Python 3.10+
- NumPy, SciPy
- scikit-learn (GaussianProcessRegressor)
- Matplotlib, Seaborn
- Jupyter Notebooks

## Author

Imperial College Business School - Machine Learning & AI Programme  
Capstone Project, Modules 12-25

## License

This project is for educational purposes as part of the Imperial College Executive Education programme.
