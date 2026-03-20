# Bayesian Black-Box Optimisation (BBO) Capstone Project

## Non-Technical Summary

This project tackles the challenge of finding optimal inputs for eight unknown "black-box" functions—systems where we can observe outputs but cannot see the internal workings. Using Bayesian optimisation over 13 weekly iterations, we intelligently balanced exploring new regions against exploiting known promising areas, making informed decisions with limited data. Key breakthroughs included F1's diagonal discovery (from zero to 1.40e-4), F4's sign reversal from -4.03 to +0.181, and F5's remarkable climb from 1,089 to 3,934 (+261%). This mirrors real-world scenarios like drug discovery, manufacturing tuning, or hyperparameter optimisation where each evaluation is expensive.

---

## 📚 Documentation

| Document                             | Description                                                                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| [**Datasheet**](docs/DATASHEET.md)   | Complete documentation of the dataset: motivation, composition, collection process, and limitations                         |
| [**Model Card**](docs/MODEL_CARD.md) | Detailed description of the optimisation approach: strategy evolution, performance, assumptions, and ethical considerations |

---

## Results Overview

### Final Performance (After Round 13)

| Function | Dims | Initial Best | Final Best | Improvement     | Strategy            |
| -------- | ---- | ------------ | ---------- | --------------- | ------------------- |
| **F1**   | 2D   | 0            | 1.40e-4    | 🚀 Breakthrough | Diagonal discovery  |
| **F2**   | 2D   | 0.611        | 0.651      | +6.5%           | Region refinement   |
| **F3**   | 3D   | -0.035       | -0.0275    | +21.4%          | Gradient following  |
| **F4**   | 4D   | -4.03        | +0.181     | 🚀 Sign reversal| Consistent gradient |
| **F5**   | 4D   | 1,089        | 3,934      | **+261%**       | Boundary pushing    |
| **F6**   | 5D   | -0.71        | -0.635     | +10.6%          | Revert-to-best      |
| **F7**   | 6D   | 1.365        | 1.397      | +2.3%           | Small perturbations |
| **F8**   | 8D   | 9.60         | 9.792      | +2.0%           | Plateau refinement  |

### Key Breakthroughs

- **F1 (Rounds 8-11)**: After 8 rounds of near-zero outputs, discovered the optimum lies along x=y diagonal
- **F4 (Rounds 5-13)**: Achieved sign reversal from -4.03 to +0.181 through consistent gradient following
- **F5 (All rounds)**: Outstanding monotonic improvement from 1,089 to 3,934 (+261%)

---

## Repository Structure

```
bbo-capstone/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Original .npy files (see data access instructions)
│   └── processed/            # Accumulated data after each round
├── docs/
│   ├── DATASHEET.md          # Dataset documentation ⭐
│   └── MODEL_CARD.md         # Model documentation ⭐
├── notebooks/
│   └── bbo_optimization.ipynb    # Main analysis notebook
├── src/
│   ├── surrogate.py          # GP surrogate model
│   ├── acquisition.py        # Acquisition functions
│   └── utils.py              # Helper functions
└── results/
    ├── queries.csv           # Complete query history (13 rounds)
    └── figures/              # Visualisations
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/elmurci/imperial-ml-ai-capstone-project.git
cd imperial-ml-ai-capstone-project

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/bbo_optimization.ipynb
```

---

## Methodology

### Strategy Evolution

| Phase               | Rounds | Approach                    | Key Insight                             |
| ------------------- | ------ | --------------------------- | --------------------------------------- |
| Exploration         | 1-3    | Corner/boundary testing     | Exploitation often more reliable        |
| Gradient Estimation | 4-6    | Directional inference       | F4/F5 have clear gradients              |
| Refinement          | 7-9    | Refine + revert-to-best     | Know when to abandon failed exploration |
| Exploitation        | 10-13  | Aggressive gradient pursuit | F1 diagonal, F5 boundary pushing        |

### Core Techniques

1. **Gaussian Process Surrogate**: Approximate unknown functions with uncertainty estimates
2. **Acquisition Functions**: UCB/EI to balance exploration-exploitation
3. **Gradient Following**: Use round-over-round changes to infer direction
4. **Revert-to-Best**: Abandon failed refinements, return to proven queries
5. **Per-Function Strategies**: Tailored approaches based on observed behaviour

---

## Data Access

Initial `.npy` data files are provided by Imperial College Business School via the course portal (Mini-lesson 12.8). They are not stored in this repository due to licensing.

```python
import numpy as np
X = np.load("data/raw/function_1/initial_inputs.npy")
Y = np.load("data/raw/function_1/initial_outputs.npy")
```

---

## Lessons Learned

1. **Exploitation > Exploration** with limited queries — aggressive exploration often wasted budget
2. **Function-specific strategies** essential — one-size-fits-all approaches fail
3. **Revert-to-best** critical for noisy functions (F2, F3, F6)
4. **Breakthroughs unpredictable** — F1's diagonal discovery came after 8 failed rounds
5. **Diminishing returns** real for high-dimensional functions (F7, F8)

---

## Technologies

- Python 3.10+
- NumPy, SciPy, Pandas
- scikit-learn (GaussianProcessRegressor)
- Matplotlib, Seaborn
- Jupyter Notebooks

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Initial seed data (.npy files) are provided by Imperial College Business School and are not included in this repository due to licensing restrictions.
