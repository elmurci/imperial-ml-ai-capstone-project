# Model Card: Gaussian Process Surrogate for BBO

## Model Details

### Model Description
This project uses Gaussian Process (GP) regression as a surrogate model to approximate unknown black-box functions. The GP provides both predictions and uncertainty estimates, enabling informed exploration-exploitation decisions.

### Model Type
- **Primary**: Gaussian Process Regressor (scikit-learn)
- **Alternative**: Neural Network surrogate (PyTorch) for high-dimensional functions

### Model Version
- scikit-learn GaussianProcessRegressor v1.2+
- Custom acquisition functions built on top

### Developer
Student project for Imperial College Business School ML & AI Programme

### Model Date
January 2025 (ongoing through Module 25)

---

## Intended Use

### Primary Intended Uses
- Approximate unknown black-box functions from limited observations
- Guide query selection via acquisition functions
- Balance exploration (uncertainty reduction) and exploitation (optimising known regions)

### Primary Intended Users
- Students learning Bayesian optimisation
- Practitioners applying BO to expensive-to-evaluate functions

### Out-of-Scope Use Cases
- Real-time production systems (GP inference is O(n³))
- Very high-dimensional problems (>10D without dimensionality reduction)
- Non-smooth or discontinuous functions

---

## Training Data

### Data Sources
Synthetic black-box function evaluations provided by Imperial College

### Data Volume
| Function | Initial Points | After Round 5 |
|----------|----------------|---------------|
| F1 (2D) | 10 | 15 |
| F2 (2D) | 10 | 15 |
| F3 (3D) | 15 | 20 |
| F4 (4D) | 30 | 35 |
| F5 (4D) | 20 | 25 |
| F6 (5D) | 20 | 25 |
| F7 (6D) | 30 | 35 |
| F8 (8D) | 40 | 45 |

### Preprocessing
- Input normalisation: Already in [0, 1) range
- Output normalisation: Applied per-function for numerical stability

---

## Model Architecture

### Gaussian Process Configuration

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=42
)
```

### Kernel Choice Rationale
- **RBF (Radial Basis Function)**: Assumes smooth, continuous functions
- **ConstantKernel**: Scales overall variance
- **WhiteKernel**: Captures observation noise

### Acquisition Functions

1. **Upper Confidence Bound (UCB)**
   ```
   UCB(x) = μ(x) + β * σ(x)
   ```
   - β controls exploration-exploitation trade-off
   - Higher β → more exploration

2. **Expected Improvement (EI)**
   ```
   EI(x) = E[max(f(x) - f_best, 0)]
   ```
   - Focuses on probability of improvement over current best

---

## Evaluation

### Metrics
- **Best output found**: Primary success metric
- **Cumulative regret**: Difference from (unknown) true optimum
- **Convergence rate**: Improvement per query

### Results by Function (After Round 4)

| Function | Initial Best | Current Best | Improvement |
|----------|-------------|--------------|-------------|
| F1 | 0 | ~0 | Searching |
| F2 | 0.611 | 0.651 | +6.5% |
| F3 | -0.035 | -0.031 | +11.4% |
| F4 | -4.03 | -0.0033 | +99.9% |
| F5 | 1088.86 | 1808.33 | +66.1% |
| F6 | -0.71 | -0.678 | +4.5% |
| F7 | 1.365 | 1.390 | +1.8% |
| F8 | 9.60 | 9.787 | +1.9% |

### Evaluation Limitations
- True optima unknown until course completion
- Limited queries prevent extensive validation
- No cross-validation due to sequential nature

---

## Ethical Considerations

### Potential Harms
- None identified (synthetic educational data)
- No personal data involved

### Limitations Leading to Potential Harms
- N/A for this educational context

### Mitigations
- N/A

---

## Limitations & Biases

### Known Limitations

1. **Smoothness Assumption**
   - GP with RBF kernel assumes smooth functions
   - May underperform on discontinuous or highly non-linear functions

2. **Curse of Dimensionality**
   - GP uncertainty estimates degrade in high dimensions
   - F7 (6D) and F8 (8D) show slower convergence

3. **Local Optima**
   - F1's extremely weak signal suggests possible multi-modal structure
   - GP may get trapped in local optima

4. **Computational Cost**
   - GP inference is O(n³) — not scalable beyond ~1000 points
   - Not an issue for this project (~50 points max)

5. **Kernel Hyperparameter Sensitivity**
   - Results depend on kernel choice and hyperparameter optimisation
   - Default settings may not suit all functions

### Biases

1. **Exploration Bias in Early Rounds**
   - Initial GP predictions are highly uncertain
   - May over-explore before sufficient data accumulated

2. **Exploitation Bias in Later Rounds**
   - As GP becomes more confident, exploration may be under-prioritised
   - Risk of missing global optima

---

## Recommendations

### When to Use This Model
- Limited function evaluations (expensive queries)
- Smooth, continuous objective functions
- Low-to-medium dimensionality (≤6D works well)
- Need for uncertainty quantification

### When to Consider Alternatives
- Very high dimensionality (>10D): Consider neural network surrogates
- Discrete/combinatorial inputs: Consider tree-based surrogates
- Non-smooth functions: Consider random forest surrogates
- Many observations available: Consider neural networks or ensembles

### Best Practices
1. Start with exploration-heavy acquisition (high β in UCB)
2. Gradually shift to exploitation as data accumulates
3. Monitor for convergence plateaus
4. Use gradient information from round-by-round changes
5. Revert to successful regions if exploration fails

---

## Caveats & Additional Information

### Interpretability
- GP provides mean prediction (μ) and uncertainty (σ) at any point
- Acquisition function values explain why specific points were chosen
- Kernel lengthscales indicate feature importance

### Updates
Model updated after each round with new observations:
```python
gp.fit(X_updated, Y_updated)
```

### Contact
Imperial College Business School, Executive Education programme
