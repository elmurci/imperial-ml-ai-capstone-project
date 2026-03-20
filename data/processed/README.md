# Processed Data Directory

This directory contains accumulated data from all optimization rounds.

## Structure

After each round, new observations are appended to the initial data:

```
data/processed/
├── function_1/
│   ├── inputs.npy    # Initial + Round 1-N inputs
│   └── outputs.npy   # Initial + Round 1-N outputs
├── function_2/
│   └── ...
└── ...
```

## Updating After Each Round

```python
import numpy as np

# Load current data
X = np.load("data/processed/function_1/inputs.npy")
Y = np.load("data/processed/function_1/outputs.npy")

# Append new observation
X_new = np.array([[0.123456, 0.654321]])  # Your query
Y_new = np.array([0.42])                   # Portal response

X_updated = np.vstack([X, X_new])
Y_updated = np.append(Y, Y_new)

# Save
np.save("data/processed/function_1/inputs.npy", X_updated)
np.save("data/processed/function_1/outputs.npy", Y_updated)
```

## Final Status (Round 13 — COMPLETE)

| Function | Initial Points | Total Points (Round 13) |
|----------|----------------|-------------------------|
| F1 | 10 | 23 |
| F2 | 10 | 23 |
| F3 | 15 | 28 |
| F4 | 30 | 43 |
| F5 | 20 | 33 |
| F6 | 20 | 33 |
| F7 | 30 | 43 |
| F8 | 40 | 53 |

**Total observations across all functions: 279**
