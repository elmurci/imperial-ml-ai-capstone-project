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

## Current Status

| Function | Initial Points | Total Points (Round 5) |
|----------|----------------|------------------------|
| F1 | 10 | 15 |
| F2 | 10 | 15 |
| F3 | 15 | 20 |
| F4 | 30 | 35 |
| F5 | 20 | 25 |
| F6 | 20 | 25 |
| F7 | 30 | 35 |
| F8 | 40 | 45 |
