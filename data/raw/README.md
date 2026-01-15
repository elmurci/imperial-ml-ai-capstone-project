# Raw Data Directory

This directory should contain the original `.npy` files provided by Imperial College Business School.

## Expected Structure

```
data/raw/
├── function_1/
│   ├── initial_inputs.npy   # Shape: (10, 2)
│   └── initial_outputs.npy  # Shape: (10,)
├── function_2/
│   ├── initial_inputs.npy   # Shape: (10, 2)
│   └── initial_outputs.npy  # Shape: (10,)
├── function_3/
│   ├── initial_inputs.npy   # Shape: (15, 3)
│   └── initial_outputs.npy  # Shape: (15,)
├── function_4/
│   ├── initial_inputs.npy   # Shape: (30, 4)
│   └── initial_outputs.npy  # Shape: (30,)
├── function_5/
│   ├── initial_inputs.npy   # Shape: (20, 4)
│   └── initial_outputs.npy  # Shape: (20,)
├── function_6/
│   ├── initial_inputs.npy   # Shape: (20, 5)
│   └── initial_outputs.npy  # Shape: (20,)
├── function_7/
│   ├── initial_inputs.npy   # Shape: (30, 6)
│   └── initial_outputs.npy  # Shape: (30,)
└── function_8/
    ├── initial_inputs.npy   # Shape: (40, 8)
    └── initial_outputs.npy  # Shape: (40,)
```

## How to Obtain

1. Download from Mini-lesson 12.8 on the course portal
2. Extract the ZIP file
3. Place the function folders in this directory

## Loading Data

```python
import numpy as np

X = np.load("data/raw/function_1/initial_inputs.npy")
Y = np.load("data/raw/function_1/initial_outputs.npy")
```

**Note**: These files are not stored in the repository due to size. See the main README for data access instructions.
