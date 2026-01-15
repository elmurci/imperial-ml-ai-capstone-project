"""
Surrogate Model Module for BBO Capstone Project

This module implements Gaussian Process surrogate models for approximating
unknown black-box functions in Bayesian optimization.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from typing import Tuple, Optional


class GPSurrogate:
    """
    Gaussian Process surrogate model for black-box function approximation.
    
    Attributes:
        kernel: GP kernel function
        gp: Fitted GaussianProcessRegressor
        X: Training inputs
        Y: Training outputs
    """
    
    def __init__(
        self,
        kernel_type: str = "rbf",
        length_scale: float = 0.5,
        noise_level: float = 0.1,
        n_restarts: int = 10,
        random_state: int = 42
    ):
        """
        Initialize the GP surrogate model.
        
        Args:
            kernel_type: Type of kernel ('rbf' or 'matern')
            length_scale: Initial length scale for RBF/Matern kernel
            noise_level: Initial noise level for WhiteKernel
            n_restarts: Number of optimizer restarts for hyperparameter tuning
            random_state: Random seed for reproducibility
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        self.random_state = random_state
        
        self.kernel = self._create_kernel()
        self.gp = None
        self.X = None
        self.Y = None
        
    def _create_kernel(self):
        """Create the GP kernel based on specified type."""
        if self.kernel_type == "rbf":
            base_kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == "matern":
            base_kernel = Matern(length_scale=self.length_scale, nu=2.5)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        return ConstantKernel(1.0) * base_kernel + WhiteKernel(noise_level=self.noise_level)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GPSurrogate":
        """
        Fit the GP model to training data.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            Y: Output array of shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        self.X = np.array(X)
        self.Y = np.array(Y).ravel()
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=self.random_state
        )
        
        self.gp.fit(self.X, self.Y)
        return self
    
    def predict(
        self, 
        X: np.ndarray, 
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with the GP model.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            return_std: Whether to return standard deviation
            
        Returns:
            mean: Predicted mean of shape (n_samples,)
            std: Predicted standard deviation (if return_std=True)
        """
        if self.gp is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return self.gp.predict(X, return_std=return_std)
    
    def update(self, x_new: np.ndarray, y_new: float) -> "GPSurrogate":
        """
        Update the model with a new observation.
        
        Args:
            x_new: New input point
            y_new: New output value
            
        Returns:
            self: Updated model
        """
        x_new = np.array(x_new).reshape(1, -1)
        y_new = np.array([y_new])
        
        self.X = np.vstack([self.X, x_new])
        self.Y = np.append(self.Y, y_new)
        
        return self.fit(self.X, self.Y)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        Get the best observed point.
        
        Returns:
            x_best: Input with highest output
            y_best: Highest observed output
        """
        if self.Y is None:
            raise RuntimeError("No data available.")
            
        idx_best = np.argmax(self.Y)
        return self.X[idx_best], self.Y[idx_best]


def load_function_data(function_num: int, data_dir: str = "data/raw") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load initial data for a specific function.
    
    Args:
        function_num: Function number (1-8)
        data_dir: Directory containing the data
        
    Returns:
        X: Input array
        Y: Output array
    """
    import os
    
    func_dir = os.path.join(data_dir, f"function_{function_num}")
    X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
    Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
    
    return X, Y


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data for testing
    X_train = np.random.rand(10, 2)
    Y_train = np.sin(X_train[:, 0] * np.pi) + np.cos(X_train[:, 1] * np.pi)
    
    # Fit surrogate
    surrogate = GPSurrogate(kernel_type="rbf")
    surrogate.fit(X_train, Y_train)
    
    # Make predictions
    X_test = np.array([[0.5, 0.5]])
    mean, std = surrogate.predict(X_test)
    
    print(f"Prediction at {X_test[0]}: mean={mean[0]:.4f}, std={std[0]:.4f}")
    
    # Get best point
    x_best, y_best = surrogate.get_best()
    print(f"Best observed: x={x_best}, y={y_best:.4f}")
