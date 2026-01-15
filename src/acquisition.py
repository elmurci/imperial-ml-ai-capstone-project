"""
Acquisition Functions Module for BBO Capstone Project

This module implements acquisition functions for Bayesian optimization,
which guide the selection of the next query point.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Callable, Tuple, Optional


def upper_confidence_bound(
    X: np.ndarray,
    gp,
    beta: float = 2.0
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB(x) = μ(x) + β * σ(x)
    
    Higher β encourages exploration; lower β encourages exploitation.
    
    Args:
        X: Points to evaluate, shape (n_points, n_dims)
        gp: Fitted GP model with predict(X, return_std=True) method
        beta: Exploration-exploitation trade-off parameter
        
    Returns:
        ucb_values: UCB values at each point
    """
    mean, std = gp.predict(X, return_std=True)
    return mean + beta * std


def expected_improvement(
    X: np.ndarray,
    gp,
    y_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.
    
    EI(x) = E[max(f(x) - y_best - ξ, 0)]
    
    Args:
        X: Points to evaluate, shape (n_points, n_dims)
        gp: Fitted GP model
        y_best: Best observed output value
        xi: Exploration parameter (small positive value)
        
    Returns:
        ei_values: EI values at each point
    """
    mean, std = gp.predict(X, return_std=True)
    
    # Avoid division by zero
    std = np.maximum(std, 1e-9)
    
    # Standardized improvement
    z = (mean - y_best - xi) / std
    
    # Expected improvement formula
    ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
    
    # Set EI to 0 where std is essentially 0
    ei[std < 1e-9] = 0.0
    
    return ei


def probability_of_improvement(
    X: np.ndarray,
    gp,
    y_best: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition function.
    
    PI(x) = P(f(x) > y_best + ξ)
    
    Args:
        X: Points to evaluate, shape (n_points, n_dims)
        gp: Fitted GP model
        y_best: Best observed output value
        xi: Exploration parameter
        
    Returns:
        pi_values: PI values at each point
    """
    mean, std = gp.predict(X, return_std=True)
    
    # Avoid division by zero
    std = np.maximum(std, 1e-9)
    
    z = (mean - y_best - xi) / std
    
    return norm.cdf(z)


def optimize_acquisition(
    acquisition_func: Callable,
    bounds: np.ndarray,
    n_restarts: int = 10,
    n_random: int = 1000
) -> Tuple[np.ndarray, float]:
    """
    Optimize an acquisition function to find the next query point.
    
    Uses a combination of random sampling and local optimization.
    
    Args:
        acquisition_func: Function that takes X and returns acquisition values
        bounds: Array of (min, max) for each dimension, shape (n_dims, 2)
        n_restarts: Number of local optimization restarts
        n_random: Number of random points to evaluate initially
        
    Returns:
        x_best: Best point found
        acq_best: Acquisition value at best point
    """
    n_dims = len(bounds)
    
    # Random search phase
    X_random = np.random.uniform(
        bounds[:, 0], 
        bounds[:, 1], 
        size=(n_random, n_dims)
    )
    acq_random = acquisition_func(X_random)
    
    # Select top points for local optimization
    top_indices = np.argsort(acq_random)[-n_restarts:]
    
    best_x = None
    best_acq = -np.inf
    
    # Local optimization from each starting point
    for idx in top_indices:
        x0 = X_random[idx]
        
        # Minimize negative acquisition (to maximize)
        result = minimize(
            lambda x: -acquisition_func(x.reshape(1, -1))[0],
            x0,
            bounds=[(b[0], b[1]) for b in bounds],
            method='L-BFGS-B'
        )
        
        if -result.fun > best_acq:
            best_acq = -result.fun
            best_x = result.x
    
    return best_x, best_acq


def suggest_next_point(
    gp,
    bounds: np.ndarray,
    acquisition: str = "ucb",
    **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Suggest the next query point using Bayesian optimization.
    
    Args:
        gp: Fitted GP surrogate model
        bounds: Array of (min, max) for each dimension
        acquisition: Acquisition function ('ucb', 'ei', or 'pi')
        **kwargs: Additional arguments for acquisition function
        
    Returns:
        x_next: Suggested next query point
        acq_value: Acquisition value at suggested point
    """
    # Get best observed value for EI/PI
    if hasattr(gp, 'get_best'):
        _, y_best = gp.get_best()
    else:
        y_best = np.max(gp.Y) if hasattr(gp, 'Y') else 0.0
    
    # Define acquisition function
    if acquisition == "ucb":
        beta = kwargs.get('beta', 2.0)
        acq_func = lambda X: upper_confidence_bound(X, gp, beta=beta)
    elif acquisition == "ei":
        xi = kwargs.get('xi', 0.01)
        acq_func = lambda X: expected_improvement(X, gp, y_best, xi=xi)
    elif acquisition == "pi":
        xi = kwargs.get('xi', 0.01)
        acq_func = lambda X: probability_of_improvement(X, gp, y_best, xi=xi)
    else:
        raise ValueError(f"Unknown acquisition function: {acquisition}")
    
    # Optimize acquisition function
    x_next, acq_value = optimize_acquisition(
        acq_func,
        bounds,
        n_restarts=kwargs.get('n_restarts', 10),
        n_random=kwargs.get('n_random', 1000)
    )
    
    return x_next, acq_value


def format_query(x: np.ndarray) -> str:
    """
    Format a query point for submission to the capstone portal.
    
    Args:
        x: Query point array
        
    Returns:
        Formatted string like "0.123456-0.654321"
    """
    # Clip to valid range [0, 1)
    x = np.clip(x, 0.0, 0.999999)
    
    # Format each value to 6 decimal places
    formatted = [f"{val:.6f}" for val in x]
    
    return "-".join(formatted)


if __name__ == "__main__":
    # Example usage
    from surrogate import GPSurrogate
    
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.rand(10, 2)
    Y_train = np.sin(X_train[:, 0] * np.pi) + np.cos(X_train[:, 1] * np.pi)
    
    # Fit surrogate
    surrogate = GPSurrogate()
    surrogate.fit(X_train, Y_train)
    
    # Define bounds
    bounds = np.array([[0, 1], [0, 1]])
    
    # Test different acquisition functions
    for acq_name in ["ucb", "ei", "pi"]:
        x_next, acq_val = suggest_next_point(
            surrogate.gp, 
            bounds, 
            acquisition=acq_name
        )
        print(f"{acq_name.upper()}: x={x_next}, acq={acq_val:.4f}")
        print(f"  Portal format: {format_query(x_next)}")
