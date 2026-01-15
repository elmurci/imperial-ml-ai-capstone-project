"""
BBO Capstone Project - Source Module

This module provides tools for Bayesian black-box optimisation:
- surrogate: Gaussian Process surrogate models
- acquisition: Acquisition functions (UCB, EI, PI)
- utils: Helper functions for data loading and visualization
"""

from .surrogate import GPSurrogate, load_function_data
from .acquisition import (
    upper_confidence_bound,
    expected_improvement,
    probability_of_improvement,
    suggest_next_point,
    format_query
)
from .utils import (
    FUNCTION_INFO,
    append_observation,
    get_best_observation,
    plot_convergence,
    plot_2d_function,
    print_submission_format
)

__version__ = "0.1.0"
__author__ = "Imperial College ML & AI Programme"
