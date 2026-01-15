"""
Utility Functions for BBO Capstone Project

Helper functions for data loading, visualization, and result tracking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os


# Function metadata
FUNCTION_INFO = {
    1: {"dims": 2, "initial_points": 10, "analogy": "Radiation detection"},
    2: {"dims": 2, "initial_points": 10, "analogy": "Drug efficacy"},
    3: {"dims": 3, "initial_points": 15, "analogy": "Manufacturing quality"},
    4: {"dims": 4, "initial_points": 30, "analogy": "Process optimization"},
    5: {"dims": 4, "initial_points": 20, "analogy": "Resource allocation"},
    6: {"dims": 5, "initial_points": 20, "analogy": "Side effect minimization"},
    7: {"dims": 6, "initial_points": 30, "analogy": "Robot control"},
    8: {"dims": 8, "initial_points": 40, "analogy": "Complex system tuning"},
}


def load_function_data(
    function_num: int, 
    data_dir: str = "data/raw"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data for a specific function.
    
    Args:
        function_num: Function number (1-8)
        data_dir: Directory containing the data
        
    Returns:
        X: Input array of shape (n_points, n_dims)
        Y: Output array of shape (n_points,)
    """
    func_dir = os.path.join(data_dir, f"function_{function_num}")
    
    X = np.load(os.path.join(func_dir, "initial_inputs.npy"))
    Y = np.load(os.path.join(func_dir, "initial_outputs.npy"))
    
    return X, Y.ravel()


def save_function_data(
    X: np.ndarray, 
    Y: np.ndarray, 
    function_num: int, 
    data_dir: str = "data/processed"
) -> None:
    """
    Save updated data for a function.
    
    Args:
        X: Input array
        Y: Output array
        function_num: Function number (1-8)
        data_dir: Directory to save data
    """
    func_dir = os.path.join(data_dir, f"function_{function_num}")
    os.makedirs(func_dir, exist_ok=True)
    
    np.save(os.path.join(func_dir, "inputs.npy"), X)
    np.save(os.path.join(func_dir, "outputs.npy"), Y)


def append_observation(
    X: np.ndarray, 
    Y: np.ndarray, 
    x_new: np.ndarray, 
    y_new: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append a new observation to existing data.
    
    Args:
        X: Existing inputs
        Y: Existing outputs
        x_new: New input point
        y_new: New output value
        
    Returns:
        X_updated: Updated inputs
        Y_updated: Updated outputs
    """
    x_new = np.array(x_new).reshape(1, -1)
    X_updated = np.vstack([X, x_new])
    Y_updated = np.append(Y, y_new)
    
    return X_updated, Y_updated


def format_query(x: np.ndarray) -> str:
    """
    Format a query point for portal submission.
    
    Args:
        x: Query point array
        
    Returns:
        Formatted string "0.xxxxxx-0.xxxxxx-..."
    """
    x = np.clip(x, 0.0, 0.999999)
    return "-".join([f"{val:.6f}" for val in x])


def parse_query(query_str: str) -> np.ndarray:
    """
    Parse a query string back to array.
    
    Args:
        query_str: String like "0.123456-0.654321"
        
    Returns:
        Array of values
    """
    return np.array([float(v) for v in query_str.split("-")])


def get_best_observation(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Get the best observation from data.
    
    Args:
        Y: Output values
        X: Input values
        
    Returns:
        x_best: Best input
        y_best: Best output
        idx_best: Index of best observation
    """
    idx_best = np.argmax(Y)
    return X[idx_best], Y[idx_best], idx_best


def plot_convergence(
    results: Dict[int, List[float]],
    title: str = "Optimization Progress",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot convergence curves for all functions.
    
    Args:
        results: Dict mapping function_num to list of best values per round
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for func_num in range(1, 9):
        ax = axes[func_num - 1]
        
        if func_num in results:
            values = results[func_num]
            rounds = range(len(values))
            
            ax.plot(rounds, values, 'b-o', linewidth=2, markersize=6)
            ax.fill_between(rounds, values, alpha=0.3)
            
            # Mark best
            best_round = np.argmax(values)
            ax.scatter([best_round], [values[best_round]], 
                      color='red', s=100, zorder=5, marker='*')
        
        ax.set_title(f"Function {func_num} ({FUNCTION_INFO[func_num]['dims']}D)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Best Output")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_2d_function(
    X: np.ndarray,
    Y: np.ndarray,
    gp=None,
    title: str = "Function Landscape",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a 2D function with observations and GP prediction.
    
    Args:
        X: Observed inputs (n_points, 2)
        Y: Observed outputs (n_points,)
        gp: Optional fitted GP for contour plot
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid for GP prediction
    if gp is not None:
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(0, 1, 50)
        X1, X2 = np.meshgrid(x1, x2)
        X_grid = np.column_stack([X1.ravel(), X2.ravel()])
        
        Y_pred, Y_std = gp.predict(X_grid, return_std=True)
        Y_pred = Y_pred.reshape(X1.shape)
        
        # Contour plot
        contour = ax.contourf(X1, X2, Y_pred, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='Predicted Output')
    
    # Plot observations
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', 
                        s=100, edgecolors='black', linewidth=1, zorder=5)
    plt.colorbar(scatter, ax=ax, label='Observed Output')
    
    # Mark best point
    idx_best = np.argmax(Y)
    ax.scatter([X[idx_best, 0]], [X[idx_best, 1]], 
              color='gold', s=200, marker='*', edgecolors='black', 
              linewidth=2, zorder=10, label='Best')
    
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_results_summary(queries_csv: str) -> pd.DataFrame:
    """
    Create a summary table from queries CSV.
    
    Args:
        queries_csv: Path to queries.csv file
        
    Returns:
        Summary DataFrame
    """
    df = pd.read_csv(queries_csv)
    
    # Get latest results per function
    summary = []
    for func_num in range(1, 9):
        func_data = df[df['function'] == f'F{func_num}']
        
        if len(func_data) > 0:
            latest = func_data.iloc[-1]
            best_row = func_data.loc[func_data['best_so_far'].idxmax()]
            
            summary.append({
                'Function': f'F{func_num}',
                'Dimensions': FUNCTION_INFO[func_num]['dims'],
                'Rounds': len(func_data) - 1,  # Subtract initial
                'Best Output': best_row['best_so_far'],
                'Best Query': best_row['query'],
                'Analogy': FUNCTION_INFO[func_num]['analogy']
            })
    
    return pd.DataFrame(summary)


def print_submission_format(queries: Dict[int, np.ndarray]) -> None:
    """
    Print queries in portal submission format.
    
    Args:
        queries: Dict mapping function_num to query array
    """
    print("=" * 60)
    print("SUBMISSION FORMAT FOR CAPSTONE PORTAL")
    print("=" * 60)
    
    for func_num in sorted(queries.keys()):
        query = queries[func_num]
        formatted = format_query(query)
        dims = FUNCTION_INFO[func_num]['dims']
        print(f"\nFunction {func_num} ({dims}D):")
        print(f"  {formatted}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Function Information:")
    for func_num, info in FUNCTION_INFO.items():
        print(f"  F{func_num}: {info['dims']}D, {info['initial_points']} initial points")
        print(f"         Analogy: {info['analogy']}")
    
    # Example query formatting
    x = np.array([0.123456, 0.654321])
    print(f"\nFormatted query: {format_query(x)}")
    
    # Example parsing
    query_str = "0.123456-0.654321"
    x_parsed = parse_query(query_str)
    print(f"Parsed query: {x_parsed}")
