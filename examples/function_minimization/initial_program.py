# EVOLVE-BLOCK-START
"""Function minimization example for OpenPACEvolve"""
import numpy as np


def evaluate_function(x, y):
    """
    Complex non-convex function with multiple local minima.
    
    f(x, y) = sin(x) * cos(y) + sin(x*y) + (x^2 + y^2)/20
    
    Global minimum is approximately at (-1.704, 0.678) with value -1.519
    """
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.
    
    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)
        
    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    
    for _ in range(iterations):
        # Simple random search
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)
        
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
    
    return best_x, best_y, best_value
# EVOLVE-BLOCK-END


def run_search():
    """Entry point for the evaluator."""
    return search_algorithm()


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x:.4f}, {y:.4f}) with value {value:.4f}")
    print("Target: (-1.704, 0.678) with value -1.519")
