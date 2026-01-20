"""
Evaluator for the function minimization example.

Tests how well the evolved search algorithm finds the global minimum.
"""

import importlib.util
import numpy as np
import time
import traceback
from typing import Any, Dict

# Known global minimum (approximate)
GLOBAL_MIN_X = -1.704
GLOBAL_MIN_Y = 0.678
GLOBAL_MIN_VALUE = -1.519


def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Evaluate the search algorithm by running it multiple times.
    
    Args:
        program_path: Path to the program file.
        
    Returns:
        Dictionary with metrics and artifacts.
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "run_search"):
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": "Missing run_search function"},
                "error": "Missing run_search function",
            }
        
        # Run multiple trials
        num_trials = 10
        values = []
        distances = []
        times = []
        success_count = 0
        
        for trial in range(num_trials):
            try:
                start_time = time.time()
                result = program.run_search()
                elapsed = time.time() - start_time
                
                # Parse result
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        x, y = result
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                    else:
                        continue
                else:
                    continue
                
                x = safe_float(x)
                y = safe_float(y)
                value = safe_float(value)
                
                if np.isnan(x) or np.isnan(y) or np.isnan(value):
                    continue
                
                distance = np.sqrt((x - GLOBAL_MIN_X)**2 + (y - GLOBAL_MIN_Y)**2)
                
                values.append(value)
                distances.append(distance)
                times.append(elapsed)
                success_count += 1
                
            except Exception as e:
                continue
        
        if success_count == 0:
            return {
                "metrics": {"combined_score": 0.0, "reliability_score": 0.0},
                "artifacts": {"error": "All trials failed"},
                "error": "All trials failed",
            }
        
        # Calculate metrics
        avg_value = float(np.mean(values))
        avg_distance = float(np.mean(distances))
        avg_time = float(np.mean(times))
        
        value_score = 1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE))
        distance_score = 1.0 / (1.0 + avg_distance)
        reliability_score = success_count / num_trials
        speed_score = 1.0 / (1.0 + avg_time)
        
        # Quality multiplier based on distance
        if avg_distance < 0.5:
            quality_mult = 1.5
        elif avg_distance < 1.5:
            quality_mult = 1.2
        elif avg_distance < 3.0:
            quality_mult = 1.0
        else:
            quality_mult = 0.7
        
        base_score = 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability_score
        combined_score = base_score * quality_mult
        
        return {
            "metrics": {
                "value_score": value_score,
                "distance_score": distance_score,
                "reliability_score": reliability_score,
                "speed_score": speed_score,
                "combined_score": combined_score,
            },
            "artifacts": {
                "avg_value": f"{avg_value:.4f}",
                "avg_distance_to_target": f"{avg_distance:.4f}",
                "avg_time_seconds": f"{avg_time:.3f}",
                "success_rate": f"{reliability_score:.2%}",
                "target_value": f"{GLOBAL_MIN_VALUE:.4f}",
            },
        }
        
    except Exception as e:
        return {
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": str(e), "traceback": traceback.format_exc()},
            "error": str(e),
        }


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """Quick first-stage evaluation with single trial."""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        if not hasattr(program, "run_search"):
            return {"metrics": {"combined_score": 0.0}, "error": "Missing run_search"}
        
        result = program.run_search()
        
        if isinstance(result, tuple) and len(result) >= 2:
            x, y = result[:2]
            value = result[2] if len(result) > 2 else (
                np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
            )
            
            distance = np.sqrt((x - GLOBAL_MIN_X)**2 + (y - GLOBAL_MIN_Y)**2)
            value_score = 1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE))
            distance_score = 1.0 / (1.0 + distance)
            
            combined = 0.6 * value_score + 0.4 * distance_score
            
            return {
                "metrics": {"combined_score": combined, "runs_successfully": 1.0},
                "artifacts": {"stage1_distance": f"{distance:.4f}"},
            }
        
        return {"metrics": {"combined_score": 0.0}, "error": "Invalid result format"}
        
    except Exception as e:
        return {"metrics": {"combined_score": 0.0}, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """Full evaluation."""
    return evaluate(program_path)
