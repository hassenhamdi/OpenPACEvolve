"""
Public API for OpenPACEvolve.

Provides convenient entry points for running evolutions.
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from pacevolve.config import Config, load_config
from pacevolve.controller import Controller
from pacevolve.database import Program
from pacevolve.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class OpenPACEvolve:
    """
    Main class for running OpenPACEvolve evolutions.
    
    Example:
        evolve = OpenPACEvolve(
            initial_program_path="program.py",
            evaluator_path="evaluator.py",
            config_path="config.yaml",
        )
        best = evolve.run(iterations=100)
        print(best.code)
    """
    
    def __init__(
        self,
        initial_program_path: str,
        evaluator_path: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize OpenPACEvolve.
        
        Args:
            initial_program_path: Path to initial program.
            evaluator_path: Path to evaluator module.
            config_path: Path to YAML config file.
            config: Config object (overrides config_path).
            output_dir: Output directory for results.
        """
        self.initial_program_path = initial_program_path
        self.evaluator_path = evaluator_path
        self.config = config or load_config(config_path)
        self.output_dir = output_dir
        
        self._controller: Optional[Controller] = None
    
    def run(self, iterations: Optional[int] = None) -> Optional[Program]:
        """
        Run evolution synchronously.
        
        Args:
            iterations: Number of iterations (overrides config).
            
        Returns:
            Best program found, or None if evolution failed.
        """
        return asyncio.run(self.run_async(iterations))
    
    async def run_async(self, iterations: Optional[int] = None) -> Optional[Program]:
        """
        Run evolution asynchronously.
        
        Args:
            iterations: Number of iterations (overrides config).
            
        Returns:
            Best program found, or None if evolution failed.
        """
        self._controller = Controller(
            initial_program_path=self.initial_program_path,
            evaluator_path=self.evaluator_path,
            config=self.config,
            output_dir=self.output_dir,
        )
        
        return await self._controller.run(iterations)
    
    @property
    def best_program(self) -> Optional[Program]:
        """Get the best program found."""
        if self._controller:
            return self._controller.database.get_global_best()
        return None
    
    @property
    def best_score(self) -> float:
        """Get the best score achieved."""
        if self._controller:
            return self._controller.best_score
        return 0.0


def run_evolution(
    initial_program: Union[str, Path],
    evaluator: Union[str, Path, Callable],
    config_path: Optional[str] = None,
    iterations: int = 100,
    output_dir: Optional[str] = None,
    **config_overrides,
) -> Optional[Program]:
    """
    Run evolution with minimal configuration.
    
    Args:
        initial_program: Path to initial program or inline code string.
        evaluator: Path to evaluator module or callable.
        config_path: Optional path to config file.
        iterations: Number of iterations.
        output_dir: Output directory.
        **config_overrides: Override specific config values.
        
    Returns:
        Best program found.
        
    Example:
        result = run_evolution(
            initial_program="examples/program.py",
            evaluator="examples/evaluator.py",
            iterations=50,
        )
        print(f"Best score: {result.score}")
    """
    import tempfile
    
    # Handle inline code
    if isinstance(initial_program, str) and not Path(initial_program).exists():
        # Assume it's inline code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(initial_program)
            initial_program_path = f.name
    else:
        initial_program_path = str(initial_program)
    
    # Handle callable evaluator
    if callable(evaluator):
        # Create wrapper module
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(f'''
import pickle
import base64

_evaluator_func = pickle.loads(base64.b64decode("{__import__('base64').b64encode(__import__('pickle').dumps(evaluator)).decode()}"))

def evaluate(program_path):
    return _evaluator_func(program_path)
''')
            evaluator_path = f.name
    else:
        evaluator_path = str(evaluator)
    
    # Load and modify config
    config = load_config(config_path)
    config.max_iterations = iterations
    
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Run evolution
    evolve = OpenPACEvolve(
        initial_program_path=initial_program_path,
        evaluator_path=evaluator_path,
        config=config,
        output_dir=output_dir,
    )
    
    return evolve.run(iterations)


def evolve_function(
    func: Callable,
    test_cases: list,
    iterations: int = 50,
    **kwargs,
) -> Optional[Program]:
    """
    Evolve a Python function to pass test cases.
    
    Args:
        func: Function to evolve.
        test_cases: List of (input, expected_output) tuples.
        iterations: Number of iterations.
        **kwargs: Additional config overrides.
        
    Returns:
        Best program found.
        
    Example:
        def bubble_sort(arr):
            for i in range(len(arr)):
                for j in range(len(arr)-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        
        result = evolve_function(
            bubble_sort,
            test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
            iterations=30,
        )
    """
    import inspect
    import tempfile
    
    # Extract function code
    source_code = inspect.getsource(func)
    func_name = func.__name__
    
    # Create initial program
    initial_code = f'''
{source_code}

def run_function():
    """Entry point for evaluation."""
    return {func_name}
'''
    
    # Create evaluator
    evaluator_code = f'''
import importlib.util

def evaluate(program_path):
    """Evaluate the evolved function."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "run_function"):
        return {{"combined_score": 0.0, "error": "Missing run_function"}}
    
    func = module.run_function()
    
    test_cases = {test_cases!r}
    passed = 0
    
    for inputs, expected in test_cases:
        try:
            if isinstance(inputs, tuple):
                result = func(*inputs)
            else:
                result = func(inputs)
            
            if result == expected:
                passed += 1
        except Exception as e:
            continue
    
    score = passed / len(test_cases)
    return {{"combined_score": score, "passed": passed, "total": len(test_cases)}}
'''
    
    # Save files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(initial_code)
        initial_path = f.name
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(evaluator_code)
        evaluator_path = f.name
    
    return run_evolution(
        initial_program=initial_path,
        evaluator=evaluator_path,
        iterations=iterations,
        **kwargs,
    )
