"""
OpenPACEvolve - Open-source implementation of PACEvolve framework

PACEvolve (Progress-Aware Consistent Evolution) is an evolutionary agent framework
that addresses:
- Context Pollution via Hierarchical Context Management
- Mode Collapse via Momentum-Based Backtracking  
- Weak Collaboration via Self-Adaptive Crossover Sampling
"""

from pacevolve._version import __version__
from pacevolve.api import OpenPACEvolve, run_evolution, evolve_function
from pacevolve.config import Config, load_config
from pacevolve.evaluation_result import EvaluationResult

__all__ = [
    "__version__",
    "OpenPACEvolve",
    "run_evolution",
    "evolve_function",
    "Config",
    "load_config",
    "EvaluationResult",
]
