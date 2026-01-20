"""Hierarchical Context Management (HCM) components for OpenPACEvolve."""

from pacevolve.context.idea_pool import Idea, IdeaPool
from pacevolve.context.hypothesis import Hypothesis, HypothesisManager
from pacevolve.context.failure_memory import FailureMemory
from pacevolve.context.pruning import ContextPruner

__all__ = [
    "Idea",
    "IdeaPool", 
    "Hypothesis",
    "HypothesisManager",
    "FailureMemory",
    "ContextPruner",
]
