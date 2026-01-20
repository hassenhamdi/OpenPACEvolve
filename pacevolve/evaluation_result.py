"""
Evaluation result dataclass for OpenPACEvolve.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """
    Result of evaluating a program.
    
    Attributes:
        metrics: Dictionary of metric name to value (higher is better).
        artifacts: Optional dictionary of execution artifacts for feedback.
        error: Optional error message if evaluation failed.
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def combined_score(self) -> float:
        """Get the combined score, or compute from metrics."""
        if "combined_score" in self.metrics:
            return self.metrics["combined_score"]
        
        # Average all metrics if no combined_score
        if not self.metrics:
            return 0.0
        
        return sum(self.metrics.values()) / len(self.metrics)
    
    @property
    def is_valid(self) -> bool:
        """Check if evaluation was successful."""
        return self.error is None and self.combined_score > 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            error=data.get("error"),
        )
    
    @classmethod
    def failure(cls, error: str) -> "EvaluationResult":
        """Create a failed evaluation result."""
        return cls(
            metrics={"combined_score": 0.0},
            error=error,
        )
