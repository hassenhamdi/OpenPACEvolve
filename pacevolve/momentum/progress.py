"""
Progress Tracking for Momentum-Based Backtracking.

Implements Relative Progress and EWMA momentum tracking.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """State snapshot for backtracking."""
    generation: int
    best_score: float
    momentum: float
    timestamp: float


class ProgressTracker:
    """
    Track Relative Progress and momentum for an island.
    
    Implements the core metrics from PACEvolve:
    - Relative Progress (Rt): Scale-invariant improvement measure
    - Momentum (mt): EWMA of relative progress
    """
    
    def __init__(
        self,
        target: float = 0.0,
        beta: float = 0.9,
        intervention_threshold: float = 0.1,
        initial_score: Optional[float] = None,
    ):
        """
        Initialize the progress tracker.
        
        Args:
            target: Target score to reach (lower bound, e.g., 0.0 for minimization).
            beta: EWMA decay factor (higher = more smoothing).
            intervention_threshold: Trigger intervention when momentum < threshold.
            initial_score: Initial best score (set when first evaluation done).
        """
        self.target = target
        self.beta = beta
        self.intervention_threshold = intervention_threshold
        
        # Track state
        self.initial_score = initial_score
        self.prev_score: Optional[float] = initial_score
        self.best_score: Optional[float] = initial_score
        self.momentum = 1.0  # Start with high momentum
        
        # History for backtracking
        self.history: List[ProgressState] = []
        self.generation = 0
    
    def initialize(self, initial_score: float) -> None:
        """Initialize with the first evaluation score."""
        self.initial_score = initial_score
        self.prev_score = initial_score
        self.best_score = initial_score
        self._save_state()
    
    def calculate_relative_progress(
        self,
        prev_score: float,
        curr_score: float,
    ) -> float:
        """
        Calculate Relative Progress (Rt).
        
        Rt = (s_{t-1} - s_t) / (s_{t-1} - r)
        
        This is the fraction of the previous performance gap that was closed.
        Returns 0 if no improvement.
        
        Note: For maximization problems, we flip the direction.
        """
        if curr_score <= prev_score:
            # No improvement
            return 0.0
        
        # For maximization: higher is better
        # Gap is how far we are from target (assuming target is max achievable, e.g., 1.0)
        # We measure progress as improvement relative to remaining gap
        max_target = 1.0  # Assuming normalized scores
        
        prev_gap = max_target - prev_score
        if prev_gap <= 0:
            return 0.0  # Already at target
        
        improvement = curr_score - prev_score
        relative_progress = improvement / prev_gap
        
        return min(relative_progress, 1.0)  # Cap at 1.0
    
    def update(self, new_score: float) -> float:
        """
        Update tracking with a new evaluation score.
        
        Args:
            new_score: Score from latest evaluation.
            
        Returns:
            Updated momentum value.
        """
        self.generation += 1
        
        if self.best_score is None:
            self.initialize(new_score)
            return self.momentum
        
        # Check if this is a new best
        is_improvement = new_score > self.best_score
        
        if is_improvement:
            # Calculate relative progress
            relative_progress = self.calculate_relative_progress(
                self.best_score, new_score
            )
            
            # Update best score
            self.prev_score = self.best_score
            self.best_score = new_score
        else:
            relative_progress = 0.0
        
        # Update momentum using EWMA
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_progress
        
        # Save state for backtracking
        self._save_state()
        
        logger.debug(
            f"Gen {self.generation}: score={new_score:.4f}, "
            f"best={self.best_score:.4f}, Rt={relative_progress:.4f}, "
            f"momentum={self.momentum:.4f}"
        )
        
        return self.momentum
    
    def should_intervene(self) -> bool:
        """Check if momentum is below threshold, triggering intervention."""
        return self.momentum < self.intervention_threshold
    
    def get_absolute_progress(self) -> float:
        """
        Calculate Absolute Progress (Ai).
        
        Ai = (s_0 - s_best) / (s_0 - r)
        
        For maximization: Ai = (s_best - s_0) / (target - s_0)
        """
        if self.initial_score is None or self.best_score is None:
            return 0.0
        
        max_target = 1.0  # Assuming normalized scores
        initial_gap = max_target - self.initial_score
        
        if initial_gap <= 0:
            return 1.0  # Already at target initially
        
        improvement = self.best_score - self.initial_score
        absolute_progress = improvement / initial_gap
        
        return max(0.0, min(1.0, absolute_progress))
    
    def _save_state(self) -> None:
        """Save current state to history."""
        import time
        
        if self.best_score is not None:
            state = ProgressState(
                generation=self.generation,
                best_score=self.best_score,
                momentum=self.momentum,
                timestamp=time.time(),
            )
            self.history.append(state)
    
    def get_state_at_generation(self, generation: int) -> Optional[ProgressState]:
        """Get state at a specific generation."""
        for state in self.history:
            if state.generation == generation:
                return state
        return None
    
    def reset_to_generation(self, generation: int) -> bool:
        """
        Reset tracker to state at a specific generation.
        
        Returns True if reset was successful.
        """
        state = self.get_state_at_generation(generation)
        if state is None:
            return False
        
        self.best_score = state.best_score
        self.prev_score = state.best_score
        self.momentum = 1.0  # Reset momentum after backtrack
        self.generation = generation
        
        # Trim history
        self.history = [s for s in self.history if s.generation <= generation]
        
        logger.info(f"Reset to generation {generation}, score={state.best_score:.4f}")
        return True
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "target": self.target,
            "beta": self.beta,
            "intervention_threshold": self.intervention_threshold,
            "initial_score": self.initial_score,
            "best_score": self.best_score,
            "momentum": self.momentum,
            "generation": self.generation,
            "history": [
                {
                    "generation": s.generation,
                    "best_score": s.best_score,
                    "momentum": s.momentum,
                    "timestamp": s.timestamp,
                }
                for s in self.history
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProgressTracker":
        """Deserialize from dictionary."""
        tracker = cls(
            target=data.get("target", 0.0),
            beta=data.get("beta", 0.9),
            intervention_threshold=data.get("intervention_threshold", 0.1),
            initial_score=data.get("initial_score"),
        )
        tracker.best_score = data.get("best_score")
        tracker.momentum = data.get("momentum", 1.0)
        tracker.generation = data.get("generation", 0)
        
        for h in data.get("history", []):
            tracker.history.append(ProgressState(
                generation=h["generation"],
                best_score=h["best_score"],
                momentum=h["momentum"],
                timestamp=h["timestamp"],
            ))
        
        return tracker
