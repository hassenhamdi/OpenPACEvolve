"""
Backtracking Manager for Momentum-Based Backtracking.

Implements power-law based state reversion when momentum drops.
"""

import logging
import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.momentum.progress import ProgressTracker, ProgressState

logger = logging.getLogger(__name__)


class BacktrackingManager:
    """
    Momentum-based backtracking with power-law state reversion.
    
    When an island's momentum drops below threshold, this triggers
    a backtrack to a previous state selected via power-law distribution.
    """
    
    def __init__(
        self,
        power_law_alpha: float = 1.5,
        min_backtrack_generations: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the backtracking manager.
        
        Args:
            power_law_alpha: Power-law exponent (higher = prefer recent states).
            min_backtrack_generations: Minimum generations back to consider.
            seed: Random seed for reproducibility.
        """
        self.power_law_alpha = power_law_alpha
        self.min_backtrack_gens = min_backtrack_generations
        self.rng = random.Random(seed)
    
    def should_backtrack(self, tracker: "ProgressTracker") -> bool:
        """Check if backtracking should be triggered."""
        return tracker.should_intervene()
    
    def select_backtrack_generation(
        self,
        tracker: "ProgressTracker",
    ) -> Optional[int]:
        """
        Select a generation to backtrack to using power-law distribution.
        
        Power-law favors more recent states but allows jumping back
        further with decreasing probability.
        
        Returns None if no suitable backtrack point exists.
        """
        history = tracker.history
        current_gen = tracker.generation
        
        if len(history) < self.min_backtrack_gens:
            return None
        
        # Calculate valid backtrack range
        min_gen = max(0, current_gen - len(history) + 1)
        max_gen = current_gen - self.min_backtrack_gens
        
        if max_gen <= min_gen:
            return None
        
        # Generate power-law distributed backtrack distance
        # P(k) ∝ k^(-α) where k is distance from current generation
        
        valid_gens = []
        weights = []
        
        for state in history:
            if min_gen <= state.generation <= max_gen:
                # Distance from current generation
                distance = current_gen - state.generation
                if distance >= self.min_backtrack_gens:
                    # Power-law weight (higher for closer generations)
                    weight = distance ** (-self.power_law_alpha)
                    valid_gens.append(state.generation)
                    weights.append(weight)
        
        if not valid_gens:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Sample
        selected_gen = self.rng.choices(valid_gens, probabilities)[0]
        
        logger.info(
            f"Selected backtrack generation {selected_gen} "
            f"(current: {current_gen}, distance: {current_gen - selected_gen})"
        )
        
        return selected_gen
    
    def execute_backtrack(
        self,
        tracker: "ProgressTracker",
        target_generation: int,
    ) -> bool:
        """
        Execute backtrack to target generation.
        
        Args:
            tracker: Progress tracker to backtrack.
            target_generation: Generation to backtrack to.
            
        Returns:
            True if backtrack was successful.
        """
        success = tracker.reset_to_generation(target_generation)
        
        if success:
            logger.info(f"Backtracked to generation {target_generation}")
        else:
            logger.warning(f"Failed to backtrack to generation {target_generation}")
        
        return success
    
    def handle_intervention(
        self,
        tracker: "ProgressTracker",
    ) -> Optional[int]:
        """
        Handle a momentum-triggered intervention.
        
        Decides whether to backtrack and to which generation.
        
        Returns the backtrack generation, or None if no backtrack.
        """
        if not self.should_backtrack(tracker):
            return None
        
        target_gen = self.select_backtrack_generation(tracker)
        
        if target_gen is not None:
            self.execute_backtrack(tracker, target_gen)
        
        return target_gen
