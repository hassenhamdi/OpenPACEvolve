"""
Action Weight Calculator for Self-Adaptive Crossover Sampling.

Implements the action weighting mechanism from PACEvolve.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ActionWeights:
    """Weights for different actions in self-adaptive sampling."""
    backtrack_weight: float = 0.0
    crossover_weights: Dict[int, float] = field(default_factory=dict)
    best_partner_id: int = -1
    
    @property
    def total_weight(self) -> float:
        """Get total weight for normalization."""
        return self.backtrack_weight + sum(self.crossover_weights.values())
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get action probabilities."""
        total = self.total_weight
        if total <= 0:
            return {}
        
        probs = {"backtrack": self.backtrack_weight / total}
        for island_id, weight in self.crossover_weights.items():
            probs[f"crossover_{island_id}"] = weight / total
        
        return probs


class ActionWeightCalculator:
    """
    Calculate adaptive weights for crossover/backtrack actions.
    
    Implements the action weighting from PACEvolve §3.3:
    - Crossover weights based on partner island progress
    - Backtrack weight based on dominance and stagnation
    - Synergy bonus for high-performing similar islands
    """
    
    def __init__(self, synergy_bonus_weight: float = 1.0):
        """
        Initialize the calculator.
        
        Args:
            synergy_bonus_weight: Weight for synergy bonus calculation.
        """
        self.synergy_weight = synergy_bonus_weight
    
    def calculate_absolute_progress(
        self,
        initial_score: float,
        current_score: float,
        target: float = 1.0,
    ) -> float:
        """
        Calculate Absolute Progress (Ai).
        
        Ai = (s_best - s_0) / (target - s_0)
        
        For maximization: measures how far we've progressed toward target.
        """
        initial_gap = target - initial_score
        if initial_gap <= 0:
            return 1.0  # Already at or past target
        
        improvement = current_score - initial_score
        progress = improvement / initial_gap
        
        return max(0.0, min(1.0, progress))
    
    def calculate_similarity(self, progress_i: float, progress_j: float) -> float:
        """
        Calculate similarity between two islands' progress.
        
        S = max(0, 1 - |Ai - Aj|)
        """
        return max(0.0, 1.0 - abs(progress_i - progress_j))
    
    def calculate_crossover_weight(
        self,
        triggered_progress: float,
        partner_progress: float,
    ) -> float:
        """
        Calculate base crossover weight for a partner island.
        
        w_Cj = max(0, Aj - Ai)
        
        Favors islands with higher progress.
        """
        return max(0.0, partner_progress - triggered_progress)
    
    def calculate_synergy_bonus(
        self,
        triggered_progress: float,
        partner_progress: float,
        similarity: float,
    ) -> float:
        """
        Calculate synergy bonus for high-performing similar islands.
        
        w_syn = S * Ai * Aj
        
        High when both islands have high progress and are similar.
        """
        return (
            self.synergy_weight 
            * similarity 
            * triggered_progress 
            * partner_progress
        )
    
    def calculate_backtrack_weight(
        self,
        triggered_progress: float,
        best_partner_progress: float,
        similarity: float,
    ) -> float:
        """
        Calculate backtrack weight.
        
        w_BT = w_dominance + w_stagnation
        
        - Dominance: Triggered island outperforms all others
        - Stagnation: Both islands have similar low progress
        """
        # Dominance component
        w_dominance = max(0.0, triggered_progress - best_partner_progress)
        
        # Stagnation component
        low_progress_triggered = 1.0 - triggered_progress
        low_progress_partner = 1.0 - best_partner_progress
        w_stagnation = similarity * low_progress_triggered * low_progress_partner
        
        return w_dominance + w_stagnation
    
    def calculate_all_weights(
        self,
        triggered_island_id: int,
        triggered_progress: float,
        partner_progresses: Dict[int, float],
    ) -> ActionWeights:
        """
        Calculate all action weights for a triggered island.
        
        Args:
            triggered_island_id: ID of the island that triggered intervention.
            triggered_progress: Absolute progress of triggered island.
            partner_progresses: Dict of island_id -> absolute progress for partners.
            
        Returns:
            ActionWeights with all computed weights.
        """
        weights = ActionWeights()
        
        if not partner_progresses:
            # No partners, only backtrack is possible
            weights.backtrack_weight = 1.0
            return weights
        
        # Find best partner
        best_partner_id = max(partner_progresses.keys(), key=lambda k: partner_progresses[k])
        best_partner_progress = partner_progresses[best_partner_id]
        weights.best_partner_id = best_partner_id
        
        # Calculate similarity with best partner
        similarity = self.calculate_similarity(triggered_progress, best_partner_progress)
        
        # Calculate crossover weights for each partner
        for partner_id, partner_progress in partner_progresses.items():
            base_weight = self.calculate_crossover_weight(triggered_progress, partner_progress)
            
            # Add synergy bonus for best partner
            if partner_id == best_partner_id:
                synergy = self.calculate_synergy_bonus(
                    triggered_progress, partner_progress, similarity
                )
                base_weight += synergy
            
            weights.crossover_weights[partner_id] = base_weight
        
        # Calculate backtrack weight
        weights.backtrack_weight = self.calculate_backtrack_weight(
            triggered_progress, best_partner_progress, similarity
        )
        
        logger.debug(
            f"Island {triggered_island_id} weights: "
            f"BT={weights.backtrack_weight:.3f}, "
            f"crossover={weights.crossover_weights}"
        )
        
        return weights
