"""
Crossover Sampler for Self-Adaptive Crossover Sampling.

Implements probabilistic action selection and crossover execution.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from pacevolve.sampling.action_weights import ActionWeights

if TYPE_CHECKING:
    from pacevolve.context.idea_pool import IdeaPool, Idea

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions in self-adaptive sampling."""
    BACKTRACK = auto()
    CROSSOVER = auto()
    CONTINUE = auto()  # No intervention needed


@dataclass
class Action:
    """Selected action with parameters."""
    action_type: ActionType
    partner_island_id: Optional[int] = None
    
    def __str__(self) -> str:
        if self.action_type == ActionType.BACKTRACK:
            return "BACKTRACK"
        elif self.action_type == ActionType.CROSSOVER:
            return f"CROSSOVER with island {self.partner_island_id}"
        else:
            return "CONTINUE"


class CrossoverSampler:
    """
    Self-adaptive action selection for PACEvolve.
    
    Implements:
    - Probabilistic selection based on action weights
    - Crossover between islands (sharing best ideas)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the sampler.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = random.Random(seed)
    
    def sample_action(self, weights: ActionWeights) -> Action:
        """
        Sample an action based on computed weights.
        
        P(a) = w_a / Σw
        
        Args:
            weights: Computed action weights.
            
        Returns:
            Selected action.
        """
        total = weights.total_weight
        
        if total <= 0:
            # No valid actions, default to continue
            return Action(action_type=ActionType.CONTINUE)
        
        # Build action list with probabilities
        actions = []
        probs = []
        
        # Backtrack action
        if weights.backtrack_weight > 0:
            actions.append(Action(action_type=ActionType.BACKTRACK))
            probs.append(weights.backtrack_weight / total)
        
        # Crossover actions
        for island_id, weight in weights.crossover_weights.items():
            if weight > 0:
                actions.append(Action(
                    action_type=ActionType.CROSSOVER,
                    partner_island_id=island_id
                ))
                probs.append(weight / total)
        
        if not actions:
            return Action(action_type=ActionType.CONTINUE)
        
        # Sample action
        selected = self.rng.choices(actions, probs)[0]
        
        logger.info(f"Selected action: {selected}")
        return selected
    
    def execute_crossover(
        self,
        source_pool: "IdeaPool",
        target_pool: "IdeaPool",
        num_ideas: int = 1,
    ) -> int:
        """
        Execute crossover by sharing ideas between islands.
        
        Copies top ideas from source island to target island.
        
        Args:
            source_pool: Idea pool of source (partner) island.
            target_pool: Idea pool of target (triggered) island.
            num_ideas: Number of ideas to transfer.
            
        Returns:
            Number of ideas transferred.
        """
        # Get top ideas from source
        source_ideas = sorted(
            source_pool.active_ideas,
            key=lambda i: i.best_score,
            reverse=True
        )[:num_ideas]
        
        transferred = 0
        for idea in source_ideas:
            # Check if target already has similar idea
            existing = None
            for target_idea in target_pool.active_ideas:
                if target_idea.description.lower() == idea.description.lower():
                    existing = target_idea
                    break
            
            if existing:
                # Update existing idea if source has better score
                if idea.best_score > existing.best_score:
                    existing.summary = (
                        f"{existing.summary}\n"
                        f"[Crossover] Partner achieved {idea.best_score:.4f}"
                    )
                    logger.debug(f"Updated idea from crossover: {idea.description[:50]}...")
            else:
                # Add new idea
                new_idea = target_pool.add_idea(idea.description)
                new_idea.summary = f"[Crossover] Imported from partner (score: {idea.best_score:.4f})"
                transferred += 1
                logger.info(f"Transferred idea: {idea.description[:50]}...")
        
        return transferred
    
    def handle_action(
        self,
        action: Action,
        island_pools: dict,  # island_id -> IdeaPool
        triggered_island_id: int,
    ) -> bool:
        """
        Handle the selected action.
        
        Args:
            action: The selected action.
            island_pools: Dict of island_id -> IdeaPool.
            triggered_island_id: ID of the triggered island.
            
        Returns:
            True if action was executed.
        """
        if action.action_type == ActionType.CONTINUE:
            return True
        
        if action.action_type == ActionType.BACKTRACK:
            # Backtracking is handled by BacktrackingManager
            return True
        
        if action.action_type == ActionType.CROSSOVER:
            if action.partner_island_id is None:
                return False
            
            source_pool = island_pools.get(action.partner_island_id)
            target_pool = island_pools.get(triggered_island_id)
            
            if source_pool is None or target_pool is None:
                logger.warning(f"Could not find pools for crossover")
                return False
            
            transferred = self.execute_crossover(source_pool, target_pool)
            logger.info(
                f"Crossover: transferred {transferred} ideas "
                f"from island {action.partner_island_id} to {triggered_island_id}"
            )
            return True
        
        return False
