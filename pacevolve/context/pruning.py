"""
Context Pruning for Hierarchical Context Management.

Implements bi-level pruning strategy for ideas and hypotheses.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.llm.base import BaseLLM
    from pacevolve.context.idea_pool import Idea, IdeaPool
    from pacevolve.context.failure_memory import FailureMemory

logger = logging.getLogger(__name__)


class ContextPruner:
    """
    Bi-level context pruning for Hierarchical Context Management.
    
    Implements:
    1. Hypothesis-level pruning: Compress experiment history per idea
    2. Idea-level pruning: Eliminate low-performing ideas
    """
    
    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        idea_cap: int = 10,
        min_experiments_before_prune: int = 3,
        low_performing_threshold: float = 0.3,
    ):
        """
        Initialize the pruner.
        
        Args:
            llm: LLM for intelligent pruning decisions.
            idea_cap: Maximum active ideas to maintain.
            min_experiments_before_prune: Minimum experiments before pruning an idea.
            low_performing_threshold: Threshold below which ideas are candidates for pruning.
        """
        self.llm = llm
        self.idea_cap = idea_cap
        self.min_experiments = min_experiments_before_prune
        self.low_threshold = low_performing_threshold
    
    async def prune_idea_pool(
        self,
        idea_pool: "IdeaPool",
        failure_memory: "FailureMemory",
        best_score: float,
    ) -> List[str]:
        """
        Prune ideas from the pool when over capacity.
        
        Args:
            idea_pool: The idea pool to prune.
            failure_memory: Memory to record pruned failures.
            best_score: Current best score for comparison.
            
        Returns:
            List of pruned idea IDs.
        """
        active_ideas = idea_pool.active_ideas
        
        if len(active_ideas) <= self.idea_cap:
            return []
        
        # Identify candidates for pruning
        candidates = []
        for idea in active_ideas:
            if idea.experiment_count < self.min_experiments:
                continue  # Too early to judge
            
            # Check if significantly underperforming
            relative_score = idea.best_score / best_score if best_score > 0 else 0
            if relative_score < self.low_threshold:
                candidates.append(idea)
        
        if not candidates:
            # No clear underperformers, use LLM to decide
            if self.llm:
                candidates = await self._llm_select_for_pruning(
                    active_ideas,
                    len(active_ideas) - self.idea_cap,
                )
            else:
                # Simple heuristic: prune oldest low performers
                sorted_ideas = sorted(
                    active_ideas,
                    key=lambda i: (i.best_score, -i.experiment_count)
                )
                candidates = sorted_ideas[:len(active_ideas) - self.idea_cap]
        
        # Prune candidates
        pruned_ids = []
        for idea in candidates[:len(active_ideas) - self.idea_cap]:
            reason = f"Low performance ({idea.best_score:.4f}) after {idea.experiment_count} experiments"
            idea_pool.prune_idea(idea.id, reason)
            
            # Record in failure memory
            failure_memory.record_failure(
                description=idea.description,
                reason=reason,
                metrics={"best_score": idea.best_score},
                idea_id=idea.id,
            )
            
            pruned_ids.append(idea.id)
        
        return pruned_ids
    
    async def _llm_select_for_pruning(
        self,
        ideas: List["Idea"],
        num_to_prune: int,
    ) -> List["Idea"]:
        """Use LLM to select ideas for pruning."""
        if not self.llm or num_to_prune <= 0:
            return []
        
        # Build prompt
        idea_descriptions = []
        for i, idea in enumerate(ideas):
            idea_descriptions.append(
                f"{i+1}. \"{idea.description}\" - "
                f"Best: {idea.best_score:.4f}, Experiments: {idea.experiment_count}"
            )
        
        ideas_str = "\n".join(idea_descriptions)
        
        prompt = f"""We need to reduce the number of active ideas from {len(ideas)} to {len(ideas) - num_to_prune}.

Current Ideas:
{ideas_str}

Select {num_to_prune} idea(s) to drop based on:
1. Has the idea been thoroughly investigated with low results?
2. Is there potential for breakthrough with more experiments?
3. Prioritize dropping ideas that are old, lack potential, or extensively explored without improvement.

Respond with ONLY the numbers of ideas to drop, separated by commas (e.g., "2, 5")."""

        from pacevolve.llm.base import Message
        
        try:
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content="You are a research advisor helping prioritize experiments."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.3,
                max_tokens=50,
            )
            
            # Parse response
            content = response.content.strip()
            indices = []
            for part in content.split(","):
                try:
                    idx = int(part.strip()) - 1
                    if 0 <= idx < len(ideas):
                        indices.append(idx)
                except ValueError:
                    continue
            
            return [ideas[i] for i in indices[:num_to_prune]]
            
        except Exception as e:
            logger.warning(f"LLM pruning selection failed: {e}")
            return []
