"""
Idea Pool for Hierarchical Context Management.

Maintains a persistent pool of macro-level conceptual ideas with
LLM-based classification for deduplication.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A micro-level experimental hypothesis (specific code implementation)."""
    id: str
    idea_id: str
    code: str
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    
    @property
    def score(self) -> float:
        """Get the combined score of this hypothesis."""
        return self.metrics.get("combined_score", 0.0)


@dataclass
class Idea:
    """
    A macro-level conceptual idea in the evolution search.
    
    Ideas represent high-level approaches (e.g., "Use simulated annealing")
    while hypotheses are specific implementations of those ideas.
    """
    id: str
    description: str
    summary: str = ""  # Summarized findings from experiments
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best_score: float = 0.0
    experiment_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    is_active: bool = True
    pruned_reason: Optional[str] = None
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to this idea."""
        self.hypotheses.append(hypothesis)
        self.experiment_count += 1
        self.last_updated = time.time()
        
        # Update best score
        if hypothesis.score > self.best_score:
            self.best_score = hypothesis.score
    
    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        """Get the best-performing hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.score)
    
    def to_context_string(self, max_hypotheses: int = 3) -> str:
        """Convert idea to context string for prompts."""
        lines = [f"## Idea: {self.description}"]
        
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        
        lines.append(f"Best Score: {self.best_score:.4f}")
        lines.append(f"Experiments: {self.experiment_count}")
        
        # Include top hypotheses
        if self.hypotheses:
            sorted_hyps = sorted(self.hypotheses, key=lambda h: h.score, reverse=True)
            lines.append("\nTop Experiments:")
            for i, hyp in enumerate(sorted_hyps[:max_hypotheses]):
                lines.append(f"  {i+1}. Score: {hyp.score:.4f}")
        
        return "\n".join(lines)


class IdeaPool:
    """
    Persistent pool of macro-level conceptual ideas.
    
    Implements the Idea component of Hierarchical Context Management:
    - Decomposes high-level ideas from specific solutions
    - Uses LLM-based classification to merge similar ideas
    - Maintains a knowledge base of explored approaches
    """
    
    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        max_ideas: int = 20,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the idea pool.
        
        Args:
            llm: LLM for idea classification.
            max_ideas: Maximum ideas to maintain.
            similarity_threshold: Threshold for merging similar ideas.
        """
        self.llm = llm
        self.max_ideas = max_ideas
        self.similarity_threshold = similarity_threshold
        
        self.ideas: Dict[str, Idea] = {}
        self._idea_order: List[str] = []  # Track insertion order
    
    @property
    def active_ideas(self) -> List[Idea]:
        """Get list of active (non-pruned) ideas."""
        return [idea for idea in self.ideas.values() if idea.is_active]
    
    @property
    def all_ideas(self) -> List[Idea]:
        """Get all ideas including pruned ones."""
        return list(self.ideas.values())
    
    def add_idea(self, description: str) -> Idea:
        """
        Add a new idea to the pool.
        
        Args:
            description: Description of the conceptual idea.
            
        Returns:
            The created or merged Idea.
        """
        # Check if idea already exists (exact match)
        for existing in self.active_ideas:
            if existing.description.lower() == description.lower():
                logger.debug(f"Exact match found for idea: {description[:50]}...")
                return existing
        
        # Create new idea
        idea = Idea(
            id=str(uuid.uuid4()),
            description=description,
        )
        
        self.ideas[idea.id] = idea
        self._idea_order.append(idea.id)
        
        logger.info(f"Added new idea: {description[:50]}...")
        return idea
    
    async def add_or_merge_idea(self, description: str) -> Idea:
        """
        Add a new idea or merge with existing similar idea using LLM.
        
        Args:
            description: Description of the conceptual idea.
            
        Returns:
            The created or merged Idea.
        """
        if not self.llm or not self.active_ideas:
            return self.add_idea(description)
        
        # Use LLM to classify if idea is conceptually similar to existing
        existing_idea = await self._classify_idea(description)
        
        if existing_idea:
            logger.info(f"Merged idea with existing: {existing_idea.description[:50]}...")
            return existing_idea
        
        return self.add_idea(description)
    
    async def _classify_idea(self, new_description: str) -> Optional[Idea]:
        """
        Use LLM to check if new idea is conceptually similar to existing.
        
        Returns the matching idea if found, None otherwise.
        """
        if not self.llm:
            return None
        
        active = self.active_ideas
        if not active:
            return None
        
        # Build classification prompt
        idea_list = "\n".join([
            f"{i+1}. {idea.description}"
            for i, idea in enumerate(active)
        ])
        
        prompt = f"""Given a new idea and a list of existing ideas, determine if the new idea is conceptually equivalent to any existing idea.

New Idea: {new_description}

Existing Ideas:
{idea_list}

If the new idea is conceptually the same as an existing idea (just with different wording or minor details), respond with the number of the matching idea.
If the new idea is genuinely novel, respond with "0" (zero).

Respond with only a single number."""

        from pacevolve.llm.base import Message
        
        try:
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content="You are an expert at classifying conceptual ideas. Be concise."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=10,
            )
            
            # Parse response
            content = response.content.strip()
            try:
                match_idx = int(content)
                if 1 <= match_idx <= len(active):
                    return active[match_idx - 1]
            except ValueError:
                pass
                
        except Exception as e:
            logger.warning(f"Idea classification failed: {e}")
        
        return None
    
    def select_ideas_for_context(self, num_ideas: int = 5) -> List[Idea]:
        """
        Select ideas to include in the generation context.
        
        Balances between:
        - Top-performing ideas (exploitation)
        - Less-explored ideas (exploration)
        - Recently active ideas
        
        Args:
            num_ideas: Number of ideas to select.
            
        Returns:
            List of selected ideas.
        """
        active = self.active_ideas
        if len(active) <= num_ideas:
            return active
        
        # Score each idea for selection
        scored_ideas = []
        for idea in active:
            # Combine performance, exploration potential, and recency
            performance_score = idea.best_score
            exploration_score = 1.0 / (1.0 + idea.experiment_count)
            recency_score = 1.0 / (1.0 + (time.time() - idea.last_updated) / 3600)
            
            combined = 0.5 * performance_score + 0.3 * exploration_score + 0.2 * recency_score
            scored_ideas.append((combined, idea))
        
        # Sort and select top ideas
        scored_ideas.sort(key=lambda x: x[0], reverse=True)
        return [idea for _, idea in scored_ideas[:num_ideas]]
    
    def get_idea(self, idea_id: str) -> Optional[Idea]:
        """Get an idea by ID."""
        return self.ideas.get(idea_id)
    
    def prune_idea(self, idea_id: str, reason: str) -> None:
        """Mark an idea as pruned."""
        idea = self.ideas.get(idea_id)
        if idea:
            idea.is_active = False
            idea.pruned_reason = reason
            logger.info(f"Pruned idea: {idea.description[:50]}... Reason: {reason}")
    
    def to_context_string(self, max_ideas: int = 5) -> str:
        """Convert the idea pool to a context string for prompts."""
        selected = self.select_ideas_for_context(max_ideas)
        
        if not selected:
            return "No ideas explored yet."
        
        sections = [idea.to_context_string() for idea in selected]
        return "\n\n".join(sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ideas": {
                idea_id: {
                    "id": idea.id,
                    "description": idea.description,
                    "summary": idea.summary,
                    "best_score": idea.best_score,
                    "experiment_count": idea.experiment_count,
                    "is_active": idea.is_active,
                    "pruned_reason": idea.pruned_reason,
                }
                for idea_id, idea in self.ideas.items()
            },
            "idea_order": self._idea_order,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm: Optional["BaseLLM"] = None) -> "IdeaPool":
        """Deserialize from dictionary."""
        pool = cls(llm=llm)
        
        for idea_data in data.get("ideas", {}).values():
            idea = Idea(
                id=idea_data["id"],
                description=idea_data["description"],
                summary=idea_data.get("summary", ""),
                best_score=idea_data.get("best_score", 0.0),
                experiment_count=idea_data.get("experiment_count", 0),
                is_active=idea_data.get("is_active", True),
                pruned_reason=idea_data.get("pruned_reason"),
            )
            pool.ideas[idea.id] = idea
        
        pool._idea_order = data.get("idea_order", list(pool.ideas.keys()))
        return pool
