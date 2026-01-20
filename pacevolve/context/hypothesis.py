"""
Hypothesis management for Hierarchical Context Management.

Manages micro-level experimental hypotheses with summarization.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.llm.base import BaseLLM
    from pacevolve.context.idea_pool import Idea

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """
    A micro-level experimental hypothesis.
    
    Represents a specific code implementation that tests a high-level idea.
    """
    id: str
    idea_id: str
    code: str
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    
    @property
    def score(self) -> float:
        """Get the combined score."""
        return self.metrics.get("combined_score", 0.0)
    
    def to_summary_string(self) -> str:
        """Convert to summary string for context compression."""
        metrics_str = ", ".join(f"{k}: {v:.3f}" for k, v in self.metrics.items())
        return f"Gen {self.generation} | Score: {self.score:.4f} | {metrics_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "idea_id": self.idea_id,
            "code": self.code,
            "metrics": self.metrics,
            "artifacts": {k: str(v)[:1000] for k, v in self.artifacts.items()},
            "generation": self.generation,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            idea_id=data["idea_id"],
            code=data["code"],
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            generation=data.get("generation", 0),
            created_at=data.get("created_at", time.time()),
            parent_id=data.get("parent_id"),
        )


class HypothesisManager:
    """
    Manages hypotheses for Hierarchical Context Management.
    
    Handles:
    - Adding hypotheses to ideas
    - Summarization when cap is reached
    - Pruning low-performing hypotheses
    """
    
    def __init__(
        self,
        llm: Optional["BaseLLM"] = None,
        max_hypotheses_per_idea: int = 10,
        summarization_trigger: int = 8,
    ):
        """
        Initialize the hypothesis manager.
        
        Args:
            llm: LLM for summarization.
            max_hypotheses_per_idea: Maximum hypotheses per idea.
            summarization_trigger: Trigger summarization at this count.
        """
        self.llm = llm
        self.max_hypotheses = max_hypotheses_per_idea
        self.summarization_trigger = summarization_trigger
    
    def create_hypothesis(
        self,
        idea_id: str,
        code: str,
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None,
        generation: int = 0,
        parent_id: Optional[str] = None,
    ) -> Hypothesis:
        """Create a new hypothesis."""
        return Hypothesis(
            id=str(uuid.uuid4()),
            idea_id=idea_id,
            code=code,
            metrics=metrics,
            artifacts=artifacts or {},
            generation=generation,
            parent_id=parent_id,
        )
    
    async def add_hypothesis_to_idea(
        self,
        idea: "Idea",
        hypothesis: Hypothesis,
    ) -> None:
        """
        Add a hypothesis to an idea, triggering summarization if needed.
        
        Args:
            idea: The idea to add the hypothesis to.
            hypothesis: The hypothesis to add.
        """
        idea.add_hypothesis(hypothesis)
        
        # Check if we need to summarize
        if len(idea.hypotheses) >= self.summarization_trigger:
            await self.summarize_and_prune(idea)
    
    async def summarize_and_prune(self, idea: "Idea") -> None:
        """
        Summarize hypotheses and prune to maintain cap.
        
        Distills accumulated experiment histories into key findings.
        """
        if len(idea.hypotheses) <= self.max_hypotheses // 2:
            return  # Not enough to prune
        
        # Generate summary using LLM if available
        if self.llm:
            idea.summary = await self._generate_summary(idea)
        else:
            idea.summary = self._generate_simple_summary(idea)
        
        # Keep only top-performing hypotheses
        sorted_hyps = sorted(idea.hypotheses, key=lambda h: h.score, reverse=True)
        keep_count = max(3, self.max_hypotheses // 2)
        
        idea.hypotheses = sorted_hyps[:keep_count]
        
        logger.info(
            f"Summarized idea '{idea.description[:30]}...' - "
            f"pruned to {len(idea.hypotheses)} hypotheses"
        )
    
    async def _generate_summary(self, idea: "Idea") -> str:
        """Generate LLM-based summary of hypotheses."""
        if not self.llm:
            return self._generate_simple_summary(idea)
        
        # Build prompt with hypothesis details
        hyp_details = []
        for hyp in idea.hypotheses:
            details = [
                f"Score: {hyp.score:.4f}",
                f"Metrics: {hyp.metrics}",
            ]
            if hyp.artifacts:
                details.append(f"Feedback: {list(hyp.artifacts.values())[:2]}")
            hyp_details.append(" | ".join(details))
        
        hyp_list = "\n".join(f"- {d}" for d in hyp_details)
        
        prompt = f"""Summarize the key findings from these experiments testing the idea: "{idea.description}"

Experiments:
{hyp_list}

Provide a concise 2-3 sentence summary of:
1. What approaches work well
2. What approaches don't work
3. Recommended next directions

Keep it brief and actionable."""

        from pacevolve.llm.base import Message
        
        try:
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content="You are a research assistant summarizing experimental results."),
                    Message(role="user", content=prompt),
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return self._generate_simple_summary(idea)
    
    def _generate_simple_summary(self, idea: "Idea") -> str:
        """Generate simple summary without LLM."""
        if not idea.hypotheses:
            return "No experiments yet."
        
        best = max(idea.hypotheses, key=lambda h: h.score)
        worst = min(idea.hypotheses, key=lambda h: h.score)
        
        return (
            f"Tested {len(idea.hypotheses)} variations. "
            f"Best: {best.score:.4f}, Worst: {worst.score:.4f}. "
            f"Average: {sum(h.score for h in idea.hypotheses) / len(idea.hypotheses):.4f}"
        )
