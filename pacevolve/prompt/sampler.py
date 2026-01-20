"""
Prompt Sampler for OpenPACEvolve.

Samples context from idea pools and programs for prompt construction.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.context.idea_pool import Idea, IdeaPool
    from pacevolve.context.failure_memory import FailureMemory
    from pacevolve.database import Program

logger = logging.getLogger(__name__)


class PromptSampler:
    """
    Context sampler for prompt construction.
    
    Samples relevant context from:
    - Idea pools (top and diverse ideas)
    - Failure memory
    - Top programs
    """
    
    def __init__(
        self,
        num_top_ideas: int = 3,
        num_diverse_ideas: int = 2,
        num_top_programs: int = 3,
        max_artifact_bytes: int = 10000,
    ):
        """
        Initialize sampler.
        
        Args:
            num_top_ideas: Number of top-performing ideas to include.
            num_diverse_ideas: Number of diverse ideas to include.
            num_top_programs: Number of top programs to include.
            max_artifact_bytes: Maximum bytes for artifact context.
        """
        self.num_top_ideas = num_top_ideas
        self.num_diverse_ideas = num_diverse_ideas
        self.num_top_programs = num_top_programs
        self.max_artifact_bytes = max_artifact_bytes
    
    def sample_idea_context(
        self,
        idea_pool: "IdeaPool",
    ) -> str:
        """
        Sample context from idea pool.
        
        Returns formatted string with top ideas and their summaries.
        """
        ideas = idea_pool.select_ideas_for_context(
            self.num_top_ideas + self.num_diverse_ideas
        )
        
        if not ideas:
            return "No ideas explored yet."
        
        sections = []
        for i, idea in enumerate(ideas):
            label = "Top" if i < self.num_top_ideas else "Diverse"
            sections.append(f"### {label} Idea {i+1}: {idea.description}")
            sections.append(f"Best Score: {idea.best_score:.4f}")
            if idea.summary:
                sections.append(f"Summary: {idea.summary}")
            sections.append("")
        
        return "\n".join(sections)
    
    def sample_failure_context(
        self,
        failure_memory: "FailureMemory",
        max_failures: int = 5,
    ) -> str:
        """
        Sample context from failure memory.
        
        Returns formatted string with recent failures to avoid.
        """
        return failure_memory.get_failure_summary(max_failures)
    
    def sample_experiment_history(
        self,
        idea: "Idea",
        max_experiments: int = 5,
    ) -> str:
        """
        Sample experiment history for an idea.
        
        Returns formatted string with recent hypothesis results.
        """
        if not idea.hypotheses:
            return "No experiments yet for this idea."
        
        # Get recent experiments
        sorted_hyps = sorted(
            idea.hypotheses,
            key=lambda h: h.created_at,
            reverse=True
        )[:max_experiments]
        
        lines = [f"Recent experiments for '{idea.description}':"]
        for hyp in sorted_hyps:
            lines.append(f"- Gen {hyp.generation}: Score {hyp.score:.4f}")
            if hyp.artifacts:
                artifact_str = str(hyp.artifacts)[:200]
                lines.append(f"  Feedback: {artifact_str}")
        
        return "\n".join(lines)
    
    def sample_artifact_context(
        self,
        program: "Program",
    ) -> str:
        """
        Sample artifact context from a program.
        
        Returns formatted string with evaluation artifacts.
        """
        if not program.artifacts:
            return "No execution feedback available."
        
        lines = ["## Evaluation Feedback"]
        total_bytes = 0
        
        for key, value in program.artifacts.items():
            value_str = str(value)
            if total_bytes + len(value_str) > self.max_artifact_bytes:
                value_str = value_str[:self.max_artifact_bytes - total_bytes] + "..."
            
            lines.append(f"### {key}")
            lines.append(value_str)
            total_bytes += len(value_str)
            
            if total_bytes >= self.max_artifact_bytes:
                lines.append("... (truncated)")
                break
        
        return "\n".join(lines)
    
    def sample_top_programs_context(
        self,
        programs: List["Program"],
    ) -> str:
        """
        Sample context from top programs.
        
        Returns formatted string with best programs for inspiration.
        """
        if not programs:
            return "No programs available."
        
        sorted_progs = sorted(programs, key=lambda p: p.score, reverse=True)
        top = sorted_progs[:self.num_top_programs]
        
        lines = ["## Top Programs for Inspiration"]
        for i, prog in enumerate(top):
            lines.append(f"### Program {i+1} (Score: {prog.score:.4f})")
            # Include code snippet
            code_preview = prog.code[:500] + "..." if len(prog.code) > 500 else prog.code
            lines.append(f"```python\n{code_preview}\n```")
        
        return "\n".join(lines)
