"""
Prompt Templates for OpenPACEvolve.

Provides default templates with stochastic variations.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_SYSTEM_MESSAGE = """You are an expert programmer specializing in code optimization and algorithm design.
Your task is to improve programs through evolutionary search, generating new variations that achieve better performance.

Key principles:
1. Understand the current approach and its limitations
2. Consider alternative algorithms or data structures
3. Make targeted improvements based on evaluation feedback
4. Ensure code correctness while optimizing for the target metrics"""

DEFAULT_IDEA_GENERATION = """Given the current state of the evolutionary search, propose a new high-level idea to explore.

Current Best Ideas:
{idea_context}

Failure History:
{failure_context}

Current Best Score: {best_score}

Propose a novel, high-level approach that:
1. Is conceptually different from existing ideas
2. Has potential to improve upon current best
3. Avoids known failed approaches

Respond with a single sentence describing the high-level idea."""

DEFAULT_CODE_EVOLUTION = """Improve the following code based on the evaluation feedback and high-level idea.

## Current Code
```python
{current_code}
```

## High-Level Idea to Pursue
{idea}

## Recent Experiment History
{experiment_history}

## Evaluation Feedback
{artifacts}

## Target
Optimize for: {optimization_target}
Current Score: {current_score}
Target Score: {target_score}

Generate an improved version of the code that implements the suggested idea.
Return ONLY the complete improved code, no explanations."""


class PromptTemplates:
    """
    Prompt templates with stochastic variations.
    
    Supports loading custom templates from directory or using defaults.
    """
    
    def __init__(
        self,
        template_dir: Optional[str] = None,
        system_message: Optional[str] = None,
        use_stochasticity: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize templates.
        
        Args:
            template_dir: Directory with custom template files.
            system_message: Custom system message.
            use_stochasticity: Enable template variations.
            seed: Random seed.
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.use_stochasticity = use_stochasticity
        self.rng = random.Random(seed)
        
        # Load templates
        self.system_message = system_message or self._load_template(
            "system_message.txt", DEFAULT_SYSTEM_MESSAGE
        )
        self.idea_generation = self._load_template(
            "idea_generation.txt", DEFAULT_IDEA_GENERATION
        )
        self.code_evolution = self._load_template(
            "code_evolution.txt", DEFAULT_CODE_EVOLUTION
        )
        
        # Stochastic variations
        self.variations: Dict[str, List[str]] = {
            "greeting": [
                "Let's improve this code:",
                "Here's an opportunity to enhance:",
                "Consider this improvement:",
                "Time to optimize:",
            ],
            "encouragement": [
                "You can do better!",
                "There's room for improvement.",
                "Let's push the boundaries.",
                "Aim higher!",
            ],
            "approach_suggestion": [
                "Consider trying",
                "You might explore",
                "One approach is",
                "Think about",
            ],
        }
    
    def _load_template(self, filename: str, default: str) -> str:
        """Load template from file or use default."""
        if self.template_dir:
            path = self.template_dir / filename
            if path.exists():
                return path.read_text()
        return default
    
    def _apply_variations(self, template: str) -> str:
        """Apply stochastic variations to template."""
        if not self.use_stochasticity:
            return template
        
        for key, options in self.variations.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                choice = self.rng.choice(options)
                template = template.replace(placeholder, choice)
        
        return template
    
    def get_system_message(self) -> str:
        """Get system message with variations."""
        return self._apply_variations(self.system_message)
    
    def get_idea_generation_prompt(
        self,
        idea_context: str,
        failure_context: str,
        best_score: float,
    ) -> str:
        """Get idea generation prompt."""
        prompt = self.idea_generation.format(
            idea_context=idea_context,
            failure_context=failure_context,
            best_score=f"{best_score:.4f}",
        )
        return self._apply_variations(prompt)
    
    def get_code_evolution_prompt(
        self,
        current_code: str,
        idea: str,
        experiment_history: str,
        artifacts: str,
        optimization_target: str,
        current_score: float,
        target_score: float = 1.0,
    ) -> str:
        """Get code evolution prompt."""
        prompt = self.code_evolution.format(
            current_code=current_code,
            idea=idea,
            experiment_history=experiment_history,
            artifacts=artifacts,
            optimization_target=optimization_target,
            current_score=f"{current_score:.4f}",
            target_score=f"{target_score:.4f}",
        )
        return self._apply_variations(prompt)
