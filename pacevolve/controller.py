"""
Main Controller for OpenPACEvolve.

Orchestrates the evolution process with PACEvolve algorithm.
"""

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from pacevolve.config import Config, load_config
from pacevolve.context.idea_pool import Idea
from pacevolve.context.hypothesis import Hypothesis, HypothesisManager
from pacevolve.context.pruning import ContextPruner
from pacevolve.database import Database, Island, Program
from pacevolve.evaluator import Evaluator
from pacevolve.llm.ensemble import EnsembleLLM
from pacevolve.momentum.backtracking import BacktrackingManager
from pacevolve.sampling.action_weights import ActionWeightCalculator
from pacevolve.sampling.crossover import CrossoverSampler, ActionType
from pacevolve.prompt.templates import PromptTemplates
from pacevolve.prompt.sampler import PromptSampler

logger = logging.getLogger(__name__)


class Controller:
    """
    Main evolution controller implementing PACEvolve algorithm.
    
    Orchestrates:
    - Multi-island evolution
    - Hierarchical Context Management
    - Momentum-Based Backtracking
    - Self-Adaptive Crossover Sampling
    """
    
    def __init__(
        self,
        initial_program_path: str,
        evaluator_path: str,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the controller.
        
        Args:
            initial_program_path: Path to initial program.
            evaluator_path: Path to evaluator module.
            config: Configuration object.
            config_path: Path to config YAML.
            output_dir: Output directory for checkpoints.
        """
        # Load configuration
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        self.initial_program_path = initial_program_path
        self.evaluator_path = evaluator_path
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM
        self.llm = EnsembleLLM(config.llm, seed=config.random_seed)
        
        # Initialize database
        self.database = Database(
            num_islands=config.database.num_islands,
            population_size=config.database.population_size,
            migration_interval=config.database.migration_interval,
            migration_rate=config.database.migration_rate,
            llm=self.llm,
            seed=config.random_seed,
        )
        
        # Initialize evaluator
        self.evaluator = Evaluator(
            evaluator_path=evaluator_path,
            config=config.evaluator,
        )
        
        # Initialize HCM components
        self.hypothesis_manager = HypothesisManager(
            llm=self.llm,
            max_hypotheses_per_idea=config.context.max_hypotheses_per_idea,
            summarization_trigger=config.context.hypothesis_summarization_trigger,
        )
        self.context_pruner = ContextPruner(
            llm=self.llm,
            idea_cap=config.context.idea_cap,
            min_experiments_before_prune=config.context.min_experiments_before_prune,
        )
        
        # Initialize MBB components
        self.backtracking_manager = BacktrackingManager(
            power_law_alpha=config.momentum.power_law_alpha,
            min_backtrack_generations=config.momentum.min_backtrack_generations,
            seed=config.random_seed,
        )
        
        # Initialize sampling components
        self.weight_calculator = ActionWeightCalculator(
            synergy_bonus_weight=config.sampling.synergy_bonus_weight,
        )
        self.crossover_sampler = CrossoverSampler(seed=config.random_seed)
        
        # Initialize prompt components
        self.prompt_templates = PromptTemplates(
            template_dir=config.prompt.template_dir,
            system_message=config.prompt.system_message,
            use_stochasticity=config.prompt.use_template_stochasticity,
            seed=config.random_seed,
        )
        self.prompt_sampler = PromptSampler(
            num_top_ideas=config.prompt.num_top_programs,
            num_diverse_ideas=config.prompt.num_diverse_programs,
            max_artifact_bytes=config.prompt.max_artifact_bytes,
        )
        
        # Load initial program
        self.initial_code = Path(initial_program_path).read_text()
        
        # Tracking
        self.current_iteration = 0
        self.best_score = 0.0
        self.start_time: Optional[float] = None
    
    async def run(self, max_iterations: Optional[int] = None) -> Program:
        """
        Run the evolution process.
        
        Args:
            max_iterations: Maximum iterations (overrides config).
            
        Returns:
            Best program found.
        """
        max_iter = max_iterations or self.config.max_iterations
        self.start_time = time.time()
        
        logger.info(f"Starting evolution for {max_iter} iterations")
        
        # Initialize islands with initial program
        await self._initialize_islands()
        
        # Main evolution loop
        for iteration in range(max_iter):
            self.current_iteration = iteration
            self.database.total_iterations = iteration
            
            logger.info(f"=== Iteration {iteration + 1}/{max_iter} ===")
            
            # Evolve each island
            for island in self.database.islands:
                await self._evolve_island(island, iteration)
            
            # Check migration
            if self.database.should_migrate(iteration):
                self.database.migrate()
            
            # Checkpoint
            if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(iteration)
            
            # Log progress
            best = self.database.get_global_best()
            if best:
                self.best_score = best.score
                logger.info(f"Best score: {self.best_score:.4f}")
        
        # Final checkpoint
        self._save_checkpoint(max_iter)
        
        elapsed = time.time() - self.start_time
        logger.info(f"Evolution complete in {elapsed:.1f}s. Best: {self.best_score:.4f}")
        
        return self.database.get_global_best()
    
    async def _initialize_islands(self) -> None:
        """Initialize all islands with the initial program."""
        # Evaluate initial program
        program_path = self._save_program(self.initial_code, "initial")
        result = await self.evaluator.evaluate(program_path)
        
        initial_score = result.combined_score
        logger.info(f"Initial program score: {initial_score:.4f}")
        
        # Add to each island
        for island in self.database.islands:
            program = Program(
                id=str(uuid.uuid4()),
                code=self.initial_code,
                metrics=result.metrics,
                artifacts=result.artifacts,
                generation=0,
                island_id=island.id,
            )
            island.add_program(program)
            island.progress_tracker.initialize(initial_score)
            
            # Add initial idea
            idea = island.idea_pool.add_idea("Initial Approach")
            hypothesis = self.hypothesis_manager.create_hypothesis(
                idea_id=idea.id,
                code=self.initial_code,
                metrics=result.metrics,
                artifacts=result.artifacts,
                generation=0,
            )
            idea.add_hypothesis(hypothesis)
    
    async def _evolve_island(self, island: Island, iteration: int) -> None:
        """Evolve a single island for one iteration."""
        island.generation = iteration
        
        # Check for momentum-based intervention
        if island.progress_tracker.should_intervene():
            await self._handle_intervention(island)
        
        # Generate or select idea
        idea = await self._get_idea_for_evolution(island)
        
        # Get parent program
        parent = island.best_program or island.programs[-1] if island.programs else None
        if not parent:
            return
        
        # Generate new code
        new_code = await self._generate_code(island, idea, parent)
        if not new_code:
            return
        
        # Evaluate
        program_path = self._save_program(new_code, f"island{island.id}_gen{iteration}")
        result = await self.evaluator.cascade_evaluate(program_path)
        
        # Create program and hypothesis
        program = Program(
            id=str(uuid.uuid4()),
            code=new_code,
            metrics=result.metrics,
            artifacts=result.artifacts,
            generation=iteration,
            island_id=island.id,
            parent_id=parent.id,
            idea_id=idea.id,
        )
        
        hypothesis = self.hypothesis_manager.create_hypothesis(
            idea_id=idea.id,
            code=new_code,
            metrics=result.metrics,
            artifacts=result.artifacts,
            generation=iteration,
            parent_id=parent.id,
        )
        
        # Add to island and idea
        self.database.add_program(program, island.id)
        await self.hypothesis_manager.add_hypothesis_to_idea(idea, hypothesis)
        
        # Prune context if needed
        await self.context_pruner.prune_idea_pool(
            island.idea_pool,
            self.database.failure_memory,
            self.best_score,
        )
        
        logger.debug(
            f"Island {island.id}: score={result.combined_score:.4f}, "
            f"idea='{idea.description[:30]}...'"
        )
    
    async def _handle_intervention(self, island: Island) -> None:
        """Handle momentum-triggered intervention on an island."""
        # Calculate weights
        island_progresses = self.database.get_island_progresses()
        triggered_progress = island_progresses.pop(island.id, 0.0)
        
        weights = self.weight_calculator.calculate_all_weights(
            triggered_island_id=island.id,
            triggered_progress=triggered_progress,
            partner_progresses=island_progresses,
        )
        
        # Sample action
        action = self.crossover_sampler.sample_action(weights)
        
        if action.action_type == ActionType.BACKTRACK:
            self.backtracking_manager.handle_intervention(island.progress_tracker)
        elif action.action_type == ActionType.CROSSOVER:
            if action.partner_island_id is not None:
                pool_map = {i.id: i.idea_pool for i in self.database.islands}
                self.crossover_sampler.handle_action(action, pool_map, island.id)
    
    async def _get_idea_for_evolution(self, island: Island) -> Idea:
        """Get or generate an idea for evolution."""
        # Try to select existing idea with potential
        selected = island.idea_pool.select_ideas_for_context(1)
        if selected and selected[0].experiment_count < 5:
            return selected[0]
        
        # Generate new idea
        idea_context = self.prompt_sampler.sample_idea_context(island.idea_pool)
        failure_context = self.prompt_sampler.sample_failure_context(
            self.database.failure_memory
        )
        
        prompt = self.prompt_templates.get_idea_generation_prompt(
            idea_context=idea_context,
            failure_context=failure_context,
            best_score=self.best_score,
        )
        
        try:
            from pacevolve.llm.base import Message
            response = await self.llm.generate(
                messages=[
                    Message(role="system", content=self.prompt_templates.system_message),
                    Message(role="user", content=prompt),
                ],
                temperature=0.9,
                max_tokens=200,
            )
            
            new_idea_desc = response.content.strip()
            return await island.idea_pool.add_or_merge_idea(new_idea_desc)
            
        except Exception as e:
            logger.warning(f"Idea generation failed: {e}")
            # Fall back to existing idea
            if selected:
                return selected[0]
            return island.idea_pool.add_idea("General Improvement")
    
    async def _generate_code(
        self,
        island: Island,
        idea: Idea,
        parent: Program,
    ) -> Optional[str]:
        """Generate new code variation."""
        # Build prompt
        experiment_history = self.prompt_sampler.sample_experiment_history(idea)
        artifacts = self.prompt_sampler.sample_artifact_context(parent)
        
        prompt = self.prompt_templates.get_code_evolution_prompt(
            current_code=parent.code,
            idea=idea.description,
            experiment_history=experiment_history,
            artifacts=artifacts,
            optimization_target="combined_score",
            current_score=parent.score,
        )
        
        try:
            new_code = await self.llm.generate_code(
                prompt=prompt,
                system_message=self.prompt_templates.get_system_message(),
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            
            # Validate code length
            if len(new_code) > self.config.max_code_length:
                logger.warning(f"Generated code too long: {len(new_code)} chars")
                return None
            
            return new_code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def _save_program(self, code: str, name: str) -> str:
        """Save program to file and return path."""
        path = self.output_dir / "programs" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        return str(path)
    
    def _save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint."""
        path = self.output_dir / "checkpoints" / f"checkpoint_{iteration}" / "database.json"
        self.database.save_checkpoint(str(path))
        
        # Save best program
        best = self.database.get_global_best()
        if best:
            best_path = self.output_dir / "best_program.py"
            best_path.write_text(best.code)
