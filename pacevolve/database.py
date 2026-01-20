"""
Island-based Database for OpenPACEvolve.

Implements multi-island evolution with MAP-Elites quality-diversity.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pacevolve.llm.base import BaseLLM

from pacevolve.context.idea_pool import IdeaPool
from pacevolve.context.failure_memory import FailureMemory
from pacevolve.momentum.progress import ProgressTracker

logger = logging.getLogger(__name__)


@dataclass  
class Program:
    """A program in the database."""
    id: str
    code: str
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    island_id: int = 0
    parent_id: Optional[str] = None
    idea_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    @property
    def score(self) -> float:
        """Get the combined score."""
        return self.metrics.get("combined_score", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "code": self.code,
            "metrics": self.metrics,
            "artifacts": {k: str(v)[:1000] for k, v in self.artifacts.items()},
            "generation": self.generation,
            "island_id": self.island_id,
            "parent_id": self.parent_id,
            "idea_id": self.idea_id,
            "hypothesis_id": self.hypothesis_id,
            "created_at": self.created_at,
        }


@dataclass
class Island:
    """An island (subpopulation) in the database."""
    id: int
    idea_pool: IdeaPool
    progress_tracker: ProgressTracker
    programs: List[Program] = field(default_factory=list)
    generation: int = 0
    best_program: Optional[Program] = None
    
    @property
    def best_score(self) -> float:
        """Get best score in this island."""
        if self.best_program:
            return self.best_program.score
        return 0.0
    
    def add_program(self, program: Program) -> None:
        """Add a program to this island."""
        program.island_id = self.id
        self.programs.append(program)
        
        # Update best
        if self.best_program is None or program.score > self.best_program.score:
            self.best_program = program
    
    def get_top_programs(self, n: int = 3) -> List[Program]:
        """Get top n programs by score."""
        return sorted(self.programs, key=lambda p: p.score, reverse=True)[:n]


class Database:
    """
    Multi-island database with MAP-Elites for OpenPACEvolve.
    
    Features:
    - Multiple islands evolving independently
    - Periodic migration between islands
    - Global failure memory
    - Checkpoint/resume support
    """
    
    def __init__(
        self,
        num_islands: int = 5,
        population_size: int = 1000,
        migration_interval: int = 50,
        migration_rate: float = 0.1,
        llm: Optional["BaseLLM"] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the database.
        
        Args:
            num_islands: Number of islands.
            population_size: Max programs per island.
            migration_interval: Migrate every N generations.
            migration_rate: Fraction of top programs to migrate.
            llm: LLM for context management.
            seed: Random seed.
        """
        self.num_islands = num_islands
        self.population_size = population_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.llm = llm
        
        import random
        self.rng = random.Random(seed)
        
        # Initialize islands
        self.islands: List[Island] = []
        for i in range(num_islands):
            self.islands.append(Island(
                id=i,
                idea_pool=IdeaPool(llm=llm),
                progress_tracker=ProgressTracker(),
            ))
        
        # Global components
        self.failure_memory = FailureMemory()
        self.global_best: Optional[Program] = None
        self.total_iterations = 0
    
    def get_island(self, island_id: int) -> Island:
        """Get an island by ID."""
        return self.islands[island_id]
    
    def add_program(self, program: Program, island_id: int) -> None:
        """Add a program to a specific island."""
        island = self.islands[island_id]
        island.add_program(program)
        
        # Update global best
        if self.global_best is None or program.score > self.global_best.score:
            self.global_best = program
            logger.info(f"New global best: {program.score:.4f}")
        
        # Update progress tracker
        island.progress_tracker.update(program.score)
        
        # Prune if over population limit
        if len(island.programs) > self.population_size:
            self._prune_island(island)
    
    def _prune_island(self, island: Island) -> None:
        """Prune island to population size."""
        if len(island.programs) <= self.population_size:
            return
        
        # Keep top programs
        island.programs = sorted(
            island.programs, 
            key=lambda p: p.score, 
            reverse=True
        )[:self.population_size]
    
    def should_migrate(self, generation: int) -> bool:
        """Check if migration should occur."""
        return generation > 0 and generation % self.migration_interval == 0
    
    def migrate(self) -> None:
        """
        Migrate top programs between adjacent islands.
        
        Uses ring topology: island i sends to island (i+1) % n
        """
        if len(self.islands) < 2:
            return
        
        n_migrate = max(1, int(self.population_size * self.migration_rate))
        
        # Collect migrants from each island
        migrants = []
        for island in self.islands:
            top_programs = island.get_top_programs(n_migrate)
            migrants.append(top_programs)
        
        # Transfer to next island in ring
        for i, island in enumerate(self.islands):
            source_idx = (i - 1) % len(self.islands)
            for program in migrants[source_idx]:
                # Create copy with new ID
                migrant = Program(
                    id=str(uuid.uuid4()),
                    code=program.code,
                    metrics=program.metrics.copy(),
                    generation=island.generation,
                    island_id=island.id,
                    parent_id=program.id,
                )
                island.programs.append(migrant)
        
        logger.info(f"Migrated {n_migrate} programs between islands")
    
    def get_global_best(self) -> Optional[Program]:
        """Get the best program across all islands."""
        return self.global_best
    
    def get_island_progresses(self) -> Dict[int, float]:
        """Get absolute progress for all islands."""
        return {
            island.id: island.progress_tracker.get_absolute_progress()
            for island in self.islands
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save database to checkpoint."""
        import json
        
        checkpoint = {
            "num_islands": self.num_islands,
            "population_size": self.population_size,
            "migration_interval": self.migration_interval,
            "total_iterations": self.total_iterations,
            "failure_memory": self.failure_memory.to_dict(),
            "global_best": self.global_best.to_dict() if self.global_best else None,
            "islands": [],
        }
        
        for island in self.islands:
            island_data = {
                "id": island.id,
                "generation": island.generation,
                "idea_pool": island.idea_pool.to_dict(),
                "progress_tracker": island.progress_tracker.to_dict(),
                "programs": [p.to_dict() for p in island.programs[:100]],  # Limit saved programs
                "best_program": island.best_program.to_dict() if island.best_program else None,
            }
            checkpoint["islands"].append(island_data)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, llm: Optional["BaseLLM"] = None) -> "Database":
        """Load database from checkpoint."""
        import json
        
        with open(path, "r") as f:
            checkpoint = json.load(f)
        
        db = cls(
            num_islands=checkpoint["num_islands"],
            population_size=checkpoint["population_size"],
            migration_interval=checkpoint.get("migration_interval", 50),
            llm=llm,
        )
        
        db.total_iterations = checkpoint.get("total_iterations", 0)
        db.failure_memory = FailureMemory.from_dict(checkpoint.get("failure_memory", {}))
        
        if checkpoint.get("global_best"):
            data = checkpoint["global_best"]
            db.global_best = Program(
                id=data["id"],
                code=data["code"],
                metrics=data.get("metrics", {}),
                generation=data.get("generation", 0),
            )
        
        db.islands = []
        for island_data in checkpoint.get("islands", []):
            island = Island(
                id=island_data["id"],
                idea_pool=IdeaPool.from_dict(island_data.get("idea_pool", {}), llm),
                progress_tracker=ProgressTracker.from_dict(
                    island_data.get("progress_tracker", {})
                ),
                generation=island_data.get("generation", 0),
            )
            
            # Load programs
            for p_data in island_data.get("programs", []):
                program = Program(
                    id=p_data["id"],
                    code=p_data["code"],
                    metrics=p_data.get("metrics", {}),
                    generation=p_data.get("generation", 0),
                    island_id=island.id,
                )
                island.programs.append(program)
            
            if island_data.get("best_program"):
                bp = island_data["best_program"]
                island.best_program = Program(
                    id=bp["id"],
                    code=bp["code"],
                    metrics=bp.get("metrics", {}),
                    generation=bp.get("generation", 0),
                )
            
            db.islands.append(island)
        
        logger.info(f"Loaded checkpoint from {path}")
        return db
