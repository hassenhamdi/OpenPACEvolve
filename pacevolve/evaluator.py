"""
Evaluator for OpenPACEvolve.

Handles program evaluation with cascade stages and artifact collection.
"""

import asyncio
import importlib.util
import logging
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pacevolve.config import EvaluatorConfig
from pacevolve.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Program evaluator with cascade evaluation support.
    
    Features:
    - Cascade evaluation to filter bad programs early
    - Timeout protection
    - Artifact collection for feedback
    - Parallel evaluation
    """
    
    def __init__(
        self,
        evaluator_path: Optional[str] = None,
        evaluator_func: Optional[Callable] = None,
        config: Optional[EvaluatorConfig] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            evaluator_path: Path to evaluator Python file.
            evaluator_func: Direct evaluator function.
            config: Evaluator configuration.
        """
        self.evaluator_path = evaluator_path
        self.evaluator_func = evaluator_func
        self.config = config or EvaluatorConfig()
        
        # Load evaluator module if path provided
        self._evaluator_module = None
        if evaluator_path:
            self._load_evaluator_module(evaluator_path)
    
    def _load_evaluator_module(self, path: str) -> None:
        """Load the evaluator module from file."""
        spec = importlib.util.spec_from_file_location("evaluator", path)
        if spec and spec.loader:
            self._evaluator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._evaluator_module)
    
    async def evaluate(
        self,
        program_path: str,
        stage: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate a program.
        
        Args:
            program_path: Path to the program file.
            stage: Specific evaluation stage (None for full evaluation).
            
        Returns:
            EvaluationResult with metrics and artifacts.
        """
        try:
            # Run evaluation with timeout
            result = await asyncio.wait_for(
                self._run_evaluation(program_path, stage),
                timeout=self.config.timeout,
            )
            return result
            
        except asyncio.TimeoutError:
            return EvaluationResult.failure(
                f"Evaluation timed out after {self.config.timeout}s"
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult.failure(str(e))
    
    async def _run_evaluation(
        self,
        program_path: str,
        stage: Optional[int],
    ) -> EvaluationResult:
        """Run the actual evaluation."""
        # Use provided function or module
        if self.evaluator_func:
            result = self.evaluator_func(program_path)
        elif self._evaluator_module:
            # Determine which stage function to call
            if stage == 1 and hasattr(self._evaluator_module, "evaluate_stage1"):
                result = self._evaluator_module.evaluate_stage1(program_path)
            elif stage == 2 and hasattr(self._evaluator_module, "evaluate_stage2"):
                result = self._evaluator_module.evaluate_stage2(program_path)
            elif hasattr(self._evaluator_module, "evaluate"):
                result = self._evaluator_module.evaluate(program_path)
            else:
                return EvaluationResult.failure("No evaluate function found")
        else:
            return EvaluationResult.failure("No evaluator configured")
        
        # Convert result to EvaluationResult if needed
        if isinstance(result, EvaluationResult):
            return result
        elif isinstance(result, dict):
            metrics = result.get("metrics", {})
            if not metrics:
                # Assume all keys are metrics
                metrics = {k: v for k, v in result.items() 
                          if isinstance(v, (int, float)) and k != "error"}
            return EvaluationResult(
                metrics=metrics,
                artifacts=result.get("artifacts", {}),
                error=result.get("error"),
            )
        else:
            return EvaluationResult.failure(f"Invalid evaluation result type: {type(result)}")
    
    async def cascade_evaluate(
        self,
        program_path: str,
    ) -> EvaluationResult:
        """
        Run cascade evaluation with multiple stages.
        
        Each stage must pass a threshold before proceeding.
        """
        if not self.config.cascade_evaluation:
            return await self.evaluate(program_path)
        
        thresholds = self.config.cascade_thresholds
        
        for stage, threshold in enumerate(thresholds, 1):
            result = await self.evaluate(program_path, stage=stage)
            
            if result.error or result.combined_score < threshold:
                logger.debug(
                    f"Stage {stage} failed: score={result.combined_score:.4f} < {threshold}"
                )
                return result
            
            logger.debug(
                f"Stage {stage} passed: score={result.combined_score:.4f} >= {threshold}"
            )
        
        # Full evaluation after passing all stages
        return await self.evaluate(program_path)


class SubprocessEvaluator:
    """
    Evaluator that runs programs in subprocess for isolation.
    """
    
    def __init__(
        self,
        evaluator_path: str,
        timeout: int = 60,
    ):
        """
        Initialize subprocess evaluator.
        
        Args:
            evaluator_path: Path to evaluator module.
            timeout: Subprocess timeout in seconds.
        """
        self.evaluator_path = evaluator_path
        self.timeout = timeout
    
    async def evaluate(self, program_path: str) -> EvaluationResult:
        """Evaluate program in subprocess."""
        import pickle
        
        # Create runner script
        script = f'''
import sys
import pickle
sys.path.insert(0, "{Path(self.evaluator_path).parent}")

import importlib.util
spec = importlib.util.spec_from_file_location("evaluator", "{self.evaluator_path}")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

result = evaluator.evaluate("{program_path}")
print(pickle.dumps(result).hex())
'''
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
            
            if proc.returncode != 0:
                return EvaluationResult.failure(
                    f"Subprocess error: {stderr.decode()[:500]}"
                )
            
            # Parse result
            result_hex = stdout.decode().strip()
            result = pickle.loads(bytes.fromhex(result_hex))
            
            if isinstance(result, dict):
                return EvaluationResult(
                    metrics=result.get("metrics", result),
                    artifacts=result.get("artifacts", {}),
                )
            return result
            
        except asyncio.TimeoutError:
            proc.kill()
            return EvaluationResult.failure(f"Timeout after {self.timeout}s")
        finally:
            Path(script_path).unlink(missing_ok=True)
