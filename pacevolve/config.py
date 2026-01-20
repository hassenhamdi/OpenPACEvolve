"""
Configuration management for OpenPACEvolve.

Handles loading and validation of YAML configuration files with dataclass mapping.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dacite
import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model."""
    name: str
    weight: float = 1.0


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    models: List[LLMModelConfig] = field(default_factory=lambda: [
        LLMModelConfig(name="gemini-2.0-flash-lite", weight=0.8),
        LLMModelConfig(name="gemini-2.0-flash", weight=0.2),
    ])
    evaluator_models: Optional[List[LLMModelConfig]] = None
    api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5


@dataclass
class ContextConfig:
    """Hierarchical Context Management configuration."""
    # Idea pool settings
    max_ideas: int = 20  # Maximum ideas to maintain
    idea_cap: int = 10  # Ideas considered per generation
    
    # Hypothesis settings
    max_hypotheses_per_idea: int = 10
    hypothesis_summarization_trigger: int = 8
    
    # Pruning settings
    prune_low_performing_threshold: float = 0.3
    min_experiments_before_prune: int = 3


@dataclass
class MomentumConfig:
    """Momentum-Based Backtracking configuration."""
    # EWMA settings
    beta: float = 0.9  # Momentum decay factor
    intervention_threshold: float = 0.1  # ε_rel threshold
    
    # Backtracking settings
    power_law_alpha: float = 1.5  # Power-law distribution parameter
    min_backtrack_generations: int = 5


@dataclass  
class SamplingConfig:
    """Self-Adaptive Crossover Sampling configuration."""
    # Action weights
    enable_adaptive_sampling: bool = True
    synergy_bonus_weight: float = 1.0
    stagnation_threshold: float = 0.2


@dataclass
class DatabaseConfig:
    """Database and island configuration."""
    db_path: Optional[str] = None
    in_memory: bool = True
    log_prompts: bool = True
    
    # Population settings
    population_size: int = 1000
    archive_size: int = 100
    
    # Island settings
    num_islands: int = 5
    migration_interval: int = 50
    migration_rate: float = 0.1
    
    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    
    # MAP-Elites feature dimensions
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10
    diversity_reference_size: int = 20


@dataclass
class PromptConfig:
    """Prompt configuration."""
    template_dir: Optional[str] = None
    system_message: str = "You are an expert coder helping to improve programs through evolution."
    evaluator_system_message: str = "You are an expert code reviewer."
    
    # Context sampling
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    
    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)
    
    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20480
    artifact_security_filter: bool = True


@dataclass
class EvaluatorConfig:
    """Evaluator configuration."""
    timeout: int = 300
    max_retries: int = 3
    
    # Cascade evaluation
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    
    # Parallel evaluation
    parallel_evaluations: int = 4
    
    # LLM feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1


@dataclass
class EvolutionTraceConfig:
    """Evolution trace logging configuration."""
    enabled: bool = False
    format: str = "jsonl"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class Config:
    """Complete OpenPACEvolve configuration."""
    # General settings
    max_iterations: int = 100
    checkpoint_interval: int = 10
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    output_dir: str = "openpacevolve_output"
    
    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"
    
    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variable references in configuration values."""
    if isinstance(value, str):
        # Match ${VAR} or $VAR patterns
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replace_env(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        
        return re.sub(pattern, replace_env, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses defaults.
        
    Returns:
        Config object with all settings.
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    # Resolve environment variables
    raw_config = _resolve_env_vars(raw_config)
    
    # Convert to Config dataclass
    config = dacite.from_dict(
        data_class=Config,
        data=raw_config,
        config=dacite.Config(
            cast=[tuple],
            strict=False,
        )
    )
    
    # Set API key from environment if not specified
    if config.llm.api_key is None:
        config.llm.api_key = os.environ.get("OPENAI_API_KEY")
    
    return config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save.
        config_path: Path to save YAML file.
    """
    import dataclasses
    
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj
    
    config_dict = to_dict(config)
    
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
