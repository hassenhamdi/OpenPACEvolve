"""
Ensemble LLM for OpenPACEvolve.

Combines multiple LLM models with weighted selection.
"""

import logging
import random
from typing import List, Optional

from pacevolve.config import LLMConfig, LLMModelConfig
from pacevolve.llm.base import BaseLLM, LLMResponse, Message
from pacevolve.llm.openai import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


class EnsembleLLM(BaseLLM):
    """
    Ensemble of multiple LLM models with weighted selection.
    
    Selects a model for each generation based on configured weights.
    Includes fallback logic if a model fails.
    """
    
    def __init__(self, config: LLMConfig, seed: Optional[int] = None):
        """
        Initialize the ensemble.
        
        Args:
            config: LLM configuration with model weights.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.models: List[tuple[LLMModelConfig, OpenAICompatibleLLM]] = []
        self.rng = random.Random(seed)
        
        # Initialize all models
        for model_config in config.models:
            client = OpenAICompatibleLLM(
                model=model_config.name,
                api_key=config.api_key or "",
                api_base=config.api_base,
                timeout=config.timeout,
                retries=config.retries,
                retry_delay=config.retry_delay,
            )
            self.models.append((model_config, client))
        
        # Compute total weight for normalization
        self.total_weight = sum(m.weight for m, _ in self.models)
    
    def _select_model(self) -> tuple[LLMModelConfig, OpenAICompatibleLLM]:
        """Select a model based on weights."""
        if not self.models:
            raise ValueError("No models configured in ensemble")
        
        if len(self.models) == 1:
            return self.models[0]
        
        # Weighted random selection
        r = self.rng.uniform(0, self.total_weight)
        cumulative = 0.0
        
        for model_config, client in self.models:
            cumulative += model_config.weight
            if r <= cumulative:
                return model_config, client
        
        # Fallback to last model
        return self.models[-1]
    
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """Generate using weighted model selection with fallback."""
        # Try selected model first
        model_config, client = self._select_model()
        
        try:
            logger.debug(f"Using model: {model_config.name}")
            return await client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Model {model_config.name} failed: {e}")
            
            # Try fallback models
            for fallback_config, fallback_client in self.models:
                if fallback_config.name == model_config.name:
                    continue
                    
                try:
                    logger.info(f"Falling back to model: {fallback_config.name}")
                    return await fallback_client.generate(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_config.name} failed: {fallback_error}")
                    continue
            
            # All models failed
            raise RuntimeError(f"All models in ensemble failed. Last error: {e}")
    
    async def generate_code(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate code using weighted model selection."""
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=prompt),
        ]
        
        response = await self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return self.extract_code(response.content)
