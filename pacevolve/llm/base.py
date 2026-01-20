"""
Base LLM interface for OpenPACEvolve.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    
    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMResponse with generated content.
        """
        pass
    
    @abstractmethod
    async def generate_code(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate code from a prompt.
        
        Args:
            prompt: The user prompt describing the code to generate.
            system_message: System message for context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Generated code string.
        """
        pass
    
    def extract_code(self, response: str) -> str:
        """
        Extract code from an LLM response.
        
        Handles common patterns like markdown code blocks.
        """
        import re
        
        # Try to extract from markdown code blocks
        code_block_pattern = r"```(?:python|py)?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Return the longest code block
            return max(matches, key=len).strip()
        
        # If no code blocks, return the response as-is
        return response.strip()
