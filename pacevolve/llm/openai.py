"""
OpenAI-compatible LLM client for OpenPACEvolve.

Supports OpenAI, Google Gemini, and any OpenAI-compatible API.
"""

import asyncio
import logging
from typing import List, Optional

from openai import AsyncOpenAI

from pacevolve.llm.base import BaseLLM, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """
    LLM client using OpenAI-compatible API.
    
    Works with:
    - OpenAI (api.openai.com)
    - Google Gemini (generativelanguage.googleapis.com)
    - Local models via Ollama, vLLM, etc.
    - Any OpenAI-compatible proxy
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 60,
        retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name (e.g., "gpt-4", "gemini-2.0-flash-lite").
            api_key: API key for authentication.
            api_base: Base URL for the API.
            timeout: Request timeout in seconds.
            retries: Number of retries for failed requests.
            retry_delay: Delay between retries in seconds.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        
        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )
    
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        # Convert Message objects to dicts
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        
        last_error = None
        for attempt in range(self.retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=message_dicts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    **kwargs
                )
                
                # Extract response content
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Build usage dict
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                
                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage=usage,
                    finish_reason=choice.finish_reason,
                    raw_response=response,
                )
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.retries}): {e}"
                )
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        raise RuntimeError(f"LLM request failed after {self.retries} retries: {last_error}")
    
    async def generate_code(
        self,
        prompt: str,
        system_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate code from a prompt."""
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
