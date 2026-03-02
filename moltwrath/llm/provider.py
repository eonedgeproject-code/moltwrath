"""LLM provider abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Standardized LLM response across providers."""
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    provider_name: str = "base"

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages to LLM and get response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        """Stream response chunks from LLM."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"
