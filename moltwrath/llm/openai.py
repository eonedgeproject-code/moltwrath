"""OpenAI LLM provider."""

from __future__ import annotations

import json
from typing import Any

from moltwrath.llm.provider import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI/GPT provider with function calling support."""

    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Clean messages for OpenAI format
        clean_msgs = []
        for m in messages:
            msg = {"role": m["role"], "content": m["content"]}
            if m.get("name"):
                msg["name"] = m["name"]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            clean_msgs.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": clean_msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Parse tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw=response.model_dump(),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        client = self._get_client()

        clean_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": clean_msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        async for chunk in await client.chat.completions.create(**kwargs):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
