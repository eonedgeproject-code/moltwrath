"""Anthropic/Claude LLM provider."""

from __future__ import annotations

from typing import Any

from moltwrath.llm.provider import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with tool use support."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Extract system message
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                role = "user" if m["role"] in ("user", "tool") else "assistant"
                chat_msgs.append({"role": role, "content": m["content"]})

        # Merge consecutive same-role messages
        merged = []
        for msg in chat_msgs:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": merged,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if system.strip():
            kwargs["system"] = system.strip()

        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        # Parse response
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            tokens_used=(response.usage.input_tokens + response.usage.output_tokens)
            if response.usage
            else 0,
            finish_reason=response.stop_reason or "stop",
            raw=response.model_dump(),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        client = self._get_client()

        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                role = "user" if m["role"] in ("user", "tool") else "assistant"
                chat_msgs.append({"role": role, "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_msgs,
            "max_tokens": self.max_tokens,
        }
        if system.strip():
            kwargs["system"] = system.strip()

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
