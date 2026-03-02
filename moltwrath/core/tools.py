"""Tool system — define and register tools for agents."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from moltwrath.core.types import ToolCall


class ToolSchema(BaseModel):
    """JSON Schema representation of a tool for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool:
    """Wraps a callable as an agent tool with schema extraction."""

    def __init__(
        self,
        func: Callable[..., Any] | Callable[..., Coroutine],
        name: str | None = None,
        description: str | None = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.is_async = inspect.iscoroutinefunction(func)
        self._schema = self._extract_schema()

    def _extract_schema(self) -> ToolSchema:
        """Extract JSON schema from function signature."""
        sig = inspect.signature(self.func)
        hints = {}
        try:
            hints = inspect.get_annotations(self.func)
        except Exception:
            pass

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            prop: dict[str, Any] = {}
            annotation = hints.get(param_name, str)

            # Map Python types → JSON schema types
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            prop["type"] = type_map.get(annotation, "string")

            properties[param_name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": self.schema.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "input_schema": self.schema.parameters,
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function."""
        if self.is_async:
            return await self.func(**kwargs)
        return self.func(**kwargs)

    async def run(self, tool_call: ToolCall) -> ToolCall:
        """Execute from a ToolCall object and return result."""
        try:
            result = await self.execute(**tool_call.arguments)
            tool_call.result = result
        except Exception as e:
            tool_call.result = f"Error: {e}"
        return tool_call

    # ─── Decorator API ────────────────────────────────────────────────────

    @staticmethod
    def define(
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """Decorator to define a tool.

        Usage:
            @Tool.define(name="search", description="Search the web")
            async def search(query: str) -> str:
                return f"Results for {query}"
        """
        def decorator(func: Callable) -> Tool:
            return Tool(func, name=name, description=description)
        return decorator


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self, format: str = "openai") -> list[dict[str, Any]]:
        """Get all tool schemas in specified format."""
        if format == "anthropic":
            return [t.to_anthropic_format() for t in self._tools.values()]
        return [t.to_openai_format() for t in self._tools.values()]

    async def execute(self, tool_call: ToolCall) -> ToolCall:
        """Execute a tool call."""
        tool = self._tools.get(tool_call.name)
        if not tool:
            tool_call.result = f"Error: Unknown tool '{tool_call.name}'"
            return tool_call
        return await tool.run(tool_call)


# ─── Global Registry ─────────────────────────────────────────────────────────

_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    return _global_registry
