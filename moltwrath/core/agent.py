"""Core Agent — autonomous AI unit with memory, tools, and LLM integration."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator

from moltwrath.core.memory import Memory
from moltwrath.core.tools import Tool, ToolRegistry
from moltwrath.core.types import (
    AgentConfig,
    AgentEvent,
    AgentResult,
    AgentStatus,
    EventType,
    Message,
    MessageRole,
    ToolCall,
)


class Agent:
    """An autonomous AI agent with memory, tools, and reasoning capabilities.

    Usage:
        agent = Agent(
            name="researcher",
            role="Research Assistant",
            instructions="You research topics thoroughly.",
            llm=OpenAIProvider(model="gpt-4"),
            tools=[search_tool, browse_tool],
        )
        result = await agent.run("Find latest DeFi trends")
    """

    def __init__(
        self,
        name: str,
        role: str = "general",
        instructions: str = "",
        llm: Any = None,  # BaseLLMProvider
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        config: AgentConfig | None = None,
        max_iterations: int = 10,
        on_event: Any = None,  # Callable for event streaming
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role
        self.instructions = instructions
        self.llm = llm
        self.max_iterations = max_iterations
        self.status = AgentStatus.IDLE
        self._on_event = on_event

        # Tool registry
        self.tool_registry = ToolRegistry()
        if tools:
            self.tool_registry.register_many(tools)

        # Memory
        system_prompt = self._build_system_prompt()
        self.memory = memory or Memory(system_prompt=system_prompt)
        if not memory:
            self.memory.system_prompt = system_prompt

        # Config
        self.config = config or AgentConfig(
            name=name, role=role, instructions=instructions
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt from agent config."""
        parts = []

        if self.role:
            parts.append(f"You are a {self.role}.")

        if self.instructions:
            parts.append(self.instructions)

        tool_names = self.tool_registry.list_tools()
        if tool_names:
            parts.append(
                f"You have access to these tools: {', '.join(tool_names)}. "
                "Use them when needed to accomplish your task."
            )

        parts.append(
            "Always think step-by-step. If a task requires multiple steps, "
            "plan them out before executing."
        )

        return "\n\n".join(parts)

    async def _emit(self, event_type: EventType, data: dict[str, Any] = {}, task_id: str = "") -> None:
        """Emit an event for streaming."""
        if self._on_event:
            event = AgentEvent(
                type=event_type,
                agent_name=self.name,
                task_id=task_id,
                data=data,
            )
            if callable(self._on_event):
                result = self._on_event(event)
                if hasattr(result, "__await__"):
                    await result

    async def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> AgentResult:
        """Execute a task with the agentic loop.

        1. Send prompt + history to LLM
        2. If LLM returns tool calls → execute tools → feed results back
        3. Repeat until LLM returns final response or max iterations
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        all_tool_calls: list[ToolCall] = []

        await self._emit(EventType.AGENT_START, {"prompt": prompt}, task_id)

        # Add context to memory if provided
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            self.memory.add_message(Message(
                role=MessageRole.SYSTEM,
                content=f"Additional context:\n{context_str}",
            ))

        # Add user prompt
        self.memory.add_user_message(prompt)

        if not self.llm:
            self.status = AgentStatus.ERROR
            return AgentResult(
                agent_name=self.name,
                task_id=task_id,
                output="Error: No LLM provider configured",
                status=AgentStatus.ERROR,
            )

        # ─── Agentic Loop ─────────────────────────────────────────────
        iteration = 0
        final_output = ""

        while iteration < self.max_iterations:
            iteration += 1

            await self._emit(
                EventType.AGENT_THINKING,
                {"iteration": iteration},
                task_id,
            )

            # Call LLM
            messages = self.memory.get_conversation()
            tools = self.tool_registry.get_schemas(
                format=getattr(self.llm, "provider_name", "openai")
            )

            response = await self.llm.chat(
                messages=[m.model_dump() for m in messages],
                tools=tools if tools else None,
            )

            # Check if LLM wants to use tools
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_call = ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                    )

                    await self._emit(
                        EventType.AGENT_TOOL_CALL,
                        {"tool": tool_call.name, "args": tool_call.arguments},
                        task_id,
                    )

                    # Execute tool
                    tool_call = await self.tool_registry.execute(tool_call)
                    all_tool_calls.append(tool_call)

                    await self._emit(
                        EventType.AGENT_TOOL_RESULT,
                        {"tool": tool_call.name, "result": str(tool_call.result)},
                        task_id,
                    )

                    # Add tool result to memory
                    self.memory.add_tool_message(
                        content=str(tool_call.result),
                        tool_call_id=tool_call.id,
                    )
            else:
                # LLM returned final text response
                final_output = response.content
                self.memory.add_assistant_message(final_output)

                await self._emit(
                    EventType.AGENT_RESPONSE,
                    {"output": final_output},
                    task_id,
                )
                break

        # ─── Build Result ──────────────────────────────────────────────
        duration_ms = (time.time() - start_time) * 1000
        self.status = AgentStatus.COMPLETED

        result = AgentResult(
            agent_name=self.name,
            task_id=task_id,
            output=final_output or "Max iterations reached without final response.",
            messages=list(self.memory.get_conversation()),
            tool_calls=all_tool_calls,
            tokens_used=getattr(response, "tokens_used", 0) if "response" in dir() else 0,
            duration_ms=duration_ms,
            status=self.status,
        )

        await self._emit(EventType.AGENT_COMPLETE, {"duration_ms": duration_ms}, task_id)

        return result

    def reset(self) -> None:
        """Reset agent state."""
        self.memory.clear_short_term()
        self.status = AgentStatus.IDLE

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', role='{self.role}', status={self.status})"
