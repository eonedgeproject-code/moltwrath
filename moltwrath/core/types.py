"""Core type definitions for MoltWrath framework."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SwarmStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    AUCTION = "auction"
    DIRECTOR = "director"
    PARALLEL = "parallel"


# ─── Core Models ──────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class AgentConfig(BaseModel):
    name: str
    role: str = "general"
    instructions: str = ""
    model: str = "gpt-4"
    provider: str = "openai"
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[str] = Field(default_factory=list)
    memory_enabled: bool = True
    max_memory_items: int = 100
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    agent_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = 10
    timeout: float = 300.0


class AgentResult(BaseModel):
    agent_name: str
    task_id: str
    output: str
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens_used: int = 0
    duration_ms: float = 0
    status: AgentStatus = AgentStatus.COMPLETED
    metadata: dict[str, Any] = Field(default_factory=dict)


class SwarmResult(BaseModel):
    task_id: str
    strategy: SwarmStrategy
    agent_results: list[AgentResult] = Field(default_factory=list)
    final_output: str = ""
    total_tokens: int = 0
    total_duration_ms: float = 0
    status: TaskStatus = TaskStatus.COMPLETED


# ─── Events (for WebSocket streaming) ────────────────────────────────────────

class EventType(str, Enum):
    AGENT_START = "agent.start"
    AGENT_THINKING = "agent.thinking"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_RESPONSE = "agent.response"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"
    SWARM_START = "swarm.start"
    SWARM_HANDOFF = "swarm.handoff"
    SWARM_COMPLETE = "swarm.complete"
    TASK_UPDATE = "task.update"


class AgentEvent(BaseModel):
    type: EventType
    agent_name: str
    task_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
