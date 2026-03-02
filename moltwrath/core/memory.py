"""Agent memory system — short-term conversation + long-term vector storage."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from moltwrath.core.types import Message, MessageRole


class MemoryEntry(BaseModel):
    """A single memory entry."""
    id: str
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0

    @staticmethod
    def generate_id(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:12]


class Memory:
    """Agent memory with sliding window + persistent recall."""

    def __init__(
        self,
        max_short_term: int = 50,
        max_long_term: int = 500,
        system_prompt: str | None = None,
    ):
        self.system_prompt = system_prompt
        self._short_term: deque[Message] = deque(maxlen=max_short_term)
        self._long_term: dict[str, MemoryEntry] = {}
        self._max_long_term = max_long_term

    # ─── Short-Term (Conversation History) ────────────────────────────────

    def add_message(self, message: Message) -> None:
        """Add a message to short-term memory."""
        self._short_term.append(message)

    def add_user_message(self, content: str) -> None:
        self.add_message(Message(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        self.add_message(Message(role=MessageRole.ASSISTANT, content=content))

    def add_tool_message(self, content: str, tool_call_id: str) -> None:
        self.add_message(Message(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
        ))

    def get_conversation(self) -> list[Message]:
        """Get full conversation history with system prompt."""
        messages = []
        if self.system_prompt:
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt,
            ))
        messages.extend(list(self._short_term))
        return messages

    def get_last_n(self, n: int = 10) -> list[Message]:
        """Get last N messages."""
        items = list(self._short_term)
        return items[-n:]

    def clear_short_term(self) -> None:
        """Clear conversation history."""
        self._short_term.clear()

    # ─── Long-Term (Persistent Knowledge) ─────────────────────────────────

    def store(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store information in long-term memory. Returns entry ID."""
        entry_id = MemoryEntry.generate_id(content)
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._long_term[entry_id] = entry

        # Evict low-importance entries if over limit
        if len(self._long_term) > self._max_long_term:
            self._evict()

        return entry_id

    def recall(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Recall memories by keyword search or tags."""
        entries = list(self._long_term.values())

        if tags:
            entries = [e for e in entries if any(t in e.tags for t in tags)]

        if query:
            query_lower = query.lower()
            entries = [
                e for e in entries
                if query_lower in e.content.lower()
            ]

        # Sort by importance × recency
        entries.sort(
            key=lambda e: (e.importance, e.created_at.timestamp()),
            reverse=True,
        )

        # Update access counts
        for entry in entries[:limit]:
            entry.access_count += 1

        return entries[:limit]

    def forget(self, entry_id: str) -> bool:
        """Remove a specific memory."""
        if entry_id in self._long_term:
            del self._long_term[entry_id]
            return True
        return False

    def _evict(self) -> None:
        """Remove least important memories when over capacity."""
        entries = sorted(
            self._long_term.values(),
            key=lambda e: e.importance * (1 + e.access_count * 0.1),
        )
        # Remove bottom 10%
        remove_count = max(1, len(entries) // 10)
        for entry in entries[:remove_count]:
            del self._long_term[entry.id]

    # ─── Serialization ────────────────────────────────────────────────────

    def export_long_term(self) -> list[dict[str, Any]]:
        """Export long-term memories as dicts."""
        return [e.model_dump() for e in self._long_term.values()]

    def import_long_term(self, data: list[dict[str, Any]]) -> None:
        """Import long-term memories from dicts."""
        for item in data:
            entry = MemoryEntry(**item)
            self._long_term[entry.id] = entry

    @property
    def short_term_count(self) -> int:
        return len(self._short_term)

    @property
    def long_term_count(self) -> int:
        return len(self._long_term)

    def summary(self) -> dict[str, Any]:
        return {
            "short_term": self.short_term_count,
            "long_term": self.long_term_count,
            "has_system_prompt": self.system_prompt is not None,
        }
