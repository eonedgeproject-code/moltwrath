"""Tests for core agent functionality."""

import pytest
from moltwrath.core import Agent, Tool, Memory
from moltwrath.core.types import AgentStatus


def test_agent_creation():
    agent = Agent(name="test", role="tester")
    assert agent.name == "test"
    assert agent.role == "tester"
    assert agent.status == AgentStatus.IDLE


def test_memory():
    mem = Memory(max_short_term=10)
    mem.add_user_message("hello")
    mem.add_assistant_message("hi there")
    assert mem.short_term_count == 2


def test_memory_long_term():
    mem = Memory()
    entry_id = mem.store("SOL is trading at $142", importance=0.8, tags=["crypto"])
    assert mem.long_term_count == 1
    results = mem.recall(tags=["crypto"])
    assert len(results) == 1


def test_tool_creation():
    @Tool.define(name="test_tool", description="A test tool")
    def my_tool(query: str) -> str:
        return f"Result: {query}"

    assert my_tool.name == "test_tool"


@pytest.mark.asyncio
async def test_tool_execution():
    @Tool.define(name="greet", description="Greet someone")
    async def greet(name: str) -> str:
        return f"Hello {name}!"

    result = await greet.execute(name="MoltWrath")
    assert result == "Hello MoltWrath!"
