"""Agent CRUD + execution routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CreateAgentRequest(BaseModel):
    name: str
    role: str = "general"
    instructions: str = ""
    model: str = "gpt-4"
    provider: str = "openai"
    temperature: float = 0.7
    tools: list[str] = []


class RunAgentRequest(BaseModel):
    prompt: str
    context: dict[str, Any] = {}


@router.post("")
async def create_agent(req: CreateAgentRequest, request: Request):
    """Create a new agent."""
    storage = request.app.state.storage

    from moltwrath.core.agent import Agent
    from moltwrath.llm.openai import OpenAIProvider
    from moltwrath.llm.anthropic import AnthropicProvider
    from moltwrath.utils.config import get_settings

    settings = get_settings()

    # Select provider
    if req.provider == "anthropic":
        llm = AnthropicProvider(
            model=req.model,
            temperature=req.temperature,
            api_key=settings.anthropic_api_key,
        )
    else:
        llm = OpenAIProvider(
            model=req.model,
            temperature=req.temperature,
            api_key=settings.openai_api_key,
        )

    agent = Agent(
        name=req.name,
        role=req.role,
        instructions=req.instructions,
        llm=llm,
    )

    # Store in memory + DB
    request.app.state.agents[req.name] = agent
    await storage.save_agent({
        "id": agent.id,
        "name": agent.name,
        "role": agent.role,
        "instructions": agent.instructions,
        "config": req.model_dump(),
    })

    return {"id": agent.id, "name": agent.name, "status": "created"}


@router.get("")
async def list_agents(request: Request):
    """List all agents."""
    storage = request.app.state.storage
    agents = await storage.list_agents()
    return {"agents": agents, "count": len(agents)}


@router.get("/{name}")
async def get_agent(name: str, request: Request):
    """Get agent details."""
    storage = request.app.state.storage
    agent = await storage.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agent


@router.post("/{name}/run")
async def run_agent(name: str, req: RunAgentRequest, request: Request):
    """Execute a task with an agent."""
    agents = request.app.state.agents

    if name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not loaded")

    agent = agents[name]
    result = await agent.run(req.prompt, context=req.context)

    # Save task
    storage = request.app.state.storage
    await storage.save_task({
        "id": result.task_id,
        "agent_id": agent.id,
        "prompt": req.prompt,
        "output": result.output,
        "status": result.status.value,
        "tokens_used": result.tokens_used,
        "duration_ms": result.duration_ms,
    })

    return {
        "task_id": result.task_id,
        "output": result.output,
        "tokens_used": result.tokens_used,
        "duration_ms": result.duration_ms,
        "tool_calls": len(result.tool_calls),
        "status": result.status.value,
    }
