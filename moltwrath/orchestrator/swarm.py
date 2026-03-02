"""Swarm — multi-agent coordination with different strategies."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from moltwrath.core.agent import Agent
from moltwrath.core.types import (
    AgentEvent,
    AgentResult,
    EventType,
    SwarmResult,
    SwarmStrategy,
    TaskStatus,
)


class Swarm:
    """Coordinate multiple agents using different strategies.

    Strategies:
        - round_robin: Each agent processes sequentially, passing context forward
        - parallel: All agents process simultaneously, results merged
        - auction: Agents "bid" on tasks based on relevance, best match executes
        - director: A director agent delegates to specialist agents

    Usage:
        swarm = Swarm(
            agents=[researcher, analyst, writer],
            strategy="round_robin",
        )
        result = await swarm.execute("Analyze Solana DeFi trends")
    """

    def __init__(
        self,
        agents: list[Agent],
        strategy: str | SwarmStrategy = SwarmStrategy.ROUND_ROBIN,
        director: Agent | None = None,
        on_event: Any = None,
    ):
        self.agents = {a.name: a for a in agents}
        self.agent_list = agents
        self.strategy = SwarmStrategy(strategy)
        self.director = director
        self._on_event = on_event

    async def _emit(self, event_type: EventType, data: dict, task_id: str = "") -> None:
        if self._on_event:
            event = AgentEvent(
                type=event_type,
                agent_name="swarm",
                task_id=task_id,
                data=data,
            )
            if callable(self._on_event):
                result = self._on_event(event)
                if hasattr(result, "__await__"):
                    await result

    async def execute(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> SwarmResult:
        """Execute a task using the configured strategy."""
        task_id = str(uuid.uuid4())[:8]
        start = time.time()

        await self._emit(EventType.SWARM_START, {
            "strategy": self.strategy.value,
            "agents": list(self.agents.keys()),
            "prompt": prompt,
        }, task_id)

        match self.strategy:
            case SwarmStrategy.ROUND_ROBIN:
                result = await self._round_robin(prompt, context, task_id)
            case SwarmStrategy.PARALLEL:
                result = await self._parallel(prompt, context, task_id)
            case SwarmStrategy.AUCTION:
                result = await self._auction(prompt, context, task_id)
            case SwarmStrategy.DIRECTOR:
                result = await self._director_strategy(prompt, context, task_id)
            case _:
                result = await self._round_robin(prompt, context, task_id)

        result.total_duration_ms = (time.time() - start) * 1000
        result.total_tokens = sum(r.tokens_used for r in result.agent_results)

        await self._emit(EventType.SWARM_COMPLETE, {
            "duration_ms": result.total_duration_ms,
            "agents_used": len(result.agent_results),
        }, task_id)

        return result

    # ─── Strategies ───────────────────────────────────────────────────────

    async def _round_robin(
        self, prompt: str, context: dict | None, task_id: str
    ) -> SwarmResult:
        """Sequential execution — each agent builds on the previous."""
        results: list[AgentResult] = []
        current_prompt = prompt
        ctx = context or {}

        for agent in self.agent_list:
            await self._emit(EventType.SWARM_HANDOFF, {
                "to_agent": agent.name,
                "prompt": current_prompt[:200],
            }, task_id)

            result = await agent.run(current_prompt, context=ctx, task_id=task_id)
            results.append(result)

            # Pass output as context to next agent
            ctx[f"{agent.name}_output"] = result.output
            current_prompt = (
                f"Previous agent ({agent.name}) produced:\n{result.output}\n\n"
                f"Original task: {prompt}\n\n"
                "Build upon this and add your expertise."
            )

        return SwarmResult(
            task_id=task_id,
            strategy=SwarmStrategy.ROUND_ROBIN,
            agent_results=results,
            final_output=results[-1].output if results else "",
            status=TaskStatus.COMPLETED,
        )

    async def _parallel(
        self, prompt: str, context: dict | None, task_id: str
    ) -> SwarmResult:
        """Parallel execution — all agents process simultaneously."""
        tasks = [
            agent.run(prompt, context=context, task_id=task_id)
            for agent in self.agent_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        agent_results = []
        outputs = []
        for r in results:
            if isinstance(r, AgentResult):
                agent_results.append(r)
                outputs.append(f"[{r.agent_name}]: {r.output}")

        final_output = "\n\n---\n\n".join(outputs)

        return SwarmResult(
            task_id=task_id,
            strategy=SwarmStrategy.PARALLEL,
            agent_results=agent_results,
            final_output=final_output,
            status=TaskStatus.COMPLETED,
        )

    async def _auction(
        self, prompt: str, context: dict | None, task_id: str
    ) -> SwarmResult:
        """Auction — agents bid on relevance, best match executes."""
        best_agent = self.agent_list[0]  # Default
        best_score = 0.0

        # Simple keyword matching for agent selection
        prompt_lower = prompt.lower()
        for agent in self.agent_list:
            score = 0.0
            role_words = agent.role.lower().split()
            for word in role_words:
                if word in prompt_lower:
                    score += 1.0
            instruction_words = agent.instructions.lower().split()[:20]
            for word in instruction_words:
                if len(word) > 3 and word in prompt_lower:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_agent = agent

        await self._emit(EventType.SWARM_HANDOFF, {
            "to_agent": best_agent.name,
            "score": best_score,
        }, task_id)

        result = await best_agent.run(prompt, context=context, task_id=task_id)

        return SwarmResult(
            task_id=task_id,
            strategy=SwarmStrategy.AUCTION,
            agent_results=[result],
            final_output=result.output,
            status=TaskStatus.COMPLETED,
        )

    async def _director_strategy(
        self, prompt: str, context: dict | None, task_id: str
    ) -> SwarmResult:
        """Director — a director agent delegates to specialists."""
        if not self.director:
            return await self._round_robin(prompt, context, task_id)

        agent_info = "\n".join(
            f"- {a.name} ({a.role}): {a.instructions[:100]}"
            for a in self.agent_list
        )

        director_prompt = (
            f"You are the director. Delegate this task to the best agent(s).\n\n"
            f"Available agents:\n{agent_info}\n\n"
            f"Task: {prompt}\n\n"
            f"Respond with the agent name(s) to use, one per line."
        )

        director_result = await self.director.run(director_prompt, task_id=task_id)

        # Parse which agents to use
        selected = []
        for agent in self.agent_list:
            if agent.name.lower() in director_result.output.lower():
                selected.append(agent)

        if not selected:
            selected = [self.agent_list[0]]

        results = [director_result]
        outputs = []

        for agent in selected:
            await self._emit(EventType.SWARM_HANDOFF, {
                "to_agent": agent.name,
                "delegated_by": "director",
            }, task_id)

            result = await agent.run(prompt, context=context, task_id=task_id)
            results.append(result)
            outputs.append(result.output)

        return SwarmResult(
            task_id=task_id,
            strategy=SwarmStrategy.DIRECTOR,
            agent_results=results,
            final_output="\n\n".join(outputs),
            status=TaskStatus.COMPLETED,
        )

    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.name] = agent
        self.agent_list.append(agent)

    def remove_agent(self, name: str) -> None:
        if name in self.agents:
            del self.agents[name]
            self.agent_list = [a for a in self.agent_list if a.name != name]
