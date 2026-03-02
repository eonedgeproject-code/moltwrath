"""Router — intelligent task routing to the best agent."""

from __future__ import annotations

from typing import Any, Callable

from moltwrath.core.agent import Agent
from moltwrath.core.types import AgentResult


class Route:
    """A routing rule mapping patterns to agents."""

    def __init__(
        self,
        agent: Agent,
        keywords: list[str] | None = None,
        matcher: Callable[[str], bool] | None = None,
        priority: int = 0,
    ):
        self.agent = agent
        self.keywords = [k.lower() for k in (keywords or [])]
        self.matcher = matcher
        self.priority = priority

    def matches(self, prompt: str) -> float:
        """Return match score (0-1) for a given prompt."""
        if self.matcher:
            return 1.0 if self.matcher(prompt) else 0.0

        prompt_lower = prompt.lower()
        if not self.keywords:
            return 0.0

        matched = sum(1 for k in self.keywords if k in prompt_lower)
        return matched / len(self.keywords)


class Router:
    """Route tasks to the best-matching agent.

    Usage:
        router = Router()
        router.add(researcher, keywords=["research", "find", "search"])
        router.add(coder, keywords=["code", "build", "implement"])
        router.add(writer, keywords=["write", "draft", "compose"])

        result = await router.route("Research the latest Solana updates")
    """

    def __init__(self, fallback: Agent | None = None):
        self.routes: list[Route] = []
        self.fallback = fallback

    def add(
        self,
        agent: Agent,
        keywords: list[str] | None = None,
        matcher: Callable[[str], bool] | None = None,
        priority: int = 0,
    ) -> "Router":
        self.routes.append(Route(agent, keywords, matcher, priority))
        return self

    def find_agent(self, prompt: str) -> Agent | None:
        """Find the best matching agent for a prompt."""
        scored = []
        for route in self.routes:
            score = route.matches(prompt) + (route.priority * 0.1)
            if score > 0:
                scored.append((score, route.agent))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return scored[0][1]
        return self.fallback

    async def route(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult | None:
        """Route a task to the best agent and execute."""
        agent = self.find_agent(prompt)
        if not agent:
            return None
        return await agent.run(prompt, context=context)
