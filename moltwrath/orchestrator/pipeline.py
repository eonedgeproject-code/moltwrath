"""Pipeline — sequential agent chain with data transformation between steps."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from moltwrath.core.agent import Agent
from moltwrath.core.types import AgentResult, SwarmResult, SwarmStrategy, TaskStatus


@dataclass
class PipelineStep:
    """A single step in the pipeline."""
    agent: Agent
    transform: Callable[[str], str] | None = None  # Transform output before next step
    condition: Callable[[AgentResult], bool] | None = None  # Skip if condition returns False
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.agent.name


class Pipeline:
    """Sequential agent pipeline with conditional branching and transforms.

    Usage:
        pipeline = Pipeline()
        pipeline.add(researcher)
        pipeline.add(analyst, transform=lambda x: f"Analyze this: {x}")
        pipeline.add(writer, condition=lambda r: len(r.output) > 100)

        result = await pipeline.execute("Research DeFi trends")
    """

    def __init__(self):
        self.steps: list[PipelineStep] = []

    def add(
        self,
        agent: Agent,
        transform: Callable[[str], str] | None = None,
        condition: Callable[[AgentResult], bool] | None = None,
        name: str = "",
    ) -> "Pipeline":
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(PipelineStep(
            agent=agent,
            transform=transform,
            condition=condition,
            name=name,
        ))
        return self

    async def execute(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> SwarmResult:
        """Execute the pipeline sequentially."""
        task_id = str(uuid.uuid4())[:8]
        start = time.time()
        results: list[AgentResult] = []
        current_input = prompt
        ctx = context or {}

        for step in self.steps:
            # Check condition
            if step.condition and results:
                if not step.condition(results[-1]):
                    continue

            # Apply transform
            if step.transform:
                current_input = step.transform(current_input)

            # Execute
            result = await step.agent.run(current_input, context=ctx, task_id=task_id)
            results.append(result)

            # Feed output forward
            ctx[f"{step.name}_output"] = result.output
            current_input = result.output

        return SwarmResult(
            task_id=task_id,
            strategy=SwarmStrategy.ROUND_ROBIN,
            agent_results=results,
            final_output=results[-1].output if results else "",
            total_duration_ms=(time.time() - start) * 1000,
            total_tokens=sum(r.tokens_used for r in results),
            status=TaskStatus.COMPLETED,
        )
