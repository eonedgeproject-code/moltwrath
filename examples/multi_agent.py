"""Example: Multi-agent swarm with different strategies."""

import asyncio

from moltwrath.core import Agent, Tool
from moltwrath.llm import OpenAIProvider, AnthropicProvider
from moltwrath.orchestrator import Swarm, Pipeline, Router


# ─── Define Agents ────────────────────────────────────────────────────────────

researcher = Agent(
    name="researcher",
    role="Research Specialist",
    instructions="You research topics thoroughly, finding key facts and data.",
    llm=OpenAIProvider(model="gpt-4"),
)

analyst = Agent(
    name="analyst",
    role="Data Analyst",
    instructions="You analyze information, find patterns, and draw conclusions.",
    llm=OpenAIProvider(model="gpt-4"),
)

writer = Agent(
    name="writer",
    role="Content Writer",
    instructions="You write clear, engaging content based on research and analysis.",
    llm=OpenAIProvider(model="gpt-4"),
)


async def demo_swarm():
    """Demo: Swarm with round-robin strategy."""
    print("\n🐝 SWARM — Round Robin")
    print("=" * 60)

    swarm = Swarm(
        agents=[researcher, analyst, writer],
        strategy="round_robin",
    )

    result = await swarm.execute("Analyze the current state of Solana DeFi")

    print(f"Agents used: {len(result.agent_results)}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"\nFinal output:\n{result.final_output[:500]}...")


async def demo_pipeline():
    """Demo: Sequential pipeline with transforms."""
    print("\n🔗 PIPELINE — Sequential Chain")
    print("=" * 60)

    pipeline = Pipeline()
    pipeline.add(researcher)
    pipeline.add(analyst, transform=lambda x: f"Analyze this research:\n{x}")
    pipeline.add(writer, transform=lambda x: f"Write a summary based on:\n{x}")

    result = await pipeline.execute("Top 5 Solana DeFi protocols by TVL")

    print(f"Steps completed: {len(result.agent_results)}")
    print(f"\nFinal output:\n{result.final_output[:500]}...")


async def demo_router():
    """Demo: Intelligent task routing."""
    print("\n🔀 ROUTER — Task Routing")
    print("=" * 60)

    router = Router(fallback=researcher)
    router.add(researcher, keywords=["research", "find", "discover", "search"])
    router.add(analyst, keywords=["analyze", "compare", "data", "metrics"])
    router.add(writer, keywords=["write", "draft", "create", "compose"])

    prompts = [
        "Research the latest Solana MEV strategies",
        "Analyze TVL growth across DeFi protocols",
        "Write a thread about AI agents in crypto",
    ]

    for prompt in prompts:
        agent = router.find_agent(prompt)
        print(f"  '{prompt[:50]}...' → {agent.name if agent else 'none'}")


async def main():
    await demo_router()
    # Uncomment below to run with real LLM (requires API keys):
    # await demo_swarm()
    # await demo_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
