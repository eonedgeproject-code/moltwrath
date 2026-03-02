"""Example: Simple single agent with tools."""

import asyncio

from moltwrath.core import Agent, Tool
from moltwrath.llm import OpenAIProvider


# ─── Define Tools ─────────────────────────────────────────────────────────────

@Tool.define(name="calculate", description="Perform math calculations")
async def calculate(expression: str) -> str:
    try:
        result = eval(expression)  # In production, use a safe math parser
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@Tool.define(name="get_price", description="Get crypto token price")
async def get_price(token: str) -> str:
    # Mock prices — replace with real API
    prices = {"SOL": 142.50, "ETH": 3200.00, "BTC": 67000.00}
    price = prices.get(token.upper())
    if price:
        return f"{token.upper()}: ${price:,.2f}"
    return f"Price not found for {token}"


# ─── Create Agent ─────────────────────────────────────────────────────────────

async def main():
    agent = Agent(
        name="crypto_analyst",
        role="Crypto Market Analyst",
        instructions=(
            "You analyze cryptocurrency markets and provide insights. "
            "Use tools to get prices and perform calculations when needed."
        ),
        llm=OpenAIProvider(model="gpt-4"),
        tools=[calculate, get_price],
    )

    # Run a task
    result = await agent.run("What's the current SOL price and how much is 100 SOL worth?")

    print(f"\n{'='*60}")
    print(f"Agent: {result.agent_name}")
    print(f"Output: {result.output}")
    print(f"Tools used: {len(result.tool_calls)}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"Tokens: {result.tokens_used}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
