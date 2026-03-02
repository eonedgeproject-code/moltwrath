# ⚡ MOLTWRATH — AI Agent Orchestration Framework

> Shed the old. Execute with wrath. Deploy intelligent agent swarms with memory, tools, and real-time coordination.

Built with Python + FastAPI. Multi-agent system framework for autonomous AI operations.

## Architecture

```
moltwrath/
├── core/          # Agent engine, memory, tools
├── orchestrator/  # Swarm coordination, pipelines, routing
├── plugins/       # Extensible plugin system
├── llm/           # LLM provider abstraction (OpenAI, Anthropic)
├── api/           # FastAPI REST + WebSocket endpoints
├── storage/       # Persistent storage layer
└── utils/         # Config, logging, helpers
```

## Quick Start

```bash
pip install -e ".[dev]"
cp .env.example .env
moltwrath serve --reload
```

## Create Your First Agent

```python
from moltwrath.core import Agent, Tool
from moltwrath.llm import OpenAIProvider

@Tool.define(name="search", description="Search the web")
async def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    name="researcher",
    role="Research Assistant",
    instructions="You research topics thoroughly.",
    llm=OpenAIProvider(model="gpt-4"),
    tools=[search],
)

result = await agent.run("Find latest Solana DeFi trends")
```

## Multi-Agent Swarm

```python
from moltwrath.orchestrator import Swarm

swarm = Swarm(
    agents=[researcher, analyst, writer],
    strategy="round_robin",  # or "auction", "director", "parallel"
)
result = await swarm.execute("Analyze top 10 Solana protocols")
```

## Swarm Strategies

| Strategy | Description |
|----------|-------------|
| `round_robin` | Sequential — each agent builds on previous output |
| `parallel` | Simultaneous — all agents process at once |
| `auction` | Best-match — agents bid on relevance |
| `director` | Delegator — director agent assigns to specialists |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agents` | Create agent |
| GET | `/agents` | List agents |
| POST | `/agents/{name}/run` | Execute agent task |
| WS | `/ws/agents/{name}` | Real-time stream |
| GET | `/health` | Health check |

## License

MIT
