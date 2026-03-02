"""WebSocket endpoint for real-time agent event streaming."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, channel: str = "default") -> None:
        await ws.accept()
        if channel not in self.connections:
            self.connections[channel] = []
        self.connections[channel].append(ws)

    def disconnect(self, ws: WebSocket, channel: str = "default") -> None:
        if channel in self.connections:
            self.connections[channel] = [
                c for c in self.connections[channel] if c != ws
            ]

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        if channel in self.connections:
            for ws in self.connections[channel]:
                try:
                    await ws.send_json(data)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/agents/{agent_name}")
async def agent_stream(ws: WebSocket, agent_name: str):
    """Stream real-time events from an agent."""
    await manager.connect(ws, channel=agent_name)

    try:
        while True:
            data = await ws.receive_json()

            # Check if agent exists
            agents = ws.app.state.agents
            if agent_name not in agents:
                await ws.send_json({"error": f"Agent '{agent_name}' not found"})
                continue

            agent = agents[agent_name]

            # Set up event streaming
            async def stream_event(event):
                await ws.send_json(event.model_dump(mode="json"))

            agent._on_event = stream_event

            # Execute task
            prompt = data.get("prompt", "")
            context = data.get("context", {})

            result = await agent.run(prompt, context=context)

            await ws.send_json({
                "type": "result",
                "task_id": result.task_id,
                "output": result.output,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
            })

    except WebSocketDisconnect:
        manager.disconnect(ws, channel=agent_name)


@router.websocket("/events")
async def global_events(ws: WebSocket):
    """Stream all agent events globally."""
    await manager.connect(ws, channel="global")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws, channel="global")
