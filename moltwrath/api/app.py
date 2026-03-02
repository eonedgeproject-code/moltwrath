"""FastAPI application — REST + WebSocket API for MoltWrath agents."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from moltwrath.api.routes.agents import router as agents_router
from moltwrath.api.routes.tasks import router as tasks_router
from moltwrath.api.routes.ws import router as ws_router
from moltwrath.storage.sqlite import SQLiteStorage
from moltwrath.utils.config import get_settings
from moltwrath.utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logger = setup_logger(level=settings.log_level)
    logger.info(f"🧠 MoltWrath starting on port {settings.api_port}")

    # Init storage
    storage = SQLiteStorage()
    await storage.connect()
    app.state.storage = storage
    app.state.agents = {}  # In-memory agent registry

    yield

    await storage.close()
    logger.info("🧠 MoltWrath shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="🧠 MoltWrath",
        description="AI Agent Orchestration Framework",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(agents_router, prefix="/agents", tags=["agents"])
    app.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
    app.include_router(ws_router, prefix="/ws", tags=["websocket"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "framework": "moltwrath", "version": "0.1.0"}

    @app.get("/")
    async def root():
        return {
            "name": "🧠 MoltWrath",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
