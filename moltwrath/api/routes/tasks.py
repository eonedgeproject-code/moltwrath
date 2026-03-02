"""Task management routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def list_tasks(request: Request, status: str | None = None, limit: int = 50):
    """List tasks with optional status filter."""
    storage = request.app.state.storage
    tasks = await storage.list_tasks(status=status, limit=limit)
    return {"tasks": tasks, "count": len(tasks)}


@router.get("/{task_id}")
async def get_task(task_id: str, request: Request):
    """Get task details."""
    storage = request.app.state.storage
    task = await storage.get_task(task_id)
    if not task:
        return {"error": "Task not found"}
    return task
