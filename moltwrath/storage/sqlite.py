"""SQLite storage for agent data, tasks, and memory persistence."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import aiosqlite


class SQLiteStorage:
    """Async SQLite storage layer."""

    def __init__(self, db_path: str = "moltwrath.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._create_tables()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def _create_tables(self) -> None:
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                role TEXT DEFAULT '',
                instructions TEXT DEFAULT '',
                config TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                prompt TEXT NOT NULL,
                output TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                tokens_used INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(agent_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_name);
        """)
        await self._db.commit()

    # ─── Agents ───────────────────────────────────────────────────────────

    async def save_agent(self, agent_data: dict[str, Any]) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO agents (id, name, role, instructions, config, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                agent_data["id"],
                agent_data["name"],
                agent_data.get("role", ""),
                agent_data.get("instructions", ""),
                json.dumps(agent_data.get("config", {})),
                datetime.utcnow().isoformat(),
            ),
        )
        await self._db.commit()

    async def get_agent(self, name: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM agents WHERE name = ?", (name,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(zip([d[0] for d in cursor.description], row))

    async def list_agents(self) -> list[dict]:
        cursor = await self._db.execute("SELECT * FROM agents ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(zip([d[0] for d in cursor.description], r)) for r in rows]

    # ─── Tasks ────────────────────────────────────────────────────────────

    async def save_task(self, task_data: dict[str, Any]) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO tasks
               (id, agent_id, prompt, output, status, tokens_used, duration_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_data["id"],
                task_data.get("agent_id"),
                task_data["prompt"],
                task_data.get("output", ""),
                task_data.get("status", "pending"),
                task_data.get("tokens_used", 0),
                task_data.get("duration_ms", 0),
                json.dumps(task_data.get("metadata", {})),
            ),
        )
        await self._db.commit()

    async def get_task(self, task_id: str) -> dict | None:
        cursor = await self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(zip([d[0] for d in cursor.description], row))

    async def list_tasks(self, status: str | None = None, limit: int = 50) -> list[dict]:
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [dict(zip([d[0] for d in cursor.description], r)) for r in rows]

    # ─── Memories ─────────────────────────────────────────────────────────

    async def save_memory(self, agent_name: str, entry: dict[str, Any]) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO memories (id, agent_name, content, importance, tags)
               VALUES (?, ?, ?, ?, ?)""",
            (
                entry["id"],
                agent_name,
                entry["content"],
                entry.get("importance", 0.5),
                json.dumps(entry.get("tags", [])),
            ),
        )
        await self._db.commit()

    async def load_memories(self, agent_name: str) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM memories WHERE agent_name = ? ORDER BY importance DESC",
            (agent_name,),
        )
        rows = await cursor.fetchall()
        return [dict(zip([d[0] for d in cursor.description], r)) for r in rows]
