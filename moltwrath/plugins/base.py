"""Plugin system — extend MoltWrath with custom plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from moltwrath.core.agent import Agent
from moltwrath.core.tools import Tool


class BasePlugin(ABC):
    """Base class for MoltWrath plugins."""

    name: str = "base_plugin"
    version: str = "0.1.0"
    description: str = ""

    @abstractmethod
    def setup(self) -> None:
        """Initialize plugin resources."""
        ...

    def get_tools(self) -> list[Tool]:
        """Return tools this plugin provides."""
        return []

    def get_agents(self) -> list[Agent]:
        """Return pre-configured agents this plugin provides."""
        return []

    def teardown(self) -> None:
        """Cleanup plugin resources."""
        pass


class PluginRegistry:
    """Load and manage plugins."""

    def __init__(self):
        self._plugins: dict[str, BasePlugin] = {}

    def register(self, plugin: BasePlugin) -> None:
        plugin.setup()
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> BasePlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return list(self._plugins.keys())

    def get_all_tools(self) -> list[Tool]:
        tools = []
        for plugin in self._plugins.values():
            tools.extend(plugin.get_tools())
        return tools

    def get_all_agents(self) -> list[Agent]:
        agents = []
        for plugin in self._plugins.values():
            agents.extend(plugin.get_agents())
        return agents

    def teardown_all(self) -> None:
        for plugin in self._plugins.values():
            plugin.teardown()
