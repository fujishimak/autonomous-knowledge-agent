"""Mounts the MCP servers in memory and hands LangChain tools to agents by name.

    toolbox = build_default_toolbox()
    await toolbox.connect()                       # or: async with build_default_toolbox() as toolbox
    tools = toolbox.for_agent(["get_reservations", "get_subscriptions"])
    ...
    await toolbox.aclose()

The MCP tools are async-only, so agents that use them must be run with `ainvoke` / `astream`.
"""

from contextlib import AsyncExitStack
from pathlib import Path

from agentic.mcp_servers.cultpass.repository import (
    CultpassRepository,
    create_cultpass_engine,
)
from agentic.mcp_servers.cultpass.tools import build_cultpass_server
from agentic.mcp_servers.udahub.knowledge_index import (
    KnowledgeIndex,
    create_udahub_engine,
)
from agentic.mcp_servers.udahub.repository import UdaHubRepository
from agentic.mcp_servers.udahub.tools import build_udahub_server
from fastmcp import Client, FastMCP
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools


class McpToolbox:
    """Keeps one in-memory client session open per server and indexes their tools by name."""

    def __init__(self, servers: dict[str, FastMCP]):
        self._servers = servers
        self._stack: AsyncExitStack | None = None
        self._tools: dict[str, BaseTool] = {}
        self._tool_server: dict[str, str] = {}

    # -- lifecycle --------------------------------------------------------- #
    async def connect(self) -> "McpToolbox":
        """Open every server and load its tools. Safe to call twice."""
        if self._stack is not None:
            return self
        stack = AsyncExitStack()
        try:
            for server_name, server in self._servers.items():
                client = await stack.enter_async_context(Client(server))
                for tool in await load_mcp_tools(client.session):
                    if tool.name in self._tools:
                        raise ValueError(
                            f"Tool name '{tool.name}' is defined by both "
                            f"'{self._tool_server[tool.name]}' and '{server_name}'"
                        )
                    self._tools[tool.name] = tool
                    self._tool_server[tool.name] = server_name
        except BaseException:
            await stack.aclose()
            self._tools.clear()
            self._tool_server.clear()
            raise
        self._stack = stack
        return self

    async def aclose(self) -> None:
        if self._stack is not None:
            stack, self._stack = self._stack, None
            await stack.aclose()
            self._tools.clear()
            self._tool_server.clear()

    async def __aenter__(self) -> "McpToolbox":
        return await self.connect()

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    # -- lookup ------------------------------------------------------------ #
    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def tool(self, name: str) -> BaseTool:
        """One tool by name, for plumbing the graph calls itself (not bound to an agent)."""
        return self.for_agent([name])[0]

    def for_agent(self, names: list[str]) -> list[BaseTool]:
        """The tools with these names, in the given order. Unknown names raise KeyError."""
        if self._stack is None:
            raise RuntimeError("Toolbox is not connected. Call `await toolbox.connect()` first.")
        missing = [name for name in names if name not in self._tools]
        if missing:
            raise KeyError(f"Unknown tools {missing}. Available: {sorted(self._tools)}")
        return [self._tools[name] for name in names]


def build_default_toolbox(
    cultpass_db_path: str | Path | None = None,
    udahub_db_path: str | Path | None = None,
) -> McpToolbox:
    """The CultPass and UDA-Hub servers on their default databases (solution/data/...).

    Not connected yet: call `await toolbox.connect()` or use it with `async with`.
    """
    cultpass_server = build_cultpass_server(
        CultpassRepository(create_cultpass_engine(cultpass_db_path))
    )
    udahub_engine = create_udahub_engine(udahub_db_path)
    udahub_server = build_udahub_server(
        UdaHubRepository(udahub_engine), KnowledgeIndex(udahub_engine)
    )
    return McpToolbox({"cultpass": cultpass_server, "udahub": udahub_server})
