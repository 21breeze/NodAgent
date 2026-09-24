from pathlib import Path
from typing import List

from fastmcp import Client
from langchain.mcp import (
    MCPAdapter,
)
from langchain_core.tools import (
    BaseTool,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

EXTERNAL_INFO_SERVER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "mcp_servers"
    / "external_info_server.py"
)


def create_external_mcp_client() -> Client:
    """
    Create a FastMCP client for the local
    stdio MCP server.

    The client will start the MCP server
    Python process when the connection opens.
    """

    return Client(
        EXTERNAL_INFO_SERVER_PATH
    )


async def list_external_tools() -> List[
    BaseTool
]:
    """
    Discover MCP tools and adapt them into
    LangChain BaseTool objects.
    """

    client = (
        create_external_mcp_client()
    )

    async with MCPAdapter(
        client
    ) as adapter:
        tools = await (
            adapter.list_tools(
                cache_mode="refresh"
            )
        )

        return tools
