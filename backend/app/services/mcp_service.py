import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

from dotenv import load_dotenv
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

load_dotenv(
    PROJECT_ROOT / ".env"
)


EXTERNAL_INFO_SERVER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "mcp_servers"
    / "external_info_server.py"
)


def build_external_mcp_config() -> Dict[
    str,
    Any,
]:
    """
    Build the MCP multi-server configuration.

    external_info:
        NodAgent local MCP server.

    github:
        Official GitHub MCP server running
        through Docker.
    """

    github_token = os.getenv(
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    )

    servers: Dict[
        str,
        Dict[str, Any],
    ] = {
        "external_info": {
            "command": sys.executable,
            "args": [
                str(
                    EXTERNAL_INFO_SERVER_PATH
                )
            ],
        }
    }

    if github_token:
        servers["github"] = {
            "command": "docker",

            "args": [
                "run",
                "-i",
                "--rm",

                "-e",
                (
                    "GITHUB_PERSONAL_ACCESS_TOKEN"
                ),

                "-e",
                "GITHUB_READ_ONLY=1",

                "-e",
                (
                    "GITHUB_TOOLSETS="
                    "repos,issues,"
                    "pull_requests"
                ),

                (
                    "ghcr.io/github/"
                    "github-mcp-server"
                ),
            ],

            "env": {
                (
                    "GITHUB_PERSONAL_ACCESS_TOKEN"
                ): github_token,
            },
        }

    return {
        "mcpServers": servers
    }


def create_external_mcp_client() -> Client:
    """
    Create one aggregated FastMCP client.

    It can connect to:

    - NodAgent local external-info server
    - Official GitHub MCP server
    """

    config = (
        build_external_mcp_config()
    )

    return Client(config)


async def list_external_tools() -> List[
    BaseTool
]:
    """
    Discover tools from every configured
    MCP server and adapt them into
    LangChain BaseTool objects.

    This function is intended for discovery
    and diagnostics.
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
