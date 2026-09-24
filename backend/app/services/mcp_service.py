import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import (
    StreamableHttpTransport,
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


GITHUB_MCP_URL = os.getenv(
    "GITHUB_MCP_URL",
    "http://127.0.0.1:8082",
)


def create_external_info_client() -> Client:
    """
    NodAgent local MCP server.

    Development transport:
    stdio.
    """

    return Client(
        EXTERNAL_INFO_SERVER_PATH
    )


def create_github_mcp_client() -> Client:
    """
    GitHub official MCP server.

    Transport:
    Streamable HTTP.

    GitHub HTTP MCP requires the client
    to send its GitHub access token in
    the Authorization header.
    """

    github_token = os.getenv(
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    )

    if not github_token:
        raise RuntimeError(
            "GITHUB_PERSONAL_ACCESS_TOKEN "
            "is not configured."
        )

    transport = (
        StreamableHttpTransport(
            url=GITHUB_MCP_URL,
            headers={
                "Authorization": (
                    f"Bearer "
                    f"{github_token}"
                ),

                "X-MCP-Readonly": (
                    "true"
                ),

                "X-MCP-Toolsets": (
                    "repos,"
                    "issues,"
                    "pull_requests"
                ),
            },
        )
    )

    return Client(
        transport=transport
    )
