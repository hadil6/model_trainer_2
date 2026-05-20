"""MCP server connection configuration.

Each server runs as an independent Python process spoken to over stdio.
`MultiServerMCPClient` from `langchain_mcp_adapters` takes this dict and
spawns/connects to every server, aggregating their `@mcp.tool()` functions
into a single list of LangChain-compatible tools.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root — the CWD each MCP server process should resolve imports from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Absolute path to the Python interpreter that launched this process.
# Critical on Windows where `python` without a full path may resolve to a
# different venv than the one running the orchestrator.
PYTHON_BIN = sys.executable or "python"


def _server_path(name: str) -> str:
    return str(PROJECT_ROOT / "mcp_servers" / f"{name}.py")


MCP_SERVERS_CONFIG: dict[str, dict] = {
    "data_prep": {
        "command":   PYTHON_BIN,
        "args":      [_server_path("data_prep_server")],
        "transport": "stdio",
        "env": {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONUNBUFFERED": "1",
        },
    },
    "evaluation": {
        "command":   PYTHON_BIN,
        "args":      [_server_path("evaluation_server")],
        "transport": "stdio",
        "env": {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONUNBUFFERED": "1",
        },
    },
    "storage": {
        "command":   PYTHON_BIN,
        "args":      [_server_path("storage_server")],
        "transport": "stdio",
        "env": {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONUNBUFFERED": "1",
        },
    },
}


# Tool namespacing: the data agent is the only planner and owns every tool.
DATA_AGENT_SERVERS: list[str] = ["data_prep", "evaluation", "storage"]


def filter_config(servers: list[str]) -> dict[str, dict]:
    """Return a subset of MCP_SERVERS_CONFIG restricted to the given names."""
    return {name: MCP_SERVERS_CONFIG[name] for name in servers if name in MCP_SERVERS_CONFIG}
