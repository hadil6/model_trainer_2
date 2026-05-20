"""MCP HTTP/SSE server entry point.

Run with:
    python -m agents.data_agent.mcp_server.server [--port 8001]

Data agent connects via:
    MultiServerMCPClient({"data_tools": {"url": "http://localhost:8001/sse", "transport": "sse"}})
"""

from __future__ import annotations

import argparse
import functools
import logging

import anyio
import uvicorn
from mcp.server.fastmcp import FastMCP

from agents.data_agent.mcp_server.export import tool_finish
from agents.data_agent.mcp_server.generation import (
    tool_auto_fill_qa,
    tool_check_coherence,
    tool_deduplicate_qa,
    tool_detect_domain,
    tool_evaluate_and_refine_qa,
    tool_evaluate_readiness,
    tool_generate_qa,
)
from agents.data_agent.mcp_server.profiling import (
    tool_assess_readiness,
    tool_clean_text,
    tool_merge_corpus,
    tool_profile_file,
    tool_review_ocr_relevance,
    tool_summarize_file,
)
from core.config import get_settings
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)

mcp = FastMCP("data-agent-tools")


# ── Async wrappers for long-running sync tools ────────────────────────────────
# Sync tools block the asyncio event loop, preventing SSE heartbeats (sent every
# 15 s by sse_starlette) from reaching the client. Running them in a thread pool
# keeps the event loop free and avoids spurious "Connection closed" errors on
# large documents.

@functools.wraps(tool_profile_file)
async def _async_profile_file(job_id: str, filename: str) -> dict:
    return await anyio.to_thread.run_sync(lambda: tool_profile_file(job_id, filename))


@functools.wraps(tool_summarize_file)
async def _async_summarize_file(job_id: str, filename: str, task: str = "") -> dict:
    return await anyio.to_thread.run_sync(lambda: tool_summarize_file(job_id, filename, task))


@functools.wraps(tool_generate_qa)
async def _async_generate_qa(
    job_id: str, filename: str, task: str, pass_num: int = 1, model_id: str = ""
) -> dict:
    return await anyio.to_thread.run_sync(
        lambda: tool_generate_qa(job_id, filename, task, pass_num, model_id)
    )


@functools.wraps(tool_auto_fill_qa)
async def _async_auto_fill_qa(
    job_id: str, target_model: str, task: str, target_peft: str = "qlora", model_id: str = ""
) -> dict:
    return await anyio.to_thread.run_sync(
        lambda: tool_auto_fill_qa(job_id, target_model, task, target_peft, model_id)
    )


@functools.wraps(tool_evaluate_and_refine_qa)
async def _async_evaluate_and_refine_qa(job_id: str, task: str, sample_size: int = 20) -> dict:
    return await anyio.to_thread.run_sync(
        lambda: tool_evaluate_and_refine_qa(job_id, task, sample_size)
    )


# Register all tools on the single app instance
mcp.add_tool(_async_profile_file,           name="profile_file_tool")
mcp.add_tool(_async_summarize_file,         name="summarize_file")
mcp.add_tool(tool_clean_text,               name="clean_text_tool")
mcp.add_tool(tool_review_ocr_relevance,     name="review_ocr_relevance")
mcp.add_tool(tool_merge_corpus,             name="merge_corpus")
mcp.add_tool(tool_assess_readiness,         name="assess_readiness")
mcp.add_tool(tool_detect_domain,            name="detect_domain")
mcp.add_tool(tool_check_coherence,          name="check_coherence")
mcp.add_tool(_async_generate_qa,            name="generate_qa")
mcp.add_tool(_async_evaluate_and_refine_qa, name="evaluate_and_refine_qa")
mcp.add_tool(tool_deduplicate_qa,           name="deduplicate_qa")
mcp.add_tool(tool_evaluate_readiness,       name="evaluate_readiness")
mcp.add_tool(_async_auto_fill_qa,           name="auto_fill_qa")
mcp.add_tool(tool_finish,                   name="finish")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Agent MCP Server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    setup_logging()
    port = args.port or get_settings().mcp_port
    logger.info("Starting MCP server on %s:%d", args.host, port)
    uvicorn.run(mcp.sse_app(), host=args.host, port=port, timeout_keep_alive=1800)


if __name__ == "__main__":
    main()
