"""Data Agent — LangGraph ReAct (plan → act → observe).

Public API
----------
run_data_agent(inputs: DataAgentInput) -> dict

Files must already be saved to artifacts/{job_id}/input_files/ before calling.
The agent connects to the MCP server at http://localhost:{mcp_port}/sse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph

from agents.data_agent.state import DataAgentInput, DataAgentState
from agents.llm_client import get_llm_client
from core.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Data Preparation Agent. Your job is to prepare a fine-tuning dataset from uploaded files.

CORE PRINCIPLE: All uploaded files must be treated as ONE unified knowledge source.
Extract text from every file, merge everything into a single corpus, then generate
Q&A pairs from that unified corpus — NOT file by file.

TOOL EXECUTION ORDER (follow this sequence precisely):

── PHASE 1 : Profile & clean every file ──────────────────────────────────────

For EACH uploaded file (repeat steps 1-4 for every filename in the list):
  1. profile_file_tool(job_id, filename)
  2. summarize_file(job_id, filename, task)
  3. clean_text_tool(job_id, filename)              [PDF and TXT only — skip for CSV/XLSX]
  4. review_ocr_relevance(job_id, filename, domain) [PDF only, if OCR pages exist — skip otherwise]

── PHASE 1b : Free GPU after all profiling done ──────────────────────────────

  5. release_gpu(job_id)
     → Frees marker-pdf and sentence-transformer VRAM inside the MCP server.
     → Call ONCE after ALL files have completed steps 1-4.
     → Critical: ensures the GPU is free before SWIFT training starts.

── PHASE 2 : Analyse the dataset as a whole ──────────────────────────────────

  6. detect_domain(job_id)                          [once, after ALL files profiled]
  7. assess_readiness(job_id, task, user_prompt=user_goal)
     → Pass user_goal from the state snapshot as user_prompt (NOT task).
     → If verdict == "BLOCKED": call finish immediately.
  8. check_coherence(job_id, task)
     → If score < 0.6: call finish immediately.

── PHASE 3 : Build the unified corpus ────────────────────────────────────────

  9. merge_corpus(job_id)
     → Merges ALL cleaned/extracted texts into artifacts/{job_id}/merged_corpus.txt
     → This single file is the sole input for QA generation.

── PHASE 4 : Generate pairs from the unified corpus (two passes) ─────────────

  10. generate_qa(job_id, filename="merged_corpus", task, pass_num=1, model_id=target_model)
  11. generate_qa(job_id, filename="merged_corpus", task, pass_num=2, model_id=target_model)
     → Always use filename="merged_corpus" — never generate per original file.
     → Always pass model_id=target_model so the correct dataset format is used.
     → For task="summarization": generates (document→summary) pairs with larger chunks (2000 chars).
     → For all other tasks: generates Q&A pairs with standard chunks (500 chars).

── PHASE 5 : Refine, validate, export ────────────────────────────────────────

  12. evaluate_and_refine_qa(job_id, task, sample_size=20)
  13. deduplicate_qa(job_id)
  14. evaluate_readiness(job_id, task, target_peft="qlora", target_model=target_model)
  15. auto_fill_qa(job_id, target_model, task, target_peft="qlora", model_id=target_model)
      → Only if readiness verdict is NOT_READY or READY_WITH_WARNINGS (insufficient pairs).
      → Skip if verdict is READY.
      → Always pass model_id=target_model to ensure consistent dataset format.
  16. finish(job_id, task, target_model)

PLANNING RULES:
- Call ONE tool at a time.
- Track completed_steps to avoid re-running the same tool on the same file.
- Always inject job_id into every tool call.
- In Phase 4, ALWAYS pass filename="merged_corpus" to generate_qa — never the original filenames.
- For auto_fill_qa: only call it if n_pairs < requirements.target.
- Output ONLY a JSON object: {"tool": "...", "args": {...}, "reasoning": "...", "confidence": 0.0}
- Do not output any text outside the JSON.
"""


def _tool_schema_str(tools: list[BaseTool]) -> str:
    lines: list[str] = []
    for t in tools:
        schema = {}
        if t.args_schema:
            try:
                schema = t.args_schema.model_json_schema()
            except Exception:
                try:
                    schema = t.args_schema.schema()
                except Exception:
                    pass
        lines.append(
            f"Tool: {t.name}\n"
            f"  Description: {t.description}\n"
            f"  Args: {json.dumps(schema.get('properties', {}), indent=4)}"
        )
    return "\n\n".join(lines)


def _build_state_snapshot(state: DataAgentState) -> str:
    return json.dumps({
        "job_id": state["job_id"],
        "filenames": state["filenames"],
        "task": state["task"],
        "user_goal": state.get("user_goal", ""),
        "domain": state["domain"],
        "language": state["language"],
        "target_model": state.get("target_model"),
        "iteration": state.get("iteration", 0),
        "completed_steps": state.get("completed_steps", []),
        "qa_count": state.get("qa_count", 0),
        "coherence_passed": state.get("coherence_passed"),
        "readiness_verdict": state.get("readiness_verdict"),
    }, indent=2)


def _extract_action_json(text: str) -> dict | None:
    """Find the last JSON object containing a 'tool' key using balanced-brace scanning."""
    best: dict | None = None
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, escape, j = 0, False, False, i
        while j < len(text):
            c = text[j]
            if escape:
                escape = False
            elif c == "\\" and in_str:
                escape = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i : j + 1])
                            if isinstance(obj, dict) and "tool" in obj:
                                best = obj
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i += 1
    return best


def _parse_action(raw: str, fallback_job_id: str, fallback_task: str) -> dict:
    # Direct parse (LLM returned pure JSON)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Balanced-brace scan — finds the last well-formed action object
    obj = _extract_action_json(raw)
    if obj is not None:
        return obj

    logger.warning("plan_parse_failed raw=%s", raw[:200])
    return {
        "tool": "finish",
        "args": {"job_id": fallback_job_id, "task": fallback_task},
        "reasoning": "parse error fallback",
        "confidence": 0.5,
    }


def _extract_tool_result(raw: Any) -> dict:
    """Normalise MCP tool output to a plain dict.

    langchain-mcp-adapters returns results as a list of content blocks:
      [{"type": "text", "text": "<JSON string>", "id": "..."}]
    We extract the first text block and parse it as JSON.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"raw": text}
        return {"raw": str(raw)}
    return {"raw": str(raw)}


async def run_data_agent(inputs: DataAgentInput) -> dict:
    """Entry point. Files must already be in artifacts/{job_id}/input_files/."""
    settings = get_settings()
    mcp_url = f"http://localhost:{settings.mcp_port}/sse"

    # ── Fetch tools once (each .ainvoke() call opens its own session) ─────────
    client = MultiServerMCPClient({
        "data_tools": {
            "url": mcp_url,
            "transport": "sse",
            "timeout": 7200,
            "sse_read_timeout": 7200,
        }
    })
    tools: list[BaseTool] = await client.get_tools()
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    logger.info(
        "data_agent tools loaded: %s", [t.name for t in tools]
    )

    tool_schema_text = _tool_schema_str(tools)

    # ── Node closures ─────────────────────────────────────────────────────────

    async def plan_node(state: DataAgentState) -> dict:
        llm = get_llm_client()
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT + "\n\nAVAILABLE TOOLS:\n" + tool_schema_text,
            },
            {
                "role": "user",
                "content": "Current state:\n" + _build_state_snapshot(state),
            },
        ]
        raw = llm.chat(messages)
        action = _parse_action(raw, state["job_id"], state["task"])
        logger.info(
            "plan job=%s iter=%d tool=%s reason=%s",
            state["job_id"], state.get("iteration", 0),
            action.get("tool"), action.get("reasoning", "")[:80],
        )
        return {
            "messages": [AIMessage(content=json.dumps(action))],
            "iteration": state.get("iteration", 0) + 1,
        }

    async def act_node(state: DataAgentState) -> dict:
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if not last_ai:
            return {"error": "No plan message found"}

        try:
            action = json.loads(last_ai.content)
        except Exception:
            return {"error": "Could not parse plan JSON"}

        tool_name = action.get("tool", "")
        args: dict[str, Any] = action.get("args") or {}

        if "job_id" not in args:
            args["job_id"] = state["job_id"]

        tool = tool_map.get(tool_name)
        if tool is None:
            logger.error("act_unknown_tool tool=%s available=%s", tool_name, list(tool_map))
            observation = {"tool": tool_name, "success": False, "error": f"Unknown tool: {tool_name}"}
        else:
            try:
                raw_result = await tool.ainvoke(args)
                result = _extract_tool_result(raw_result)
                observation = {"tool": tool_name, "success": True, "result": result}
            except Exception as exc:
                logger.error("act_failed tool=%s error=%s", tool_name, exc)
                observation = {"tool": tool_name, "success": False, "error": str(exc)}

        obs_msg = HumanMessage(content=json.dumps(observation, ensure_ascii=False))

        filename = args.get("filename", "")
        step_key = f"{tool_name}:{filename}" if filename else tool_name
        updates: dict[str, Any] = {
            "messages": [obs_msg],
            "completed_steps": [step_key],
        }

        if observation["success"]:
            r = observation.get("result") or {}
            if tool_name == "generate_qa":
                updates["qa_count"] = state.get("qa_count", 0) + (r.get("pairs_generated") or 0)
            elif tool_name == "check_coherence":
                updates["coherence_passed"] = float(r.get("score", 1.0)) >= 0.6
            elif tool_name == "evaluate_readiness":
                updates["readiness_verdict"] = r.get("verdict")
            elif tool_name == "finish":
                updates["result"] = r

        return updates

    def route_after_act(state: DataAgentState) -> str:
        if state.get("result") is not None:
            return END
        if state.get("error"):
            return END
        max_iter = max(20, 5 * len(state.get("filenames", [])) + 10)
        if state.get("iteration", 0) >= max_iter:
            logger.warning("data_agent budget exhausted job=%s", state["job_id"])
            return END
        return "plan"

    # ── Build and run graph ───────────────────────────────────────────────────

    graph = StateGraph(DataAgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("act",  act_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", route_after_act, {"plan": "plan", END: END})
    compiled = graph.compile()

    # Pre-mark files already profiled on disk so the LLM skips re-extraction.
    from data.artifact_store import profile_path as _pp
    pre_done = [
        f"profile_file_tool:{fn}"
        for fn in inputs["filenames"]
        if _pp(inputs["job_id"], fn).exists()
    ]

    initial: DataAgentState = {
        "job_id":           inputs["job_id"],
        "filenames":        inputs["filenames"],
        "task":             inputs["task"],
        "user_goal":        inputs.get("user_goal", ""),
        "domain":           inputs.get("domain", ""),
        "language":         inputs.get("language", "en"),
        "target_model":     inputs.get("target_model"),
        "completed_steps":  pre_done,
        "iteration":        0,
        "qa_count":         0,
        "coherence_passed": None,
        "readiness_verdict": None,
        "messages": [HumanMessage(
            content=f"Start job {inputs['job_id']} — task: {inputs['task']}"
        )],
        "result": None,
        "error":  None,
    }

    max_iter = max(30, 6 * len(inputs["filenames"]) + 15)
    final_state = await compiled.ainvoke(initial, config={"recursion_limit": max_iter * 2 + 10})
    return final_state.get("result") or {
        "job_id": inputs["job_id"],
        "verdict": "incomplete",
        "error": final_state.get("error"),
    }
