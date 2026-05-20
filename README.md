# Model Trainer — Fine-Tuning Prep Pipeline

A LangGraph + MCP pipeline that ingests raw documents, produces a formatted
Q&A fine-tuning dataset (Alpaca / ShareGPT), and judges it for fine-tuning
readiness — all inside one data agent under a thin orchestrator.

## Architecture

```
    ┌────────────────────────┐
    │  orchestrator_agent    │  Thin supervisor — calls the data agent
    └──────────┬─────────────┘
               ▼
         ┌────────────┐
         │ data_agent │       LangGraph StateGraph ReAct loop
         └──────┬─────┘
                ▼
    MCP servers (stdio, one process each):
      • data_prep_server  — profile / clean / generate_qa / dedup /
                            evaluate_dataset_readiness / export
      • evaluation_server — score_qa_pairs / llm_as_judge /
                            generate_eval_report
      • storage_server    — read_file / read_pdf / write_json[l]
```

The data agent runs a single ReAct loop (`plan → act → observe`) driven by an
Azure OpenAI chat model. Tools come from FastMCP servers via
`langchain-mcp-adapters`.

## Layout

```
    agents/
      data_agent.py          StateGraph: ingest → QA → export → judge
      orchestrator_agent.py  Supervisor (route → data → aggregate)

    mcp_servers/
      data_prep_server.py    Profilers + Q&A + dedup + readiness + export
      evaluation_server.py   LLM scoring + LLM-as-judge + eval_report writer
      storage_server.py      read_file / read_pdf / write_json[l]

    shared/
      state.py               TypedDict states (DataAgent, Orchestrator)
      prompts.py             System prompts + ReAct output contract
      mcp_config.py          MCP server launch configuration

    app/                     Domain code reused by the MCP servers
      core/config.py         Settings (Azure OpenAI creds, embedding, …)
      domain/profilers/      CSV/PDF/TXT/XLSX profilers
      domain/services/       LLM client, readiness judge, QA generation
      schemas/               Pydantic artifacts (QAPair, ProfileArtifact, …)

    main.py                  Typer CLI (data | run | serve)
    app/api.py               FastAPI HTTP layer (run + SSE log stream)
    frontend/                React + Vite + TypeScript UI
    requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt
cp .env.example .env                # then fill in Azure credentials
```

Required env vars (see `.env.example`):

- `AZURE_API_KEY`
- `AZURE_ENDPOINT`
- `AZURE_API_VERSION` (default `2025-04-01-preview`)
- `DEPLOYMENT_NAME` (default `gpt-5.4-mini`)

## Run

### Data agent directly

```bash
python main.py data \
  --files docs/a.pdf docs/b.pdf \
  --domain medical --language fr \
  --target-model Qwen/Qwen2.5-7B
```

### Through the orchestrator

```bash
python main.py run \
  --files docs/a.pdf \
  --goal "Fine-tune a medical QA bot" \
  --domain medical --language fr \
  --target-model Qwen/Qwen2.5-7B
```

### Web UI (recommended)

Start the backend (serves `POST /api/runs`, SSE at `/api/runs/{id}/events`):

```bash
python main.py serve --host 127.0.0.1 --port 8000
```

In a second terminal, install + run the frontend:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite proxies every `/api/*` call to `http://127.0.0.1:8000`, so no extra
config is needed. Open the browser, upload files, fill in the task / language /
domain / target model, and click **Run pipeline** — agent and MCP-server logs
stream live into the page until the final verdict is shown.

## Data agent workflow

See [shared/prompts.py::DATA_AGENT_SYSTEM](shared/prompts.py):

    1.  profile_file                 — per file
    2.  semantic_summarize           — per profiled file
    3.  clean_text                   — per pdf/txt
    4.  review_ocr_relevance         — only if ocr_pages > 0
    5.  assess_readiness             — once
    6.  check_files_coherence        — MANDATORY (hard stop if < 0.6)
    7.  generate_qa                  — per file, up to 2 passes
    8.  evaluate_and_refine_qa       — once
    9.  deduplicate_qa               — once
    10. evaluate_dataset_readiness   — once (6-axis judge)
    11. score_qa_pairs               — once (numeric LLM scoring)
    12. llm_as_judge                 — once (per-pair PASS/FAIL)
    13. export_sharegpt | _alpaca    — once
    14. generate_eval_report         — writes eval_report.json
    15. (synthesize_multiturn)       — optional
    16. finish

Verdict thresholds (data agent + final report):

- `READY` — every axis ≥ 0.70
- `WARN` — at least one axis in [0.50, 0.70), none < 0.50
- `NOT_READY` — any axis < 0.50 OR a blocking issue was raised

## Orchestrator behavior

`route → data → route → aggregate → end`. Deterministic guards take precedence
over the LLM:

- `files` present + no `data_agent_result` → `data_agent`
- `data_agent_result` present → `aggregate`
- `route_count ≥ 4` → force `aggregate` (infinite-loop guard)

## Debugging

- MCP server logs go to **stderr**; stdout is reserved for the MCP protocol.
- `PYTHONUNBUFFERED=1` is set in `mcp_config.py` for live log streaming.
- Iteration caps: data agent = `max(30, 6·n_files + 15)`, orchestrator = 4 routes.
