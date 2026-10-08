---
title: Agno personal studies
Updated: 10/2026
---

# Agno personal studies

Agno agent experiments with local LLMs (Ollama, or oMLX). This readme references existing code in this folder, and how to test it. The [agno chapter summarizes](https://jbcodeforce.github.io/ML-studies/genAI/agno/) the studies.

Upgrade to the [last agno SDK](https://www.agno.com/products/sdk) to get some new features. The folder structure tries to reflect agno/cookbook folder.


## oMLX agents

Two ways to access oMLX server: 1/ remote on another server on LAN, or thunderbolt connection (169.254.130.185 (M5) or 183 (M3)). The proposed setting is to use both, for example the embedder on the local machine as it needs less memory.

For local, start an oMLX server: [start_oLMX.sh](./start_omlx.sh). Verify with:
```bash
curl -s http://127.0.0.1:7999/v1/models -H "Authorization: Bearer local-key" | jq .
#
curl -s http://169.254.130.185:7999/v1/models -H "Authorization: Bearer local-key" | jq .
```

Set environment variables in an .env file, and reference this file with ML_ENV_FILE.

```sh
export ML_ENV_FILE=.env
```

The .env has the following env variables:
```sh
LLM_BASE_URL=
LLM_MODEL=
LLM_TEMPERATURE=0.4
LLM_API_KEY=local-key
EMBEDDER_MODEL=bge-small-en-v1.5-8bit
EMBEDDER_BASE_URL=http://10.0.0.148:7999/v1
```

The config.py prepare environment variables and can be reused for all omlx demos.

The agent in `first_mlx_agent_with_tool` uses an OpenAI-compatible API. pointing to oMLX server on LAN.

```bash
cd code
uv run python -m agents.agno._00_quickstart.first_mlx_agent_with_tool
```

## Cursor + local oMLX (Codestral on :7999)

| Setting | Value |
|---------|--------|
| oMLX base URL | `http://127.0.0.1:7999/v1` |
| API key | `local-key` |
| Model id (from `GET /v1/models`) | `Codestral-22B-v0.1-4bit` |
| Admin UI | http://127.0.0.1:7999/admin |

### Configuration

- **User settings**: `cursor.openai.baseUrl` → `http://127.0.0.1:7999/v1` in `~/Library/Application Support/Cursor/User/settings.json`
- **Cursor state DB**: `openAIBaseUrl`, `cursorAuth/openAIKey`, and custom model names
- **MCP tools** in (`~/.cursor/mcp.json`) 

After changes, **reload Cursor** (Developer: Reload Window).

### Navigate to Agno source and API inside VScode or Cursor

To jump to agno source with **F12** (or Cmd+Click):

1. **Interpreter** – The venv with `agno` is at `code/.venv`. The project sets this via `.vscode/settings.json` and `pyrightconfig.json` (venvPath + venv). If you still get “no definition found”:
   - Open Command Palette → **Python: Select Interpreter**.
   - Pick the one under **ML-studies/code/.venv** (e.g. `Python 3.12.x ('.venv': venv)` with path ending in `ML-studies/code/.venv`).
2. **Sync** – Run `uv sync` from `ML-studies/code` or from `src/agentic/agno` so `agno` is installed in that venv.
3. **Reload** – Reload the window (Command Palette → **Developer: Reload Window**) after changing the interpreter or config.
4. **Multi-root** – If the workspace has several roots (e.g. MyAIAssistant + ML-studies), ensure the interpreter you select is the one from **ML-studies/src/.venv**, not another project’s venv.

In **Cursor**: add `https://docs.agno.com/llms-full.txt` under Preferences → Cursor Settings → Indexing & Docs so the AI can use the Agno API when coding.


## List of studies and agents in this folder

### Root-level scripts

| Source | Intent | Status |
|--------|--------|--------|
| [`first_mlx_agent_with_tool.py`](./_00_quickstart/first_mlx_agent_with_tool.py) | Finance agent backed by an oMLX LLM via an OpenAI-compatible server. Demonstrates instructions, tools ( YFinance), SQLite session storage, structured output (`BaseModel`), streaming, and datetime context. It uses prompt based workflow | 10/2026 returns wrong results | 
| [`first_agent_os.py`](./first_agent_os.py) | Exposes the finance agent through Agno AgentOS (FastAPI) so it can be used from [os.agno.com](https://os.agno.com/). Uses [`config.yaml`](./config.yaml) for quick prompts. |


| [`ollama_agent_with_tool.py`](./ollama/ollama_agent_with_tool.py) | Same finance-agent pattern as above, using Ollama with native tool calling. Baseline for comparing Ollama vs MLX/oMLX tool support. | |
| [`ollama_self_learning_agent_with_tool.py`](./ollama/ollama_self_learning_agent_with_tool.py) | Self-learning agent: saves insights to a knowledge base with human-in-the-loop confirmation before persisting learnings. Extends the Ollama finance agent with memory and knowledge patterns from the Agno cookbook. | |
| [`ollama_knowledge.py`](./ollama_knowledge.py) | Agentic search over a Flink knowledge base (Chroma vector store + SqliteDb contents). Implements the [agent search over knowledge](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_search_over_knowledge.py) cookbook pattern. |

| [`deep_researcher.py`](./deep_researcher.py) | Single-file deep researcher: reads a research paper (file upload), summarizes it, and proposes a learning path. Uses Ollama via OpenAI-compatible API. |
| [`olmx_deep_researcher.py`](./olmx_deep_researcher.py) | Same deep-researcher pattern as `deep_researcher.py`, targeting a local oMLX server (`:7999`). |
| [`olmx_learning.py`](./olmx_learning.py) | LearningMachine demo: oMLX/Codestral for chat, Ollama for background extraction of user profile and memories. Documents the split when local models lack reliable tool calling. |
| [`startoLMX.sh`](./startoLMX.sh) | Starts oMLX on `http://127.0.0.1:7999/v1` with models from `~/.lmstudio/models`. |
| [`cursor_omlx.md`](./cursor_omlx.md) | Cursor IDE configuration for routing chat/completions to local oMLX. |

### [`deep_researcher/`](./deep_researcher/)

Step-by-step build of a multi-agent investment research system based on [Agno deep research](https://docs.agno.com/use-cases/deep-research/overview). Modular layout with dedicated agent definitions and tests.

| Source | Intent |
|--------|--------|
| [`deep_research_agents.py`](./deep_researcher/deep_research_agents.py) | Agent definitions: market analyst (DuckDuckGo + YFinance), financial analyst, technical analyst, risk officer, memo writer, committee chair. |
| [`main.py`](./deep_researcher/main.py) | Workflow entry point: wires agents into a `Parallel` + `Step` pipeline for investment memo generation. |
| [`tests/`](./deep_researcher/tests/) | Integration tests for YFinance, DuckDuckGo, agent wiring, and workflow execution. |

### [`llm-wiki/`](./llm-wiki/)

Karpathy-style personal wiki: immutable sources, curated markdown pages, SqliteDb sessions, and Chroma embeddings. See [`llm-wiki/README.md`](./llm-wiki/README.md).

| Source | Intent |
|--------|--------|
| [`wiki_cli.py`](./llm-wiki/wiki_cli.py) | CLI entry point: `chat`, `ask`, `ingest`, `reindex`, `index-folder`. |
| [`llm_wiki/agent.py`](./llm-wiki/llm_wiki/agent.py) | Wiki agent factory with knowledge retrieval over `wiki/` and indexed corpus. |
| [`llm_wiki/indexing.py`](./llm-wiki/llm_wiki/indexing.py) | Embed and index markdown into Chroma. |
| [`llm_wiki/tools.py`](./llm-wiki/llm_wiki/tools.py) | Agent tools for reading and writing wiki pages. |
| [`wiki/`](./llm-wiki/wiki/) | Curated markdown knowledge base (pages, `index.md`, `log.md`). |

### [`workflows/`](./workflows/)

Agno workflow examples running locally. See [`workflows/README.md`](./workflows/README.md).

| Source | Intent |
|--------|--------|
| [`daily_ai_news_search_summary.py`](./workflows/daily_ai_news_search_summary.py) | Four-step workflow: prepare search input, research team (HackerNews + web search), prepare writer input, summary writer. Demonstrates `Step` events, session/run IDs, team composition, and SQLite workflow persistence. |



## Use user preferences

* [Works with Memory manager](https://docs.agno.com/memory/working-with-memories/overview) to keep user preference, with instructions that should include:
    ```markdown
    ## Memory

    You have memory of user preferences (automatically provided in context). Use this to:
    - Tailor recommendations to their interests
    - Consider their risk tolerance
    - Reference their investment goals
    ```
    And add this to the agent:
    ```python
        db=agent_db,
        memory_manager=memory_manager,
        enable_agentic_memory=True,
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_runs=3,
    ```

* Use ` add_history_to_context=True` to keep multi-turn conversations. [see history doc.](https://docs.agno.com/database/chat-history)
    ```python
    # Get user-assistant message pairs
    chat_history = agent.get_chat_history(session_id="chat_123")

    # Get all messages from the session
    messages = agent.get_session_messages(session_id="chat_123")

    # Get the last run output with metrics
    last_run = agent.get_last_run_output()
    ```

* Human in a loop before executing tool: [example](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/human_in_the_loop.py)
    ```python
    @tool(requires_confirmation=True)
    def save_learning(title: str, learning: str) -> str:
        ...
    
        Agent(
            ...
            tools= [save_learnings]
            knowledge=learnings_kb,
            search_knowledge=True,
        )
    ```
## Agno cookbook relevant examples

* [Agentic Search over Knowledge](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_search_over_knowledge.py), which is implemented with flink knowledge in [knowledge](./knowledge). 
* [State management](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_state_management.py)
* [Typed input-output](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_typed_input_output.py)
