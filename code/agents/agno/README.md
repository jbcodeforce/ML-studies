---
title: Agno personal studies
Updated: 07/2026
---

# Agno personal studies

Agno agent experiments with local LLMs (Ollama, or oMLX). This readme references existing code in this folder ,adn how to test it. The [agno chapter summarizes](https://jbcodeforce.github.io/ML-studies/genAI/agno/) the studies.

## Navigate to Agno source and API inside VScode or Cursor

To jump to agno source with **F12** (or Cmd+Click):

1. **Interpreter** – The venv with `agno` is at `code/.venv`. The project sets this via `.vscode/settings.json` and `pyrightconfig.json` (venvPath + venv). If you still get “no definition found”:
   - Open Command Palette → **Python: Select Interpreter**.
   - Pick the one under **ML-studies/code/.venv** (e.g. `Python 3.12.x ('.venv': venv)` with path ending in `ML-studies/code/.venv`).
2. **Sync** – Run `uv sync` from `ML-studies/code` or from `src/agentic/agno` so `agno` is installed in that venv.
3. **Reload** – Reload the window (Command Palette → **Developer: Reload Window**) after changing the interpreter or config.
4. **Multi-root** – If the workspace has several roots (e.g. MyAIAssistant + ML-studies), ensure the interpreter you select is the one from **ML-studies/src/.venv**, not another project’s venv.

In **Cursor**: add `https://docs.agno.com/llms-full.txt` under Preferences → Cursor Settings → Indexing & Docs so the AI can use the Agno API when coding.

### GitHub source (libs/agno/agno)

Source tree: [agno-agi/agno – libs/agno/agno](https://github.com/agno-agi/agno/tree/main/libs/agno/agno)

#### Agno cookbook relevant examples

* [Agentic Search over Knowledge](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_search_over_knowledge.py), which is implemented with flink knowledge in [knowledge](./knowledge). 
* [State management](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_state_management.py)
* [Typed input-output](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_typed_input_output.py)

## MLX agent

Start an oMLX server: [start_oLMX.sh](./start_omlx.sh), and [cursor_omlx.md](./cursor_omlx.md) for Cursor IDE setup, or use a remote LAN server. Set environment variables in an .env file, and reference this file with ML_ENV_FILE.

```sh
export ML_ENV_FILE=.env
```

The agent in `first_mlx_agent_with_tool` uses an OpenAI-compatible API. 

```bash
cd coge/agents/agno
uv run python first_mlx_agent_with_tool.py
```

Default server URL is `http://127.0.0.1:7999`. Override with the `LLM_BASE_URL` environment variable or pass `base_url` to `create_mlx_agent()`.

### Using the agent in code

```python
from first_mlx_agent_with_tool import create_mlx_agent

agent = create_mlx_agent()  # uses MLX_BASE_URL or http://127.0.0.1:1337
agent.print_response("What is 2+2?", stream=True)

# Custom URL or no tools
agent = create_mlx_agent(base_url="http://localhost:5000", tools=[])
```

## How to from Agno cookbook

### Use user preferences

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

## For Cursor Configuration with Local llm and grok

See openai url as: https://amperage-earthly-reacquire.ngrok-free.dev/v1 using ngrok.com. 


## List of samples in this folder

### Root-level scripts

| Source | Intent |
|--------|--------|
| [`first_mlx_agent_with_tool.py`](./first_mlx_agent_with_tool.py) | Finance agent backed by an MLX LLM via an OpenAI-compatible server. Demonstrates instructions, tools (DuckDuckGo, YFinance), SQLite session storage, structured output (`BaseModel`), streaming, and datetime context. Entry point: `uv run python first_mlx_agent_with_tool.py` or `agno-mlx`. |
| [`ollama_agent_with_tool.py`](./ollama_agent_with_tool.py) | Same finance-agent pattern as above, using Ollama with native tool calling. Baseline for comparing Ollama vs MLX/oMLX tool support. |
| [`ollama_self_learning_agent_with_tool.py`](./ollama_self_learning_agent_with_tool.py) | Self-learning agent: saves insights to a knowledge base with human-in-the-loop confirmation before persisting learnings. Extends the Ollama finance agent with memory and knowledge patterns from the Agno cookbook. |
| [`ollama_knowledge.py`](./ollama_knowledge.py) | Agentic search over a Flink knowledge base (Chroma vector store + SqliteDb contents). Implements the [agent search over knowledge](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_search_over_knowledge.py) cookbook pattern. |
| [`first_agent_os.py`](./first_agent_os.py) | Exposes the finance agent through Agno AgentOS (FastAPI) so it can be used from [os.agno.com](https://os.agno.com/). Uses [`config.yaml`](./config.yaml) for quick prompts. |
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
