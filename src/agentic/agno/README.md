# Agno agentic studies

Agno agent experiments with local LLMs (Ollama, or oMLX).

## Navigate to Agno source and API

To jump to agno source with **F12** (or Cmd+Click):

1. **Interpreter** – The venv with `agno` is at `src/.venv`. The project sets this via `.vscode/settings.json` and `pyrightconfig.json` (venvPath + venv). If you still get “no definition found”:
   - Open Command Palette → **Python: Select Interpreter**.
   - Pick the one under **ML-studies/src/.venv** (e.g. `Python 3.12.x ('.venv': venv)` with path ending in `ML-studies/src/.venv`).
2. **Sync** – Run `uv sync` from `ML-studies/src` or from `src/agentic/agno` so `agno` is installed in that venv.
3. **Reload** – Reload the window (Command Palette → **Developer: Reload Window**) after changing the interpreter or config.
4. **Multi-root** – If the workspace has several roots (e.g. MyAIAssistant + ML-studies), ensure the interpreter you select is the one from **ML-studies/src/.venv**, not another project’s venv.

In **Cursor**: add `https://docs.agno.com/llms-full.txt` under Settings → Indexing & Docs so the AI can use the Agno API when coding.

### GitHub source (libs/agno/agno)

Source tree: [agno-agi/agno – libs/agno/agno](https://github.com/agno-agi/agno/tree/main/libs/agno/agno)

| Import | Source (GitHub) |
|--------|-----------------|
| `agno.agent` | [agent/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/agent) |
| `agno.models.ollama` | [models/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models) |
| `agno.models.openai.like` | [models/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models) |
| `agno.tools` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.tools.duckduckgo` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.tools.yfinance` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.db.sqlite` | [db/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/db) |
| `agno.knowledge` | [knowledge/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/knowledge) |
| `agno.utils` | [utils/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/utils) |
| `agno.vectordb.chroma` | [vectordb/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/vectordb) |

## Core Concepts

* [Agents](https://docs.agno.com/agents/overview) are a stateful control loop around a stateless LLM. 
* [Database](https://docs.agno.com/database/overview) to get persistent storage for sessions, context, memory, learnings, and evaluation datasets.
* [Storage](https://docs.agno.com/database/session-storage) for conversation history. Sessions are stored automaticaly once a database is added to the agent
* [Memory](https://docs.agno.com/memory/overview) for  user preferences
* [State] is structured data the agent actively manages: counters, lists, flags. An agent can use across runs. State variables can be injected into instructions with {variable_name}

## Development approach

* Declare and unit test all the tools to be used
* Prepare some queries  to validate
* Use the same integration pattern with backend LLM, externalize URL, API keys, model reference.
* Fine tune the instructions
* Integrate agent in Workflow and/r Team


## MLX agent

The agent in `first_mlx_agent_with_tool` uses an OpenAI-compatible API. Start an MLX server (e.g. [mlx-llm-server](https://pypi.org/project/mlx-llm-server/) or [mlx-openai-server](https://github.com/cubist38/mlx-openai-server)).

MLX being slow, 05/2026, [oMLX](https://github.com/jundot/omlx) is a prefered choice: see [olmx_deep_researcher.py](./olmx_deep_researcher.py), [startoLMX.sh](./startoLMX.sh), and [cursor_omlx.md](./cursor_omlx.md) for Cursor IDE setup.

```bash
# Terminal 1: start MLX server (example with mlx-llm-server)
pip install mlx-llm-server
mlx-llm-server --model /path/to/mlx-model

# Terminal 2: run the agent
uv run python first_mlx_agent_with_tool.py
# or after install: agno-mlx
```

Default server URL is `http://127.0.0.1:1337`. Override with the `MLX_BASE_URL` environment variable or pass `base_url` to `create_mlx_agent()`.

### Using the agent in code

```python
from first_mlx_agent_with_tool import create_mlx_agent

agent = create_mlx_agent()  # uses MLX_BASE_URL or http://127.0.0.1:1337
agent.print_response("What is 2+2?", stream=True)

# Custom URL or no tools
agent = create_mlx_agent(base_url="http://localhost:5000", tools=[])
```

It looks the MLX models are not supporting tool calling.

## Ollama with tools

Use Ollama code: (ex. ollama_agent_with_tool.py which has tool calling and structure input/output)

```python
from agno.models.ollama import Ollama
...
model = Ollama(
    id=DEFAULT_LLM_MODEL
)
```

To run:
```sh
uv run python ollama_agent_with_tool.py
```

## Ollama with persistence, knowledge, self learning, human in the loop

To run:
```sh
uv run python ollama_self_learning_agent_with_tool.py.py
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

## Agno cookbook relevant examples

* [Agentic Search over Knowledge](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_search_over_knowledge.py), which is implemented with flink knowledge in [ollama_knowledge](./ollama_knowledge.py). This example use db for Agent to keep session: run is with `uv run ollama_knowledge.py ` 
* [State management](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_state_management.py)
* [Typed input-output](https://github.com/agno-agi/agno/blob/main/cookbook/00_quickstart/agent_with_typed_input_output.py)


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
