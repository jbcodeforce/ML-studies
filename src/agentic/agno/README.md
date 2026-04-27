# Agno agentic studies

Agno agent experiments with local LLMs (Ollama, or MLX).

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
* [storage](https://docs.agno.com/database/session-storage) for conversation history. Sessions are stored automaticaly once a database is added to the agent
* [memory](https://docs.agno.com/memory/overview) for  user preferences
* [state] is structured data the agent actively manages: counters, lists, flags. An agent can use across runs. State variables can be injected into instructions with {variable_name}


## MLX agent

The agent in `first_mlx_agent_with_tool` uses an OpenAI-compatible API. Start an MLX server (e.g. [mlx-llm-server](https://pypi.org/project/mlx-llm-server/) or [mlx-openai-server](https://github.com/cubist38/mlx-openai-server)) then run:

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
