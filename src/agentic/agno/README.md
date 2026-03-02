# Agno agentic studies

Agno agent experiments with local LLMs (Ollama, MLX).

## Navigate to Agno source and API

- **[AGNO_REFERENCE.md](AGNO_REFERENCE.md)** – Links to the Agno API docs and GitHub source so you can open the reference or jump to the implementation (e.g. `agno.agent`, `agno.models.ollama`, `agno.tools`).
- In **Cursor**: add `https://docs.agno.com/llms-full.txt` under Settings → Indexing & Docs so the AI can use the Agno API when coding.

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