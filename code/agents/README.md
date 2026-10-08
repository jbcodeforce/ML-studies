# Agents

Agentic AI with Agno and old studies in LangGraph, as I do not like LG, I will remove code once moved to agno.

## Layout

| Path | Description |
| --- | --- |
| `agno/` | Agno agents (Ollama, MLX, deep researcher) |
| `langgraph/` | LangGraph ReAct, RAG, human-in-the-loop |

## Environment

```sh
cd code && uv sync --extra agents
# in code
source .venv/bin/activate
```

## Run example like

```sh
uv run python -m agents.agno.first_mlx_agent_with_tool
```

See [agentic.md](../../docs/genAI/agentic.md) and [agno.md](../../docs/genAI/agno.md).
