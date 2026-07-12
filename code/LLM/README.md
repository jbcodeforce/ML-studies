# LLM

LangChain, LlamaIndex, Ollama, HuggingFace, and shared `mlstudies` utilities.

## Layout

| Path | Description |
| --- | --- |
| `langchain/` | Provider integrations, RAG, Q&A, feature stores |
| `llamaindex/` | LlamaIndex query engines |
| `ollama/` | Local Ollama chat scripts |
| `huggingface/` | HuggingFace Hub experiments |
| `shared/mlstudies/` | Shared config, LLM, and RAG factories |

## Environment

```sh
cd code && uv sync --extra llm
# Provider-specific extras: --extra openai, --extra anthropic, etc.
```

See [coding/langchain.md](../../docs/coding/langchain.md) and [genAI/review.md](../../docs/genAI/review.md).
