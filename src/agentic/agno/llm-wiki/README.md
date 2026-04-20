# LLM Wiki (Agno)

Karpathy-style personal wiki: immutable sources in `docs/` from this repo, curated indexing  markdown in `wiki/`, plus Agno **SqliteDb** (sessions + knowledge contents) and **Chroma** (embeddings) under `data/`.

System behavior is driven by [techno/claude/llm-wiki.md](../../../../techno/claude/llm-wiki.md) (override with `LLM_WIKI_RULES_PATH`).

## Layout

```
../../../docs # sources — never modified by the agent
llm-wiki/          
  wiki/          # markdown pages, index.md, log.md
  data/          # contents.db, agent.db, chroma/ (created at runtime)
  llm_wiki/      # Python package
  wiki_cli.py    # entry point
```

## Prerequisites

- **Ollama** running (`ollama serve`) with a chat model (default `mistral:7b-instruct`, override `LLM_WIKI_MODEL`).
- An **embedding** model matching dimensions (default `nomic-embed-text` with **768** dims; set `LLM_WIKI_EMBEDDER` and `LLM_WIKI_EMBED_DIM` if you use another model).

Dependencies come from the parent Agno project: run `uv sync` from `ML-studies/src` (or your project venv with `agno` + `chromadb` installed).

## Commands

From parent directory:

```bash
# Interactive chat (default)
uv run llm-wiki/wiki_cli.py chat

# Single question
uv run llm-wiki/wiki_cli.py ask What is in the wiki about Japan?

# Prompt a full ingest pass for one file already placed under raw/
uv run llm-wiki/wiki_cli.py ingest my-notes.md

# Re-embed every wiki/**/*.md into the vector store
uv run llm-wiki/wiki_cli.py reindex

# Bulk-embed an existing tree (e.g. ML-studies/docs) into the same knowledge base
# Use --dry-run to list files without calling the embedder. From this package directory:
uv run llm-wiki/wiki_cli.py index-folder ../../../../docs --dry-run
uv run llm-wiki/wiki_cli.py index-folder ../../../../docs --ext md,txt --max-files 500
```

Chunks from `index-folder` are stored with metadata `kind: corpus`, `source_root`, and `relative_path` so they stay distinct from `wiki-*` entries created by `reindex`.

## Environment

| Variable | Purpose |
|----------|---------|
| `LLM_WIKI_RULES_PATH` | Full path to the markdown rules file (default: repo `techno/claude/llm-wiki.md`) |
| `LLM_WIKI_MODEL` | Ollama chat model id |
| `LLM_WIKI_EMBEDDER` | Ollama embedding model id |
| `LLM_WIKI_EMBED_DIM` | Embedding vector size (must match the embedder) |
| `LLM_WIKI_MAX_RESULTS` | Max knowledge chunks retrieved per query |

## Module API

```python
from pathlib import Path
from llm_wiki.paths import wiki_paths, ensure_layout
from llm_wiki.agent import create_wiki_agent

paths = wiki_paths(Path("/path/to/llm-wiki"))
ensure_layout(paths)
agent, knowledge = create_wiki_agent(paths)
agent.print_response("Summarize wiki/index.md", markdown=True)
```
