---
name: agno-agent-builder
description: Scaffold agno agents with local tools, system prompts, and persistent memory. Use when building agno agents with custom tools, knowledge bases, or SQLite-backed context.
---

# Agno Agent Builder

A structured skill for scaffolding agno agents with:
- Custom local tools (`@tool` decorator)
- Rich system prompts (`instructions` parameter)
- Persistent memory via `SqliteDb` + `Knowledge` + `ChromaDb`

## Quick Start

```bash
# Start Ollama first (required for local LLM)
ollama serve

# Run an example
cd <skill-dir>/examples/qa-agent
python agent.py
```

## Project Structure

```
your-agent/
├── agent.py              # Main agent entry point
├── tools/
│   ├── custom_tool.py    # Local tool definitions
│   └── additional_tool.py # More tools as needed
├── memory/
│   ├── main.py           # Knowledge base setup
│   └── sqlite_context.py # Raw SQLite for general memory
└── requirements.txt
```

## Components

### 1. Model Setup

```python
from agno.models.ollama import Ollama

model = Ollama(id="llama3.2", temperature=0.4)
```

### 2. Local Tools (`@tool`)

Tools can be plain functions decorated with `@tool`. Set `requires_confirmation=True` for human-in-the-loop approval.

```python
from agno.tools import tool
from agno.db.sqlite import SqliteDb
import json

agent_db = SqliteDb(db_file="tmp/agents.db")

@tool
def save_insight(title: str, insight: str) -> str:
    """Save a reusable insight to memory."""
    payload = {
        "title": title,
        "insight": insight,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    agent_db.insert("memories.json", json.dumps(payload))
    return f"Saved: '{title}'"
```

### 3. System Prompt (`instructions`)

Use multi-line strings with markdown. Structure with headers: Workflow, Rules, Rules.

```python
instructions = """\
You are a QA Agent — an automated software quality assurance specialist.

## Workflow

1. Understand — Clarify what's being tested
2. Plan — Define test cases and acceptance criteria
3. Execute — Run the tests using available tools
4. Report — Summarize findings with severity and status

## Rules

- Always include test case ID in reports
- Severity: Critical / Major / Minor / Info
- Status: Pass / Fail / Blocker / N/A
- Never invent test results — report what tools return
"""
```

### 4. Persistent Memory

Three approaches, from simple to advanced:

#### A. SQLite for structured data

```python
from agno.db.sqlite import SqliteDb

agent_db = SqliteDb(db_file="tmp/agents.db")

agent = Agent(
    model=model,
    instructions=instructions,
    db=agent_db,
    add_datetime_to_context=True,
)
```

#### B. Chroma + Knowledge for semantic search

```python
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

kb = Knowledge(
    name="QA Knowledge",
    vector_db=ChromaDb(
        name="qa-knowledge",
        collection="qa-facts",
        path="tmp/chromadb",
        persistent_client=True,
        search_type=SearchType.hybrid,
        hybrid_rrf_k=60,
        embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
    ),
    max_results=5,
    contents_db=agent_db,
)

agent = Agent(
    model=model,
    instructions=instructions,
    knowledge=kb,
    search_knowledge=True,
)
```

#### C. Files-based memory (simplest)

```python
from pathlib import Path

memory_path = Path("tmp/agent-memory.txt")

# Pre-populate with context
memory_path.write_text("Project uses pytest, Django, Redis...")

instructions = f"""\
## Project Memory
{memory_path.read_text()}

## Workflow
..."""
```

## Configuration

```python
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "llama3.2"  # or "mistral:7b-instruct"
DEFAULT_LLM_TEMPERATURE = 0.4
```

## Key Agent Parameters

| Parameter | Purpose |
|-----------|---------|
| `instructions` | System prompt / behavioral guide |
| `tools` | List of tools the agent can call |
| `db` | SqliteDb for structured memory |
| `knowledge` | Knowledge base for semantic recall |
| `search_knowledge` | Enable auto-search before responding |
| `add_datetime_to_context` | Timestamp each query |
| `add_history_to_context` | Include prior conversation |
| `num_history_runs` | How many prior messages to include |
| `output_schema` | Pydantic model for structured responses |
| `markdown` | Render output as markdown |

## Examples

| Example | Description |
|---------|-------------|
| `examples/qa-agent/` | QA agent with tools + knowledge base |
| `examples/self-learning-agent/` | Self-learning agent with human-in-the-loop |

## Testing

```bash
# Basic test
python -c "from examples.qa-agent.agent import finance_agent; print('Import OK')"

# Interactive session
cd examples/qa-agent && python agent.py
```

## Full Reference

See `references/architecture-guide.md` for the complete architecture reference, troubleshooting, and advanced patterns.
