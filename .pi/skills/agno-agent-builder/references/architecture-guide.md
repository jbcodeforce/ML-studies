# Agno Agent Architecture Reference

## Core Concepts

### Agent

```python
from agno.agent import Agent

agent = Agent(
    name="My Agent",
    model=Ollama(id="llama3.2"),
    instructions="Your system prompt here...",
    tools=[tool1, tool2],      # Optional: list of tools
    db=SqliteDb(db_file="tmp/memory.db"),  # Optional: persistent memory
    knowledge=Knowledge(...),  # Optional: semantic knowledge base
    search_knowledge=True,     # Optional: auto-search knowledge before responding
    add_datetime_to_context=True,  # Add timestamp to every query
    add_history_to_context=True,   # Include prior messages
    num_history_runs=5,            # How many prior messages to include
    markdown=True,                 # Render output as markdown
    output_schema=MySchema,       # Optional: structured response via Pydantic
)
```

### Tools (`@tool`)

Tools extend the agent's capabilities. Each tool is a regular Python function decorated with `@tool`.

```python
from agno.tools import tool

@tool
def my_tool(arg1: str, arg2: int = 0) -> str:
    """One-line description of what this tool does.

    Args:
        arg1: First argument (with type hint)
        arg2: Second argument with default

    Returns:
        Description of the return value
    """
    result = arg1 * arg2
    return str(result)
```

**Options:**
- `requires_confirmation=True` — Human must approve before the tool executes
- `retries=3` — Retry on failure
- `timeout=60` — Max execution time in seconds

```python
@tool(requires_confirmation=True)
def sensitive_operation(data: str) -> str:
    """An operation that requires human approval before executing."""
    ...
```

### Models

Local models via Ollama:

```python
from agno.models.ollama import Ollama

model = Ollama(
    id="llama3.2",          # Model name in Ollama
    base_url="http://127.0.0.1:11434",  # Ollama server URL
    temperature=0.4,        # Lower = more deterministic
)
```

Supported models: `llama3.2`, `mistral:7b-instruct`, `codellama`, etc.

### Structured Output (Pydantic)

Force the agent to return structured data:

```python
from pydantic import BaseModel, Field

class MyOutput(BaseModel):
    field1: str = Field(..., description="Required field")
    field2: int = Field(0, description="Optional field with default")

agent = Agent(
    model=model,
    instructions=instructions,
    output_schema=MyOutput,  # Agent will structure response to match this
)
```

### Persistent Memory: SQLite

```python
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="tmp/agent.db")

agent = Agent(model=model, db=db)
```

Tools can write structured data:
```python
db.insert("my_table.json", json.dumps({"key": "value"}))
# Retrieves: db.retrieve_records("my_table.json")
# Counts: db.count_records("my_table.json")
```

### Persistent Memory: Chroma Knowledge Base

Semantic search over a vector store:

```python
from agno.knowledge import Knowledge
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.reader.text_reader import TextReader
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

kb = Knowledge(
    name="My Knowledge",
    vector_db=ChromaDb(
        name="my-kb",
        collection="my-collection",
        path="tmp/chromadb",  # Persistent path
        persistent_client=True,
        search_type=SearchType.hybrid,  # Hybrid = BM25 + embeddings
        hybrid_rrf_k=60,
        embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
    ),
    max_results=5,
    contents_db=db,
)

agent = Agent(model=model, knowledge=kb, search_knowledge=True)
```

## Workflow Patterns

### Pattern 1: Simple Tool-Using Agent

```python
agent = Agent(
    model=model,
    instructions="""\
You are a code reviewer. Read files with analyze_code and give feedback.
""",
    tools=[analyze_code],
    db=db,
)
```

### Pattern 2: Knowledge-Augmented Agent

```python
agent = Agent(
    model=model,
    instructions="""\
Always check your knowledge base before answering.
""",
    tools=[analyze_code, check_tests],
    db=db,
    knowledge=kb,
    search_knowledge=True,
)
```

### Pattern 3: Self-Learning Agent

```python
@tool(requires_confirmation=True)
def save_learning(title: str, content: str) -> str:
    # Human approves → saves to knowledge base
    kb.insert(name=title, text_content=content)

agent = Agent(
    model=model,
    tools=[save_learning, ...],
    knowledge=kb,
    search_knowledge=True,
)
```

### Pattern 4: Multi-Agent System (AgentOS)

```python
from agno.os import AgentOS

agent_os = AgentOS(
    id="My Multi-Agent",
    agents=[finance_agent, qa_agent, learning_agent],
    config="config.yaml",
    tracing=True,
)

app = agent_os.get_app()
```

## Running Locally

### Prerequisites

```bash
# Install Ollama
brew install ollama    # macOS
# or: https://ollama.com/download

# Pull a model
ollama pull llama3.2
ollama pull mistral:7b-instruct

# Start the server
ollama serve
```

### Run an Agent

```bash
# Interactive mode
python agent.py

# As part of AgentOS
python first_agent_os.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'agno'` | `pip install agno` |
| `Ollama not running` | Run `ollama serve` |
| `Model not found` | Run `ollama pull llama3.2` |
| `ChromaDb not found` | First run auto-creates `tmp/chromadb/` |
| `sqlite3 error` | Check write permissions on the .db file |
| `Knowledge base empty` | Need to insert entries first via tools |

## File Layout Reference

```
project/
├── agent.py                # Main agent definition + CLI
├── tools/
│   ├── custom_tool.py      # Tool implementations
│   └── additional_tool.py
├── memory/
│   ├── sqlite_context.py   # SQLite-based memory
│   └── knowledge_base.py   # ChromaDb knowledge base
├── config.yaml             # AgentOS config
├── requirements.txt
└── tmp/                    # Auto-created data directory
    ├── chromadb/           # ChromaDB vector store
    ├── agent.db            # SQLite database
    └── qa-memory.db
```
