# agno-agent-builder — Skill for Pi / Claude Code

> Scaffold agno agents with local tools, system prompts, and persistent memory.

## What This Skill Does

Provides a structured workflow for building agno agents featuring:

1. **Custom local tools** — Functions decorated with `@tool` that the agent can call
2. **Rich system prompts** — Multi-section `instructions` with workflow, rules, and examples
3. **Persistent memory** — Three tiers:
   - **SQLite** — Structured, queryable state
   - **Chroma + Knowledge** — Semantic search over vector embeddings
   - **Files** — Simplest form of persistent context

## Included Examples

| Example | Description |
|---------|-------------|
| `examples/qa-agent/` | Full QA agent: code analysis, test running, issue tracking, knowledge-base-powered recall |
| `examples/self-learning-agent/` | Agent that saves insights over time with human-in-the-loop approval |

## Quick Start

```bash
# Ensure omlx is running
omlx start

# From project root
cd .pi/skills/agno-agent-builder
```

### Run the QA Agent

```bash
cd examples/qa-agent
python agent.py
```

### Run the Self-Learning Agent

```bash
cd examples/self-learning-agent
python agent.py
```

## How to Use

### 1. Load the Skill

```bash
# In Pi settings or CLI
--skill .pi/skills/agno-agent-builder
```

### 2. Read the Main Guide

The `SKILL.md` in the skill root contains:
- Quick-start templates for models, tools, memory, and prompts
- Component reference (Model, Tools, Instructions, Memory)
- Key agent parameters table
- Troubleshooting guide

### 3. Study the Examples

| File | What It Shows |
|------|---------------|
| `examples/qa-agent/SKILL.md` | QA-specific guidance |
| `examples/qa-agent/agent.py` | Complete working agent |
| `examples/self-learning-agent/SKILL.md` | Self-learning guidance |
| `examples/self-learning-agent/agent.py` | Complete working agent |
| `references/architecture-guide.md` | Full architecture reference |

## Minimum Working Example

```python
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.db.sqlite import SqliteDb

model = Ollama(id="llama3.2")
db = SqliteDb(db_file="tmp/memory.db")

@tool
def my_tool(arg: str) -> str:
    """Do something."""
    return f"Processed: {arg}"

agent = Agent(
    name="My Agent",
    model=model,
    instructions="""\
You are a helpful assistant.
Use my_tool when the user asks you to process something.
""",
    tools=[my_tool],
    db=db,
    add_datetime_to_context=True,
    markdown=True,
)
```

## Skill Loading

The skill is automatically discovered by Pi when you're in a `.pi/skills/` directory.
It registers `/skill:agno-agent-builder` as a command, and its descriptions help the agent
know when to load it.
