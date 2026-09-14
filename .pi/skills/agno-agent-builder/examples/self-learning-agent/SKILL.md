---
name: self-learning-agent
description: An agno agent that self-learns from conversations via human-in-the-loop knowledge capture. Use for agents that should improve over time by saving insights to a knowledge base.
---

# Self-Learning Agent

An agno agent that:
- Queries its knowledge base before answering
- Saves new insights using a custom `save_learning` tool
- Requires human confirmation before saving (to avoid noise)

## Architecture

```
Self-Learning Agent
├── Model: Ollama
├── Tools:
│   ├── save_learning (human-in-the-loop) → inserts to ChromaDb
│   ├── search_learnings → query knowledge base
│   └── ... (domain-specific tools)
├── Knowledge: ChromaDb with hybrid search
└── Memory: SqliteDb for structured state
```

## Quick Start

```bash
ollama serve
cd /Users/jerome/Documents/Code/ML-studies/code/agents/agno/.pi/skills/agno-agent-builder/examples/self-learning-agent
python agent.py
```

## What is "Save a Learning"?

A learning is a reusable insight that can be applied to future questions. The agent asks for human confirmation before saving, and you can reject any that seem wrong or too specific.

### Good Learnings

- "Tech stocks P/E ratios typically range 20-35x"
- "Our CI pipeline fails on Python 3.11 due to setuptools version"
- "The logger uses set-level(), not setLevel()"

### Bad Learnings

- "P/E ratios vary" (too vague)
- "Python is a programming language" (obvious)
- Raw data dumps (not reusable)

## File Layout

```
self-learning-agent/
├── SKILL.md
└── agent.py
```

## Configuration

Edit `agent.py` to:
- Change `DEFAULT_LLM_MODEL`
- Add more tools
- Customize `INSTRUCTIONS`
- Set `MAX_MEMORY_SIZE`
