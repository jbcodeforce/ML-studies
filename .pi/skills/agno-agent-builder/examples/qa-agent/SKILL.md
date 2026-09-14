---
name: qa-agent
description: A Software Quality Assurance agent with local tools, a knowledge base for project context, and persistent memory. Use for automated testing, bug triage, and project-aware QA tasks.
---

# QA Agent

An agno agent that performs software quality assurance tasks: understanding requirements, generating test cases, reporting defects, and learning from past findings.

## Architecture

```
QA Agent
├── Model: Ollama (llama3.2 or mistral)
├── Tools:
│   ├── analyze_code       → Inspect source files
│   ├── check_tests        → Run existing tests
│   ├── document_issue     → Log a bug finding
│   └── review_pr          → Review pull request changes
├── Memory:
│   ├── Knowledge base (ChromaDb) → Project context, past issues, patterns
│   └── SQLite → Structured test results, session memory
└── Output: Structured QA reports via Pydantic schema
```

## Quick Start

```bash
# Start Ollama
ollama serve

# Run interactively
cd /Users/jerome/Documents/Code/ML-studies/code/agents/agno/.pi/skills/agno-agent-builder/examples/qa-agent
python agent.py
```

## Prompting Examples

- "Check all Python tests and report coverage"
- "Review the src folder and suggest improvements"
- "Document issue: Login form returns 500 for empty password"
- "Review this PR: add-auth-header.patch"

## Workflow

1. **Analyze** — Use `analyze_code` to read source files
2. **Test** — Use `check_tests` to run the test suite
3. **Report** — Use `document_issue` to log findings
4. **Learn** — Insights are saved to the knowledge base for future reference

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Ollama not running` | Run `ollama serve` in another terminal |
| `Knowledge DB not found` | First run creates `tmp/chromadb/` automatically |
| `SQLite error` | Check `tmp/qa-memory.db` is writable |
