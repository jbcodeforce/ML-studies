---
title: "Claude Code"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/claude/index.md]
related: [anthropic-claude, agentic-ai, agentic-design-patterns]
tags: [claude-code, coding-assistant, agentic, cli, mcp, plugins]
---

# Claude Code

Claude Code is an agentic coding assistant from Anthropic that addresses coding, discovery, and design tasks. It operates as a CLI tool with powerful capabilities for autonomous code generation, code review, and project exploration.

## Architecture and Memory

Claude Code uses a **CLAUDE.md** file as persistent memory across sessions. This file can contain style guidelines, common commands, and project-specific knowledge, and is automatically loaded into context. Conversation history is stored locally and loaded into context.

## Key Capabilities

- **Sub-agents**: Can launch sub-agents for parallel investigation tasks
- **To-do management**: Built-in task tracking
- **Agentic search**: Autonomous codebase exploration
- **Plugins marketplace**: Extensible via plugins containing skills, hooks, agents, and commands
- **Context management**: /clear between unrelated tasks, /compact to summarize mid-session

## Integration Playbook

The recommended approach for integrating Claude Code with existing repos involves three steps:
1. Create a targeted CLAUDE.md (exclude anything the LLM can infer from code)
2. Adopt Test-Driven Development
3. Plan first, then scale with skills and plugins

Best practices favor CLIs over MCP, aggressive context management, and using planning for uncertain or multi-file changes.

## Execution Environments

Claude Code supports multiple backends:
- **Anthropic API**: Default cloud execution
- **Google Vertex AI**: Via VPN with gcloud auth; supports models like claude-haiku-4-5
- **Local LLMs**: Via Ollama with custom Modelfile configuration (e.g., gemma4 with 64k context), though with reduced capability compared to native Claude

## Notable Patterns

The source references the "Wiki LLM" concept (from Andrej Karpathy's core principles) as a pattern for LLM-powered documentation and knowledge management.

## Sources
- [Source Document](../summaries/claude-index.md)

## Related
- [Anthropic Claude](anthropic-claude.md)
- [Agentic AI](agentic-ai.md)
- [Agentic Design Patterns](agentic-design-patterns.md)