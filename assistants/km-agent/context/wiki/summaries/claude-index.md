# Claude Code Summary

Claude Code is an agentic coding assistant from Anthropic that extends beyond code generation into discovery and design. It operates as a CLI with local conversation history and persistent memory via a **CLAUDE.md** file that stores style guidelines, commands, and project-specific knowledge.

Key capabilities include launching sub-agents for parallel investigation, built-in to-do management, agentic codebase search, and a plugin marketplace for extending functionality with custom skills, hooks, agents, and commands.

Integration with existing repos follows a three-step playbook: create a focused CLAUDE.md (excluding redundant information), adopt TDD, and plan before scaling with skills. Best practices emphasize CLIs over MCP, aggressive context management with `/clear` and `/compact` commands, and using planning for multi-file changes.

Claude Code supports multiple execution backends: Anthropic API (default), Google Vertex AI via VPN, and local LLMs through Ollama with custom configurations—though local models have reduced capability compared to native Claude.

The source also references the "Wiki LLM" pattern from Andrej Karpathy's principles for LLM-powered documentation.

## Connections
- Relates to **Agentic AI** patterns of autonomous action and tool use
- Connects to **Anthropic Claude** as the underlying model family
- Demonstrates **Agentic Design Patterns** including multi-agent collaboration and human-in-the-loop workflows