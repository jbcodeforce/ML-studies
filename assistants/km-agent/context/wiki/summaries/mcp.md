---
title: "Model Context Protocol"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/mcp.md]
---

# Model Context Protocol — Summary

The Model Context Protocol (MCP) is an open protocol defined by Anthropic that enables AI assistants (e.g., Claude, Cursor) to invoke external tools and services in a standardized way. MCP provides a common interface through which AI models can discover, call, and receive results from arbitrary external systems — effectively extending the model's capabilities beyond what is baked into its training.

Key points:
- **Origin**: Defined by Anthropic, the creators of Claude.
- **Purpose**: Standardize how AI assistants interact with external tools/services, avoiding custom integrations per model or per service.
- **Ecosystem**: Cursor IDE supports a directory of MCP-compatible servers; developers can build and register their own MCP servers (e.g., a custom CLI tool exposed as an MCP server).
- **Relationship to other concepts**: MCP sits alongside tool-calling/function-calling capabilities in LLMs, providing a protocol-level standard rather than a model-specific feature. It complements agentic AI patterns where the agent needs reliable, standardized access to external resources.

See also: [Tool Calling in LLMs](../concepts/tool-calling.md), [Agentic AI](../concepts/agentic-ai.md), [Anthropic and Claude Models](../concepts/anthropic-claude.md).