---
title: "Model Context Protocol (MCP)"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/mcp.md]
related: [tool-calling, agentic-ai, anthropic-claude]
tags: [mcp, tool-calling, protocol, anthropic, ai-assistants, cursor]
---

# Model Context Protocol (MCP)

The **Model Context Protocol (MCP)** is an open protocol defined by Anthropic that standardizes how AI assistants interact with external tools and services. MCP provides a common interface through which AI models—such as Claude or editor-based assistants like Cursor—can discover, invoke, and receive structured results from arbitrary external systems.

## Overview

MCP addresses the integration challenge that arises when AI assistants need to extend beyond their built-in capabilities. Rather than requiring custom adapters for every model-tool pair, MCP defines a protocol-level standard:

- **Tool Discovery**: AI assistants can discover available tools through the MCP server.
- **Standardized Invocation**: Tools are called using a consistent protocol, abstracting away model-specific calling conventions.
- **Structured Results**: Responses from tools follow a uniform format that the assistant can interpret.

## Ecosystem

- **Cursor IDE** maintains a [directory of supported MCP servers](https://cursor.com/docs/context/mcp/directory), enabling developers to connect their AI-assisted workflows to external services.
- **Custom MCP Servers**: Developers can author their own MCP servers to expose any service—CLI tools, APIs, databases, or domain-specific systems—as callable tools for AI assistants.

## Relationship to Related Concepts

MCP sits above the model-level **tool-calling** capability. While tool calling is a feature of specific LLMs (e.g., Claude, GPT), MCP provides a protocol standard that works across models and services. This makes it a natural fit for **agentic AI** systems, where agents need reliable, standardized access to external resources.

## Sources
- [MCP](../summaries/mcp.md)

## Related
- [Tool Calling in LLMs](tool-calling.md)
- [Agentic AI](agentic-ai.md)
- [Anthropic and Claude Models](anthropic-claude.md)