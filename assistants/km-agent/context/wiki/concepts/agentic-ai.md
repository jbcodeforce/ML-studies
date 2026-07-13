---
title: "Agentic AI"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/agentic.md]
related: [agentic-memory-management, react-pattern, agentic-design-patterns, small-specialist-agents, langgraph]
tags: [agentic, ai, llm, orchestration, scaffolding, crew-ai, langgraph, agno]
---

# Agentic AI

Agentic AI is an orchestrator pattern where the LLM decides what actions to take from the current query and context, following Lilian Weng's reference architecture. The agent uses planning, memory, tools, and execution loops to solve complex tasks.

## Core Principle

Agent results quality is only 20–30% linked to the LLM model — the remaining 80% depends on the quality of the scaffolding: deterministic code, well-designed tools, consistent prompts, and system architecture. The recommended approach is to code deterministic logic first, then integrate it as tools for the model.

## Architecture Components

- **Planning**: Chain of Thought, Tree of Thoughts, or LLM+P with external long-horizon planners
- **Memory**: Short-term (context window), long-term (vector store), and entity memory (subjects of interactions)
- **Tools**: CLI interfaces, MCP servers, and external services — each designed as a single composable capability
- **Neuro-symbolic**: Expert system modules combined with general-purpose LLMs, with the LLM routing to the best tool

## Freedom Spectrum

Agentic implementations exist on a spectrum from full code control to full LLM autonomy:

| Type | Decide Output | Decide Steps | Determine Sequences |
|------|--------------|-------------|---------------------|
| Code | Code | Code | Code |
| LLM Call | LLM step | Code | Code |
| Chain | Multiple LLM calls | Code | Code |
| Router | LLM | LLM (no cycles) | Code |
| State Machine | LLM | LLM (with cycles) | Code |
| Agent (Autonomous) | LLM | LLM | LLM |

## Frameworks

Well-established frameworks include Agno, LangGraph, CrewAI, Pydantic AI, AutoGen, and the OpenAI SDK. The OpenAI SDK has become a de-facto standard for tool calling, supported by multiple LLM vendors including Anthropic, Mistral, and Groq.

## Best Practices

- Define clear roles, focused prompts, and limited tool sets per agent
- Use multiple small specialist agents rather than one big agent with many tools
- Template prompts consistently; adopt a composable CLI/text approach
- Design for production: logging, version management, automated deployment, and monitoring
- Tasks should run asynchronously with clear expectations and context

## Challenges

- High cost of open-loop agent execution
- Instability when new models are released
- Poor tool selection with large tool lists (Multi-Action-Agent problem)
- "In-the-middle" context window issue where mid-prompt instructions are ignored
- Current implementations not fully production-ready for general use

## Guidelines

- Adapt agent and task granularity; test sequential and parallel execution
- Add QA agents to review results
- Tools should be versatile, fault-tolerant, and support caching
- Use guardrails to prevent loops, hallucinations, and inconsistent outcomes

## Sources
- [Agentic AI](../summaries/agentic.md)

## Related
- [Agentic Memory Management](agentic-memory-management.md)
- [ReAct Pattern](react-pattern.md)
- [Agentic Design Patterns](agentic-design-patterns.md)
- [Small Specialist Agents](small-specialist-agents.md)
- [LangGraph](langgraph.md)