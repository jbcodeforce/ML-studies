---
title: "Agno Framework"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/agno.md]
related: [agentic-ai, rag-architecture, tool-calling, langchain-framework, langgraph, llamaindex]
tags: [agno, agents, framework, rag, skills, knowledge-base, python]
---

# Agno Framework

Agno is a minimalist, production-ready Python SDK for building AI agents and agentic solutions. It emphasizes deterministic behavior, transparency, and simplicity.

## Core Architecture

- **Agents**: Stateful control loops around stateless LLMs. Support multiple model backends including Ollama (local), OpenAI-compatible servers, and Anthropic.
- **Database**: Persistent storage for sessions, context, memory, learnings, and evaluation datasets.
- **Storage**: Automatic conversation history once a database is attached to an agent.
- **Memory**: Tracks user preferences across sessions.
- **State**: Structured data (counters, lists, flags) actively managed by the agent across runs, injectable into instructions via `{variable_name}`.

## Knowledge (RAG)

Agno's Knowledge system provides the primary mechanism for giving agents access to documents. Key features:

- **Agentic search**: Agent receives a `search_knowledge_base` tool and decides when/how to query.
- **Vector databases**: Qdrant, ChromaDB, with OpenAI or Ollama embedders.
- **Search types**: Vector (semantic), keyword (full-text), or hybrid (recommended default).
- **Reranking**: Two-stage retrieval with Cohere, SentenceTransformer, Infinity, or Bedrock rerankers.
- **Filtering**: Metadata-based filtering with simple dict filters or powerful FilterExpr (AND, OR, NOT, EQ, IN, GT, LT). Agentic filtering lets the agent dynamically build filters from user queries.
- **Multi-tenant isolation**: `isolate_vector_search` ensures separate data namespaces per knowledge instance sharing a vector store.
- **Content lifecycle**: Skip duplicates, remove outdated content, track via contents database.
- **Knowledge Graph**: LightRAG support for entity/relationship extraction and multi-hop graph reasoning.

## Skills

Skills are self-contained, modular packages of domain expertise organized in directories with instructions (SKILL.md), optional scripts, and references. They use lazy loading — the system prompt only holds a lightweight summary, and detailed instructions load on demand. Skills enable:

- Progressive discovery workflow (browse → load → reference → execute)
- Swappable language models without rewriting skill logic
- Team-level skill assignment in multi-agent systems

## Comparison

Agno positions itself alongside LangChain, LangGraph, and LlamaIndex as an alternative agent framework, but with a focus on minimalism and production readiness rather than broad composability.

## Sources
- [Agno studies](../summaries/agno.md)

## Related
- [Agentic AI](agentic-ai.md)
- [RAG Architecture](rag-architecture.md)
- [LangChain Framework](langchain-framework.md)
- [LangGraph](langgraph.md)
- [LlamaIndex Library](llamaindex.md)
- [Tool Calling in LLMs](tool-calling.md)