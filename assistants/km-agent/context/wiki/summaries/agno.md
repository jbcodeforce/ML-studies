# Agno Framework Studies

Agno is a Python SDK for building AI agents and agentic solutions, positioning itself as a minimalist, production-ready alternative to frameworks like LangChain and LlamaIndex. It emphasizes deterministic behavior, transparency, and simplicity.

The framework's core concepts include **Agents** (stateful control loops around stateless LLMs), **Knowledge** (RAG with vector search, filtering, and reranking), and **Skills** (modular domain-expertise packages with lazy loading). Agents support multiple model backends — Ollama for local development, OpenAI-compatible servers, and Anthropic.

Knowledge management is a major strength: agents can agendically decide when to search a knowledge base, use hybrid vector+keyword search, apply metadata filters (including dynamic agentic filtering), and rerank results for improved quality. Multi-tenant isolation and content lifecycle management (skip duplicates, remove outdated, track via contents DB) support production use.

Skills are self-contained packages with instructions, scripts, and references, using a progressive discovery workflow where the agent only loads detailed instructions on demand. This keeps system prompts lean and enables model-agnostic skill definitions. Skills can be assigned to teams in multi-agent setups, where the team leader orchestrates skill discovery and execution.

The source includes code examples using both local (Ollama + ChromaDB) and cloud (OpenAI + Qdrant) configurations.