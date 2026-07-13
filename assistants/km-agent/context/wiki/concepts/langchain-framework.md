---
title: "LangChain Framework"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/langchain.md]
related: [haystack-ai-framework, llm-agentic-workflows, rag-architecture]
code: [code/LLM/langchain/openAI, code/LLM/langchain/ollama, code/LLM/langchain/bedrock]
tags: [langchain, llm, framework, chains, prompt-templates, output-parsers]
---

# LangChain Framework

LangChain is an open-source framework for developing applications powered by large language models. It addresses the "glue code" problem in LLM development by providing composable building blocks for connecting LLMs to external data sources, managing conversations, and orchestrating multi-step workflows.

## Core Architecture

LangChain emphasizes composition and modularity. Developers can combine predefined components or create custom ones to address specific use cases.

### Ecosystem
- **LangChain**: Python and JavaScript/TypeScript libraries (Java under construction)
- **LangServe**: Deploy LangChain chains as REST APIs
- **LangSmith**: Platform for debugging, testing, evaluating, and monitoring chains
- **Prompt Hub**: Predefined, reusable prompt templates

### Core Components

1. **Model I/O** — Interfaces for LLMs and ChatModels, prompt templates (string and chat), output parsers
2. **Chains** — Composable sequences of components (LLMChain, SequentialChain, RouterChain)
3. **Memory** — Stateful conversation management (buffer, window, token, summary, vector store, knowledge graph)
4. **Retrievers** — Semantic search over embeddings in vector stores (FAISS, ChromaDB, OpenSearch)
5. **Agents** — LLM-driven orchestration where the model decides tool usage in a loop
6. **Tools** — Functions that agents can invoke, with structured input/output schemas

## Value Proposition

- **Context awareness**: Apps that can reason using LLMs with external data
- **Modularity**: Mix and match components for different use cases
- **Multi-backend support**: Works with OpenAI, Anthropic, Mistral, Ollama, WatsonX, AWS Bedrock, and more
- **RAG-ready**: Built-in support for embeddings, text splitting, vector stores, and similarity search

## Sources
- [LangChain Study](../summaries/langchain.md)

## Related
- [Haystack AI Framework](haystack-ai-framework.md)
- [LLM-Driven Agentic Workflows](llm-agentic-workflows.md)
- [RAG Architecture](rag-architecture.md)