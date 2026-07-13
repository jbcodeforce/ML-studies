---
title: "OpenAI Platform Overview"
created: 2026-07-12
updated: 2026-07-12
source: raw/studies/genAI/openai.md
tags: [openai, genai, llm, assistants-api, gpt]
---

# OpenAI Platform Overview

OpenAI provides a comprehensive generative AI platform including ChatGPT, multiple model families (GPT-4, DALL·E, TTS, Whisper, Embeddings), an SDK, and REST APIs. Personal data is not used for model training, though API data may be retained for 30 days (with zero-retention options available).

## Assistants API

The Assistants API is a key abstraction for building AI applications. Assistants combine a model, instructions, and tools (Code Interpreter, Retrieval, Function Calling) into reusable entities. Key concepts include:

- **Assistants**: Created via API with a name, instructions, tools, and model specification.
- **Threads**: Represent conversations between users and assistants, automatically managing context window truncation.
- **Runs**: Execute an assistant on a thread, with pollable states (Queued, In Progress, Requires Actions, Completed, Failed, etc.).
- **Code Interpreter**: Allows execution of user-provided Python or Node.js code on OpenAI-hosted infrastructure.

The Assistants API simplifies development by handling message history management and context window constraints automatically.

## Key Connections

- Relates to [Agentic AI](../concepts/agentic-ai.md) through tool-calling and function invocation capabilities.
- Provides the underlying models for [RAG Architecture](../concepts/rag-architecture.md) implementations.
- The Assistants API offers an alternative approach to [LangGraph](../concepts/langgraph.md) for stateful AI workflows.