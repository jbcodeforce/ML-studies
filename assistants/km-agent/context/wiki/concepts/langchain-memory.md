---
title: "LangChain Memory"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/langchain.md]
related: [langchain-framework, private-chatbot-architecture]
tags: [langchain, memory, conversation, state, token-buffer]
---

# LangChain Memory

Large Language Models are inherently stateless — they do not remember anything between calls. LangChain's Memory subsystem provides abstractions for persisting conversation state, enabling chatbots and agents to maintain context across interactions.

## Memory Types

| Type | Description |
| --- | --- |
| **ConversationBufferMemory** | Stores full conversation history as strings; simplest form |
| **ConversationBufferWindowMemory** | Keeps the last *k* exchanges; limits context growth and cost |
| **ConversationTokenBufferMemory** | Limits memory by token count rather than exchange count |
| **ConversationSummaryMemory** | Maintains a running summary of the conversation via an LLM |
| **VectorStoreRetrieverMemory** | Stores conversation history in a vector store for semantic retrieval |
| **KnowledgeGraphMemory** | Maintains an entity knowledge graph from conversation context |

## Key Concepts

- Memory is essentially a container that saves `{"input": ...}` and `{"output": ...}` pairs
- As conversations grow, context size increases and so does API cost (tokens are billed)
- `ConversationChain` is a predefined chain that loads context from memory
- The `verbose=True` flag on chains enables tracing of chain execution

## Sources
- [LangChain Study](../summaries/langchain.md)

## Related
- [LangChain Framework](langchain-framework.md)
- [Private Chatbot Architecture](private-chatbot-architecture.md)