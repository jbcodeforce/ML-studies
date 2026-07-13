---
title: "Agentic Memory Management"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/agentic.md]
related: [agentic-ai, langgraph, langchain-memory]
tags: [agentic, memory, short-term, long-term, entity-memory, vector-store]
---

# Agentic Memory Management

Memory is critical for agents to maintain context, learn from interactions, and improve over time. There are three types of memory in agentic systems.

## Short-Term Memory

Short-term memory holds the current conversation context within the LLM's context window. It includes:

- Current user query and conversation history
- Intermediate results from tool calls
- Agent scratchpad for reasoning steps

Short-term memory also helps exchange data between agents. In LangGraph, this is implemented via message-based state with checkpointing for persistence across requests:

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# Thread-based conversation tracking
config = {"configurable": {"thread_id": "user-123"}}
```

## Long-Term Memory

Long-term memory persists beyond the context window using vector stores, supporting maximum inner product search. It enables:

- Retrieval of relevant past interactions
- Knowledge base queries
- Learning from historical data
- Self-improvement of agents

```python
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()
relevant_docs = retriever.invoke(user_query)
```

## Entity Memory

Entity memory is a third type that tracks information about specific subjects (people, organizations, locations) mentioned in conversations. CrewAI implements this as short-term memory extracted via NLP, enabling agents to maintain coherent understanding of entities across conversations without re-extracting information.

## Memory in Practice

For multi-agent systems, short-term memory is used during crew execution of a task and is shared between agents even before task completion. Long-term memory is used after task execution and can inform future tasks, stored in a database. Agent memory enables learning from previous executions and self-improvement.

## Sources
- [Agentic AI](../summaries/agentic.md)

## Related
- [Agentic AI](agentic-ai.md)
- [LangGraph](langgraph.md)
- [LangChain Memory](langchain-memory.md)