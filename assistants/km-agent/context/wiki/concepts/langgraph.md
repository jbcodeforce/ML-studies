---
title: "LangGraph"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/langgraph.md]
related: [langchain-framework, lcel, tool-calling, llm-agentic-workflows, rag-architecture]
tags: [langgraph, agentic, stateful, graphs, multi-agent, checkpointing, react, human-in-the-loop]
---

# LangGraph

LangGraph is a library for building stateful, multi-actor LLM applications using graph-based workflows. Unlike traditional DAGs, LangGraph supports cycles, enabling iterative agent patterns such as ReAct (Reasoning and Acting).

## Core Concepts

LangGraph models applications as graphs with three primary elements:

**States** — Typed collections (e.g., `MessageState` or custom `TypedDict` schemas) that carry data between nodes. The default `MessageState` maintains a list of conversation messages.

**Nodes** — Units of work, implemented as functions or runnables. Each node receives the full graph state, performs computation, and returns an updated state.

**Edges** — Connections between nodes. Conditional edges evaluate node output to dynamically route execution along different paths, enabling flexible workflows.

## Agent Development

LangGraph provides a graph-native approach to agent construction, replacing the deprecated `AgentExecutor` API. The development workflow is:

1. Define tools the agent can use
2. Specify the state schema and persistence requirements
3. Construct the workflow as a graph with nodes, edges, and conditional routing
4. Compile the graph into a LangChain Runnable with optional checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["action"])
```

Once compiled, the graph can be invoked via `invoke()` for single responses or `stream()` for token-by-token output.

## Persistence and Human-in-the-Loop

Built-in checkpointers (`MemorySaver`, `SqliteSaver`, `AsyncSqliteSaver`, `PostgresSaver`) persist graph state between steps, enabling:

- **Resume from checkpoints** — Continue execution from a saved state
- **Human-in-the-loop** — Interrupt execution before or after specific nodes, allowing human review or input before proceeding
- **Thread isolation** — Each conversation is tracked via a unique `thread_id` in the config

## Tool Calling

LangGraph integrates tool calling through `ToolNode`, which executes tool functions identified by the LLM. The graph cycles between agent and tool nodes until the LLM produces a response without `tool_calls`. The pattern:

1. Agent node invokes LLM with tool definitions in the prompt
2. LLM returns a response containing `tool_calls` structured arguments
3. Tool node executes the matching function and returns results
4. Agent node re-invokes with the tool results appended to the message history
5. Loop continues until the LLM produces a final answer

## Key Use Cases

- **ReAct Pattern** — Iterative reasoning, action, and observation cycles for complex problem-solving
- **Adaptive RAG** — Query routing, document retrieval, and quality grading within a graph workflow
- **Multi-Agent Teams** — Composable subgraphs with independent state, enabling collaborative multi-agent systems
- **Human-in-the-Loop** — Execution pausing at designated nodes for human approval or correction
- **Streaming** — Real-time token streaming via `astream_events` for responsive chat interfaces

## Related Concepts

LangGraph extends the LangChain ecosystem, leveraging LCEL-style runnables and complementing LangChain's memory and tool-calling systems. It is particularly suited for production-ready agentic workflows requiring cyclic control flow, state persistence, and multi-step orchestration.

## Sources
- [LangGraph Study Notes](../summaries/langgraph.md)

## Related
- [LangChain Framework](langchain-framework.md)
- [LangChain Expression Language (LCEL)](lcel.md)
- [Tool Calling in LLMs](tool-calling.md)
- [LLM-Driven Agentic Workflows](llm-agentic-workflows.md)
- [RAG Architecture](rag-architecture.md)