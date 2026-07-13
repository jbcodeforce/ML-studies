# LangGraph Summary

LangGraph is a library built on LangChain for creating stateful, multi-actor LLM applications using cyclic graph-based workflows. Unlike traditional DAGs, LangGraph allows cycles, enabling iterative agent patterns like ReAct (Reasoning and Acting).

## Key Concepts

**Graph Structure**: Applications are modeled as graphs with three core components:
- **States**: Typed collections of messages (MessageState) or custom schemas (TypedDict) passed between nodes
- **Nodes**: Units of work (functions or runnables) that receive and update the graph state
- **Edges**: Connections between nodes, including conditional edges that route based on node output

**Agent Implementation**: LangGraph provides a graph-based approach to building agents, replacing the deprecated AgentExecutor API. Agents are constructed by defining tools, state schemas, workflow graphs, and persistence mechanisms, then compiling into a LangChain Runnable.

**Persistence & Human-in-the-Loop**: Built-in checkpointing (MemorySaver, SqliteSaver, PostgreSQL) enables state persistence between steps, supporting human-in-the-loop patterns via `interrupt_before`/`interrupt_after` node pausing. Each conversation thread is identified by a unique `thread_id`.

**Tool Calling**: ToolNode integration allows LLMs to call external tools (e.g., Tavily search). The graph cycles between agent and tool nodes until the LLM produces a final response without tool calls.

## Use Cases

- **ReAct Pattern**: Reasoning and Acting with iterative thought-action-observation cycles
- **Adaptive RAG**: Query routing, document retrieval, and quality grading in a graph workflow
- **Multi-Agent Teams**: Composable subgraphs with independent state tracking
- **Human-in-the-Loop**: Interrupt execution before/after nodes for human approval
- **Streaming**: Token streaming via async `astream_events` for real-time output

## Code Examples

The source includes multiple implementations: basic LLM-only graphs, tool-calling agents, ReAct patterns (custom and prebuilt), Adaptive RAG, human-in-the-loop flows, Mistral integration, and streaming demos. Code is located in the `code/agents/langgraph/` directory of the ML-studies repository.

## Related Concepts

LangGraph extends LangChain's framework, leveraging LCEL-style runnables and complementing LangChain's memory systems. It enables agentic workflows with cyclic control flow, tool calling, and state persistence—key components for building production-ready AI agents.