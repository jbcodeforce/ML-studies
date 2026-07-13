---
title: "Tool Calling in LLMs"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/langchain.md]
related: [langchain-framework, llm-agentic-workflows, haystack-ai-framework]
tags: [tool-calling, agents, function-calling, langchain, llm]
---

# Tool Calling in LLMs

Tool calling enables an LLM to detect when external functions (tools) should be invoked and to generate structured arguments for those functions. The LLM does not execute the tools itself — it produces a structured response that the application orchestrator uses to call the appropriate tools.

## How It Works

1. **Tool definitions** are passed to the LLM via `llm.bind_tools([tool_definitions])`, embedding structured system prompts
2. The LLM responds with a `tool_calls` array containing `name`, `args`, and `id` fields
3. The application extracts the tool name and arguments, executes the tool, and returns the **observation** to the LLM
4. This loops until the LLM produces an `AgentFinish` response instead of `AgentAction`

## Tool Definition Approaches

- **`@tool` decorator**: Annotate a plain function to expose it as a tool (function name = tool name, docstring = description)
- **BaseModel subclassing**: Define a Pydantic model that becomes a tool schema
- **StructuredTool**: Use the dataclass for explicit tool configuration
- **`load_tools`**: Load predefined tools from LangChain's catalog

## Key Considerations

- Tool descriptions must be sensibly named and well-documented for the LLM to choose correctly
- For large tool catalogs, use **embedding-based tool selection** to dynamically pick the top-N relevant tools at runtime, avoiding context window overflow
- LangChain provides `convert_to_openai_function` to map tools to OpenAI function calling format
- Many providers support tool calling: Anthropic, Cohere, Google, Mistral, OpenAI

## Scratchpad Pattern

Agent prompts often include an `agent_scratchpad` (`MessagesPlaceholder`) that holds intermediate steps — pairs of `AgentAction` + tool output — enabling the LLM to reason about prior actions.

## Sources
- [LangChain Study](../summaries/langchain.md)

## Related
- [LangChain Framework](langchain-framework.md)
- [LLM-Driven Agentic Workflows](llm-agentic-workflows.md)
- [Haystack AI Framework](haystack-ai-framework.md)