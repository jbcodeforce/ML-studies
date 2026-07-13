---
title: "OpenAI Assistants API"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/openai.md]
related: [openai-platform, langgraph, agentic-ai, rag-architecture]
tags: [openai, assistants, api, threads, runs, code-interpreter, function-calling]
---

# OpenAI Assistants API

The OpenAI Assistants API provides a high-level abstraction for building stateful, multi-turn AI applications. It manages conversation history, tool execution, and context window constraints automatically, simplifying the development of AI assistants.

## Core Concepts

### Assistants
An Assistant is a configurable entity that combines:
- A name and system-level instructions
- A specific LLM model (e.g., `gpt-4-turbo-preview`)
- A set of available tools (Code Interpreter, Retrieval, Function Calling)

Each Assistant is created via API and can be reused across multiple conversations.

### Threads
A Thread represents a single conversation between a user and one or more Assistants. Key features:
- Messages are added to the Thread by the user or assistant
- The Thread automatically manages message history
- Context window is truncated automatically when conversations exceed the model's length limit
- Threads persist conversation state between interactions

### Runs
A Run executes an Assistant on a specific Thread. Runs progress through several states:
- **Queued**: Waiting to begin execution
- **In progress**: Currently processing
- **Requires actions**: Waiting for external function call results
- **Expired**: Timed out without completion
- **Completed**: Successfully finished
- **Failed**: Encountered an error
- **Cancelling / Cancelled**: User-initiated cancellation

Run states can be polled via API to monitor progress and handle async operations.

### Tools
Assistants can be equipped with several tool types:
- **Code Interpreter**: Executes Python or Node.js code on OpenAI-hosted infrastructure, enabling computation and data analysis
- **Retrieval**: Searches through uploaded documents to answer questions based on provided knowledge
- **Function Calling**: Invokes external APIs or services defined by the developer via tool definitions

## Example Usage

```python
from openai import OpenAI
client = OpenAI()

# Create an assistant
assistant = client.beta.assistants.create(
  name="Math Tutor",
  instructions="You are a personal math tutor. Write and run code to answer math questions.",
  tools=[{"type": "code_interpreter"}],
  model="gpt-4-turbo-preview",
)

# Create a conversation thread
thread = client.beta.threads.create()

# Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Can you help me?"
)

# Execute the assistant on the thread
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions="Please ..."
)
```

## Sources
- [OpenAI Platform Overview](../summaries/openai.md)

## Related
- [OpenAI Platform](openai-platform.md)
- [LangGraph](langgraph.md)
- [Agentic AI](agentic-ai.md)
- [RAG Architecture](rag-architecture.md)
- [Tool Calling in LLMs](tool-calling.md)