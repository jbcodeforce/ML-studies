---
title: "LangChain Expression Language (LCEL)"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/langchain.md]
related: [langchain-framework, llm-agentic-workflows]
code: [code/LLM/langchain/openAI]
tags: [lcel, langchain, declarative, runnables, streaming, async]
---

# LangChain Expression Language (LCEL)

LangChain Expression Language (LCEL) is a declarative syntax for defining chains in LangChain. It uses a Unix shell pipe–style operator (`|`) to compose runnables, where the output of one runnable becomes the input of the next.

## Syntax

```python
chain = prompt | model | output_parser
```

LCEL is built on top of the **Runnable** interface, which exposes synchronous and async `invoke` methods, batch execution, and streaming.

## Key Capabilities

- **Streaming**: LCEL supports streaming LLM results for responsive applications
- **Async communication**: Runnables support `ainvoke` for async execution
- **Parallelism**: `RunnableParallel` and `RunnablePassthrough` enable concurrent data flow
- **Retries and fallbacks**: Built-in resilience patterns
- **Intermediate results**: Access outputs between pipeline stages
- **Type coercion**: Automatic schema handling between runnables

## Runnable Primitives

- **RunnableLambda**: Wraps a callable function as a Runnable
- **RunnablePassthrough**: Passes input through while optionally adding keys via `.assign()`
- **RunnableParallel**: Runs multiple runnables in parallel, merging outputs into a dict
- **Runnable.bind()**: Pass constant arguments into a chain where they are not part of preceding outputs

## Sources
- [LangChain Study](../summaries/langchain.md)

## Related
- [LangChain Framework](langchain-framework.md)
- [LLM-Driven Agentic Workflows](llm-agentic-workflows.md)