---
title: "Prompt Chaining"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
related: [prompt-engineering, chain-of-thought, tree-of-thoughts]
tags: [prompt-chaining, prompting, llm, workflow]
---

# Prompt Chaining

Prompt chaining breaks a complex prompt task into subtasks, feeding the response of one subtask as input to the next. This creates a chain of prompts.

## How It Works

1. A complex task is decomposed into simpler subtasks.
2. Each subtask is sent to the LLM as a separate prompt.
3. The output from one subtask becomes the input (or context) for the next.
4. The final output is the result of the last subtask.

## Benefits

- **Better performance**: Breaking tasks down improves overall output quality.
- **Transparency**: Each step is visible, making it easier to debug.
- **Controllability**: Individual steps can be tuned or validated independently.
- **Reliability**: Errors in one step don't necessarily cascade to the entire task.

## Use Cases

- **Conversational assistants**: Multi-turn conversations where context builds incrementally.
- **Document QA**: A first prompt extracts important quotes from a document given a question; a second prompt uses those quotes to generate the answer.
- **Response validation**: A second prompt can be used to validate or refine the output of a first prompt.

## Sources
- [Prompt Engineering Summary](../summaries/prompt-eng.md)

## Related
- [Prompt Engineering](prompt-engineering.md)
- [Chain of Thought](chain-of-thought.md)
- [Tree of Thoughts](tree-of-thoughts.md)