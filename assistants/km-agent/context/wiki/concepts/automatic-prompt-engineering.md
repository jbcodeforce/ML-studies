---
title: "Automatic Prompt Engineering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
related: [prompt-engineering, chain-of-thought, prompt-chaining]
tags: [ape, automatic-prompt-engineering, dspy, prompting, llm]
---

# Automatic Prompt Engineering

Automatic Prompt Engineering (APE) is an approach to automate the creation and selection of prompts using LLMs. Instead of manually crafting prompts, APE uses one LLM as an inference model (to generate candidate prompts) and another LLM as a scoring model (to evaluate their effectiveness).

## How It Works

1. **Inference Model**: Generates candidate prompts for a given task.
2. **Scoring Model**: Evaluates each candidate prompt by running it and scoring the output.
3. **Selection**: The best-scoring prompt is selected for use.

This approach can scale prompt optimization and remove the need for manual trial-and-error.

## DSPy

**DSPy** (https://github.com/stanfordnlp/dspy) is a notable framework for algorithmically optimizing LM prompts and weights. It provides tools for automatically searching the prompt space and finding optimal configurations.

## Sources
- [Prompt Engineering Summary](../summaries/prompt-eng.md)

## Related
- [Prompt Engineering](prompt-engineering.md)
- [Chain of Thought](chain-of-thought.md)
- [Prompt Chaining](prompt-chaining.md)