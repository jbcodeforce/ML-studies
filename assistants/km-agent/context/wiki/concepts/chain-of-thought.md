---
title: "Chain of Thought"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
related: [prompt-engineering, prompt-chaining, tree-of-thoughts]
code: [code/LLM/langchain/bedrock/]
tags: [chain-of-thought, cot, prompting, reasoning, llm]
---

# Chain of Thought

Chain-of-Thought (CoT) prompting uses intermediate reasoning steps to solve complex tasks. It is particularly effective for arithmetic, commonsense, and symbolic reasoning tasks.

## How It Works

CoT prompting asks the model to generate intermediate reasoning steps before producing a final answer. This allows the model to "show its work" and produce more accurate results on complex problems.

## Zero-Shot CoT

The zero-shot CoT technique achieves good results by simply adding the sentence **"Let's think step by step"** to the prompt, without providing any examples.

Example:
```
explain Quantum mechanics to a high school student. Let's think step by step.
A:
```

## Self-Consistency

Self-consistency is a technique that builds on CoT by sampling multiple, diverse reasoning paths through few-shot CoT prompts. The system then selects the most consistent answer across all generated paths, improving reliability.

## RLHF Connection

**Reinforcement Learning from Human Feedback (RLHF)** has been adopted to scale instruction tuning, wherein the model is aligned to better fit human preferences.

## Practical Testing

CoT can be tested with Bedrock LLMs. Testing code is available in the `code/LLM/langchain/bedrock/` directory. Results vary by model — Anthropic Claude v2 produces good answers, while AI21 Jurassic-2 Mid returns poor answers.

## Sources
- [Prompt Engineering Summary](../summaries/prompt-eng.md)

## Related
- [Prompt Engineering](prompt-engineering.md)
- [Prompt Chaining](prompt-chaining.md)
- [Tree of Thoughts](tree-of-thoughts.md)