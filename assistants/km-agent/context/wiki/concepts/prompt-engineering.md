---
title: "Prompt Engineering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
related: [chain-of-thought, instruction-tuning, prompt-chaining, tree-of-thoughts, automatic-prompt-engineering]
code: [code/LLM/langchain/bedrock/]
tags: [prompt-engineering, llm, genai, prompting]
---

# Prompt Engineering

Prompt engineering is the practice of developing and optimizing prompts to efficiently use Large Language Models (LLMs) for a variety of applications. It is still a major research topic.

## Definition

A **prompt** is an input that the model uses as the basis for generating text — comprising instructions and context passed to a language model to achieve a desired task. While all information may be encoded in the model, knowledge extraction can be hit or miss.

## Key Characteristics

- **Prompt sensitivity**: LLMs are very sensitive to small perturbations of the prompt. A single typo or word change can alter the output.
- **Robustness evaluation**: There is still need to evaluate models' robustness to prompts.
- **Domain knowledge**: Prompts can help incorporate domain knowledge on specific tasks and improve interpretability.

## Core Elements of a Prompt

A prompt typically contains any combination of:
- **Instruction**: What the model should do
- **Context**: Background information
- **Input data**: The data to operate on
- **Output indicator**: Expected format or type of output

## Tuning Parameters

Several parameters influence prompt responses:
- **Temperature**: Controls randomness of output
- **Top-P**: Nucleus sampling parameter
- **Max length**: Maximum token limit
- **Stop sequence**: Token at which generation halts
- **Frequency penalty**: Penalty on next token already present in response
- **Presence penalty**: Limits repeating phrases

## Best Practices (OpenAI)

1. Write clear instructions (e.g., "ask for brief replies", "expert-level writing")
2. Provide reference text
3. Split complex tasks into simpler subtasks
4. Give the model time to "think"

## Prompting Approaches

- **Zero-shot prompting**: A single question with no examples or prior instruction.
- **Few-shot prompting**: Includes sample Q&A pairs to condition the model on new context.
- **Directional Stimulus Prompting**: Providing hints to guide the response.

## Sources
- [Prompt Engineering Summary](../summaries/prompt-eng.md)

## Related
- [Chain of Thought](chain-of-thought.md)
- [Instruction Tuning](instruction-tuning.md)
- [Prompt Chaining](prompt-chaining.md)
- [Tree of Thoughts](tree-of-thoughts.md)
- [Automatic Prompt Engineering](automatic-prompt-engineering.md)