---
title: "Prompt Engineering Summary"
created: 2026-07-12
sources: [raw/studies/genAI/prompt-eng.md]
---

# Prompt Engineering Summary

Prompt engineering is the practice of developing and optimizing prompts to efficiently use LLMs for various applications. A prompt is the input (instructions and context) passed to a language model to achieve a desired task. The field remains a major research topic.

## Key Points

- **Prompt sensitivity**: LLMs are highly sensitive to small perturbations; a single typo or word change can alter outputs significantly.
- **Instruction tuning**: Modern LLMs are fine-tuned using instruction tuning, enabling them to respond to prompts without task-specific fine-tuning (no weight updates).
- **Tuning parameters**: Temperature, Top-P, max length, stop sequences, frequency penalty, and presence penalty all influence response quality.

## Prompting Techniques Covered

1. **Zero-shot**: A single question with no examples or instructions.
2. **Few-shot**: Includes sample Q&A pairs to guide the model.
3. **Chain of Thought (CoT)**: Uses intermediate reasoning steps; zero-shot CoT works by adding "Let's think step by step." Effective for arithmetic, commonsense, and symbolic reasoning.
4. **Prompt Chaining**: Breaks complex tasks into subtasks, passing responses between them. Increases transparency, controllability, and reliability.
5. **Tree of Thoughts**: Generalizes CoT by organizing thoughts in a tree structure, using BFS/DFS search to find optimal thought sequences.
6. **Automatic Prompt Engineering (APE)**: Uses LLMs as both inference and scoring models to optimize prompts algorithmically. DSPy is a notable framework.
7. **Self-Consistency**: Samples multiple diverse reasoning paths via few-shot CoT and selects the most consistent answer.
8. **Directional Stimulus Prompting**, **Program-Aided Language Models (PAL)**, and **ReAct Prompting** are additional techniques.

## RLHF

Reinforcement Learning from Human Feedback scales instruction tuning by aligning models to human preferences.

## OpenAI Best Practices

- Write clear instructions
- Provide reference text
- Split complex tasks into simpler subtasks
- Give the model time to "think"

## Practical Resources

- Code samples available under `code/LLM/langchain/bedrock/` with Bedrock client tests.
- AWS Bedrock text playground and workshop notebooks are referenced for experimentation.