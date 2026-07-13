---
title: "AI Discussions Review"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/review.md]
tags: [genai, review, llm, skills, prompt-engineering, rag, agentic]
---

# AI Discussions Review

A curated discussion guide covering eight key areas for Gen AI conversations, interviews, and chit-chat, with references to code examples and study materials.

## Overview

This document serves as a topical index linking to deeper material across the ML-studies codebase. It is organized into eight discussion themes:

1. **LLM Fundamentals** — Transformer architecture, pre-training, fine-tuning, RAG, embeddings, and inference parameters (temperature, top-p). Includes code samples for OpenAI API, LangChain chains, FAISS retrieval, and local LLMs via Ollama.

2. **Prompt Engineering** — Zero-shot/few-shot prompting, Chain of Thought, prompt chaining, Tree of Thoughts, Automatic Prompt Engineering (APE), and ReAct. Code examples include CoT with Bedrock, Program-Aided Language with Claude, and a 5-stage critical thinking framework.

3. **LLM Project Examples** — Hands-on RAG implementations (multi-query, fusion, HyDE, adaptive routing), vector store construction with ChromaDB, agentic workflows with Agno, and tool-calling agents. Includes a Streamlit RAG demo and end-to-end PDF Q&A app.

4. **Research Updates** — Current topics in few-shot/zero-shot learning, instruction tuning, Chain of Thought, Tree of Thoughts, Agentic AI, MCP, and Hermes agents.

5. **Model Architectures** — Transformer networks (GPT-3 vs Codex), self-attention, encodings, encoder-decoder vs decoder-only models, transfer learning, and distributed training (DDP).

6. **Skills** — Packaging reusable agent capabilities as self-contained folders with `SKILL.md` (YAML frontmatter + procedural markdown), scripts, and templates. Key principles: hyper-focused descriptions as triggers, deterministic scripts over natural language for non-improvisational steps, preventing plan drift with clear step boundaries, and paired simulation evaluation. References research from Le Chen, ACES framework, and SkillComposer.

7. **Fine-Tuning Techniques** — Supervised fine-tuning, parameter-efficient fine-tuning, few-shot learning, instruction tuning, and tradeoffs between RAG and fine-tuning.

8. **Production Engineering** — Tokenization, embeddings, deployment patterns, streaming APIs, feature stores (Feast, FeatureForm), model evaluation, and monitoring (LiteLLM proxy with Prometheus).

## Connections

This review cross-references material in the genAI section (index, rag, prompt-eng, agentic, agno, mcp, hermes, openai, anthropic) and coding tools (langchain, langgraph, ddp, haystack). It ties together discussion topics with concrete code examples in the ML-studies repository.