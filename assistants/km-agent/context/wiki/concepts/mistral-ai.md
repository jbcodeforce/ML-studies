---
title: "Mistral.ai"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/mistral.md]
tags: [mistral, llm, genai, open-source, mixture-of-experts]
---

# Mistral.ai

Mistral.ai is a French startup specializing in mixture-of-experts (MoE) language models with both open-weights and commercial offerings.

## Open-Weights Models
- **Mistral 7B** — Lightweight open model
- **Mixtral 8x7B** — MoE-based open model
- **Mixtral 8x22B** — Larger MoE open model

## Commercial Models
- **Mistral Small** — Classification, customer support, text generation
- **Mistral Medium** — General-purpose commercial tier
- **Mistral Large** — Complex reasoning, code generation, RAG, agents
- **Mistral Embeddings** — Retrieval embeddings (55 score on MTEB)
- **Codetral** — Code generation specialist

All models support fine-tuning. Function calling is available on Mistral Small, Large, and 8x22B.

## Deployment
Models are available as Docker images and can run locally via [SkyPilot](https://skypilot.readthedocs.io/), [Ollama](https://ollama.com/), or Docker Compose.

## Mixture of Experts (MoE)
MoE architecture combines multiple specialist models, routing inputs to relevant experts via a gating network. This allows handling complex, diverse data where each expert extracts different patterns — for example, language-pair-specific experts in translation.

## Sources
- [Mistral.ai Raw Notes](../summaries/mistral.md)

## Related
- [Cohere Platform](cohere-platform.md)
- [Anthropic and Claude Models](anthropic-claude.md)
- [Transformer Architecture](transformer-architecture.md)