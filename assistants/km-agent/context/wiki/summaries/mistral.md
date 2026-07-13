# Mistral.ai Summary

Mistral.ai is a French startup building mixture-of-experts (MoE) language models with both open-source and commercial offerings.

**Key Models:**
- **Open-weights**: Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
- **Commercial**: Mistral Small (classification, customer support), Mistral Large (complex reasoning, RAG, agents), Mistral Embeddings (55 MTEB score), Codetral (code generation)

All models support fine-tuning; function calling is available on Mistral Small, Large, and 8x22B. Models can be deployed locally via Docker, SkyPilot, or Ollama.

**Mixture of Experts (MoE):** A neural architecture where multiple specialist sub-models ("experts") handle different subsets of the input space, with a gating network routing inputs and combining outputs. This enables scaling model capacity while keeping per-token compute manageable. Example: language-pair-specific experts in translation.

**Connections:** MoE is a key architectural pattern complementing transformer-based LLMs. Mistral competes with Cohere, Anthropic/Claude, and OpenAI in the commercial LLM space.