---
title: "Mixture of Experts"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/mistral.md]
related: [mistral-ai, transformer-architecture]
tags: [mixture-of-experts, moe, neural-networks, architecture]
---

# Mixture of Experts (MoE)

Mixture of Experts is a neural network architecture that combines multiple specialized sub-models ("experts") to make predictions or decisions. Each expert specializes in a specific subset of the input space and provides its own prediction. A **gating network** determines which experts to activate for a given input, and their outputs are combined to produce the final result.

## Key Characteristics

- **Specialization**: Each expert focuses on a particular domain or pattern within the input space
- **Conditional Computation**: Only a subset of experts is activated per input, reducing compute costs
- **Gating Network**: A learned routing mechanism selects relevant experts dynamically

## Use Cases

- **Language translation**: Experts organized by language pairs
- **Complex diverse data**: Different experts extract different aspects or patterns
- **Large-scale LLMs**: Models like Mixtral 8x7B and Mixtral 8x22B use MoE to achieve high capability with efficient compute

## Advantages over Dense Models

MoE architectures can scale model capacity (total parameters) while keeping per-inference compute manageable, since only a fraction of experts processes each token.

## Sources
- [Mistral.ai Raw Notes](../summaries/mistral.md)

## Related
- [Mistral AI](mistral-ai.md)
- [Transformer Architecture](transformer-architecture.md)
- [LLM Training Pipeline](llm-training-pipeline.md)