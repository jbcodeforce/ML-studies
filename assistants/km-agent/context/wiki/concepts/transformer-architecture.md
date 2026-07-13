---
title: "Transformer Architecture"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/index.md]
related: [generative-ai-overview, text-embeddings, llm-training-pipeline]
tags: [transformer, architecture, attention, nlp, neural-networks]
---

# Transformer Architecture

The Transformer is a neural network architecture used to generate the next word in a sentence using probability. It is the foundation of most modern Generative AI models.

## Core Mechanism

The **self-attention mechanism** helps weight the significance of different words by taking into account previously seen context. The attention mechanism computes similarity between tokens (from embeddings) in a sequence. The closer two words are in vector space, the higher their attention scores.

## Processing Pipeline

1. **Tokenization**: Words, prefixes, suffixes, and punctuation are assigned matching tokens (average 5 chars per token)
2. **Embedding**: Tokens are transformed into numerical vectors
3. **Positional Encoding**: Predefined vectors are added to embeddings, ensuring sentences with same words in different order get different vectors
4. **Multi-Head Attention**: Added at every block of the feedforward network; several different embeddings modify vectors to add context
5. **Transformer Blocks**: Layers stacked on top of each other, with feed-forward neural networks within each layer
6. **Softmax Layer**: Converts scores into probabilities (summing to 1)
7. **Output Tokens**: Results are converted back into readable text

## Transformer Types

1. **Encoder-only**: Generate no human-readable content; used for efficient content querying and similarity search
2. **Encoder-decoder**: Treats every NLP problem as text-to-text conversion (e.g., translation)
3. **Decoder-only**: Used for text generation

Only encoder-decoder and decoder-only are **generative** models.

## Key Properties

- Transformers process the entire input at once during learning, enabling **parallelization**
- An encoder component converts input text into embeddings
- A decoder component consumes embeddings to emit output text
- Models are trained on terabytes of text data (books, articles, websites)

## Sub-word Tokenization

Combines the benefits of character and word tokenization by breaking down rare words into smaller units while keeping frequent words as unique entities.

## Sources
- [Generative AI Index](../summaries/genAI-index.md)

## Related
- [Generative AI Overview](generative-ai-overview.md)
- [Text Embeddings](text-embeddings.md)
- [LLM Training Pipeline](llm-training-pipeline.md)