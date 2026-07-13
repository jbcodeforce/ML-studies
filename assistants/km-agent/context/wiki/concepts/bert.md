---
title: "BERT"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/nlp.md]
related: [text-embeddings, transformer-architecture, transfer-learning]
tags: [bert, nlp, transformer, masked-language-model, google]
---

# BERT

**BERT** (Bidirectional Encoder Representations from Transformers) is a family of masked-language models published in 2018 by researchers at Google.

## Overview

BERT uses bidirectional training, meaning it learns from both left and right context simultaneously. It is based on the Transformer architecture, specifically the encoder stack.

## Characteristics

- **Smaller than modern LLMs** — BERT is significantly smaller than current foundation models, making it more resource-efficient
- **Cost-effective** — for tasks where BERT performs adequately, it is a practical choice for developers due to lower compute requirements
- **Masked language modeling** — trained by predicting masked (hidden) tokens in a sentence, learning bidirectional context
- **Performance tradeoff** — generally underperforms larger foundation models on complex tasks because of its smaller size

## When to Use BERT

- Tasks that do not require the full capacity of large foundation models
- Environments with limited compute resources
- Tasks where bidirectional context is important but generation is not required

## Sources
- [NLP Summary](../summaries/nlp.md)

## Related
- [Text Embeddings](text-embeddings.md)
- [Transformer Architecture](transformer-architecture.md)
- [Transfer Learning](transfer-learning.md)