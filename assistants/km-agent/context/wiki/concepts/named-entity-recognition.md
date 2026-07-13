---
title: "Named Entity Recognition"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/nlp.md]
related: [text-embeddings, bert, classification]
tags: [ner, nlp, entity-extraction, information-extraction, nlp-task]
---

# Named Entity Recognition

**Named Entity Recognition (NER)** is a Natural Language Processing (NLP) technique used to identify and extract important entities from unstructured text data.

## How It Works

NER is achieved using neural networks trained on labeled data. The model learns to recognize patterns in text that correspond to entity types such as:

- **Persons** — names of people
- **Organizations** — companies, institutions
- **Locations** — cities, countries, landmarks
- **Dates/Times** — temporal expressions
- **Monetary values** — currency amounts
- **Miscellaneous entities** — product names, events, etc.

## Approaches

1. **Neural network-based** — traditional approach using NN models trained on labeled datasets to recognize entity patterns
2. **GenAI prompt-based** — newer approaches leverage generative AI models with appropriate prompts to perform NER without explicit training data

## Use Cases

- Information extraction from documents
- Knowledge graph population
- Search and retrieval enhancement
- Document categorization and tagging

## Sources
- [NLP Summary](../summaries/nlp.md)

## Related
- [Text Embeddings](text-embeddings.md)
- [BERT](bert.md)
- [Classification](classification.md)