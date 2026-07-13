---
title: "NLP Summary"
created: 2026-07-12
updated: 2026-07-12
---

# Natural Language Processing (NLP)

## Summary

This document covers foundational NLP concepts including embeddings, BERT, and Named Entity Recognition.

**Embeddings** map data points into lower-dimensional vector spaces that capture semantic and syntactic relationships. Word2Vec (Google, 2014) introduced dense word embeddings enabling arithmetic on word vectors (e.g., similarity computation). Sentence embeddings vectorize complete sentences for semantic similarity. Embedding sizes typically range from 200–1000 dimensions. The embedding process is computationally expensive (days to complete) but models are usually saved and reused.

**Key embedding types**: word, sentence, image, graph, video. CLIP (OpenAI) embeds both text and images in the same vector space, enabling text-to-image generation.

**Cosine similarity** is the standard metric for comparing embeddings — it equals the cosine of the angle between two vectors.

**Use cases**: RAG with sentence embeddings and similarity search, product recommendations, anomaly detection, LLM token embedding, audio/video embedding.

**BERT** (Bidirectional Encoder Representations from Transformers) is a family of masked-language models from Google (2018). It is significantly smaller than modern LLMs, making it cost-effective for certain tasks, though it generally underperforms larger foundation models.

**Named Entity Recognition (NER)** identifies and extracts important entities (people, organizations, locations, etc.) from unstructured text using neural networks trained on labeled data. Some approaches leverage GenAI models with appropriate prompting.