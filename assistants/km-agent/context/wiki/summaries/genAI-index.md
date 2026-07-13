# Generative AI Overview

**Source:** `raw/studies/genAI/index.md` | **Created:** Aug 2023, Updated Aug 2024

## Summary

Generative AI combines neural network models to create new content (text, images, music, videos) from queries. Modern Gen AI models are based on the **Transformer architecture** and pre-trained on vast unlabeled datasets with 7B to 500B+ parameters. These models, called **foundation models (FMs)**, learn grammar, facts, and reasoning from internet-scale data.

## Key Concepts

- **Transformer Architecture**: Uses self-attention mechanisms to weigh word significance using context. Processing pipeline: tokenization → embedding → positional encoding → multi-head attention blocks → softmax → output tokens.
- **Training Pipeline**: Two-stage process — **pre-training** (predict next token on large corpus) and **fine-tuning** (task-specific adaptation). Evaluated via perplexity metric.
- **Transformer Types**: Encoder-only (similarity search), Encoder-decoder (text-to-text conversion), Decoder-only (text generation). Only the latter two are generative.
- **Customization Spectrum** (simplest to complex): Zero-shot inference → Prompt engineering → Few-shot → RAG → Fine-tuning → Pre-train FM → Build FM from scratch.

## Use Cases

Four main categories: customer experience (chatbots, summarization), employee productivity (code generation, Q&A agents), creativity (marketing, ideation), and business process optimization (intelligent document processing, data augmentation). Industries impacted include supply chain, quality control, education, safety, and travel.

## Challenges

LLMs face significant enterprise concerns: **accuracy** (hallucination, probabilistic outputs), **specificity** (trained on general data, not enterprise data), **cost** (training, inference, hosting), **skills gap**, **reliability** (no true reasoning or planning), and **legal issues** (IP, copyright, bias). A single LLM is unlikely to solve every business problem effectively.

## Key Players

Open-source models include BLOOM (46 languages), FLAN (instruction-tuned), and Mistral (Mixture of Experts). Proprietary models: GPT (OpenAI), Jurassic (AI21), Claude (Anthropic), Gemini (Google). Platforms: HuggingFace, Replicate, Vercel, Ollama, Amazon SageMaker.

## Connections
- Relates to [Transformer architecture](wiki/concepts/generative-ai-overview.md) in this article
- Connects to RAG, embeddings, and agentic AI concepts in the wiki
- Depends on prompt engineering and fine-tuning techniques