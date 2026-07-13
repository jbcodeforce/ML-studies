---
title: "Vertex AI"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/gcp/index.md]
related: [google-cloud-ai-platform, gemini-llm, generative-ai-overview]
tags: [vertex-ai, google-cloud, ml-platform, gen-ai, grounding, rag]
---

# Vertex AI

Vertex AI is Google Cloud's unified platform for machine learning model training and Generative AI application development. It provides managed services for custom model training as well as access to pre-built Gen AI tools.

## Capabilities

- **Multi-language SDKs**: Python, Node.js, Go, Java, C#, and REST API
- **Request Augmentation**: Three methods to connect models to external knowledge:
  - **Grounding**: Connects model output to verifiable sources, reducing hallucinations. Can use Google Search for public knowledge or Vertex AI Search for enterprise data.
  - **RAG**: Retrieval-Augmented Generation
  - **Function Calling**: Allows models to invoke external APIs
- **Safety Checking**: Both prompts and responses are evaluated against safety categories

## Generative AI Workflow

Vertex AI supports a complete generative AI workflow from prompt to production, with integrated tools for testing, grounding, and deployment.

## Pricing

Cost is determined by the combination of Vertex AI tools and services used, plus associated storage, compute, and other Google Cloud resources.

## Sources
- [Google AI Platform](../summaries/index-gcp.md)

## Related
- [Google Cloud AI Platform](google-cloud-ai-platform.md)
- [Gemini LLM](gemini-llm.md)
- [Generative AI Overview](generative-ai-overview.md)