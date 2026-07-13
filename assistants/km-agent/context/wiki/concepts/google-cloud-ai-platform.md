---
title: "Google Cloud AI Platform"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/gcp/index.md]
related: [vertex-ai, gemini-llm, gemma-llm, document-ai-gcp]
tags: [gcp, google-cloud, ai-platform, ml, infrastructure]
---

# Google Cloud AI Platform

Google Cloud Platform (GCP) provides a comprehensive suite of managed services for developing, training, and deploying machine learning and Generative AI models. The platform is organized around several key service families.

## Core AI/ML Services

- **Vertex AI**: Unified platform for custom model training and Gen AI applications with SDKs in multiple languages
- **Model Garden**: Catalog of Google and third-party models
- **Gemini**: Multimodal LLM accepting text, image, video, audio, and document inputs
- **Gemma**: Open-weight family of lightweight LLMs (9B and 27B parameters)
- **Document AI**: Document processing and understanding service

## Development Infrastructure

- **Colab**: Browser-based Jupyter notebook environment with VM kernel and enterprise-grade security
- **Cloud Workstation**: Fully managed dev environment supporting containerized editors; includes Gemini Code Assist
- **Cloud Shell**: Free browser-based shell with 50h/week quota for infrastructure management and app development
- **Cloud Run**: Serverless platform for deploying web applications

## Compute

- **Compute Engine**: Linux/Windows VMs; e2-micro instance available in free tier; spot instances for cost savings
- **TPU**: Custom hardware designed for matrix operations common in ML; usable as worker nodes in GKE

## Sources
- [Google AI Platform](../summaries/index-gcp.md)

## Related
- [Vertex AI](vertex-ai.md)
- [Gemini LLM](gemini-llm.md)
- [Gemma LLM](gemma-llm.md)
- [Document AI (GCP)](document-ai-gcp.md)