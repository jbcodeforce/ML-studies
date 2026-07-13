---
title: "LLM Reference Architecture"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/ref-arch.md]
related: [rag-architecture, text-embeddings, agentic-ai, model-context-protocol]
tags: [llm, architecture, reference-architecture, data-pipeline, streaming, vector-store, orchestrator]
---

# LLM Reference Architecture

The LLM Reference Architecture is a six-component pattern for building production LLM solutions, originally defined by A16Z and extended with event-streaming capabilities. Developing custom LLM applications requires managing infrastructure across data, models, pipelines, prompts, context windows, application states, observability, embeddings, storage, caching, and augmented generation.

## Six Core Components

### 1. Data Pipelines
Batch processing that combines unstructured documents with structured data (CSV, JSON, SQL tables). Modern pipelines may invoke LLMs directly to build embeddings and persist them in a vector store, forming the ingestion stage of the RAG process.

### 2. Streaming
Event-driven integration where business services and microservices generate events that become part of the application's future context. A streaming pipeline consumes events, creates embeddings via LLM calls, and pushes them to the vector store for real-time knowledge enrichment.

### 3. Embeddings
Numerical vector representations of document chunks. Supported by open-source solutions like the Sentence Transformers library (Hugging Face) or proprietary hosted APIs.

### 4. Vector Store
Persists vectors with indexing and similarity search capabilities. Options include Faiss, ChromaDB, AWS OpenSearch, Redis, Kendra, OpenSearch Serverless, RDS for PostgreSQL, Aurora PostgreSQL, and Pinecone.

### 5. Hosted LLM
Model serving accessed via API, enabling applications to leverage LLM capabilities without managing inference infrastructure.

### 6. Orchestrator
The solution code that connects all components. Manages session caching in distributed cloud environments, performs semantic search via the vector store, and exposes APIs for chatbot or Q&A user interfaces.

## Architecture Extensions
The baseline A16Z architecture is extended with **event streaming** as a live source of data and knowledge, bridging traditional batch data pipelines with real-time event-driven architectures.

## Sources
- [Reference Architecture for LLM Solution](../summaries/ref-arch.md)

## Related
- [RAG Architecture](rag-architecture.md)
- [Text Embeddings](text-embeddings.md)
- [Agentic AI](agentic-ai.md)
- [Model Context Protocol](model-context-protocol.md)