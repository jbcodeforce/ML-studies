# Reference Architecture for LLM Solutions

## Main Thesis
Developing and serving custom LLMs requires extensive infrastructure spanning data management, embeddings, vector storage, orchestration, and integration with streaming/event-driven systems. The A16Z reference architecture provides a foundational six-component model, extended here with event streaming capabilities.

## Key Components
1. **Data Pipelines** — Batch processing combining unstructured documents with structured data (CSV, JSON, SQL). Modern pipelines may call LLMs directly to build embeddings, feeding into vector stores via RAG patterns.
2. **Streaming** — Event-driven pipelines consuming microservice events, creating embeddings via LLM calls, and pushing to vector stores for real-time context enrichment.
3. **Embeddings** — Numerical vector representations of document chunks, created via open-source tools (Sentence Transformers from Hugging Face) or proprietary hosted APIs.
4. **Vector Store** — Persistence layer with indexing and similarity search. Options include Faiss, ChromaDB, AWS OpenSearch, Redis, Pinecone, and various PostgreSQL variants.
5. **Hosted LLM** — Model serving via API, enabling access without local inference infrastructure.
6. **Orchestrator** — Connects all components; manages session caching, semantic search, and exposes APIs for chatbot or Q&A interfaces.

## Connections
- Extends the RAG architecture pattern by adding event streaming as a knowledge source.
- Relates to the Agentic AI orchestrator concept, which ties components together.
- Vector store and embeddings concepts are foundational to RAG and semantic search.