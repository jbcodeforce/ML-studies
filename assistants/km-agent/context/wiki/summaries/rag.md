# Retrieval Augmented Generation (RAG) — Summary

**Source:** `raw/studies/genAI/rag.md`

Retrieval Augmented Generation (RAG) is a technique for supplementing LLMs with data outside their pre-training corpus, addressing knowledge cut-offs and reducing hallucinations by grounding responses in enterprise knowledge.

## Core Architecture

RAG follows three stages:
1. **Indexing** — Ingest documents, split into chunks (typically 256-512 tokens), compute embeddings, and store in a vector database.
2. **Retrieval** — Retrieve relevant chunks via similarity search (cosine similarity) and pass them as context.
3. **Generation** — LLM generates responses using the retrieved context via in-context learning.

## Key Components

- **Document Pipeline**: Text extraction from various formats (PDF, HTML, Doc), chunk creation aligned with document structure, and embedding computation.
- **Vector Stores**: ChromaDB, FAISS, Elasticsearch, Annoy, HNSW. Best practice is to isolate collections per domain.
- **Embeddings**: Numerical vector representations of document chunks enabling semantic similarity search.

## Challenges of Naive RAG

Naive RAG struggles with scalability, single-shot prompts, lack of query understanding, no decomposition, no tool use, no reflection or memory.

## Advanced Techniques

- **Query Transformations**: Multiple Query, RAG Fusion, Answer Recursively, Answer Individually, HyDE (Hypothetical Document Embedding).
- **Query Routing**: Route queries to domain-specific index subsets.
- **Knowledge Graph Integration**: Enhances RAG with ontology, document hierarchies, answer augmentation, and personalization.
- **Hybrid Search**: Combines semantic search with metadata filtering and keyword search (BM25).

## Frozen RAG

A variant with no training — data is purely in context. Prompt engineering and chunk/embedding selection are critical.

## Assessment & Evaluation

Key scoping questions cover end users, data sources, security, PII, chunk sizing, compliance, deployment, and expected queries. Evaluation approaches include FaaS (Facts as a Function) and ARES (Automated Retrieval Augmented Generation Evaluation System).

## Connections

- Builds on **text embeddings** and **reranking** for retrieval quality.
- Integrates with **LangChain** and **LangGraph** for orchestration.
- Enhanced by **knowledge graphs** for ontology and answer augmentation.
- Relates to **prompt engineering** for in-context learning optimization.