---
title: "Retrieval Augmented Generation (RAG)"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/rag.md]
related: [text-embeddings, reranking, langchain-framework, knowledge-graph-integration, prompt-engineering]
code: [code/LLM/langchain/openAI/]
tags: [rag, retrieval-augmented-generation, vector-store, embeddings, query-transformations, knowledge-graph, in-context-learning]
---

# Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is a technique that supplements generative text models with data outside of their pre-training corpus, addressing the knowledge cut-off limitation of LLMs and reducing hallucinations by grounding responses in enterprise knowledge.

## Core Architecture

RAG follows a three-stage process:

1. **Indexing** — Batch processing to ingest documents from various sources, split them into chunks, compute embeddings, and store them in a vector database. Chunking strategies balance preserving context with accuracy; typical chunk sizes range from 256-512 tokens.

2. **Retrieval** — Retrieves relevant document chunks from the vector store using similarity search (cosine similarity between embeddings) and passes them to the LLM as context.

3. **Generation** — The LLM generates a natural language response using the retrieved context, leveraging its in-context learning capability.

## Document Pipeline

The preparation pipeline is critical for retrieval quality:

- **Text extraction** — Isolate relevant textual information from various formats (PDF, HTML, Doc, Markdown). This is the most expensive task. Tools like Unstructured.io and Apache Spark can be used.
- **Chunk creation** — Segment text into smaller chunks. Align chunking breaks with document structure (sections, titles). Overlapping between chunks preserves semantic continuity.
- **Embedding** — Convert chunks to numerical vector representations using embedding models.

## Vector Stores

Vector databases store document chunks and their embeddings for efficient similarity search. Popular options include:
- ChromaDB, FAISS, Elasticsearch, Annoy, HNSW

Best practice: isolate vector store **collections** per domain to reduce noise and improve accuracy.

## Challenges of Naive RAG

Naive RAG has significant limitations:
- Hard to scale reliably on large knowledge corpora
- Limited to single-shot prompts
- No query understanding — just semantic search
- No query decomposition, tool use, reflection, or memory

## Advanced RAG Techniques

### Query Transformations
- **Multiple Query** — LLM generates 4-5 variations of the user's question to cast a wider retrieval net
- **RAG Fusion** — Apply merging logic with filtering and heuristics
- **Answer Recursively** — Chain Q&A, using previous responses as context
- **Answer Individually** — Answer sub-questions separately, then synthesize
- **HyDE (Hypothetical Document Embedding)** — Generate a hypothetical document first, then retrieve similar passages

### Query Routing
Route queries to different index subsets based on domain classification using logical and semantic routing.

### Knowledge Graph Integration
Knowledge graphs enhance RAG by:
- Providing enterprise-specific ontology and term definitions
- Adding document hierarchies to guide chunk selection
- Augmenting answers with information missing from the vector database
- Personalizing responses and eliminating repetition

### Hybrid Search
Combines semantic search with metadata filtering and keyword search (e.g., BM25) for improved retrieval quality.

## Frozen RAG

Frozen RAG uses no training — data is passed purely in context. The prompt drives the LLM to maximize in-context learning. Selection of the right data chunks and embedding model is crucial.

## Retriever Considerations

- Fine-tune RAG models for domain-specific retrieval and generation
- Optimize retrieval parameters: number of documents, vector size, text length, parallel queries
- Use metadata for hybrid search strategies
- Evaluate retrieval quality using similarity matrices and rule-based systems
- Address hallucination prevention, privacy protection, and source quality control
- Start small with guardrails in place

## RAG Assessment Scoping Questions

Key questions before implementation:
- Who is the end user and what is the brand impact?
- What are the data sources, pipelines, and security boundaries?
- How often does documentation change?
- How to handle PII and malicious content?
- What chunk size and overlap are appropriate?
- What are the most common expected queries?
- How to handle out-of-domain queries?
- What are the compliance and regulatory requirements?
- Self-hosted or API-based deployment? Expected latency and cost?

## Evaluation

Traditional evaluation uses human annotations. Newer approaches include:
- **FaaS (Facts as a Function)** — Callable functions using JSON objects for interpretable evaluation
- **ARES** — Uses LLM-generated query-passage-answer triples and fine-tuned LLM judges

## Sources
- [Retrieval Augmented Generation (RAG)](../summaries/rag.md)

## Related
- [Text Embeddings](text-embeddings.md)
- [Reranking](reranking.md)
- [LangChain Framework](langchain-framework.md)
- [Prompt Engineering](prompt-engineering.md)