---
title: "OpenSearch"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/opensearch.md]
related: [bm25-ranking, sharding, full-text-search]
tags: [search, distributed-systems, analytics, opensearch, lucene]
---

# OpenSearch

OpenSearch is a distributed search and analytics engine, providing scalable full-text search and document retrieval capabilities. It is built on the Lucene library.

## Architecture

OpenSearch operates on a cluster of nodes, each requiring significant memory and disk resources. A cluster manager node is elected to orchestrate cluster-level operations.

## Core Concepts

- **Documents**: Units of information stored in JSON format.
- **Indexes**: Collections of documents that can be searched.
- **Shards**: Indexes are split into shards for distribution across the cluster. Each shard is a full Lucene index. Shard size should be limited to 10–50 GB.
- **Primary and Replica Shards**: A shard may be a primary (original) or a replica (copy). Replica shards are distributed to different nodes than their corresponding primary shards for fault tolerance.
- **Search Terms**: Individual words in a search query.

## Search and Relevance

When searching, OpenSearch matches query terms against document terms and assigns a relevance score to each document. Relevance scoring combines:

- **Term Frequency**: How often a term appears in a document.
- **Inverse Document Frequency**: How rare the term is across the document collection.

OpenSearch uses the [BM25](bm25-ranking.md) ranking algorithm to calculate these relevance scores.

## Use Cases

OpenSearch is commonly used for:
- Full-text search engines
- Log analytics and monitoring
- RAG (Retrieval-Augmented Generation) document retrieval
- Knowledge bases requiring fast document search

## Sources
- [OpenSearch](../summaries/opensearch.md)

## Related
- [BM25 Ranking](bm25-ranking.md)
- [Sharding](sharding.md)
- [Full-Text Search](full-text-search.md)