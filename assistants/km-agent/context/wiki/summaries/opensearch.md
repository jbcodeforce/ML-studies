# OpenSearch Summary

**Main Thesis:** OpenSearch is a distributed search and analytics engine built on Lucene, providing scalable full-text search and document retrieval.

**Key Concepts:**
- **Documents**: JSON-format units that store information; collected into **indexes**.
- **Cluster architecture**: Nodes require significant memory and disk; a cluster manager node is elected to orchestrate cluster-level operations.
- **Shards**: Indexes are split into shards (each a full Lucene index). Limit shard size to 10–50 GB. Shards can be **primary** (original) or **replica** (copy), with replicas distributed to different nodes for fault tolerance.
- **Search terms**: Individual words in a query. OpenSearch matches query terms to document words and assigns relevance scores.
- **Relevance scoring**: Uses **term frequency** (how often a term appears) combined with **inverse document frequency** (rarity across documents). The **BM25 ranking algorithm** computes document relevance scores.

**Connections:** OpenSearch is a key technology for search infrastructure, relevant to RAG systems (retrieval of documents), knowledge bases, and any system requiring fast full-text search over large document collections.