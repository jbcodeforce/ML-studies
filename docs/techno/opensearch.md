# OpenSearch

[OpenSearch](https://opensearch.org/docs/latest/about/)  is a distributed search and analytics engine. 

# Main concepts

**Documents** are units that store information, and are stored in json format. An **index** is a collection of documents.

Search are done on one or more nodes. Nodes need a lot of memory and disk. In cluster, there is a node which is elected as cluster manager, and orchestrates cluster-level operations.

OpenSearch splits indexes into **shards**. Each shard is actually a full Lucene index.  Limit shard size to 10–50 GB. A shard may be either a *primary* (original) shard or a *replica* (copy) shard. OpenSearch distributes replica shards to different nodes than their corresponding primary shards.

Doing a search, OpenSearch matches the words in the query to the words in the documents, and each document has a relevance score. Individual words in a search query are called search **terms**.

For relevance score, the **term frequency** is used, combined with the **inverse document frequency** which measure the number of document in which the word occurs. OpenSearch uses the BM25 ranking algorithm to calculate document relevance scores.