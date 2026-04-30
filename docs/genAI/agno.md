# Agno studies

[Agno](https://www.agno.com/) seems to be one of the best SDK for developing agents. [See my code with ollama as local server](https://github.com/jbcodeforce/ML-studies/tree/master/src/agentic/agno).

**The Core Concepts**

* [Agents](https://docs.agno.com/agents/overview) are a stateful control loop around a stateless LLM. 
* [Database](https://docs.agno.com/database/overview) to get persistent storage for sessions, context, memory, learnings, and evaluation datasets.
* [storage](https://docs.agno.com/database/session-storage) for conversation history. Sessions are stored automaticaly once a database is added to the agent
* [memory](https://docs.agno.com/memory/overview) for  user preferences
* [state]() is structured data the agent actively manages: counters, lists, flags. An agent can use across runs. State variables can be injected into instructions with {variable_name}

## Agent

=== "Ollama"
    ```
    from agno.agent import Agent
    from agno.models.ollama import Ollama
    agent = Agent(model=Ollama(id="gemma4:26b"), markdown=True)
    ```

## Knowledge

The simplest way to give an agent access to documents. Content is automatically retrieved and injected into the system prompt before the agent responds. With an agentic approach, the agent gets a search_knowledge_base tool and decides when to query the knowledge base. The agent can choose to search multiple times, refine queries, or skip searching entirely.

This is the default behavior when you set knowledge on an Agent.

Steps:

1. Create a Knowledge base with a vector database

    === "OpenAI"
        ```python
        from agno.knowledge.knowledge import Knowledge
        from agno.vectordb.qdrant import Qdrant
        from agno.vectordb.search import SearchType
        from agno.knowledge.embedder.openai import OpenAIEmbedder
        
        knowledge = Knowledge(
            vector_db=Qdrant(
                collection="basic_rag",
                url=qdrant_url,
                search_type=SearchType.hybrid,
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
        )
        # Agentic by setting search_knowledge=True,
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            knowledge=knowledge,
            search_knowledge=True,
            markdown=True,
        )
        ```

    === "Local"
        Using ollama for embedding
        ```python
        from agno.knowledge.knowledge import Knowledge
        from agno.knowledge.embedder.ollama import OllamaEmbedder
        from agno.vectordb.chroma import ChromaDb
        from agno.vectordb.search import SearchType

        knowledge_chroma = Knowledge(
            vector_db=ChromaDb(
                    collection="local_demo",
                    search_type=SearchType.hybrid,
                    embedder=OllamaEmbedder(
                        id="nomic-embed-text",
                        dimensions=768,
                    ),
                ),
            )
        ```


2. Load a document, from local files, URLs, raw text, topics (Wikipedia/ArXiv), and batch operations.
    ```python
    from agno.knowledge.knowledge import Knowledge
    from agno.knowledge.reader.wikipedia_reader import WikipediaReader
    await knowledge.ainsert(
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            metadata={"source": "url"},
        )
    await knowledge.ainsert(
            name="CV",
            path="cookbook/07_knowledge/testing_resources/cv_1.pdf",
            metadata={"source": "local_file"},
        )
     await knowledge.ainsert(
            name="Company Info",
            text_content="Acme Corp was founded in 2020. They build AI tools for developers.",
            metadata={"source": "text"},
        )
     await knowledge.ainsert(
            topics=["Retrieval-Augmented Generation"],
            reader=WikipediaReader(),
        )
    ```
3. Create an Agent with search_knowledge=True (the default)
4. Ask questions - agent decides when to search

In production, knowledge needs to be managed over time:

- Skip re-inserting content that already exists
    ```python
     await knowledge.ainsert(
            name="Recipes",
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            skip_if_exists=True,  # Won't re-process since content hash matches
        )
    ```
- Remove outdated content
    ```python
    await knowledge.aremove_vectors_by_name("Recipes")
    ```
- Track content status with a contents database
    ```python
    knowledge = Knowledge(
        name="Lifecycle Demo",
        vector_db=Qdrant(
            collection="lifecycle_demo",
            url=qdrant_url,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
        # Contents DB tracks ingested content, status, and metadata
        contents_db=SqliteDb(
            db_file="tmp/agent.db",
        ),
    )
    ```
- Re-index when content changes

### Different search mechanism
Knowledge supports three search types. Each has different strengths:

- Vector: Semantic similarity search. Finds conceptually related content
  even when exact words don't match.
- Keyword: Full-text search. Fast and precise for exact term matching.
- Hybrid: Combines vector + keyword. Best of both worlds. Recommended default.

```python
from agno.vectordb.search import SearchType
search_types = [
            (SearchType.vector, "Vector (semantic similarity)"),
            (SearchType.keyword, "Keyword (full-text search)"),
            (SearchType.hybrid, "Hybrid (vector + keyword)"),
        ]
vector_db = ChromaDb(
    collection="studies",
    path=get_vstore_path(),
    persistent_client=True,
    search_type=SearchType.hybrid,
    embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
)
```

### Chunking Strategies

```python
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.markdown import MarkdownChunking
markdown_reader = PDFReader(chunking_strategy=MarkdownChunking())

await knowledge.ainsert(url=pdf_url, reader=markdown_reader)
```

### Reranking

Reranking is a two-stage retrieval process:

1. First, retrieve candidate results using vector/hybrid search
2. Then, a reranker model scores and reorders results by relevance

This dramatically improves result quality, especially for complex queries.

Supported rerankers:

- CohereReranker: Cohere's rerank models (recommended)
- SentenceTransformerReranker: Local reranking with BAAI/bge models
- InfinityReranker: Self-hosted reranking
- BedrockReranker: AWS Bedrock reranking

```python
from agno.knowledge.reranker.cohere import CohereReranker
kn = Knowledge(
    vector_db=ChromaDb(
            collection="local_demo",
            search_type=SearchType.hybrid,
            embedder=OllamaEmbedder(
                        id="nomic-embed-text",
                        dimensions=768,
                    ),
            reranker=CohereReranker(model="rerank-multilingual-v3.0"),
    )
)
```

### Filtering

Filters let you narrow search results based on document metadata. This is essential for multi-user, multi-topic, or access-controlled systems. 

Two stages of filtering:

1. On load: Tag documents with metadata at insert time
2. On search: Apply filters when the agent searches

Filter approaches:

- Dict filters: Simple key-value matching {"category": "recipes"}
- FilterExpr: Powerful expressions with AND, OR, NOT, EQ, IN, GT, LT

In the knowledge definition, Embedders convert text into vectors for semantic search.

```python
from agno.filters import AND, EQ, GT, IN, NOT, OR
agent_dict = Agent(
        model=OpenAIResponses(id="gpt-5.2"),
        knowledge=knowledge,
        search_knowledge=True,
        knowledge_filters={"cuisine": "thai"},
        markdown=True,
    )
# OR
knowledge_filters=[OR(EQ("category", "recipes"), EQ("category", "docs"))],
knowledge_filters=[GT("difficulty", 2)],
knowledge_filters=[NOT(EQ("category", "docs"))],
```

With agentic filtering enabled, the agent inspects available metadata keys in the knowledge base and dynamically builds filters from the user query.

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    knowledge=knowledge,
    search_knowledge=True,
    enable_agentic_knowledge_filters=True,
    markdown=True,
)
```

### Sharing vector store
When multiple Knowledge instances share the same vector database, use isolate_vector_search to ensure each instance only searches its own data.

This is essential for multi-tenant applications where different users or departments should only access their own documents.

Behavior:

- isolate_vector_search=False (default): Searches ALL vectors in the database.
- isolate_vector_search=True: Only searches vectors tagged with this instance's name.

Approach:

- Crearte one vector store
- Create as many knowledge (with different name) as needed using the same vector store 
- Insert different documents inside each knowledge
- create as many agents as knowledgem and route query to the agent.

### Knowledge Graph

Unlike standard vector-based RAG, LightRAG:

- Extracts entities and relationships from documents
- Builds a knowledge graph for multi-hop reasoning
- Supports graph-traversal queries