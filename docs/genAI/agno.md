---
title: "Agno studies"
source: local-import
ingested: 2026-06-19
tags: []
type: article
compiled: false
---

# Agno studies

[Agno](https://www.agno.com/) seems to be one of the best SDK for developing agents and agentic solutions. [See my code with ollama, or oMLX as local server](https://github.com/jbcodeforce/ML-studies/tree/master/code/agents/agno). It is a minimalist, production-ready that emphasizes deterministic behavior transparency and simplicity. It has a lot of powerful tools and constrcuts, like knowledge, learning, team, workflow, and rest API.

**The Core Concepts**

* [Agents](https://docs.agno.com/agents/overview) are a stateful control loop around a stateless LLM. 
* [Database](https://docs.agno.com/database/overview) to get persistent storage for sessions, context, memory, learnings, and evaluation datasets.
* [Tools](https://docs.agno.com/tools/overview)
* [storage](https://docs.agno.com/database/session-storage) for conversation history. Sessions are stored automaticaly once a database is added to the agent
* [memory](https://docs.agno.com/memory/overview) for  user preferences
* [Learning](https://docs.agno.com/learning/overview) to capture user profiles, memories, and knowledge over time
* [Knowledge and Rag](https://docs.agno.com/knowledge/overview) to manage domain specific information. [See my own code translation from Agno cookbook to run locally](https://github.com/jbcodeforce/ML-studies/tree/master/code/agents/agno/knowledge), and the bigger usage in [km-agent](https://github.com/jbcodeforce/km-agent)
* [state]() is structured data the agent actively manages: counters, lists, flags. An agent can use across runs. State variables can be injected into instructions with {variable_name}

## Agent

The [Agno cookbook quickstarts](https://github.com/agno-agi/agno.git) present the building blocks to get started and the [first_mlx_agent_with_tool.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/agno/first_mlx_agent_with_tool.py) or [ollama_agent_with_tool.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/agno/ollama_agent_with_tool.py) are other example with oMLX or Ollama.

=== "Ollama"
    ```python
    from agno.agent import Agent
    from agno.models.ollama import Ollama
    agent = Agent(model=Ollama(id="gemma4:26b"), markdown=True)
    ```

=== "OpenAI  - oMLX local"
    Also used for any OpenAI compatible LLMs (base_url finishes with '/v1')
    ```python
    from agno.models.openai.like import OpenAILike
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")  # finish with /v1
    model =OpenAILike(id=model_id, api_key=api_key.strip(), base_url=base_url)
    agent = Agent(model=model,
    ```

=== "Anthropic"
    ```python
    api_key = os.getenv("ANTHROPIC_API_KEY")
    Claude(id=mid, api_key=api_key.strip())
    ```

## Tools

Agent needs [tools](https://docs.agno.com/tools/overview) to perform some more advanced actions on external systems. 


## Development approach

* Declare and unit test all the tools to be used
* Prepare some queries  to validate
* Use the same integration pattern with backend LLM, externalize URL, API keys, model reference.
* Fine tune the instructions
* Integrate agent in Workflow and/or Team

## Knowledge

The simplest way to give an agent access to documents. Agno (07/2026) supports different ways to manage knowledge:

1. RAG with content automatically retrieved and injected into the system prompt before the agent responds. 
1. With an agentic approach, the agent gets a `search_knowledge_base` tool and decides when to query the knowledge base. The agent can choose to search multiple times, refine queries, or skip searching entirely.
1. Graph

*07/2026: integrated code study of Agno cookbook knowledge folder*

Knowledge supports three search types. Each has different strengths:

- Vector: Semantic similarity search. Finds conceptually related content even when exact words don't match.
- Keyword: Full-text search. Fast and precise for exact term matching.
- Hybrid: Combines vector + keyword.

#### Things to consider to develop a good searchable content

* Define the goal of the search and how users process those query: just chat, build more content, human in the loop
* List of sources a manifest of URLs or  local files
* Vector sizefor the embeddings and embedder selection: Are you looking for an English-only or multilingual embedding model?
* Define the type and size of chunking. Chunking determines how documents are split into pieces for embedding and search. The right strategy depends on your content type.
* Select vector store according to scaling needs: dedicated vector store techno or SLQ based database with vectors support

* Load documents with idempotency or avoid reloading the same document
* How to assess quality of the responses

#### Steps:

1. Create a Knowledge base with a vector database. [See OMLX with Qdrant](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/agno/knowledge/omlx_adv_rag.py)
    ```python
    k=Knowledge(
    vector_db=Qdrant(
        collection="basic_rag",
        url=qdrant_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
    ```

    * Traditional RAG is agent with `add_knowledge_to_context` and `search_knowledge=false`
        ```python
        agent = Agent(
            model=OpenAILike(id=DEFAULT_LLM_MODEL),
            knowledge=knowledge,
            add_knowledge_to_context=True,
            search_knowledge=False,
        ```
    * Agentic RAG use search knowledge base tool: `search_knowledge=True`
        ```python
        agent = Agent(
            model=OpenAILike(id=DEFAULT_LLM_MODEL),
            knowledge=knowledge,
            search_knowledge=True,
            markdown=True,
        )
        ```


2. Load documents using [Readers](https://docs.agno.com/knowledge/concepts/readers/overview). Use metadata to help filtering on search. This is the basic code.
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
    await knowledge.ainsert(
            name="Agno README",
            remote_content=github_config.file("README.md", repo="agno-agi/agno"),
        )

    ```

    The `Knowledge` class supports loading content from many sources: local files, URLs, raw text, topics (Wikipedia/ArXiv), and batch operations. So [this code](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/agno/knowledge/prepare_vs.py) is a more sophisticate document processor based on manifests.
    `knowledge.insert()` automatically selects the right reader based on file extension or URL. It is also possible to integrate with file from github via [GitHubConfig](https://docs.agno.com/knowledge/concepts/cloud-storage#githubconfig)

    For embedder, it is possible to use local embedders like [FastEmbedEmbedder](https://docs.agno.com/knowledge/concepts/embedder/qdrant-fastembed/qdrant-fastembed), or using openaiembedder with configuration for local LLM.
3. Create an Agent with search_knowledge=True (the default)
4. Ask questions - agent decides when to search

#### Document processing
In production, knowledge needs to be managed with minimum governance:

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
- Track content status with a contents database(could be manifest file with sha256 on source content)
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

### Different search mechanisms

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

Unlike standard vector-based RAG, [LightRAG](https://github.com/hkuds/lightrag):

- Extracts entities and relationships from documents
- Builds a knowledge graph for multi-hop reasoning
- Supports graph-traversal queries

## Knowledge Tools
KnowledgeTools provides a richer set of tools for knowledge interaction beyond basic search:

- think: Agent reasons about the query before searching
- search: Standard knowledge base search
- analyze: Deep analysis of search results

```python
knowledge_tools = KnowledgeTools(
    knowledge=knowledge,
    enable_think=True,
    enable_search=True,
    enable_analyze=True,
    add_few_shot=True,
)
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[knowledge_tools],
    markdown=True,
)
```

It is also possible to implement custom knowledge sources on non-standard sources like database, API, specific file types. There is a protocol to implement as class inheritance.

```python
from agno.knowledge.protocol import KnowledgeProtocol
class MyKnowledge(KnowledgeProtocol):
    def __init__(self):
        self.documents: list[Document] = []

    def add(self, name: str, content: str) -> None:
        self.documents.append(Document(name=name, content=content))

    def _search(self, query: str, limit: int = 5) -> List[Document]:
        ...
     # --- Required protocol methods ---

    def build_context(self, **kwargs) -> str:
        return "Use the search tool to find information in the knowledge base."

    def get_tools(self, **kwargs) -> List[Callable]:
        return []

    async def aget_tools(self, **kwargs) -> List[Callable]:
        return []

    # --- Optional: enables search_knowledge feature ---

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        max_results = kwargs.get("max_results", 5)
        return self._search(query, limit=max_results)

    async def aretrieve(self, query: str, **kwargs) -> List[Document]:
        return self.retrieve(query, **kwargs)
```

## Skills

[Agno Skills](https://docs.agno.com/skills/overview) are self-contained, modular packages of domain expertise. Instead of bloating an AI agent’s core prompt with endless instructions, skills allow an agent to discover, load, and execute specialized capabilities on demand.


* Every skill is typically organized in its own directory, packaging everything the agent needs in one place:
    * Instructions (SKILL.md): The "brain" of the skill. It provides detailed guidance on when the skill should be triggered and exactly how the agent should apply it.
    * Scripts: Optional executable code templates (e.g., Python scripts) that the agent can run to perform tasks automatically.
    * References: Supporting documentation, cheat sheets, or examples.

* `skills` is specified as part of the agent creation, and will lead to modified system prompt
    ```python
    a = Agent(
                name=name,
                model=model,
                skills=Skills(loaders=[LocalSkills(str(skill_dir), validate=False)]),
                tools=agent_tools,
                instructions=instructions,
                markdown=True,
            )
    ```

    Example of modified prompt:
    ```
    ...
         - Call get_skill_instructions('ksql-to-flink') before translating.  
    ...
    <skills_system>                                          
      ## What are Skills?                                    
      Skills are packages of domain expertise that extend your capabilities. Each skill contains:  
      - **Instructions**: Detailed guidance on when and how to apply the skill
      - **Scripts**: Executable code templates you can use or adapt
      - **References**: Supporting documentation (guides, cheatsheets, examples)                                               
      ## IMPORTANT: How to Use Skills                        
      **Skill names are NOT callable functions.** You cannot call a skill directly by its name. Instead, you MUST use the provided skill access tools: 
      1. `get_skill_instructions(skill_name)` - Load the full instructions for a skill
      2. `get_skill_reference(skill_name, reference_path)` - Access specific documentation
      3. `get_skill_script(skill_name, script_path, execute=False)` - Read or run scripts

      ## Progressive Discovery Workflow
      1. **Browse**: Review the skill summaries below to understand what's available
      2. **Load**: When a task matches a skill, call `get_skill_instructions(skill_name)` first
      3. **Reference**: Use `get_skill_reference` to access specific documentation as needed
      4. **Scripts**: Use `get_skill_script` to read or execute scripts from a skill

      **IMPORTANT**: References are documentation files (NOT executable). Only use `get_skill_script` when `<scripts>` lists actual script files. If `<scripts>none</scripts>`, do NOT call `get_skill_script`.
      This approach ensures you only load detailed instructions when actually needed.
      
      ## Available Skills
      <skill>
      <name>ksql-to-flink</name>
      <description>Translates Confluent ksqlDB SQL scripts to Apache Flink SQL with proper streaming semantics. Use when converting ksqlDB to Flink, migrating CREATE STREAM scripts, or when the useR asks to migrate ksql to Flink SQL.</description>
      <scripts>none<scripts>
      <references>confluent-sql-deploy.md, examples.md, flink-deploy-setup.md, translation-rules.md</references>                                                                                       
      </skill>
      </skills_system> 
    ```


* Instead of forcing the AI model to hold all instructions in its context window (which wastes tokens and can cause confusion), Agno uses lazy loading. The agent's core system prompt only contains a lightweight summary of the skill.When you prompt the agent with a task that matches the skill, it automatically discovers the skill, pulls the full instructions, and accesses the required references to complete the request.
* Because of this modular design, you can easily swap out the underlying language model without having to rewrite the skill itself. The agent focuses on orchestrating, reasoning and planning.
* If you are building complex multi-agent systems, skills can be assigned directly to a Team. The team leader acts as the coordinator and can discover, load, and execute skills to evaluate standards or oversee tasks, without delegating the work to individual member agents.