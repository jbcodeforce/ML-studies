# LangChain Study Summary

**Source:** `raw/studies/coding/langchain.md` | **Ingested:** 2026-06-19

LangChain is an open-source framework for building applications powered by large language models. It addresses the "glue code" problem in LLM development — connecting LLMs to external data, managing conversation history, and composing multi-step workflows.

## Key Points

- **Modular Architecture**: LangChain emphasizes composition via composable building blocks — prompt templates, output parsers, chains, and tools — supporting Python, TypeScript, and Java.
- **Core Products**: LangChain libraries, LangServe (REST API deployment), and LangSmith (debugging, testing, monitoring).
- **Model I/O**: Supports both LLMs (text completion) and ChatModels (conversation with AIMessage/HumanMessage). Uses prompt templates (string and chat) and a Prompt Hub for reusable templates.
- **Chains & LCEL**: Chains combine components into coherent workflows. LCEL (`prompt | model | output_parser`) is a declarative, Unix-pipe-style syntax supporting streaming, async, parallelism, retries, and fallbacks.
- **Memory**: LLMs are stateless; LangChain provides memory abstractions (ConversationBuffer, WindowMemory, TokenBuffer, Summary, VectorStore, KnowledgeGraph) to simulate conversation context.
- **RAG**: Retrieval-Augmented Generation augments LLMs with external data via embeddings, text splitting, and vector stores (FAISS, ChromaDB, OpenSearch). Includes semantic search, HyDE, and chunking strategies.
- **Agents**: Unlike chains (developer-defined sequence), agents let the LLM decide tool usage in a loop. Tool calling allows structured function invocation. LangGraph (separate study) is recommended over deprecated AgentExecutor.
- **Tool Calling**: Models can detect and invoke tools with structured arguments. Tool definitions can be embedded in prompts, dynamically selected via embeddings, or loaded from LangChain's predefined tool catalog.
- **Q&A and Chatbots**: Standard pipelines combine retrievers, vector stores, and LLMs. Chatbots add memory and retrievers on top.
- **Evaluations**: Use QAGenerateChain to build question-answer pairs for evaluation.
- **Code Examples**: The study links numerous code samples across OpenAI, Ollama, Anthropic, Mistral, WatsonX, and AWS Bedrock backends.

## Connections

- Related to **Haystack** (another LLM app framework, already compiled).
- Introduces concepts of **RAG**, **Agents**, **Tool Calling**, **LCEL**, and **LangChain Memory**.
- Connects to **LangGraph** (uncompiled companion study) for agent orchestration.