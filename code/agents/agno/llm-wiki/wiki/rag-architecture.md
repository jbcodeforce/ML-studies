# RAG Architecture


**Summary**: An architectural pattern that enhances LLMs with retrieval capabilities from external document stores and vector databases.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/sol-design.md


**Last updated**: 2026-04-18

---


A [[rag-architecture]] (Retrieval-Augmented Generation) solution extends a standard chatbot architecture by introducing document and vector store management (source: sol-design.md).


### Core Components


- **Document Management**: Systems for managing and ingesting the unstructured or semi-structured data that will be used as context.


- **Vector Store Management**: The use of vector databases to store and retrieve embeddings, enabling efficient similarity searches for retrieving relevant document chunks.


- **Contextual Retrieval**: The process of retrieving relevant information from the managed documents to augment the LLM's prompt, providing more accurate and up-to-date responses.


### Use Cases and Risks


- **Internal Enterprise Use**: Highly effective for providing staff with access to internal knowledge bases.


- **B2B/B2C Considerations**: While useful for B2B, exposing LLMs directly to consumers in regulated businesses is considered high-risk (as of mid-2024) (source: sol-design.md).


## Related pages


- [[ai-solution-designs]]
- [[po-processing]]