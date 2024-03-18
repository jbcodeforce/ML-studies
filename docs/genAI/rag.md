# Retrieval augmented generation (RAG)

RAG is the act of supplementing generative text models with data outside of what it was trained on. This is applied to businesses who want to include proprietary information which was not previously used in a foundation model training set but does have the ability to search. Technical documentation which is not public is a good example of the usage of RAG.

The following diagram illustrates a classical RAG process using AWS SageMaker and OpenSearch.

![](./diagrams/rag.drawio.png)

And a classical RAG with LangChain:

![](../coding/diagrams/rag-process.drawio.png)

RAG produces great quality result, due to augmenting use-case specific context coming directly from vectorized information stores. It has the highest degree of flexibility when it comes to changes in the architecture. We can change the embedding model, vector store and LLM independently with minimal to moderate impact on other components.

Training from scratch produces the highest quality result amongst Prompt, RAG, fine tuning, but cost far more and need deep data science skill set.

[See hands-on with LangChain](../coding/langchain.md/#retrieval-augmented-generation).