# Some LLM and LangChain implementations

Use python docker image with the script `./startPythonDocker.sh`

note "invalid request":

    In any code connected to AWS, if there is this kind of message: ` when calling the InvokeModel operation: The security token included in the request is invalid`, be sure to get the AWS_SESSION_TOKEN environment variable set.

## Pre-requisite

* Install lib for Bedrock and AWS: 

    ```sh
    cd bedrock
    pip install -r requirements
    ```

* Install langchain and other dependencies

    ```sh
    cd ..
    pip install -r requirements
    ```

## A RAG pipeline for Q&A

The code is based on the content from [this article](https://python.langchain.com/docs/use_cases/question_answering/), and the code is [qa-pipeline.py](./qa-pipeline.py).

* Install chromadb, as vector store, with: `pip install chromadb`
* Get a client to Bedrock using boto3.
* Use [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.web_base.WebBaseLoader.html) to load content form a website giving its URL
* Split the document in smaller chunks
* Define Bedrock embeddings and use it to encode chunks to a vector store like chromaDB
* Build a prompt
* Use LLM with retriever in a question and answer chain

```python
qa_chain = RetrievalQA.from_chain_type(
    claude_llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
result=qa_chain({"query": question})
```

## RAG on markdown files and opensearch

Same approach but it uses OpenSearch as vector store, and markdown files as source. See [this code](./qa-chat-md-os.py) which uses a chatbot interface.