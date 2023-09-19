# Some LLM and LangChain implementations

Use python docker image with the script `./startPythonDocker.sh`, it includes not released yet AWS SDK.

note "invalid request":

    In any code connected to AWS, if there is this kind of message: ` when calling the InvokeModel operation: The security token included in the request is invalid`, be sure to get the AWS_SESSION_TOKEN environment variable set.

## Getting started

See code of [langchain-1.py](./langchain-1.py) to get basic of creating a AWS Bedrock client and use AWS Titan LLM in different prompts.

The `utils` folder includes the code to interact with AWS Bedrock using last release of boto3.

## FeatureForm examples

It uses FeatureForm tutorial to load the transactions data into online and offline storage. The demonstration is based on FeatureForm tutorial and LangChain prompt with feature store code.

![](../docs/data/diagrams/featureform-llm.drawio.png)

To run it:

* Start the different dockers container with `docker-compose -f featureform-docker-compose.yaml`
* Then connect to the mypython env: `docker exec -ti mypython bash`
* `pip install featureform` if not done before.
* `export FEATUREFORM_HOST=featureform:7878`
* Define the feature store for the Transactions entity: 

    ```sh
    featureform apply ./ff-definitions.py --insecure
    ```

    The output looks like

    ```sh
     Resource Type              Name (Variant)                                      Status      Error 
    Provider                   local-mode ()                                       CREATED           
    Provider                   postgres-quickstart ()                              CREATED           
    Provider                   redis-quickstart ()                                 CREATED           
    SourceVariant              average_user_transaction (default)                  READY             
    SourceVariant              transactions (default)                              READY             
    FeatureVariant             avg_transactions (default)                          READY             
    LabelVariant               fraudulent (default)                                READY     
    ```

* Look at the dashboard about the definition: http://localhost:8082

![](./images/featureform-ui.png)

* [Optional:] We can serve a training dataset from Postgresql using: `python ff-training.py`, or get data from the on-line store with `python ff-serving.py`.
* Run a prompt to Bedrock with Claude LLM and feature retreived from FeatureForm.

```sh
python ff-langchain-prompt.py

#output

Given your average amount per transaction of $5000, I would definitely consider you a high roller!
No chicken jokes for you. 😎
```

## A RAG pipeline for Q&A

The code is based on the content from [this article](https://python.langchain.com/docs/use_cases/question_answering/), and the code is [qa-pipeline.py](./qa-pipeline.py).

* Install chromadb with: `pip install chromadb`
* Get a client to Bedrock using boto3.
* Use [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.web_base.WebBaseLoader.html) to load a website URL
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