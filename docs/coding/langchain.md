# LangChain notes

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing applications powered by language models, connecting them to external data sources.

The core building block of LangChain applications is the LLMChain:

* A LLM
* Prompt templates
* Output parsers

Below is [an example](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/feast-prompt.py) of getting a LLM api using AWS Bedrock service, build a prompt using Feast feature stores, a chain using Langchain and call it with the value of one of the driver_id.

```python
titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=bedrock_client)
prompt_template = FeastPromptTemplate(input_variables=["driver_id"])

chain = LLMChain(llm=titan_llm, prompt=prompt_template)
# run has positional arguments or keyword arguments, 
print(chain.run(1001))
```

The standard interface that LangChain provides has two methods:

* `predict`: Takes in a string, returns a string
* `predict_messages`: Takes in a list of messages, returns a message.

Modules are extendable interfaces to Langchain.

## Use cases

* **Q&A**: ask questions on a knowledge corpus, LLM helps understanding the text and the questions.

    ![](./diagrams/qa-llm.drawio.png)

    The pipeline to build the Q&A over existing document is illustrated in the figure below:

    ![](./diagrams/lg-pipeline.drawio.png)

* **Chatbots**: Aside from basic prompting and LLMs, memory and retrieval are the core components of a chatbot. ChatModels do not need LLM, as they are conversational. 

    ![](./diagrams/chatbot.drawio.png)

* **Code Understanding**
* Extraction
* Summarization: summarize call transcripts, meetings transcripts, books, articles, blog posts, and other relevant content.
* **[Web scraping](https://python.langchain.com/docs/use_cases/web_scraping)** for LLM based web research. It uses the same process: document/page loading, transformation with tool like BeautifulSoup, to HTML2Text.

## Model I/O

* Model I/O are building blocks to interface with any language model. It facilitates the interface of model input (prompts) with the LLM model to produce the model output.
* A **prompt** for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output. See the[Prompt template](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/).
* Two prompt templates: [string prompt](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.base.StringPromptTemplate.html) templates and [chat prompt](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html) templates.
* We can build custom prompt by extending existing default templates. An example is a 'few-shot-examples' in a chat prompt usine [FewShotChatMessagePromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples_chat).
* Feature stores, like [Feast](https://github.com/feast-dev/feast), can be a great way to keep information abount the user fresh, and LangChain provides an easy way to combine that data with LLMs.

### Examples

Getting started with langchain, the [following code (llm/langchain-1.py)](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langchain-1.py) regroups the getting started examples from LangChain docs to illustrate calls to LLM.

 [This is an example of LLM Chain with AWS Bedrock Titan llm and Feast as feature store](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/feast-prompt.py)





???- info "Getting started with Feast"
    Use `pip install feast` then the `feast` CLI with `feast init my_feature_repo` to create a Feature Store then `feast apply` to create entity, feature views, and services. Then `feast ui` + [http://localhost:8888](http://localhost:8888) to act on the store. See [my summary on Feast](../../data/features/#feast-open-source)

???- info "LLM and FeatureForm"
    See [FeatureForm](https://docs.featureform.com/) as another open-source feature store solution and the LangChain sample with [Claude LLM](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/ff-langchain-prompt.py)

## Retrieval Augmented Generation

The goal is to add custom dataset not already part of a model training set and use it as input to the LLM. This is the Retrieval Augmented Generation or RAG and illustrated in figure below:

![](./diagrams/rag-process.drawio.png)

The process is to get data from the different sources, load, cut into smaller pieces, extract what is necessary, transform the sentences into numerical vectors. Creating chunks is necessary because language models generally have a limit to the amount of text they can deal with.
During the interaction with the end-user, the system (a chain in LangChain) retrieves the data most relevant to the question asked, and passes it to LLM in the generation step.

* Embeddings capture the semantic meaning of the text to help do similarity search
* Persist the embeddings into a Vector store. ChromaDB is common, but OpenSearch can also being used.
* Retriever includes semantic search and efficient algorithm to prepare the prompt. To improve on vector similarity search we can generate variants of the input question.

Combine chat history with new question to ask follow up questions.

## Chains

Chains allow us to combine multiple components together to create a single, coherent application, and also  combine chains.

[LLMChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html) is the basic chain to integrate with a LLM.

ConversationChain 

### Summarization chain

Always assess the size of the content to send, as the approach can be different: for big doc we need to split those doc.

* Small text to summarize, with [bedrock client](https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/utils/bedrock.py) and use the invoke_model on the clientm see the code in [https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/small-summarization.py](small-summarization.py)
* For big document, langchain provides the load_summarize_chain to summarize by chunks and get the summary of the summaries:

???- code "Using langchain summarize chain"
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.llms.bedrock import Bedrock
    from langchain.chains.summarize import load_summarize_chain

    llm = Bedrock(
        model_id=modelId,
        model_kwargs={
            "max_tokens_to_sample": 1000,

        },
        client=boto3_bedrock,
    ) 

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
    )
    docs = text_splitter.create_documents([letter])

    summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)
    output = summary_chain.run(docs)
    ```

### Q&A chain

## Deeper dive

* [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
* [Retrieval and RAG blog.](https://blog.langchain.dev/retrieval/)


    ???- "LangChain code"
        ```python
        from utils import bedrock, print_ww
        from langchain.llms.bedrock import Bedrock

        inference_modifier = {
            "max_tokens_to_sample": 4096,
            "temperature": 0.5,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"],
        }

        textgen_llm = Bedrock(
            model_id="anthropic.claude-v1",
            client=boto3_bedrock,
            model_kwargs=inference_modifier,
        )
        response = textgen_llm("""Write an email from Bob, Customer Service Manager, to the customer "John Doe" that provided negative feedback on the service provided by our customer support engineer.\n\nHuman:""")
        ```

    ???- code "Langchain with prompt template, and variables to add more context."
        ```python
        from langchain import PromptTemplate

        multi_var_prompt = PromptTemplate(
            input_variables=["customerServiceManager", "customerName", "feedbackFromCustomer"], 
            template="""Create an apology email from the Service Manager {customerServiceManager} to {customerName}. 
        in response to the following feedback that was received from the customer: {feedbackFromCustomer}.
        """
        )

        prompt = multi_var_prompt.format(customerServiceManager="Bob", 
                                        customerName="John Doe", 
                                        feedbackFromCustomer="""Hello Bob,
            I am very disappointed with the recent experience I had when I called your customer support.
            I was expecting an immediate call back but it took three days for us to get a call back.
            The first suggestion to fix the problem was incorrect. Ultimately the problem was fixed after three days.
            We are very unhappy with the response provided and may consider taking our business elsewhere.
            """
            )
        ```

    ???- code "Splitting long text and summarize with map-reduce"
        ```python
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains.summarize import load_summarize_chain
        from langchain.llms.bedrock import Bedrock

        llm = Bedrock(
            model_id="amazon.titan-tg1-large",
            model_kwargs={
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            },
            client=boto3_bedrock,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100
        )
        summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)
        docs = text_splitter.create_documents([letter])
        output = summary_chain.run(docs)
        ```

    ???- code "Build embeddings from a corpus"
        ```python
        from langchain.embeddings import BedrockEmbeddings
        from langchain.llms.bedrock import Bedrock
        import numpy as np
        from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
        from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

        llm = Bedrock(model_id="anthropic.claude-v1", client=boto3_bedrock, model_kwargs=                   {'max_tokens_to_sample':200})
        bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

        # prepare docs
        loader = PyPDFDirectoryLoader("./data/")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1000,
            chunk_overlap  = 100,
        )
        docs = text_splitter.split_documents(documents)
        # Use in-memory vectorDB
        from langchain.chains.question_answering import load_qa_chain
        from langchain.vectorstores import FAISS
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper

        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings,
        )

        wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
        ```

    ???- code "Search similarity in vector DB, and then query"
        See code above for import and vector RB preparatio 
        ```python
        query_embedding = vectorstore_faiss.embedding_function(query)
        relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)
        answer = wrapper_store_faiss.query(question=query, llm=llm)
        # OR with retrievalQA
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate

        prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Assistant:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        query = "Is it possible that I get sentenced to jail due to failure in filings?"
        result = qa({"query": query})
        print_ww(result['result'])
        ```
    
    ???- code "Chatbot with LangChain"
        ```python
        from langchain.chains import ConversationChain
        from langchain.llms.bedrock import Bedrock
        from langchain.memory import ConversationBufferMemory

        titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=boto3_bedrock)
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=titan_llm, verbose=True, memory=memory
        )

        print_ww(conversation.predict(input="Hi there!"))
        ```


