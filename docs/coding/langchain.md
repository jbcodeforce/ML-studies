# LangChain notes

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing applications powered by language models, connecting them to external data sources.

The core building block of LangChain applications is the LLMChain:

* A LLM
* Prompt templates
* Output parsers

Below is [an example](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/feast-prompt.py) of getting a LLM api, build a prompt, a chain and call it.

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

The [following code (llm/langchain-1.py)](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/langchain-1.py) regroups the getting started examples from LangChain docs to illustrate calls to LLM.

Modules are extendable interfaces to Langchain.

## Use cases

* **Q&A**: ask questions on a knowledge corpus, LLM helps understanding the text and the questions.

    ![](./diagrams/qa-llm.drawio.png)

    The pipeline to build the Q&A over existing document is illustrated in the figure below:

    ![](./diagrams/lg-pipeline.drawio.png)

* **Chatbots**: Aside from basic prompting and LLMs, memory and retrieval are the core components of a chatbot. ChatModels do not need LLM, as they are conversational. 

    ![](./diagrams/chatbot.drawio.png)

## Model I/O

* Model I/O are building blocks to interface with any language model.
* A **prompt** for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output. See the[Prompt template](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/).
* Two prompt templates: [string prompt](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.base.StringPromptTemplate.html) templates and [chat prompt](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.chat.ChatPromptTemplate.html) templates.
* We can build custom prompt by extending existing default templates. An example is a 'few-shot-examples' in a chat prompt usine [FewShotChatMessagePromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples_chat). 
* Feature stores, like [Feast](https://github.com/feast-dev/feast), can be a great way to keep information abount the user fresh, and LangChain provides an easy way to combine that data with LLMs. [This is an example of LLM Chain with AWS Bedrock Titan llm and Feast as feature store](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/feast-prompt.py)

???- info "Getting started with Feast"
    Use `pip install feast` then the `feast` CLI with `feast init my_feature_repo` to create a Feature Store then `feast apply` to create entity, feature views, and services. Then `feast ui` + [http://localhost:8888](http://localhost:8888) to act on the store. See [my summary on Feast](../../data/features/#feast-open-source)

???- info "LLM and FeatureForm"
    See [FeatureForm](https://docs.featureform.com/) as another open-source feature store solution and the LangChain sample with [Claude LLM](https://github.com/jbcodeforce/ML-studies/tree/master/llm-langchain/ff-langchain-prompt.py)

## Retrieval

Get custom dataset not already part of a model training set. The goal is to retrieve the data and pass it to LLM in the generation step. This is the Retrieval Augmented Generation or RAG and illustrated in figure below:

![](./diagrams/rag-process.drawio.png)

* Embeddings capture the semantic meaning of the text to help do similarity search
* Persist the embeddings into a vector store. ChromaDB is common, but OpenSearch can also being used.
* Retriever includes semantic search and efficient algorithm to prepare the prompt. To improve on vector similarity search we can generate variants of the input question.

## Chains

Chains allow us to combine multiple components together to create a single, coherent application. 
[LLMChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html)

ConversationChain 

##  Deeper dive

* [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
* [Retrieval and RAG blog.](https://blog.langchain.dev/retrieval/)