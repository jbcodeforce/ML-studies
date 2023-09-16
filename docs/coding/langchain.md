# LangChain notes

[LangChain](https://python.langchain.com/docs/get_started/introduction) is a framework for developing applications powered by language models.

The core building block of LangChain applications is the LLMChain:

* A LLM
* Prompt templates
* Output parsers

The standard interface that LangChain provides has two methods:

* `predict`: Takes in a string, returns a string
* `predict_messages`: Takes in a list of messages, returns a message.

The [following code (llm/langchain-1.py)](https://github.com/jbcodeforce/ML-studies/tree/master/llm/langchain-1.py) regroups the getting started examples from LangChain docs.

Modules are extendable interfaces to Langchain.

## Model I/O

* building blocks to interface with any language model.
* A **prompt** for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output. See [Prompt template](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/).
* Feature stores, like [Feast](https://github.com/feast-dev/feast), can be a great way to keep information abount the user fresh, and LangChain provides an easy way to combine that data with LLMs. [This is an example of LLM Chain with AWS Bedrock titan llm and Feast as feature store](https://github.com/jbcodeforce/ML-studies/tree/master/llm/feast-prompt.py)

???- info "Getting started with Feast"
    Use `pip install feast` then the `feast` CLI with `feast init my_feature_repo` then `feast apply` to create entity, feasture views, and services. Then `feast ui` + [http://localhost:8888](http://localhost:8888) to act on the store. See [my summary on Feast](../../data/features/#feast-open-source)


## Chains

Chains allow us to combine multiple components together to create a single, coherent application. 
[LLMChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html)

