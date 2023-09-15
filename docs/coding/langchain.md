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


## Chains

Chains allow us to combine multiple components together to create a single, coherent application.

