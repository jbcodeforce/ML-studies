# LLM with core API like Ollama or OpenAI API

This folder includes different examples using pure lower level APIs without langchain or langgraph.

## Chat with ollama API

* Chat with ollama mistral model.
    ```sh
    uv run chat_with_mistral.py          

    ```


## Asynch chat with ollama

## Chat with a llm running within Ollama using OpenAI SDK

* mistral model by default
    ```sh
    uv run chat_with_ollama_openai_api.py
    ```
* other model from `ollama list`
    ```sh 
    uv run chat_with_ollama_openai_api.py --model gpt-oss:20b 
    ```


## Combine with vector store for RAG
