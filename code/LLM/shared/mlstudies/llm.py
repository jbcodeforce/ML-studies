"""LLM client factory functions."""

from typing import Any
from . import config


def get_openai_chat(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    **kwargs,
) -> Any:
    """Get an OpenAI chat model.
    
    Args:
        model: Model name (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
        temperature: Sampling temperature
        **kwargs: Additional arguments passed to ChatOpenAI
        
    Returns:
        ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_anthropic_chat(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    **kwargs,
) -> Any:
    """Get an Anthropic chat model.
    
    Args:
        model: Model name (claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.)
        temperature: Sampling temperature
        **kwargs: Additional arguments passed to ChatAnthropic
        
    Returns:
        ChatAnthropic instance
    """
    from langchain_anthropic import ChatAnthropic
    
    return ChatAnthropic(
        api_key=config.ANTHROPIC_API_KEY,
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_mistral_chat(
    model: str = "mistral-large-latest",
    temperature: float = 0.0,
    **kwargs,
) -> Any:
    """Get a Mistral chat model.
    
    Args:
        model: Model name
        temperature: Sampling temperature
        **kwargs: Additional arguments passed to ChatMistralAI
        
    Returns:
        ChatMistralAI instance
    """
    from langchain_mistralai import ChatMistralAI
    
    return ChatMistralAI(
        api_key=config.MISTRAL_API_KEY,
        model=model,
        temperature=temperature,
        **kwargs,
    )


def get_openai_embeddings(
    model: str = "text-embedding-3-small",
    **kwargs,
) -> Any:
    """Get OpenAI embeddings model.
    
    Args:
        model: Embedding model name
        **kwargs: Additional arguments
        
    Returns:
        OpenAIEmbeddings instance
    """
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY,
        model=model,
        **kwargs,
    )


def get_watsonx_llm(
    model_id: str = "meta-llama/llama-3-8b-instruct",
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    **kwargs,
) -> Any:
    """Get an IBM WatsonX LLM.
    
    Args:
        model_id: Model identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional arguments
        
    Returns:
        WatsonxLLM instance
    """
    from langchain_ibm import WatsonxLLM
    
    parameters = {
        "decoding_method": "sample",
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 1,
    }
    
    return WatsonxLLM(
        model_id=model_id,
        url=config.IBM_WATSONX_URL,
        project_id=config.IBM_WATSON_PROJECT_ID,
        params=parameters,
        **kwargs,
    )

