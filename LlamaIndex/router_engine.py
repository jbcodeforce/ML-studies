"""
An agentic approach of RAG using tool callings
Load a pdf document, split it using sentence splitter 
"""
import sys
from dotenv import load_dotenv
# import nest_asyncio
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv("../.env")

def load_docs(fn: str):
    docs = SimpleDirectoryReader(input_files=[fn]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(docs)
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    print("--- Build two indexes one for the summary, one for the vector (for semantic query based on similarity)")
    summary_index =SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)
    return summary_index,vector_index

def get_router_engine(summary_index,vector_index):
    # Build query engines
    summary_query_engine = summary_index.as_query_engine( response_mode="tree_summarize", use_async=True)
    vector_query_engine = vector_index.as_query_engine()


    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    query_engine= RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[ summary_tool, vector_tool],
        verbose=True
    )
    return query_engine

if __name__ == "__main__":
    filename=sys.argv[1]
    print(f"--> Build indexes from content of {filename}")
    summary_index,vector_index = load_docs(filename)
    engine=get_router_engine(summary_index,vector_index)
    response = engine.query("What is the summary of the document?")
    print(str(response))
    print(len(response.source_nodes))
    response = engine.query(
        "How do agents share information with other agents?"
    )
    print(str(response))    