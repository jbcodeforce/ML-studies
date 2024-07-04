"""
The graph is doing query analysis to route the query to no-retrieval, single-shot RAG or iterative RAG
"""
import os
from dotenv import load_dotenv 
from typing import List, Literal
from typing_extensions import TypedDict
from pprint import pprint
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, START, END

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv("../../.env")
DOMAIN_VS_PATH="./agent_domain_store"
LLM_MODEL="gpt-3.5-turbo-0125"
_embd = OpenAIEmbeddings()
_vs = None

web_search_tool = TavilySearchResults(k=3)

def build_rag_content():
    global _embd
    print("---- BUILD CORPUS CONTENT -----")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list= [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(doc_list)
    _vs = Chroma.from_documents(persist_directory=DOMAIN_VS_PATH, documents=doc_splits, collection_name="agentic_corpus", embedding=_embd)


def retrieve(state):
    global _vs
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = _vs.as_retriever().invoke(question)
    return {"documents": documents, "question": question}

# ===========================Tool to search the web ======================

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}


#================== Define Router Agent ===================
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

_router_agent = None
def get_or_build_router_agent():
    global _router_agent
    if not _router_agent:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Prompt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        _router_agent = route_prompt | structured_llm_router
    return _router_agent


# =================== Define Generation Agent =======================
_generation_agent =None
def get_or_build_generation_agent():
    global _generation_agent
    if not _generation_agent:
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        _generation_agent= prompt | llm | StrOutputParser()
    return _generation_agent

def generate(state):
    # Generate answer 
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = get_or_build_generation_agent().invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# =================================== Graph elements =================
class AdaptiveRagState(TypedDict):
    question: str
    generation: str
    documents: List[str]


def route_question(state):
    print("---ROUTE QUESTION---")
    q = state["question"]
    source = get_or_build_router_agent().invoke({"question": q})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


    
def define_graph():
    """Assess to use retriever or web search"""
    workflow = StateGraph(AdaptiveRagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()



def process_unrelated_question(app):
    inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            pprint(value["keys"], indent=2, width=80, depth=None)

    # Final generation
    pprint(value["generation"])
    
if __name__ == "__main__":
    if not os.path.isdir(DOMAIN_VS_PATH):
        build_rag_content()
    else:
        _vs=Chroma(persist_directory=DOMAIN_VS_PATH,embedding_function=_embd)
    app=define_graph()
    process_unrelated_question(app)