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

# =============================== RAG preparation ============================= 
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



# =============================== Grader of the retrieved documents  ======
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

_retrieval_grader= None
def get_or_build_retrieval_grader():
    global _retrieval_grader
    if not _retrieval_grader:
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        _retrieval_grader = grade_prompt | structured_llm_grader
    return _retrieval_grader

# ========================================= Answer Grader ============================
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    
_answer_grader=None
def get_or_build_answer_grader():
    global _answer_grader 
    if not _answer_grader:
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        answer_prompt  = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                    ),
                ]
            )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeAnswer)
        _answer_grader = answer_prompt  | structured_llm_grader | StrOutputParser()
    return _answer_grader


_question_rewriter= None
def get_or_build_question_rewriter():
    global _question_rewriter 
    if not _question_rewriter:
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                    ),
                ]
            )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        _question_rewriter = re_write_prompt | llm | StrOutputParser()
    return _question_rewriter



# ============================ Hallucination Grader =================
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

_hallucination_grader= None
def get_or_build_hallucination_grader():
    global _retrieval_grader
    if not _retrieval_grader:
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.."""
        hallucination_prompt  = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        _hallucination_grader = hallucination_prompt  | structured_llm_grader  
    return _hallucination_grader

    
# =================================== Graph elements =================
class AdaptiveRagState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def generate(state):
    # Generate answer 
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = get_or_build_generation_agent().invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def retrieve(state):
    global _vs
    print("---RETRIEVE---")
    question = state["question"]
    documents = _vs.as_retriever().invoke(question)
    return {"documents": documents, "question": question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

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

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = get_or_build_retrieval_grader().invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score # type: ignore
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = get_or_build_question_rewriter().invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = get_or_build_hallucination_grader().invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = get_or_build_answer_grader().invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
    
    
def define_graph():
    """Assess to use retriever or web search"""
    workflow = StateGraph(AdaptiveRagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query) 
    
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
   
    return workflow.compile()


# ================================= test functions ================================= 
def process_unrelated_question(app):
    inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            #pprint(value["keys"], indent=2, width=80, depth=None)

    # Final generation
    pprint(value["generation"])

def process_corpus_related_question(app):
    inputs = {
    "question": "how does agent memory work?"
    }
    print(f"\n @@@> {inputs}")
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            #pprint(value["keys"], indent=2, width=80, depth=None)

    # Final generation
    pprint(value["generation"])
    
     
if __name__ == "__main__":
    if not os.path.isdir(DOMAIN_VS_PATH):
        build_rag_content()
    else:
        _vs=Chroma(persist_directory=DOMAIN_VS_PATH,embedding_function=_embd)
    print(_vs)
    app=define_graph()
    #process_unrelated_question(app)
    process_corpus_related_question(app)