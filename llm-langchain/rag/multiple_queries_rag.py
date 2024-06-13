import os,sys, json
from dotenv import load_dotenv
from operator import itemgetter
import langchain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma

DOMAIN_VS_PATH="./agent_domain_store"

langchain.debug = True

def define_multi_query():
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    return ChatPromptTemplate.from_template(template)

def build_multi_query_chain():
    prompt_perspectives = define_multi_query()
    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    vectorstore=Chroma(persist_directory=DOMAIN_VS_PATH,embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain

def get_unique_union(documents: list[list]):

    flat_docs = [doc.json() for sublist in documents for doc in sublist]
    unique_docs=  list(set(flat_docs)) 
    return [json.loads(doc) for doc in unique_docs]
    
def build_rag_chain(retrieval_chain):
    template = """Using as much content for the given context, answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    return (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    print("---> Welcome to demo for multi query transformation")
    if not os.path.isdir(DOMAIN_VS_PATH):
        print("You need to run build_agent_domain_rag.py before")
        sys.exit(1)
    load_dotenv("../../.env")
    print("\t 1/ Build chain")
    retrieval_chain=build_multi_query_chain()
    final_rag_chain = build_rag_chain(retrieval_chain)
    question = "What is task decomposition for LLM agents?"
    resp=final_rag_chain.invoke({"question":question})
    print(resp)