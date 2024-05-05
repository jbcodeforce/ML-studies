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

def define_HyDE(question: str):
    template = """
    Please write a scientific paper passage to answer the question

    Question: {question}
    Passage:"""
    prompt= ChatPromptTemplate.from_template(template)
    generate_docs_for_retrieval = (
        prompt 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
    )
   
    return generate_docs_for_retrieval

def retrieve_docs(generate_docs_for_retrieval, question):
    vectorstore=Chroma(persist_directory=DOMAIN_VS_PATH,embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    retrieval_chain = generate_docs_for_retrieval | retriever 
    return retrieval_chain.invoke({"question": question})


def build_rag_chain():
    template = """Using as much content for the given context, answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    return (
        prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    print("---> Welcome to demo for Rag fusion using 4 queries")
    if not os.path.isdir(DOMAIN_VS_PATH):
        print("You need to run build_agent_domain_rag.py before")
        sys.exit(1)
    load_dotenv("../../.env")
    print("\t 1/ Build a HyDE")
    question = "What is task decomposition for LLM agents?"
    generate_docs_for_retrieval = define_HyDE(question)
    retrieved_docs=retrieve_docs(generate_docs_for_retrieval, question)
    final_rag_chain = build_rag_chain()
    resp=final_rag_chain.invoke({"context": retrieved_docs, "question":question})
    print(resp)