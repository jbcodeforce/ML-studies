"""
A simple tool to interact with LLM and RAG with a specific knowledge based
"""
import os, sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


DOMAIN_VS_PATH="./agent_domain_store"
"""
Query the domain store, with a text chat
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def buildRetrieverChain():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    prompt = hub.pull("rlm/rag-prompt")
    print(f"---> {prompt}")
    vectorstore=Chroma(persist_directory=DOMAIN_VS_PATH,embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    chain =  (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

"""
Stream the output as it arrives. but in a sync context
"""
def run_continue_chat(chain): 
    print("\t 2/Starting iterative question and answer\n\n")
    finish=False
    chat_history = []
    while not finish:
        question=input("Ask questions (q to quit): ")
        if question=='q' or question=='quit' or question=='Q':
            finish=True
        else:
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n\n")
    print("Thank you , that was a nice chat !!")

if __name__ == "__main__":
    print("---> Welcome to query the Multi Agent knowledge based")
    if not os.path.isdir(DOMAIN_VS_PATH):
        print("You need to run build_agent_domain_rag.py before")
        sys.exit(1)
    load_dotenv("../../.env")
    print("\t 1/ Build chain")
    chain = buildRetrieverChain()
    run_continue_chat(chain)