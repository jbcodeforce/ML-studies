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
    template = """
    You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
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
    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    return retrieval_chain

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = doc.json()
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (json.loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return a list of tuples, each containing the document and its fused score
    return reranked_results
    
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
    print("---> Welcome to demo for Rag fusion using 4 queries")
    if not os.path.isdir(DOMAIN_VS_PATH):
        print("You need to run build_agent_domain_rag.py before")
        sys.exit(1)
    load_dotenv("../../.env")
    print("\t 1/ Build chain")
    retrieval_chain=build_multi_query_chain()
    final_rag_chain = build_rag_chain(retrieval_chain)
    question = "What are the main components of an LLM-powered autonomous agent system?"
    resp=final_rag_chain.invoke({"question":question})
    print(resp)