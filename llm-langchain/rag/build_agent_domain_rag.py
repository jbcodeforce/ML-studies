"""
The presence of mixed-domain data in the same vector store can introduce noise and potentially degrade performance.
Isolating vector stores for each domain can help maintain domain-specific information and improve model accuracy within individual domains

This code demonstrate creating a vector store from a blog and keep content related to Agent in the same vector
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma.Chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

URL_ARTICLE="https://lilianweng.github.io/posts/2023-06-23-agent/"
DOMAIN_VS_PATH="./agent_domain_store"

def load_multi_agent_blog():
    """
    Load a blog from the internet, parse it with beautiful soup taking into account only the post content and headers
    return the documents
    """
    loader = WebBaseLoader(
            web_paths=(URL_ARTICLE,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
    return  loader.load()
    
def build_indexing(docs): 
    """
    Split the documents in input, with a specific chunk size. The overlap size is important to try to keep
    related subjects together. Perform the indexing of the split by doing a vector creation via embedding.
    Persist to a vector store so it can be reused by real time inference
    """ 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    print(f" The number of document splits: {len(splits)}")
    vectorstore =  Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),
                                         persist_directory=DOMAIN_VS_PATH)
    return vectorstore

if __name__ == "__main__":
    print("---> Welcome to build indexing on Multi Agent knowledge")
    load_dotenv("../../.env")
    print(f"---> 1/ load: {URL_ARTICLE}")
    docs=load_multi_agent_blog()
    print(f"---> The length of the page content: {len(docs[0].page_content)}")
    print(f"---> 2/ build and persist vector store in: {DOMAIN_VS_PATH}")
    vectorstore = build_indexing(docs) 
    print(f"---> 3 Test semantic search on this vector store with the following \nquestion: What are the approaches to Task Decomposition?")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")
    print(f"\t Number of documents retrieved {len(retrieved_docs)}")
    print(f"\n\nResults: {retrieved_docs[0].page_content}")

