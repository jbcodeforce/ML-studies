from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain
import os

def load_documents_from_blog(embeddings):
    print("--- Load source document from langchain.com/user_guide web site")
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    print("\tSplit the documents into chunks")
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(docs)
    print("\tCreate embeddings and save them in Chromadb vector stores")
    vectorstore=Chroma.from_documents(documents=all_splits, 
                                        embedding=embeddings,
                                        persist_directory=CHROMA_DB_FOLDER)

    return vectorstore

"""
Add Retriever to get better data, by crawling a LangChain product documentation from the web using BeautifulSoup
then Chroma vector store
"""
load_dotenv(dotenv_path="../../.env")

embeddings = OpenAIEmbeddings()
CHROMA_DB_FOLDER="./chroma_db"

vectorstore=None

if os.path.isdir(CHROMA_DB_FOLDER):
    vectorstore=load_documents_from_blog(embeddings)
else:
    vectorstore=Chroma(persist_directory=CHROMA_DB_FOLDER,embedding_function=embeddings)

# create a retrieval chain
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


print("--- call openAI via a retrieval LangChain chain")

response = retrieval_chain.invoke({"input": "how can LangSmith help with testing?"})
print(response["answer"])