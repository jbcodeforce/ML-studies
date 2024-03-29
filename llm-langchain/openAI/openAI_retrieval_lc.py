from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain
import os


"""
Add Retrieval to get better data, by crawling a product documentation from the web using BeautifulSoup
then FAISS vector store
"""
load_dotenv(dotenv_path="../../.env")

print(" Load source document from web site")
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
# index it into a vectorstore
embeddings = OpenAIEmbeddings()
print(" Split the documents into chunks")
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
print(" Create embeddings and save them in vector stores")
vector = FAISS.from_documents(documents, embeddings)

# create a retrieval chain

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


print("--- call openAI via a retrieval LangChain chain")

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])