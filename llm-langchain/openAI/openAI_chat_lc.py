from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
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

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])


retriever = vector.as_retriever()
retriever_chain  = create_history_aware_retriever(llm, retriever, prompt)


print("--- Simulate a conversation with a human")

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

# returns documents about testing in LangSmith. This is because the LLM generated a new query,
#  combining the chat history with the follow-up question.

print("continue the conversation with these retrieved documents in mind.")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print(response["answer"])