from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
import os, sys


"""
Use the Langchain user guide, Chromadb and use chat history
"""
load_dotenv(dotenv_path="../../.env")

embeddings = OpenAIEmbeddings()
CHROMA_DB_FOLDER="./chroma_db"

vectorstore=None

if os.path.isdir(CHROMA_DB_FOLDER):
    vectorstore=Chroma(persist_directory=CHROMA_DB_FOLDER,embedding_function=embeddings)
else:
    print("Need to run openAI_retrieval.py program to create the vectorDB")
    sys.exit(1)


# create a retrieval chain

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])


retriever = vectorstore.as_retriever()
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
