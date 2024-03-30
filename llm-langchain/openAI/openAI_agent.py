from langchain_openai import ChatOpenAI,OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os

def get_data_from_source():
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents

def build_retreiver_for_FAISS(documents):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    return vector.as_retriever()

print("--- Welcome to a basic Agent with LangChain")
load_dotenv(dotenv_path="../../.env")
docs= get_data_from_source()
retriever = build_retreiver_for_FAISS(docs)
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [retriever_tool, search]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

resp=agent_executor.invoke({"input": "how can langsmith help with testing?"})
print(resp)
resp=agent_executor.invoke({"input": "what is the weather in SF?"})
print(resp)
