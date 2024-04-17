from langchain_openai import ChatOpenAI,OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os,sys


print("--- Welcome to a basic Agent with LangChain")
load_dotenv(dotenv_path="../../.env")
embeddings = OpenAIEmbeddings()
CHROMA_DB_FOLDER="./chroma_db"

vectorstore=None

if os.path.isdir(CHROMA_DB_FOLDER):
    vectorstore=Chroma(persist_directory=CHROMA_DB_FOLDER,embedding_function=embeddings)
else:
    print("Need to run openAI_retrieval to create the vectorDB")
    sys.exit(1)


retriever = vectorstore.as_retriever()

print("--- Agent needs tools, so use retriever as one of the tool, and Tavily Search as the second one")
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [retriever_tool, search]

print("--- get a predefined prompt from LangChain hub")
#prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo", temperature=0)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

resp=agent_executor.invoke({"input": "how can LangSmith help with testing?"})
print(resp)
resp=agent_executor.invoke({"input": "what is the weather in SF?",
                            "chat_history": [
                                HumanMessage(content="how can LangSmith help with testing"),
                                AIMessage(content="LangSmith can help with testing by providing services such as test automation, test management, and quality assurance. They offer solutions to streamline testing processes, improve test coverage, and ensure the quality of software products. LangSmith's expertise in testing can assist organizations in achieving efficient and effective testing practices to deliver high-quality software products."),
                            ]})
print(resp)
