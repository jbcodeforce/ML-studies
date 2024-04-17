from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("hwchase17/react-chat-json")
print(prompt[0])

llm = ChatOpenAI()

agent = create_json_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

resp=agent_executor.invoke({"input": "what is LangServe?"})

print(resp['output'])