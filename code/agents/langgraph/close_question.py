""" A sample to demonstrate how to use a node to ask a question 
and from the response call a tool. then generate the response
"""
from dotenv import load_dotenv
load_dotenv()
import operator
from typing import Annotated, TypedDict, Union, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain.agents import create_openai_functions_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentFinish
from langchain_core.runnables import RunnableConfig
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt.tool_executor import ToolExecutor

from langgraph.checkpoint.memory import MemorySaver

class CloseQuestion(TypedDict):
    question: str
    response: str
    variable: str


def ask_close_question(cq: CloseQuestion) -> CloseQuestion:
    return cq

close_q_tool = Tool(
    name="AskCloseQuestion",
    func=ask_close_question,
    description="useful for when you need to ask closed questions to gather missing data",
)

class UserData(TypedDict):
    user_id: str
    first_name: str
    last_name: str
    risk_factor: float = .5

# ==== tool used ====
def get_user_data(user_id : str) -> UserData:
    return UserData(user_id, "Bob", "TheBuilder", risk_factor=.8)

get_user_data_tool = Tool(
    name="get_user_data",
    func=get_user_data,
    description="tool to get user data given its user_id",
)

tools = [get_user_data_tool]

llm = ChatMistralAI(model="mistral-small-latest")
llm.bind_tools(tools)
text = """you are an assistant to gather information about the user and then leverage tools"""
prompt = ChatPromptTemplate.from_messages([
                        ("system", text),
                        MessagesPlaceholder(variable_name="chat_history", optional=True),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
                    ])
agent_runnable = prompt | llm 

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    user_id: Optional[str] = ""
    close_question: str = ""

class ControlledConversationAssistant():

    def __init__(self):
        self.tool_executor = ToolExecutor(tools)
        workflow = StateGraph(AgentState)
        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.run_agent)
        workflow.add_node("action", self.execute_tools)
        workflow.add_edge("action", "agent")
        # First is to ask a question to the user
        workflow.add_node("first_agent", self.build_close_question)
        workflow.set_entry_point("first_agent")
        workflow.add_edge("first_agent", END)
        
        workflow.add_conditional_edges(
                "agent",
                self.should_continue,
                {
                    "continue": "action",
                    "end": END,
                },
            )
        workflow.add_conditional_edges(
                "first_agent",
                self.ask_close_question,
                {
                    "continue": "agent",
                    "end": END,
                },
            )
        self.graph = workflow.compile(checkpointer=MemorySaver())  # 
       

    def build_close_question(self,state):
        user_id= state["user_id"]
        question_asked = state["close_question"]
        if user_id is None and question_asked is None:
            msg = AIMessage(content="what is your user id, give the answer in the form user_id= ?")
            return {"chat_history": [msg], "close_question": "user_id"}
        else:
            return None
    
    def ask_close_question(self,state):
        last_message = state["chat_history"][-1]
        if isinstance(last_message, HumanMessage):
            return "continue"
        else:
            return "end"
        
    def run_agent(self,state):
        messages = state['chat_history']
        agent_outcome = agent_runnable.invoke(messages)
        return {"chat_history": [agent_outcome]}

    def execute_tools(self, state):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = state["agent_outcome"]
        output = self.tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}
    
    def should_continue(self,state):
        messages = state["chat_history"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    


if __name__ == '__main__':
    graph = ControlledConversationAssistant().graph
    config = {"configurable": {"thread_id": "t01"}}
    print("[Human]: how to get a loan?")
    m=HumanMessage(content="how to get a loan?")
    rep= graph.invoke({"chat_history": [m]}, config, debug=True)

    print(f"\n\t\t{rep['chat_history'][-1].content}")
    print("[Human]: my user_id=4567")
    m=HumanMessage(content="my user_id=4567")
    rep= graph.invoke({"chat_history": [m], "user_id": "4567"},  config, debug = True)
    print(f"\n\t\t{rep['chat_history'][-1].content}")
    print(rep['chat_history'])