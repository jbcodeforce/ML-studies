from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentActionMessageLog, AgentFinish
import json
from dotenv import load_dotenv

"""
Simple tool calling with structured response, custom parser
"""

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )
    
load_dotenv(dotenv_path="../../.env")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(temperature=0)
search = TavilySearchResults(max_results=1)
# bind function 
llm_with_tools = llm.bind_functions([search, Response])

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)



agent_executor = AgentExecutor(
    agent=agent, tools=[search], verbose=True, handle_parsing_errors=True
)

resp=agent_executor.invoke({"input": "what is LangServe?"}, return_only_outputs=True,)

print(resp)