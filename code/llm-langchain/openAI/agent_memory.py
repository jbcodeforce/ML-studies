from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
# format_to_openai_tool_messages: Convert (AgentAction, tool output) tuples into FunctionMessages.

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

MEMORY_KEY = "chat_history"

chat_history = []

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

def defineAgent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [get_word_length]
    prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are very powerful assistant, but don't know current events",
                    ),
                    MessagesPlaceholder(variable_name=MEMORY_KEY),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
    llm_with_tools = llm.bind_tools(tools)
    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                 "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
    return agent, tools

if __name__ == "__main__":
    print(f"Test tool function {get_word_length.invoke('abc')}")
    openAI_agent, tools = defineAgent()
    agent_executor = AgentExecutor(agent=openAI_agent, tools=tools, verbose=True)
    input1 = "how many letters in the word educa?"
    result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=input1),
            AIMessage(content=result["output"]),
        ]
    )
    print(list(agent_executor.stream({"input": "is that a real word?", 
                                      "chat_history": chat_history})))