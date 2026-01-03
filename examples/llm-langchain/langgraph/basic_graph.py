from dotenv import load_dotenv
import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ibm import WatsonxLLM
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import gradio as gr

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# The add_messages function in our State will append the llm's response messages to whatever messages
# are already in the state.
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

load_dotenv("../../.env")
print("Build LangGraph and LLM api")
# defines the structure of our chatbot as a "state machine".
graph_builder = StateGraph(State)
#llm = ChatAnthropic(model="claude-3-haiku-20240307")
parameters = {
            "decoding_method": "sample",
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "temperature": 0.5,
            "top_k": 50,
            "top_p": 1,
        }
project_id=os.environ.get("IBM_WATSON_PROJECT_ID")
llm = WatsonxLLM(
            #model_id="ibm/granite-13b-instruct-v2",
            model_id="meta-llama/llama-3-8b-instruct",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id,
            params=parameters,
        )

graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(interactive=True)
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        for event in graph.stream({"messages": ("user", history[-1][0])}):
            for value in event.values():
                history[-1][1] = value["messages"][-1]
                return history

    # When enter key in textbox: delegate to user function to add user message to the history
    msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(
        # send to chatbot the same message to get an answer.
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

print("http://localhost:7860")    
demo.queue()
demo.launch()
