
"""
Based on deeplearning.ai training
"""
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory

load_dotenv(dotenv_path="../../.env")

llm = ChatOpenAI(temperature = 0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm= llm,
    memory=memory,
    verbose=True   # trace the chain
)


def basic_conversation_memory():
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm= llm,
        memory=memory,
        verbose=True   # trace the chain
    )
    rep=conversation.predict(input="Hi my name is Jerome")
    print(rep)
    rep=conversation.predict(input="give me a brief value proposition of using LlamaIndex?")
    print(rep)
    rep=conversation.predict(input="What is my name?")
    print(rep)
    print(memory.buffer)

def short_term_memory():
    short_term_mem = ConversationBufferWindowMemory(k=1)
    short_term_mem.save_context({"input": "hi this is me"}, {"output": "how can I help you"})
    short_term_mem.save_context({"input": "may be bring me a big bottle of beer"}, {"output": "I do not know how to do that"})
    # will return the last message only as k =1
    print(short_term_mem.load_memory_variables({}))

def play_with_knowledge_graph_memory():
    memory = ConversationKGMemory(llm=llm)
    memory.save_context({"input": "say hi to Julie"}, {"output": "who is Julie"})
    memory.save_context({"input": "Julie is my daughter"}, {"output": "okay"})
    memory.save_context({"input": "Julie is the sister of Mathieu"}, {"output": "okay"})
    memory.save_context({"input": "Mathieu is a boy"}, {"output": "okay"})
    print(memory.get_current_entities("Who is Julie's brother?"))

play_with_knowledge_graph_memory()