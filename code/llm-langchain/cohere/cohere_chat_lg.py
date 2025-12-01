from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

chat = ChatCohere()
messages = [HumanMessage(content="What is cohere ai?")]
print(chat.invoke(messages))