from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os,sys

try:
    api_key=os.environ.get("ANTHROPIC_API_KEY")
except KeyError:
    print(" Set the ANTHROPIC_API_KEY environment variable")
    sys.exit(1)
chat = ChatAnthropic(temperature=0, anthropic_api_key=api_key, model_name="claude-3-opus-20240229")
print("Call Claude")
prompt = ChatPromptTemplate.from_messages(
    [("human", "Give me a list of famous tourist attractions in Japan")]
)
chain = prompt | chat
for chunk in chain.stream({}):
    print(chunk.content, end="", flush=True)


