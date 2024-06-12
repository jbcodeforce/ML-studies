from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import os,sys
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

try:
    api_key=os.environ.get("ANTHROPIC_API_KEY")
except KeyError:
    print(" Set the ANTHROPIC_API_KEY environment variable")
    sys.exit(1)
    
chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
print("Call Claude")

system = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])


chain = prompt | chat
rep=chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "I love Langchain and langgraph",
    }
)
print(rep)

