import os, sys
from langchain.llms import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

module_path = "."
sys.path.append(os.path.abspath(module_path))
from bedrock.utils import bedrock, print_ww

def buildBedrockClient():
    return bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None)
        )

def buildLLM():
    return  Bedrock(
                client=buildBedrockClient(),
                model_id="anthropic.claude-v2",
            )           

def simple_chat_chain(llm):
    text="what would you recommend to loose weight"
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )
    response=conversation.predict(input=text)
    print(response)


def useSimplePrompt(llm):
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    prompt.format(product="colorful socks")
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )
    print(conversation.predict(input=prompt.format(product="colorful socks")))

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        return text.strip().split(", ")



'''
LLMChain is the simplest chain in langchain.
This chain will take input variables, pass those to a prompt template to create a prompt, 
pass the prompt to an LLM, and then pass the output through an (optional) output parser.
It uses a chat model with human and system messages.
'''    
def buildConversationChain(llm):
    template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        output_parser=CommaSeparatedListOutputParser()
    )
    print(chain.run("colors"))



if __name__ == "__main__":
    llm = buildLLM()
    print("--- First prompt ---")
    simple_chat_chain(llm)
    print("--- Second prompt ---")
    useSimplePrompt(llm)
    print("--- Chat Chain ---")
    buildConversationChain(llm)