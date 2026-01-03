"""
A simple text chat with Gemini
@author Jerome Boyer
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def build_gemini_chain():
    """
    Build a langchain chain using Google Gemini model
    Returns:
        a langchain chain
    """
    gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    prompt=ChatPromptTemplate.from_messages([
                     ("system", "You are an expert in Google cloud products and services"),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    ])
    return   (
        prompt
        | gemini_model
        | StrOutputParser()
    )



def run_continue_chat(chain_with_llm):
    """
    Simple text interface to enter user's input and stream the output from LLM
    Parameters:
    - llm the langchain chain to call
    """
    print("\t Starting iterative question and answer\n\n")
    finish=False
    chat_history = []
    while not finish:
        question=input("Ask questions (q to quit): ")
        if question=='q' or question=='quit' or question=='Q':
            finish=True
        else:
            msg=""
            for chunk in chain_with_llm.stream({"input": question, "chat_history": chat_history}):
                print(chunk, end="", flush=True)
                msg +=chunk
            chat_history.append(msg)
            print("\n\n")
    print("Thank you , that was a nice chat !!")

if __name__ == "__main__":
    print("---> Welcome to Chat with Gemini")
    print("\t 1/ Build chain\n")
    llm = build_gemini_chain()
    print("\t 2/ Let's chat\n")
    run_continue_chat(llm)