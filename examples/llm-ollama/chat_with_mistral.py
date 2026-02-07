
from ollama import chat
from pydantic import BaseModel
import os

mistral_model_name=os.getenv("LLM_MODEL","mistral-small:latest")
agent_system_prompt = "You are an expert in game coding and implementation  answer the user question."

def chat_with_ollama(question: str) -> str:
    messages = [
      {"role": "system", "content": agent_system_prompt},
       {"role": "user", "content": f"question: {question}" }
    ]
    result= ""
    for part in chat(model=mistral_model_name,
                 messages=messages, 
                 options={'temperature': 0.2},
                 stream=True,
                 # think=True  -- only available for specific models
                ):
        result+=part['message']['content']
        print(part['message']['content'])
    return result


if __name__ == '__main__':
    print("Chat with mistral until entering an empty question")
    done = False
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            answer = chat_with_ollama(question)
            print(answer)

