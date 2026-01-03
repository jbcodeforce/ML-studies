
from ollama import chat, AsyncClient
import asyncio
from pydantic import BaseModel
import os

mistral_model_name=os.getenv("LLM_MODEL","gpt-oss:20b")
agent_system_prompt = "You are an expert in AI, answer the user question."

def chat_with_ollama(question: str) -> str:
    messages = [
      {"role": "system", "content": agent_system_prompt},
       {"role": "user", "content": f"question: {question}" }
    ]

    answer = chat(model=mistral_model_name,
                 messages=messages, 
                 options={'temperature': 0.2}
                )
    return answer['message']['content']

async def main():
    client = AsyncClient()
    done = False
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            messages = [
                {"role": "system", "content": agent_system_prompt},
                {"role": "user", "content": f"question: {question}" }
                ]
            answer = await client.chat(model=mistral_model_name, messages=messages)
            print(answer['message']['content'])


if __name__ == '__main__':
    print("Chat with Gpt OSS 20b until entering an empty question")
    asyncio.run(main())

