from openai import OpenAI
import os
import argparse

model_name=os.getenv("LLM_MODEL","mistral-small:latest")
agent_system_prompt = "You are an expert in AI, answer the user question."
llm_api_key=os.getenv("LLM_API_KEY","ollama_test_key")
llm_base_url=os.getenv("LLM_BASE_URL","http://localhost:11434/v1")
llm_client=OpenAI(api_key=llm_api_key, base_url=llm_base_url)


def chat_with_ollama(question: str, model: str = None) -> str:
    messages = [
      {"role": "system", "content": agent_system_prompt},
       {"role": "user", "content": f"question: {question}" }
    ]
    response = llm_client.chat.completions.create(
            model=model or model_name,
            messages=messages
        )
    return response.choices[0].message.content
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with Ollama')
    parser.add_argument('--model', type=str, default=model_name, help='Model to use')
    args = parser.parse_args()
    print(f"Chat with {args.model} until entering an empty question")
    done = False
    while not done:
        print("Question >:")
        question = input()
        if not question or 'bye' in question:
            done = True
        else:
            answer = chat_with_ollama(question, args.model)
            print(answer)