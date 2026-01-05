from openai import OpenAI
import os

model_name=os.getenv("LLM_MODEL","llama3.3:latest")
agent_system_prompt = "You are an expert in AI, answer the user question."
llm_api_key=os.getenv("LLM_API_KEY","ollama_test_key")
llm_base_url=os.getenv("LLM_BASE_URL","http://localhost:11434/v1")
llm_client=OpenAI(api_key=llm_api_key, base_url=llm_base_url)

def _load_prompts():
    fname = "prompt.txt"
    with open(fname, "r") as f:
        system_prompt= f.read()
    return system_prompt

def chat_with_ollama(question: str) -> str:
    messages = [
       {"role": "system", "content": "You are an expert Socratic partner and critical thinking aide."},
       {"role": "user", "content": f"question: {question}" }
    ]
    response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
    return response.choices[0].message.content
    
def save_conversation(answer: str):
    print("Saving the conversation to a file")
    with open("conversation.txt", "a") as f:
        f.write(f"{answer}\n")

if __name__ == '__main__':
    print("Chat with Deep Thinker until entering an empty question or bye or /save")
    system_prompt = _load_prompts()
    done = False
    print("Subject >:")
    while not done:  
        question = input()
        if not question or 'bye' in question:
            done = True
        elif question == "/save":
            save_conversation(answer)
        else:
            full_question = system_prompt.replace("{subject}", question)
            print(full_question)
            answer = chat_with_ollama(full_question)
            print(answer)
            print("my comment >:")


