"""
Only use open AI api
 https://github.com/openai/openai-python 
"""
from openai import OpenAI


from dotenv import load_dotenv


load_dotenv("../../.env")
LLM_MODEL="gpt-3.5-turbo"
client = OpenAI()

def get_completion(prompt, model= LLM_MODEL):
    messages = [{ "role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model, messages=messages, temperature=0)
    return response.choices[0].message.content


print(get_completion("where is san francisco?"))

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

print(get_completion(prompt))