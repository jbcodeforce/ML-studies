from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from dotenv import load_dotenv
import os

print("--- Welcome to a basic QA with Mistral")
load_dotenv(dotenv_path="../../.env")

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"

client = MistralClient(api_key=api_key)

chat_response = client.chat(
    model=model,
    messages=[ChatMessage(role="user", content="What is the best French cheese?")]
)

print(chat_response.choices[0].message.content)