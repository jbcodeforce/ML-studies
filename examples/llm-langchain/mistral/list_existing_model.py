from dotenv import load_dotenv
load_dotenv()

from mistralai.client import MistralClient


model = "mistral-large-latest"
client = MistralClient()
for model_card in client.list_models().data:
    print(f"model id: {model_card.id} and can be used for fine tuning {model_card.permission[0].allow_fine_tuning}")