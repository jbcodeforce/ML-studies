import requests,os,sys
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

load_dotenv(dotenv_path="../../.env")
try:
    project_id=os.environ.get("IBM_WATSON_PROJECT_ID")
    watson_api_key=os.environ.get("IBM_WATSONX_APIKEY")
    watsonx_url=os.environ.get("IBM_WATSONX_URL")
except KeyError:
    print(" Set the IBM_API_KEY environment variable")
    sys.exit(1)



print("Get an IAM token from IBM Cloud")
url     = "https://iam.cloud.ibm.com/identity/token"
headers = { "Content-Type" : "application/x-www-form-urlencoded" }
data    =  "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey="+watson_api_key

response  = requests.post( url, headers=headers, data=data)
iam_token = response.json()["access_token"]
print(f"--> Token: {iam_token}")


from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
import json

print(f"\n\nExample of parameters {GenParams().get_example_values()}\n\n")
print("--> existing models in WatsonX.ai:")
print(json.dumps( ModelTypes._member_names_, indent=2 ) )

credentials = Credentials(
                   url = watsonx_url,
                   api_key =watson_api_key,
                  )

client = APIClient(credentials)

parameters = {
            "decoding_method": "sample",
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "temperature": 0.5,
            "top_k": 50,
            "top_p": 1,
        }

model = Model(
    model_id=ModelTypes.FLAN_UL2,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

generated_response = model.generate_text(prompt="what are the solar system planetes?")
print(generated_response)

