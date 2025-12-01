import requests,os,sys
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

load_dotenv(dotenv_path="../../.env")
try:
    watson_api_key=os.environ.get("IBM_API_KEY")
    project_id=os.environ.get("IBM_WATSON_PROJECT_ID")
    watson_api_key=os.environ.get("WATSONX_APIKEY")
    watsonx_url=os.environ.get("WATSONX_URL")
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
                   api_key =watson_api_key
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
    credentials=Credentials(
        api_key = watson_api_key,
        url = watsonx_url,
        project_id=project_id
    )

generated_response = model.generate_text_stream(prompt="what are the solar system planetes?")

for chunk in generated_response:
    print(chunk, end='', flush=True)

"""    
llm = WatsonxLLM(
            #model_id="ibm/granite-13b-instruct-v2",
            model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q",
            url=watsonx_url,
            project_id=project_id,
            params=parameters,
        )

print(llm.invoke("Who is man's best friend?"))


url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """As technical writer can you improve the following sentence:

I will take a imaginary business case of car ride on autonomous vehicles.

For this discussion, the simplest requirements include, booking a ride to go from one location to another location, manage a fleet of autonomous cars, get payment from the customer once the trip is done, get estimation for the ride duration and price to present a proposition to the end-users.

The modern synchronous based microservice component view may look like in the following figure:""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-13b-chat-v2",
	"project_id": "e44c2b60-dc53-4e6d-99a0-cf5ca013d17b",
	"moderations": {
		"hap": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		}
	}
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer " + iam_token
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()
print(data)
"""