import requests,os,sys
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path="../.env")
try:
    api_key=os.environ.get("IBM_API_KEY")
    project_id=os.environ.get("IBM_WATSON_PROJECT_ID")
    watson_api_key=os.environ.get("WATSONX_APIKEY")
except KeyError:
    print(" Set the IBM_API_KEY environment variable")
    sys.exit(1)

print(api_key)
# Get an IAM token from IBM Cloud
url     = "https://iam.cloud.ibm.com/identity/token"
headers = { "Content-Type" : "application/x-www-form-urlencoded" }
data    =  "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey="+api_key

response  = requests.post( url, headers=headers, data=data)
iam_token = response.json()["access_token"]

print(iam_token)

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
import json

print( json.dumps( ModelTypes._member_names_, indent=2 ) )


parameters = {
            "decoding_method": "sample",
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "temperature": 0.5,
            "top_k": 50,
            "top_p": 1,
        }

llm = WatsonxLLM(
            model_id="ibm/granite-13b-instruct-v2",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id,
            params=parameters,
            api_key=watson_api_key
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
				"enabled": true,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": true
				}
			},
			"output": {
				"enabled": true,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": true
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