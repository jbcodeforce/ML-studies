import requests,os
from dotenv import load_dotenv

load_dotenv()

api_key= os.environ["IBM_API_KEY"]

# Get an IAM token from IBM Cloud
url     = "https://iam.cloud.ibm.com/identity/token"
headers = { "Content-Type" : "application/x-www-form-urlencoded" }
data    = "apikey=" + api_key + "&grant_type=urn:ibm:params:oauth:grant-type:apikey"
response  = requests.post( url, headers=headers, data=data, auth=api_key )
iam_token = response.json()["access_token"]

print(iam_token)
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
import json

print( json.dumps( ModelTypes._member_names_, indent=2 ) )