import json
import os
import sys

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=False
)

resp= boto3_bedrock.list_foundation_models()
print(json.dumps(resp,indent=4))