mkdir dependencies
cd dependencies
curl -sS https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip > sdk.zip
python -m zipfile -e sdk.zip . && rm sdk.zip
pip install --no-build-isolation --force-reinstall     ./awscli-*-py3-none-any.whl     ./boto3-*-py3-none-any.whl     ./botocore-*-py3-none-any.whl