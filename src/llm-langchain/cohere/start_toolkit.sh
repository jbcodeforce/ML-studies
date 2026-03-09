
source ../../.env
echo $COHERE_API_KEY
docker run -e COHERE_API_KEY=$COHERE_API_KEY -e TAVILY_API_KEY=$TAVILY_API_KEY -p 8001:8000 -p 4001:4000 ghcr.io/cohere-ai/cohere-toolkit:latest
