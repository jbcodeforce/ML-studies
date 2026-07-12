# Sample sandbox to get Ollama running locally

[Ollama](https://ollama.com/library) is a framework to LLM locally. [The CLI](https://github.com/ollama/ollama).

## Starting Ollama with CLI

```sh
ollama run llama3.2
```

## Docker compose

Create a service like:

```yaml
  ollama:
    image: ollama/ollama
    hostname: ollama
    volumes:
        - ./start-ollama.sh:/start-ollama.sh
    container_name: ollama
    entrypoint: ["/usr/bin/bash", "/start-ollama.sh"]
    ports:
        - 11434:11434
```

The start_ollama script load llama3.2. Snapshot the container once created:

```sh
docker commit <container_id> jbcodeforce/ollama-llama3.2
```

## With OWL Framework

Use docker compose file in the `owl` directory to start the backend.
