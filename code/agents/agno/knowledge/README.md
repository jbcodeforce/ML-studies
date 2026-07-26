# Own knowledge code with local embedder and LLM server

07/2026. Based on agno different examples.

## Start Qdrant Vector Store using container

* Be sure container is started 
    ```sh
    container system start
    ```

* Start the vector store with local storage
    ```sh
    ./start_qdrant.sh
    ```
* See administration console at [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Prepare vector store

The code will take the two first chapter of the astronomy chapter.

```sh
uv run knowledge/prepare_vs.py
```

## Query the agent

```sh
uv run knowledge/omlx_adv_rag.py
```