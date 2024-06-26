# Mistral.ai

French Startup to build mixture of experts based LLMs with open source offering. 
 
* The open-weights models are Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
* The commercial models (Mistral Small, Mistral Medium, Mistral Large, and Mistral Embeddings (retrieval score of 55 on MTEB), codetral for code generation.

[Models description and benchmarks notes.](https://docs.mistral.ai/getting-started/models/)

Model can be fine tuned.

| model | type of usage |
| --- | --- |
| **Mistral Small** | Classification, Customer support, text gen. | 
| **Mistral 8x22B** | intermediate tasks that require moderate reasoning - like Data extraction, Summarizing a Document, Writing a Job Description, or Writing Product Descriptions |
| **Mistral Large** | Complex tasks that require large reasoning capabilities or are highly specialized - like Synthetic Text Generation, Code Generation, RAG, or Agents |

Function calling is supported by Mistral Small, Large, 8x22B.

Mistral delivers docker image for the model. To run locally with [skypilot](https://skypilot.readthedocs.io/) or [Ollama](https://ollama.com/) and [docker compose](https://github.com/jbcodeforce/ML-studies/blob/master/e2e-demos/ollama-mistral/docker-compose.yaml).


???- info "SkyPilot"
    [skypilot](https://skypilot.readthedocs.io/) delivers a CLI to deploy batch job on cluster deployed on AWS or GCP. Cluster of EC2 machines are created dynamically, as well as security group, security key,... The concept of task is used to run ML job by requesting specific resources like TPU, GPU, disk... The resources are created in the user's cloud account. It  allows easy movement of data between task VMs and cloud object stores.
---

## Mixture of Experts

MoE combines multiple models to make predictions or decisions. Each expert specializes in a specific subset of the input space and provides its own prediction. The predictions of the experts are then combined, typically using a gating network, to produce the final output.

It is useful when dealing with complex and diverse data, each expert extract different aspects or patterns in the data.

MoE in language translation may use experts by language pairs

