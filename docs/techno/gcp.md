# AI at Google Cloud

Google offers a set of managed services to develop ML models and use Generative AI models. The [model garden](https://cloud.google.com/model-garden) exposes a catalog of Google or 3nd party models.

## Cloud Engine

Run virtual machines on Google infrastructure. 

To create Linux or Windows based VM. e2-micro is free for < 30Gb storage and 1GB of outbound data transfers. [Spot instances to pay for less](https://cloud.google.com/compute/docs/instances/spot). [See pricing calculator](https://cloud.google.com/products/calculator).

Install Apache HTTP server:

```sh
sudo apt update && sudo apt -y install apache2
```


## Cloud Workstation

[Cloud Workstation](https://cloud.google.com/workstations) is a fully managed dev env. It supports any code editors and applications that can be run in a container. And it supports Gemini Code Assist. [Pricing](https://cloud.google.com/workstations/pricing) is based of per-hour usage, management fees, control plane and network fees.

## Cloud Shell

Manage infrastructure and develop our applications from any browser with Cloud Shell. Free for all users, but has a weekly quotas of 50h.

To run Gcloud cli locally [see install instructions.](https://cloud.google.com/sdk/docs/install)

## Vertex AI

Managed services for custom model training, but also app on top of Gen AI. It includes a [SDK in Python](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal), nodejs, Go, Java, C# or REST.API.

Pricing is based on the Vertex AI tools and services, storage, compute, and Google Cloud resources used.

The generative AI workflow:

![](https://cloud.google.com/static/vertex-ai/generative-ai/docs/images/generative-ai-workflow.png)

Interesting features:

* offers multiple request augmentation methods that give the model access to external APIs and real-time information: Grounding, RAG amd function calling.
* checks both the prompt and response for how much the prompt or response belongs to a safety category

???- info "Grounding"
        [Grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) is the ability to connect model output to verifiable sources of information. It reduces model hallucinations, links model responses to specific information, and it enhances the trustworthiness. LLM response is based on Google Search to get public knowledge as facts. Grounding can be done with enterprise data using Vertex AI Search.

## Colab

A new notebook experience with enterprise-grade privacy and security.

## Gemini models

Gemini i a multi-modal (text, audio, image, video) model, Imagen (image generation), Codey are dedicated Gen AI models.

## 