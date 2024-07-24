# Google AI platform

Google offers a set of managed services to develop ML models and use Generative AI models. The [model garden](https://cloud.google.com/model-garden) exposes a catalog of Google or 3nd party models.

## Most important products

* [Vertex AI](https://cloud.google.com/vertex-ai?hl=en)
* [Gemini LLM]() Prompt and test in Vertex AI with Gemini, using text, images, video, or code. Gemini is multimodal model, it accepts text, image, video, audio and document data as input and produces text output.
* [Gemma](https://blog.google/technology/developers/google-gemma-2/), a family of lightweight, state-of-the-art open models from 9B or 27 billion parameters. Based on Gemini embeddings. Uses a 256 k tokenizers. It supports  text-to-text, decoder-only large language models, with open weights for both pre-trained variants and instruction-tuned variants. Should be able to run on small device. [Available on Hugging Face](https://huggingface.co/google/gemma-2-9b), [kaggle](https://www.kaggle.com/models/google/gemma).
* [Search Generative Experience (Search lab)](https://search.google/ways-to-search/search-labs/)
* [Google Cloud run]() Serverless platform to deploy any web app
* [Colab](https://colab.research.google.com/) A new notebook experience with enterprise-grade privacy and security.
* [TPU](https://cloud.google.com/tpu?hl=en) designed ships specifically for matrix operations common in machine learning. Can be used in worker node of GKE.

### Cloud Engine

Run virtual machines on Google infrastructure. 

To create Linux or Windows based VM. e2-micro is free for < 30Gb storage and 1GB of outbound data transfers. [Spot instances to pay for less](https://cloud.google.com/compute/docs/instances/spot). [See pricing calculator](https://cloud.google.com/products/calculator).

Install Apache HTTP server:

```sh
sudo apt update && sudo apt -y install apache2
```

### Colab

Creating a Jupyter notebook in Google Drive will start colab. It can execute any python code and gets a VM as kernel

Simple [introduction video](). 


### Cloud Workstation

[Cloud Workstation](https://cloud.google.com/workstations) is a fully managed dev env. It supports any code editors and applications that can be run in a container. And it supports Gemini Code Assist. [Pricing](https://cloud.google.com/workstations/pricing) is based of per-hour usage, management fees, control plane and network fees.

### Cloud Shell

Manage infrastructure and develop our applications from any browser with Cloud Shell. Free for all users, but has a weekly quotas of 50h.


## Vertex AI

Managed services for custom model training, but also app on top of Gen AI. It includes a [SDK in Python](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal), nodejs, Go, Java, C# or REST.API.

Pricing is based on the Vertex AI tools and services, storage, compute, and Google Cloud resources used.

The generative AI workflow:

![](https://cloud.google.com/static/vertex-ai/generative-ai/docs/images/generative-ai-workflow.png)

Interesting features:

* offers multiple request augmentation methods that give the model access to external APIs and real-time information: Grounding, RAG amd function calling.
* checks both the prompt and response for how much the prompt or response belongs to a safety category

???+ info "Grounding"
     [Grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) is the ability to connect model output to verifiable sources of information. It reduces model hallucinations, links model responses to specific information, and it enhances the trustworthiness. LLM responses are based on Google Search to get public knowledge as facts. Grounding can be done with enterprise data using Vertex AI Search.

## Document AI

  **Google Document AI** is used to process and understand documents:

* Extracts and understands structured information from a wide variety of document types, including PDFs, images, and scanned documents.
* It leverages advanced natural language processing (NLP) and computer vision technologies to deliver high-accuracy document extraction and understanding. 
* Trained to extract data from invoices, receipts, and contracts.
* It allows to create custom document models to extract information from specialized or domain-specific documents.
* Different natural languages are supported.
* Scale up and down ensuring high throughput and low latency.
* Can run on-line or with batch processing.
* No-code tools to let developers quickly set up and configure document processing pipelines without writing any code.
* Integrated with monitoring and logging services.
* It is possible to combine with custom MM models to do better entity extraction.

Document processing is a complex domain in AI as it groups optical character recognition, NLP, Entity Extraction, machine translation and data lost prevention. Documents can be categorized into three different groups:

1. General docs, a basic content that should be parsed with general ML models, which includes OCR, structured form parser, and document quality analysis. 
1. Specialized docs, the content needs some specialized models pre-trained on w2s, driving licenses, invoices, expense reports, contracts which are high variance document types...
1. Custom docs, where the content needs to be parsed with custom models developed by the enterprise on their own domain. 

Document AI uses document processors, that represents the interface to a machine learning model, it is responsible to classify, parse, analyze a Document. Developers need to create processor instances in their project to use Document AI. A processor can be generalized, specialized or custom. Each is addressing a specific task, like optical character recognition, general form parsing, classification, or parsing specialized document types like invoices, expenses..., 

The document is defined as a Document Object to include raw text, layout, extracted entities, languages...

Developer needs to create service account for the Document AI application, then grant the DocumentAI API User role.

Document AI workbench helps developers to develop custom document processors, from new one or an existing ones. Uptraining is the way to add custom field for the entity extraction by extending the schema. While new model creation follows the standard ML process of labelling, training, evaluation and deployment.

It uses F1 score, accuracy and recall metrics to assess the quality of the entity extractions.

## Hands-on

* [Install gcloud](https://cloud.google.com/sdk/docs/install)
* Use Gemini inside VSCode

## Deeper dive

* [GCP architecture](https://cloud.google.com/architecture/framework) and [ML specific](https://cloud.google.com/architecture/framework/system-design/ai-ml)
* [LLM comparator](https://github.com/pair-code/llm-comparator)