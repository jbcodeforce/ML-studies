# Google Cloud AI Platform

This source documents Google's managed AI/ML services on GCP, covering development, training, and deployment of ML and Generative AI models. Key products include Vertex AI (managed training and Gen AI applications), Gemini multimodal LLM, Gemma open-weight models (9B/27B parameters), Cloud Run (serverless deployment), Colab (notebook environment), TPU (accelerated ML compute), and Cloud Workstation/Shell for development.

## Vertex AI
Vertex AI is Google's unified platform for custom model training and Gen AI applications. It supports multiple SDKs (Python, Node.js, Go, Java, C#, REST) and provides request augmentation methods including Grounding (connecting model output to verifiable sources to reduce hallucinations), RAG, and function calling. It also includes safety checking for prompts and responses. Pricing is based on tools, storage, compute, and cloud resources.

## Document AI
Google Document AI processes and understands documents using NLP and computer vision. It extracts structured information from PDFs, images, and scanned documents. Pre-trained models exist for invoices, receipts, contracts, and other specialized document types. Custom models can be built using Document AI Workbench. Document processors are the interface to ML models and can be generalized, specialized, or custom. Evaluation uses F1 score, accuracy, and recall metrics. Supports multi-language, online and batch processing, and no-code pipeline configuration.

## Google LLMs
- **Gemini**: Multimodal LLM accepting text, image, video, audio, and document inputs; produces text output. Accessible via Vertex AI.
- **Gemma**: Open-weight family of lightweight LLMs (9B and 27B parameter variants), decoder-only, with pre-trained and instruction-tuned versions. Available on Hugging Face and Kaggle. Designed to run on small devices.

## Cloud Infrastructure
- **Compute Engine**: Linux/Windows VMs; e2-micro instance is free-tier eligible; spot instances offer cost savings.
- **Colab**: Browser-based Jupyter notebook with VM kernel; enterprise-grade security.
- **Cloud Workstation**: Fully managed dev environment supporting any containerized editor; includes Gemini Code Assist.
- **Cloud Shell**: Free browser-based shell with 50h/week quota.
- **TPU**: Hardware designed for ML matrix operations, usable in GKE worker nodes.
- **Cloud Run**: Serverless platform for deploying web apps.