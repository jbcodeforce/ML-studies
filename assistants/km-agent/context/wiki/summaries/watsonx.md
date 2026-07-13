# WatsonX.ai Summary

WatsonX.ai is IBM's enterprise AI platform that provides a unified environment for traditional ML model development and LLM experimentation. The platform supports the full lifecycle: train, validate, tune, and deploy AI models or LLMs through WatsonX Studio.

## Key Capabilities

**Studio Environment**: Covers both traditional ML and LLM workflows in a single platform, with AutoAI for no-code ML model creation.

**Prompt Lab**: An interactive environment with three sandbox modes (chat, structured, freeform) for building and testing prompts. Prompts are tokenized before model invocation, with usage costs calculated by token count. Features include session history with version control, one-shot prompting examples, and export to Jupyter notebooks.

**Foundation Models**: Supports open-source LLMs from Mistral, Llama, and IBM's own Granite models. The **codellama-34b-instruct-hf** model excels at code tasks. "Instruct" models perform better at structured output and following instructions. Model size does not guarantee better results.

**AI Guardrails**: Built-in controls to prevent harmful input and output text.

**Fine-tuning & Prompt Tuning**: WatsonX supports fine-tuning on proprietary data and prompt tuning (soft prompts without changing model weights). Prompt tuning is useful for business-specific terminology where LLMs struggle with domain-specific language. A one-time tuning can outperform multi-shot prompting at lower cost.

**Synthetic Data Generation**: WatsonX can generate synthetic tabular data conforming to existing schemas, supporting categorical values, numerical distributions (Kolmogorov-Smirnov, Anderson-Darling), anonymization, and correlation building between columns.

## Technical Integration

Access is via Python SDK (`ibm-watsonx-ai`), LangChain (`langchain_ibm`), or LlamaIndex. Authentication uses API keys and IAM tokens through IBM Cloud IAM. Inference parameters include greedy vs. sampling modes with temperature, top-k, and top-p controls.

## Granite Models

IBM's Granite foundation models use a decoder-only transformer architecture, trained on filtered enterprise data spanning internet, academic, code, legal, and finance domains. Training data is filtered to remove hate speech, copyrighted materials, duplicates, and undesirable content. Granite models deliver strong performance on instruction-following tasks.