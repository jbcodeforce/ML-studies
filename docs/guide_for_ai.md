# Complete AI/ML Learning Guide

A structured, end-to-end roadmap for learning Artificial Intelligence, Machine Learning, Deep Learning, Generative AI, and Agentic Systems. This guide integrates content from this repository with external resources for comprehensive coverage.

---

## Phase 0: Prerequisites

Build programming and tooling fundamentals required before entering AI/ML.

### 0.1 Python for AI/ML

Python basics and essential data-science libraries. See the [coding environment setup](./coding/index.md) for local development configuration.

#### Environment Setup with uv

[uv](https://docs.astral.sh/uv/) is the recommended package manager for this repository. It provides fast dependency resolution and virtual environment management.

```sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project and sync dependencies
uv sync

# Run scripts
uv run python script.py
```

#### Key Libraries

| Library | Description | Documentation |
|---------|-------------|--------------|
| NumPy | Array computing and numerical operations | [coding/index.md#numpy](./coding/index.md#numpy) |
| Pandas | Data manipulation and analysis | [pandas.md](./coding/pandas.md) |
| Matplotlib | Data visualization | [visualization.md](./coding/visualization.md) |
| Seaborn | Statistical graphics | [coding/index.md#seaborn](./coding/index.md#seaborn) |
| PyTorch | Pytorch library | [coding/pytorch.md](./coding/pytorch.md)

**External Resources:**

- [Python for Data Science - freeCodeCamp (Full Course)](https://www.youtube.com/watch?v=LHBE6Q9XlzI)
- [Python Object Oriented Programming - freeCodeCamp](https://www.youtube.com/watch?v=Ej_02ICOIgs)

### 0.2 Git Basics

Version control is required for all ML and AI work.

**External Resources:**

- [Git & GitHub Crash Course for Beginners](https://www.youtube.com/watch?v=RGOj5yH7evk)

### 0.3 Linux Commands (Optional)

Useful for development environments and servers.

**External Resources:**

- [The 50 Most Popular Linux & Terminal Commands - freeCodeCamp](https://www.youtube.com/watch?v=ZtqBQ68cfJc)

---

## Phase 1: Mathematical Foundations

Understand why ML and DL models work, not just how to use them.

### 1.1 Linear Algebra (Core of ML)

Core concepts: Scalars, vectors, matrices, tensors; vector operations (dot product, cross product, norm); matrix operations (multiplication, transpose, inverse); eigenvalues and eigenvectors; PCA intuition.

**Used in:** Neural Networks, Linear Regression, PCA, Embeddings, Attention, Transformers.

**External Resources:**

- [Essence of Linear Algebra - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

### 1.2 Probability and Statistics (ML Reasoning)

See [Mathematical Foundations](./concepts/maths.md) for:

- Probability basics and conditional probability
- Bayes theorem (prior, likelihood, posterior)
- Data distributions (Gaussian, Poisson, Uniform)
- Covariance and correlation
- Normalization techniques

**Key Notebooks:**

- [Conditional Probability Exercise](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/ConditionalProbabilityExercise.ipynb)
- [Distributions Notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Distributions.ipynb)

**External Resources:**

- [Statistics Fundamentals - StatQuest](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)

### 1.3 Calculus (Optimization and Learning)

See [ML Concepts - Cost Function](./ml/index.md#cost-function) for:

- Gradient and direction of steepest descent
- Gradient descent algorithm
- Learning rate and convergence

**External Resources:**

- [Essence of Calculus - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

### 1.4 Bias-Variance Tradeoff

See [ML Concepts](./concepts/index.md) for detailed coverage of:

- Variance and model consistency
- Bias and prediction accuracy
- Regularization (L1/Lasso, L2/Ridge, Elastic Net)
- Overfitting and underfitting

---

## Phase 2: Core Machine Learning

Learn classical ML using feature-based models and structured data.

### 2.1 Introduction to Machine Learning

See [Machine Learning Overview](./ml/index.md) for:

- ML vs AI vs Deep Learning
- Supervised, unsupervised, reinforcement learning
- Classification vs regression
- ML workflow and system design

### 2.2 Data Understanding and Preprocessing

See [Feature Engineering](./data/features.md) for:

- Handling missing values
- Categorical encoding (ordinal, one-hot)
- Feature scaling and normalization
- Mutual information for feature selection
- Creating new features

### 2.3 Supervised Learning - Regression

See [ML Index - Regression](./ml/index.md#supervised-learning) for:

- Linear, multiple, and polynomial regression
- Hypothesis functions
- Cost functions (MSE)
- Gradient descent

**Metrics:** MAE, MSE, RMSE, R-squared. See [Performance Metrics](./concepts/index.md#common-performance-metrics-used).

### 2.4 Supervised Learning - Classification

See [Classifiers](./ml/classifier.md) for detailed implementations:

| Algorithm | Description | Code |
|-----------|-------------|------|
| Perceptron | Basic neural unit | [TestPerceptron.py](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestPerceptron.py) |
| Adaline | Adaptive Linear Neuron | [TestAdaline.py](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestAdaline.py) |
| Logistic Regression | Probability-based classification | [classifier.md#logistic-regression](./ml/classifier.md#logistic-regression) |
| SVM | Maximum margin classification | [SVM-IRIS.py](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/SVM-IRIS.py) |
| Decision Trees | Rule-based learning | [DecisionTreeIRIS.py](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/DecisionTreeIRIS.py) |
| Random Forest | Ensemble learning | [classifier.md#random-forests](./ml/classifier.md#combining-weak-to-strong-learners-via-random-forests) |
| KNN | Instance-based learning | [KNN Notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/KNN.ipynb) |

**Metrics:** Confusion matrix, accuracy, precision, recall, F1, ROC-AUC.

### 2.5 Unsupervised Learning

See [Unsupervised Learning](./ml/unsupervised.md) for:

- K-Means clustering
- Cluster labels and distance features
- Dimensionality reduction

### 2.6 Model Selection and Validation

See [ML System](./ml/index.md#ml-system) for:

- Train/validation/test split
- Cross-validation (K-fold, LOOCV)
- Bias-variance tradeoff in practice

### 2.7 ML Libraries (Hands-On)

See [Coding Index](./coding/index.md) for environment setup and:

- [NumPy basics](./coding/index.md#numpy)
- [Pandas for data manipulation](./coding/pandas.md)
- [Scikit-learn](./coding/sklearn.md)
- [Visualization with Matplotlib](./coding/visualization.md)

**External Resources:**

- [Machine Learning with Python and Scikit-Learn - freeCodeCamp](https://www.youtube.com/watch?v=pqNCD_5r0IU)
- [Stanford CS229: Machine Learning (Andrew Ng)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

---

## Phase 3: Deep Learning and Advanced ML

Build and understand neural-network-based systems end-to-end.

### 3.1 Neural Network Fundamentals

See [Deep Learning](./ml/deep-learning.md) for:

- Neuron structure and activation
- Input, hidden, and output layers
- Activation functions (Sigmoid, ReLU, Softmax)
- Forward and backward propagation

### 3.2 PyTorch Framework

See [PyTorch](./coding/pytorch.md) for comprehensive coverage:

- Tensors and GPU computation
- Neural network modules (`torch.nn`)
- Optimizers and loss functions
- Training workflows

**Key Notebooks:**

- [Tensor Basics](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/get_started/torch-tensor-basic.ipynb)
- [Basic ML Workflow](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/get_started/workflow-basic.ipynb)
- [Classification](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/classifier.ipynb)

### 3.3 Classification Neural Networks

See [Classification Architecture](./ml/deep-learning.md#classification-neural-network-architecture) for:

- Layer design and hyperparameters
- Loss functions (Cross entropy, BCE)
- Optimizer selection (SGD, Adam)

### 3.4 Convolutional Neural Networks (CNNs)

See [CNN Section](./ml/deep-learning.md#convolutional-neural-network) for:

- Convolution and pooling layers
- Image processing architecture
- Feature extraction

**Code Examples:**

- [LeNet in Keras](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/neuralnetwork/lenet_in_keras.ipynb)
- [Fashion CNN](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer-vision/fashion_cnn.py)

### 3.5 Transfer Learning

See [Transfer Learning](./ml/deep-learning.md#transfer-learning) for:

- Pre-trained models usage
- Fine-tuning strategies
- Feature freezing

**External Resources:**

- [Neural Networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Neural Networks: Zero to Hero - Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [PyTorch for Deep Learning - Full Course](https://www.learnpytorch.io/)
- [MIT 6.S191: Introduction to Deep Learning](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
- [Dive into Deep Learning (Book)](https://d2l.ai)

---

## Phase 4: LLMs, NLP and Generative AI

Use transformer-based LLMs to build real-world AI applications.

### 4.1 Generative AI Foundations

See [Generative AI Overview](./genAI/index.md) for:

- Transformer architecture
- Pre-training and fine-tuning
- NLP processing and tokenization
- Embeddings and context windows

### 4.2 LLM Fundamentals

See [GenAI Concepts](./genAI/index.md#concepts) for:

- Encoder-decoder architectures
- Inference parameters (Temperature, Top-K, Top-P)
- Model selection considerations

### 4.3 Prompt Engineering

See [Prompt Engineering](./genAI/prompt-eng.md) for:

- Zero-shot and few-shot prompting
- Chain of Thought (CoT)
- Prompt chaining
- Tree of Thoughts
- Automatic Prompt Engineering

### 4.4 Embeddings and Vector Databases

See [GenAI - Vector Database](./genAI/index.md#vector-database) and [NLP Embeddings](./ml/nlp.md#embedding) for:

- Similarity search
- ChromaDB, FAISS, OpenSearch
- Embedding models

### 4.5 Retrieval-Augmented Generation (RAG)

See [RAG](./genAI/rag.md) for comprehensive coverage:

- RAG architecture (indexing, retrieval, generation)
- Document pipelines and chunking
- Retriever considerations
- Advanced RAG techniques (multi-query, RAG fusion, HyDE)
- Knowledge graph integration

### 4.6 LLM Providers

| Provider | Documentation |
|----------|--------------|
| OpenAI | [openai.md](./genAI/openai.md) |
| Anthropic Claude | [anthropic.md](./genAI/anthropic.md) |
| Mistral | [mistral.md](./genAI/mistral.md) |
| Cohere | [cohere.md](./genAI/cohere.md) |

### 4.7 LLM Development Frameworks

| Framework | Documentation |
|-----------|--------------|
| LangChain | [langchain.md](./coding/langchain.md) |
| LlamaIndex | [llama-index.md](./coding/llama-index.md) |
| Haystack | [haystack.md](./coding/haystack.md) |

**External Resources:**

- [Intro to Large Language Models - Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
- [Fine-Tuning LLM Models - Generative AI Course](https://www.youtube.com/watch?v=eC6Hd1hFvos)

---

## Phase 5: Agentic Systems and AI System Design

Design autonomous, goal-driven AI systems with tools, memory, orchestration, and safety controls.

### 5.1 From LLMs to Agents

See [Agentic AI](./genAI/agentic.md) for:

- Agent reference architecture
- Planning strategies (CoT, Tree of Thoughts, ReAct)
- Memory systems (short-term, long-term, entity)
- Tool integration

### 5.2 Agent Design Patterns

See [Agentic Guidelines](./genAI/agentic.md#guidelines) for:

- Role definition and focus
- Tool selection and management
- Multi-agent cooperation
- Guardrails and control

### 5.3 LangGraph for Agent Orchestration

See [LangGraph](./coding/langgraph.md) for:

- Stateful multi-actor applications
- Graph-based workflows
- Conditional edges and routing
- Human-in-the-loop patterns
- Persistence and checkpointing

**Key Patterns:**

- [ReAct Implementation](./coding/langgraph.md#reasoning-and-acting-react-implementation)
- [Adaptive RAG](./coding/langgraph.md#adaptive-rag)
- [Human in the Loop](./coding/langgraph.md#human-in-the-loop)

### 5.4 Multi-Agent Frameworks

| Framework | Description | Documentation |
|-----------|-------------|--------------|
| LangGraph | Graph-based orchestration | [langgraph.md](./coding/langgraph.md) |
| CrewAI | Multi-agent collaboration | [agentic.md#crewai](./genAI/agentic.md#crewai) |
| AutoGen | Conversable agents | [agentic.md#autogen](./genAI/agentic.md#autogen) |
| OpenSSA | Small Specialist Agents | [agentic.md#openssa](./genAI/agentic.md#openssa) |

### 5.5 Model Context Protocol (MCP)

See [MCP](./genAI/mcp.md) for:

- Standardized tool integration
- Context management
- Protocol implementation

### 5.6 Agent Use Cases

See [Agentic Use Cases](./genAI/agentic.md#use-cases) for examples:

- Research and writing agents
- Customer support crews
- Sales lead analysis
- Job application tailoring

**Code Examples:**

- [Research Agent](https://github.com/jbcodeforce/ML-studies/blob/master/techno/crew-ai/research-agent.py)
- [Support Crew](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/support_crew.py)
- [Customer Outreach](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai/customer_outreach.py)

**External Resources:**

- [AI Agents for Beginners - Microsoft](https://github.com/microsoft/ai-agents)
- [AI Agents Course - Hugging Face](https://huggingface.co/learn/agents-course)
- [LangGraph Complete Course](https://www.youtube.com/watch?v=R8KB-Zcynxc)
- [Building AI Agents from Scratch](https://www.youtube.com/watch?v=vik_kl-OW4k)
- [Agentic AI by Andrew Ng](https://www.youtube.com/watch?v=sal78ACtGTc)

---

## Supporting Topics

### UI Frameworks for AI Applications

| Framework | Documentation |
|-----------|--------------|
| Streamlit | [streamlit.md](./techno/streamlit.md) |
| Gradio | [gradio/index.md](./techno/gradio/index.md) |
| NiceGUI | [nicegui.md](./techno/nicegui.md) |
| Taipy | [taipy/index.md](./techno/taipy/index.md) |

### Cloud and Infrastructure

| Platform | Documentation |
|----------|--------------|
| GCP | [gcp/index.md](./techno/gcp/index.md) |
| Feature Stores | [feature_store.md](./techno/feature_store.md) |
| Airflow | [airflow.md](./techno/airflow.md) |

### Methodology

See [Methodology](./methodology/index.md) for project planning approaches.

---

## Books and Resources

See the [main index](./index.md#books-and-other-sources) for a comprehensive list of books and resources including:

- Python Machine Learning - Sebastian Raschka
- Collective Intelligence - Toby Segaran
- Stanford ML Course - Andrew Ng
- Dive into Deep Learning
- Kaggle competitions
- Papers with Code

---

## Learning Path Recommendations

### Beginner Path (2-3 months)

1. Phase 0: Python basics, NumPy, Pandas
2. Phase 1: Linear algebra essentials, probability basics
3. Phase 2: Scikit-learn classifiers, regression basics

### Intermediate Path (3-4 months)

1. Phase 3: PyTorch fundamentals, neural networks
2. Phase 3: CNNs and transfer learning
3. Phase 4: LLM basics, prompt engineering

### Advanced Path (3-4 months)

1. Phase 4: RAG implementation, fine-tuning
2. Phase 5: LangGraph agents
3. Phase 5: Multi-agent systems, production deployment

---

## Notes

- This guide prioritizes practical implementation with code examples from this repository.
- External video resources are provided for topics requiring deeper theoretical understanding.

