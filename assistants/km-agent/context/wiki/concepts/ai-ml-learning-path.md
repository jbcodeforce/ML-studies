---
title: "AI/ML Learning Path"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/guide_for_ai.md]
related: [python-ml-development-environment, mathematical-foundations, ml-pipeline, pytorch-library, transformer-architecture, rag-architecture, agentic-ai, langgraph]
tags: [learning-path, curriculum, roadmap, ai, ml, beginner, intermediate, advanced]
---

# AI/ML Learning Path

A structured five-phase progression from prerequisites through agentic AI, providing a comprehensive roadmap for mastering artificial intelligence and machine learning.

## Phase Structure

### Phase 0: Prerequisites
Python fundamentals, environment setup with uv, Git version control, and optional Linux commands. Focus on data-science libraries: NumPy, Pandas, Matplotlib, Seaborn, PyTorch.

### Phase 1: Mathematical Foundations
- **Linear Algebra**: Scalars, vectors, matrices, tensors, PCA, eigenvalues
- **Probability & Statistics**: Bayes theorem, distributions, covariance, normalization
- **Calculus**: Gradients, gradient descent, learning rate
- **Bias-Variance Tradeoff**: Regularization (L1/Lasso, L2/Ridge, Elastic Net)

### Phase 2: Core Machine Learning
- Supervised learning: regression (linear, polynomial) and classification (perceptron, Adaline, logistic regression, SVM, decision trees, random forest, KNN)
- Unsupervised learning: K-Means clustering, dimensionality reduction
- Model validation: cross-validation, train/validation/test splits
- Performance metrics: MAE, MSE, RMSE, R², precision, recall, F1, ROC-AUC

### Phase 3: Deep Learning
- Neural network fundamentals: neurons, layers, activation functions (Sigmoid, ReLU, Softmax)
- PyTorch: tensors, GPU computation, modules, optimizers
- Classification networks: cross-entropy, BCE loss, SGD/Adam
- CNNs: convolution, pooling, image processing
- Transfer learning: pre-trained models, fine-tuning, feature freezing

### Phase 4: LLMs and Generative AI
- Transformer architecture and inference parameters
- Prompt engineering: zero-shot, few-shot, CoT, prompt chaining, Tree of Thoughts
- Embeddings and vector databases (ChromaDB, FAISS, OpenSearch)
- RAG: indexing, retrieval, generation, advanced techniques (multi-query, RAG fusion, HyDE)
- LLM frameworks: LangChain, LlamaIndex, Haystack

### Phase 5: Agentic Systems
- Agent architecture: planning, memory, tool integration
- Design patterns: ReAct, Human-in-the-Loop, multi-agent cooperation
- LangGraph: stateful workflows, conditional edges, checkpointing
- Multi-agent frameworks: CrewAI, AutoGen, OpenSSA
- Model Context Protocol (MCP)

## Recommended Timelines

| Track | Duration | Phases |
|-------|----------|--------|
| Beginner | 2-3 months | 0-2 |
| Intermediate | 3-4 months | 3-4 |
| Advanced | 3-4 months | 4-5 |

## Supporting Resources

- **UI Frameworks**: Streamlit, Gradio, NiceGUI, Taipy
- **Cloud/Infrastructure**: GCP, Feature Stores, Airflow
- **Methodology**: Project planning approaches

## Sources
- [Complete AI/ML Learning Guide](../summaries/guide_for_ai.md)

## Related
- [Python ML Development Environment](python-ml-development-environment.md)
- [Mathematical Foundations](mathematical-foundations.md)
- [PyTorch Library](pytorch-library.md)
- [Transformer Architecture](transformer-architecture.md)
- [RAG Architecture](rag-architecture.md)
- [Agentic AI](agentic-ai.md)
- [LangGraph](langgraph.md)