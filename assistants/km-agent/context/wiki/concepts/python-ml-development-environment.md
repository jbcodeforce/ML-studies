---
title: "Python ML Development Environment"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/index.md]
related: [distributed-data-parallel, rag-architecture, llm-agentic-workflows, haystack-ai-framework]
tags: [python, uv, numpy, scipy, matplotlib, seaborn, pytorch, development-environment]
---

# Python ML Development Environment

The Python ML development environment in the ML-studies repository uses a modern toolchain centered around **uv** for package management and a subject-based code organization.

## uv Package Manager

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust that replaces pip, pip-tools, pipx, poetry, pyenv, virtualenv, and more. It was adopted in January 2026 as the recommended tool for managing the codebase.

**Installation:**
```sh
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

## Virtual Environments

Two virtual environments are maintained:

1. **code/.venv** — For Python/PyTorch scripts
   - `uv sync --extra pytorch` for PyTorch and computer vision
   - `uv sync --extra llm --extra agents` for LLM and agent scripts

2. **jupyter/.venv** — For notebook kernels
   - `uv sync --extra deep-learning` for Keras/TensorFlow notebooks
   - Notebooks live under `code/<subject>/` but use the jupyter kernel

## Core Libraries

### NumPy
Array computing in Python with axes-based dimensions. Supports operations like `np.arange()`, matrix products via `.dot` or `@`, and array creation with `np.zeros()` and `np.ones()`.

### SciPy
A collection of mathematical algorithms and convenience functions built on top of NumPy. Includes probability distributions (`scipy.stats.norm`), optimization, and other scientific computing tools.

### Matplotlib
Data visualization library for presenting figures among multiple axes. Classic import pattern uses `matplotlib.pyplot` as `plt` and `matplotlib` as `mpl`.

### Seaborn
A high-level interface for drawing attractive and informative statistical graphics. Built on top of Matplotlib and integrated with Pandas for easy data plotting.

### PyTorch
Deep learning framework installed via conda or pip. Includes torchvision and torchaudio. Code samples cover fundamentals, CNNs, transfer learning, and distributed training.

## Code Organization

Code is organized by **subject** under `code/` with a mapping documented in `code/SUBJECTS.md`. Key subject areas include:

- **Classification**: Perceptron, Adaline, SVM, decision trees, KNN
- **Regression**: Lasso/Ridge regularization, logistic regression
- **Deep Learning**: PyTorch fundamentals, RNNs, Keras CNNs
- **Computer Vision**: CNNs, transfer learning with EfficientNet
- **LLM**: LangChain integrations, RAG implementations, Ollama local LLM
- **Agents**: LangGraph patterns including ReAct and human-in-the-loop
- **Statistics**: Probability, distributions, PCA, K-Means

## Docker & Jupyter

- **Docker**: Kaggle's Docker images available for CPU and GPU-based development
- **Jupyter**: VSCode Jupyter extension with kernel selection and variable inspection
- **Legacy**: Per-folder `pyproject.toml` setup is deprecated in favor of uv

## Sources
- [Coding Index](../summaries/index.md)

## Related
- [Distributed Data Parallel](distributed-data-parallel.md)
- [RAG Architecture](rag-architecture.md)
- [LLM-Driven Agentic Workflows](llm-agentic-workflows.md)
- [Haystack AI Framework](haystack-ai-framework.md)