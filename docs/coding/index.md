---
title: "Coding"
source: local-import
ingested: 2026-06-19
tags: []
type: article
compiled: false
---

# Coding

???- info "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

    09/10/2023: Add PyTorch

    12/2023: Clean Jupyter

    01/2026: Migrate to uv for package management

    07/2026: Reorganize code by subject under `code/`; two venvs (code + jupyter)

## Environments

Two virtual environments: **code** for Python/PyTorch scripts, **jupyter** for notebook kernels.

### Code / PyTorch (`code/.venv`)

```sh
cd code
uv venv .venv
source .venv/bin/activate
uv sync --extra pytorch          # PyTorch and computer vision
uv sync --extra llm --extra agents   # LLM and agent scripts
```

See [code/SUBJECTS.md](https://github.com/jbcodeforce/ML-studies/blob/master/code/SUBJECTS.md) for subject folder mapping.

### Jupyter (`jupyter/.venv`)

```sh
cd jupyter
uv venv .venv
source .venv/bin/activate
uv sync --extra deep-learning     # optional: Keras/TensorFlow notebooks
uv run python -m ipykernel install --user --name ml-studies-jupyter
```

Notebooks live under `code/<subject>/` but use the **jupyter** kernel.

### uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust. It replaces pip, pip-tools, pipx, poetry, pyenv, virtualenv, and more.

**Installation:**

```sh
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

**Project setup:**

```sh
# Code environment (scripts, PyTorch, LLM)
cd code && uv venv .venv && uv sync --extra pytorch

# Jupyter environment (notebooks under code/<subject>/)
cd jupyter && uv venv .venv && uv sync
```

**Quick commands:**

```sh
uv run python script.py
uv run jupyter lab
```

Subject folders are documented in [code/SUBJECTS.md](https://github.com/jbcodeforce/ML-studies/blob/master/code/SUBJECTS.md).

### Legacy per-folder setup (deprecated)

Each project folder contains a `pyproject.toml` for dependency management.

### VSCode

#### [Jupyter Notebook](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

* To select an environment, use the `Python: Select Interpreter` command from the Command Palette (⇧⌘P)
* Use `Create: New Jupyter Notebook` from command Palette
* Select a kernel using the kernel picker in the top right.
* Within a Python Notebook, it's possible to view, inspect, sort, and filter the variables within the current Jupyter session, using `Variables` in toolbar.
* We can offload intensive computation in a Jupyter Notebook to other computers by connecting to a remote Jupyter server. Use server URL with security token.

### Run Kaggle image

As an alternate Kaggle has a more complete [docker image](https://github.com/Kaggle/docker-python) to start with. 

```sh
# CPU based
docker run --rm -v $(pwd):/home -it gcr.io/kaggle-images/python /bin/bash
# GPU based
docker run -v $(pwd):/home --runtime nvidia --rm -it gcr.io/kaggle-gpu-images/python /bin/bash
```

## Important Python Libraries

### [numpy](https://numpy.org/)

* Array computing in Python. [Numpy official quickstar.t](https://numpy.org/devdocs/user/quickstart.html)
* NumPy dimensions are called axes.
    ```python
    import numpy as np
    a = np.array([2, 3, 4])
    b = np.ones((2, 3, 4), dtype=np.int16)
    c = np.zeros((3, 4))
    ```

* Create a sequence of number: `np.arange(10, 30, 5)`
* Matrix product: using .dot or @
    ```python
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[2, 0], [3, 4]])
    A @ B
    A.dot(B)
    ```

### [scipy](https://scipy.org/)

SciPy is a collection of mathematical algorithms and convenience functions built on top of NumPy. See [product documentation](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide).

* Get a normal distribution function: use the probability density function (pdf)
    ```python
    from scipy.stats import norm
    x = np.arange(-3, 3, 0.01)
    y=norm.pdf(x)
    ```

### [MatPlotLib](https://matplotlib.org/stable/users/index.html)

Persent figure among multiple axes, from the data for human analysis.


* Classic import
    ```
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib as mpl
    ```

* See [Notebook](https://github.com/jbcodeforce/ML-studies/blob/master/code/shared/MatPlotLib.ipynb)

### [Seaborn](https://seaborn.pydata.org/)

Seaborn provides a high-level interface for drawing attractive and informative statistical graphics. Based on top of MatPlotLib and integrated with Pandas.

[See the introduction for different examples](https://seaborn.pydata.org/tutorial/introduction.html)


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.relplot(
    data=masses_data,
    x="age", y="shape", 
    hue="density", size="density"
)
plt.show()
```

### PyTorch

Via conda or pip, install `pytorch torchvision torchaudio`.

Example of getting started code in `code/deep-learning/fundamentals/`.

[Summary of the library and deeper studies](./pytorch.md)

## Code Samples

Code is organized by **subject** under [`code/`](https://github.com/jbcodeforce/ML-studies/tree/master/code): scripts and notebooks live together per topic. See [SUBJECTS.md](https://github.com/jbcodeforce/ML-studies/blob/master/code/SUBJECTS.md). End-to-end demos remain in `e2e-demos/`.

### Perceptron

Located in `code/perceptron/`:

| Code | Description |
| --- | --- |
| [test_perceptron.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/perceptron/test_perceptron.py) | Perceptron classifier for iris flowers using identity activation |

### Classification

Located in `code/classification/`:

| Code | Description |
| --- | --- |
| [test_adaline.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/test_adaline.py) | ADAptive LInear NEuron with linear activation function |
| [svm_iris.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/svm_iris.py) | Support Vector Machine on iris dataset |
| [decision_tree_iris.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/decision_tree_iris.py) | Decision tree classification |

### Regression

Located in `code/regression/`:

| Code | Description |
| --- | --- |
| [demo_lasso_ridge.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/regression/demo_lasso_ridge.py) | L1/L2 regularization comparison |
| [classify_with_pipe.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/logistic-regression/classify_with_pipe.py) | Logistic regression pipeline (`code/logistic-regression/`) |

### PyTorch Deep Learning

Located in `code/deep-learning/` and `code/computer-vision/`:

| Code | Description |
| --- | --- |
| [get_started/](https://github.com/jbcodeforce/ML-studies/tree/master/code/deep-learning/get_started) | Tensor basics, workflow notebooks |
| [classifications.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/classifications.ipynb) | Binary classification with neural networks |
| [multiclass-classifier.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/multiclass-classifier.ipynb) | Multi-class classification |
| [fashion_cnn.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/computer-vision/fashion_cnn.py) | CNN on Fashion MNIST |
| [transfer_learning.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/computer-vision/transfer_learning.py) | Transfer learning with EfficientNet |
| [ddp/](https://github.com/jbcodeforce/ML-studies/tree/master/code/deep-learning/ddp) | Distributed Data Parallel training |

### LangChain and LLM Integration

Located in `code/LLM/langchain/`:

| Code | Description |
| --- | --- |
| [openai/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/openai) | OpenAI API integration, agents, streaming |
| [anthropic/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/anthropic) | Claude integration |
| [bedrock/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/bedrock) | AWS Bedrock with CoT prompts |
| [mistral/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/mistral) | Mistral AI tool calling |
| [gemini/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/gemini) | Google Gemini chat |
| [cohere/](https://github.com/jbcodeforce/ML-studies/tree/master/code/LLM/langchain/cohere) | Cohere integration |

### RAG Implementations

Located in `code/LLM/langchain/rag/`:

| Code | Description |
| --- | --- |
| [build_agent_domain_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/langchain/rag/build_agent_domain_rag.py) | Build RAG with ChromaDB and OpenAI |
| [multiple_queries_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/langchain/rag/multiple_queries_rag.py) | Multi-query RAG pattern |
| [rag_fusion.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/langchain/rag/rag_fusion.py) | RAG fusion with reciprocal rank |
| [rag_hyde.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/langchain/rag/rag_hyde.py) | Hypothetical Document Embedding |

### LangGraph Agent Patterns

Located in `code/agents/langgraph/`:

| Code | Description |
| --- | --- |
| [first_graph_with_tool.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/first_graph_with_tool.py) | Basic graph with tool calling |
| [react_lg.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/react_lg.py) | ReAct pattern implementation |
| [adaptive_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/adaptive_rag.py) | Adaptive RAG with routing |
| [human_in_loop.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/human_in_loop.py) | Human-in-the-loop pattern |
| [ask_human_graph.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/ask_human_graph.py) | Human approval workflow |
| [stream_agent_node.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/agents/langgraph/stream_agent_node.py) | Streaming agent output |

### Ollama Local LLM

Located in `code/LLM/ollama/`:

| Code | Description |
| --- | --- |
| [chat_with_mistral.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/ollama/chat_with_mistral.py) | Chat with local Mistral |
| [async_chat_with_mistral.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/ollama/async_chat_with_mistral.py) | Async chat streaming |
| [chat_with_ollama_openai_api.py](https://github.com/jbcodeforce/ML-studies/blob/master/code/LLM/ollama/chat_with_ollama_openai_api.py) | Ollama with OpenAI-compatible API |

### End-to-End Demos

Located in `e2e-demos/`:

| Demo | Description |
| --- | --- |
| [qa_retrieval/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/qa_retrieval) | Q&A with RAG and ChromaDB |
| [chat_with_pdf/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/chat_with_pdf) | PDF document chat application |
| [streaming-demo/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/streaming-demo) | WebSocket streaming with LangGraph |
| [resume_tuning/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/resume_tuning) | Resume optimization with LLM |
| [think_deeply/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/think_deeply) | Deep reasoning with LLM |
| [gemini_cmd/](https://github.com/jbcodeforce/ML-studies/tree/master/e2e-demos/gemini_cmd) | Gemini CLI integration |

### UI Frameworks

Located in `techno/`:

| Framework | Code |
| --- | --- |
| CrewAI | [techno/crew-ai/](https://github.com/jbcodeforce/ML-studies/tree/master/techno/crew-ai) - Multi-agent examples |
| Streamlit | [techno/streamlit/](https://github.com/jbcodeforce/ML-studies/tree/master/techno/streamlit) - Dashboard apps |
| Gradio | [techno/gradio/](https://github.com/jbcodeforce/ML-studies/tree/master/techno/gradio) - ML interfaces |
| NiceGUI | [techno/nicegui/](https://github.com/jbcodeforce/ML-studies/tree/master/techno/nicegui) - Python web UI |
| Taipy | [techno/taipy/](https://github.com/jbcodeforce/ML-studies/tree/master/techno/taipy) - Data apps |

### Jupyter Notebooks by subject

Notebooks live under `code/<subject>/`. Use the [jupyter environment](https://github.com/jbcodeforce/ML-studies/blob/master/jupyter/README.md) kernel.

| Notebook | Topic |
| --- | --- |
| [ConditionalProbabilityExercise.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/statistics/ConditionalProbabilityExercise.ipynb) | Probability and Bayes |
| [Distributions.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/statistics/Distributions.ipynb) | Statistical distributions |
| [LinearRegression.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/regression/LinearRegression.ipynb) | Linear regression basics |
| [KNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/KNN.ipynb) | K-Nearest Neighbors |
| [DecisionTree.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/classification/DecisionTree.ipynb) | Decision tree classifier |
| [KMeans.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/unsupervised/KMeans.ipynb) | K-Means clustering |
| [PCA.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/unsupervised/PCA.ipynb) | Principal Component Analysis |
| [Keras-CNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/computer-vision/Keras-CNN.ipynb) | CNN with Keras |
| [Keras-RNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/code/deep-learning/Keras-RNN.ipynb) | RNN with Keras |
