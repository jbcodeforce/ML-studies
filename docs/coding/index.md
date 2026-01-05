# Coding

???- info "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

    09/10/2023: Add PyTorch

    12/2023: Clean Jupyter

    01/2026: Migrate to uv for package management

## Environments

### uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust. It replaces pip, pip-tools, pipx, poetry, pyenv, virtualenv, and more.

**Installation:**

```sh
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

**Project setup for examples and demo as subfolder within this project:**

```sh
# Initialize a new project with pyproject.toml
uv init

# Create virtual environment and install dependencies
uv sync

# Add a dependency
uv add numpy pandas torch

# Run a script
uv run python script.py

# Run Jupyter
uv run jupyter lab
```

**Quick environment for existing projects:**

```sh
# Create venv and install from pyproject.toml or requirements.txt
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

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

* See [Notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/MatPlotLib.ipynb)

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

Example of getting started code in deep-neural-net folder. 

[Summary of the library and deeper studies](./pytorch.md)

## Code Samples

Code is organized in three main folders: `examples/` for library-specific samples, `e2e-demos/` for end-to-end applications, and `notebooks/` for Jupyter notebooks.

### Machine Learning Classifiers

Located in `examples/ml-python/classifiers/`:

| Code | Description |
| --- | --- |
| [TestPerceptron.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/ml-python/classifiers/TestPerceptron.py) | Perceptron classifier for iris flowers using identity activation |
| [TestAdaline.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/ml-python/classifiers/TestAdaline.py) | ADAptive LInear NEuron with linear activation function |
| [SVM-IRIS.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/ml-python/classifiers/SVM-IRIS.py) | Support Vector Machine on iris dataset |
| [DecisionTreeIRIS.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/ml-python/classifiers/DecisionTreeIRIS.py) | Decision tree classification |
| [demo_lasso_ridge.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/ml-python/demo_lasso_ridge.py) | L1/L2 regularization comparison |

### PyTorch Deep Learning

Located in `examples/pytorch/`:

| Code | Description |
| --- | --- |
| [get_started/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/pytorch/get_started) | Tensor basics, workflow notebooks |
| [classification/classifier.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/examples/pytorch/classification/classifier.ipynb) | Binary classification with neural networks |
| [classification/multiclass-classifier.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/examples/pytorch/classification/multiclass-classifier.ipynb) | Multi-class classification |
| [computer-vision/fashion_cnn.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/pytorch/computer-vision/fashion_cnn.py) | CNN on Fashion MNIST |
| [computer-vision/transfer_learning.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/pytorch/computer-vision/transfer_learning.py) | Transfer learning with EfficientNet |
| [ddp/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/pytorch/ddp) | Distributed Data Parallel training |

### LangChain and LLM Integration

Located in `examples/llm-langchain/`:

| Code | Description |
| --- | --- |
| [openai/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/openai) | OpenAI API integration, agents, streaming |
| [anthropic/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/anthropic) | Claude integration |
| [bedrock/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/bedrock) | AWS Bedrock with CoT prompts |
| [mistral/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/mistral) | Mistral AI tool calling |
| [gemini/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/gemini) | Google Gemini chat |
| [cohere/](https://github.com/jbcodeforce/ML-studies/tree/master/examples/llm-langchain/cohere) | Cohere integration |

### RAG Implementations

Located in `examples/llm-langchain/rag/`:

| Code | Description |
| --- | --- |
| [build_agent_domain_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/rag/build_agent_domain_rag.py) | Build RAG with ChromaDB and OpenAI |
| [multiple_queries_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/rag/multiple_queries_rag.py) | Multi-query RAG pattern |
| [rag_fusion.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/rag/rag_fusion.py) | RAG fusion with reciprocal rank |
| [rag_hyde.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/rag/rag_hyde.py) | Hypothetical Document Embedding |

### LangGraph Agent Patterns

Located in `examples/llm-langchain/langgraph/`:

| Code | Description |
| --- | --- |
| [first_graph_with_tool.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/first_graph_with_tool.py) | Basic graph with tool calling |
| [react_lg.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/react_lg.py) | ReAct pattern implementation |
| [adaptive_rag.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/adaptive_rag.py) | Adaptive RAG with routing |
| [human_in_loop.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/human_in_loop.py) | Human-in-the-loop pattern |
| [ask_human_graph.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/ask_human_graph.py) | Human approval workflow |
| [stream_agent_node.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-langchain/langgraph/stream_agent_node.py) | Streaming agent output |

### Ollama Local LLM

Located in `examples/llm-ollama/`:

| Code | Description |
| --- | --- |
| [chat_with_mistral.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-ollama/chat_with_mistral.py) | Chat with local Mistral |
| [async_chat_with_mistral.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-ollama/async_chat_with_mistral.py) | Async chat streaming |
| [chat_with_ollama_openai_api.py](https://github.com/jbcodeforce/ML-studies/blob/master/examples/llm-ollama/chat_with_ollama_openai_api.py) | Ollama with OpenAI-compatible API |

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

### Jupyter Notebooks

Located in `notebooks/`:

| Notebook | Topic |
| --- | --- |
| [ConditionalProbabilityExercise.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/ConditionalProbabilityExercise.ipynb) | Probability and Bayes |
| [Distributions.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Distributions.ipynb) | Statistical distributions |
| [LinearRegression.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/LinearRegression.ipynb) | Linear regression basics |
| [KNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/KNN.ipynb) | K-Nearest Neighbors |
| [DecisionTree.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/DecisionTree.ipynb) | Decision tree classifier |
| [KMeans.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/KMeans.ipynb) | K-Means clustering |
| [PCA.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/PCA.ipynb) | Principal Component Analysis |
| [Keras-CNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Keras-CNN.ipynb) | CNN with Keras |
| [Keras-RNN.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/Keras-RNN.ipynb) | RNN with Keras |