# Code by Subject

Canonical mapping of subjects to folders, documentation, and environments.

| Subject | Folder | Documentation | Venv |
| --- | --- | --- | --- |
| Perceptron | [perceptron/](perceptron/) | [classifier.md](../docs/ml/classifier.md) | `code/.venv` |
| Logistic regression | [logistic-regression/](logistic-regression/) | [classifier.md#logistic-regression](../docs/ml/classifier.md) | `code/.venv` |
| Classification | [classification/](classification/) | [classifier.md](../docs/ml/classifier.md) | `jupyter/.venv` for notebooks |
| Regression | [regression/](regression/) | [classifier.md](../docs/ml/classifier.md) | `jupyter/.venv` |
| Computer vision | [computer-vision/](computer-vision/) | [deep-learning.md](../docs/ml/deep-learning.md) | `code/.venv --extra pytorch` |
| Deep learning | [deep-learning/](deep-learning/) | [deep-learning.md](../docs/ml/deep-learning.md) | `code/.venv --extra pytorch` |
| Unsupervised | [unsupervised/](unsupervised/) | [unsupervised.md](../docs/ml/unsupervised.md) | `jupyter/.venv` |
| Anomaly detection | [anomaly-detection/](anomaly-detection/) | [anomaly.md](../docs/anomaly.md) | `code/.venv` |
| Statistics / EDA | [statistics/](statistics/) | [concepts/maths.md](../docs/concepts/maths.md) | `jupyter/.venv` |
| NLP | [nlp/](nlp/) | [nlp.md](../docs/ml/nlp.md) | `jupyter/.venv` |
| LLM | [LLM/](LLM/) | [genAI/](../docs/genAI/index.md) | `code/.venv --extra llm` |
| Agents | [agents/](agents/) | [agentic.md](../docs/genAI/agentic.md) | `code/.venv --extra agents` |
| Shared utilities | [shared/](shared/) | [coding/index.md](../docs/coding/index.md) | either |

## GitHub link pattern

```
https://github.com/jbcodeforce/ML-studies/blob/master/code/<subject>/<file>
```

## Environments

- **Code / PyTorch**: `cd code && uv venv .venv && uv sync --extra pytorch`
- **Jupyter notebooks**: `cd jupyter && uv venv .venv && uv sync`

Notebooks under `code/<subject>/` use the **jupyter** kernel. Runnable `.py` scripts use **code/.venv**.
