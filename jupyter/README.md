# Jupyter Environment

Virtual environment for notebooks located under `code/<subject>/`.

## Setup

```sh
cd jupyter
uv venv .venv
source .venv/bin/activate
uv sync --extra pytorch        # required for PyTorch notebooks (classification, computer-vision, deep-learning)
uv sync --extra deep-learning   # optional: TF/Keras notebooks
uv sync --extra spark           # optional: Spark scripts
uv run python -m ipykernel install --user --name ml-studies-jupyter
```

In VS Code or Jupyter, select the **ml-studies-jupyter** kernel when opening notebooks under `code/`.

Runnable `.py` scripts use `code/.venv` instead. See [code/SUBJECTS.md](../code/SUBJECTS.md).

## Verify notebooks

```sh
uv run python verify_notebooks.py
```
