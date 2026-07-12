# Deep Learning

PyTorch fundamentals, Keras/TensorFlow, distributed training, and LLM fine-tuning.

## Layout

| Path | Description |
| --- | --- |
| `get_started/` | Tensor basics, workflow notebooks |
| `ddp/` | Distributed Data Parallel training |
| `neuralnetwork/` | Keras CNN architectures (LeNet, AlexNet) |
| `tensorflow/` | TensorFlow intro |
| `fundamentals/` | Sigmoid, plotting, LLM fine-tuning |

## Notebooks

`Keras.ipynb`, `Keras-RNN.ipynb`, `Tensorflow.ipynb`, `DeepLearningProject-Solution.ipynb`, `FinalProjectAssignment.ipynb`

## Environment

```sh
cd code && uv sync --extra pytorch --extra deep-learning
```

See [coding/ddp.md](../../docs/coding/ddp.md) for distributed training.
