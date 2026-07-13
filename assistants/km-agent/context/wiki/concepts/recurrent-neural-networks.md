---
title: "Recurrent Neural Networks"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [deep-learning, neural-networks, nlp]
code: [code/deep-learning/]
tags: [rnn, lstm, sequential-data, time-series, nlp, deep-learning]
---

# Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are designed to process sequential data where the order of inputs matters. Unlike feedforward networks, RNNs maintain a hidden state that captures information about previous inputs in the sequence.

## The Vanishing Gradient Problem

Standard RNNs suffer from the vanishing gradient problem, making it difficult to learn long-range dependencies in sequences. Gradients become extremely small during backpropagation through many time steps, preventing effective weight updates for earlier inputs.

## Long Short-Term Memory (LSTM)

LSTM networks address the vanishing gradient problem with a sophisticated gating mechanism that controls information flow through the network:

- **Forget gate**: Decides what information to discard from the cell state
- **Input gate**: Decides which new values to update into the cell state
- **Output gate**: Decides what to output based on the current cell state

### Example LSTM Model

```python
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

## Use Cases

- **Sentiment analysis**: Classifying text (e.g., movie reviews) as positive or negative based on word sequences
- **Time series prediction**: Forecasting stock prices, weather patterns, or sensor data
- **Language modeling**: Predicting the next word in a sequence
- **Speech recognition**: Converting audio sequences to text

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Deep Learning](deep-learning.md)
- [Neural Networks](neural-networks.md)