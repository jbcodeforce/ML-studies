# Time Series Forecasting with ML

This note is about time series analytics and neural network forecasting, from statistical baselines through recurrent networks, transformers, and time-series foundation models.

## 1. Core Foundations & Classical Analytics

In general neural networks often overfit when applied naively to series where a simple statistical baseline or tree-based model would perform better. The following techniques should be considered:

* Decomposition & Stationarity: Deconstruct series into trend, seasonality, and residual components (additive vs. multiplicative). Test for unit roots using the Augmented Dickey-Fuller (ADF) test or KPSS test to determine whether differencing or log transforms are needed.
* Autocorrelation Analysis: Plot Auto-Correlation Functions (ACF) and Partial Auto-Correlation Functions (PACF) to identify autoregressive and moving average properties, which also determine your lookback window length ($p$).
* Baseline Benchmarks: Always evaluate deep models against a naive drift, Seasonal Naive, Exponential Smoothing (Holt-Winters), and SARIMAX.
* Tree-Based Feature Engineering: Gradient-boosted decision trees (LightGBM, XGBoost, CatBoost) with engineered lag features, rolling statistics (mean, standard deviation over 7/14/30 days), and calendar Fourier terms frequently match or outperform neural networks on tabular time series.

## 2. Neural Network Architectures for Forecasting

Deep learning excels primarily when dealing with high-dimensional multivariate series, non-linear cross-series dependencies, long lookback horizons, or cold-start scenarios where one global model trains across thousands of interrelated time series.

| Architecture | Representative Models | Primary Strengths | Best Use Cases |
| --- | --- | --- | --- |
| Recurrent (RNNs) | LSTM, GRU, xLSTM | Explicit sequential memory states; natural handling of step-by-step inputs | Sensor telemetry, medium-horizon univariate/multivariate series |
| Convolutional (CNNs) | Temporal Convolutional Networks (TCN) | Dilated causal 1D convolutions; parallelizable training; no vanishing gradients | Multi-scale temporal patterns with large historical windows |
| Decomposition MLPs | N-BEATS, N-HiTS | Extremely fast; interpretable trend/seasonality outputs; no recurrent loops | Fast univariate forecasting across millions of series |
| Transformer-Based | PatchTST, iTransformer, Informer | Tokenizes time patches rather than single steps; extracts long-range correlations | Long-horizon multivariate forecasting with complex cross-series signals |
| Zero-Shot Foundation Models | Chronos (T5-based), TimesFM, Lag-Llama | Pre-trained across open time series; zero-shot and few-shot inference | Rapid baseline generation and cold-start datasets |

## 3. 3. Validation Strategy & Practical Toolstack

Time series data violates the standard machine learning assumption of independent and identically distributed samples. Rigorous evaluation requires temporal discipline.

1. Strict Temporal Splitting: 
Never use random train/test splits or standard $K$-fold cross-validation, as they introduce severe lookahead bias (data leakage). Use rolling-origin cross-validation (expanding or sliding windows).
1. Scaler Hygiene: When applying MinMax or Standard scaling, fit the scaler only on the training window and transform the test window using those fitted parameters.
1. Loss Functions & Metrics:
    * Point Forecasts: Use Mean Absolute Error (MAE) for robustness against outliers, Root Mean Squared Error (RMSE) to penalize large deviations, and Weighted Absolute Percentage Error (WAPE) for intermittent or zero-heavy series
    * Probabilistic Forecasts: Train models with Quantile Loss (pinball loss) to output prediction intervals ($p_{10}, p_{50}, p_{90}$) instead of a single deterministic number.
1. Core Python Ecosystem:
    * Data Handling & EDA: pandas, polars, statsmodels.
    * Unified Modeling Frameworks: Darts (covers ARIMA to PyTorch Transformers under an intuitive Scikit-Learn API) and Nixtla NeuralForecast (scalable implementations of N-BEATS, PatchTST, and TimesFM).
    * Deep Learning Frameworks: PyTorch and PyTorch Lightning for custom sequence-to-sequence or encoder-decoder architectures.
    
To see a hands-on implementation of recurrent networks for sequential prediction, see the [RNN and LSTM Time Series Forecasting Tutorial](https://www.youtube.com/watch?v=S8tpSG6Q2H0&t=6s). It provides a practical walk-through of structuring sliding window tensors and training recurrent models on sequential data using Python.