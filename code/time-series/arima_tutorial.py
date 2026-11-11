import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
This tutorial consider the statsmodels macrodata dataset. It collects various macro data recorded quarterly from 1959 to 2009.
We will explore the dataset, check for stationarity, and fit an ARIMA model to one of the time series.

See source: https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressive_integrated_moving_average.html
"""
from statsmodels.datasets import macrodata

# ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# ADF test
from statsmodels.tsa.stattools import adfuller

sns.set_style("darkgrid")

# Load the macrodata dataset
data = macrodata.load_pandas().data
# select the realgdp ts
ts = data[["year", "quarter", "realgdp"]]
# build a quarterly period index
ts.index = pd.PeriodIndex(
    ts["year"].astype(int).astype(str) + "Q" + ts["quarter"].astype(int).astype(str),
    freq="Q",
)
ts.index.name = "quarter"
# drop old columns not needed
ts = ts.drop(columns=["year", "quarter"])
print(ts.head(2))
# ts summary statistics
print("\nReal GDP summary statistics:")
print(ts.describe().map("{:.2f}".format))
print("""
# The time-series has 203 observation, a median of 6559.59 and an average of 7221.17.
# Since the median and the average are close, probably there are not outliers in the series.
""")
# set the index as timestamp for plotting
ts_plot = ts.to_timestamp()
# plot ts
plt.plot(ts_plot)
plt.title("Real GDP: quarterly frequence")
plt.xlabel("Year")
plt.ylabel("Real GDP")
plt.show()
"""
As the chart highlights, there is no clear seasonal pattern and an evident upward trend in real GDP
This suggests the time series is not-stationary, and it needs differencing 
(of at least order = 1) to become stationary. That is, d is likely to be >= 1.
"""

# Define training and test sets
# define the training %
split_index = int(len(ts) * 0.90)
# training set
train = ts.iloc[:split_index]
# validation set
val = ts.iloc[split_index:]

"""
ARIMA requires specifying (p,d,q).

Augmented Dickey-Fuller (ADF) test, autocorrelation (ACF) and partial autocorrelation (PACF) 
plots help in determining them. However, the most efficient approach is to set up the 
simplest ARIMA configuration and eventually adding complexity by trial-and-errors.
"""

# Apply first differencing
ts_diff = train.diff().dropna()
# Augmented Dickey-Fuller test
adf_test = adfuller(ts_diff, result_object=True)
# show p-values
print(f"Augmented Dickey-Fuller p-value: {adf_test.pvalue:.6f}")
print("The series becomes stationary by differencing by 1, as the ADF test and the plot remark. Therefore d = 1.")

"""
We now consider the auto-correlation ACF and partial auto-correlation PACF plots to assess p and q 
maximum values. ACF helps in identifying MA(q), while PACF helps in defining AR(p).
"""

print("plot autocorrelation function (ACF)")
plot_acf(ts_diff, lags=30)
plt.show()
print("plot partial autocorrelation function (PACF)")
plot_pacf(ts_diff, lags=30)
plt.show()

print("""
Both ACF and PACF show significant spikes at the first few lags, suggesting the presence 
of a short-term correlation. Altogether, the plots highlighting a time-series with an 
initial correlation that lose importance after the very first few lags.
""")

# define the model. Adding drift (trend = “t”) allows the model to capture long-term growth.
arima = ARIMA(train, order=(1, 1, 1), trend="t")
# fit the model
arima_fit = arima.fit()
print(arima_fit.summary())

"""
From the Arima summary table, we can see that at the top-left, we have the variable and the model. 
On the top-right: metrics to assess performances and used for comparison.
In the middle:variables’ coefficent, along with their standard error, z-statistic, p-values and 95% confidence interval. 
In this specific case, x1 is the constant term (drift), 
ar.L1 is the autoregressive term and sigma2 is the variance of residuals (i.e error)

At the bottom: diagnostics tests used to evaluate properties as errors autocorrelation, 
normality and homoskedasticity, allowing to evaluate the overall model’s reliability.
"""

print("\nRunning forecasting...")
# predict real GDP values (avg + confidence interval)
predictions = arima_fit.get_forecast(steps=len(val))
# get point prediction
predictions_mean = predictions.predicted_mean
# build the confidence interval
conf_int = predictions.conf_int()
# show prediction
print("Real GDP predictions:")
print("-------------------------------------")
print(predictions_mean)
print("\nassess model performances by comparing prediction with actual values")
print("-------------------------------------")
val_series = val.squeeze()
# compute rmse
rmse = np.sqrt(np.mean((val_series - predictions_mean.values) ** 2))
# Note: Ensure predictions and actual values are aligned before computing errors
# show it
print("Validation RMSE:")
print("----------------")
print(f"{rmse: .3f}")
print("The RMSE corresponds to roughly 3.5% of the average GDP level")

# Convert train to timestamp
train_plot = train.to_timestamp()
# Convert val to timestamp
val_plot = val.to_timestamp()
# Align the prediction series w/ validation
predictions_mean.index = val_plot.index
# Align the CI w/ validation
conf_int.index = val_plot.index
# keep only last 30 points
train_tail = train_plot[-30:]
# Bridge first validation points to make plot continuous
bridge_point = val_plot.iloc[:1]
train_tail = pd.concat([train_tail, bridge_point])


plt.figure(figsize=(12, 6))
# last 30 train points
plt.plot(train_tail, label="Train", color="blue")
# actual validation points
plt.plot(val_plot, label="Actual", color="orange")
# forecast
plt.plot(predictions_mean, label="Forecast", color="red")
# confidence interval
plt.fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="pink",
    alpha=0.3,
    label="Confidence Interval",
)
# labels & title
plt.title("GDP ARIMA Forecast vs Actual")
plt.xlabel("Year")
plt.ylabel("Real GDP")
plt.legend()
plt.show()

print("""
As the RMSE and the actual-vs-predicted plot reveal, the model captures the real-GDP 
upward trend reasonably well. However, it fails in catching some short-term fluctuations 
along with some temporal dynamics
""")