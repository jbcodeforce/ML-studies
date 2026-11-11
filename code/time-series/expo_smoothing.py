"""
Exponential Smoothing: SES, Holt, Holt-Winters, and Simulations

This script demonstrates various exponential smoothing techniques from statsmodels:
  1. Loading data (5 datasets: oildata, air, livestock2, livestock3, aust)
  2. Simple Exponential Smoothing (oildata, 3 variants)
  3. Holt's Method (air, 3 variants: linear trend, exponential trend, damped trend)
  4. Seasonally adjusted data (livestock2, 5 fits + param table)
  5. Plots of seasonally adjusted data (level/slope of fit2 and fit4)
  6. Comparison (livestock2 + livestock3, 5 models side by side)
  7. Holt-Winters Seasonal (aust, 4 fits with table + forecast plot)
  8. Internals (aust, fit1/fit2 without Box-Cox, y_t, l_t, b_t, s_t, fitted + forecast)
  9. Simulation and confidence intervals (2 variants of simulate)

See source: https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
See: Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing


# ============================================================================ #
# 1. Loading Data
# ============================================================================ #

# ---- Oil production in Saudi Arabia ----
data = [
    446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325,
    485.1494, 506.0482, 526.792, 514.2689, 494.211,
]
index = pd.date_range(start="1996", end="2008", freq="YE")
oildata = pd.Series(data, index)

# ---- Air pollution data ----
data = [
    17.5534, 21.86, 23.8866, 26.9293, 26.8885, 28.8314, 30.0751,
    30.9535, 30.1857, 31.5797, 32.5776, 33.4774, 39.0216,
    41.3864, 41.5966,
]
index = pd.date_range(start="1990", end="2005", freq="YE")
air = pd.Series(data, index)

# ---- Livestock (sheep in Asia) ----
data = [
    263.9177, 268.3072, 260.6626, 266.6394, 277.5158, 283.834, 290.309,
    292.4742, 300.8307, 309.2867, 318.3311, 329.3724, 338.884,
    339.2441, 328.6006, 314.2554, 314.4597, 321.4138, 329.7893,
    346.3852, 352.2979, 348.3705, 417.5629, 417.1236, 417.7495,
    412.2339, 411.9468, 394.6971, 401.4993, 408.2705, 414.2428,
]
index = pd.date_range(start="1970", end="2001", freq="YE")
livestock2 = pd.Series(data, index)

# ---- Livestock 3 ----
data = [407.9979, 403.4608, 413.8249, 428.105, 445.3387, 452.9942, 455.7402]
index = pd.date_range(start="2001", end="2008", freq="YE")
livestock3 = pd.Series(data, index)

# ---- International visitor nights in Australia (quarterly) ----
data = [
    41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829,
    42.9777, 48.9015, 31.1802, 37.7179, 40.4202, 51.2069,
    31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764,
    45.6423, 59.7668, 35.1919, 44.3197, 47.9137,
]
index = pd.period_range(start="2005Q1", end="2010Q4", freq="Q-OCT")
aust = pd.Series(data, index=index)


# ============================================================================ #
# 2. Simple Exponential Smoothing
# ============================================================================ #

def simple_exp_smoothing(dta: pd.Series, name: str = "oildata"):
    """Forecast with Simple Exp. Smoothing (3 variants of alpha)."""
    fit1 = SimpleExpSmoothing(dta, initialization_method="heuristic").fit(
        smoothing_level=0.2, optimized=False
    )
    fcast1 = fit1.forecast(3).rename(r"$\alpha=0.2$")

    fit2 = SimpleExpSmoothing(dta, initialization_method="heuristic").fit(
        smoothing_level=0.6, optimized=False
    )
    fcast2 = fit2.forecast(3).rename(r"$\alpha=0.6$")

    fit3 = SimpleExpSmoothing(dta, initialization_method="estimated").fit()
    fcast3 = fit3.forecast(3).rename(
        r"$\alpha={:.3f}".format(fit3.model.params["smoothing_level"])
    )

    ax = dta.plot(marker="o", color="black", figsize=(12, 8))
    ax.plot(fit1.fittedvalues, marker="o", color="blue")
    (line1,) = ax.plot(fcast1, marker="o", color="blue")
    ax.plot(fit2.fittedvalues, marker="o", color="red")
    (line2,) = ax.plot(fcast2, marker="o", color="red")
    ax.plot(fit3.fittedvalues, marker="o", color="green")
    (line3,) = ax.plot(fcast3, marker="o", color="green")

    ax.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])
    ax.set_xlabel("Year")
    ax.set_ylabel(name + " (millions of tonnes)")
    ax.set_title("Simple Exponential Smoothing Forecasts")
    return fit1, fit2, fit3


# ============================================================================ #
# 3. Holt's Method
# ============================================================================ #

def holt_method(air_data: pd.Series):
    """Fit Holt's method (3 variants: linear, exponential, damped)."""
    # Linear trend with explicit params
    fit1 = Holt(air_data, initialization_method="estimated").fit(
        smoothing_level=0.8, smoothing_trend=0.2, optimized=False
    )
    fcast1 = fit1.forecast(5).rename("Holt's linear trend")

    # Exponential trend
    fit2 = Holt(air_data, exponential=True, initialization_method="estimated").fit(
        smoothing_level=0.8, smoothing_trend=0.2, optimized=False
    )
    fcast2 = fit2.forecast(5).rename("Exponential trend")

    # Additive damped trend
    fit3 = Holt(air_data, damped_trend=True, initialization_method="estimated").fit(
        smoothing_level=0.8, smoothing_trend=0.2
    )
    fcast3 = fit3.forecast(5).rename("Additive damped trend")

    ax = air_data.plot(marker="o", color="black", figsize=(12, 8))
    ax.plot(fit1.fittedvalues, color="blue")
    (line1,) = ax.plot(fcast1, marker="o", color="blue")
    ax.plot(fit2.fittedvalues, color="red")
    (line2,) = ax.plot(fcast2, marker="o", color="red")
    ax.plot(fit3.fittedvalues, color="green")
    (line3,) = ax.plot(fcast3, marker="o", color="green")

    ax.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])
    return fit1, fit2, fit3


# ============================================================================ #
# 4. Seasonally Adjusted Data (5 Holt fits on livestock2)
# ============================================================================ #

def holt_seasonal_comparison(liv_data: pd.Series):
    """Fit 5 Holt models and compare parameters in a table."""
    fit1 = SimpleExpSmoothing(liv_data, initialization_method="estimated").fit()
    fit2 = Holt(liv_data, initialization_method="estimated").fit()
    fit3 = Holt(liv_data, exponential=True, initialization_method="estimated").fit()
    fit4 = Holt(
        liv_data, damped_trend=True, initialization_method="estimated"
    ).fit(damping_trend=0.98)
    fit5 = Holt(
        liv_data, exponential=True, damped_trend=True, initialization_method="estimated"
    ).fit()

    params = [
        "smoothing_level",
        "smoothing_trend",
        "damping_trend",
        "initial_level",
        "initial_trend",
    ]
    results = pd.DataFrame(
        index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$l_0$", r"$b_0$", "SSE"],
        columns=["SES", "Holt's", "Exponential", "Additive", "Multiplicative"],
    )
    results["SES"] = [fit1.params[p] for p in params] + [fit1.sse]
    results["Holt's"] = [fit2.params[p] for p in params] + [fit2.sse]
    results["Exponential"] = [fit3.params[p] for p in params] + [fit3.sse]
    results["Additive"] = [fit4.params[p] for p in params] + [fit4.sse]
    results["Multiplicative"] = [fit5.params[p] for p in params] + [fit5.sse]

    return results, fit1, fit2, fit3, fit4, fit5


# ============================================================================ #
# 5. Plots of Seasonally Adjusted Data (level/slope of fit2 and fit4)
# ============================================================================ #

def plot_level_slope_level_slopes(
    fit2, fit4, title: str = "Seasonally Adjusted Data"
):
    """Plot level and slope components for Holt's linear trend and additive damped trend."""
    fig, ax = plt.subplots(figsize=(12, 10), nrows=2, sharex=True)

    # Fit2 (Holt's linear trend)
    pd.DataFrame(np.c_[fit2.level, fit2.trend]).rename(
        columns={0: "level", 1: "slope"}
    ).plot(subplots=True, ax=ax[0])
    ax[0].set_title("Holt's linear trend")

    # Fit4 (Additive damped trend)
    pd.DataFrame(np.c_[fit4.level, fit4.trend]).rename(
        columns={0: "level", 1: "slope"}
    ).plot(subplots=True, ax=ax[1])
    ax[1].set_title("Additive damped trend")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()


# ============================================================================ #
# 6. Comparison: livestock2 + livestock3 (5 models side-by-side)
# ============================================================================ #

def plot_comparison(liv2, liv3):
    """Compare 5 forecasting models across livestock2 and livestock3."""
    fit1 = SimpleExpSmoothing(liv2, initialization_method="estimated").fit()
    fcast1 = fit1.forecast(9).rename("SES")

    fit2 = Holt(liv2, initialization_method="estimated").fit()
    fcast2 = fit2.forecast(9).rename("Holt's")

    fit3 = Holt(liv2, exponential=True, initialization_method="estimated").fit()
    fcast3 = fit3.forecast(9).rename("Exponential")

    fit4 = Holt(
        liv2, damped_trend=True, initialization_method="estimated"
    ).fit(damping_trend=0.98)
    fcast4 = fit4.forecast(9).rename("Additive Damped")

    fit5 = Holt(
        liv2, exponential=True, damped_trend=True, initialization_method="estimated"
    ).fit()
    fcast5 = fit5.forecast(9).rename("Multiplicative Damped")

    ax = liv2.plot(color="black", marker="o", figsize=(12, 8))
    liv3.plot(ax=ax, color="black", marker="o", legend=False)

    fcast1.plot(ax=ax, color="red", legend=True)
    fcast2.plot(ax=ax, color="green", legend=True)
    fcast3.plot(ax=ax, color="blue", legend=True)
    fcast4.plot(ax=ax, color="cyan", legend=True)
    fcast5.plot(ax=ax, color="magenta", legend=True)

    ax.set_ylabel("Livestock, sheep in Asia (millions)")
    plt.show()


# ============================================================================ #
# 7. Holt-Winters Seasonal (aust, 4 fits + table + plot)
# ============================================================================ #

def holt_winters_seasonal(aust_data: pd.Series):
    """Fit 4 Holt-Winters models and compare parameters."""
    fit1 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        use_boxcox=True,
        initialization_method="estimated",
    ).fit()

    fit2 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        use_boxcox=True,
        initialization_method="estimated",
    ).fit()

    fit3 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated",
    ).fit()

    fit4 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated",
    ).fit()

    params = [
        "smoothing_level",
        "smoothing_trend",
        "damping_trend",
        "smoothing_seasonal",
        "initial_level",
        "initial_trend",
    ]
    results = pd.DataFrame(
        index=[
            r"$\alpha$",
            r"$\beta$",
            r"$\phi$",
            r"$\gamma$",
            r"$l_0$",
            r"$b_0$",
            "SSE",
        ]
    )
    results["Additive"] = [fit1.params[p] for p in params] + [fit1.sse]
    results["Multiplicative"] = [fit2.params[p] for p in params] + [fit2.sse]
    results["Additive Dam"] = [fit3.params[p] for p in params] + [fit3.sse]
    results["Multiplica Dam"] = [fit4.params[p] for p in params] + [fit4.sse]

    ax = aust_data.plot(
        figsize=(10, 6),
        marker="o",
        color="black",
        title="Forecasts from Holt-Winters' multiplicative method",
    )
    ax.set_ylabel(
        "International visitor night in Australia (millions)"
    )
    ax.set_xlabel("Year")
    fit1.fittedvalues.plot(ax=ax, style="--", color="red")
    fit2.fittedvalues.plot(ax=ax, style="--", color="green")

    fit1.forecast(8).rename("Holt-Winters (add-add-seasonal)").plot(
        ax=ax, style="--", marker="o", color="red", legend=True
    )
    fit2.forecast(8).rename(
        "Holt-Winters (add-mul-seasonal)"
    ).plot(ax=ax, style="--", marker="o", color="green", legend=True)

    plt.show()
    return results, fit1, fit2, fit3, fit4


# ============================================================================ #
# 8. Internals: y_t, l_t, b_t, s_t, fitted, and forecasts
# ============================================================================ #

def show_internals(aust_data: pd.Series):
    """Show the internals of add-add and add-mul models (no Box-Cox)."""
    fit1 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        initialization_method="estimated",
    ).fit()

    fit2 = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        initialization_method="estimated",
    ).fit()

    for fit, label in [(fit1, "Additive"), (fit2, "Multiplicative")]:
        print(f"\n{'='*50}")
        print(f"Model: {label}")
        print(f"{'='*50}")

        df = pd.DataFrame(
            np.c_[aust_data, fit.level, fit.trend, fit.season, fit.fittedvalues],
            columns=[r"$y_t$", r"$l_t$", r"$b_t$", r"$s_t$", r"$\hat{y}_t$"],
            index=aust_data.index,
        )

        forecasts = fit.forecast(8).rename(r"$\hat{y}_t$").to_frame()
        df = pd.concat([df, forecasts], axis=0, sort=True)

        print(df.to_string(float_format="%.2f"))

        # Plot level, slope, and seasonal components
        states = pd.DataFrame(
            np.c_[fit.level, fit.trend, fit.season],
            columns=["level", "slope", "seasonal"],
            index=aust_data.index,
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        states["level"].plot(ax=axes[0])
        axes[0].set_title(f"Level — {label}")
        states["slope"].plot(ax=axes[1])
        axes[1].set_title(f"Slope — {label}")
        states["seasonal"].plot(ax=axes[2])
        axes[2].set_title(f"Seasonal — {label}")
        plt.suptitle(f"Holt-Winters Internals ({label})")
        plt.tight_layout()
        plt.show()


# ============================================================================ #
# 9. Simulation and Confidence Intervals
# ============================================================================ #

def simulate_forecasts(aust_data: pd.Series):
    """Simulate future values using two variants of the simulate() method."""
    fit = ExponentialSmoothing(
        aust_data,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        initialization_method="estimated",
    ).fit()

    # Variant 1: 8-step ahead, 1000 reps, multiplicative error
    print("Variant 1: 8-step simulations with multiplicative error (random errors)")
    simulations1 = fit.simulate(8, repetitions=1000, error="mul")
    print(simulations1.describe())

    # Plot variant 1
    ax = aust_data.plot(
        figsize=(10, 6),
        marker="o",
        color="black",
        title="Forecasts and simulations from Holt-Winters' multiplicative method",
    )
    ax.set_ylabel(
        "International visitor night in Australia (millions)"
    )
    ax.set_xlabel("Year")
    fit.fittedvalues.plot(ax=ax, style="--", color="green")
    simulations1.plot(ax=ax, style="-", alpha=0.05, color="grey", legend=False)
    fit.forecast(8).rename(
        "Holt-Winters (add-mul-seasonal)"
    ).plot(
        ax=ax, style="--", marker="o", color="green", legend=True
    )
    plt.show()

    # Variant 2: 16-step ahead, anchored at 2009Q1, bootstrap random errors
    print("\nVariant 2: 16-step simulations anchored at 2009Q1 with bootstrap")
    simulations2 = fit.simulate(
        16,
        anchor=pd.Period("2009Q1", freq="Q-OCT"),
        repetitions=100,
        error="mul",
        random_errors="bootstrap",
    )
    print(simulations2.describe())

    # Plot variant 2
    ax = aust_data.plot(
        figsize=(10, 6),
        marker="o",
        color="black",
        title="Forecasts and simulations from Holt-Winters' multiplicative method",
    )
    ax.set_ylabel(
        "International visitor night in Australia (millions)"
    )
    ax.set_xlabel("Year")
    fit.fittedvalues.plot(ax=ax, style="--", color="green")
    simulations2.plot(ax=ax, style="-", alpha=0.05, color="grey", legend=False)
    fit.forecast(8).rename(
        "Holt-Winters (add-mul-seasonal)"
    ).plot(
        ax=ax, style="--", marker="o", color="green", legend=True
    )
    plt.show()


# ============================================================================ #
# Main
# ============================================================================ #

if __name__ == "__main__":
    # 2. Simple Exponential Smoothing
    print("2. Simple Exponential Smoothing (oildata)")
    fit1, fit2, fit3 = simple_exp_smoothing(oildata, name="Oil")
    print("Fit params:", fit3.model.params)
    print()

    # 3. Holt's Method
    print("3. Holt's Method (air)")
    fit1_h, fit2_h, fit3_h = holt_method(air)
    print("Fit params:", fit3_h.model.params)
    print()

    # 4. Seasonally Adjusted Data
    print("4. Seasonally Adjusted Data (livestock2, 5 models)")
    results, f1, f2, f3, f4, f5 = holt_seasonal_comparison(livestock2)
    print(results)
    print()

    # 5. Plots of Level and Slope
    print("5. Plots of Level and Slope")
    plot_level_slope_level_slopes(f2, f4, "Level and Slope Components")
    print()

    # 6. Comparison
    print("6. Comparison (livestock2 + livestock3, 5 models)")
    plot_comparison(livestock2, livestock3)
    print()

    # 7. Holt-Winters Seasonal
    print("7. Holt-Winters Seasonal (aust, 4 fits)")
    hw_results, hw1, hw2, hw3, hw4 = holt_winters_seasonal(aust)
    print(hw_results)
    print()

    # 8. Internals
    print("8. Internals (aust, add-add and add-mul)")
    show_internals(aust)
    print()

    # 9. Simulation
    print("9. Simulation and Confidence Intervals (aust)")
    simulate_forecasts(aust)
    print()

    print("All done!")
