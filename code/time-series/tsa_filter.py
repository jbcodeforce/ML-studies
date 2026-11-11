"""
Time Series Filtering: HP, Baxter-King, Christiano-Fitzgerald, and Hamilton Filters

This script demonstrates four popular time-series filtering methods from statsmodels:
  1. Hodrick-Prescott (HP) filter — trend vs. cyclical decomposition
  2. Baxter-King (BK) approximate band-pass filter
  3. Christiano-Fitzgerald (CF) approximate band-pass filter
  4. Hamilton filter

The dataset used is the quarterly US macroeconomic data from 1959Q1–2009Q3
supplied by statsmodels.datasets.macrodata.

See source: https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_filters.html
"""

import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
from statsmodels.datasets import macrodata
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def setup_data():
    """Load the macro data and attach a proper DatetimeIndex."""
    dta = macrodata.load_pandas().data

    # Build a DatetimeIndex from the quarterly period labels
    index = pd.Index(sm.tsa.datetools.dates_from_range("1959Q1", "2009Q3"))
    dta.index = index

    # Remove redundant year/quarter columns
    del dta["year"]
    del dta["quarter"]

    return dta


# --------------------------------------------------------------------------- #
# 1. Hodrick-Prescott Filter
# --------------------------------------------------------------------------- #
def hp_filter_gdp(dta: pd.DataFrame):
    """Decompose real GDP into cyclical and trend components using the HP filter."""
    gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(dta["realgdp"])

    gdp_decomp = dta[["realgdp"]].copy()
    gdp_decomp["cycle"] = gdp_cycle
    gdp_decomp["trend"] = gdp_trend
    return gdp_decomp


def plot_hp_filter(gdp_decomp: pd.DataFrame):
    """Plot real GDP and the extracted HP trend for 2000 onward."""
    fig, ax = plt.subplots(figsize=(12, 8))
    gdp_decomp[["realgdp", "trend"]]["2000-03-31":].plot(ax=ax, fontsize=16)
    ax.set_title("Real GDP and the extracted trend using the HP filter")


# --------------------------------------------------------------------------- #
# 2. Baxter-King Approximate Band-Pass Filter
# --------------------------------------------------------------------------- #
def bk_filter_infl_unemp(dta: pd.DataFrame):
    """Apply the BK filter to inflation and unemployment series."""
    bk_cycles = sm.tsa.filters.bkfilter(dta[["infl", "unemp"]])
    return bk_cycles


def plot_bk_filter(bk_cycles: pd.DataFrame):
    """Plot the cycle components extracted by the BK filter."""
    fig, ax = plt.subplots(figsize=(12, 10))
    bk_cycles.plot(ax=ax, style=["r--", "b-"])
    ax.set_title("Cycle extracted using Baxter & King's filter")


# --------------------------------------------------------------------------- #
# 3. Christiano-Fitzgerald Approximate Band-Pass Filter
# --------------------------------------------------------------------------- #
def cf_filter_infl_unemp(dta: pd.DataFrame):
    """Apply the CF filter to inflation and unemployment series."""
    cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])
    return cf_cycles


def adfuller_series(dta: pd.DataFrame):
    """Print ADF tests for unemployment and inflation."""
    print("=== ADF Test Results ===")
    print(adfuller(dta["unemp"], result_object=True))
    print(adfuller(dta["infl"], result_object=True))
    print("=" * 40)


def plot_cf_filter(cf_cycles: pd.DataFrame):
    """Plot the cycle components extracted by the CF filter."""
    fig, ax = plt.subplots(figsize=(14, 10))
    cf_cycles.plot(ax=ax, style=["r--", "b-"])
    ax.set_title("Cycle extracted using Christiano & Fitzgerald's filter")


# --------------------------------------------------------------------------- #
# 4. Hamilton Filter
# --------------------------------------------------------------------------- #
def hamilton_filter_infl_unemp(dta: pd.DataFrame, h=8, p=4):
    """Apply Hamilton's filter to inflation and unemployment series.

    Parameters
    ----------
    dta : pd.DataFrame
        DataFrame containing 'infl' and 'unemp' columns.
    h : int, default 8
        Number of leads (forecast horizon).
    p : int, default 4
        Number of lags.

    Returns
    -------
    tuple of DataFrame
        (cycles, trend) DataFrames with cyclical and trend components.
    """
    h_cycles, h_trend = sm.tsa.filters.hamilton_filter(dta[["infl", "unemp"]], h=h, p=p)
    return h_cycles, h_trend


def plot_hamilton_filter(h_cycles: pd.DataFrame):
    """Plot the cycle components extracted by Hamilton's filter."""
    fig, ax = plt.subplots(figsize=(14, 10))
    h_cycles.plot(ax=ax, style=["r--", "b-"])
    ax.set_title("Cycle extracted using Hamilton's filter")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print(f"Dataset: {macrodata.NOTE}")
    print(f"\nFirst 10 observations of the data:")
    print()

    # ---- Load and inspect the data ----
    dta = setup_data()
    print(dta.head(10))

    # ---- 1. Hodrick-Prescott Filter ----
    print("\n" + "=" * 50)
    print("1. Hodrick-Prescott Filter on Real GDP")
    print("=" * 50)
    gdp_decomp = hp_filter_gdp(dta)
    print(gdp_decomp.tail())
    plot_hp_filter(gdp_decomp)
    plt.tight_layout()
    plt.savefig("/Users/jerome/Documents/Code/ML-studies/code/time-series/hp_filter_gdp.png", dpi=150)
    plt.show()

    # ---- 2. Baxter-King Filter ----
    print("\n" + "=" * 50)
    print("2. Baxter-King Approximate Band-Pass Filter on Inflation & Unemployment")
    print("=" * 50)
    bk_cycles = bk_filter_infl_unemp(dta)
    print(bk_cycles.tail())
    plot_bk_filter(bk_cycles)
    plt.tight_layout()
    plt.savefig("/Users/jerome/Documents/Code/ML-studies/code/time-series/bk_filter_infl_unemp.png", dpi=150)
    plt.show()

    # ---- 3. Christiano-Fitzgerald Filter ----
    print("\n" + "=" * 50)
    print("3. Christiano-Fitzgerald Approximate Band-Pass Filter on Inflation & Unemployment")
    print("=" * 50)
    adfuller_series(dta)
    cf_cycles = cf_filter_infl_unemp(dta)
    print(cf_cycles.head(10))
    plot_cf_filter(cf_cycles)
    plt.tight_layout()
    plt.savefig("/Users/jerome/Documents/Code/ML-studies/code/time-series/cf_filter_infl_unemp.png", dpi=150)
    plt.show()

    # ---- 4. Hamilton Filter ----
    print("\n" + "=" * 50)
    print("4. Hamilton Filter on Inflation & Unemployment (h=8, p=4)")
    print("=" * 50)
    h_cycles, h_trend = hamilton_filter_infl_unemp(dta, h=8, p=4)
    print(h_cycles.head(13))
    plot_hamilton_filter(h_cycles)
    plt.tight_layout()
    plt.savefig("/Users/jerome/Documents/Code/ML-studies/code/time-series/hamilton_filter_infl_unemp.png", dpi=150)
    plt.show()

    print("\nAll plots saved. Done.")
