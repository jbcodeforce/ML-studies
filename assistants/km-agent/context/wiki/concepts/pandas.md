---
title: "Pandas"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/pandas.md]
related: [python-ml-development-environment]
tags: [pandas, dataframe, python, data-manipulation]
---

# Pandas

Pandas is the primary tool data scientists use for exploring and manipulating data. It provides the `DataFrame` structure — a tabular data container with rows (records) and columns, where each entry holds a value. Pandas integrates with many file formats and data sources out of the box, including CSV, Excel, SQL, JSON, and Parquet.

## Core Operations

- **Loading data**: `pd.read_csv()` with options for missing value handling (`na_values`), column naming, and other format-specific parameters.
- **Inspecting data**: `describe()` for statistical summaries, `head()` for previewing rows.
- **Handling missing data**: Locate null values with `.isnull()` and remove them with `dropna()`.
- **Extracting features**: Selecting columns and converting to `.values` for use with ML libraries like scikit-learn.

## Integration with ML Pipelines

Pandas DataFrames serve as the standard intermediate format for preparing data before passing to ML libraries. Feature columns and target labels can be extracted as NumPy arrays for use with scikit-learn classifiers and other algorithms.

## Sources
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Kaggle's Pandas Training](https://www.kaggle.com/learn/pandas)

## Related
- [Python ML Development Environment](python-ml-development-environment.md)