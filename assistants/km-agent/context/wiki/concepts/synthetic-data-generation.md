---
title: "Synthetic Data Generation"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/watsonx.md]
related: [watsonx-ai, data-management, feature-engineering]
tags: [synthetic-data, data-generation, ai, privacy, watsonx]
---

# Synthetic Data Generation

Synthetic data generation creates artificial data that conforms to existing schemas and statistical properties, useful for augmenting or replacing real data in AI model development.

## Purpose

- **Augment existing datasets**: Increase data volume for model training.
- **Protect sensitive data**: Replace real PII or confidential information.
- **Mitigate bias**: Generate balanced data distributions.

## Methods

WatsonX uses statistical methods including:
- **Kolmogorov-Smirnov test**: Compares cumulative distribution functions.
- **Anderson-Darling test**: Sensitive to tail distributions.

## Capabilities

- **Categorical data**: Generates categorical values from provided strings with occurrence frequencies.
- **Numerical data**: Uses standard distributions with configurable mean and deviation.
- **Anonymization**: Columns can be anonymized during generation.
- **Correlation building**: Profiles datasets and builds correlations between columns to reflect real-world relationships.
- **Schema conformance**: Generated data conforms to the existing dataset schema.

## Output

Generated data can be exported in formats such as .xls.

## Sources
- [WatsonX.ai](../summaries/watsonx.md)

## Related
- [WatsonX.ai](watsonx-ai.md)
- [Data Management](data-management.md)