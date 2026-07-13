---
title: "Feature Store"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/feature_store.md]
related: [feature-engineering, ml-pipeline, data-management]
tags: [machine-learning, ml-infrastructure, feature-management]
---

# Feature Store

A **feature store** is an operational data system for managing and serving machine learning features to models in production. It provides a centralized repository with metadata management, point-in-time correct retrieval, and unified access for both offline training and online inference.

## Core Value Propositions

- **Feature reuse** via a central searchable repository with metadata.
- **Decoupling** ML systems from data sources, bringing stability through a single data access layer.
- **Facilitated deployment** with a centralized registry and service layer.
- **Point-in-time correctness** when exporting feature datasets for model training, avoiding data leakage.

## Architecture

Feature stores typically consist of:

1. **Feature Definition Layer** — Declarative transformations (SQL-like or Python) that define how raw data becomes features.
2. **Offline Store** — Batch storage (e.g., S3, data lake) used by data scientists for model training and notebooks.
3. **Online Store** — Low-latency distributed key-value store serving the latest feature values for real-time inference.
4. **Feature Service** — A curated set of features that power a specific model, accessible via SDK or HTTP API.
5. **Feature Repository** — A collection of version-controlled definitions that can be deployed with CI/CD.

## Typical Workflow

1. Create and validate feature definitions in a notebook
2. Run feature pipelines interactively to verify correctness
3. Register features in the centralized repository
4. Generate training data for model testing
5. Apply changes to a live production workspace

## Platforms

### Tecton
A hosted SaaS feature platform (proprietary fork of Feast) that manages feature definitions as transformation pipelines. Supports batch, streaming, and real-time processing, reusing infrastructure like AWS EMR, Databricks, Spark, and Snowflake. Sources include Kafka, MSK, Kinesis, S3, Delta Lake, DynamoDB, EMR, Athena, and Redshift.

### Feast
An open-source feature store serving features from a low-latency online store (real-time prediction) or an offline store (training). Feast is not an ETL/ELT tool.

### FeatureForm
An open-source feature store that transforms existing infrastructure into a feature store. It can orchestrate data flow from Spark to Redis, supports native embeddings and vector databases for both inference and training, and runs on Kubernetes or locally on Minikube.

## LLM Integration

Feature stores can be used to enrich LLM prompts with real-time signals derived from customer events and streaming data, enabling powerful insights to be passed as context to language models.

## Sources
- [Feature Store](../summaries/feature_store.md)

## Related
- [Feature Engineering](feature-engineering.md)
- [ML Pipeline](ml-pipeline.md)
- [Data Management](data-management.md)