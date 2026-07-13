---
title: "Feature Store"
source: studies:techno/feature_store.md
ingested: 2026-06-19
compiled: true
tags: [feature-store, ml-infrastructure, feast, tecton, featureform]
---

# Feature Store Summary

A feature store is an operational data system for managing and serving machine learning features to models in production. It decouples ML systems from underlying data infrastructure, providing a centralized repository with metadata management, point-in-time correct retrieval, and unified access for both offline training and online inference.

**Key platforms covered:**

- **Tecton** — A hosted SaaS feature platform (proprietary fork of Feast) that manages feature definitions as transformation pipelines. It supports batch, streaming, and real-time processing, reusing infrastructure like AWS EMR, Databricks, Spark, and Snowflake. Tecton provides an offline store (on S3) for training and an online store (distributed key-value) for low-latency inference. Feature Services bundle features that power a model, accessible via SDK or HTTP API.

- **Feast** — An open-source feature store that serves features from a low-latency online store (real-time prediction) or an offline store (training). Core value propositions include feature reuse, decoupling ML from data infrastructure, centralized registry for production deployment, and point-in-time correct feature retrieval to avoid data leakage. Feast is not an ETL/ELT tool.

- **FeatureForm** — An open-source feature store that transforms existing infrastructure into a feature store. It can orchestrate data flow from Spark to Redis, supports native embeddings and vector databases for both inference and training, and runs on Kubernetes or locally on Minikube. Data scientists can push transformation, feature, and training-set definitions to a centralized repository.

**Common architecture patterns:**
1. Feature definitions are created and validated in notebooks
2. Feature pipelines are run to ensure correctness
3. Features are registered in a centralized repository
4. Training data is generated for model testing
5. Changes are applied to production workspaces via CI/CD

Feature stores can also enrich LLM prompts with real-time signals derived from customer events and streaming data.