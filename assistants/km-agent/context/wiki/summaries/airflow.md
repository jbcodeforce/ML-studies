---
title: "Apache Airflow"
source: studies:techno/airflow.md
ingested: 2026-06-19
tags: [orchestration, workflow, batch-processing]
type: summary
---

# Apache Airflow Summary

Apache Airflow is a batch-oriented workflow orchestration platform built on a Python framework, designed to connect with any technology.

## Key Value Propositions
- **Flexible Deployment**: Can be deployed locally or on a distributed cluster.
- **Extensible**: Components are extensible to fit specific environments.
- **Version Controlled**: Supports rollback to previous versions of workflows.
- **Batch-Focused**: Not designed for event-based workflows or streaming. Can be combined with Kafka to batch-process data in topics.

## Core Concepts
- Workflows are represented as **Directed Acyclic Graphs (DAGs)**.
- The platform follows a component-based architecture.

Airflow fills the orchestration gap for batch ML pipelines and data processing workflows, complementing streaming technologies like Kafka.