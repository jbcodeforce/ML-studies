---
title: "LLM-Driven Agentic Workflows"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/architecture/po-processing.md]
related: [document-understanding, cloud-native-architecture]
tags: [llm, agentic, langgraph, gemini, automation, orchestration]
---

# LLM-Driven Agentic Workflows

Agentic workflows use Large Language Models (LLMs) as reasoning engines that can plan, decide, and execute multi-step processes, often integrating with external systems and expert knowledge bases.

## Overview

Agentic workflows represent a shift from simple LLM prompting to structured, stateful processes where the LLM acts as an autonomous agent that can:
- Reason about complex inputs
- Make decisions about next steps
- Call external tools and APIs
- Maintain conversation state across interactions
- Integrate with rule-based systems and expert knowledge

## Key Components

### LangGraph
A framework for building conversational applications with LLMs that provides:
- **State management**: Tracks conversation state across turns
- **Function calling**: Integrates LLM decisions with external system calls
- **Flow control**: Supports conditional branching and loops
- **Configuration trees**: Define structured flows for guided extraction and decision-making

### Multi-Modal LLMs (Google Gemini)
Modern LLMs like Gemini support multiple input types:
- Text, images, audio, video, and code
- Multi-size models (Nano, Pro, Ultra) for different performance/latency needs
- Safety controls across harassment, hate speech, explicit content, and dangerous content
- High benchmark performance (e.g., 90.0% on MMLU)

### Expert System Integration
Agentic workflows can bridge unstructured AI reasoning with structured rule engines:
- The LLM interprets unstructured input and determines what information is needed
- A configuration tree drives targeted extraction from documents
- Results feed into expert systems for validation and decision-making

## Example: PO Configuration Flow

In a purchase order processing context:
1. Document AI extracts structured data from uploaded POs
2. A LangGraph agent determines what additional information is needed
3. The agent queries the expert system using a configuration tree
4. If unstructured requests arise, Gemini provides entity extraction and reasoning
5. The agent orchestrates the flow between extraction, configuration, and validation

## Sources
- [Purchase Order Processing with AI and Hybrid Cloud](../summaries/po-processing.md)