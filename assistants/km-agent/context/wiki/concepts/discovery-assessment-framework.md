---
title: "Discovery Assessment Framework"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/methodology/index.md]
related: [agentic-ai-implementation-methodology, interface-characteristics-evaluation]
tags: [discovery, assessment, genai, use-case, business-needs, evaluation, scoping]
---

# Discovery Assessment Framework

A structured set of assessment questions for evaluating enterprise readiness and identifying opportunities for generative AI adoption.

## Purpose

The Discovery Assessment helps organizations understand their current state, identify suitable use cases, and plan generative AI implementation. It covers six dimensions:

### Research for Opportunities
Key questions include:
- What manual or repetitive processes could be automated with generative AI?
- Where do employees spend the most time gathering information?
- What customer pain points could be addressed with more natural conversation?
- What expert skills are scarce that AI models could supplement?
- What insights could be uncovered from large volumes of unstructured data?

### Use Cases and Business Needs
- What are the potential use cases (B2B, B2C, Employees)?
- Is the use case a strategic priority?
- What value is associated with the use case?
- Are subject matter experts available?
- What are the current user challenges and pains?

### Experience in AI
- Are you using AI in current business applications?
- What are current/past successes adopting AI?
- What ML support is needed for technical staff?
- How will you monitor model performance and detect drift?

### Generative AI Current Adoption
- How familiar with Generative AI and its common use cases?
- What GenAI technologies have you/are you evaluating?
- What is your risk appetite for model hallucination?
- How do you plan domain adaptation (pre-train, fine-tune, in-context prompting)?
- How frequently does data change?

### Integration Needs
- Is it a new solution or extending an existing one?
- Where is data coming from?
- What type of systems to integrate with?
- Any expected performance requirements?

### Security and Compliance Needs
- Code privacy and IP related code control

## Model Evaluation Steps

1. Select models based on specific use case and tasks
2. **Human calibration**: Understand behavior on certain tasks, fine-tune prompts, assess against ground truth using cosine similarity; ROUGE scores for summarization comparison
3. **Automated evaluation**: Test scenarios with data preparation; LLM as judge using accuracy, coherence, factuality, completeness metrics
4. MLOps integration with self-correction

## Gen AI Specific Scoping

1. Complete the discovery assessment
2. Define key metrics and evaluation methods (accuracy for Document Q&A and Summarization)
3. Define a list of expected questions with correct answers; for summarization, prepare sample summaries and questions

## Sources
- [Agentic AI Implementation Solution Methodology](../summaries/methodology-index.md)

## Related
- [Agentic AI Implementation Methodology](agentic-ai-implementation-methodology.md)
- [LLM Reference Architecture](llm-reference-architecture.md)