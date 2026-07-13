---
title: "Bayesian Inference"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [mathematical-foundations, probability-rules, anomaly-detection]
tags: [bayesian, probability, machine-learning, inference]
---

# Bayesian Inference

Bayesian inference is a statistical approach based on probability theory that uses Bayes' theorem to update beliefs about hypotheses in light of new evidence.

## Bayes' Theorem

The core formula for the probability of a hypothesis H given evidence E:

**P(H|E) = P(E|H) × P(H) / P(E)**

Where:
- **P(H)** — the **prior**: probability of the hypothesis before seeing evidence
- **P(E|H)** — the **likelihood**: probability of the evidence assuming the hypothesis is true
- **P(E)** — the **evidence**: total probability of seeing the evidence, computed as P(H)×P(E|H) + P(¬H)×P(E|¬H)
- **P(H|E)** — the **posterior**: updated probability of the hypothesis after seeing evidence

## Frequentist vs. Bayesian

In machine learning, there are two main statistical approaches:

- **Bayesian**: Based on probability theory; updates probabilities using Bayes' theorem as new data arrives. Handles uncertainty and complex data well.
- **Frequentist**: Based on statistical inference; uses hypothesis testing, confidence intervals, and p-values. More common when data is abundant and relationships are well-defined.

## The Steve Example

A classic demonstration of base rate importance:

> Steve is described as "meek and tidy soul, with a need for order and a passion for detail." Is Steve more likely a librarian or a farmer?

Despite the intuitive answer being "librarian," Bayes' theorem shows:

- **Prior**: ~20 farmers per librarian → P(farmer) = 20/21, P(librarian) = 1/21
- **Likelihood**: 40% of librarians match the description, 10% of farmers do
- **Result**: Farmers matching = 20 × 0.10 = 2.0; Librarians matching = 1 × 0.40 = 0.4

Steve is ~5× more likely to be a **farmer**. P(librarian | description) = 4/(4+20) ≈ **16.7%**

## Key Takeaways

- Always consider **base rates (priors)** before updating beliefs
- New evidence **updates** but does not **replace** prior knowledge
- Rationality is about recognizing which facts are relevant
- Seeing evidence restricts the space of possibilities

## Geometric Interpretation

A unit square can visualize Bayes' theorem:
1. Divide the square into regions proportional to prior probabilities
2. Shade areas within each region where evidence holds (proportional to likelihood)
3. The posterior = ratio of shaded hypothesis area to total shaded area

This shows how conditioning on evidence changes probability distributions.

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
- [Probability Rules](probability-rules.md)
- [Anomaly Detection](anomaly-detection.md)