---
title: "Probability Rules"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [mathematical-foundations, bayesian-inference]
tags: [probability, statistics, machine-learning]
---

# Probability Rules

Core probability formulas for independent, dependent, disjoint, and non-mutually exclusive events.

## Independent Events

When two events A and B are independent (one does not affect the other):

**P(A ∧ B) = P(A) × P(B)**

## Dependent Events (Conditional Probability)

When events are related, use Bayes' rule:

**P(A|B) = P(B|A) × P(A) / P(B)**

## Disjoint (Mutually Exclusive) Events

Events A and B cannot both happen simultaneously:

**P(A ∨ B) = P(A) + P(B)**

## Non-Mutually Exclusive Events

When A and B can overlap, subtract the intersection to avoid double-counting (inclusion-exclusion principle):

**P(A ∨ B) = P(A) + P(B) − P(A ∧ B)**

### Example: Card Probability

*What is the probability that a card chosen from a standard 52-card deck will be a Jack or a heart?*

- P(Jack) = 4/52
- P(Heart) = 13/52
- P(Jack of Hearts) = 1/52

P(Jack ∨ Heart) = 4/52 + 13/52 − 1/52 = **16/52**

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
- [Bayesian Inference](bayesian-inference.md)