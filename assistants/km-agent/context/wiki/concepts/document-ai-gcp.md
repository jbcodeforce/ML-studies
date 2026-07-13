---
title: "Document AI (GCP)"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/gcp/index.md]
related: [google-cloud-ai-platform, document-understanding]
tags: [document-ai, google-cloud, nlp, ocr, entity-extraction, machine-learning]
---

# Document AI (GCP)

Google Document AI is a service for processing and understanding documents using advanced NLP and computer vision. It extracts structured information from various document types including PDFs, images, and scanned documents.

## Processor Types

Document AI uses **Document Processors** as the interface to ML models, responsible for classifying, parsing, and analyzing documents. Three types exist:

1. **Generalized Processors**: For basic content — OCR, structured form parsing, document quality analysis
2. **Specialized Processors**: Pre-trained on specific document types — invoices, receipts, contracts, driving licenses, expense reports (high-variance document types)
3. **Custom Processors**: Built by enterprises for domain-specific documents

## Capabilities

- **Multi-format**: PDFs, images, scanned documents
- **Multi-language**: Supports different natural languages
- **Processing Modes**: Online and batch processing
- **No-Code Tools**: Configure document processing pipelines without writing code
- **Monitoring & Logging**: Integrated with cloud monitoring services
- **Custom Model Integration**: Combine with custom ML models for enhanced entity extraction

## Document AI Workbench

Workbench helps developers create custom document processors, either from scratch or by extending existing ones:

- **Uptraining**: Add custom fields for entity extraction by extending the schema
- **New Model Creation**: Follows standard ML process — labelling, training, evaluation, deployment

## Evaluation Metrics

Quality of entity extraction is assessed using:
- F1 score
- Accuracy
- Recall

## Setup Requirements

- Create a service account for the Document AI application
- Grant the DocumentAI API User role
- Create processor instances in the project

## Document Categories

1. **General docs**: Basic content parsed with general ML models
2. **Specialized docs**: Content needing specialized pre-trained models
3. **Custom docs**: Content requiring enterprise-built custom models

## Sources
- [Google AI Platform](../summaries/index-gcp.md)

## Related
- [Google Cloud AI Platform](google-cloud-ai-platform.md)
- [Document Understanding](document-understanding.md)