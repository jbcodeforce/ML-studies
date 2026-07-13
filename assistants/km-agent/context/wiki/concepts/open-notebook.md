---
title: "Open Notebook"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/genAI/opennotebook.md]
related: [rag-architecture, generative-ai-overview, private-chatbot-architecture, text-embeddings]
tags: [open-notebook, self-hosted, rag, notebooklm, privacy, open-source, ai-tools]
---

# Open Notebook

**Open Notebook** is an open-source, self-hosted alternative to Google's [notebookLM](https://notebooklm.google.com/), providing AI-powered research and writing capabilities while keeping data private. It implements RAG-based querying over user-uploaded documents with citation-backed answers.

## Core Capabilities

- **Multi-format ingestion**: PDFs, Google Docs, Slides, Markdown, plain text, web URLs, YouTube links, and audio/video with transcripts.
- **RAG-based querying**: Answer questions grounded in uploaded sources with citations to exact source snippets.
- **Content generation**: Produce study guides, briefing documents, FAQs, timelines, blog posts, and presentation outlines from personal materials.
- **Learning tools**: Generate flashcards, quizzes, and interactive "Learning Guide" tutoring experiences.
- **Audio overviews**: Transform written content into podcast-style audio summaries.

## Architecture & Deployment

Open Notebook is deployable via **Docker Compose**. The workflow is:
1. Create a notebook with a description (used as LLM context).
2. Upload sources and embed them.
3. Configure the AI provider, model, and embeddings via `.env` configuration.

Enabling a new provider requires uncommenting the API URL and keys in the environment file used by Docker Compose.

## Privacy Advantage

Unlike Google's notebookLM, Open Notebook is self-hosted, meaning uploaded documents and queries never leave the user's infrastructure. This makes it suitable for sensitive research, enterprise documents, and private study materials.

## Sources
- [AI Notebook](../summaries/opennotebook.md)

## Related
- [RAG Architecture](rag-architecture.md)
- [Generative AI Overview](generative-ai-overview.md)
- [Private Chatbot Architecture](private-chatbot-architecture.md)
- [Text Embeddings](text-embeddings.md)