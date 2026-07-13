# AI Notebook — Open Notebook Summary

**Open Notebook** is an open-source alternative to Google's [notebookLM](https://notebooklm.google.com/), an AI-powered research and writing tool. While Google's notebookLM provides powerful RAG-based querying and content analysis, it raises privacy concerns since Google may use uploaded data. Open Notebook delivers comparable capabilities with self-hosted deployment, keeping data private.

## Value Propositions

Open Notebook enables users to:
- **Understand, organize, and create content** from uploaded sources including PDFs, Google Docs, Slides, Markdown, plain text, web URLs, YouTube links, and audio/video with transcripts.
- **Summarize and extract information** from complex documents or multiple sources.
- **Answer specific questions** grounded only in uploaded files, with citations back to source snippets.
- **Generate content** such as study guides, briefing documents, FAQs, timelines, blog posts, and presentation outlines from personal materials.
- **Support learning** with flashcards, quizzes, and interactive "Learning Guide" tutoring based on course readings.
- **Create audio overviews** — podcast-style audio summaries of written content.

## Deployment

The project is deployable via Docker Compose. Setup involves:
1. Creating a notebook with a description (used as context for the LLM).
2. Uploading sources and embedding them.
3. Configuring the AI provider, model, and embeddings by uncommenting API URLs and keys in the `.env` file.

The project repository is at GitHub (lfnovo/open-notebook), with documentation including a [getting-started guide](https://github.com/lfnovo/open-notebook/blob/main/docs/getting-started/first-notebook.md).

## Connection to Wiki Concepts

Open Notebook is a practical implementation of **RAG Architecture** — it embeds uploaded documents into a vector store and retrieves relevant snippets to ground LLM responses. It also relates to **Private Chatbot Architecture** by offering a self-hosted alternative to cloud AI tools, and to **Generative AI Overview** as a consumer-facing Gen AI application.