# AI Notebook

Google created [notebookLM](https://notebooklm.google.com/) as AI-powered research and writing tool. It is a nice technology for deeper analysis and doing RAG based query. The problem is Google will use the data sent. The open source has the same capability with [open notebook]()

## Value propositions

* understand, organize, and create new content from their uploaded source
    * Documents: PDFs, Google Docs, Google Slides, Markdown files, and plain text.
    * Web Content: Public website URLs and YouTube video links.
    * Audio/Video: Audio files and video content (that has a transcript).

## Use cases
* Summarizing and Extracting Information: from complex documents or multiple sources.
* Answering Specific Questions: using only the information within uploaded files, providing citations back to the exact source snippet for verification.
* Content Generation: Creating various outputs like study guides, briefing documents, FAQs, timelines, or even blog posts and presentation outlines, all grounded in personal source material.
* Learning and Study: Generating flashcards and quizzes, or using the "Learning Guide" feature for a personalized, interactive tutoring experience based on your course readings or research papers.
* Audio Overviews: Transforming written content into engaging, podcast-style audio summaries, allowing for on-the-go learning and content consumption

[Notebook LM tutorial](https://www.youtube.com/watch?v=UG0DP6nVnrc)

## Getting started

See [docker compose]() and the version I used `Documents/Code/open-notebook` with one version for Flink-studies.

* Setup provider, model and embeddings

The process:
1. create notebook with a description as it is used in the context for the LLM
1. Upload sources and embed them
1. 

## How to

[First notebook](https://github.com/lfnovo/open-notebook/blob/main/docs/getting-started/first-notebook.md).

### Enable a new provider
uncomment api url and keys in the .env used by docker compose.