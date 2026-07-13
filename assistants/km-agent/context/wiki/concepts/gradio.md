---
title: "Gradio"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/gradio/index.md]
related: [streamlit, nicegui]
tags: [gradio, ui, demo, python, ml]
---

# Gradio

Gradio is a Python library designed for rapidly building user interfaces to demonstrate ML models and applications. Its primary advantage is the ability to wrap any Python function with an interactive UI in minimal code.

## Core Features

- **Function wrapping**: Any Python function can automatically become a UI component.
- **Blocks API**: The `gr.Blocks` construct enables custom interface design; components added within `with` clauses are automatically registered.
- **Event-driven interactivity**: Components that act as inputs to event listeners become interactive automatically.
- **Shared state**: Global variables declared at the top of the program are shared across connected sessions, an advantage over alternatives like Streamlit.
- **Chatbot components**: Dedicated `Chatbot` and `ChatInterface` components streamline chatbot demo development.

## Usage

Gradio can be run in development or server mode:

```sh
# Continuous development
gradio main.py
# Server mode
python main.py
```

The app serves on `http://localhost:7860` by default.

## Advanced Patterns

- **Read/write outputs**: Set `interactive=True` on output components like `gr.Textbox` to allow user editing.
- **Accordions**: Nested accordion panels can organize complex interface layouts.
- **Dynamic updates**: Component configurations can be updated programmatically via event listeners.

Gradio is particularly well-suited for rapid prototyping and demoing of ML/AI applications.

## Sources
- [Develop UI with Gradio](../summaries/gradio-index.md)

## Related
- [Streamlit](streamlit.md)
- [NiceGUI](nicegui.md)