---
title: "NiceGUI"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/nicegui.md]
related: [streamlit, gradio]
tags: [python, ui-framework, fastapi, web-ui]
---

# NiceGUI

NiceGUI is an open-source Python library for building web user interfaces with robust state management across pages. It sits alongside other Python UI frameworks like Streamlit and Gradio, but differentiates itself through strong state handling and integration with the FastAPI web framework.

## Architecture

NiceGUI uses a component-based architecture built around pages, components, events, and handlers. Components are arranged on pages using layouts that provide grids, tabs, carousels, expansions, and menus.

## Customization

The library supports styling and behavior customization through:
- **Tailwind CSS classes** for styling
- **Quasar components** for extended UI behavior

## Server

NiceGUI runs on FastAPI, a modern high-performance Python web framework, providing async support and automatic API documentation.

## Sources
- [NiceGUI](../summaries/nicegui.md)

## Related
- [Streamlit](streamlit.md)
- [Gradio](gradio.md)