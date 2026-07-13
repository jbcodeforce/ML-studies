# Gradio Summary

Gradio is a Python library for rapidly building user interfaces to demonstrate ML models and applications. It allows developers to wrap a Python function with an interactive UI in minimal code. Key advantages over alternatives like Streamlit include the ability to share data between pages and elements.

## Key Features
- **Function wrapping**: Any Python function can be wrapped with a UI automatically.
- **Blocks API**: `gr.Blocks` enables custom interface design; components are automatically added within `with` clauses.
- **Event-driven interactivity**: Any component acting as an input to an event listener becomes interactive.
- **Shared state**: Global variables at the top of the program are shared between connected sessions.
- **Chatbot support**: Dedicated `Chatbot` and `ChatInterface` components for building chatbot demos quickly.

## Development
- Run with `gradio main.py` for continuous development or `python main.py` for server mode.
- Serves on `http://localhost:7860` by default.
- Supports read/write outputs via `interactive=True` and accordion inputs for complex interfaces.

Gradio is particularly well-suited for quick prototyping and demoing of ML/AI applications.