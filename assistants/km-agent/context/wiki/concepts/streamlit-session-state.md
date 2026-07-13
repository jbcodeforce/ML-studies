---
title: "Streamlit Session State"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/techno/streamlit.md]
related: [streamlit]
tags: [python, streamlit, session-state, state-management]
---

# Streamlit Session State

`st.session_state` is Streamlit's built-in mechanism for preserving and sharing data across page navigations and reruns within a single user session.

## Purpose

In Streamlit, scripts rerun from top to bottom on every interaction. `st.session_state` provides a persistent dictionary-like object that survives reruns, enabling:
- Cross-page data sharing
- Form state persistence
- User configuration storage

## Usage Pattern

**Saving data to session state** (e.g., from a settings form):
```python
if save_button:
    data = {
        "callWithVectorStore": callWithVectorStore,
        "callWithDecisionService": callWithDecisionService,
        "llm_provider": llm_provider
    }
    st.session_state["app_config"] = data
```

**Reading data from session state** (on another page):
```python
if 'app_config' not in st.session_state:
    st.session_state['app_config'] = {
        "callWithVectorStore": False,
        "callWithDecisionService": False,
        "llm_provider": "openAI"
    }
app_config = st.session_state['app_config']
```

## Best Practices

- Always initialize with sensible defaults to avoid `KeyError` on first access.
- Use it for lightweight configuration and state, not for large data objects.

## Sources
- [Streamlit](../summaries/streamlit.md)

## Related
- [Streamlit](streamlit.md)