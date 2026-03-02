# Agno API reference and source navigation

Use these links to open the official Agno API docs and GitHub source when you need to look up a function or class.

## F12 (Go to Definition)

To jump to agno source with **F12** (or Cmd+Click):

1. **Interpreter** – The venv with `agno` is at `src/.venv`. The project sets this via `.vscode/settings.json` and `pyrightconfig.json` (venvPath + venv). If you still get “no definition found”:
   - Open Command Palette → **Python: Select Interpreter**.
   - Pick the one under **ML-studies/src/.venv** (e.g. `Python 3.12.x ('.venv': venv)` with path ending in `ML-studies/src/.venv`).
2. **Sync** – Run `uv sync` from `ML-studies/src` or from `src/agentic/agno` so `agno` is installed in that venv.
3. **Reload** – Reload the window (Command Palette → **Developer: Reload Window**) after changing the interpreter or config.
4. **Multi-root** – If the workspace has several roots (e.g. MyAIAssistant + ML-studies), ensure the interpreter you select is the one from **ML-studies/src/.venv**, not another project’s venv.

## API documentation

- [Agno docs](https://docs.agno.com/)
- [API reference overview](https://docs.agno.com/reference-api/overview)
- [Agent reference](https://docs.agno.com/reference/agents/agent)
- [Documentation index (llms.txt)](https://docs.agno.com/llms.txt) – discover all pages

## Cursor: index Agno docs for navigation

To let Cursor use the Agno API when coding:

1. Open **Settings** → **Indexing & Docs**
2. Add: `https://docs.agno.com/llms-full.txt`

Cursor will index the Agno reference so you can get accurate suggestions and the AI can cite the API.

## GitHub source (libs/agno/agno)

Source tree: [agno-agi/agno – libs/agno/agno](https://github.com/agno-agi/agno/tree/main/libs/agno/agno)

| Import | Source (GitHub) |
|--------|-----------------|
| `agno.agent` | [agent/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/agent) |
| `agno.models.ollama` | [models/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models) |
| `agno.models.openai.like` | [models/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/models) |
| `agno.tools` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.tools.duckduckgo` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.tools.yfinance` | [tools/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/tools) |
| `agno.db.sqlite` | [db/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/db) |
| `agno.knowledge` | [knowledge/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/knowledge) |
| `agno.utils` | [utils/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/utils) |
| `agno.vectordb.chroma` | [vectordb/](https://github.com/agno-agi/agno/tree/main/libs/agno/agno/vectordb) |

To open a specific file, browse the folder above and open the `.py` file (e.g. `agent/agent.py`, `models/ollama.py`).
