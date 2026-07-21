# km-agent for ML-studies

[See the km-agent project](https://github.com/jbcodeforce/km-agent/tree/main) as it is used to manage the knowledge graph from this repository and do deeper research.

km-agent backend, agent team, and the chat UI run from the km-agent clone; context, knowledge and configuration live here.

## Prerequisites

- km-agent clone at path recorded in `.kma-home`
- macOS 26+ with Apple `container` CLI
- `uv`, `npm`, OMLX (or configured cloud LLM)
- `docs/` with km-agent raw frontmatter (see `docs/.manifest.json`)

## Quick start

```bash
# 1. Copy and edit secrets (if not already done)
cp example.env .env
# edit .env — LLM keys, ports

# 2. Start stack (Postgres + AgentOS + chat UI)
./starter-mac.sh --dev --frontend

# 3. Verify
./verify_config.sh --frontend

# 4. Compile docs into context/wiki/
./compile-docs.sh
```

## Layout

| Path | Purpose |
|------|---------|
| `context/raw/` | Researcher-ingested sources |
| `context/wiki/` | Compiled wiki (index, concepts, summaries) |
| `context/ontology/` | Derived RDF graph |
| `.env` | Runtime configuration (gitignored) |
| `.kma-home` | Absolute path to km-agent clone |

## Docs

- [km-agent USER_GUIDE](/Users/jerome/Documents/Code/km-agent/docs/USER_GUIDE.md)
- UC-2 compile workflow in USER_GUIDE
