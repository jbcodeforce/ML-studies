# Cursor + local oMLX (Codestral on :7999)

## Quick reference

| Setting | Value |
|---------|--------|
| oMLX base URL | `http://127.0.0.1:7999/v1` |
| API key | `localkey` |
| Model id (from `GET /v1/models`) | `Codestral-22B-v0.1-4bit` |
| Admin UI | http://127.0.0.1:7999/admin |

## 1. Start oMLX

```bash
cd /Users/jerome/Documents/Code/ML-studies/src/agentic/agno
./startoLMX.sh
```

Verify:

```bash
curl -s http://127.0.0.1:7999/v1/models -H "Authorization: Bearer localkey" | jq .
```

## 2. Cursor IDE

Configuration is applied via:

- **User settings**: `cursor.openai.baseUrl` → `http://127.0.0.1:7999/v1` in `~/Library/Application Support/Cursor/User/settings.json`
- **Cursor state DB**: `openAIBaseUrl`, `cursorAuth/openAIKey`, and custom model names in reactive storage

After changes, **reload Cursor** (Developer: Reload Window).

In **Cursor Settings → Models**:

1. **Override OpenAI Base URL**: `http://127.0.0.1:7999/v1`
2. **OpenAI API Key**: `localkey`
3. Enable the custom model **Codestral-22B-v0.1-4bit** (exact id from `/v1/models`)
4. Select it in Chat (`Cmd+L`)

### What works / what does not

| Surface | Local oMLX |
|---------|------------|
| Chat / Cmd+K | Yes (OpenAI-compatible override) |
| Agent / Composer | Partial — tool loop quality varies; may need cloud Composer as fallback |
| Tab completion | No — stays on Cursor built-in models |
| MCP tools (`~/.cursor/mcp.json`) | Yes — independent of LLM endpoint |

## 3. Cursor CLI

The CLI (`~/.cursor/cli-config.json`) uses Cursor cloud (`api2.cursor.sh`). There is **no** supported way to point the CLI agent at oMLX.

**Terminal alternatives** (same model):

```bash
# Claude Code (Anthropic API on oMLX)
export ANTHROPIC_BASE_URL="http://127.0.0.1:7999"
export ANTHROPIC_API_KEY="localkey"
claude --model Codestral-22B-v0.1-4bit

# OpenAI-compatible clients
export OPENAI_BASE_URL="http://127.0.0.1:7999/v1"
export OPENAI_API_KEY="localkey"
```

## 4. Troubleshooting

| Issue | Fix |
|-------|-----|
| 401 Unauthorized | Use `Authorization: Bearer localkey` |
| Wrong model name | Copy id from `curl .../v1/models` (directory name, not HuggingFace path) |
| Chat template error on Codestral | Ensure `tokenizer_config.json` includes `chat_template`; reload model in oMLX admin |
| Cursor still hits old URL | Reload window; check Settings → Models shows `127.0.0.1:7999` |

## 5. Agno

See [`olmx_deep_researcher.py`](./olmx_deep_researcher.py) — uses the same base URL, key, and model id.
