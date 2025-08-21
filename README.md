# OpenAI Agents (Rust)

Harmony-aligned, OpenAI-compatible agent orchestration in Rust — no mocks, no automatic fallbacks.

This crate provides a library and CLI to build agents that call OpenAI-compatible models (cloud or OSS) with robust tool orchestration, optional realtime/voice support, and an environment-first configuration model. It aims for practical parity with the Python SDK while staying idiomatic in Rust.

## Highlights

- No mocks or hidden fallbacks: strict, explicit errors; no auto-retries or provider defaults.
- OpenAI-compatible models: Chat Completions, Harmony-style Responses, LiteLLM pass-through.
- Tools: OpenAI tool schema and legacy function_call supported; configurable tool-choice.
- Realtime/Voice: SSE-based realtime module and STT/TTS clients (OpenAI-compatible endpoints).
- Extensible: dynamic plugin system and a simple tool registry.

## Quickstart

1. Create a .env with your model server details (base_url is required):

```bash
# Local OSS (e.g., vLLM / OpenAI-compatible)
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=openai/gpt-oss-120b
# Optional if your server requires auth
# OPENAI_API_KEY=sk-...

# Logging
RUST_LOG=info
```

1. Build and test:

```bash
cargo test -q
```

1. Run the CLI (starts the MCP server and a default agent):

```bash
cargo run
```

By default, the MCP server binds to <http://127.0.0.1:8080> and the runtime registers:

- EchoAgent (simple example that calls your configured model)
- A configured agent using the experimental realtime model

## Configuration

The config loader supports both file-based and environment-based configuration.

- File: set `OPENAI_AGENTS_CONFIG` (default: `./config.yaml`). The loader also reads variables with prefix `OPENAI_AGENTS__` to override file keys.
- Environment: common provider-style variables are respected globally:
  - `OPENAI_BASE_URL` (required)
  - `OPENAI_MODEL` (required)
  - `OPENAI_API_KEY` (optional)
  - `RUST_LOG` (optional)

Schema (`src/config/schema.rs`):

- `api_key: String` (optional)
- `model: String` (required)
- `base_url: String` (required)
- `log_level: String`
- `plugins_path: PathBuf` (defaults to `~/.config/openai_agents/plugins`)
- `max_concurrent_requests: Option<usize>`

Important policy: base_url is required and there are no provider defaults baked into the loader. If a value is missing, it stays empty and you’ll see a clear error where used.

### Env override notes

- File-based overrides: `OPENAI_AGENTS__BASE_URL`, `OPENAI_AGENTS__MODEL`, etc. map onto file keys when using a config file.
- Global overrides: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `RUST_LOG` always overlay the active config at runtime.

## Model backends

- OpenAI Chat Completions (`/chat/completions`): `src/model/openai_chat.rs`
  - Tool schema: supports `tool_calls` and legacy `function_call`.
  - Tool-choice: auto/none and object forms; optional env overrides (see below).
- Harmony-style OSS Responses (`/responses`): `src/model/gpt_oss_responses.rs`
- LiteLLM pass-through: `src/model/litellm.rs` (for aggregating providers behind a single base_url)

Common env toggles for OpenAI-compatible servers (especially vLLM):

- `VLLM_MIN_PAYLOAD` (bool): minimal payload (model+messages only).
- `VLLM_FORCE_FUNCTIONS` (bool): send legacy `functions`/`function_call` instead of `tools`.
- `VLLM_DISABLE_PARALLEL_TOOL_CALLS` (bool): don’t send `parallel_tool_calls: true`.
- `VLLM_TOOL_CHOICE` (string): one of `auto`, `none`, `object:auto`, `object:none`.
- `VLLM_DISABLE_TOOLS_IN_LLM` (bool): don’t pass tool specs to the LLM from the runner.
- `VLLM_DEBUG_PAYLOAD` (bool): pretty-print request JSON at debug level.

## Tools and the runner

The `Runner` orchestrates tool execution and model turns.

- Behavior modes: `RunLlmAgain`, `StopOnFirstTool`, `StopAtTools([..])`, or a custom decider.
- Messages: constructed with optional system instructions and user input; tool outputs are appended as proper `tool` messages with `tool_call_id` when available.
- Error policy: strict — unknown or disabled tool requests return explicit errors; no implicit retries.

Tools live under `src/tools/` with a simple `Tool` trait and a registry for discovery. Tools can optionally expose an OpenAI tool spec so the model can call them via `tool_calls`.

## Realtime and voice

- Realtime: `src/model/openai_realtime.rs` and `src/realtime.rs` provide building blocks for SSE streaming flows.
- Voice: `src/voice/` includes STT (`audio/transcriptions`) and TTS (`audio/speech`) clients, with a pipeline for simple voice interactions.

## Plugins

Plugins are dynamically loadable and initialized at runtime:

- Loader/registry in `src/plugin/loader.rs` and `src/plugin/mod.rs`.
- Default search path is `~/.config/openai_agents/plugins` (configurable via `plugins_path`).

## Parity with openai-agents-python

What’s implemented:

- Model coverage: OpenAI Chat Completions, Harmony-aligned OSS Responses, LiteLLM pass-through.
- Tool orchestration: `tool_calls` and legacy `function_call`; optional tool-choice; parallel tool-calls control.
- Sessions/memory scaffolding and tracing hooks; explicit errors instead of silent fallbacks.
- Realtime SSE client and voice pipeline (STT/TTS) counterparts.

Partial/roadmap:

- Agent-as-tool helper ( ergonomic wrapper ).
- Guardrails and handoffs deeper integration in the runner.
- Rich tracing exporters and span coverage.
- Hosted tool adapters and more bundled tools.

## No mocks or fallbacks — by design

- The env loader doesn’t invent values and does not inject provider defaults.
- Model calls don’t auto-retry or downgrade; errors are explicit and surfaced early.
- Comments and docs avoid "fallback" semantics; behaviors are intentional and visible.

## Development

Build, test, and run locally:

```bash
cargo test -q
cargo run
```

Optional config file example (`config.yaml`):

```yaml
base_url: "http://localhost:8000/v1"
model: "openai/gpt-oss-120b"
log_level: "info"
plugins_path: "~/.config/openai_agents/plugins"
```

You can override any file keys with `OPENAI_AGENTS__<KEY>` (double underscore maps to nested keys) or use the global provider-style variables documented above.

## Repository layout (selected)

```text
src/
  lib.rs           # crate exports
  main.rs          # CLI: loads config, starts MCP server, runs agents
  agent/           # Agent traits, runtime, runner
  model/           # Models: OpenAI Chat, OSS Responses, LiteLLM, Realtime
  tools/           # Tool trait + registry + function tools
  plugin/          # Plugin loader/registry
  config/          # Config schema + loader (env-first)
  realtime/, voice/  # Streaming + STT/TTS
```

## License

MIT — see `LICENSE` for details. Contributions welcome.

See also: `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`.
