# OpenAI Agents SDK Rust Port – Parity Checklist

This document tracks 1:1 parity between the Python SDK (`openai-agents-python/src/agents`) and the Rust port (`src/`).

Status legend: [ ] Todo, [~] Partial, [x] Done

## Core Agent

- [~] Agent configuration (instructions, prompt, tools, handoffs, model selection, model settings, guardrails, output type, hooks, tool-use behavior)
- [ ] Agent `as_tool` equivalent
- [ ] `get_system_prompt` / `get_prompt` behavior
- [ ] `get_all_tools` incl. MCP tools
- [ ] Clone/shallow copy semantics

## Runner / Orchestration

- [x] Single-run engine that builds model inputs and returns structured result
- [ ] Tool invocation roundtrip and configurable tool-use behavior
- [ ] Handoffs delegation
- [ ] Finalization and output extraction

## Tools & MCP

- [x] Tool trait and basic registry
- [ ] FunctionTool decorator/adapter and enablement predicates
- [ ] MCP client for tool discovery and (optional) strict schema conversion

## Models

- [~] Expanded model interface for `get_response` and streaming
- [ ] Responses compatibility (tools, handoffs, prompts, output schema, tracing)
- [ ] OpenAI Chat/Responses implementations

## Prompts & Model Settings

- [ ] Prompt configuration and resolution
- [ ] Model settings parity

## Guardrails

- [ ] Input guardrails
- [ ] Output guardrails

## Tracing

- [~] Tracing init
- [ ] Spans/processors/provider parity and data-inclusion controls

## Realtime

- [~] Streaming traits and mock
- [ ] Event types and session orchestration

## Output Schema & Results

- [~] Placeholder output schema trait and run result struct
- [ ] Strict/non-strict schemas and helpers

## Handoffs

- [ ] Handoff struct and delegation integration

## Voice / Extensions / Utils

- [ ] Voice parity
- [ ] Extension interfaces parity

---

Next steps prioritized:

1) Flesh out tool invocation and tool-use behavior in runner.
2) Add MCP client abstraction and integrate into agent tool discovery.
3) Expand model response types to carry tool calls and deltas; wire to OpenAI endpoint.
