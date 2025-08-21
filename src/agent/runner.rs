use crate::agent::traits::{Agent, AgentContext};
use crate::error::AgentError;
use crate::model::Model;
use crate::results::RunResult;
use crate::tools::registry::ToolRegistry;
use crate::utils::env::var_bool;
use serde_json::json;

/// Tool-use behavior modes analogous to the Python SDK behavior.
#[derive(Clone)]
pub enum ToolUseBehavior {
    RunLlmAgain,
    StopOnFirstTool,
    StopAtTools(Vec<String>),
    Custom(BehaviorFn),
}

/// Result of deciding whether tool outputs are final.
#[derive(Clone, Debug, Default)]
pub struct ToolsToFinalOutputResult {
    pub is_final_output: bool,
    pub final_output: Option<String>,
}

/// Callback to decide final output based on tool results.
pub type BehaviorFn = std::sync::Arc<
    dyn Fn(&AgentContext, &[(String, String)]) -> ToolsToFinalOutputResult + Send + Sync,
>;

/// Minimal runner scaffold to orchestrate a single agent call.
pub struct Runner;

impl Runner {
    pub async fn run<A: Agent>(agent: &A, ctx: &AgentContext) -> Result<(), AgentError> {
        // For now, just run the agent.
        agent.run(ctx).await
    }

    /// Execute tools and collect outputs; may short-circuit based on behavior.
    pub async fn run_tools_collect(
        registry: &ToolRegistry,
        ctx: &AgentContext,
        input: &str,
        behavior: &ToolUseBehavior,
    ) -> Result<(Vec<(String, String)>, Option<String>), AgentError> {
        let mut results: Vec<(String, String)> = Vec::new();
        for tool in registry.all() {
            if tool.is_enabled(ctx).await {
                let name = tool.name().to_string();
                let out = tool.call(input).await?;
                // If StopAtTools, check if this tool is among the stop set.
                match behavior {
                    ToolUseBehavior::StopOnFirstTool => {
                        return Ok((vec![(name, out.clone())], Some(out)));
                    }
                    ToolUseBehavior::StopAtTools(stop_list) => {
                        if stop_list.iter().any(|n| n == &name) {
                            return Ok((vec![(name, out.clone())], Some(out)));
                        }
                    }
                    _ => {}
                }
                results.push((name, out));
            }
        }
        Ok((results, None))
    }

    /// Minimal agent loop: try tools first (optional), then call the model with instructions.
    pub async fn run_agent_with_model<M: Model + ?Sized>(
        model: &M,
        ctx: &AgentContext,
        instructions: Option<&str>,
        input: &str,
        behavior: ToolUseBehavior,
    ) -> Result<RunResult, AgentError> {
        // First attempt: run tools if any are enabled.
        let (tool_results, early_final) =
            Self::run_tools_collect(&ctx.tools, ctx, input, &behavior).await?;
        if let Some(out) = early_final {
            if !tool_results.is_empty() {
                tracing::info!(
                    target: "runner",
                    tool_count = tool_results.len(),
                    tools = %serde_json::json!(tool_results),
                    "early stop with tool outputs"
                );
            }
            return Ok(RunResult {
                id: None,
                text: Some(out),
                tool_outputs: tool_results,
            });
        }

        // If custom behavior, allow it to decide final output.
        if let ToolUseBehavior::Custom(decider) = &behavior {
            let res = decider(ctx, &tool_results);
            if res.is_final_output {
                return Ok(RunResult {
                    id: None,
                    text: res.final_output,
                    tool_outputs: vec![],
                });
            }
        }

        // No tool output or no tools available; call the model directly.
        let combined_input = if tool_results.is_empty() {
            input.to_string()
        } else {
            let mut agg = String::from(input);
            for (name, out) in &tool_results {
                agg.push_str("\n\nTool ");
                agg.push_str(name);
                agg.push_str(" output:\n");
                agg.push_str(out);
            }
            agg
        };

        let max_turns = 3;
        // Build initial chat messages (system + user)
        let mut messages: Vec<serde_json::Value> = Vec::new();
        if let Some(sys) = instructions {
            messages.push(json!({"role": "system", "content": sys}));
        }
        messages.push(json!({"role": "user", "content": input}));
        // Collect OpenAI tool specs for enabled tools
        let mut tool_specs: Vec<serde_json::Value> = Vec::new();
        for t in ctx.tools.all() {
            if t.openai_tool_spec().is_some() && t.is_enabled(ctx).await {
                if let Some(spec) = t.openai_tool_spec() {
                    tool_specs.push(spec);
                }
            }
        }
        // Compatibility: allow disabling passing tools to the LLM via env flag
        let disable_tools_in_llm = var_bool("VLLM_DISABLE_TOOLS_IN_LLM", false);
        let mut previous_response_id: Option<String> = None;
        let disable_tools_next_turn = false;
        let mut collected_tool_outputs: Vec<(String, String)> = tool_results.clone();
        for _turn in 0..max_turns {
            let resp = model
                .get_response(
                    instructions,
                    &combined_input,
                    None,
                    Some(&messages),
                    if tool_specs.is_empty() || disable_tools_in_llm || disable_tools_next_turn {
                        None
                    } else {
                        Some(&tool_specs)
                    },
                    None,
                    None,
                    None,
                    false,
                    previous_response_id.as_deref(),
                    None,
                )
                .await?;

            if let Some(rid) = &resp.id {
                previous_response_id = Some(rid.clone());
            }

            if resp.tool_calls.is_empty() {
                if let Some(text) = &resp.text {
                    messages.push(json!({"role": "assistant", "content": text}));
                }
                return Ok(RunResult {
                    id: resp.id,
                    text: resp.text,
                    tool_outputs: collected_tool_outputs,
                });
            }

            // Add assistant message for proper round-trip.
            let all_have_ids = resp
                .tool_calls
                .iter()
                .all(|tc| tc.id.is_some() || tc.call_id.is_some());
            if all_have_ids {
                // Use tool_calls schema; set content to null per Harmony compatibility
                messages.push(json!({
                    "role": "assistant",
                    "content": serde_json::Value::Null,
                    "tool_calls": resp.tool_calls.iter().map(|tc| json!({
                        "id": tc.id.clone().or(tc.call_id.clone()),
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                        "call_id": tc.call_id,
                    })).collect::<Vec<_>>()
                }));
            } else {
                // Legacy function_call schema supports only one function call per message.
                if let Some(tc0) = resp.tool_calls.first() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": serde_json::Value::Null,
                        "function_call": {"name": tc0.name, "arguments": tc0.arguments},
                    }));
                }
            }

            // Execute requested tool calls if available.
            let mut executed_any_tool = false;
            let mut missing_tools: Vec<String> = Vec::new();
            let mut _new_tool_outputs: Vec<(String, String)> = Vec::new();
            for tc in resp.tool_calls {
                if let Some(tool) = ctx.tools.get_by_name(&tc.name) {
                    if tool.is_enabled(ctx).await {
                        let out = tool
                            .call_with_context(ctx, tc.id.as_deref(), &tc.arguments)
                            .await?;
                        // Append a proper tool message for the next model turn.
                        if let Some(link_id) = tc.call_id.clone().or(tc.id.clone()) {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": link_id,
                                "content": out
                            }));
                        } else {
                            // Legacy function message
                            messages.push(json!({
                                "role": "function",
                                "name": tc.name,
                                "content": out
                            }));
                        }
                        _new_tool_outputs.push((tc.name.clone(), out.clone()));
                        executed_any_tool = true;
                    } else {
                        missing_tools.push(tc.name.clone());
                    }
                } else {
                    missing_tools.push(tc.name.clone());
                }
            }
            if !_new_tool_outputs.is_empty() {
                collected_tool_outputs.extend(_new_tool_outputs);
                tracing::info!(
                    target: "runner",
                    total_tools = collected_tool_outputs.len(),
                    last_batch = %serde_json::json!(collected_tool_outputs),
                    "collected tool outputs"
                );
            }

            if !missing_tools.is_empty() {
                return Err(AgentError::Other(format!(
                    "model requested unknown or disabled tools: {}",
                    missing_tools.join(", ")
                )));
            }
            if !executed_any_tool {
                return Err(AgentError::Other(
                    "model returned tool_calls but none could be executed".into(),
                ));
            }
            // combined_input remains the same; messages carry the tool outputs.
        }

        // Final model call to produce an answer after tool outputs were added to messages.
        let resp = model
            .get_response(
                instructions,
                &combined_input,
                None,
                Some(&messages),
                if tool_specs.is_empty() || disable_tools_in_llm {
                    None
                } else {
                    Some(&tool_specs)
                },
                None,
                None,
                None,
                false,
                previous_response_id.as_deref(),
                None,
            )
            .await?;
        let res = RunResult {
            id: resp.id,
            text: resp.text,
            tool_outputs: collected_tool_outputs,
        };
        if !res.tool_outputs.is_empty() {
            tracing::info!(
                target: "runner",
                tool_count = res.tool_outputs.len(),
                tools = %serde_json::json!(res.tool_outputs),
                "final result with tool outputs"
            );
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::traits::AgentContext;
    use crate::client::OpenAiClient;

    use async_trait::async_trait;
    use std::sync::Arc;

    struct EchoTool;
    #[async_trait]
    impl crate::tools::traits::Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        async fn call(&self, input: &str) -> Result<String, crate::error::AgentError> {
            Ok(input.to_string())
        }
    }

    use crate::model::openai_chat::OpenAiChat;

    #[tokio::test]
    async fn runner_returns_tool_outputs_on_stop_first() {
        // Load from .env if present and use env overrides; then fallback for test defaults
        let _ = dotenvy::dotenv();
        let mut cfg = crate::config::load_from_env();
        if cfg.base_url.is_empty() {
            cfg.base_url = "http://localhost".into();
        }
        if cfg.model.is_empty() {
            cfg.model = "openai/gpt-oss-120b".into();
        }
        // Avoid auth in unit test
        cfg.api_key = String::new();
        let client = Arc::new(OpenAiClient::new(cfg.clone()));
        let plugins = Arc::new(crate::plugin::loader::PluginRegistry::new());
        let mut reg = crate::tools::registry::ToolRegistry::new();
        reg.register(EchoTool);
        let ctx = AgentContext {
            config: Arc::new(cfg.clone()),
            client,
            plugins,
            tools: Arc::new(reg),
        };
        let model = OpenAiChat::new(cfg).without_auth();
        let res = Runner::run_agent_with_model(
            &model,
            &ctx,
            None,
            "hi",
            ToolUseBehavior::StopOnFirstTool,
        )
        .await
        .unwrap();
        assert_eq!(res.text.as_deref(), Some("hi"));
        assert_eq!(res.tool_outputs.len(), 1);
        assert_eq!(res.tool_outputs[0].0, "echo");
        assert_eq!(res.tool_outputs[0].1, "hi");
    }
}
