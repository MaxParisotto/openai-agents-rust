use crate::agent::traits::{Agent, AgentContext};
use crate::error::AgentError;
use crate::model::Model;
use crate::results::RunResult;
use crate::tools::registry::ToolRegistry;
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
            return Ok(RunResult { text: Some(out) });
        }

        // If custom behavior, allow it to decide final output.
        if let ToolUseBehavior::Custom(decider) = &behavior {
            let res = decider(ctx, &tool_results);
            if res.is_final_output {
                return Ok(RunResult {
                    text: res.final_output,
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
        let disable_tools_in_llm = std::env::var("VLLM_DISABLE_TOOLS_IN_LLM")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false);
        for _turn in 0..max_turns {
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
                    None,
                    None,
                )
                .await?;

            if resp.tool_calls.is_empty() {
                if let Some(text) = &resp.text {
                    messages.push(json!({"role": "assistant", "content": text}));
                }
                return Ok(RunResult { text: resp.text });
            }

            // Add assistant message for proper round-trip.
            let all_have_ids = resp.tool_calls.iter().all(|tc| tc.id.is_some());
            if all_have_ids {
                // Use tool_calls schema; set content to null per Harmony compatibility
                messages.push(json!({
                    "role": "assistant",
                    "content": serde_json::Value::Null,
                    "tool_calls": resp.tool_calls.iter().map(|tc| json!({
                        "id": tc.id,
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
        for tc in resp.tool_calls {
                if let Some(tool) = ctx.tools.get_by_name(&tc.name) {
                    if tool.is_enabled(ctx).await {
                        let out = tool
                            .call_with_context(ctx, tc.id.as_deref(), &tc.arguments)
                            .await?;
                        // Append a proper tool message for the next model turn.
            let link_id = tc.call_id.clone().or(tc.id.clone());
            if link_id.is_some() {
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
                        executed_any_tool = true;
                    }
                }
            }

            if !executed_any_tool {
                break;
            }
            // combined_input remains the same; messages carry the tool outputs.
        }

        // Fallback final call without tool calls processed.
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
                None,
                None,
            )
            .await?;
        Ok(RunResult { text: resp.text })
    }
}
