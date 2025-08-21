use std::sync::Arc;

use openai_agents_rust::agent::traits::AgentContext;
use openai_agents_rust::agent::{Runner, ToolUseBehavior};
use openai_agents_rust::client::OpenAiClient;
use openai_agents_rust::config::load_from_path;
use openai_agents_rust::model::openai_chat::OpenAiChat;
use openai_agents_rust::results::RunResult;
use openai_agents_rust::tools::function::FunctionTool;
use openai_agents_rust::tools::registry::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    // Init simple logging for debugging
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    // Load config (env-only is fine; config.yaml is optional)
    let config_path =
        std::env::var("OPENAI_AGENTS_CONFIG").unwrap_or_else(|_| "config.yaml".into());
    let config = load_from_path(&config_path)?;
    println!("Using model: {}", config.model);
    println!("Base URL: {}", config.base_url);

    // Build model pointing to vLLM-compatible endpoint
    let model = OpenAiChat::new(config.clone());

    // Prepare tools registry with an uppercase tool expecting {"text": string}
    let mut registry = ToolRegistry::new();
    let no_tools = std::env::var("VLLM_NO_TOOLS")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if !no_tools {
        registry.register(FunctionTool::new(
            "uppercase",
            "Uppercase the input string. Expects JSON: {\"text\": string}",
            |s| {
                let text = serde_json::from_str::<serde_json::Value>(s)
                    .ok()
                    .and_then(|v| {
                        v.get("text")
                            .and_then(|t| t.as_str())
                            .map(|t| t.to_string())
                    })
                    .unwrap_or_else(|| s.to_string());
                Ok(text.to_uppercase())
            },
        ));
    } else {
        println!("Tool registry disabled via VLLM_NO_TOOLS=1");
    }

    // Build a minimal AgentContext
    let ctx = AgentContext {
        config: Arc::new(config.clone()),
        client: Arc::new(OpenAiClient::new(config.clone())),
        plugins: Arc::new(openai_agents_rust::plugin::loader::PluginRegistry::new()),
        tools: Arc::new(registry),
    };

    // Simple instruction to encourage tool use
    let instructions = Some(
        "You can call tools. If the user asks to shout or uppercase, call the `uppercase` tool with {\"text\": string}.",
    );
    let input = if no_tools {
        "Say hello"
    } else {
        "Please shout: rust agents"
    };

    let res: RunResult = Runner::run_agent_with_model(
        &model,
        &ctx,
        instructions,
        input,
        ToolUseBehavior::RunLlmAgain,
    )
    .await
    .map_err(|e| format!("runner error: {e}"))?;
    println!("Model result: {:?}", res.text);
    Ok(())
}
