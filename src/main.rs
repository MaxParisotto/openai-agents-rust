use std::{net::SocketAddr, sync::Arc};

use tokio::sync::broadcast;

use openai_agents_rust::agent::{self, runtime::AgentRuntime};
use openai_agents_rust::client::OpenAiClient;
use openai_agents_rust::config::load_from_path;
use openai_agents_rust::error::AgentError;
use openai_agents_rust::mcp_server::{self, AppState, start_server};
use openai_agents_rust::plugin::loader::PluginRegistry;
use openai_agents_rust::tracing::init_tracing;

#[tokio::main]
async fn main() -> Result<(), AgentError> {
    // Load .env for local development (non-fatal if missing)
    let _ = dotenvy::dotenv();
    // Initialise global tracing (respect RUST_LOG).
    init_tracing();

    // Load configuration – defaults to ./config.yaml or the path set in
    // the OPENAI_AGENTS_CONFIG environment variable.
    let config_path =
        std::env::var("OPENAI_AGENTS_CONFIG").unwrap_or_else(|_| "config.yaml".to_string());
    let config = load_from_path(&config_path)?;

    // Initialise core components.
    let client = Arc::new(OpenAiClient::new(config.clone()));
    let plugins = Arc::new(PluginRegistry::load_from_dir(&config.plugins_path)?);
    let (broadcaster, _receiver) = broadcast::channel(100);

    // Shared application state for the MCP server.
    let state = AppState {
        config: Arc::new(config.clone()),
        client,
        plugins,
        broadcaster,
    };

    // Build the agent runtime and register the default agent.
    let mut runtime = AgentRuntime::new(config);
    openai_agents_rust::agent::register_default_agent(&mut runtime);

    // Start the agent runtime in a background task.
    let agents_handle = tokio::spawn(async move {
        // Propagate any AgentError from the runtime start.
        runtime.start().await.map_err(|e| {
            tracing::error!("Agent runtime error: {}", e);
            e
        })
    });

    // Start the MCP HTTP server.
    let addr: SocketAddr = "127.0.0.1:8080".parse().expect("Invalid bind address");
    tracing::info!("Starting MCP server at http://{addr}");
    let server_handle = tokio::spawn(start_server(state, addr));

    // Wait for both tasks to finish (they run indefinitely until the process is stopped).
    // Await both background tasks and propagate any errors.
    let agents_res = agents_handle
        .await
        .map_err(|e| AgentError::Other(e.to_string()))?;
    let server_res = server_handle
        .await
        .map_err(|e| AgentError::Other(e.to_string()))?;
    // Each task returns Result<(), AgentError>.
    agents_res?;
    server_res?;

    Ok(())
}
