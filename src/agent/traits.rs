use crate::client::OpenAiClient;
use crate::config::Config;
use crate::error::AgentError;
use crate::plugin::loader::PluginRegistry;
use crate::tools::registry::ToolRegistry;
use async_trait::async_trait;
use std::sync::Arc;

/// Core Agent trait – all agents must implement this.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run the agent with the provided context.
    async fn run(&self, ctx: &AgentContext) -> Result<(), AgentError>;
}

/// Context passed to each agent during execution.
pub struct AgentContext {
    pub config: Arc<Config>,
    pub client: Arc<OpenAiClient>,
    pub plugins: Arc<PluginRegistry>,
    pub tools: Arc<ToolRegistry>,
}
