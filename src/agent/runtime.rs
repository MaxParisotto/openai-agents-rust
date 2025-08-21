use crate::agent::traits::{Agent, AgentContext};
use crate::client::OpenAiClient;
use crate::config::Config;
use crate::error::AgentError;
use crate::plugin::loader::PluginRegistry;
use crate::tools::registry::ToolRegistry;
use std::sync::Arc;

/// Runtime that holds a collection of agents and executes them.
pub struct AgentRuntime {
    pub agents: Vec<Arc<dyn Agent>>,
    pub config: Arc<Config>,
    pub client: Arc<OpenAiClient>,
    pub plugins: Arc<PluginRegistry>,
    pub tools: Arc<ToolRegistry>,
}

impl AgentRuntime {
    /// Create a new runtime from a configuration.
    pub fn new(config: Config) -> Self {
        let config = Arc::new(config);
        let client = Arc::new(OpenAiClient::new((*config).clone()));
        let plugins = Arc::new(
            PluginRegistry::load_from_dir(&config.plugins_path)
                .unwrap_or_else(|_| PluginRegistry::new()),
        );
        let mut tools = ToolRegistry::new();
        // Demo tool – uppercase
        tools.register(crate::tools::function::FunctionTool::new(
            "uppercase",
            "Uppercase the input string. Expects JSON: {\"text\": string}",
            |s| {
                // Try to parse OpenAI tool call args as JSON: {"text": "..."}
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

        Self {
            agents: Vec::new(),
            config,
            client,
            plugins,
            tools: Arc::new(tools),
        }
    }

    /// Register an agent with the runtime.
    pub fn register<A: Agent + 'static>(&mut self, agent: A) {
        self.agents.push(Arc::new(agent));
    }

    /// Start all registered agents.
    pub async fn start(&self) -> Result<(), AgentError> {
        let ctx = AgentContext {
            config: Arc::clone(&self.config),
            client: Arc::clone(&self.client),
            plugins: Arc::clone(&self.plugins),
            tools: Arc::clone(&self.tools),
        };
        for agent in &self.agents {
            agent.run(&ctx).await?;
        }
        Ok(())
    }
}
