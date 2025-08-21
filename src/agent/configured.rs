use crate::agent::runner::Runner;
use crate::agent::traits::{Agent, AgentContext};
use crate::agent::types::AgentConfig;
use crate::error::AgentError;
use crate::model::Model;
use async_trait::async_trait;

/// A configurable agent that calls the model and optionally executes tools.
pub struct ConfiguredAgent<M: Model + 'static> {
    pub config: AgentConfig,
    pub model: M,
}

impl<M: Model + 'static> ConfiguredAgent<M> {
    pub fn new(name: impl Into<String>, model: M) -> Self {
        Self {
            config: AgentConfig::new(name),
            model,
        }
    }
}

#[async_trait]
impl<M: Model + 'static> Agent for ConfiguredAgent<M> {
    async fn run(&self, ctx: &AgentContext) -> Result<(), AgentError> {
        let default_instructions = format!("Agent: {}", self.config.name);
        let instructions = self
            .config
            .instructions
            .as_deref()
            .or(Some(default_instructions.as_str()));
        let behavior = self.config.tool_use_behavior.clone();
        let res =
            Runner::run_agent_with_model(&self.model, ctx, instructions, "say hello", behavior)
                .await?;
        tracing::info!(
            "ConfiguredAgent({}) -> {}",
            self.config.name,
            res.text.unwrap_or_default()
        );
        Ok(())
    }
}
