pub mod configured;
pub mod runner;
pub mod runtime;
pub mod traits;
pub mod types;

pub use runner::{Runner, ToolUseBehavior};
pub use runtime::AgentRuntime;
pub use traits::{Agent, AgentContext};
pub use types::AgentConfig;

use async_trait::async_trait;

use crate::{error::AgentError, model::Model, model::openai_chat::OpenAiChat};

/// Simple echo agent that forwards a fixed prompt to a model.
pub struct EchoAgent {
    model: Box<dyn Model>,
}

impl EchoAgent {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self { model }
    }
}

#[async_trait]
impl Agent for EchoAgent {
    async fn run(&self, _ctx: &AgentContext) -> Result<(), AgentError> {
        let response = self.model.generate("Hello from EchoAgent").await?;
        tracing::info!("EchoAgent response: {}", response);
        Ok(())
    }
}

/// Register a default EchoAgent in the provided runtime.
pub fn register_default_agent(runtime: &mut AgentRuntime) {
    let model = Box::new(OpenAiChat::new((*runtime.config).clone()));
    let agent = EchoAgent::new(model);
    runtime.register(agent);

    // Also register a configured agent using the realtime placeholder model.
    let rt_model = crate::model::openai_realtime::OpenAiRealtime::new((*runtime.config).clone());
    let configured = crate::agent::configured::ConfiguredAgent::new("configured", rt_model);
    runtime.register(configured);
}
