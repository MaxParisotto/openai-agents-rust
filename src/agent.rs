use async_trait::async_trait;
use std::sync::Arc;

mod runtime;
mod traits;

pub use runtime::AgentRuntime;
pub use traits::{Agent, AgentContext};

use crate::{
    model::openai_chat::OpenAiChat,
    model::Model,
    error::AgentError,
};

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
}