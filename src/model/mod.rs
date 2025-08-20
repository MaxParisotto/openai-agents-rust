pub mod openai_chat;
pub mod openai_realtime;
pub mod litellm;

use async_trait::async_trait;
use crate::error::AgentError;

/// Core trait for all model implementations.
#[async_trait]
pub trait Model: Send + Sync {
    /// Perform a generation request and return the raw response as a string.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError>;
}