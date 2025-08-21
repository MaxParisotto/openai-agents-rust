pub mod litellm;
pub mod openai_chat;
pub mod openai_realtime;

use crate::error::AgentError;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;

/// Core trait for all model implementations.
#[async_trait]
pub trait Model: Send + Sync {
    /// Perform a generation request and return the raw response as a string.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError>;

    /// Rich response method (scaffold) to align with Python's get_response signature.
    /// Default implementation wraps `generate` with a simple string prompt.
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &str,
        _model_settings: Option<HashMap<String, String>>, // placeholder
        _messages: Option<&[Value]>,                      // chat messages for full fidelity
        _tools: Option<&[Value]>,                         // OpenAI tool specs
        _tool_choice: Option<Value>,                      // tool_choice config
        _output_schema: Option<&str>,                     // placeholder
        _handoffs: Option<&[String]>,                     // placeholder
        _tracing_enabled: bool,
        _previous_response_id: Option<&str>,
        _prompt_config: Option<&str>,
    ) -> Result<ModelResponse, AgentError> {
        // Fallback: collapse into a single prompt string.
        let text = if let Some(messages) = _messages {
            // Try to find the last user message content as prompt.
            let last_user = messages.iter().rev().find_map(|m| {
                let role = m.get("role")?.as_str()?;
                if role == "user" {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            });
            match last_user {
                Some(s) => self.generate(&s).await?,
                None => {
                    let mut s = String::new();
                    if let Some(sys) = system_instructions {
                        s.push_str(sys);
                        s.push_str("\n\n");
                    }
                    s.push_str(input);
                    self.generate(&s).await?
                }
            }
        } else {
            let mut s = String::new();
            if let Some(sys) = system_instructions {
                s.push_str(sys);
                s.push_str("\n\n");
            }
            s.push_str(input);
            self.generate(&s).await?
        };
        Ok(ModelResponse {
            id: None,
            text: Some(text),
            tool_calls: vec![],
        })
    }
}

/// Simplified model response structure with tool-call scaffold.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub id: Option<String>,
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

/// Streamed event variants.
#[derive(Debug, Clone)]
pub enum ModelStreamEvent {
    TextDelta(String),
    ToolCallDelta(ToolCall),
}
