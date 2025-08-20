use crate::config::Config;
use crate::error::AgentError;
use crate::client::request::{ChatRequest, ChatResponse};
use reqwest::Client;
use std::sync::Arc;

/// Wrapper around the OpenAI HTTP client.
pub struct OpenAiClient {
    pub http: Client,
    pub config: Arc<Config>,
}

impl OpenAiClient {
    /// Create a new client from the given configuration.
    pub fn new(config: Config) -> Self {
        let http = Client::builder()
            .user_agent("openai-agents-rust")
            .build()
            .expect("Failed to build reqwest client");
        Self {
            http,
            config: Arc::new(config),
        }
    }

    /// Send a chat completion request.
    pub async fn chat_completion(
        &self,
        req: ChatRequest,
    ) -> Result<ChatResponse, AgentError> {
        let url = "https://api.openai.com/v1/chat/completions";
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&req)
            .send()
            .await
            .map_err(AgentError::from)?;

        let chat_resp = response
            .json::<ChatResponse>()
            .await
            .map_err(AgentError::from)?;
        Ok(chat_resp)
    }
}