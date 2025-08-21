mod request;

use crate::client::request::{ChatRequest, ChatResponse};
use crate::config::Config;
use crate::error::AgentError;
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
            .user_agent("openai-agents-rust") // TODO: ensure the OpenAiClient respects Config.base_url if/when used for direct calls.
            .build()
            .expect("Failed to build reqwest client");
        Self {
            http,
            config: Arc::new(config),
        }
    }

    /// Send a chat completion request.
    pub async fn chat_completion(&self, req: ChatRequest) -> Result<ChatResponse, AgentError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut rb = self.http.post(url);
        if !self.config.api_key.is_empty() {
            rb = rb.bearer_auth(&self.config.api_key);
        }
        let response = rb.json(&req).send().await.map_err(AgentError::from)?;

        let chat_resp = response
            .json::<ChatResponse>()
            .await
            .map_err(AgentError::from)?;
        Ok(chat_resp)
    }
}
