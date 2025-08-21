use async_trait::async_trait;
use crate::error::AgentError;
use crate::config::Config;
use crate::model::Model;
use reqwest::Client;

/// Simple wrapper around any LLM that follows the OpenAI‑compatible API.
pub struct LiteLLM {
    client: Client,
    config: Config,
}

impl LiteLLM {
    /// Create a new instance with the given configuration.
    pub fn new(config: Config) -> Self {
        let client = Client::builder()
            .user_agent("openai-agents-rust")
            .build()
            .expect("Failed to build reqwest client");
        Self { client, config }
    }
}

#[async_trait]
impl Model for LiteLLM {
    /// Sends a chat completion request to the configured endpoint.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError> {
        // For demonstration we reuse the OpenAI chat endpoint.
        let url = "https://api.openai.com/v1/chat/completions";
        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [{ "role": "user", "content": prompt }],
                "max_tokens": 512,
            }))
            .send()
            .await
            .map_err(AgentError::from)?;

        let text = resp.text().await.map_err(AgentError::from)?;
        Ok(text)
    }
}