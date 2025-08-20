use async_trait::async_trait;
use crate::error::AgentError;
use crate::config::Config;
use crate::model::Model;
use reqwest::Client;

/// Simple OpenAI Chat model implementation.
pub struct OpenAiChat {
    client: Client,
    config: Config,
}

impl OpenAiChat {
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
impl Model for OpenAiChat {
    /// Sends a chat completion request to the OpenAI API.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError> {
        // Placeholder request – in a real implementation you would construct the
        // appropriate JSON payload and deserialize the response.
        let url = format!(
            "https://api.openai.com/v1/chat/completions"
        );
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [{ "role": "user", "content": prompt }],
                "max_tokens": 512,
            }))
            .send()
            .await
            .map_err(AgentError::from)?;

        let text = resp
            .text()
            .await
            .map_err(AgentError::from)?;

        Ok(text)
    }
}