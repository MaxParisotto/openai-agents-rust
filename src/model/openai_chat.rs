use crate::config::Config;
use crate::error::AgentError;
use crate::model::{Model, ModelResponse, ToolCall};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

/// Simple OpenAI Chat model implementation.
pub struct OpenAiChat {
    client: Client,
    config: Config,
    base_url: String,
    auth_token: Option<String>,
}

impl OpenAiChat {
    /// Create a new instance with the given configuration.
    pub fn new(config: Config) -> Self {
        let client = Client::builder()
            .user_agent("openai-agents-rust")
            .build()
            .expect("Failed to build reqwest client");
        let auth_token = if config.api_key.is_empty() {
            None
        } else {
            Some(config.api_key.clone())
        };
        let base_url = config.base_url.clone();
        Self {
            client,
            config,
            base_url,
            auth_token,
        }
    }

    /// Override the base URL (e.g., http://192.168.3.40:8000/v1)
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Disable authentication for local/open endpoints.
    pub fn without_auth(mut self) -> Self {
        self.auth_token = None;
        self
    }
}

// Response shapes for OpenAI chat.completions
#[derive(Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}
#[derive(Deserialize)]
struct ToolCallJson {
    #[serde(rename = "type")]
    _type: Option<String>,
    id: Option<String>,
    function: Option<FunctionCall>,
}
#[derive(Deserialize)]
struct Message {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallJson>>,
}
#[derive(Deserialize)]
struct Choice {
    message: Message,
}
#[derive(Deserialize)]
struct ChatCompletion {
    choices: Vec<Choice>,
}

fn parse_chat_completion(body: ChatCompletion) -> ModelResponse {
    let mut text: Option<String> = None;
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(first) = body.choices.into_iter().next() {
        text = first.message.content;
        if let Some(tcs) = first.message.tool_calls {
            for tc in tcs.into_iter() {
                if let Some(func) = tc.function {
                    tool_calls.push(ToolCall {
                        id: tc.id,
                        name: func.name,
                        arguments: func.arguments,
                    });
                }
            }
        }
    }
    ModelResponse {
        id: None,
        text,
        tool_calls,
    }
}

#[async_trait]
impl Model for OpenAiChat {
    /// Sends a chat completion request to the OpenAI API.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.client.post(&url);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req
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

    /// Rich response with basic parsing of text and tool calls.
    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        input: &str,
        _model_settings: Option<std::collections::HashMap<String, String>>,
        messages: Option<&[serde_json::Value]>,
        tools: Option<&[serde_json::Value]>,
        tool_choice: Option<serde_json::Value>,
        _output_schema: Option<&str>,
        _handoffs: Option<&[String]>,
        _tracing_enabled: bool,
        _previous_response_id: Option<&str>,
        _prompt_config: Option<&str>,
    ) -> Result<ModelResponse, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);
        // Build messages array if not provided.
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        if let Some(provided) = messages {
            msgs.extend_from_slice(provided);
        } else {
            if let Some(sys) = system_instructions {
                msgs.push(serde_json::json!({"role": "system", "content": sys}));
            }
            msgs.push(serde_json::json!({"role": "user", "content": input}));
        }

        let mut req = self.client.post(&url);
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req
            .json(&{
                let mut payload = serde_json::json!({
                    "model": self.config.model,
                    "messages": msgs,
                    "max_tokens": 512,
                });
                if let Some(t) = tools {
                    payload["tools"] = serde_json::Value::Array(t.to_vec());
                }
                if let Some(choice) = tool_choice {
                    payload["tool_choice"] = choice;
                }
                payload
            })
            .send()
            .await
            .map_err(AgentError::from)?;

        let body: ChatCompletion = resp.json().await.map_err(AgentError::from)?;
        Ok(parse_chat_completion(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_only() {
        let json = serde_json::json!({
            "choices": [
                { "message": { "content": "Hello", "tool_calls": null } }
            ]
        });
        let body: ChatCompletion = serde_json::from_value(json).unwrap();
        let res = parse_chat_completion(body);
        assert_eq!(res.text.as_deref(), Some("Hello"));
        assert!(res.tool_calls.is_empty());
    }

    #[test]
    fn parse_with_tool_calls() {
        let json = serde_json::json!({
            "choices": [
                { "message": {
                    "content": null,
                    "tool_calls": [
                        { "type": "function", "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" } },
                        { "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"NYC\"}" } }
                    ]
                }}
            ]
        });
        let body: ChatCompletion = serde_json::from_value(json).unwrap();
        let res = parse_chat_completion(body);
        assert!(res.text.is_none());
        assert_eq!(res.tool_calls.len(), 2);
        assert_eq!(res.tool_calls[0].name, "search");
        assert_eq!(res.tool_calls[0].arguments, "{\"q\":\"rust\"}");
        assert_eq!(res.tool_calls[1].name, "get_weather");
    }
}
