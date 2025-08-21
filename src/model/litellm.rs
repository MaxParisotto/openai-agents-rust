use crate::config::Config;
use crate::error::AgentError;
use crate::model::{Model, ModelResponse, ToolCall};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

/// Simple wrapper around any LLM that follows the OpenAI‑compatible API.
pub struct LiteLLM {
    client: Client,
    config: Config,
    base_url: String,
    auth_token: Option<String>,
}

impl LiteLLM {
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
    Self { client, config, base_url, auth_token }
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

    /// Set a custom auth token (Bearer). Use None to disable.
    pub fn with_auth_token(mut self, token: Option<impl Into<String>>) -> Self {
        self.auth_token = token.map(|t| t.into());
        self
    }
}

#[async_trait]
impl Model for LiteLLM {
    /// Sends a chat completion request to the configured endpoint.
    async fn generate(&self, prompt: &str) -> Result<String, AgentError> {
        // For demonstration we reuse the OpenAI chat endpoint.
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
                    "temperature": 0.2,
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

        #[derive(Deserialize)]
        struct FunctionCall {
            name: String,
            arguments: serde_json::Value,
        }
        #[derive(Deserialize)]
        struct ToolCallJson {
            #[serde(rename = "type")]
            _type: Option<String>,
            id: Option<String>,
            call_id: Option<String>,
            function: Option<FunctionCall>,
        }
        #[derive(Deserialize)]
        struct Message {
            content: Option<String>,
            tool_calls: Option<Vec<ToolCallJson>>,
            function_call: Option<FunctionCall>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
        }
        #[derive(Deserialize)]
        struct ChatCompletion {
            choices: Vec<Choice>,
        }

        let status = resp.status();
        let body_text = resp.text().await.map_err(AgentError::from)?;
        if !status.is_success() {
            return Err(AgentError::Other(format!(
                "HTTP {} error: {}",
                status, body_text
            )));
        }
        match serde_json::from_str::<ChatCompletion>(&body_text) {
            Ok(body) => {
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
                                    arguments: match func.arguments {
                                        serde_json::Value::String(s) => s,
                                        other => other.to_string(),
                                    },
                                    call_id: tc.call_id,
                                });
                            }
                        }
                    } else if let Some(func) = first.message.function_call {
                        tool_calls.push(ToolCall {
                            id: None,
                            name: func.name,
                            arguments: match func.arguments {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            },
                            call_id: None,
                        });
                    }
                }
                Ok(ModelResponse {
                    id: None,
                    text,
                    tool_calls,
                })
            }
            Err(_) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body_text) {
                    let text = v
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|arr| arr.get(0))
                        .and_then(|c0| {
                            c0.get("message")
                                .and_then(|m| m.get("content"))
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                                .or_else(|| {
                                    c0.get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                })
                        });
                    return Ok(ModelResponse {
                        id: None,
                        text,
                        tool_calls: vec![],
                    });
                }
                Ok(ModelResponse {
                    id: None,
                    text: Some(body_text),
                    tool_calls: vec![],
                })
            }
        }
    }
}
