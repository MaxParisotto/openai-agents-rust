use crate::config::Config;
use crate::error::AgentError;
use crate::model::{Model, ModelResponse, ToolCall};
use crate::utils::env::var_bool;
use crate::utils::env::var_opt;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use tracing::debug;

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
    // Some servers return a JSON object instead of a stringified JSON.
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
    // Legacy function_call support (pre-tool_calls schema)
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
                        arguments: match func.arguments {
                            serde_json::Value::String(s) => s,
                            other => other.to_string(),
                        },
                        call_id: tc.call_id,
                    });
                }
            }
        } else if let Some(func) = first.message.function_call {
            // Legacy single function call
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
        let mut rb = self.client.post(&url);
        if let Some(token) = &self.auth_token {
            rb = rb.bearer_auth(token);
        }
        let resp = rb
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [{ "role": "user", "content": prompt }],
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

        // Env toggles for compatibility
        let minimal_payload = var_bool("VLLM_MIN_PAYLOAD", false);
        let force_functions = var_bool("VLLM_FORCE_FUNCTIONS", false);
        // Default to enabling parallel tool calls for Harmony unless explicitly disabled
        let disable_parallel = var_bool("VLLM_DISABLE_PARALLEL_TOOL_CALLS", false);
        // Optional override: Values: "auto", "none", "object:auto", "object:none"
        let tool_choice_override = var_opt("VLLM_TOOL_CHOICE");

        // Prepare payload
        let mut payload = if minimal_payload {
            serde_json::json!({
                "model": self.config.model,
                "messages": msgs,
            })
        } else {
            serde_json::json!({
                "model": self.config.model,
                "messages": msgs,
                "max_tokens": 512,
                "temperature": 0.2,
            })
        };
        let have_tools = if let Some(t) = tools {
            if force_functions {
                // Build legacy functions list
                let mut functions: Vec<serde_json::Value> = Vec::new();
                for tool in t.iter() {
                    if let Some(obj) = tool.as_object() {
                        if obj.get("type").and_then(|v| v.as_str()) == Some("function") {
                            if let Some(func) = obj.get("function") {
                                functions.push(func.clone());
                            }
                        }
                    }
                }
                if !functions.is_empty() {
                    payload["functions"] = serde_json::Value::Array(functions);
                    payload["function_call"] = serde_json::json!("auto");
                }
            } else if !minimal_payload {
                payload["tools"] = serde_json::Value::Array(t.to_vec());
                if !disable_parallel {
                    payload["parallel_tool_calls"] = serde_json::Value::Bool(true);
                }
            }
            true
        } else {
            false
        };
        if !minimal_payload {
            if let Some(choice) = &tool_choice {
                payload["tool_choice"] = choice.clone();
            } else if have_tools && !force_functions {
                // Harmony: omit tool_choice to let server decide, unless override is provided
                if let Some(tc) = tool_choice_override.as_deref() {
                    match tc {
                        "object:auto" => {
                            payload["tool_choice"] = serde_json::json!({"type": "auto"})
                        }
                        "object:none" => {
                            payload["tool_choice"] = serde_json::json!({"type": "none"})
                        }
                        "none" => payload["tool_choice"] = serde_json::json!("none"),
                        "auto" => payload["tool_choice"] = serde_json::json!("auto"),
                        _ => {}
                    }
                }
            }
        }

        if var_bool("VLLM_DEBUG_PAYLOAD", false) {
            if let Ok(pretty) = serde_json::to_string_pretty(&payload) {
                debug!(target: "openai_chat", payload = %pretty, "request payload");
            }
        }
        debug!(
            target: "openai_chat",
            url = %url,
            model = %self.config.model,
            have_tools = %have_tools,
            force_functions = %force_functions,
            tool_choice = %tool_choice.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "<none>".into()),
            messages_len = payload["messages"].as_array().map(|a| a.len()).unwrap_or(0),
            "sending chat.completions"
        );
        let mut req1 = self.client.post(&url);
        if let Some(token) = &self.auth_token {
            req1 = req1.bearer_auth(token);
        }
        let resp1 = req1.json(&payload).send().await.map_err(AgentError::from)?;
        let status = resp1.status();
        let body_text = resp1.text().await.map_err(AgentError::from)?;
        debug!(target: "openai_chat", "request completed status={} have_tools={}", status, have_tools);
        if !status.is_success() {
            let truncated = if body_text.len() > 2000 {
                format!("{}...<truncated>", &body_text[..2000])
            } else {
                body_text.clone()
            };
            return Err(AgentError::Other(format!(
                "chat.completions failed (status: {}). The server returned an error while tools={} force_functions={}. No automatic retries are performed. Verify the endpoint supports your requested schema (tool_calls vs functions) or adjust config (e.g., VLLM_FORCE_FUNCTIONS, VLLM_TOOL_CHOICE). Response body: {}",
                status,
                if have_tools { "enabled" } else { "disabled" },
                if var_bool("VLLM_FORCE_FUNCTIONS", false) {
                    "on"
                } else {
                    "off"
                },
                truncated
            )));
        }

        match serde_json::from_str::<ChatCompletion>(&body_text) {
            Ok(body) => Ok(parse_chat_completion(body)),
            Err(e) => {
                let truncated = if body_text.len() > 2000 {
                    format!("{}...<truncated>", &body_text[..2000])
                } else {
                    body_text
                };
                Err(AgentError::Other(format!(
                    "Failed to parse chat.completions response: {}. Expected OpenAI chat format with choices[0].message. Body: {}",
                    e, truncated
                )))
            }
        }
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
