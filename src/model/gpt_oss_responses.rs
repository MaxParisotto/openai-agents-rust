use crate::config::Config;
use crate::error::AgentError;
use crate::model::{Model, ModelResponse, ToolCall};
use crate::utils::env::var_bool;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct GptOssResponses {
    client: Client,
    config: Config,
    base_url: String,
    auth_token: Option<String>,
}

impl GptOssResponses {
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
        Self {
            client,
            base_url: config.base_url.clone(),
            config,
            auth_token,
        }
    }

    fn url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum InputUnion {
    Str(String),
    Items(Vec<InputItem>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum InputItem {
    #[allow(dead_code)]
    #[serde(rename = "message")]
    Message { role: String, content: String },
    #[allow(dead_code)]
    #[serde(rename = "function_call")]
    FunctionCall {
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Serialize)]
struct FunctionToolDefinition {
    #[serde(rename = "type")]
    ty: String,
    name: String,
    parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize)]
struct ResponsesRequestBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: InputUnion,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<FunctionToolDefinition>>, // browser/code interpreter omitted for now
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>, // "auto" | "none"
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum OutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(rename = "role")]
        _role: String,
        content: Vec<TextPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        name: String,
        arguments: String,
        id: String,
        call_id: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        #[allow(dead_code)]
        call_id: String,
        #[allow(dead_code)]
        output: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct TextPart {
    #[allow(dead_code)]
    #[serde(rename = "type")]
    _ty: String,
    text: String,
}

#[derive(Deserialize)]
struct ResponsesObject {
    output: Vec<OutputItem>,
    #[allow(dead_code)]
    id: Option<String>,
}

fn map_openai_tools_to_oss(
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<FunctionToolDefinition>> {
    let mut out = Vec::new();
    if let Some(arr) = tools {
        for t in arr.iter() {
            if let Some(obj) = t.as_object() {
                if obj.get("type").and_then(|v| v.as_str()) == Some("function") {
                    if let Some(func) = obj.get("function").and_then(|v| v.as_object()) {
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let description = func
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let parameters = func
                            .get("parameters")
                            .cloned()
                            .unwrap_or(serde_json::json!({"type":"object"}));
                        out.push(FunctionToolDefinition {
                            ty: "function".into(),
                            name,
                            parameters,
                            description,
                            strict: Some(false),
                        });
                    }
                }
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn adapt_messages_to_input(messages: Option<&[serde_json::Value]>) -> InputUnion {
    if let Some(msgs) = messages {
        let mut items: Vec<InputItem> = Vec::new();
        for m in msgs.iter() {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
            match role {
                "user" | "assistant" | "system" => {
                    if let Some(content) = m.get("content").and_then(|v| v.as_str()) {
                        items.push(InputItem::Message {
                            role: role.into(),
                            content: content.into(),
                        });
                    }
                }
                "tool" => {
                    if let Some(call_id) = m.get("tool_call_id").and_then(|v| v.as_str()) {
                        let out = m
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        items.push(InputItem::FunctionCallOutput {
                            call_id: call_id.into(),
                            output: out,
                        });
                    }
                }
                _ => {}
            }
            // Do not inject function_call items into input for OSS Responses.
            // The model server expects function_call_output linked via previous_response_id.
        }
        if items.is_empty() {
            InputUnion::Str("".into())
        } else {
            InputUnion::Items(items)
        }
    } else {
        InputUnion::Str("".into())
    }
}

#[async_trait]
impl Model for GptOssResponses {
    async fn generate(&self, prompt: &str) -> Result<String, AgentError> {
        let mut req = self.client.post(self.url());
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let body = ResponsesRequestBody {
            instructions: None,
            input: InputUnion::Str(prompt.to_string()),
            model: Some(self.config.model.clone()),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            max_output_tokens: Some(512),
            temperature: Some(0.2),
            previous_response_id: None,
            store: None,
        };
        let resp = req.json(&body).send().await.map_err(AgentError::from)?;
        let status = resp.status();
        let text = resp.text().await.map_err(AgentError::from)?;
        if !status.is_success() {
            return Err(AgentError::Other(format!(
                "HTTP {} error: {}",
                status, text
            )));
        }
        Ok(text)
    }

    async fn get_response(
        &self,
        system_instructions: Option<&str>,
        _input: &str,
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
        let mut req = self.client.post(self.url());
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }

        let input = adapt_messages_to_input(messages);
        let tools_mapped = map_openai_tools_to_oss(tools);
        let tool_choice_str = tool_choice.and_then(|v| v.as_str().map(|s| s.to_string()));
        let disable_prev = var_bool("OSS_DISABLE_PREVIOUS_RESPONSE", false)
            || var_bool("OSS_TOOL_OUTPUT_AS_TEXT", false);
        let body = ResponsesRequestBody {
            instructions: system_instructions.map(|s| s.to_string()),
            input,
            model: Some(self.config.model.clone()),
            tools: tools_mapped,
            tool_choice: tool_choice_str,
            parallel_tool_calls: Some(true),
            max_output_tokens: Some(512),
            temperature: Some(0.2),
            previous_response_id: if disable_prev {
                None
            } else {
                _previous_response_id.map(|s| s.to_string())
            },
            store: if disable_prev { None } else { Some(true) },
        };
        if var_bool("OSS_DEBUG_PAYLOAD", false) {
            if let Ok(j) = serde_json::to_string_pretty(&body) {
                tracing::debug!(target = "gpt_oss_responses", payload = %j, "OSS Responses request body");
            }
        }
        if var_bool("OSS_DEBUG_HTTP", false) {
            if let Ok(j) = serde_json::to_string_pretty(&body) {
                eprintln!("OSS Responses REQUEST: {}", j);
            }
        }
        let resp = req.json(&body).send().await.map_err(AgentError::from)?;
        let status = resp.status();
        let body_text = resp.text().await.map_err(AgentError::from)?;
        if var_bool("OSS_DEBUG_PAYLOAD", false) {
            tracing::debug!(target = "gpt_oss_responses", http_status = %status, body = %body_text, "OSS Responses response");
        }
        if var_bool("OSS_DEBUG_HTTP", false) {
            eprintln!("OSS Responses HTTP {} body: {}", status, body_text);
        }
        if !status.is_success() {
            return Err(AgentError::Other(format!(
                "HTTP {} error: {}",
                status, body_text
            )));
        }
        let parsed: ResponsesObject = serde_json::from_str(&body_text).map_err(AgentError::from)?;
        let mut text: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let resp_id = parsed.id.clone();
        for item in parsed.output.into_iter() {
            match item {
                OutputItem::Message { _role: _, content } => {
                    let mut s = String::new();
                    for p in content {
                        s.push_str(&p.text);
                    }
                    if !s.is_empty() {
                        text = Some(s);
                    }
                }
                OutputItem::FunctionCall {
                    name,
                    arguments,
                    id,
                    call_id,
                } => {
                    tool_calls.push(ToolCall {
                        id: Some(id),
                        name,
                        arguments,
                        call_id: Some(call_id),
                    });
                }
                _ => {}
            }
        }
        Ok(ModelResponse {
            id: resp_id,
            text,
            tool_calls,
        })
    }
}
