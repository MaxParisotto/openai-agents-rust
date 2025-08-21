use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;

use crate::config::Config;
use crate::error::AgentError;

/// Trait for real‑time streaming capabilities (e.g., token streams over SSE).
#[async_trait]
pub trait Realtime: Send + Sync {
    /// Start a streaming session and return a handle that yields streamed text deltas.
    async fn start_stream(&self) -> Result<Box<dyn StreamItem>, AgentError>;
}

/// Trait representing a single item yielded by a real‑time stream.
#[async_trait]
pub trait StreamItem: Send + Sync {
    /// Retrieve the next chunk of data. Returns `None` when the stream ends.
    async fn next(&mut self) -> Result<Option<String>, AgentError>;
}

/// OpenAI‑compatible Chat Completions streaming client (SSE, stream=true).
/// Construct with the prompt/messages to stream and then call `start_stream`.
pub struct OpenAiChatRealtime {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
    model: String,
    messages: Vec<Value>,
    // optional parameters
    max_tokens: Option<i32>,
    temperature: Option<f32>,
}

impl OpenAiChatRealtime {
    pub fn new_with_messages(config: Config, messages: Vec<Value>) -> Self {
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
            auth_token,
            model: config.model.clone(),
            messages,
            max_tokens: Some(512),
            temperature: Some(0.2),
        }
    }

    pub fn new_simple(config: Config, prompt: &str) -> Self {
        let messages = vec![serde_json::json!({"role":"user","content":prompt})];
        Self::new_with_messages(config, messages)
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl Realtime for OpenAiChatRealtime {
    async fn start_stream(&self) -> Result<Box<dyn StreamItem>, AgentError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": self.messages,
            "stream": true,
        });
        if let Some(mt) = self.max_tokens {
            body["max_tokens"] = serde_json::json!(mt);
        }
        if let Some(t) = self.temperature {
            body["temperature"] = serde_json::json!(t);
        }

        let mut req = self.client.post(self.url());
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req.json(&body).send().await.map_err(AgentError::from)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(AgentError::Other(format!(
                "realtime stream failed: HTTP {} — {}",
                status, text
            )));
        }

        let item = SseStreamItem::new(resp);
        Ok(Box::new(item))
    }
}

/// StreamItem backed by parsing SSE "data:" lines from an HTTP response.
struct SseStreamItem {
    stream: tokio::sync::Mutex<
        Pin<Box<dyn futures_core::Stream<Item = Result<String, AgentError>> + Send>>,
    >,
}

impl SseStreamItem {
    fn new(resp: reqwest::Response) -> Self {
        let byte_stream = resp.bytes_stream();
        let s = async_stream::try_stream! {
                let mut buf: Vec<u8> = Vec::new();
                futures_util::pin_mut!(byte_stream);
                while let Some(chunk) = byte_stream.next().await {
                    let chunk: Bytes = chunk.map_err(AgentError::from)?;
                    buf.extend_from_slice(&chunk);
                    // process complete lines
                    loop {
                        if let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                            let line = buf.drain(..=pos).collect::<Vec<u8>>();
                            let line = String::from_utf8_lossy(&line).to_string();
                            let line = line.trim();
                            if line.is_empty() { continue; }
                            if let Some(rest) = line.strip_prefix("data: ") {
                                let data = rest.trim();
                                if data == "[DONE]" { break; }
                                // Try parse JSON, extract text deltas
                                if let Ok(v) = serde_json::from_str::<Value>(data) {
                                    // OpenAI: choices[0].delta.content or choices[0].text
                                    let maybe = v
                                        .get("choices").and_then(|c| c.as_array()).and_then(|arr| arr.get(0))
                                        .and_then(|c0| c0.get("delta").and_then(|d| d.get("content")).and_then(|t| t.as_str()).map(|s| s.to_string())
                                            .or_else(|| c0.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())));
                                    if let Some(text) = maybe { if !text.is_empty() { yield text; } }
                                }
                            }
                        } else { break; }
                    }
                }
        };
        Self {
            stream: tokio::sync::Mutex::new(Box::pin(s)),
        }
    }
}

#[async_trait]
impl StreamItem for SseStreamItem {
    async fn next(&mut self) -> Result<Option<String>, AgentError> {
        let mut guard = self.stream.lock().await;
        match guard.next().await {
            Some(Ok(s)) => Ok(Some(s)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
}
