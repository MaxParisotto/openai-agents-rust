use async_trait::async_trait;
use reqwest::Client;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use crate::config::Config;
use crate::error::AgentError;

/// Trait representing a generic voice pipeline.
#[async_trait]
pub trait VoicePipeline: Send + Sync {
    /// Process an input audio buffer and return a textual transcription.
    async fn transcribe(&self, audio: &[u8]) -> Result<String, AgentError>;

    /// Convert text to synthesized audio bytes.
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>, AgentError>;
}

/// Speech‑to‑text trait.
#[async_trait]
pub trait Stt: Send + Sync {
    async fn stt(&self, audio: &[u8]) -> Result<String, AgentError>;
}

/// Text‑to‑speech trait.
#[async_trait]
pub trait Tts: Send + Sync {
    async fn tts(&self, text: &str) -> Result<Vec<u8>, AgentError>;
}

/// OpenAI‑compatible STT implementation (POST /v1/audio/transcriptions).
pub struct OpenAiStt {
    client: Client,
    base_url: String,
    model: String,
    auth_token: Option<String>,
}

impl OpenAiStt {
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
            model: config.model.clone(),
            auth_token,
        }
    }

    fn url(&self) -> String {
        format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        )
    }
}

#[derive(Deserialize)]
struct SttResponse {
    text: String,
}

#[async_trait]
impl Stt for OpenAiStt {
    async fn stt(&self, audio: &[u8]) -> Result<String, AgentError> {
        let part = Part::bytes(audio.to_vec())
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| AgentError::Other(format!("invalid audio mime: {}", e)))?;
        let form = Form::new()
            .text("model", self.model.clone())
            .part("file", part);
        let mut req = self.client.post(self.url());
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req.multipart(form).send().await.map_err(AgentError::from)?;
        let status = resp.status();
        let body = resp.text().await.map_err(AgentError::from)?;
        if !status.is_success() {
            return Err(AgentError::Other(format!(
                "stt failed (status: {}): {}",
                status, body
            )));
        }
        let parsed: SttResponse = serde_json::from_str(&body)
            .map_err(|e| AgentError::Other(format!("stt parse error: {} body={}", e, body)))?;
        Ok(parsed.text)
    }
}

/// OpenAI‑compatible TTS implementation (POST /v1/audio/speech).
pub struct OpenAiTts {
    client: Client,
    base_url: String,
    model: String,
    voice: Option<String>,
    format: Option<String>,
    auth_token: Option<String>,
}

impl OpenAiTts {
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
            model: config.model.clone(),
            voice: Some("alloy".into()),
            format: Some("wav".into()),
            auth_token,
        }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
    pub fn with_format(mut self, fmt: impl Into<String>) -> Self {
        self.format = Some(fmt.into());
        self
    }
    fn url(&self) -> String {
        format!("{}/audio/speech", self.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl Tts for OpenAiTts {
    async fn tts(&self, text: &str) -> Result<Vec<u8>, AgentError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        if let Some(v) = &self.voice {
            body["voice"] = serde_json::json!(v);
        }
        if let Some(f) = &self.format {
            body["format"] = serde_json::json!(f);
        }
        let mut req = self.client.post(self.url());
        if let Some(token) = &self.auth_token {
            req = req.bearer_auth(token);
        }
        let resp = req.json(&body).send().await.map_err(AgentError::from)?;
        let status = resp.status();
        let bytes = resp.bytes().await.map_err(AgentError::from)?;
        if !status.is_success() {
            let body = String::from_utf8_lossy(&bytes).to_string();
            return Err(AgentError::Other(format!(
                "tts failed (status: {}): {}",
                status, body
            )));
        }
        Ok(bytes.to_vec())
    }
}

/// Composed pipeline using STT and TTS.
pub struct HttpVoicePipeline {
    stt: Box<dyn Stt>,
    tts: Box<dyn Tts>,
}

impl HttpVoicePipeline {
    pub fn new(stt: Box<dyn Stt>, tts: Box<dyn Tts>) -> Self {
        Self { stt, tts }
    }
}

#[async_trait]
impl VoicePipeline for HttpVoicePipeline {
    async fn transcribe(&self, audio: &[u8]) -> Result<String, AgentError> {
        self.stt.stt(audio).await
    }
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>, AgentError> {
        self.tts.tts(text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::{Router, routing::post};

    #[tokio::test]
    async fn stt_tts_roundtrip_against_mock_server() {
        // Simple mock endpoints that accept any payloads
        let app = Router::new()
            .route(
                "/audio/transcriptions",
                post(|| async move {
                    let body = serde_json::json!({"text":"hello world"});
                    (StatusCode::OK, axum::Json(body))
                }),
            )
            .route(
                "/audio/speech",
                post(|axum::Json(_): axum::Json<serde_json::Value>| async move {
                    let audio: Vec<u8> = vec![1, 2, 3, 4, 5];
                    (StatusCode::OK, audio).into_response()
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });

    let _ = dotenvy::dotenv();
    let mut cfg = crate::config::load_from_env();
    cfg.api_key = String::new();
    cfg.model = if cfg.model.is_empty() { "whisper-1".into() } else { cfg.model };
    cfg.base_url = format!("http://{}:{}", addr.ip(), addr.port());
        let stt = OpenAiStt::new(cfg.clone());
        let tts = OpenAiTts::new(cfg.clone());
        let pipe = HttpVoicePipeline::new(Box::new(stt), Box::new(tts));

        let transcript = pipe.transcribe(b"ignored").await.unwrap();
        assert_eq!(transcript, "hello world");
        let audio = pipe.synthesize("Hi").await.unwrap();
        assert_eq!(audio, vec![1, 2, 3, 4, 5]);
    }
}
