use async_trait::async_trait;
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

/// Example implementation that simply echoes the input.
pub struct DummyVoicePipeline;

#[async_trait]
impl VoicePipeline for DummyVoicePipeline {
    async fn transcribe(&self, audio: &[u8]) -> Result<String, AgentError> {
        // In a real implementation you would call a STT service.
        Ok(String::from_utf8_lossy(audio).to_string())
    }

    async fn synthesize(&self, text: &str) -> Result<Vec<u8>, AgentError> {
        // In a real implementation you would call a TTS service.
        Ok(text.as_bytes().to_vec())
    }
}