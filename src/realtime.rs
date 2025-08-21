use async_trait::async_trait;
use crate::error::AgentError;

/// Trait for real‑time streaming capabilities (e.g., audio or token streams).
#[async_trait]
pub trait Realtime: Send + Sync {
    /// Start a streaming session and return a handle that yields streamed data.
    async fn start_stream(&self) -> Result<Box<dyn StreamItem>, AgentError>;
}

/// Trait representing a single item yielded by a real‑time stream.
#[async_trait]
pub trait StreamItem: Send + Sync {
    /// Retrieve the next chunk of data. Returns `None` when the stream ends.
    async fn next(&mut self) -> Result<Option<String>, AgentError>;
}
// Mock implementation of the Realtime streaming traits.

/// Simple stream item that yields a predefined list of string chunks.
pub struct SimpleStreamItem {
    messages: Vec<String>,
    index: usize,
}

impl SimpleStreamItem {
    pub fn new() -> Self {
        Self {
            messages: vec![
                "chunk 1".to_string(),
                "chunk 2".to_string(),
                "chunk 3".to_string(),
            ],
            index: 0,
        }
    }
}

#[async_trait]
impl StreamItem for SimpleStreamItem {
    async fn next(&mut self) -> Result<Option<String>, AgentError> {
        if self.index < self.messages.len() {
            let msg = self.messages[self.index].clone();
            self.index += 1;
            Ok(Some(msg))
        } else {
            Ok(None)
        }
    }
}

/// Mock realtime implementation that returns a `SimpleStreamItem`.
pub struct MockRealtime;

#[async_trait]
impl Realtime for MockRealtime {
    async fn start_stream(&self) -> Result<Box<dyn StreamItem>, AgentError> {
        Ok(Box::new(SimpleStreamItem::new()))
    }
}