use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::error::AgentError;

/// Trait representing a generic memory store.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Store a value under a given key.
    async fn set(&self, key: &str, value: String) -> Result<(), AgentError>;

    /// Retrieve a value for a given key.
    async fn get(&self, key: &str) -> Result<Option<String>, AgentError>;
}

/// Simple in‑memory implementation used for a single session.
#[derive(Clone, Default)]
pub struct SessionMemory {
    inner: Arc<RwLock<HashMap<String, String>>>,
}

#[async_trait]
impl Memory for SessionMemory {
    async fn set(&self, key: &str, value: String) -> Result<(), AgentError> {
        let mut map = self.inner.write().await;
        map.insert(key.to_string(), value);
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<String>, AgentError> {
        let map = self.inner.read().await;
        Ok(map.get(key).cloned())
    }
}