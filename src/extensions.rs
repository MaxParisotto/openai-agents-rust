use crate::error::AgentError;

/// Trait for optional extensions that can be loaded at runtime.
pub trait Extension: Send + Sync {
    /// Human‑readable name of the extension.
    fn name(&self) -> &str;

    /// Initialise the extension with access to configuration and client.
    fn init(&self, config: &crate::config::Config) -> Result<(), AgentError>;
}