use crate::error::AgentError;
use config::{Config as ConfigLoader, Environment, File};
use std::path::Path;

mod schema;
pub use schema::Config;

/// Load configuration from a YAML or JSON file and merge with environment variables.
///
/// The function reads the file at `path`, then overlays any environment variables
/// prefixed with `OPENAI_AGENTS_`. Environment variable names are converted to
/// lower‑case and underscores are replaced with dots to match the struct fields.
///
/// # Errors
///
/// Returns `AgentError` if the file cannot be read, parsed, or if required fields
/// are missing.
pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Config, AgentError> {
    // Initialize a new config loader
    let mut settings = ConfigLoader::default();

    // Load from the specified file (YAML or JSON). The `File` source automatically
    // detects the format based on the file extension.
    settings.merge(File::with_name(
        path.as_ref()
            .to_str()
            .ok_or_else(|| AgentError::Other("Invalid config file path".to_string()))?,
    ))?;

    // Merge in environment variables with the prefix `OPENAI_AGENTS_`.
    // This allows overrides like `OPENAI_AGENTS_API_KEY=...`.
    settings.merge(Environment::with_prefix("OPENAI_AGENTS").separator("_"))?;

    // Deserialize into our strongly‑typed `Config` struct.
    settings.try_into().map_err(|e| AgentError::Other(format!("Config error: {}", e)))
}