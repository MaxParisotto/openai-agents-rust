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
    // Build a config loader using the builder API (the `merge` method was removed in recent versions).
    let builder = ConfigLoader::builder()
        .add_source(
            File::with_name(
                path.as_ref()
                    .to_str()
                    .ok_or_else(|| AgentError::Other("Invalid config file path".to_string()))?,
            )
            .required(false),
        )
        // Use "__" as separator so single underscores remain intact (e.g., BASE_URL -> base_url)
        .add_source(Environment::with_prefix("OPENAI_AGENTS").separator("__"));

    // Build the configuration and deserialize into our strongly‑typed `Config` struct.
    let settings = builder
        .build()
        .map_err(|e| AgentError::Other(format!("Config build error: {}", e)))?;
    settings
        .try_deserialize::<Config>()
        .map_err(|e| AgentError::Other(format!("Config error: {}", e)))
}
