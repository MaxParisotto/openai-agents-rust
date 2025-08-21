use crate::error::AgentError;
use crate::utils::env::var_nonempty;
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
    let mut cfg = settings
        .try_deserialize::<Config>()
        .map_err(|e| AgentError::Other(format!("Config error: {}", e)))?;
    apply_env_overrides(&mut cfg);
    Ok(cfg)
}

/// Load configuration purely from environment variables (after .env is loaded by the caller).
/// Priority:
/// - OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, RUST_LOG
///
/// Note: No provider defaults are applied here. Missing variables remain empty
/// and should be handled by callers or higher-level config files.
pub fn load_from_env() -> Config {
    let mut cfg = Config {
        api_key: String::new(),
        model: var_nonempty("OPENAI_MODEL").unwrap_or_default(),
        base_url: var_nonempty("OPENAI_BASE_URL").unwrap_or_default(),
        log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        plugins_path: schema::default_plugins_path(),
        max_concurrent_requests: None,
    };
    if let Some(k) = var_nonempty("OPENAI_API_KEY") {
        cfg.api_key = k;
    }
    apply_env_overrides(&mut cfg);
    cfg
}

/// Apply generic provider environment overrides on top of a loaded Config.
fn apply_env_overrides(cfg: &mut Config) {
    if let Some(k) = var_nonempty("OPENAI_API_KEY") {
        cfg.api_key = k;
    }
    if let Some(u) = var_nonempty("OPENAI_BASE_URL") {
        cfg.base_url = u;
    }
    if let Some(m) = var_nonempty("OPENAI_MODEL") {
        cfg.model = m;
    }
    if let Ok(lvl) = std::env::var("RUST_LOG") {
        if !lvl.trim().is_empty() {
            cfg.log_level = lvl;
        }
    }
}
