use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration schema for the OpenAI Agents crate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// OpenAI API key.
    pub api_key: String,
    /// Model name to use (e.g., "gpt-4o-mini").
    pub model: String,
    /// Logging level (e.g., "info", "debug").
    pub log_level: String,
    /// Directory where plugins are stored.
    #[serde(default = "default_plugins_path")]
    pub plugins_path: PathBuf,
    /// Optional maximum number of concurrent requests.
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
}

fn default_plugins_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("openai_agents")
        .join("plugins")
}