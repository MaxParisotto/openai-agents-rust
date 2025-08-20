use thiserror::Error;

/// Central error type for the OpenAI Agents crate.
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Configuration error: {0}")]
    ConfigError(#[from] config::ConfigError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP request error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Serialization/Deserialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("Plugin loading error: {0}")]
    PluginError(String),

    #[error("Other error: {0}")]
    Other(String),
}