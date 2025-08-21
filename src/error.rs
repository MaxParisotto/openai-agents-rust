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
    YamlError(#[from] rust_yaml::Error),

    #[error("Plugin loading error: {0}")]
    PluginError(String),

    #[error("Other error: {0}")]
    Other(String),
}

// Allow `AgentError` to be used directly as an Axum response.
impl axum::response::IntoResponse for AgentError {
    fn into_response(self) -> axum::response::Response {
        let body = axum::Json(serde_json::json!({ "error": self.to_string() }));
        (axum::http::StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}
