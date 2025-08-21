use axum::{
    Router,
    extract::{Path, State},
    response::{
        IntoResponse, Json,
        sse::{self, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tokio::sync::broadcast::{Receiver, Sender};

use crate::{
    client::OpenAiClient,
    config::Config,
    error::AgentError,
    model::{self, Model},
    plugin::loader::PluginRegistry,
};

/// Shared application state for the MCP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub client: Arc<OpenAiClient>,
    pub plugins: Arc<PluginRegistry>,
    pub broadcaster: Sender<String>,
}

/// Request payload for the `/run` endpoint.
#[derive(Debug, Deserialize)]
pub struct RunRequest {
    pub model: String,
    pub prompt: String,
}

/// Simple response wrapper.
#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub result: String,
}

/// Handler for `/run` – forwards the prompt to the selected model.
#[axum::debug_handler]
async fn run_handler(
    State(state): State<AppState>,
    Json(payload): Json<RunRequest>,
) -> Result<Json<RunResponse>, AgentError> {
    // Instantiate the requested model.
    let model: Box<dyn Model> = match payload.model.as_str() {
        "openai_chat" => Box::new(model::openai_chat::OpenAiChat::new((*state.config).clone())),
        "openai_realtime" => Box::new(model::openai_realtime::OpenAiRealtime::new(
            (*state.config).clone(),
        )),
        "litellm" => Box::new(model::litellm::LiteLLM::new((*state.config).clone())),
        _ => {
            return Err(AgentError::Other(format!(
                "Unknown model {}",
                payload.model
            )));
        }
    };

    let result = model.generate(&payload.prompt).await?;
    // Broadcast the result to any SSE listeners.
    let _ = state.broadcaster.send(result.clone());

    Ok(Json(RunResponse { result }))
}

/// Handler for `/status/:session_id` – placeholder returning a static status.
async fn status_handler(Path(_session_id): Path<String>) -> impl IntoResponse {
    Json(serde_json::json!({ "status": "running" }))
}

/// Handler for `/events/:session_id` – Server‑Sent Events stream.
async fn events_handler(
    Path(_session_id): Path<String>,
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<sse::Event, std::convert::Infallible>>> {
    let mut rx: Receiver<String> = state.broadcaster.subscribe();
    let stream = async_stream::stream! {
        while let Ok(msg) = rx.recv().await {
            yield Ok(sse::Event::default().data(msg));
        }
    };
    Sse::new(stream)
}

/// Build the Axum router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/run", post(run_handler))
        .route("/status/:session_id", get(status_handler))
        .route("/events/:session_id", get(events_handler))
        .with_state(state)
}

/// Start the MCP server – called from `main.rs`.
pub async fn start_server(state: AppState, addr: SocketAddr) -> Result<(), AgentError> {
    let app = router(state);
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| AgentError::Other(e.to_string()))?;
    axum::serve(listener, app.into_make_service())
        .await
        .map_err(|e| AgentError::Other(e.to_string()))
}
