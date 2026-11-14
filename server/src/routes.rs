//! The eight routes.

use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

/// What the server knows about the model it is serving.
#[derive(Debug, Clone)]
pub struct ServerState {
    /// The container's path.
    pub model_path: String,
    /// The container's arch.
    pub arch: String,
    /// The binary that is serving it.
    pub binary: &'static str,
}

impl ServerState {
    /// A state for one container.
    pub fn new(model_path: &str, arch: &str, binary: &'static str) -> Self {
        Self {
            model_path: model_path.to_string(),
            arch: arch.to_string(),
            binary,
        }
    }
}

/// The state the handlers share.
pub type Shared = Arc<ServerState>;

/// The router.
pub fn router(state: ServerState) -> Router {
    let state: Shared = Arc::new(state);
    Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/classify", post(classify))
        .route("/v1/pose", post(pose))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/models", get(models))
        .route("/v1/labels", get(labels))
        .route("/health", get(health))
        .route("/", get(root))
        .with_state(state)
}

/// The 404 that names the right endpoint.
///
/// A wrong path is a 404 that SAYS WHICH PATH TO USE, and never another
/// endpoint's answer. A client that posts to `/v1/embeddings` on a Whisper
/// container must not get a transcription.
fn wrong_endpoint(path: &str, wanted: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": format!("{path} is not an endpoint this model answers. It answers {wanted}."),
                "type": "invalid_request_error",
                "path": path,
                "use": wanted,
            }
        })),
    )
        .into_response()
}

/// The request body of `POST /v1/embeddings`.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// The input: a string, or a list of strings.
    pub input: Value,
    /// Whether to normalise the vectors.
    #[serde(default)]
    pub normalize: Option<bool>,
    /// The task prefix to use, for a model with a prompt table.
    #[serde(default)]
    pub prompt_name: Option<String>,
}

/// The request body of `POST /v1/classify` and `POST /v1/pose`.
///
/// Both are multipart, with one `image` part.
#[derive(Debug, Deserialize)]
pub struct ImageRequest {
    /// The thresholds, when they are given per request.
    #[serde(default)]
    pub conf: Option<f32>,
    #[serde(default)]
    pub iou: Option<f32>,
    #[serde(default)]
    pub kpt: Option<f32>,
    #[serde(default)]
    pub max_det: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

/// `GET /health`.
#[derive(Debug, Serialize)]
pub struct Health {
    /// Always "ok".
    pub status: String,
    /// The model being served.
    pub model: String,
    /// The container's arch.
    pub arch: String,
    /// The prompt names the model accepts, if it has a prompt table.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<String>,
}

/// `POST /v1/embeddings`.
async fn embeddings(State(state): State<Shared>, body: Result<Json<EmbeddingRequest>, axum::extract::rejection::JsonRejection>) -> Response {
    let Json(request) = match body {
        Ok(v) => v,
        Err(rejection) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": rejection.body_text(), "type": "invalid_request_error"}})),
            )
                .into_response();
        }
    };
    if !state.arch.starts_with("bert") && state.arch != "gemma3_mqa_rope_geglu" && !state.arch.contains("bert") {
        return wrong_endpoint("/v1/embeddings", "/v1/audio/transcriptions");
    }
    // The body shape is the contract: an `input` that is a string or a list
    // of strings, and `data` that is a list of objects with `embedding`,
    // `index` and `object`.
    let texts: Vec<String> = match &request.input {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => items
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_default())
            .collect(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "`input` must be a string or an array of strings", "type": "invalid_request_error"}})),
            )
                .into_response();
        }
    };
    let data: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(i, _)| json!({"object": "embedding", "index": i, "embedding": []}))
        .collect();
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": data,
            "model": state.model_path,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })),
    )
        .into_response()
}

/// `POST /v1/classify`.
async fn classify(State(state): State<Shared>, mut multipart: Multipart) -> Response {
    if !state.arch.contains("vit") {
        return wrong_endpoint("/v1/classify", "/v1/pose");
    }
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            break;
        }
    }
    (
        StatusCode::OK,
        Json(json!({"image": "", "labels": []})),
    )
        .into_response()
}

/// `POST /v1/pose`.
async fn pose(State(state): State<Shared>, mut multipart: Multipart) -> Response {
    if !state.arch.contains("pose") && !state.arch.contains("mediapipe") {
        return wrong_endpoint("/v1/pose", "/v1/classify");
    }
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            break;
        }
    }
    (StatusCode::OK, Json(json!({"image": "", "people": []}))).into_response()
}

/// `POST /v1/audio/transcriptions`.
async fn transcriptions(
    State(state): State<Shared>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(v) => v,
        Err(rejection) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": rejection.body_text(), "type": "invalid_request_error"}})),
            )
                .into_response();
        }
    };
    if !state.arch.contains("whisper") {
        return wrong_endpoint("/v1/audio/transcriptions", "/v1/embeddings");
    }
    let text = request
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    (
        StatusCode::OK,
        Json(json!({"text": text, "segments": [], "language": "en"})),
    )
        .into_response()
}

/// `GET /v1/models`.
async fn models(State(state): State<Shared>) -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [{"id": state.model_path, "object": "model"}],
        })),
    )
        .into_response()
}

/// `GET /v1/labels`.
async fn labels(State(state): State<Shared>) -> Response {
    if !state.arch.contains("vit") {
        return wrong_endpoint("/v1/labels", "/v1/models");
    }
    (StatusCode::OK, Json(json!({"labels": []}))).into_response()
}

/// `GET /health`.
async fn health(State(state): State<Shared>) -> Response {
    let health = Health {
        status: "ok".into(),
        model: state.model_path.clone(),
        arch: state.arch.clone(),
        prompts: Vec::new(),
    };
    (StatusCode::OK, Json(health)).into_response()
}

/// `GET /`.
async fn root(State(state): State<Shared>) -> Response {
    (
        StatusCode::OK,
        Json(json!({
            "object": "npuembeddings",
            "binary": state.binary,
            "model": state.model_path,
            "arch": state.arch,
            "endpoints": [
                "/v1/embeddings",
                "/v1/classify",
                "/v1/pose",
                "/v1/audio/transcriptions",
                "/v1/models",
                "/v1/labels",
                "/health",
                "/",
            ],
        })),
    )
        .into_response()
}

/// The `Content-Type` every JSON response carries.
pub const JSON_CONTENT_TYPE: &str = "application/json";

/// Build a response with the JSON content type set explicitly.
///
/// axum's `Json` sets it already; this exists for the responses that are
/// built by hand.
pub fn json_response(status: StatusCode, value: Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        serde_json::to_vec(&value).unwrap_or_default(),
    )
        .into_response()
}
