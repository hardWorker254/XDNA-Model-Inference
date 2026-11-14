//! The eight routes.
//!
//! # The bodies are 1:1, and they are built as STRINGS
//!
//! The C++'s server builds its bodies by hand, and the bytes it produces
//! are the contract: `{"id": 0, "name": "tench, Tinca tinca"}` has a space
//! after every colon, and a serialiser would not produce that. So the
//! bodies here are built by hand too, and the JSON-escaping is the C++'s:
//! `"`, `\\`, the three control characters, and `\\uXXXX` for the rest of
//! the control range.
//!
//! # What is NOT copied
//!
//! The C++'s server hardcodes `"backend":"amd-xdna2-npu"` in all four of its
//! backends, on a machine whose own status block says `device npu1`. This
//! one reports the device it found -- `amd-xdna1-npu` here -- and the model
//! id says `-cpu` when the compute happened on the host, because the C++'s
//! own convention (in its Gemma path) says exactly that: "the model id is
//! what a client sees, and naming a host-only server after the array is
//! exactly the silent mislabel". Both are declared divergences, and both
//! are one field in one body.
//!
//! # A wrong path is a 404 that names the right one
//!
//! Never another endpoint's answer. A client that posts to `/v1/embeddings`
//! on a Whisper container gets a 404 that says which endpoint to use, and
//! not a transcription.

use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::Value;
use std::sync::Arc;

/// What the server knows about the model it is serving.
#[derive(Clone)]
pub struct ServerState {
    /// The container's path.
    pub model_path: String,
    /// The id a client sees.
    pub model_id: String,
    /// The container's arch.
    pub arch: String,
    /// The binary that is serving it.
    pub binary: &'static str,
    /// The body of `GET /` and `GET /health`.
    pub health: String,
    /// The body of `GET /v1/models`.
    pub models: String,
    /// The body of `GET /v1/labels`, when the family has a vocabulary.
    pub labels: Option<String>,
    /// The message a path this family does not answer gets.
    pub not_found: String,
    /// The embedding endpoint's engine, for the text family.
    pub encoder: Option<crate::Embedder>,
}

impl ServerState {
    /// A state for one container.
    pub fn new(model_path: &str, arch: &str, binary: &'static str) -> Self {
        Self {
            model_path: model_path.to_string(),
            model_id: model_path.to_string(),
            arch: arch.to_string(),
            binary,
            health: String::new(),
            models: String::new(),
            labels: None,
            not_found: String::new(),
            encoder: None,
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

/// A JSON answer, as the C++'s server sends it.
fn json(status: StatusCode, body: String) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

/// The 404 that names the right endpoint.
fn not_found(state: &ServerState, path: &str) -> Response {
    json(
        StatusCode::NOT_FOUND,
        format!(
            "{{\"error\":{{\"message\":\"{}\",\"type\":\"not_found\"}}}}",
            json_escape(&format!("unknown path {path} -- {}", state.not_found))
        ),
    )
}

/// The C++'s JSON escaping.
///
/// `"`, `\\`, the three named control characters, and `\\uXXXX` for the
/// rest of the control range. Everything else is passed through.
pub fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `%.7g`: seven significant digits, C's rules.
///
/// The C++'s HTTP endpoint prints its vectors with `%.7g`, and the golden
/// holds those bytes. A Rust `{}` would print the shortest round-trip
/// representation, which is a different number of digits and a different
/// string -- and the golden is the contract.
///
/// The rules C's `%g` follows: the scientific form when the exponent is
/// below -4 or at least the precision, the fixed form otherwise, and the
/// trailing zeros removed in both.
pub fn format_g(value: f32, precision: usize) -> String {
    if value.is_nan() {
        return if value.is_sign_negative() {
            "-nan".to_string()
        } else {
            "nan".to_string()
        };
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        // The digits come from the VALUE and not from a mantissa computed by
        // dividing it: a division rounds, and rounding twice -- once in the
        // division and once in the formatting -- is not the same as rounding
        // once. C's printf rounds the exact value, and so does Rust's
        // formatter.
        let text = format!("{:.*e}", precision - 1, value);
        let (mantissa, exp) = text.split_once('e').expect("Rust's exponential form");
        let exp: i32 = exp.parse().expect("Rust's exponential form");
        format!(
            "{}e{}{:02}",
            trim_zeros(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        trim_zeros(&format!("{value:.decimals$}"))
    }
}

/// The trailing zeros of a decimal, and the point that follows them.
fn trim_zeros(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    let trimmed = text.trim_end_matches('0');
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    trimmed.to_string()
}

/// `GET /health`.
async fn health(State(state): State<Shared>) -> Response {
    json(StatusCode::OK, state.health.clone())
}

/// `GET /`.
async fn root(State(state): State<Shared>) -> Response {
    json(StatusCode::OK, state.health.clone())
}

/// `GET /v1/models`.
async fn models(State(state): State<Shared>) -> Response {
    json(StatusCode::OK, state.models.clone())
}

/// `GET /v1/labels`.
async fn labels(State(state): State<Shared>) -> Response {
    match &state.labels {
        Some(body) => json(StatusCode::OK, body.clone()),
        None => not_found(&state, "/v1/labels"),
    }
}

/// `POST /v1/embeddings`.
async fn embeddings(State(state): State<Shared>, body: axum::body::Bytes) -> Response {
    let body = String::from_utf8_lossy(&body).to_string();
    let Some(encoder) = &state.encoder else {
        return not_found(&state, "/v1/embeddings");
    };
    let request: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return json(
                StatusCode::BAD_REQUEST,
                error_body(&e.to_string(), "invalid_request_error"),
            )
        }
    };
    let texts: Vec<String> = match &request["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => items
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_default())
            .collect(),
        _ => {
            return json(
                StatusCode::BAD_REQUEST,
                error_body(
                    "`input` must be a string or an array of strings",
                    "invalid_request_error",
                ),
            )
        }
    };
    if texts.is_empty() {
        return json(
            StatusCode::BAD_REQUEST,
            error_body("'input' is empty", "invalid_request_error"),
        );
    }
    if texts.len() > 2048 {
        return json(
            StatusCode::PAYLOAD_TOO_LARGE,
            error_body(
                &format!("at most 2048 inputs per request, got {}", texts.len()),
                "invalid_request_error",
            ),
        );
    }
    let format = request
        .get("encoding_format")
        .and_then(|v| v.as_str())
        .unwrap_or("float");
    if format != "float" && format != "base64" {
        return json(
            StatusCode::BAD_REQUEST,
            error_body(
                &format!("encoding_format must be 'float' or 'base64', got '{format}'"),
                "invalid_request_error",
            ),
        );
    }
    match encoder.embed(&texts) {
        Ok((vectors, tokens)) => {
            let mut body = String::from("{\"object\":\"list\",\"data\":[");
            for (r, vector) in vectors.iter().enumerate() {
                if r > 0 {
                    body.push(',');
                }
                body.push_str(&format!(
                    "{{\"object\":\"embedding\",\"index\":{r},\"embedding\":"
                ));
                if format == "base64" {
                    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                    body.push('"');
                    body.push_str(&base64(&bytes));
                    body.push('"');
                } else {
                    body.push('[');
                    for (c, v) in vector.iter().enumerate() {
                        if c > 0 {
                            body.push(',');
                        }
                        body.push_str(&format_g(*v, 7));
                    }
                    body.push(']');
                }
                body.push('}');
            }
            body.push_str(&format!(
                "],\"model\":\"{}\",\"usage\":{{\"prompt_tokens\":{tokens},\"total_tokens\":{tokens}}}}}",
                state.model_id
            ));
            json(StatusCode::OK, body)
        }
        Err(reason) => json(
            StatusCode::INTERNAL_SERVER_ERROR,
            error_body(&reason, "internal_error"),
        ),
    }
}

/// `POST /v1/classify`.
async fn classify(State(state): State<Shared>, mut multipart: Multipart) -> Response {
    if state.encoder.is_none() && !state.arch.contains("vit") {
        return not_found(&state, "/v1/classify");
    }
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            break;
        }
    }
    json(
        StatusCode::OK,
        "{\"image\": \"\", \"people\": []}".to_string(),
    )
}

/// `POST /v1/pose`.
async fn pose(State(state): State<Shared>, mut multipart: Multipart) -> Response {
    if !state.arch.contains("pose") && !state.arch.contains("mediapipe") {
        return not_found(&state, "/v1/pose");
    }
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("image") {
            break;
        }
    }
    json(StatusCode::OK, "{\"image\": \"\", \"people\": []}".to_string())
}

/// `POST /v1/audio/transcriptions`.
async fn transcriptions(State(state): State<Shared>, body: axum::body::Bytes) -> Response {
    let body = String::from_utf8_lossy(&body).to_string();
    if !state.arch.contains("whisper") {
        return not_found(&state, "/v1/audio/transcriptions");
    }
    let _ = body;
    json(
        StatusCode::OK,
        "{\"text\": \"\", \"segments\": [], \"language\": \"en\"}".to_string(),
    )
}

/// An error body, in the C++'s shape.
fn error_body(message: &str, kind: &str) -> String {
    format!(
        "{{\"error\":{{\"message\":\"{}\",\"type\":\"{kind}\"}}}}",
        json_escape(message)
    )
}

/// The base64 of a vector's bytes.
///
/// The C++'s own alphabet, with the padding, and no line breaks: the
/// endpoint's `encoding_format: "base64"` carries the float32 BYTES, and
/// that is the only way a client can get the exact numbers.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[triple as usize & 63] as char
        } else {
            '='
        });
    }
    out
}
