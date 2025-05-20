// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::DeploymentState;
use super::{
    error::HttpError,
    metrics::{Endpoint, InflightGuard},
    RouteDoc,
};
use axum::{
    extract::State,
    http::HeaderMap,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::{
    collections::{HashMap, HashSet},
    pin::Pin,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::wrappers::ReceiverStream;

use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionResponse, completions::CompletionResponse, nvext::NvExt,
};
use crate::types::{
    openai::{chat_completions::NvCreateChatCompletionRequest, completions::CompletionRequest},
    Annotated,
};

use dynamo_runtime::pipeline::{AsyncEngineContext, Context};

#[derive(Serialize, Deserialize)]
pub(crate) struct ErrorResponse {
    error: String,
}

impl ErrorResponse {
    /// Not Found Error
    pub fn model_not_found() -> (StatusCode, Json<ErrorResponse>) {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Model not found".to_string(),
            }),
        )
    }

    /// Service Unavailable
    /// This is returned when the service is live, but not ready.
    pub fn _service_unavailable() -> (StatusCode, Json<ErrorResponse>) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Service is not ready".to_string(),
            }),
        )
    }

    /// Internal Service Error
    /// Return this error when the service encounters an internal error.
    /// We should return a generic message to the client instead of the real error.
    /// Internal Services errors are the result of misconfiguration or bugs in the service.
    pub fn internal_server_error(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
        tracing::error!("Internal server error: {msg}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: msg.to_string(),
            }),
        )
    }

    /// Too Many Requests Error
    /// This is returned when the service is busy due to too many inflight requests for a model.
    pub fn too_many_requests(model_name: &str) -> (StatusCode, Json<ErrorResponse>) {
        (
            StatusCode::TOO_MANY_REQUESTS, // 429 Too Many Requests
            Json(ErrorResponse {
                error: format!(
                    "Too slow inflight requests for model: {}. Please try again later.",
                    model_name
                ),
            }),
        )
    }

    /// The OAI endpoints call an [`dynamo.runtime::engine::AsyncEngine`] which are specialized to return
    /// an [`anyhow::Error`]. This method will convert the [`anyhow::Error`] into an [`HttpError`].
    /// If successful, it will return the [`HttpError`] as an [`ErrorResponse::internal_server_error`]
    /// with the details of the error.
    pub fn from_anyhow(err: anyhow::Error, alt_msg: &str) -> (StatusCode, Json<ErrorResponse>) {
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorResponse::from_http_error(http_error),
            Err(err) => ErrorResponse::internal_server_error(&format!("{alt_msg}: {err}")),
        }
    }

    /// Implementers should only be able to throw 400-499 errors.
    pub fn from_http_error(err: HttpError) -> (StatusCode, Json<ErrorResponse>) {
        if err.code < 400 || err.code >= 500 {
            return ErrorResponse::internal_server_error(&err.message);
        }
        match StatusCode::from_u16(err.code) {
            Ok(code) => (code, Json(ErrorResponse { error: err.message })),
            Err(_) => ErrorResponse::internal_server_error(&err.message),
        }
    }
}

impl From<HttpError> for ErrorResponse {
    fn from(err: HttpError) -> Self {
        ErrorResponse { error: err.message }
    }
}

// A RAII guard to ensure that the context is stopped when the request is dropped.
// Request fututures are dropped in axum when the client disconnects.
// https://github.com/tokio-rs/axum/discussions/1094
// may be defused to prevent stopping the context and send a control message via
// stop_generating
struct CtxDropGuard {
    ctx: Arc<dyn AsyncEngineContext>,
    verbose: bool,
}

impl CtxDropGuard {
    fn new(ctx: Arc<dyn AsyncEngineContext>) -> Self {
        CtxDropGuard {
            ctx,
            verbose: true,
        }
    }

    fn not_print_drop(&mut self) {
        self.verbose = false;
    }
}

impl Drop for CtxDropGuard {
    fn drop(&mut self) {
        self.ctx.stop_generating();
        if self.verbose {
            tracing::info!("Detected user side cancellation for request_id: {}", self.ctx.id());
        }
    }
}

fn extract_request_id(headers: &HeaderMap) -> (String, i32) {
    let billing_id = headers
        .get("X-Baseten-Billing-Org-Id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("org-internalnobillingidprovided");

    let request_suffix = headers
        .get("X-Baseten-Request-Id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_owned()) // Use map to convert &str -> String
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let billing_model_version = headers
        .get("X-Baseten-Model-Version-ID")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("modelversionidempty");

    let priority: i32 = headers
        .get("X-Baseten-Priority")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(100);

    let request_id = format!(
        "{}--{}--{}",
        billing_id, request_suffix, billing_model_version
    );

    (request_id, priority)
}

/// OpenAI Completions Request Handler
///
/// This method will handle the incoming request for the `/v1/completions endpoint`. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
#[tracing::instrument(skip_all)]
async fn completions(
    State(state): State<Arc<DeploymentState>>,
    headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // todo - extract distributed tracing id and context id from headers
    let (request_id, priority) = extract_request_id(&headers);

    tracing::info!("request_id: {request_id}");
    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // update the request to always stream
    let inner = async_openai::types::CreateCompletionRequest {
        stream: Some(true),
        ..request.inner
    };

    let mut request_payload = CompletionRequest {
        inner,
        nvext: request.nvext,
    };
    request_payload
        .nvext
        .get_or_insert_with(NvExt::default)
        .priority = Some(priority as i64);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request_payload.inner.model;

    // todo - error handling should be more robust
    let engine = state
        .get_completions_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight = state.create_inflight_guard(model, Endpoint::Completions, streaming);

    // Check for backpressure before proceeding
    if state.is_rate_limited(model, priority) {
        inflight.mark_429();
        return Ok(ErrorResponse::too_many_requests(model).into_response());
    }

    // setup context
    // todo - inherit request_id from distributed trace details
    let request_ctx = Context::with_id(request_payload, request_id.clone());

    // issue the generate call on the engine
    let stream = engine
        .generate(request_ctx)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic
    let mut guard = CtxDropGuard::new(ctx.clone());
    if streaming {
        let stream = stream.map(move |response| {
            guard.not_print_drop(); // ensures that the guard is moved into the stream
            Event::try_from(EventConverter::from(response))
        });
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight).await?;

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }
        Ok(sse_stream.into_response())
    } else {
        let response = CompletionResponse::from_annotated_stream(stream.into())
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id,
                    e
                );
                // TODO(Michael): check if HTTPError is present in the error chain and return it
                // todo: check if HTTPError is prsent in the error chain and return it
                if e.to_string().contains("HTTPExceptionDetected") {
                    ErrorResponse::from_http_error(HttpError {
                        code: 400,
                        message: e.to_string(),
                    })
                } else {
                    ErrorResponse::internal_server_error(&format!(
                        "Failed to fold chat completions stream: {}",
                        e
                    ))
                }
            })?;
        guard.not_print_drop();
        inflight.mark_ok();
        Ok(Json(response).into_response())
    }
}

/// OpenAI Chat Completions Request Handler
///
/// This method will handle the incoming request for the /v1/chat/completions endpoint. The endpoint is a "source"
/// for an [`super::OpenAIChatCompletionsStreamingEngine`] and will return a stream of responses which will be
/// forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
#[tracing::instrument(skip_all)]
async fn chat_completions(
    State(state): State<Arc<DeploymentState>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // todo - extract distributed tracing id and context id from headers
    let (request_id, priority) = extract_request_id(&headers);
    tracing::info!("request_id: {request_id}");

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // update the request to always stream
    let inner_request = async_openai::types::CreateChatCompletionRequest {
        stream: Some(true),
        ..request.inner
    };

    let mut request_payload = NvCreateChatCompletionRequest {
        inner: inner_request,
        nvext: request.nvext,
    };
    request_payload
        .nvext
        .get_or_insert_with(NvExt::default)
        .priority = Some(priority as i64);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request_payload.inner.model;

    // todo - determine the proper error code for when a request model is not present
    tracing::trace!("Getting chat completions engine for model: {}", model);

    let engine = state
        .get_chat_completions_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight: InflightGuard =
        state.create_inflight_guard(model, Endpoint::ChatCompletions, streaming);

    if state.is_rate_limited(model, priority) {
        inflight.mark_429();
        return Ok(ErrorResponse::too_many_requests(model).into_response());
    }
    // setup context
    // todo - inherit request_id from distributed trace details
    let request_ctx = Context::with_id(request_payload, request_id.clone());

    tracing::trace!("Issuing generate call for chat completions");

    // issue the generate call on the engine
    let stream = engine
        .generate(request_ctx)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic
    let mut guard = CtxDropGuard::new(ctx.clone());
    if streaming {
        let stream = stream.map(move |response| {
            guard.not_print_drop(); // ensures that the guard is moved into the stream
            Event::try_from(EventConverter::from(response))
        });
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight).await?;

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }
        Ok(sse_stream.into_response())
    } else {
        let response = NvCreateChatCompletionResponse::from_annotated_stream(stream.into())
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold chat completions stream for request_id {}: {:?}",
                    request_id,
                    e
                );

                // todo: check if HTTPError is prsent in the error chain and return it
                if e.to_string().contains("HTTPExceptionDetected") {
                    ErrorResponse::from_http_error(HttpError {
                        code: 400,
                        message: e.to_string(),
                    })
                } else {
                    ErrorResponse::internal_server_error(&format!(
                        "Failed to fold chat completions stream: {}",
                        e
                    ))
                }
            })?;
        guard.not_print_drop();
        inflight.mark_ok();
        Ok(Json(response).into_response())
    }
}

// todo - abstract this to the top level lib.rs to be reused
// todo - move the service_observer to its own state/arc
fn check_ready(_state: &Arc<DeploymentState>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    // if state.service_observer.stage() != ServiceStage::Ready {
    //     return Err(ErrorResponse::service_unavailable());
    // }
    Ok(())
}

/// list models handler, non-standard format
async fn list_models_custom(
    State(state): State<Arc<DeploymentState>>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    check_ready(&state)?;
    let mut models = HashMap::new();

    let chat_models = state
        .chat_completion_engines
        .lock()
        .unwrap()
        .engines
        .keys()
        .cloned()
        .collect::<Vec<String>>();

    let completion_models = state
        .completion_engines
        .lock()
        .unwrap()
        .engines
        .keys()
        .cloned()
        .collect::<Vec<String>>();

    models.insert("chat_completion_models", chat_models);
    models.insert("completion_models", completion_models);

    Ok(Json(models).into_response())
}

/// openai compatible format
/// Example:
/// {
///  "object": "list",
///  "data": [
///    {
///      "id": "model-id-0",
///      "object": "model",
///      "created": 1686935002,
///      "owned_by": "organization-owner"
///    },
///    ]
/// }
async fn list_models_openai(
    State(state): State<Arc<DeploymentState>>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    check_ready(&state)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut data = Vec::new();

    let models: HashSet<String> = state
        .chat_completion_engines
        .lock()
        .unwrap()
        .engines
        .keys()
        .chain(state.completion_engines.lock().unwrap().engines.keys())
        .cloned()
        .collect();

    for model_id in models {
        data.push(ModelListing {
            id: model_id.clone(),
            object: "object",
            created,                        // Where would this come from? The GGUF?
            owned_by: "nvidia".to_string(), // Get organization from GGUF
        });
    }

    let out = ListModelOpenAI {
        object: "list",
        data,
    };
    Ok(Json(out).into_response())
}

#[derive(Serialize)]
struct ListModelOpenAI {
    object: &'static str, // always "list"
    data: Vec<ModelListing>,
}

#[derive(Serialize)]
struct ModelListing {
    id: String,
    object: &'static str, // always "object"
    created: u64,         //  Seconds since epoch
    owned_by: String,
}

/// Monitors a stream of SSE events.
/// - If the first event is an error, returns an `Err` to allow the caller to send an HTTP error.
/// - Otherwise, sets up a `ReceiverStream` to forward events.
/// - Handles client disconnects by stopping the engine and updating metrics.
/// - Propagates subsequent stream errors through the `ReceiverStream`.
async fn monitor_for_disconnects(
    mut stream: Pin<Box<dyn Stream<Item = Result<Event, axum::Error>> + Send>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight: InflightGuard, // inflight is moved and its state managed by this function or its spawned task
) -> Result<ReceiverStream<Result<Event, axum::Error>>, (StatusCode, Json<ErrorResponse>)> {
    let (tx, rx) = tokio::sync::mpsc::channel(8);

    match stream.next().await {
        None => {
            // TODO: Simplify, there should be no empty stream.
            tracing::warn!(
                "Input stream for context {} was empty. Proceeding with empty SSE stream.",
                context.id()
            );
            tokio::spawn(async move {
                // inflight is moved here
                if !tx.is_closed() {
                    // Check if client is still there
                    if tx.send(Ok(Event::default().data("[DONE]"))).await.is_ok() {
                        inflight.mark_streaming_error();
                    } else {
                        tracing::trace!("SSE client disconnected (before [DONE] for empty stream) for context: {}.", context.id());
                        context.stop_generating(); // Ensure cleanup if client disconnected
                        inflight.mark_client_drop();
                    }
                } else {
                    context.stop_generating(); // Ensure cleanup if client disconnected
                    inflight.mark_client_drop();
                }
            });
            Ok(ReceiverStream::new(rx))
        }
        Some(Err(initial_err)) => {
            context.stop_generating();

            // Check if the source of axum::Error is an HttpError
            if let Some(source) = initial_err.source() {
                if let Some(http_err_ref) = source.downcast_ref::<HttpError>() {
                    // HttpError found as source. Construct an owned HttpError
                    // and use ErrorResponse::from_http_error.
                    inflight.mark_downstream_http_error();
                    let owned_http_err = HttpError {
                        code: http_err_ref.code,
                        message: http_err_ref.message.clone(), // String needs clone
                    };
                    tracing::info!(
                        "Returining HTTP error {}: {}",
                        owned_http_err.code,
                        owned_http_err.message
                    );
                    return Err(ErrorResponse::from_http_error(owned_http_err));
                }
            }
            tracing::warn!(
                "Initial event in stream for context {} resulted in an error: {}. Aborting SSE setup.",
                context.id(),
                initial_err
            );

            // Fallback: if initial_err was not an axum::Error wrapping an HttpError,
            // or if HttpError was not found as its source.
            Err(ErrorResponse::internal_server_error(&format!(
                "Failed to start streaming, initial event error: {}",
                initial_err
            )))
        }
        Some(Ok(first_event)) => {
            // First item was successful. Proceed with streaming.
            tracing::debug!(
                "First event successful for context {}. Starting SSE forwarder task.",
                context.id()
            );
            tokio::spawn(async move {
                // inflight is moved here
                inflight.add_event_time(); // For the first event
                if tx.send(Ok(first_event)).await.is_err() {
                    tracing::trace!("SSE client disconnected (while sending first event) for context: {}. Stopping generation.", context.id());
                    context.stop_generating();
                    inflight.mark_client_drop();
                    return;
                }

                // Process subsequent events
                while let Some(event_result) = stream.next().await {
                    inflight.add_event_time(); // For subsequent events

                    match event_result {
                        Ok(event) => {
                            if tx.send(Ok(event)).await.is_err() {
                                tracing::trace!("SSE client disconnected (while sending subsequent event) for context: {}. Stopping generation.", context.id());
                                context.stop_generating();
                                inflight.mark_client_drop();
                                return;
                            }
                        }
                        Err(err) => {
                            // An error occurred in the stream after it had started.
                            // Propagate this error through the SSE channel. Axum's Sse handler
                            // will typically terminate the connection.
                            tracing::warn!("Error in ongoing SSE event stream for context {}: {}. Propagating to client.", context.id(), err);
                            let send_err_result = tx.send(Err(err)).await;
                            inflight.mark_streaming_error();
                            context.stop_generating(); // Tell engine to stop.

                            if send_err_result.is_err() {
                                tracing::trace!("SSE client disconnected (before error could be sent for ongoing stream) for context: {}.", context.id());
                                // If sending the error itself failed, it's definitely a client drop.
                                inflight.mark_client_drop(); // Override mark_error
                            }
                            return; // Exit task
                        }
                    }
                }

                // Input stream finished naturally.
                if !tx.is_closed() {
                    if tx.send(Ok(Event::default().data("[DONE]"))).await.is_ok() {
                        inflight.mark_ok();
                    } else {
                        tracing::trace!("SSE client disconnected (before [DONE] could be sent) for context: {}.", context.id());
                        context.stop_generating();
                        inflight.mark_client_drop();
                    }
                } else {
                    tracing::debug!(
                        "Input stream ended for context {}, but client was already disconnected.",
                        context.id()
                    );
                    // inflight status should have been set in the loop or when sending first_event
                }
            });
            Ok(ReceiverStream::new(rx))
        }
    }
}

struct EventConverter<T>(Annotated<T>);

impl<T> From<Annotated<T>> for EventConverter<T> {
    fn from(annotated: Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

/// Convert an Annotated into an Event
/// If the Event represents an Error, then return an axum::Error
/// The [`monitor_for_disconnects`] method will handle the error, emit to the sse stream
/// then stop the generation of completions.
impl<T: Serialize> TryFrom<EventConverter<T>> for Event {
    type Error = axum::Error;

    fn try_from(annotated: EventConverter<T>) -> Result<Self, Self::Error> {
        let annotated = annotated.0;
        let mut event = Event::default();

        if let Some(data) = annotated.data {
            event = event.json_data(data)?;
        }

        if let Some(msg) = annotated.event {
            if msg == "error" {
                let msgs = annotated
                    .comment
                    .unwrap_or_else(|| vec!["unspecified error".to_string()]);
                // If exactly two parts, try to deserialize the second part into HttpError.
                if msgs.len() == 2 {
                    if let Ok(http_error) = serde_json::from_str::<HttpError>(&msgs[1]) {
                        // Propagate the HttpError. The Axum handler will then use its IntoResponse impl.
                        return Err(axum::Error::new(http_error));
                    }
                }
                // Fallback using a joined message.
                return Err(axum::Error::new(msgs.join(" -- ")));
            }
            event = event.event(msg);
        }

        if let Some(comments) = annotated.comment {
            for comment in comments {
                event = event.comment(comment);
            }
        }

        Ok(event)
    }
}

/// Create an Axum [`Router`] for the OpenAI API Completions endpoint
/// If not path is provided, the default path is `/v1/completions`
pub fn completions_router(
    state: Arc<DeploymentState>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(completions))
        .with_state(state);
    (vec![doc], router)
}

/// Create an Axum [`Router`] for the OpenAI API Chat Completions endpoint
/// If not path is provided, the default path is `/v1/chat/completions`
pub fn chat_completions_router(
    state: Arc<DeploymentState>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/chat/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(chat_completions))
        .with_state(state);
    (vec![doc], router)
}

/// List Models
pub fn list_models_router(
    state: Arc<DeploymentState>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    // TODO: Why do we have this endpoint?
    let custom_path = path.unwrap_or("/dynamo/alpha/list-models".to_string());
    let doc_for_custom = RouteDoc::new(axum::http::Method::GET, &custom_path);

    // Standard OpenAI compatible list models endpoint
    let openai_path = "/v1/models".to_string();
    let doc_for_openai = RouteDoc::new(axum::http::Method::GET, &openai_path);

    let router = Router::new()
        .route(&custom_path, get(list_models_custom))
        .route(&openai_path, get(list_models_openai))
        .with_state(state);

    (vec![doc_for_custom, doc_for_openai], router)
}

#[cfg(test)]
mod tests {
    use super::super::ServiceHttpError;

    use super::*;

    const BACKUP_ERROR_MESSAGE: &str = "Failed to generate completions";

    fn http_error_from_engine(code: u16) -> Result<(), anyhow::Error> {
        Err(HttpError {
            code,
            message: "custom error message".to_string(),
        })?
    }

    fn other_error_from_engine() -> Result<(), anyhow::Error> {
        Err(ServiceHttpError::ModelNotFound("foo".to_string()))?
    }

    #[test]
    fn test_http_error_response_from_anyhow() {
        let err = http_error_from_engine(400).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response.error, "custom error message");
    }

    #[test]
    fn test_error_response_from_anyhow_out_of_range() {
        let err = http_error_from_engine(399).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");

        let err = http_error_from_engine(500).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");

        let err = http_error_from_engine(501).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");
    }

    #[test]
    fn test_other_error_response_from_anyhow() {
        let err = other_error_from_engine().unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.error,
            format!(
                "{}: {}",
                BACKUP_ERROR_MESSAGE,
                other_error_from_engine().unwrap_err()
            )
        );
    }
}
