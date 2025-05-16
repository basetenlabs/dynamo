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

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use prometheus::{Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};
use serde_json::json;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub use prometheus::Registry;

use super::{DeploymentState, RouteDoc};

/// Value for the `status` label in the request counter for successful requests
pub const REQUEST_STATUS_SUCCESS: &str = "success";

pub const REQUEST_STATUS_CLIENT_DROP: &str = "client_drop";

pub const REQUEST_STATUS_429: &str = "429";

/// Value for the `status` label in the request counter if the request failed
pub const REQUEST_STATUS_ERROR: &str = "error";

// Value for sending a sse-event error during the streaming
pub const REQUEST_STATUS_STREAMING_ERROR: &str = "streaming_error";

// Value for initial invalidation via HTTPError from processor
pub const REQUEST_STATUS_DOWNSTREAM_HTTP_ERROR: &str = "downstream_http_error";

/// Partial value for the `type` label in the request counter for streaming requests
pub const REQUEST_TYPE_STREAM: &str = "stream";

/// Partial value for the `type` label in the request counter for unary requests
pub const REQUEST_TYPE_UNARY: &str = "unary";

pub struct Metrics {
    request_counter: IntCounterVec,
    inflight_gauge: IntGaugeVec,
    request_duration: HistogramVec,
    ttfb: HistogramVec,
    ibl_p50: HistogramVec,
    ibl_p90: HistogramVec,
    ibl_p99: HistogramVec,
    rolling_ttfb: Mutex<HashMap<String, VecDeque<(Instant, Duration)>>>,
}

/// RAII object for inflight gauge and request counters
/// If this object is dropped without calling `mark_ok`, then the request will increment
/// the request counter with the `status` label with [`REQUEST_STATUS_ERROR`]; otherwise, it will increment
/// the counter with `status` label [`REQUEST_STATUS_SUCCESS`]
pub struct InflightGuard {
    metrics: Arc<Metrics>,
    model: String,
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    timer: Instant,
    events_times: Vec<Duration>,
}

/// Requests will be logged by the type of endpoint hit
/// This will include llamastack in the future
pub enum Endpoint {
    /// OAI Completions
    Completions,

    /// OAI Chat Completions
    ChatCompletions,
}

/// Metrics for the HTTP service
pub enum RequestType {
    /// SingleIn / SingleOut
    Unary,

    /// SingleIn / ManyOut
    Stream,
}

/// Status
pub enum Status {
    Success,
    // client side dropped
    ClientDrop,
    // rate limited
    TooManyRequests,
    // HTTP Error
    DownstreamHTTPError,
    // client side error
    StreamingError,
    // general error
    Error,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new("nv_llm")
    }
}

impl Metrics {
    /// Create Metrics with the given prefix
    /// The following metrics will be created:
    /// - `{prefix}_http_service_requests_total` - IntCounterVec for the total number of requests processed
    /// - `{prefix}_http_service_inflight_requests` - IntGaugeVec for the number of inflight requests
    /// - `{prefix}_http_service_request_duration_seconds` - HistogramVec for the duration of requests
    pub fn new(prefix: &str) -> Self {
        let request_counter = IntCounterVec::new(
            Opts::new(
                format!("{}_http_service_requests_total", prefix),
                "Total number of LLM requests processed",
            ),
            &["model", "endpoint", "request_type", "status"],
        )
        .unwrap();

        let inflight_gauge = IntGaugeVec::new(
            Opts::new(
                format!("{}_http_service_inflight_requests", prefix),
                "Number of inflight requests",
            ),
            &["model"],
        )
        .unwrap();

        let buckets = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                format!("{}_http_service_request_duration_seconds", prefix),
                "Duration of LLM requests",
            )
            .buckets(buckets),
            &["model"],
        )
        .unwrap();

        let buckets_ttfb = vec![
            0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
            3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
        ];
        let ttfb = HistogramVec::new(
            HistogramOpts::new(
                format!("{}_http_service_ttfb_seconds", prefix),
                "Time to first byte for LLM requests",
            )
            .buckets(buckets_ttfb.clone()),
            &["model"],
        )
        .unwrap();

        let mut ibl = Vec::new();
        for p in [0.5, 0.9, 0.99] {
            let ibl_p = HistogramVec::new(
                HistogramOpts::new(
                    format!(
                        "{}_http_service_ibl_p{}_seconds",
                        prefix,
                        p.to_string().replace('.', "_")
                    ),
                    format!(
                        "Time to inter byte latency for LLM requests at {} percentile",
                        p
                    ),
                )
                .buckets(buckets_ttfb.clone()),
                &["model"],
            )
            .unwrap();
            ibl.push(ibl_p);
        }

        Metrics {
            request_counter,
            inflight_gauge,
            request_duration,
            ttfb,
            ibl_p50: ibl[0].clone(),
            ibl_p90: ibl[1].clone(),
            ibl_p99: ibl[2].clone(),
            rolling_ttfb: Mutex::new(HashMap::new()),
        }
    }

    fn record_ttfb(&self, model: &str, duration: Duration) {
        let mut per_model_ttfb = self.rolling_ttfb.lock().unwrap();
        let model_completions = per_model_ttfb
            .entry(model.to_string())
            .or_insert_with(VecDeque::new);

        let now = Instant::now();
        model_completions.push_back((now, duration));
    }

    /// get recent ttfb times for the given model
    pub fn get_recent_ttfb_times(&self, model: &str) -> (Option<Duration>, Option<Duration>) {
        let mut per_model_ttfb = self.rolling_ttfb.lock().unwrap();
        if let Some(times) = per_model_ttfb
            .get_mut(model)
            .filter(|times| !times.is_empty())
        {
            // Prune to last 60 seconds
            let now = Instant::now();
            times.retain(|(start_time, _)| {
                now.duration_since(*start_time) <= Duration::from_secs(60)
            });
            // Check again after pruning
            if times.is_empty() {
                return (None, None);
            }
            let mean_time = times.iter().map(|(_, t)| *t).sum::<Duration>() / times.len() as u32;
            let median_time = if times.len() % 2 == 0 {
                let mid = times.len() / 2;
                (times[mid - 1].1 + times[mid].1) / 2
            } else {
                times[times.len() / 2].1
            };
            return (Some(mean_time), Some(median_time));
        }
        (None, None)
    }

    /// Get the number of successful requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    pub fn get_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .get()
    }

    /// Increment the counter for requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    fn inc_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .inc()
    }

    /// Get the number if inflight requests for the given model
    pub fn get_inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    fn inc_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc()
    }

    fn dec_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec()
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        registry.register(Box::new(self.ttfb.clone()))?;
        registry.register(Box::new(self.ibl_p50.clone()))?;
        registry.register(Box::new(self.ibl_p90.clone()))?;
        registry.register(Box::new(self.ibl_p99.clone()))?;
        Ok(())
    }
}

impl DeploymentState {
    /// Create a new [`InflightGuard`] for the given model and annotate if its a streaming request,
    /// and the kind of endpoint that was hit
    ///
    /// The [`InflightGuard`] is an RAII object will handle incrementing the inflight gauge and
    /// request counters.
    pub fn create_inflight_guard(
        &self,
        model: &str,
        endpoint: Endpoint,
        streaming: bool,
    ) -> InflightGuard {
        let request_type = if streaming {
            RequestType::Stream
        } else {
            RequestType::Unary
        };

        InflightGuard::new(
            self.metrics.clone(),
            model.to_string(),
            endpoint,
            request_type,
        )
    }
}

impl InflightGuard {
    fn new(
        metrics: Arc<Metrics>,
        model: String,
        endpoint: Endpoint,
        request_type: RequestType,
    ) -> Self {
        // Start the timer
        let timer = Instant::now();

        // Increment the inflight gauge when the guard is created
        metrics.inc_inflight_gauge(&model);

        // Return the RAII Guard
        InflightGuard {
            metrics,
            model,
            endpoint,
            request_type,
            status: Status::Error,
            timer,
            events_times: Vec::new(), // vec of event times
        }
    }

    pub(crate) fn add_event_time(&mut self) {
        let duration = Instant::now().duration_since(self.timer);
        if self.events_times.len() == 0 {
            // this is the first event, so we can record the TTFB
            self.metrics
                .ttfb
                .with_label_values(&[&self.model])
                .observe(duration.as_secs_f64());
            self.metrics.record_ttfb(&self.model, duration);
        }

        self.events_times.push(duration);
    }

    pub(crate) fn mark_429(&mut self) {
        self.status = Status::TooManyRequests;
    }

    pub(crate) fn mark_streaming_error(&mut self) {
        self.status = Status::StreamingError;
    }

    pub(crate) fn mark_downstream_http_error(&mut self) {
        self.status = Status::DownstreamHTTPError;
    }

    pub(crate) fn mark_client_drop(&mut self) {
        self.status = Status::ClientDrop;
    }

    pub(crate) fn mark_ok(&mut self) {
        self.status = Status::Success;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        // Decrement the gauge when the guard is dropped
        self.metrics.dec_inflight_gauge(&self.model);

        // the frequency on incrementing the full request counter is relatively low
        // if we were incrementing the counter on every forward pass, we'd use static CounterVec or
        // discrete counter object without the more costly lookup required for the following calls
        self.metrics.inc_request_counter(
            &self.model,
            &self.endpoint,
            &self.request_type,
            &self.status,
        );

        let duration = self.timer.elapsed();

        // Record the duration of the request
        self.metrics
            .request_duration
            .with_label_values(&[&self.model])
            .observe(duration.as_secs_f64());

        // Record the inter byte latency for each event, not counting the first event
        // aggreate statistics across all events of this inflight request
        if self.events_times.len() > 1 {
            // ibl values are already Duration between events, so below code is commented out
            let mut ibl_values_f64: Vec<f64> = Vec::with_capacity(self.events_times.len() - 1);
            for i in 1..self.events_times.len() {
                // Calculate IBL as the difference between consecutive event durations
                // (assuming events_duration stores cumulative durations from request start)
                let ibl_duration = self.events_times[i].saturating_sub(self.events_times[i - 1]);
                ibl_values_f64.push(ibl_duration.as_secs_f64());
            }

            if !ibl_values_f64.is_empty() {
                ibl_values_f64
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Helper to calculate percentile from sorted f64 slice.
                // Uses (L * P).ceil() - 1 rule for 0-indexed array of length L.
                let calculate_percentile = |sorted_data: &[f64], p_fraction: f64| -> Option<f64> {
                    if sorted_data.is_empty() {
                        return None;
                    }
                    let len = sorted_data.len();
                    // Calculate rank: (length * percentile_fraction), rounded up.
                    // Convert to 0-based index: rank - 1.
                    // Ensure index is within bounds [0, len-1].
                    let rank_ceil = (len as f64 * p_fraction).ceil();
                    let index = (rank_ceil as usize).saturating_sub(1).min(len - 1);
                    Some(sorted_data[index])
                };

                if let Some(p50) = calculate_percentile(&ibl_values_f64, 0.50) {
                    self.metrics
                        .ibl_p50
                        .with_label_values(&[&self.model])
                        .observe(p50);
                }
                if let Some(p90) = calculate_percentile(&ibl_values_f64, 0.90) {
                    self.metrics
                        .ibl_p90
                        .with_label_values(&[&self.model])
                        .observe(p90);
                }
                if let Some(p99) = calculate_percentile(&ibl_values_f64, 0.99) {
                    self.metrics
                        .ibl_p99
                        .with_label_values(&[&self.model])
                        .observe(p99);
                }
            }
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::ChatCompletions => write!(f, "chat_completions"),
        }
    }
}

impl Endpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Endpoint::Completions => "completions",
            Endpoint::ChatCompletions => "chat_completions",
        }
    }
}

impl RequestType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Unary => REQUEST_TYPE_UNARY,
            RequestType::Stream => REQUEST_TYPE_STREAM,
        }
    }
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Success => REQUEST_STATUS_SUCCESS,
            Status::ClientDrop => REQUEST_STATUS_CLIENT_DROP,
            Status::TooManyRequests => REQUEST_STATUS_429,
            Status::DownstreamHTTPError => REQUEST_STATUS_ERROR,
            Status::Error => REQUEST_STATUS_ERROR,
            Status::StreamingError => REQUEST_STATUS_STREAMING_ERROR,
        }
    }
}

/// Create a new router with the given path for handler_metrics and handler_health_model
pub fn router(
    registry: Registry,
    metrics: Arc<Metrics>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let registry_arc = Arc::new(registry);
    let app_state = (registry_arc, metrics.clone());

    let metrics_path = path.unwrap_or_else(|| "/metrics".to_string());
    let health_model_path_str = "/health_model/{model_name}";

    let doc_metrics = RouteDoc::new(axum::http::Method::GET, &metrics_path);
    let doc_health_model = RouteDoc::new(axum::http::Method::GET, health_model_path_str);

    let service_router = Router::new() // without v07 checks
        .route(&metrics_path, get(handler_metrics))
        .route(health_model_path_str, get(handler_health_model))
        .with_state(app_state);

    let docs = vec![doc_metrics, doc_health_model];
    (docs, service_router)
}

/// Metrics Handler
async fn handler_metrics(
    State((registry, _)): State<(Arc<Registry>, Arc<Metrics>)>,
) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = vec![];
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        )
            .into_response();
    }

    let metrics = match String::from_utf8(buffer) {
        Ok(metrics) => metrics,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response()
        }
    };

    (StatusCode::OK, metrics).into_response()
}

// handle health_model
// this route allows a get request to /health_model/{model_name}
// and returns 200 OK with the count of inflight + past 60s request end2end times as json payload
// 200: {"model_name": { "inflight": 15, "end2end_mean_time": 7.5 } }
// 400: {"model_name": { "inflight": 0, "end2end_mean_time": 0 } }

async fn handler_health_model(
    State((_, current_metrics)): State<(Arc<Registry>, Arc<Metrics>)>,
    Path(model_name): Path<String>,
) -> impl IntoResponse {
    let inflight = current_metrics.get_inflight_count(&model_name);
    let (mean_ttfb, median_ttfb) = current_metrics.get_recent_ttfb_times(&model_name);

    let (status, response_body) = match (mean_ttfb, median_ttfb) {
        (Some(mean_ttfb), Some(median_ttfb)) => {
            let body = serde_json::to_string(&json!({
                "model_name": model_name,
                "data": {
                    "inflight": inflight,
                    "mean_ttfb": mean_ttfb.as_secs_f64(),
                    "median_ttfb": median_ttfb.as_secs_f64()
                }
            }))
            .unwrap();
            (StatusCode::OK, body)
        }
        _ => {
            let body = serde_json::to_string(&json!({
                "model_name": model_name,
                "data": {
                    "inflight": inflight,
                    "mean_ttfb": mean_ttfb.unwrap_or_default().as_secs_f64(),
                    "median_ttfb": median_ttfb.unwrap_or_default().as_secs_f64()
                }
            }))
            .unwrap();
            (StatusCode::SERVICE_UNAVAILABLE, body)
        }
    };

    (status, response_body).into_response()
}
