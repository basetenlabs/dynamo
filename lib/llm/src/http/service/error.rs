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

#![allow(clippy::module_inception)]
use std::error;

use thiserror::Error;
use serde::{Deserialize, Serialize};
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;
use tracing::warn;

#[derive(Debug, Error)]
pub enum ServiceHttpError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),
}

/// Implementation of the Completion Engines served by the HTTP service should
/// map their custom errors to to this error type if they wish to return error
/// codes besides 500.
#[derive(Debug, Error, Serialize, Deserialize)]
#[error("HTTP Error {code}: {message}")]
pub struct HttpError {
    pub code: u16,
    pub message: String,
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        // Build the response with the given status code and JSON error body.
        // check if code between 400 and 599
        if self.code < 400 || self.code > 599 {
            warn!("Invalid HTTP error code: {} {}", self.code, &self.message);
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        (StatusCode::from_u16(self.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), self.message).into_response()
    }
}