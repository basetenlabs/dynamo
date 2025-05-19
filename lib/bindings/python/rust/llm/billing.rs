use super::*;
use chrono::Utc;
use llm_rs::billing::{BillingEvent, BillingPublisher};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyBillingEvent {
    pub output_tokens: i32,
    pub input_tokens: i32,
    pub organization_id: String,
    pub request_id: String,
    pub billing_model_version: String,
    pub model_name: Option<String>,
}

#[pymethods]
impl PyBillingEvent {
    #[pyo3(signature = (output_tokens, input_tokens, organization_id, request_id, billing_model_version, model_name=None))]
    #[new]
    pub fn new(
        output_tokens: i32,
        input_tokens: i32,
        organization_id: String,
        request_id: String,
        billing_model_version: String,
        model_name: Option<String>,
    ) -> Self {
        Self {
            output_tokens,
            input_tokens,
            organization_id,
            request_id,
            model_name,
            billing_model_version,
        }
    }
}

impl From<&PyBillingEvent> for BillingEvent {
    fn from(event: &PyBillingEvent) -> Self {
        let timestamp = Utc::now().timestamp_millis() as u64;
        let model_name = event.model_name.clone().unwrap_or_else(|| "".to_string());
        BillingEvent {
            timestamp,
            output_tokens: event.output_tokens,
            input_tokens: event.input_tokens,
            organization_id: event.organization_id.clone(),
            request_id: event.request_id.clone(),
            model_name: model_name,
            billing_model_version: event.billing_model_version.clone(),
        }
    }
}

#[pyclass]
pub struct PyBillingPublisher {
    inner: BillingPublisher,
}

#[pymethods]
impl PyBillingPublisher {
    #[new]
    fn new(component: Component) -> PyResult<Self> {
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        // Block on the async constructor of BillingPublisher
        let publisher = runtime.block_on(async {
            BillingPublisher::new(component.inner.clone())
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        })?;
        Ok(PyBillingPublisher { inner: publisher })
    }

    /// Publish a billing event (this will spawn onto the same Tokio reactor).
    fn publish(&self, event: &PyBillingEvent) -> PyResult<()> {
        self.inner
            .publish(event.into())
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}
