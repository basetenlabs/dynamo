use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use serde::{Deserialize, Serialize};
use chrono::Utc;
use super::*;
use llm_rs::billing::{BillingEvent, BillingPublisher};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyBillingEvent {
    pub output_tokens: i32,
    pub input_tokens: i32,
    pub organization_id: String,
    pub model_name: Option<String>,
}

#[pymethods]
impl PyBillingEvent {
    #[pyo3(signature = (output_tokens, input_tokens, organization_id, model_name=None))]
    #[new]
    pub fn new(output_tokens: i32, input_tokens: i32, organization_id: String, model_name: Option<String>) -> Self {
        Self {
            output_tokens,
            input_tokens,
            organization_id,
            model_name,
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
            model_name: model_name.clone(),
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