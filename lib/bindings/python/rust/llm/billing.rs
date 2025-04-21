use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
// chrono
use chrono::Utc;
// bring in *your* Python binding Component (and anything else from super)
use super::*;

// Import your BillingEvent and BillingPublisher from llm_rs
use llm_rs::billing::{BillingEvent, BillingPublisher};

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyBillingEvent {
    pub output_tokens: i32,
    pub input_tokens: i32,
    pub organization_id: String,
}

#[pymethods]
impl PyBillingEvent {
    #[new]
    pub fn new(output_tokens: i32, input_tokens: i32, organization_id: String) -> Self {
        Self {
            output_tokens,
            input_tokens,
            organization_id,
        }
    }
}

impl From<&PyBillingEvent> for BillingEvent {
    fn from(event: &PyBillingEvent) -> Self {
        // get utc timestamp
        let timestamp = Utc::now().timestamp_millis() as u64;
        BillingEvent {
            timestamp: timestamp,
            output_tokens: event.output_tokens,
            input_tokens: event.input_tokens,
            organization_id: event.organization_id.clone(),
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
        // component.inner is the runtime Component; clone that for the publisher.
        let publisher = BillingPublisher::new(component.inner.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        Ok(Self { inner: publisher })
    }

    fn publish(&self, event: &PyBillingEvent) -> PyResult<()> {
        self.inner
            .publish(event.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }
}
