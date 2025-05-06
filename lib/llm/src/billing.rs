use dynamo_runtime::traits::events::EventPublisher;
use dynamo_runtime::{component::Component, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use tokio::sync::mpsc;
use tracing as log;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingEvent {
    pub timestamp: u64,
    pub output_tokens: i32,
    pub input_tokens: i32,
    pub organization_id: String,
    pub request_id: String,
    pub model_name: String,
    pub billing_model_version: String,
}

impl fmt::Display for BillingEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BillingEvent(timestamp: {}, output_tokens: {}, input_tokens: {}, organization_id: {}, request_id: {}, model_name: {}, billing_model_version: {})",
            self.timestamp,
            self.output_tokens,
            self.input_tokens,
            self.organization_id,
            self.request_id,
            self.model_name,
            self.billing_model_version,
        )
    }
}

pub struct BillingPublisher {
    tx: mpsc::UnboundedSender<BillingEvent>,
}

impl BillingPublisher {
    pub fn new(component: Component) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<BillingEvent>();
        start_billing_publish_task(component, rx);
        Ok(BillingPublisher { tx })
    }

    pub fn publish(&self, event: BillingEvent) -> Result<(), mpsc::error::SendError<BillingEvent>> {
        if event.organization_id.starts_with("org-internal") {
            log::info!("Skipping internal publish. Billing event: {}", event);
            Ok(())
        } else {
            log::debug!("Publishing billing event: {:?}", event);
            self.tx.send(event)
        }
    }
}

fn start_billing_publish_task(component: Component, mut rx: mpsc::UnboundedReceiver<BillingEvent>) {
    let billing_subject = "token_events".to_string();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            // Publish to billing subject via the component's NATS client
            if let Err(e) = component.publish(&billing_subject, &event).await {
                log::error!("Failed to publish billing event: {:?}", e);
            }
        }
    });
}
