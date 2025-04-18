use anyhow::Error;
use async_nats::{RequestError, RequestErrorKind};
use tokio::time::{sleep, Duration};

const MAX_RETRIES: usize = 10;
const RETRY_DELAY_MS: u64 = 100; // Backoff delay

pub async fn retry_request<F, Fut, T>(mut f: F) -> Result<T, Error>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, Error>>,
{
    for attempt in 0..MAX_RETRIES {
        match f().await {
            Ok(response) => return Ok(response),
            Err(e) => {
                if let Some(RequestErrorKind::NoResponders) = e.downcast_ref::<RequestError>().map(|e| e.kind()) {
                    tracing::warn!("No responders (attempt {}/{}), retrying...", attempt + 1, MAX_RETRIES);
                    sleep(Duration::from_millis(RETRY_DELAY_MS * (attempt as u64 + 1))).await;
                    continue;
                } else {
                    return Err(e);
                }
            }
        }
    }

    Err(anyhow::anyhow!("No responders after {} retries", MAX_RETRIES))
}
