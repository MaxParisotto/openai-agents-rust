use tracing::{info, Level};
use tracing_subscriber::{fmt, EnvFilter};

/// Initialize global tracing subscriber.
///
/// This function sets up a `tracing_subscriber` that respects the `RUST_LOG`
/// environment variable (or defaults to `info`). It should be called once at
/// application startup, typically from `main.rs`.
pub fn init_tracing() {
    // Build a subscriber that formats logs in a human‑readable way.
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true);

    // Use `EnvFilter` to enable dynamic log level control via `RUST_LOG`.
    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Combine layers and set as the global default.
    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();

    info!(level = %Level::INFO, "Tracing initialized");
}