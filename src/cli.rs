use clap::Parser;
use crate::config::Config;
use crate::agent::AgentRuntime;
use crate::error::AgentError;

/// Command‑line interface for the OpenAI Agents crate.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Path to the configuration file (YAML or JSON)
    #[arg(short, long, default_value = "config.yaml")]
    pub config: String,
}

/// Entry point for the CLI. Parses arguments, loads configuration,
/// creates the runtime and starts it.
pub async fn run() -> Result<(), AgentError> {
    let cli = Cli::parse();
    let config = Config::load_from_path(&cli.config)?;
    let runtime = AgentRuntime::new(config);
    runtime.start().await?;
    Ok(())
}