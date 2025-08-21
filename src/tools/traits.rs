use crate::agent::traits::AgentContext;
use crate::error::AgentError;
use async_trait::async_trait;
use serde_json::Value;

/// Basic tool trait comparable to Python FunctionTool.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str {
        ""
    }
    /// Optional OpenAI Chat Completions tool spec for this tool.
    /// Return None if the tool shouldn't be exposed to the model.
    fn openai_tool_spec(&self) -> Option<Value> {
        None
    }
    /// Whether the tool is enabled in the current context.
    async fn is_enabled(&self, _ctx: &AgentContext) -> bool {
        true
    }
    /// Execute the tool with a string input and return a string output.
    async fn call(&self, input: &str) -> Result<String, AgentError>;

    /// Optional context-aware call. Default delegates to `call`.
    async fn call_with_context(
        &self,
        ctx: &AgentContext,
        _tool_call_id: Option<&str>,
        input: &str,
    ) -> Result<String, AgentError> {
        let _ = ctx; // unused by default
        self.call(input).await
    }
}
