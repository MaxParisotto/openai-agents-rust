use crate::agent::traits::AgentContext;
use crate::error::AgentError;
use crate::tools::traits::Tool;
use async_trait::async_trait;
use serde_json::json;

pub struct FunctionTool<F>
where
    F: Fn(&str) -> Result<String, AgentError> + Send + Sync + 'static,
{
    name: String,
    description: String,
    func: F,
    enabled: bool,
}

impl<F> FunctionTool<F>
where
    F: Fn(&str) -> Result<String, AgentError> + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>, description: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            func,
            enabled: true,
        }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

#[async_trait]
impl<F> Tool for FunctionTool<F>
where
    F: Fn(&str) -> Result<String, AgentError> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn openai_tool_spec(&self) -> Option<serde_json::Value> {
        Some(json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to transform."}
                    },
                    "required": ["text"],
                    "additionalProperties": false
                }
            }
        }))
    }
    async fn is_enabled(&self, _ctx: &AgentContext) -> bool {
        self.enabled
    }
    async fn call(&self, input: &str) -> Result<String, AgentError> {
        (self.func)(input)
    }
    async fn call_with_context(
        &self,
        _ctx: &AgentContext,
        _tool_call_id: Option<&str>,
        input: &str,
    ) -> Result<String, AgentError> {
        (self.func)(input)
    }
}
