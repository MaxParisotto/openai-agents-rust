use crate::agent::runner::ToolUseBehavior;

// Agent configuration scaffold to move toward parity with the Python SDK.
#[derive(Clone)]
pub struct AgentConfig {
    pub name: String,
    pub handoff_description: Option<String>,
    pub model_name: Option<String>,
    pub reset_tool_choice: bool,
    pub instructions: Option<String>,
    pub tool_use_behavior: ToolUseBehavior,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            handoff_description: None,
            model_name: None,
            reset_tool_choice: true,
            instructions: None,
            tool_use_behavior: ToolUseBehavior::RunLlmAgain,
        }
    }
}

impl AgentConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}
