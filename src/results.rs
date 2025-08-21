/// Structured run result aligning with Python's RunResult basics.
#[derive(Debug, Clone, Default)]
pub struct RunResult {
    pub id: Option<String>,
    pub text: Option<String>,
    /// Optional summaries of tool calls executed during the run: (name, output JSON or text)
    pub tool_outputs: Vec<(String, String)>,
}
