use crate::tools::traits::Tool;
use std::sync::Arc;

#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.push(Arc::new(tool));
    }
    pub fn all(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }
    pub fn get_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        for t in &self.tools {
            if t.name() == name {
                return Some(t.clone());
            }
        }
        None
    }
}
