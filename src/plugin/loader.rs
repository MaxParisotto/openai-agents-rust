use crate::plugin::traits::Plugin;
use libloading;
use std::path::Path;

/// Simple plugin registry that holds a list of plugins.
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin + Send + Sync>>,
}

impl PluginRegistry {
    /// Creates a new, empty registry.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Registers a plugin.
    pub fn register<P: Plugin + Send + Sync + 'static>(&mut self, plugin: P) {
        self.plugins.push(Box::new(plugin));
    }

    /// Register a plugin that is already boxed.
    pub fn register_box(&mut self, plugin: Box<dyn Plugin + Send + Sync>) {
        self.plugins.push(plugin);
    }

    /// Returns an iterator over the registered plugins.
    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Plugin + Send + Sync>> {
        self.plugins.iter()
    }

    /// Load plugins from a directory.
    ///
    /// This implementation scans the given directory for shared library files
    /// (`.so` on Linux, `.dylib` on macOS). Each library is expected to expose a
    /// C‑compatible symbol named `plugin_create` with the signature:
    ///
    /// ```text
    /// unsafe extern "C" fn() -> *mut dyn Plugin
    /// ```
    ///
    /// The function should allocate a concrete type that implements `Plugin` and
    /// return a raw pointer. The loader will convert the raw pointer into a
    /// `Box<dyn Plugin>` and register it. The loaded library is deliberately
    /// leaked (`std::mem::forget`) to keep it alive for the duration of the
    /// program; a production implementation would store the `Library` handles
    /// inside the registry to manage their lifetimes.
    pub fn load_from_dir<P: AsRef<Path>>(path: P) -> Result<Self, crate::error::AgentError> {
        let mut registry = Self::new();

        let entries =
            std::fs::read_dir(&path).map_err(|e| crate::error::AgentError::Other(e.to_string()))?;

        for entry in entries {
            let entry = entry.map_err(|e| crate::error::AgentError::Other(e.to_string()))?;
            let lib_path = entry.path();

            // Only consider files with typical shared‑library extensions.
            if let Some(ext) = lib_path.extension().and_then(|s| s.to_str()) {
                if ext != "so" && ext != "dylib" {
                    continue;
                }
            } else {
                continue;
            }

            // Load the library.
            let lib = unsafe {
                libloading::Library::new(&lib_path)
                    .map_err(|e| crate::error::AgentError::Other(e.to_string()))?
            };

            // Look for the expected symbol.
            unsafe {
                // The plugin constructor must return a boxed plugin that satisfies Send + Sync.
                let ctor: libloading::Symbol<unsafe fn() -> *mut (dyn Plugin + Send + Sync)> = lib
                    .get(b"plugin_create")
                    .map_err(|e| crate::error::AgentError::Other(e.to_string()))?;

                // Call the constructor to obtain a raw pointer.
                let raw = ctor();

                // Register the plugin directly from the raw pointer.
                registry.register_box(Box::from_raw(raw));
            }

            // Keep the library alive for the program's lifetime.
            // In this simple implementation we deliberately leak it.
            std::mem::forget(lib);
        }

        Ok(registry)
    }
}
impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
