pub trait Plugin {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Called when the plugin is initialized.
    fn init(&self);
}