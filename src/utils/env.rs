use std::env;

/// Read a boolean environment variable.
/// Accepts 1/true/yes/on (case-insensitive) as true; 0/false/no/off as false.
/// Returns `default` if the variable is unset or empty.
pub fn var_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) if !v.trim().is_empty() => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        _ => default,
    }
}

/// Returns the non-empty value of an environment variable or None.
pub fn var_nonempty(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|v| {
        let t = v.trim();
        if t.is_empty() { None } else { Some(t.to_string()) }
    })
}

/// Returns the raw environment variable value if present.
pub fn var_opt(name: &str) -> Option<String> {
    env::var(name).ok()
}
