//! Identifier conventions.

/// `snake_case` to `lowerCamelCase`.
pub(crate) fn lower_camel(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = false;
    for c in name.chars() {
        if c == '_' {
            upper = !out.is_empty();
        } else if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

/// `PascalCase` to `lowerCamelCase`, for Swift enum cases.
pub(crate) fn lower_first(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// A Java name as JNI spells it in a native method's symbol (JNI
/// specification, "Resolving Native Method Names").
pub(crate) fn jni_mangle(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '.' | '/' => out.push('_'),
            '_' => out.push_str("_1"),
            ';' => out.push_str("_2"),
            '[' => out.push_str("_3"),
            c if c.is_ascii_alphanumeric() => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("_0{unit:04x}"));
                }
            }
        }
    }
    out
}

/// Doc text as comment lines with `prefix` (`" * "`, `"/// "`).
pub(crate) fn comment(docs: &str, prefix: &str, indent: &str) -> String {
    docs.lines()
        .map(|line| {
            if line.is_empty() {
                format!("{indent}{}\n", prefix.trim_end())
            } else {
                format!("{indent}{prefix}{line}\n")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases() {
        assert_eq!(lower_camel("load_style_url"), "loadStyleUrl");
        assert_eq!(lower_camel("x"), "x");
        assert_eq!(lower_first("ZoomIn"), "zoomIn");
    }

    #[test]
    fn jni_names() {
        assert_eq!(jni_mangle("dev.orbis.Native"), "dev_orbis_Native");
        assert_eq!(jni_mangle("a_b"), "a_1b");
        assert_eq!(jni_mangle("é"), "_000e9");
    }
}
