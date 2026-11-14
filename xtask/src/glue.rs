//! Short names for the imports wasm-bindgen gives a module from its JS
//! glue (`./<name>_bg.js`). wasm-bindgen names each after the binding it
//! calls plus 16 hex digits of a hash (`__wbg_get_d7176cfb76d71114`), and
//! the module and the glue both carry every name: in core.wasm 500 of them
//! were 4 KiB brotli, hashes that don't compress. The glue reads each only
//! as a key of the object it hands the module, so renaming both, before
//! wasm-opt and the split, changes nothing else.

use std::collections::HashMap;
use std::path::Path;

use wasmparser::{Imports, Parser, Payload};

/// Renames the glue's imports in `<name>_bg.wasm` and `<name>.js` in `out`.
pub(crate) fn shorten_imports(out: &Path, name: &str) -> Result<(), String> {
    let wasm_path = out.join(format!("{name}_bg.wasm"));
    let js_path = out.join(format!("{name}.js"));
    let wasm = std::fs::read(&wasm_path).map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    let js =
        std::fs::read_to_string(&js_path).map_err(|e| format!("{}: {e}", js_path.display()))?;
    let glue = format!("./{name}_bg.js");
    let keys = object_keys(&js);
    // Each import entry's offset, and whether to rename it.
    let mut entries: Vec<(usize, Option<String>)> = Vec::new();
    let mut section_end = 0;
    let mut renames: HashMap<String, String> = HashMap::new();
    for payload in Parser::new(0).parse_all(&wasm) {
        let Payload::ImportSection(reader) = payload.map_err(|e| e.to_string())? else {
            continue;
        };
        section_end = usize::try_from(reader.range().end).map_err(|e| e.to_string())?;
        for group in reader.into_iter_with_offsets() {
            let (offset, group) = group.map_err(|e| e.to_string())?;
            let Imports::Single(_, import) = group else {
                return Err(format!(
                    "{}: compact imports, which shorten_imports can't rename",
                    wasm_path.display()
                ));
            };
            // A name the glue reads anywhere but as its object's key keeps
            // its name.
            let short = (import.module == glue && keys.get(import.name) == Some(&1)).then(|| {
                let short = short_name(renames.len());
                renames.insert(import.name.to_string(), short.clone());
                short
            });
            entries.push((usize::try_from(offset).map_err(|e| e.to_string())?, short));
        }
    }
    if renames.is_empty() {
        return Err(format!(
            "{}: no import of `{glue}` to rename: wasm-bindgen's output changed, so update glue.rs",
            wasm_path.display()
        ));
    }
    std::fs::write(&wasm_path, renamed_imports(&wasm, &entries, section_end)?)
        .map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    std::fs::write(&js_path, renamed_keys(&js, &renames))
        .map_err(|e| format!("{}: {e}", js_path.display()))
}

/// The `n`th short name: a letter, then letters and digits.
fn short_name(mut n: usize) -> String {
    const FIRST: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const REST: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    let mut name = vec![FIRST[n % FIRST.len()]];
    n /= FIRST.len();
    while n > 0 {
        n -= 1;
        name.push(REST[n % REST.len()]);
        n /= REST.len();
    }
    String::from_utf8(name).expect("ASCII")
}

/// The identifiers of `js` and how often each appears, counting only those
/// that every appearance writes as an object's key (`name:`), so a name
/// the glue reads anywhere else counts as never.
fn object_keys(js: &str) -> HashMap<&str, usize> {
    let mut keys: HashMap<&str, usize> = HashMap::new();
    let mut elsewhere: Vec<&str> = Vec::new();
    for (at, ident) in identifiers(js) {
        if js[at + ident.len()..].starts_with(':') {
            *keys.entry(ident).or_default() += 1;
        } else {
            elsewhere.push(ident);
        }
    }
    for ident in elsewhere {
        keys.remove(ident);
    }
    keys
}

/// `js` with each identifier `renames` names replaced.
fn renamed_keys(js: &str, renames: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(js.len());
    let mut last = 0;
    for (at, ident) in identifiers(js) {
        if let Some(short) = renames.get(ident) {
            out.push_str(&js[last..at]);
            out.push_str(short);
            last = at + ident.len();
        }
    }
    out.push_str(&js[last..]);
    out
}

/// The identifiers in `js` and where each starts. Strings and comments
/// aren't told apart: a name the glue spells in one doesn't count as a key.
fn identifiers(js: &str) -> impl Iterator<Item = (usize, &str)> {
    let bytes = js.as_bytes();
    let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'$';
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < bytes.len() && !word(bytes[i]) {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && word(bytes[i]) {
            i += 1;
        }
        (start < i).then(|| (start, &js[start..i]))
    })
}

/// `wasm` with its import section's entries (by offset, up to
/// `section_end`) given the names in `entries`.
fn renamed_imports(
    wasm: &[u8],
    entries: &[(usize, Option<String>)],
    section_end: usize,
) -> Result<Vec<u8>, String> {
    let mut section = Vec::new();
    leb128(&mut section, entries.len());
    for (i, (offset, short)) in entries.iter().enumerate() {
        let end = entries.get(i + 1).map_or(section_end, |(next, _)| *next);
        let entry = &wasm[*offset..end];
        // The module's name, then the import's, then its type.
        let module_end = string_end(entry, 0)?;
        let name_end = string_end(entry, module_end)?;
        section.extend_from_slice(&entry[..module_end]);
        match short {
            Some(short) => {
                leb128(&mut section, short.len());
                section.extend_from_slice(short.as_bytes());
            }
            None => section.extend_from_slice(&entry[module_end..name_end]),
        }
        section.extend_from_slice(&entry[name_end..]);
    }
    // The sections in order, the import section's body replaced.
    let mut out = wasm[..8].to_vec();
    let mut at = 8;
    while at < wasm.len() {
        let id = wasm[at];
        let (size, body) = read_leb128(wasm, at + 1)?;
        let end = body + size;
        out.push(id);
        if id == 2 {
            leb128(&mut out, section.len());
            out.extend_from_slice(&section);
        } else {
            out.extend_from_slice(&wasm[at + 1..end]);
        }
        at = end;
    }
    Ok(out)
}

/// Where the length-prefixed string at `at` in `bytes` ends.
fn string_end(bytes: &[u8], at: usize) -> Result<usize, String> {
    let (len, start) = read_leb128(bytes, at)?;
    let end = start + len;
    (end <= bytes.len())
        .then_some(end)
        .ok_or_else(|| "an import's name runs past its entry".to_string())
}

fn read_leb128(bytes: &[u8], mut at: usize) -> Result<(usize, usize), String> {
    let mut value = 0usize;
    let mut shift = 0;
    loop {
        let byte = *bytes.get(at).ok_or("a LEB128 runs past the module")?;
        at += 1;
        value |= usize::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, at));
        }
        shift += 7;
    }
}

pub(crate) fn leb128(out: &mut Vec<u8>, mut value: usize) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_are_distinct_identifiers() {
        let names: Vec<String> = (0..10_000).map(short_name).collect();
        let distinct: std::collections::HashSet<&String> = names.iter().collect();
        assert_eq!(distinct.len(), names.len());
        assert!(names.iter().all(|n| n.as_bytes()[0].is_ascii_alphabetic()));
        assert_eq!(names[0], "a");
        assert!(names.iter().all(|n| n.len() <= 3));
    }

    #[test]
    fn only_names_written_as_keys_alone_are_renamed() {
        let js = "const o = { __wbg_a_1: function() {}, __wbg_b_2: 1 };\n\
                  o.__wbg_b_2(); const __wbg_c_3 = { __wbg_c_3: 2 };";
        let keys = object_keys(js);
        assert_eq!(keys.get("__wbg_a_1"), Some(&1));
        assert_eq!(keys.get("__wbg_b_2"), None);
        assert_eq!(keys.get("__wbg_c_3"), None);
        let renames = HashMap::from([("__wbg_a_1".to_string(), "a".to_string())]);
        assert_eq!(renamed_keys(js, &renames), js.replacen("__wbg_a_1", "a", 1));
    }
}
