//! Short names for the imports wasm-bindgen gives a module from its JS
//! glue (`./<name>_bg.js`). wasm-bindgen names each after the binding it
//! calls plus 16 hex digits of a hash (`__wbg_get_d7176cfb76d71114`), and
//! the module and the glue both carry every name: in core.wasm 500 of them
//! were 4 KiB brotli, hashes that don't compress. The glue reads each only
//! as a key of the object it hands the module, so renaming both, before
//! wasm-opt and the split, changes nothing else. The same goes for the
//! exports the glue reads only as `wasm.<name>` ([`shorten_exports`]).

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
    std::fs::write(&wasm_path, renamed(&wasm, IMPORTS, &entries, section_end)?)
        .map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    std::fs::write(&js_path, renamed_keys(&js, &renames))
        .map_err(|e| format!("{}: {e}", js_path.display()))
}

/// Renames the module's exports in `<name>_bg.wasm` and `<name>.js` in
/// `out` that the glue reads only as `wasm.<export>`: its exported
/// functions' names (`orbismap_addImage`) and its closures' invokers',
/// which spell out their types with crates' hashes, 6 KiB in core.wasm
/// and as much in its glue.
pub(crate) fn shorten_exports(out: &Path, name: &str) -> Result<(), String> {
    let wasm_path = out.join(format!("{name}_bg.wasm"));
    let js_path = out.join(format!("{name}.js"));
    let wasm = std::fs::read(&wasm_path).map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    let js =
        std::fs::read_to_string(&js_path).map_err(|e| format!("{}: {e}", js_path.display()))?;
    let reads = export_reads(&js);
    let mut exports: Vec<(usize, &str)> = Vec::new();
    let mut section_end = 0;
    for payload in Parser::new(0).parse_all(&wasm) {
        let Payload::ExportSection(reader) = payload.map_err(|e| e.to_string())? else {
            continue;
        };
        section_end = usize::try_from(reader.range().end).map_err(|e| e.to_string())?;
        for export in reader.into_iter_with_offsets() {
            let (offset, export) = export.map_err(|e| e.to_string())?;
            exports.push((
                usize::try_from(offset).map_err(|e| e.to_string())?,
                export.name,
            ));
        }
    }
    // A short name no export kept as it is has.
    let kept: std::collections::HashSet<&str> = (exports.iter())
        .map(|&(_, name)| name)
        .filter(|name| !reads.contains(name))
        .collect();
    let mut shorts = (0..).map(short_name).filter(|s| !kept.contains(s.as_str()));
    let mut renames: HashMap<String, String> = HashMap::new();
    let entries: Vec<(usize, Option<String>)> = (exports.iter())
        .map(|&(offset, name)| {
            let short = reads.contains(name).then(|| {
                let short = shorts.next().expect("endless");
                renames.insert(name.to_string(), short.clone());
                short
            });
            (offset, short)
        })
        .collect();
    if renames.is_empty() {
        return Err(format!(
            "{}: no export the glue reads as `wasm.<name>` alone: wasm-bindgen's output changed, so update glue.rs",
            wasm_path.display()
        ));
    }
    std::fs::write(&wasm_path, renamed(&wasm, EXPORTS, &entries, section_end)?)
        .map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    std::fs::write(&js_path, renamed_keys(&js, &renames))
        .map_err(|e| format!("{}: {e}", js_path.display()))
}

/// The identifiers of `js` that every appearance reads off the module's
/// exports, as `wasm.<name>`.
fn export_reads(js: &str) -> std::collections::HashSet<&str> {
    let mut reads = std::collections::HashSet::new();
    let mut elsewhere: Vec<&str> = Vec::new();
    for (at, ident) in identifiers(js) {
        let before = &js[..at];
        let read = before.strip_suffix("wasm.").is_some_and(|before| {
            !before
                .bytes()
                .next_back()
                .is_some_and(|b| b.is_ascii_alphanumeric() || b"_$.".contains(&b))
        });
        if read {
            reads.insert(ident);
        } else {
            elsewhere.push(ident);
        }
    }
    for ident in elsewhere {
        reads.remove(ident);
    }
    reads
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

/// A section whose entries [`renamed`] renames: its id, and how many
/// strings come before an entry's name.
struct Named {
    id: u8,
    strings_before: usize,
}
/// The import section: an import's module's name, then its own.
const IMPORTS: Named = Named {
    id: 2,
    strings_before: 1,
};
/// The export section: an export's name first.
const EXPORTS: Named = Named {
    id: 7,
    strings_before: 0,
};

/// `wasm` with `section`'s entries (by offset, up to `section_end`) given
/// the names in `entries`.
fn renamed(
    wasm: &[u8],
    section: Named,
    entries: &[(usize, Option<String>)],
    section_end: usize,
) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    leb128(&mut body, entries.len());
    for (i, (offset, short)) in entries.iter().enumerate() {
        let end = entries.get(i + 1).map_or(section_end, |(next, _)| *next);
        let entry = &wasm[*offset..end];
        let mut name_start = 0;
        for _ in 0..section.strings_before {
            name_start = string_end(entry, name_start)?;
        }
        let name_end = string_end(entry, name_start)?;
        body.extend_from_slice(&entry[..name_start]);
        match short {
            Some(short) => {
                leb128(&mut body, short.len());
                body.extend_from_slice(short.as_bytes());
            }
            None => body.extend_from_slice(&entry[name_start..name_end]),
        }
        body.extend_from_slice(&entry[name_end..]);
    }
    // The sections in order, this one's body replaced.
    let mut out = wasm[..8].to_vec();
    let mut at = 8;
    while at < wasm.len() {
        let id = wasm[at];
        let (size, start) = read_leb128(wasm, at + 1)?;
        let end = start + size;
        out.push(id);
        if id == section.id {
            leb128(&mut out, body.len());
            out.extend_from_slice(&body);
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
        .ok_or_else(|| "a name runs past its entry".to_string())
}

pub(crate) fn read_leb128(bytes: &[u8], mut at: usize) -> Result<(usize, usize), String> {
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

    #[test]
    fn only_exports_read_off_the_module_alone_are_renamed() {
        let js = "wasm.orbismap_new(); wasm.memory.buffer; const memory = 1;\n\
                  wasm.__wbindgen_free(); x.wasm.other(); wasm.both(); both();";
        let mut reads: Vec<&str> = export_reads(js).into_iter().collect();
        reads.sort_unstable();
        assert_eq!(reads, ["__wbindgen_free", "orbismap_new"]);
    }
}
