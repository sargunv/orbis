//! core.wasm in two: what a map's first frames run, and the rest
//! (`core-deferred.wasm`), which loads once the map is idle, or the first
//! time the map calls into it (web/src/builds.ts).
//!
//! The line is [`LIST`], which `pnpm -C web coverage --tour --write`
//! writes from the functions no scene of its has run: one per line, the
//! FNV-1a 64 hash of the function's name without its crates' hashes
//! ([`key`]), which a build on another machine or after an unrelated
//! change still matches (`+` lines are functions a run saw run, which
//! coverage keeps so that later runs only add to them). A function that
//! isn't listed stays in core.wasm unless only deferred functions call it
//! ([`only_called_deferred`]), so a function renamed or added since the
//! list was written defers when its callers do, and a stale list otherwise
//! only defers less; the build reports how many listed functions it didn't
//! find.
//!
//! binaryen's `wasm-split` moves the listed functions out. core.wasm then
//! imports a placeholder for each one it calls, from [`NAMESPACE`], and
//! calls them through a table of them; the deferred module, made with
//! core.wasm's exports, writes the real functions into that table. A
//! custom section, [`SECTION`], names the table's export, and its absence
//! tells the loader the module is whole.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use wasmparser::{
    CompositeInnerType, ElementItems, ElementKind, ExternalKind, Operator, Parser, Payload,
    TypeRef, ValType,
};

/// The functions to defer (see the module docs).
pub(crate) const LIST: &str = "web/deferred.txt";
/// Where core.wasm imports its placeholders from.
const NAMESPACE: &str = "placeholder.deferred";
/// What wasm-split prefixes to the secondary module's name (`deferred`) to
/// make [`NAMESPACE`].
const PREFIX: &str = "placeholder";
/// The custom section naming the placeholder table's export.
const SECTION: &str = "orbis.deferred";

/// What a split moved.
pub(crate) struct Split {
    /// Functions deferred.
    pub(crate) deferred: usize,
    /// Of those, the ones the list doesn't defer but only deferred
    /// functions call ([`only_called_deferred`]).
    pub(crate) inferred: usize,
    /// Hashes in the list.
    pub(crate) listed: usize,
    /// Hashes in the list that matched no function here.
    pub(crate) missing: usize,
}

/// `name` without its crates' hashes (`orb_style[48b542d477ccbf4d]` →
/// `orb_style`), which change with a crate's build settings.
pub(crate) fn unhashed(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut rest = name;
    while let Some(open) = rest.find('[') {
        let (before, after) = rest.split_at(open);
        out.push_str(before);
        let hash = after[1..]
            .find(']')
            .map(|close| &after[1..=close])
            .filter(|hash| {
                !hash.is_empty()
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            });
        // A crate's hash follows its name; `[f32]` after `<` is a slice.
        let after_ident = before
            .bytes()
            .last()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'_');
        match hash {
            Some(hash) if after_ident => rest = &after[hash.len() + 2..],
            _ => {
                out.push('[');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The list's key for a function named `name`: FNV-1a 64 of
/// [`unhashed`]`(name)`, as web/scripts/coverage.mjs computes it.
pub(crate) fn key(name: &str) -> u64 {
    unhashed(name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |hash, b| {
            (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// [`LIST`]'s keys.
#[derive(Default)]
pub(crate) struct List {
    /// Functions to defer.
    pub(crate) deferred: HashSet<u64>,
    /// Functions a run saw run (`+` lines), which stay in core.wasm.
    pub(crate) seen: HashSet<u64>,
}

/// Reads [`LIST`]: hashes in hex, one per line, `+` before those seen
/// running, besides `#` comments.
pub(crate) fn read_list(root: &Path) -> Result<List, String> {
    let path = root.join(LIST);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(List::default()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut list = List::default();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (set, hex) = match line.strip_prefix('+') {
            Some(hex) => (&mut list.seen, hex),
            None => (&mut list.deferred, line),
        };
        set.insert(u64::from_str_radix(hex, 16).map_err(|e| format!("{LIST}: {line}: {e}"))?);
    }
    Ok(list)
}

/// Adds to `moved` every function that only moved functions call, until
/// there are no more. One renamed since the list was written (its old
/// name stale in the list, its new one in neither half of it), or added
/// since, can then only run once core-deferred.wasm has, so it goes there
/// too. A function JS or a table can reach, one with a v128 in its
/// signature, and one a run saw run (`seen`) stay.
fn only_called_deferred(functions: &Functions, seen: &[bool], moved: &mut [bool]) {
    let mut changed = true;
    while changed {
        changed = false;
        for (i, function) in functions.defined.iter().enumerate() {
            if moved[i]
                || seen[i]
                || function.v128
                || function.reachable
                || function.callers.is_empty()
            {
                continue;
            }
            if function.callers.iter().all(|&c| moved[c as usize]) {
                moved[i] = true;
                changed = true;
            }
        }
    }
}

/// Splits `whole` (core.wasm, without names) into `primary` and
/// `secondary`, deferring the functions `list` keys; `symbols` is the
/// symbol map wasm-opt wrote for `whole`, which names its functions by
/// index. With nothing to defer, `primary` is `whole` and `secondary` an
/// empty module.
pub(crate) fn split(
    whole: &Path,
    symbols: &Path,
    primary: &Path,
    secondary: &Path,
    list: &List,
    features: &[&str],
) -> Result<Split, String> {
    let read = |path: &Path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
    let functions = functions(&read(whole)?)?;
    let symbols = String::from_utf8(read(symbols)?).map_err(|e| e.to_string())?;
    // `index:name`, in the module's index space, imports first.
    let names: HashMap<u32, &str> = symbols
        .lines()
        .filter_map(|line| {
            let (index, name) = line.split_once(':')?;
            Some((index.parse().ok()?, name))
        })
        .collect();
    let mut matched = HashSet::new();
    let mut moved = vec![false; functions.defined.len()];
    let mut seen = vec![false; functions.defined.len()];
    for ((i, function), (moved, seen)) in (0u32..)
        .zip(&functions.defined)
        .zip(moved.iter_mut().zip(&mut seen))
    {
        let Some(name) = names.get(&(functions.imported + i)) else {
            continue;
        };
        let key = key(name);
        *seen = list.seen.contains(&key);
        if list.deferred.contains(&key) {
            matched.insert(key);
            // A placeholder is a JS function, and JS can't pass a v128.
            *moved = !function.v128;
        }
    }
    let listed = moved.iter().filter(|&&m| m).count();
    only_called_deferred(&functions, &seen, &mut moved);
    let mut split = String::new();
    for (i, _) in moved.iter().enumerate().filter(|(_, m)| **m) {
        // binaryen names a function the module leaves unnamed by its
        // index among those it defines.
        split.push_str(&i.to_string());
        split.push('\n');
    }
    let deferred = moved.iter().filter(|&&m| m).count();
    let result = Split {
        deferred,
        inferred: deferred - listed,
        listed: list.deferred.len(),
        missing: list.deferred.len() - matched.len(),
    };
    if deferred == 0 {
        std::fs::copy(whole, primary).map_err(|e| format!("{}: {e}", primary.display()))?;
        std::fs::write(secondary, b"\0asm\x01\0\0\0")
            .map_err(|e| format!("{}: {e}", secondary.display()))?;
        return Ok(result);
    }
    // Kept beside `whole`: the indices deferred, which the symbol map names.
    let split_file = whole.with_extension("deferred");
    std::fs::write(&split_file, split).map_err(|e| format!("{}: {e}", split_file.display()))?;
    crate::run(
        Command::new("wasm-split")
            .args(features)
            .arg(format!("--placeholder-namespace={PREFIX}"))
            .arg(format!("--split-funcs=@{}", split_file.display()))
            .arg(whole)
            .arg("-o1")
            .arg(primary)
            .arg("-o2")
            .arg(secondary),
    )?;
    if let Some((other, _)) = imports(&read(primary)?)?
        .into_iter()
        .find(|(module, _)| module.starts_with(PREFIX) && module != NAMESPACE)
    {
        return Err(format!(
            "wasm-split imported a placeholder from `{other}`, not `{NAMESPACE}`"
        ));
    }
    let table = placeholder_table(&read(secondary)?)?;
    append_section(primary, SECTION, table.as_bytes())?;
    Ok(result)
}

struct Functions {
    imported: u32,
    defined: Vec<Function>,
}

struct Function {
    /// Whether its signature takes or gives a v128.
    v128: bool,
    /// Whether anything but a direct call reaches it: an export, the
    /// start function, a table or a `ref.func`.
    reachable: bool,
    /// The defined functions that call it, by their index among those.
    callers: Vec<u32>,
}

/// The module's functions, and what calls them.
fn functions(wasm: &[u8]) -> Result<Functions, String> {
    let mut types = Vec::new();
    let mut imported = 0u32;
    let mut defined = Vec::new();
    // Functions reached other than by a direct call, and (callee, caller)
    // pairs, in the module's index space and among defined functions.
    let mut reached = Vec::new();
    let mut calls = Vec::new();
    let mut bodies = 0u32;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            Payload::TypeSection(reader) => {
                for group in reader {
                    for ty in group.map_err(|e| e.to_string())?.into_types() {
                        types.push(match &ty.composite_type.inner {
                            CompositeInnerType::Func(f) => f
                                .params()
                                .iter()
                                .chain(f.results())
                                .any(|t| *t == ValType::V128),
                            _ => false,
                        });
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    if matches!(import.map_err(|e| e.to_string())?.ty, TypeRef::Func(_)) {
                        imported += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty in reader {
                    let ty = ty.map_err(|e| e.to_string())?;
                    defined.push(Function {
                        v128: types.get(ty as usize).copied().unwrap_or(false),
                        reachable: false,
                        callers: Vec::new(),
                    });
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.map_err(|e| e.to_string())?;
                    if export.kind == ExternalKind::Func {
                        reached.push(export.index);
                    }
                }
            }
            Payload::StartSection { func, .. } => reached.push(func),
            Payload::ElementSection(reader) => {
                for element in reader {
                    match element.map_err(|e| e.to_string())?.items {
                        ElementItems::Functions(items) => {
                            for item in items {
                                reached.push(item.map_err(|e| e.to_string())?);
                            }
                        }
                        ElementItems::Expressions(_, items) => {
                            for item in items {
                                let item = item.map_err(|e| e.to_string())?;
                                for op in item.get_operators_reader() {
                                    if let Operator::RefFunc { function_index } =
                                        op.map_err(|e| e.to_string())?
                                    {
                                        reached.push(function_index);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let caller = bodies;
                bodies += 1;
                for op in body.get_operators_reader().map_err(|e| e.to_string())? {
                    match op.map_err(|e| e.to_string())? {
                        Operator::Call { function_index }
                        | Operator::ReturnCall { function_index } => {
                            calls.push((function_index, caller));
                        }
                        Operator::RefFunc { function_index } => reached.push(function_index),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let local = |index: u32| index.checked_sub(imported).map(|i| i as usize);
    for index in reached {
        if let Some(function) = local(index).and_then(|i| defined.get_mut(i)) {
            function.reachable = true;
        }
    }
    for (callee, caller) in calls {
        if let Some(function) = local(callee).and_then(|i| defined.get_mut(i))
            && !function.callers.contains(&caller)
        {
            function.callers.push(caller);
        }
    }
    Ok(Functions { imported, defined })
}

/// The module's imports, as (module, name).
fn imports(wasm: &[u8]) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| e.to_string())? {
            for import in reader.into_imports() {
                let import = import.map_err(|e| e.to_string())?;
                out.push((import.module.to_string(), import.name.to_string()));
            }
        }
    }
    Ok(out)
}

/// The export name of the table the deferred module writes its functions
/// into: the one table its element segments fill, which it imports.
fn placeholder_table(secondary: &[u8]) -> Result<String, String> {
    let mut tables = Vec::new();
    let mut filled = HashSet::new();
    for payload in Parser::new(0).parse_all(secondary) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    let import = import.map_err(|e| e.to_string())?;
                    if matches!(import.ty, TypeRef::Table(_)) {
                        tables.push(import.name.to_string());
                    }
                }
            }
            Payload::ElementSection(reader) => {
                for element in reader {
                    if let ElementKind::Active { table_index, .. } =
                        element.map_err(|e| e.to_string())?.kind
                    {
                        filled.insert(table_index.unwrap_or(0));
                    }
                }
            }
            _ => {}
        }
    }
    match filled.into_iter().collect::<Vec<_>>()[..] {
        [table] => tables.get(table as usize).cloned().ok_or_else(|| {
            format!("the deferred module fills table {table}, which it doesn't import")
        }),
        ref other => Err(format!(
            "the deferred module fills {} tables; the loader expects one",
            other.len()
        )),
    }
}

/// Appends a custom section `name` holding `payload` to the module at `path`.
fn append_section(path: &Path, name: &str, payload: &[u8]) -> Result<(), String> {
    let mut wasm = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut body = Vec::new();
    leb128(&mut body, name.len());
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(payload);
    wasm.push(0);
    leb128(&mut wasm, body.len());
    wasm.extend_from_slice(&body);
    std::fs::write(path, wasm).map_err(|e| format!("{}: {e}", path.display()))
}

fn leb128(out: &mut Vec<u8>, mut value: usize) {
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
    fn unhashed_drops_crate_hashes_only() {
        assert_eq!(
            unhashed("<orb_style[48b542d477ccbf4d]::v8::doc::StyleDoc>::apply"),
            "<orb_style::v8::doc::StyleDoc>::apply"
        );
        assert_eq!(
            unhashed("<[f32] as core[e6ec70d1861ee446]::fmt::Debug>::fmt"),
            "<[f32] as core::fmt::Debug>::fmt"
        );
        assert_eq!(unhashed("x::<[u8; 16]>"), "x::<[u8; 16]>");
        assert_eq!(unhashed("a[b"), "a[b");
    }

    #[test]
    fn defers_what_only_deferred_functions_call() {
        let function = |callers: &[u32]| Function {
            v128: false,
            reachable: false,
            callers: callers.to_vec(),
        };
        // 0 is listed; 1 and then 2 only it calls; 3 a kept function calls
        // too; 4 was seen running; 5 is exported; 6 takes a v128.
        let mut defined = vec![
            function(&[]),
            function(&[0]),
            function(&[1]),
            function(&[0, 7]),
            function(&[0]),
            function(&[0]),
            function(&[0]),
            function(&[]),
        ];
        defined[5].reachable = true;
        defined[6].v128 = true;
        let functions = Functions {
            imported: 0,
            defined,
        };
        let mut seen = vec![false; 8];
        seen[4] = true;
        let mut moved = vec![true, false, false, false, false, false, false, false];
        only_called_deferred(&functions, &seen, &mut moved);
        assert_eq!(moved, [true, true, true, false, false, false, false, false]);
    }

    #[test]
    fn key_matches_coverage() {
        // web/scripts/coverage.mjs's `key` gives the same.
        assert_eq!(key(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(key("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(key("core[0123]::f"), key("core::f"));
    }
}
