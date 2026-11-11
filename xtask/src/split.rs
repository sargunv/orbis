//! core.wasm in two: what a map's first frames run, and the rest
//! (`core-deferred.wasm`), which loads once the map is idle, or the first
//! time the map calls into it (web/src/builds.ts).
//!
//! The line is [`LIST`], which `pnpm -C web coverage --tour --write` has
//! [`write_list`] rewrite from what its scenes ran: functions none ran are
//! deferred, and those one ran stay (`+`), as do those an earlier list kept,
//! so that later runs only add to them. A function is named by its name
//! without its crates' hashes ([`unhashed`]), at three levels: that name,
//! the name with its module paths dropped ([`short`]), and that with its
//! closures' numbers dropped too ([`loose`]). The list holds, for each
//! group of the build's functions that share a loose name, the loose name
//! if they agree, else the short names that do, else the full names that
//! do, each as its FNV-1a 64 hash ([`key`]). A build finds each function's
//! class under the first of its names the list holds, so moving a module,
//! renaming a crate or adding a closure before another leaves the line
//! where it was; only code whose type or function names changed falls out
//! of it.
//!
//! A function the list doesn't name, new or renamed since it was written,
//! stays in core.wasm unless only deferred functions call it, directly or
//! through others ([`only_called_deferred`]). Such functions are mostly
//! new code, and most new code runs on first frames (two thirds of what a
//! list 53 commits old didn't name ran in the coverage scenes), so
//! deferring them would have first frames fetch core-deferred.wasm
//! synchronously. The build reports how many functions, and how much
//! code, the list doesn't name.
//!
//! binaryen's `wasm-split` moves the deferred functions out. core.wasm then
//! imports a placeholder for each one it calls, from [`NAMESPACE`], and
//! calls them through a table of them; the deferred module, made with
//! core.wasm's exports, writes the real functions into that table. A
//! custom section, [`SECTION`], names the table's export, and its absence
//! tells the loader the module is whole.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use wasmparser::{
    CompositeInnerType, ElementItems, ElementKind, ExternalKind, KnownCustom, Name, Operator,
    Parser, Payload, TypeRef, ValType,
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
    /// Functions the list doesn't name, and their code bytes.
    pub(crate) unnamed: (usize, usize),
    /// Of those, the ones that stayed in core.wasm, and their code bytes.
    pub(crate) unnamed_kept: (usize, usize),
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

fn ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// `name` with its module paths dropped, each path down to its last
/// segment (`<orb_web::map::Map>::draw` → `<Map>::draw`,
/// `core::ptr::drop_glue::<alloc::vec::Vec<u8>>` → `drop_glue::<Vec<u8>>`):
/// what moving a module or renaming a crate leaves as it was.
pub(crate) fn short(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // A path starts where an identifier does, after neither another
        // identifier's bytes nor `::`.
        let starts =
            ident_start(bytes[i]) && (i == 0 || !(ident(bytes[i - 1]) || bytes[i - 1] == b':'));
        if starts {
            // Drops each `segment::` another identifier follows.
            loop {
                let end = i + bytes[i..].iter().take_while(|&&b| ident(b)).count();
                if bytes[end..].starts_with(b"::")
                    && bytes.get(end + 2).copied().is_some_and(ident_start)
                {
                    i = end + 2;
                } else {
                    break;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("whole segments dropped from UTF-8")
}

/// [`short`]`(name)` without its closures' and shims' numbers
/// (`{closure#3}` → `{closure}`) or binaryen's number for a name that
/// repeats (a last `[1]`): what adding a closure before another leaves as
/// it was.
pub(crate) fn loose(name: &str) -> String {
    let short = short(name);
    let bytes = short.as_bytes();
    let digits = |from: usize| {
        from + bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let end = digits(i + 1);
            if end > i + 1 && bytes.get(end) == Some(&b'}') {
                i = end;
                continue;
            }
        }
        if bytes[i] == b'[' && i > 0 && bytes[i - 1] == b'>' {
            let end = digits(i + 1);
            if end > i + 1 && end + 1 == bytes.len() && bytes[end] == b']' {
                break;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("ASCII dropped from UTF-8")
}

fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, b| {
        (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A function's names in the list, finest first: FNV-1a 64 of
/// [`unhashed`]`(name)`, of [`short`] of that and of [`loose`] of that.
pub(crate) fn key(name: &str) -> [u64; 3] {
    let full = unhashed(name);
    [fnv(&full), fnv(&short(&full)), fnv(&loose(&full))]
}

/// [`LIST`]'s classes, by key: true for a function that stays in core.wasm.
#[derive(Default)]
pub(crate) struct List(HashMap<u64, bool>);

impl List {
    /// Whether the function of these [`key`]s stays (`Some(true)`) or is
    /// deferred, under the first of them the list holds.
    fn class(&self, keys: [u64; 3]) -> Option<bool> {
        keys.iter().find_map(|key| self.0.get(key).copied())
    }
}

/// Reads [`LIST`]: hashes in hex, one per line, `+` before those that
/// stay, besides `#` comments. A hash listed both ways stays.
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
        let (kept, hex) = match line.strip_prefix('+') {
            Some(hex) => (true, hex),
            None => (false, line),
        };
        let key = u64::from_str_radix(hex, 16).map_err(|e| format!("{LIST}: {line}: {e}"))?;
        *list.0.entry(key).or_default() |= kept;
    }
    Ok(list)
}

/// The list for functions `(name, kept)`: per loose name their functions
/// agree on, that; else per short name they agree on, that; else per full
/// name, that (see the module docs). Sorted, deferred first.
fn list_lines(functions: &[(&str, bool)]) -> Vec<String> {
    let keyed: Vec<([u64; 3], bool)> = functions
        .iter()
        .map(|&(name, kept)| (key(name), kept))
        .collect();
    let mut out: Vec<(bool, u64)> = Vec::new();
    // Emits, per key at `level` of `members`, the key if they agree, else
    // the next level's.
    fn emit(
        keyed: &[([u64; 3], bool)],
        members: &[usize],
        level: usize,
        out: &mut Vec<(bool, u64)>,
    ) {
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for &i in members {
            groups.entry(keyed[i].0[level]).or_default().push(i);
        }
        for (key, group) in groups {
            let kept = keyed[group[0]].1;
            if group.iter().all(|&i| keyed[i].1 == kept) {
                out.push((kept, key));
            } else if level > 0 {
                emit(keyed, &group, level - 1, out);
            }
        }
    }
    emit(&keyed, &(0..keyed.len()).collect::<Vec<_>>(), 2, &mut out);
    out.sort_unstable();
    out.dedup();
    out.into_iter()
        .map(|(kept, key)| format!("{}{key:016x}", if kept { "+" } else { "" }))
        .collect()
}

/// Rewrites [`LIST`] from `profile`, wasm-split's profile of the module at
/// `names` (`cargo xtask web --names`): what ran in it, or what the list
/// there already kept, stays, and the rest is deferred.
pub(crate) fn write_list(root: &Path, names: &Path, profile: &Path) -> Result<(), String> {
    let read = |path: &Path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
    let wasm = read(names)?;
    let profile = read(profile)?;
    let mut imported = 0u32;
    let mut defined = 0u32;
    let mut named: HashMap<u32, &str> = HashMap::new();
    for payload in Parser::new(0).parse_all(&wasm) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    if matches!(import.map_err(|e| e.to_string())?.ty, TypeRef::Func(_)) {
                        imported += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => defined = reader.count(),
            Payload::CustomSection(section) => {
                if let KnownCustom::Name(reader) = section.as_known() {
                    for name in reader {
                        if let Name::Function(map) = name.map_err(|e| e.to_string())? {
                            for naming in map {
                                let naming = naming.map_err(|e| e.to_string())?;
                                named.insert(naming.index, naming.name);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // An 8-byte hash of the module, then each defined function's first
    // call, 0 for one that never ran.
    let stamps: Vec<bool> = profile
        .get(8..)
        .unwrap_or_default()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|stamp| *stamp != [0; 4])
        .collect();
    if stamps.len() != defined as usize {
        return Err(format!(
            "the profile stamps {} functions; {} defines {defined}: was it taken of an older build?",
            stamps.len(),
            names.display()
        ));
    }
    let previous = read_list(root)?;
    let functions: Vec<(&str, bool)> = (0..defined)
        .filter_map(|i| {
            let name = *named.get(&(imported + i))?;
            let ran = stamps[i as usize];
            let kept = ran || previous.class(key(name)) == Some(true);
            Some((name, kept))
        })
        .collect();
    let lines = list_lines(&functions);
    let header = "\
# What core.wasm leaves to core-deferred.wasm, which loads once the map is
# idle or first calls one of them: the functions no scene of `pnpm -C web
# coverage --tour` has run. Each line is a hash of names (xtask/src/split.rs),
# a `+` before those that stay in core.wasm. Written by `pnpm -C web coverage
# --tour --write`.
";
    let path = root.join(LIST);
    std::fs::write(&path, format!("{header}{}\n", lines.join("\n")))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let kept = functions.iter().filter(|(_, kept)| *kept).count();
    eprintln!(
        "{LIST}: {} functions deferred and {kept} kept, in {} lines",
        functions.len() - kept,
        lines.len()
    );
    Ok(())
}

/// Adds to `moved` every function that only moved functions call, directly
/// or through others. One renamed since the list was written (its old
/// name stale in the list, its new one in neither half of it), or added
/// since, can then only run once core-deferred.wasm has, so it goes there
/// too. What stays is what core.wasm can call before then: the functions
/// JS or a table can reach, those a run saw run (`seen`), and what they
/// call through functions that stay, recursion included. Only core.wasm
/// calls a deferred function through a placeholder, so a v128 in the
/// signature of one this moves doesn't matter.
fn only_called_deferred(functions: &Functions, seen: &[bool], moved: &mut [bool]) {
    let mut stays = vec![false; moved.len()];
    let mut stack = Vec::new();
    for (i, function) in functions.defined.iter().enumerate() {
        if !moved[i] && (function.reachable || seen[i]) {
            stays[i] = true;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        for &callee in &functions.defined[i].callees {
            let callee = callee as usize;
            if !moved[callee] && !stays[callee] {
                stays[callee] = true;
                stack.push(callee);
            }
        }
    }
    for (moved, stays) in moved.iter_mut().zip(stays) {
        *moved |= !stays;
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
    let mut moved = vec![false; functions.defined.len()];
    let mut seen = vec![false; functions.defined.len()];
    let mut unnamed = vec![false; functions.defined.len()];
    for (i, function) in (0u32..).zip(&functions.defined) {
        let at = i as usize;
        let Some(name) = names.get(&(functions.imported + i)) else {
            continue;
        };
        match list.class(key(name)) {
            Some(kept) => {
                seen[at] = kept;
                // A placeholder is a JS function, and JS can't pass a v128.
                moved[at] = !kept && !function.v128;
            }
            // binaryen's own functions (`byn$mgfn-shared$…`, the body
            // functions it merged share) are no list's to name: they go
            // where their callers do.
            None => unnamed[at] = !name.starts_with("byn$"),
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
    let tally = |kept: bool| {
        (0..moved.len())
            .filter(|&i| unnamed[i] && (!kept || !moved[i]))
            .fold((0, 0), |(n, bytes), i| {
                (n + 1, bytes + functions.defined[i].size)
            })
    };
    let result = Split {
        deferred,
        inferred: deferred - listed,
        unnamed: tally(false),
        unnamed_kept: tally(true),
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
    /// The defined functions it calls, by their index among those.
    callees: Vec<u32>,
    /// Its body's bytes.
    size: usize,
}

/// The module's functions, and what they call.
fn functions(wasm: &[u8]) -> Result<Functions, String> {
    let mut types = Vec::new();
    let mut imported = 0u32;
    let mut defined = Vec::new();
    // Functions reached other than by a direct call, in the module's
    // index space.
    let mut reached = Vec::new();
    let mut bodies = 0;
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
                        callees: Vec::new(),
                        size: 0,
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
            Payload::GlobalSection(reader) => {
                for global in reader {
                    let global = global.map_err(|e| e.to_string())?;
                    for op in global.init_expr.get_operators_reader() {
                        if let Operator::RefFunc { function_index } =
                            op.map_err(|e| e.to_string())?
                        {
                            reached.push(function_index);
                        }
                    }
                }
            }
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
                let mut callees = Vec::new();
                for op in body.get_operators_reader().map_err(|e| e.to_string())? {
                    match op.map_err(|e| e.to_string())? {
                        Operator::Call { function_index }
                        | Operator::ReturnCall { function_index } => {
                            // Imports come first in the index space.
                            if let Some(callee) = function_index.checked_sub(imported) {
                                callees.push(callee);
                            }
                        }
                        Operator::RefFunc { function_index } => reached.push(function_index),
                        _ => {}
                    }
                }
                callees.sort_unstable();
                callees.dedup();
                let function: &mut Function = defined
                    .get_mut(bodies)
                    .ok_or("more function bodies than functions")?;
                function.callees = callees;
                let range = body.range();
                function.size = (range.end - range.start) as usize;
                bodies += 1;
            }
            _ => {}
        }
    }
    for index in reached {
        if let Some(function) = index
            .checked_sub(imported)
            .and_then(|i| defined.get_mut(i as usize))
        {
            function.reachable = true;
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
        let function = |callees: &[u32]| Function {
            v128: false,
            reachable: false,
            callees: callees.to_vec(),
            size: 0,
        };
        // 0 is listed, and calls 1, 3, 4, 5 and 6. 1 calls 2, which calls
        // itself, and 8, which calls 9, which calls 8 back. 3 the exported
        // 7 calls too, 4 was seen running and calls 10, 5 is exported, and
        // 6 takes a v128. 7 also calls 11, which calls 12, which calls 11.
        let mut defined = vec![
            function(&[1, 3, 4, 5, 6]),
            function(&[2, 8]),
            function(&[2]),
            function(&[]),
            function(&[10]),
            function(&[]),
            function(&[]),
            function(&[3, 11]),
            function(&[9]),
            function(&[8]),
            function(&[]),
            function(&[12]),
            function(&[11]),
        ];
        defined[5].reachable = true;
        defined[6].v128 = true;
        defined[7].reachable = true;
        let functions = Functions {
            imported: 0,
            defined,
        };
        let mut seen = vec![false; 13];
        seen[4] = true;
        let mut moved = vec![false; 13];
        moved[0] = true;
        only_called_deferred(&functions, &seen, &mut moved);
        let moved: Vec<usize> = (0..13).filter(|&i| moved[i]).collect();
        assert_eq!(moved, [0, 1, 2, 6, 8, 9]);
    }

    #[test]
    fn short_drops_module_paths() {
        assert_eq!(
            short("<orb_web::map::Map>::draw::{closure#0}"),
            "<Map>::draw::{closure#0}"
        );
        assert_eq!(
            short("core::ptr::drop_glue::<alloc::vec::Vec<orb_tile::Layer>>"),
            "drop_glue::<Vec<Layer>>"
        );
        assert_eq!(
            short("<[f32] as core::fmt::Debug>::fmt"),
            "<[f32] as Debug>::fmt"
        );
        assert_eq!(short("orb_host::redacted_url"), "redacted_url");
        assert_eq!(short("x::<[u8; 16usize]>"), "x::<[u8; 16usize]>");
        assert_eq!(short("main"), "main");
        assert_eq!(short("a::b::"), "b::");
    }

    #[test]
    fn loose_drops_closure_numbers() {
        assert_eq!(
            loose("<orb_web::map::Map>::draw::{closure#3}::{shim:vtable#0}"),
            "<Map>::draw::{closure}::{shim:vtable}"
        );
        assert_eq!(loose("core::sort::<u32>[1]"), "sort::<u32>");
        assert_eq!(loose("f::<[u8; 1]>"), "f::<[u8; 1]>");
        assert_eq!(loose("a#1"), "a#1");
    }

    #[test]
    fn keys_hash_each_level() {
        // The first is the full name's: what lists written before the
        // other levels hold.
        assert_eq!(key("")[0], 0xcbf2_9ce4_8422_2325);
        assert_eq!(key("a")[0], 0xaf63_dc4c_8601_ec8c);
        assert_eq!(key("core[0123]::f"), key("core::f"));
        assert_eq!(key("a::f")[1], key("b::f")[1]);
        assert_ne!(key("a::f")[0], key("b::f")[0]);
    }

    #[test]
    fn a_list_survives_moves_and_keeps_what_disagrees_apart() {
        let lines = list_lines(&[
            ("orb_web::map::draw", true),
            ("orb_web::map::draw::{closure#0}", false),
            ("orb_web::map::draw::{closure#1}", false),
            // The same short name, kept in one module and not the other.
            ("<orb_a::Paint>::new", true),
            ("<orb_b::Paint>::new", false),
            // The same full name twice, either way: neither is listed.
            ("dup", true),
            ("dup", false),
        ]);
        let mut list = List::default();
        for line in &lines {
            let (kept, hex) = line
                .strip_prefix('+')
                .map_or((false, line.as_str()), |h| (true, h));
            list.0.insert(u64::from_str_radix(hex, 16).unwrap(), kept);
        }
        let class = |name| list.class(key(name));
        // Moved to another module, and a closure added before the others.
        assert_eq!(class("orb_web::surface::draw"), Some(true));
        assert_eq!(class("orb_web::surface::draw::{closure#2}"), Some(false));
        assert_eq!(class("<orb_a::Paint>::new"), Some(true));
        assert_eq!(class("<orb_b::Paint>::new"), Some(false));
        assert_eq!(class("<orb_c::Paint>::new"), None);
        assert_eq!(class("dup"), None);
        assert_eq!(lines.len(), 4);
    }
}
