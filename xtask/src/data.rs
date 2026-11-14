//! Read-only data only one feature module's code reads, moved out of
//! core.wasm's memory into that module, from the linker's relocations
//! (#201).
//!
//! The web builds link with `--emit-relocs`, so the module cargo built
//! names, for every function, the data symbols its code addresses
//! (`R_WASM_MEMORY_ADDR_*`, whatever the addend: LLVM folds an index's
//! offset into the address, `T[i - 1]` addressing the object before `T`,
//! but its relocation still names `T`), and for every pointer in data the
//! symbol it points to (`reloc.DATA`). That is what the code reads before
//! wasm-opt; what it reads after differs only where wasm-opt inlined one
//! function into another (otherwise binaryen 133's `-Oz` moves a constant
//! only into a callee, as an argument every call passes, and the callee
//! then runs only from those calls; it folds no callee's constant result
//! into its callers). So the functions a module's paths name (the
//! split's seeds) that code outside the module calls are kept from being
//! inlined ([`NO_INLINE`]), and every other function's code may sit in
//! any function that calls it, directly or through others: its hosts. A
//! byte of `.rodata` moves to module M when every symbol that covers it is
//! read by code whose hosts are M's functions alone, and anything the
//! analysis can't account for keeps the byte in core.wasm, or fails the
//! build.
//!
//! The bytes leave core.wasm's data as zeros (its memory starts zeroed),
//! and the module carries them in a custom section, [`MEMORY`], which
//! web/src/builds.ts writes into the instance's memory as it links the
//! module, before the module's code can run.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use wasmparser::{
    DataKind, ExternalKind, KnownCustom, Linking, Name, Operator, Parser, Payload, RelocationType,
    SymbolInfo,
};

use crate::glue::{leb128, read_leb128};

/// What the build prefixes the name of each seed code outside its module
/// calls with, before wasm-opt, which inlines none of them
/// (`--no-inline=seed$*`).
pub(crate) const NO_INLINE: &str = "seed$";

/// The custom section a feature module's data comes in, written into the
/// instance's memory as the module comes in ([`apply`]).
pub(crate) const MEMORY: &str = "orbis.memory";

/// The module's function names by index, imports first.
pub(crate) fn names(wasm: &[u8]) -> Result<HashMap<u32, String>, String> {
    let e = |e: wasmparser::BinaryReaderError| e.to_string();
    let mut names = HashMap::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::CustomSection(custom) = payload.map_err(e)?
            && let KnownCustom::Name(reader) = custom.as_known()
        {
            for name in reader {
                if let Name::Function(map) = name.map_err(e)? {
                    for naming in map {
                        let naming = naming.map_err(e)?;
                        names.insert(naming.index, naming.name.to_string());
                    }
                }
            }
        }
    }
    Ok(names)
}

/// Prefixes [`NO_INLINE`] to the names of functions `indices` (imports
/// first) in the module at `path`; returns those names.
pub(crate) fn mark(path: &Path, indices: &HashSet<u32>) -> Result<HashSet<String>, String> {
    let wasm = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut marked = HashSet::new();
    let out = rewrite_custom(&wasm, "name", |payload| {
        let mut out = Vec::new();
        let mut at = 0;
        while at < payload.len() {
            let id = payload[at];
            let (size, start) = read_leb128(payload, at + 1)?;
            let end = start + size;
            let body = payload
                .get(start..end)
                .ok_or("a name subsection runs past")?;
            out.push(id);
            if id != 1 {
                leb128(&mut out, size);
                out.extend_from_slice(body);
                at = end;
                continue;
            }
            // Function names: `count, (index, name)*`.
            let mut names = Vec::new();
            let (count, mut i) = read_leb128(body, 0)?;
            leb128(&mut names, count);
            for _ in 0..count {
                let (index, next) = read_leb128(body, i)?;
                let (len, from) = read_leb128(body, next)?;
                let name =
                    std::str::from_utf8(&body[from..from + len]).map_err(|e| e.to_string())?;
                leb128(&mut names, index);
                if indices.contains(&(index as u32)) {
                    marked.insert(name.to_string());
                    leb128(&mut names, NO_INLINE.len() + len);
                    names.extend_from_slice(NO_INLINE.as_bytes());
                } else {
                    leb128(&mut names, len);
                }
                names.extend_from_slice(name.as_bytes());
                i = from + len;
            }
            leb128(&mut out, names.len());
            out.extend_from_slice(&names);
            at = end;
        }
        Ok(Some(out))
    })?;
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(marked)
}

/// Drops the module at `path`'s linking and relocation sections, which
/// only [`moves`] reads (from cargo's output) and which nothing keeps true
/// after wasm-bindgen.
pub(crate) fn strip_relocations(path: &Path) -> Result<(), String> {
    let mut wasm = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    for name in ["linking", "reloc.CODE", "reloc.DATA"] {
        wasm = rewrite_custom(&wasm, name, |_| Ok(None))?;
    }
    std::fs::write(path, wasm).map_err(|e| format!("{}: {e}", path.display()))
}

/// `wasm` with each custom section `name`'s payload replaced by what
/// `f` makes of it, or dropped for `None`.
fn rewrite_custom(
    wasm: &[u8],
    name: &str,
    mut f: impl FnMut(&[u8]) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<u8>, String> {
    let mut out = wasm[..8].to_vec();
    let mut at = 8;
    while at < wasm.len() {
        let (size, start) = read_leb128(wasm, at + 1)?;
        let end = start + size;
        let section = wasm.get(at..end).ok_or("a section runs past the module")?;
        let payload = if wasm[at] == 0 {
            let (len, from) = read_leb128(wasm, start)?;
            (wasm.get(from..from + len) == Some(name.as_bytes())).then_some(from + len)
        } else {
            None
        };
        match payload {
            Some(payload) => {
                if let Some(new) = f(&wasm[payload..end])? {
                    let mut body = Vec::new();
                    leb128(&mut body, name.len());
                    body.extend_from_slice(name.as_bytes());
                    body.extend_from_slice(&new);
                    out.push(0);
                    leb128(&mut out, body.len());
                    out.extend_from_slice(&body);
                }
            }
            None => out.extend_from_slice(section),
        }
        at = end;
    }
    Ok(out)
}

/// Where a function's code may run once wasm-opt is done: bit `m` for
/// feature module `m`, [`CORE`] for core.wasm.
type Hosts = u32;
const CORE: Hosts = 1 << 31;

/// The most bytes between two runs of a module's that its section
/// carries as one run.
const MERGE_GAP: u32 = 16;

/// A run of bytes moving to a module.
pub(crate) struct Run {
    pub(crate) module: usize,
    pub(crate) address: u32,
    pub(crate) bytes: Vec<u8>,
}

/// The module cargo linked, as [`moves`] reads it.
struct Linked<'a> {
    imported: u32,
    /// Each defined function's body, by its offset in the code section.
    bodies: Vec<std::ops::Range<u32>>,
    /// Each defined function's name, demangled as wasm-bindgen does.
    names: Vec<Option<String>>,
    /// Functions reached other than by a direct call (exports, the start
    /// function, tables): kept by wasm-opt, by whatever name.
    roots: Vec<bool>,
    /// Data segments: their address if active, where their bytes start in
    /// the data section, their bytes, and their name.
    segments: Vec<(Option<u32>, u32, &'a [u8], &'a str)>,
    symbols: Vec<SymbolInfo<'a>>,
    code: Vec<wasmparser::RelocationEntry>,
    data: Vec<wasmparser::RelocationEntry>,
    /// Addresses global initializers hold.
    globals: Vec<u32>,
}

fn read(wasm: &[u8]) -> Result<Linked<'_>, String> {
    let e = |e: wasmparser::BinaryReaderError| e.to_string();
    let mut linked = Linked {
        imported: 0,
        bodies: Vec::new(),
        names: Vec::new(),
        roots: Vec::new(),
        segments: Vec::new(),
        symbols: Vec::new(),
        code: Vec::new(),
        data: Vec::new(),
        globals: Vec::new(),
    };
    let mut code_start = 0;
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut segment_names = Vec::new();
    let mut roots = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.map_err(e)? {
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    if matches!(import.map_err(e)?.ty, wasmparser::TypeRef::Func(_)) {
                        linked.imported += 1;
                    }
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.map_err(e)?;
                    if export.kind == ExternalKind::Func {
                        roots.push(export.index);
                    }
                }
            }
            Payload::StartSection { func, .. } => roots.push(func),
            Payload::ElementSection(reader) => {
                for element in reader {
                    match element.map_err(e)?.items {
                        wasmparser::ElementItems::Functions(items) => {
                            for item in items {
                                roots.push(item.map_err(e)?);
                            }
                        }
                        wasmparser::ElementItems::Expressions(..) => {
                            return Err("an element segment of expressions".into());
                        }
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                for global in reader {
                    for op in global.map_err(e)?.init_expr.get_operators_reader() {
                        if let Operator::I32Const { value } = op.map_err(e)? {
                            linked.globals.push(value as u32);
                        }
                    }
                }
            }
            Payload::CodeSectionStart {
                unchecked_range, ..
            } => code_start = unchecked_range.start,
            Payload::CodeSectionEntry(body) => {
                let range = body.range();
                linked
                    .bodies
                    .push((range.start - code_start) as u32..(range.end - code_start) as u32);
            }
            Payload::DataSection(reader) => {
                let start = reader.range().start as usize;
                for data in reader {
                    let data = data.map_err(e)?;
                    let address = match data.kind {
                        DataKind::Active { offset_expr, .. } => {
                            match offset_expr.get_operators_reader().read().map_err(e)? {
                                Operator::I32Const { value } => Some(value as u32),
                                _ => None,
                            }
                        }
                        DataKind::Passive => None,
                    };
                    let at = (data.range.end as usize - data.data.len()) - start;
                    linked.segments.push((address, at as u32, data.data, ""));
                }
            }
            Payload::CustomSection(custom) => match custom.as_known() {
                KnownCustom::Name(reader) => {
                    for name in reader {
                        if let Name::Function(map) = name.map_err(e)? {
                            for naming in map {
                                let naming = naming.map_err(e)?;
                                names.insert(naming.index, demangled(naming.name));
                            }
                        }
                    }
                }
                KnownCustom::Linking(reader) => {
                    for subsection in reader.subsections() {
                        match subsection.map_err(e)? {
                            Linking::SymbolTable(map) => {
                                for symbol in map {
                                    linked.symbols.push(symbol.map_err(e)?);
                                }
                            }
                            Linking::SegmentInfo(map) => {
                                for segment in map {
                                    segment_names.push(segment.map_err(e)?.name);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                KnownCustom::Reloc(reader) => {
                    let entries = reader.entries().into_iter().collect::<Result<Vec<_>, _>>();
                    match custom.name() {
                        "reloc.CODE" => linked.code = entries.map_err(e)?,
                        "reloc.DATA" => linked.data = entries.map_err(e)?,
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    if linked.symbols.is_empty() || linked.code.is_empty() {
        return Err("the module has no relocations: link it with --emit-relocs".into());
    }
    if segment_names.len() != linked.segments.len() {
        return Err("the linking section names other segments than the module has".into());
    }
    for (segment, name) in linked.segments.iter_mut().zip(segment_names) {
        segment.3 = name;
    }
    let defined = linked.bodies.len();
    linked.names = (0..defined)
        .map(|i| names.remove(&(linked.imported + i as u32)))
        .collect();
    linked.roots = vec![false; defined];
    for index in roots {
        if let Some(i) = index.checked_sub(linked.imported) {
            linked.roots[i as usize] = true;
        }
    }
    Ok(linked)
}

/// `name` as wasm-bindgen names the function, and so binaryen.
fn demangled(name: &str) -> String {
    match rustc_demangle::try_demangle(name) {
        Ok(name) => name.to_string(),
        Err(_) => name.to_string(),
    }
}

fn is_memory_address(ty: RelocationType) -> bool {
    use RelocationType::*;
    matches!(
        ty,
        MemoryAddrLeb
            | MemoryAddrSleb
            | MemoryAddrI32
            | MemoryAddrRelSleb
            | MemoryAddrLeb64
            | MemoryAddrSleb64
            | MemoryAddrI64
            | MemoryAddrRelSleb64
            | MemoryAddrTlsSleb
            | MemoryAddrLocrelI32
            | MemoryAddrTlsSleb64
    )
}

fn is_function(ty: RelocationType) -> bool {
    use RelocationType::*;
    matches!(
        ty,
        FunctionIndexLeb
            | FunctionIndexI32
            | TableIndexSleb
            | TableIndexI32
            | TableIndexRelSleb
            | TableIndexSleb64
            | TableIndexI64
            | TableIndexRelSleb64
    )
}

fn is_table(ty: RelocationType) -> bool {
    !matches!(
        ty,
        RelocationType::FunctionIndexLeb | RelocationType::FunctionIndexI32
    )
}

/// What of `linked`'s `.rodata` only one module's code reads, by
/// [`MODULES`](crate::split::MODULES)' order: `status` gives where wasm-opt's
/// output keeps a function of a name (`Some(Some(m))` module `m`'s,
/// `Some(None)` core.wasm's, `None` where it keeps none), and `seeds` the
/// functions it inlined nowhere ([`mark`]).
pub(crate) fn moves(
    linked: &[u8],
    seeds: &HashSet<String>,
    status: impl Fn(&str) -> Option<Option<usize>>,
) -> Result<Vec<Run>, String> {
    let linked = read(linked)?;
    let n = linked.bodies.len();
    let imported = linked.imported;
    let defined = |index: u32| index.checked_sub(imported).map(|i| i as usize);
    // Data symbols: where they lie.
    let mut spans: Vec<Option<(u32, u32)>> = vec![None; linked.symbols.len()];
    for (s, symbol) in linked.symbols.iter().enumerate() {
        if let SymbolInfo::Data {
            symbol: Some(data), ..
        } = symbol
        {
            let segment =
                (linked.segments.get(data.index as usize)).ok_or("a data symbol in no segment")?;
            if let Some(address) = segment.0 {
                spans[s] = Some((address + data.offset, data.size));
            }
        }
    }
    let function = |s: u32| match linked.symbols.get(s as usize) {
        Some(&SymbolInfo::Func { index, .. }) => defined(index),
        _ => None,
    };
    // What each function's code addresses and calls.
    let mut reads: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut calls: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = linked.roots.clone();
    for entry in &linked.code {
        let f = linked
            .bodies
            .partition_point(|body| body.start <= entry.offset)
            .checked_sub(1)
            .filter(|&f| linked.bodies[f].contains(&entry.offset))
            .ok_or("a relocation outside every function")?;
        if is_memory_address(entry.ty) {
            match linked.symbols.get(entry.index as usize) {
                Some(SymbolInfo::Data { .. }) => reads[f].push(entry.index),
                _ => {
                    return Err(format!(
                        "a memory address of symbol {}, no data",
                        entry.index
                    ));
                }
            }
        } else if is_function(entry.ty)
            && let Some(callee) = function(entry.index)
        {
            calls[f].push(callee);
            // An address taken is a table's: a call through it, made
            // direct, may inline it anywhere, but it runs whole too.
            if is_table(entry.ty) {
                roots[callee] = true;
            }
        }
    }
    // Pointers in data, by the symbols that hold them.
    let mut by_address: Vec<(u32, u32, u32)> = (spans.iter().enumerate())
        .filter_map(|(s, span)| span.map(|(a, size)| (a, a + size, s as u32)))
        .filter(|&(a, end, _)| a < end)
        .collect();
    by_address.sort_unstable();
    let longest = by_address
        .iter()
        .map(|&(a, end, _)| end - a)
        .max()
        .unwrap_or(0);
    let covering = |address: u32, len: u32| {
        let from = by_address.partition_point(|&(a, ..)| a + longest <= address);
        let to = by_address.partition_point(|&(a, ..)| a < address + len);
        by_address[from..to]
            .iter()
            .filter(move |&&(a, end, _)| a < address + len && address < end)
            .map(|&(.., s)| s)
    };
    let mut points: Vec<Vec<u32>> = vec![Vec::new(); linked.symbols.len()];
    // Symbols nothing the analysis follows may reach: pinned in core.
    let mut pinned: HashSet<u32> = HashSet::new();
    for entry in &linked.data {
        let segment = (linked.segments.iter())
            .find(|s| (s.1..s.1 + s.2.len() as u32).contains(&entry.offset))
            .ok_or("a data relocation outside every segment")?;
        if is_memory_address(entry.ty) {
            let Some(address) = segment.0.map(|a| a + entry.offset - segment.1) else {
                pinned.insert(entry.index);
                continue;
            };
            let mut held = false;
            for s in covering(address, 4) {
                points[s as usize].push(entry.index);
                held = true;
            }
            if !held {
                pinned.insert(entry.index);
            }
        } else if is_function(entry.ty)
            && let Some(callee) = function(entry.index)
        {
            roots[callee] = true;
        }
    }
    for &address in &linked.globals {
        pinned.extend(covering(address, 1));
    }
    // Where each function's code may run: its own body where wasm-opt
    // keeps it, and inlined, wherever its callers' may, unless it's a
    // seed, which wasm-opt inlines nowhere.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for name in linked.names.iter().flatten() {
        *counts.entry(name).or_default() += 1;
    }
    let mut hosts: Vec<Hosts> = (0..n)
        .map(|f| {
            let Some(name) = linked.names[f].as_deref() else {
                return CORE;
            };
            if counts[name] > 1 {
                return CORE;
            }
            match status(name) {
                Some(Some(m)) => 1 << m,
                Some(None) => CORE,
                // No function of the name: a root runs whole in core.wasm
                // for all we know, by another name, and anything else
                // runs only inlined.
                None if roots[f] => CORE,
                None => 0,
            }
        })
        .collect();
    let stops: Vec<bool> = (linked.names.iter())
        .map(|name| name.as_deref().is_some_and(|n| seeds.contains(n)))
        .collect();
    let mut stack: Vec<usize> = (0..n).collect();
    while let Some(f) = stack.pop() {
        for &callee in &calls[f] {
            if !stops[callee] && hosts[callee] | hosts[f] != hosts[callee] {
                hosts[callee] |= hosts[f];
                stack.push(callee);
            }
        }
    }
    // Where each data symbol is read: its readers' hosts, and those of
    // whatever points to it.
    let mut read_by: Vec<Hosts> = vec![0; linked.symbols.len()];
    for (f, symbols) in reads.iter().enumerate() {
        for &s in symbols {
            read_by[s as usize] |= hosts[f];
        }
    }
    for &s in &pinned {
        read_by[s as usize] |= CORE;
    }
    let mut stack: Vec<usize> = (0..linked.symbols.len()).collect();
    while let Some(s) = stack.pop() {
        for &t in &points[s] {
            let t = t as usize;
            if read_by[t] | read_by[s] != read_by[t] {
                read_by[t] |= read_by[s];
                stack.push(t);
            }
        }
    }
    // Byte by byte in `.rodata`: each symbol covering it read by one
    // module's code alone.
    let mut runs: Vec<Run> = Vec::new();
    for &(address, _, bytes, name) in &linked.segments {
        let Some(address) = address.filter(|_| name == ".rodata") else {
            continue;
        };
        let mut owner: Vec<Option<Hosts>> = vec![None; bytes.len()];
        for (s, span) in spans.iter().enumerate() {
            let Some((a, size)) = *span else { continue };
            if a < address || a + size > address + bytes.len() as u32 {
                continue;
            }
            // Read by nothing the analysis saw: whatever reads it is core's.
            let by = if read_by[s] == 0 { CORE } else { read_by[s] };
            for o in &mut owner[(a - address) as usize..(a - address + size) as usize] {
                *o = Some(o.unwrap_or(0) | by);
            }
        }
        let mut i = 0;
        while i < bytes.len() {
            let module = owner[i].filter(|&h| h.count_ones() == 1 && h != CORE);
            let Some(module) = module else {
                i += 1;
                continue;
            };
            let start = i;
            while i < bytes.len() && owner[i] == Some(module) {
                i += 1;
            }
            runs.push(Run {
                module: module.trailing_zeros() as usize,
                address: address + start as u32,
                bytes: bytes[start..i].to_vec(),
            });
        }
    }
    Ok(runs)
}

/// Moves `runs` out of `primary`'s data into their modules (`paths`, by
/// [`MODULES`](crate::split::MODULES)' order): zeros in its segments, the
/// bytes in each module's [`MEMORY`] section. Fails where `primary` holds
/// other bytes than the linker laid out there.
pub(crate) fn apply(
    primary: &Path,
    paths: &[std::path::PathBuf],
    runs: &[Run],
) -> Result<(), String> {
    let mut wasm = std::fs::read(primary).map_err(|e| format!("{}: {e}", primary.display()))?;
    let mut segments: Vec<(u32, usize, usize)> = Vec::new();
    for payload in Parser::new(0).parse_all(&wasm) {
        if let Payload::DataSection(reader) = payload.map_err(|e| e.to_string())? {
            for data in reader {
                let data = data.map_err(|e| e.to_string())?;
                if let DataKind::Active { offset_expr, .. } = data.kind
                    && let Operator::I32Const { value } = offset_expr
                        .get_operators_reader()
                        .read()
                        .map_err(|e| e.to_string())?
                {
                    let at = data.range.end as usize - data.data.len();
                    segments.push((value as u32, at, data.data.len()));
                }
            }
        }
    }
    // Where `primary` holds the byte at an address; `None` for one
    // wasm-opt left out of its segments, a zero.
    let held = |address: u32| {
        (segments.iter())
            .find(|&&(a, _, len)| (a..a + len as u32).contains(&address))
            .map(|&(a, at, _)| at + (address - a) as usize)
    };
    // Each module's runs, those a few bytes apart as one: the bytes
    // between are what memory holds there anyway (core.wasm's, or another
    // module's that it writes the same), and cost less than a run's
    // header. Runs are `gap: leb128, length: leb128, bytes`, each gap
    // from the end of the run before (from zero for the first).
    let mut sections: Vec<Vec<u8>> = vec![Vec::new(); paths.len()];
    for (m, section) in sections.iter_mut().enumerate() {
        let mut merged: Vec<(u32, Vec<u8>)> = Vec::new();
        for run in runs.iter().filter(|r| r.module == m) {
            match merged.last_mut() {
                Some((address, bytes))
                    if run.address - (*address + bytes.len() as u32) <= MERGE_GAP =>
                {
                    for gap in *address + bytes.len() as u32..run.address {
                        bytes.push(held(gap).map_or(0, |at| wasm[at]));
                    }
                    bytes.extend_from_slice(&run.bytes);
                }
                _ => merged.push((run.address, run.bytes.clone())),
            }
        }
        let mut end = 0;
        for (address, bytes) in merged {
            leb128(section, (address - end) as usize);
            leb128(section, bytes.len());
            section.extend_from_slice(&bytes);
            end = address + bytes.len() as u32;
        }
    }
    for run in runs {
        for (i, &byte) in run.bytes.iter().enumerate() {
            let address = run.address + i as u32;
            match held(address) {
                Some(at) if wasm[at] == byte => wasm[at] = 0,
                None if byte == 0 => {}
                _ => {
                    return Err(format!(
                        "{}: other data at {address:#x} than the linker laid out",
                        primary.display()
                    ));
                }
            }
        }
    }
    std::fs::write(primary, wasm).map_err(|e| format!("{}: {e}", primary.display()))?;
    for (path, section) in paths.iter().zip(sections) {
        if !section.is_empty() {
            if !path.exists() {
                return Err(format!(
                    "{}: data for a module with no code",
                    path.display()
                ));
            }
            crate::split::append_section(path, MEMORY, &section)?;
        }
    }
    Ok(())
}
