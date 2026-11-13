//! core.wasm by feature: the code of plain vector maps (MVT on Web
//! Mercator) stays in it, and each feature a style declares goes to a
//! module of its own, [`MODULES`], which the map fetches once its style
//! declares that feature, beside TileJSON, and instantiates against
//! core.wasm's memory and table in each of its threads (web/src/builds.ts,
//! orb-runtime's `modules.rs`).
//!
//! A function goes to a module when its name, without its crates' hashes
//! ([`unhashed`]), names one of the module's paths ([`Module::paths`]), and
//! so do the functions only that module's code calls, directly or through
//! others ([`assign`]). The rest stays in core.wasm: what JS, a table or
//! core.wasm's other code reaches, and what more than one module calls.
//!
//! The API's module starts at the exports JS calls for queries, edits,
//! the style's document, images, snapshots and offline regions, which JS
//! makes wait for the module (web/src/engine.ts), and takes what only they
//! call: code nothing else reaches, by construction rather than by a
//! profile of what scenes ran.
//!
//! binaryen's `wasm-split --multi-split` moves them out. core.wasm then
//! imports a placeholder for each function a module holds that core.wasm
//! calls, from `placeholder.<module>`, and calls it through a table; the
//! module, made with core.wasm's exports, writes the real functions into
//! that table. The runtime never calls into a module it hasn't loaded, so
//! a placeholder throws: a call is a bug in a module's paths or in the
//! runtime's gate, and the e2e `split` check finds both. Every module of a
//! build carries [`PAIR`], a hash of its modules: one from another build
//! (a stale copy, a cache) would write other functions into the table, so
//! the loader takes only its own.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use wasmparser::{ElementItems, ExternalKind, Operator, Parser, Payload, TypeRef};

/// A feature's module.
pub(crate) struct Module {
    /// Its name: `<build>-<name>.wasm`, and orb-runtime's `Module`.
    pub(crate) name: &'static str,
    /// The paths whose functions it holds: a crate or a module of one,
    /// matched in a function's name.
    paths: &'static [&'static str],
    /// Functions of those paths that maps without the feature run too.
    except: &'static [&'static str],
}

/// The feature modules, as orb-runtime's `Module` names them: what each
/// holds, and when the map loads it.
pub(crate) const MODULES: &[Module] = &[
    // A source's tiles in MLT, as its `encoding`, TileJSON or archive says.
    Module {
        name: "mlt",
        paths: &["mlt_core::", "fastpfor::", "fsst::", "orb_tile::mlt::"],
        except: &[],
    },
    // A `pmtiles://` source.
    Module {
        name: "archive",
        paths: &["orb_source::pmtiles::", "orb_source::gzip::"],
        except: &[],
    },
    // An `mbtiles://` source.
    Module {
        name: "sqlite",
        paths: &[
            "orb_sqlite::",
            "orb_source::mbtiles::",
            "orb_runtime::native_db::",
        ],
        except: &[],
    },
    // A `raster-dem` source: terrain, hillshade and color-relief.
    Module {
        name: "dem",
        paths: &[
            "orb_tile::dem::",
            "orb_runtime::terrain::",
            "orb_runtime::elevation::",
            "orb_render::terrain::",
        ],
        except: &[],
    },
    // A projection but Web Mercator, or a source in another CRS.
    Module {
        name: "globe",
        paths: &[
            "orb_crs::warp::",
            "orb_crs::sphere::",
            "orb_crs::epsg::",
            "orb_crs::family::azimuthal::",
            "orb_crs::family::conic::",
            "orb_crs::family::datum::",
            "orb_crs::family::ellipsoid::",
            "orb_crs::family::oblique::",
            "orb_crs::family::pseudo::",
            "orb_crs::family::tm::",
            "orb_runtime::globe::",
        ],
        // Every map has its warps, empty on Web Mercator, and a sphere frame.
        except: &[
            "orb_runtime::globe::GlobeWarps>::new",
            "orb_runtime::globe::GlobeWarps>::end_frame",
            "orb_crs::sphere::frame",
        ],
    },
    // A `geojson` source.
    Module {
        name: "geojson",
        paths: &["orb_tile::geojson::", "orb_runtime::geojson::"],
        // A source's options are read with the style, before it waits.
        except: &["orb_tile::geojson::Options"],
    },
    // Queries, edits, the style's document, images, snapshots and offline
    // regions: on idle, or before the first call that needs it. JS calls
    // these exports only once the module is in (web/src/engine.ts's
    // `API_CALLS`, regions.ts's `loadCore`).
    Module {
        name: "api",
        paths: &[
            "orbismap_addImage",
            "orbismap_updateImage",
            "orbismap_removeImage",
            "orbismap_hasImage",
            "orbismap_edit",
            "orbismap_getLayer",
            "orbismap_getSource",
            "orbismap_getStyle",
            "orbismap_layerIds",
            "orbismap_sourceIds",
            "orbismap_hitTest",
            "orbismap_queryCluster",
            "orbismap_queryRendered",
            "orbismap_querySource",
            "orbismap_setStyleUrl",
            "orbismap_beginSnapshot",
            "orbismap_snapshotFrame",
            "orbismap_endSnapshot",
            "orbismap_cancelSnapshot",
            "orbismap_readNextFrame",
            "downloadRegion",
            "deleteRegion",
            "listRegions",
            // What they run, which a table or a callback's closure reaches
            // (edits not: a style URL loaded over another applies as one).
            "orb_runtime::map::Map>::query_",
            "orb_runtime::query::Query",
            "orb_runtime::query::Scene",
            "orb_runtime::query::ScreenHit",
            "orb_runtime::query::Picking",
            "orb_runtime::offline::download",
            "orb_runtime::offline::Job",
            "orb_runtime::offline::Then",
            "orb_runtime::offline::import",
            "orb_runtime::offline::Import",
            "orb_runtime::offline::font_stacks",
        ],
        except: &[],
    },
];

/// Where core.wasm imports a module's placeholders from: this, then the
/// module's name.
const PLACEHOLDERS: &str = "placeholder";
/// The custom section every module of a build carries, core.wasm too:
/// the FNV-1a 64 hash of its feature modules' bytes, in hex.
const PAIR: &str = "orbis.pair";

/// Functions this small don't start a module: binaryen folds identical
/// bodies into one function of any of their names, and a small one is
/// likeliest to be core.wasm's too. Only one module's code calling it
/// still moves it.
const MIN_SEED_BYTES: usize = 24;

/// What the split moved: per module, its functions and their code bytes.
pub(crate) struct Split {
    pub(crate) modules: Vec<(&'static str, usize, usize)>,
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

/// The module whose paths `name` names, if one does: a function that
/// such a path defines, as a free function or a method of a type there
/// (`orb_tile::mlt::decode`, `<orb_tile::mlt::Layer>::new`, `<mlt_core::X
/// as Y>::f`). Code that only names such a type, as std's generics
/// instantiated for it do, runs wherever the type's values go, which may
/// be where the feature is off (dropping an empty `Option` of it): it
/// goes to the module only if only the module's code calls it.
fn seed(name: &str) -> Option<usize> {
    let name = unhashed(name);
    let own = name.strip_prefix('<').unwrap_or(&name);
    MODULES.iter().position(|m| {
        m.paths.iter().any(|p| own.starts_with(p)) && !m.except.iter().any(|p| own.starts_with(p))
    })
}

/// Each function's module, from `seeds` (a module's paths name it): also
/// each function only one module's functions reach, through functions of
/// no module, and that nothing else reaches: not JS, a table nor
/// core.wasm's other code.
fn assign(functions: &Functions, seeds: &[Option<usize>]) -> Vec<Option<usize>> {
    let n = functions.defined.len();
    // What reaches each function: core.wasm (`usize::MAX`), or modules.
    const CORE: usize = usize::MAX;
    let mut reached: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for (i, function) in functions.defined.iter().enumerate() {
        match seeds[i] {
            Some(module) => stack.push((i, module)),
            None if function.reachable => stack.push((i, CORE)),
            None => {}
        }
    }
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    while let Some((i, from)) = stack.pop() {
        if !seen.insert((i, from)) {
            continue;
        }
        for &callee in &functions.defined[i].callees {
            let callee = callee as usize;
            if seeds[callee].is_none() {
                reached[callee].insert(from);
                stack.push((callee, from));
            }
        }
    }
    (0..n)
        .map(|i| {
            if seeds[i].is_some() || functions.defined[i].reachable {
                return seeds[i];
            }
            match reached[i].iter().collect::<Vec<_>>()[..] {
                [&only] if only != CORE => Some(only),
                _ => None,
            }
        })
        .collect()
}

/// Splits `whole` (core.wasm, without names) into `primary` and a module
/// per feature, `<prefix><name>.wasm`; `symbols` is the symbol map
/// wasm-opt wrote for `whole`, which names its functions by index.
/// Returns where each module went, by [`MODULES`]' order.
pub(crate) fn split(
    whole: &Path,
    symbols: &Path,
    primary: &Path,
    prefix: &str,
    features: &[&str],
) -> Result<(Split, Vec<PathBuf>), String> {
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
    let seeds: Vec<Option<usize>> = (0u32..)
        .zip(&functions.defined)
        .map(|(i, function)| {
            let name = names.get(&(functions.imported + i))?;
            (function.size >= MIN_SEED_BYTES || function.exported)
                .then(|| seed(name))
                .flatten()
        })
        .collect();
    let modules = assign(&functions, &seeds);
    let mut manifest = String::new();
    let mut split = Split {
        modules: Vec::new(),
    };
    for (m, module) in MODULES.iter().enumerate() {
        let held: Vec<usize> = (0..modules.len())
            .filter(|&i| modules[i] == Some(m))
            .collect();
        let bytes = held.iter().map(|&i| functions.defined[i].size).sum();
        split.modules.push((module.name, held.len(), bytes));
        if held.is_empty() {
            continue;
        }
        if !manifest.is_empty() {
            manifest.push('\n');
        }
        manifest.push_str(module.name);
        manifest.push('\n');
        for i in held {
            // binaryen names a function the module leaves unnamed by its
            // index among those it defines.
            manifest.push_str(&i.to_string());
            manifest.push('\n');
        }
    }
    let dir = primary.parent().expect("a file in a directory");
    let paths: Vec<PathBuf> = MODULES
        .iter()
        .map(|m| dir.join(format!("{prefix}{}.wasm", m.name)))
        .collect();
    for path in &paths {
        let _ = std::fs::remove_file(path);
    }
    let manifest_file = whole.with_extension("modules");
    std::fs::write(&manifest_file, manifest)
        .map_err(|e| format!("{}: {e}", manifest_file.display()))?;
    crate::run(
        Command::new("wasm-split")
            .arg("--multi-split")
            .args(features)
            .arg(format!("--placeholder-namespace-prefix={PLACEHOLDERS}"))
            .arg("--placeholdermap")
            .arg(format!("--manifest={}", manifest_file.display()))
            .arg(format!("--out-prefix={}/{prefix}", dir.display()))
            .arg(whole)
            .arg("-o")
            .arg(primary),
    )?;
    // Each placeholder's function by name, for a call into a module not
    // loaded (whose error names its placeholder).
    let map = primary.with_extension("wasm.placeholders");
    let named: String = std::fs::read_to_string(&map)
        .map_err(|e| format!("{}: {e}", map.display()))?
        .lines()
        .filter_map(|line| {
            let (slot, function) = line.split_once(':')?;
            let i: usize = function.trim().parse().ok()?;
            let module = modules
                .get(i)
                .copied()
                .flatten()
                .map_or("?", |m| MODULES[m].name);
            let name = names
                .get(&(functions.imported + i as u32))
                .map_or_else(|| function.to_string(), |n| unhashed(n));
            Some(format!("{slot}\t{module}\t{name}\n"))
        })
        .collect();
    let _ = std::fs::remove_file(&map);
    let placeholders = whole.with_extension("placeholders");
    std::fs::write(&placeholders, named).map_err(|e| format!("{}: {e}", placeholders.display()))?;
    // A build's modules know each other by the hash of theirs.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for path in paths.iter().filter(|p| p.exists()) {
        for b in read(path)? {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    let pair = format!("{hash:016x}");
    for path in std::iter::once(primary).chain(paths.iter().map(PathBuf::as_path)) {
        if path.exists() {
            append_section(path, PAIR, pair.as_bytes())?;
        }
    }
    // A module with nothing in it is an empty one, for the loader.
    for path in &paths {
        if !path.exists() {
            std::fs::write(path, b"\0asm\x01\0\0\0")
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let known: HashSet<String> = MODULES
        .iter()
        .map(|m| format!("{PLACEHOLDERS}.{}", m.name))
        .collect();
    if let Some(other) = imports(&read(primary)?)?
        .into_iter()
        .find(|module| module.starts_with(PLACEHOLDERS) && !known.contains(module))
    {
        return Err(format!(
            "wasm-split imported placeholders from `{other}`, which names no module"
        ));
    }
    Ok((split, paths))
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

struct Functions {
    imported: u32,
    defined: Vec<Function>,
}

struct Function {
    /// Whether anything but a direct call reaches it: an export, the
    /// start function, a table or a `ref.func`.
    reachable: bool,
    /// Whether it's an export, which JS calls.
    exported: bool,
    /// The defined functions it calls, by their index among those.
    callees: Vec<u32>,
    /// Its body's bytes.
    size: usize,
}

/// The module's functions, and what they call.
fn functions(wasm: &[u8]) -> Result<Functions, String> {
    let mut imported = 0u32;
    let mut defined = Vec::new();
    // Functions reached other than by a direct call, in the module's
    // index space.
    let mut reached = Vec::new();
    let mut exports = Vec::new();
    let mut bodies = 0;
    let ref_funcs = |ops: wasmparser::OperatorsReader<'_>, reached: &mut Vec<u32>| {
        for op in ops {
            if let Operator::RefFunc { function_index } = op.map_err(|e| e.to_string())? {
                reached.push(function_index);
            }
        }
        Ok::<_, String>(())
    };
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for import in reader.into_imports() {
                    if matches!(import.map_err(|e| e.to_string())?.ty, TypeRef::Func(_)) {
                        imported += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty in reader {
                    ty.map_err(|e| e.to_string())?;
                    defined.push(Function {
                        reachable: false,
                        exported: false,
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
                        exports.push(export.index);
                    }
                }
            }
            Payload::StartSection { func, .. } => reached.push(func),
            Payload::GlobalSection(reader) => {
                for global in reader {
                    let global = global.map_err(|e| e.to_string())?;
                    ref_funcs(global.init_expr.get_operators_reader(), &mut reached)?;
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
                                ref_funcs(item.get_operators_reader(), &mut reached)?;
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
    for index in exports {
        if let Some(function) = index
            .checked_sub(imported)
            .and_then(|i| defined.get_mut(i as usize))
        {
            function.exported = true;
        }
    }
    Ok(Functions { imported, defined })
}

/// The modules a module imports from.
fn imports(wasm: &[u8]) -> Result<HashSet<String>, String> {
    let mut out = HashSet::new();
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::ImportSection(reader) = payload.map_err(|e| e.to_string())? {
            for import in reader.into_imports() {
                out.insert(import.map_err(|e| e.to_string())?.module.to_string());
            }
        }
    }
    Ok(out)
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
    fn paths_name_their_modules() {
        let at = |name| seed(name).map(|m| MODULES[m].name);
        assert_eq!(at("<mlt_core[0a1b]::decoder::Parser>::parse"), Some("mlt"));
        assert_eq!(at("<orb_tile::mlt::Layer as Drop>::drop"), Some("mlt"));
        assert_eq!(at("core::ptr::drop_in_place::<orb_tile::mlt::Layer>"), None);
        assert_eq!(at("<orb_source::pmtiles::Reader>::lookup"), Some("archive"));
        assert_eq!(at("orb_tile::mvt::decode"), None);
    }

    #[test]
    fn a_module_takes_what_only_its_code_calls() {
        let function = |callees: &[u32], reachable| Function {
            reachable,
            exported: false,
            callees: callees.to_vec(),
            size: 100,
        };
        // 0 is exported and calls 1 (module 0's) and 2. 1 calls 3 and 4; 3
        // calls 5. 4 is called by 6 (module 1's) too, and 2 calls 7, which
        // 1 calls as well.
        let functions = Functions {
            imported: 0,
            defined: vec![
                function(&[1, 2], true),
                function(&[3, 4, 7], false),
                function(&[7], false),
                function(&[5], false),
                function(&[], false),
                function(&[], false),
                function(&[4], false),
                function(&[], false),
            ],
        };
        let seeds = [None, Some(0), None, None, None, None, Some(1), None];
        assert_eq!(
            assign(&functions, &seeds),
            [None, Some(0), None, Some(0), None, Some(0), Some(1), None]
        );
    }
}
