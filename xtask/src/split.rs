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
//! the style's document, images, snapshots, offline regions, fitting
//! bounds, host-fed heights, video controls and GPU timings, which JS
//! makes wait for the module (web/src/engine.ts), and takes what only they
//! call, workers' reads for queries and GeoJSON diffs included: code
//! nothing else reaches, by construction rather than by a profile of what
//! scenes ran.
//!
//! Read-only data only one module's code reaches goes with that module
//! too ([`data`]): its strings, tables and shaders' sources.
//!
//! A name is one function's alone: neither LLVM nor binaryen folds two
//! functions with one body into one of their names here (web.rs), which
//! would give a module a function core.wasm calls too; [`split`] folds
//! them within each module after.
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
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;

use wasmparser::{DataKind, ElementItems, ExternalKind, Operator, Parser, Payload, TypeRef};

mod data;

/// A feature's module.
pub(crate) struct Module {
    /// Its name: `<build>-<name>.wasm`, and orb-runtime's `Module`.
    pub(crate) name: &'static str,
    /// The paths whose functions it holds: a crate or a module of one,
    /// matched in a function's name.
    paths: &'static [&'static str],
    /// Functions of those paths that maps without the feature run too.
    except: &'static [&'static str],
    /// The other modules its code calls into, which a map that loads it
    /// loads too (orb-runtime's `modules::of_style`); the split fails on a
    /// call into one not listed.
    needs: &'static [&'static str],
}

/// What a map runs before its gate asks for any module: a style read,
/// parsed or edited, and the modules it needs worked out from it
/// (orb-runtime's `modules.rs`). What these reach stays out of the feature
/// modules whatever path names it: the maths of a projection a style
/// names, the GeoJSON features a style's document copies.
const BEFORE_GATE: &[&str] = &[
    "orb_style::v8::read",
    "orb_style::v8::parse",
    "orb_style::v8::doc::StyleDoc",
    "orb_source::endpoint::resolve_style_urls",
    "orb_runtime::modules::",
];

/// Edits of a style's document, which only the API makes, to a style
/// that may still wait for its modules: what they reach stays out of the
/// feature modules too, but may be the API's.
const EDITS: &[&str] = &["orb_style::v8::doc::StyleDoc>::apply"];

/// What a function runs before: a map's gate ([`BEFORE_GATE`]), or the
/// feature modules' but the API's ([`EDITS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Before {
    FeaturesButApi,
    Gate,
}

/// The feature modules, as orb-runtime's `Module` names them: what each
/// holds, and when the map loads it.
pub(crate) const MODULES: &[Module] = &[
    // A source's tiles in MLT, as its `encoding`, TileJSON or archive says.
    Module {
        name: "mlt",
        paths: &["mlt_core::", "fastpfor::", "fsst::", "orb_tile::mlt::"],
        except: &[],
        needs: &[],
    },
    // A `pmtiles://` source.
    Module {
        name: "archive",
        paths: &["orb_source::pmtiles::", "orb_source::gzip::"],
        except: &[],
        needs: &[],
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
        // MBTiles' tiles are gzipped.
        needs: &["archive"],
    },
    // A `raster-dem` source: terrain, hillshade and color-relief.
    Module {
        name: "dem",
        paths: &[
            "orb_tile::dem::",
            "orb_runtime::terrain::",
            "orb_runtime::elevation::",
            "orb_render::terrain::",
            // An app's heights for a raster-dem source it feeds, once
            // given (`Map::set_host_elevations`, the API's, checks the
            // source first).
            "orb_runtime::host_heights::",
            "orb_runtime::map::Map>::load_heights",
            "orb_runtime::map::Map>::stage_heights",
            "orb_runtime::map::Map>::show_batch",
            "orb_runtime::map::Map>::heights_layers",
            // The drape, once the style's terrain resolved.
            "orb_runtime::frame::DrapePlan",
            "orb_render::Renderer>::terrain",
            "orb_render::Renderer>::set_terrain_tiles",
            "orb_render::Renderer>::request_terrain",
            "orb_render::Renderer>::reserve_drape",
            "orb_render::Renderer>::carry_drape",
            "orb_render::Renderer>::set_drape_paint",
            "orb_render::Renderer>::publish_dems",
            "orb_render::Renderer>::pad_dems",
            "orb_camera::View>::unproject_terrain",
            // The DEM sources' tiles, and their pages' padding and slopes
            // for hillshades, color reliefs and the terrain.
            "orb_source::TilePyramid<orb_runtime::terrain::DemTile>>::",
            "orb_render::image::prepare::",
            "orb_render::image::Images>::make_prepare",
            "orb_render::image::Images>::prepare_dem",
            "orb_render::image::Images>::make_slopes",
            "orb_render::image::Images>::make_tables",
            "orb_render::tiles::dem_neighbours",
            // The camera over a terrain (`&dyn Elevation`).
            "orb_camera::View>::clearance",
            "orb_camera::View>::near_footprint",
            "orb_camera::View>::near_points",
            "orb_camera::View>::over",
            "orb_camera::CameraState>::settle_on",
            "orb_camera::CameraState>::recenter_on",
            "orb_camera::CameraState>::keep_above",
        ],
        // Every map has an empty store of host heights, and turns the
        // terrain off.
        except: &["orb_runtime::host_heights::HostHeights>::"],
        needs: &[],
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
            // Covers of the globe, of a projection's low-zoom plane and of
            // sources in another CRS, and the globe's labels.
            "orb_camera::View>::cover_globe",
            "orb_camera::View>::cover_warped",
            "orb_camera::View>::cover_low",
            "orb_camera::GlobeFrustum",
            "orb_symbol::placement::view::TileGlobe",
            // Vector tiles bent to a warp or the sphere (`Bend`).
            "orb_tile::TileFrame>::subdivided",
            "orb_tile::subdivide::",
            "orb_layout::fill::subdivide",
            "orb_layout::fill::cap",
            "orb_layout::fill::Bend",
            "orb_layout::line::subdivide",
            // Flights over the sphere.
            "orb_camera::animate::over_sphere",
        ],
        // Every map has its warps, empty on Web Mercator, and a sphere frame.
        except: &[
            "orb_runtime::globe::GlobeWarps>::new",
            "orb_runtime::globe::GlobeWarps>::end_frame",
            "orb_crs::sphere::frame",
        ],
        needs: &[],
    },
    // A `geojson` source.
    Module {
        name: "geojson",
        paths: &["orb_tile::geojson::", "orb_runtime::geojson::"],
        except: &[],
        needs: &[],
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
            "orbismap_cameraForBounds",
            "orbismap_hostElevations",
            "orbismap_reloadHostTiles",
            "orbismap_retryFailedTiles",
            "orbismap_controlVideo",
            "downloadRegion",
            "deleteRegion",
            "listRegions",
            "orb_web::regions::_::__wasm_bindgen_generated_downloadRegion",
            "orb_web::regions::_::__wasm_bindgen_generated_deleteRegion",
            "orb_web::regions::_::__wasm_bindgen_generated_listRegions",
            "orb_web::regions::download_region",
            "orb_web::regions::delete_region",
            "orb_web::regions::list_regions",
            // What they run, which a table or a callback's closure reaches
            // (edits not: a style URL loaded over another applies as one).
            "orb_runtime::map::Map>::query_",
            "orb_runtime::map::Map>::read_features",
            "orb_runtime::queries::",
            "orb_web::map::features_promise",
            // A query's features read off its tiles, in a worker: JS
            // loads the module in every thread before it asks.
            "orb_web::dispatch::WorkerJobs as orb_runtime::jobs::Jobs>::features",
            "orb_web::dispatch::WorkerJobs as orb_runtime::jobs::Jobs>::cluster",
            "orb_runtime::job_host::JobHost>::features",
            "orb_runtime::job_host::JobHost>::cluster",
            "orb_runtime::worker::TileWorker>::features",
            "orb_runtime::worker::TileWorker>::cluster",
            "orb_runtime::features::FeatureRequest",
            "orb_runtime::features::TileFeature",
            "orb_runtime::features::body",
            "orb_runtime::features::evaluated",
            "orb_runtime::features::parse_filter",
            "orb_runtime::features::parse_state",
            // Images an app added, which a new device takes back only
            // when there are some.
            "orb_runtime::map::Map>::add_image",
            // GPU timings, for measurements.
            "orbismap_gpuTimings",
            "orb_web::map::OrbisMap>::gpu_timings",
            "orb_web::canvas::Canvas>::take_timings",
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
            // A style loaded over another applies as an edit, which only
            // setStyleUrl (and orbismap_edit) start.
            "orb_runtime::map::Map>::diff_document",
            "orb_runtime::map::Map>::edit",
            // A GeoJSON source's data changed by an edit's diffs, which
            // workers apply too.
            "orb_style::v8::geojson::apply_diffs",
            "orb_style::v8::geojson::Diff",
        ],
        except: &[],
        needs: &[],
    },
];

/// Where core.wasm imports a module's placeholders from: this, then the
/// module's name.
const PLACEHOLDERS: &str = "placeholder";
/// The custom section every module of a build carries, core.wasm too:
/// the FNV-1a 64 hash of its feature modules' bytes, in hex.
const PAIR: &str = "orbis.pair";

/// What the split moved: per module, its functions, their code bytes and
/// its data bytes.
pub(crate) struct Split {
    pub(crate) modules: Vec<(&'static str, usize, usize, usize)>,
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
/// as Y>::f`), with its closures (`<<f::{closure#0} as FnOnce>::call_once`).
/// So does code generic over one of those closures
/// (`future_to_promise::<f::{closure#0}>`, its drop glue): only a value of
/// the closure runs it, and only `f` makes one. That a name is one
/// function's alone takes a build that folds no two functions with one
/// body into one of their names, as LLVM and binaryen would (web.rs).
/// Code that only names a module's
/// type, as std's generics instantiated for it do, runs wherever the
/// type's values go, which may be where the feature is off (dropping an
/// empty `Option` of it): it goes to the module only if only the module's
/// code calls it.
fn seed(name: &str) -> Option<usize> {
    // A body binaryen shares among similar functions, of any module's.
    if name.starts_with("byn$") {
        return None;
    }
    let name = unhashed(name);
    let named = |own: &str| {
        MODULES.iter().position(|m| {
            m.paths.iter().any(|p| names(own, p)) && !m.except.iter().any(|p| names(own, p))
        })
    };
    if let Some(m) = named(name.trim_start_matches('<')) {
        return Some(m);
    }
    // The closures named in its generic arguments: the path before each
    // `::{closure#`, back to the `<`, `(`, `,` or space that opens it.
    let mut found = None;
    for (at, _) in name.match_indices("::{closure#") {
        let Some(owner) = owner(&name[..at]) else {
            continue;
        };
        match (found, named(owner.trim_start_matches('<'))) {
            (_, None) => {}
            (None, Some(m)) => found = Some(m),
            (Some(a), Some(b)) if a == b => {}
            // Closures of two modules': neither's alone.
            (Some(_), Some(_)) => return None,
        }
    }
    found
}

/// The path that ends `text`, back to the `<`, `(`, `,` or space that
/// opens it outside its own angle brackets (`<a::B<C>>::f` of
/// `x::<<a::B<C>>::f`); `None` when nothing does: the name's own path.
fn owner(text: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (i, c) in text.char_indices().rev() {
        match c {
            '>' => depth += 1,
            '<' if depth > 0 => depth -= 1,
            '<' | '(' | ',' | ' ' if depth == 0 => return Some(&text[i + 1..]),
            _ => {}
        }
    }
    None
}

/// Whether `path` names function `own`: it starts `own` and ends where
/// a name does (`Renderer>::terrain` names `terrain`'s closures, not
/// `terrain_on`), unless it ends in `::` or `_`, a prefix of names.
fn names(own: &str, path: &str) -> bool {
    own.strip_prefix(path).is_some_and(|rest| {
        path.ends_with([':', '_'])
            || !rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// What `name` runs before, if it's code a map runs before its gate
/// ([`BEFORE_GATE`]) or an edit ([`EDITS`]).
fn before_gate(name: &str) -> Option<Before> {
    let name = unhashed(name);
    let own = name.strip_prefix('<').unwrap_or(&name);
    if EDITS.iter().any(|p| names(own, p)) {
        Some(Before::FeaturesButApi)
    } else {
        BEFORE_GATE
            .iter()
            .any(|p| own.starts_with(p))
            .then_some(Before::Gate)
    }
}

/// `seeds` without the functions code before the gate runs: `roots`
/// ([`before_gate`]) and what they call, directly or through others;
/// what only edits run keeps the API's seeds.
fn ungated(
    functions: &Functions,
    seeds: &[Option<usize>],
    roots: &[Option<Before>],
) -> Vec<Option<usize>> {
    let mut before = roots.to_vec();
    // The gate's reach first, so that an edit's stops where it starts.
    for gate in [Before::Gate, Before::FeaturesButApi] {
        let mut stack: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] == Some(gate))
            .collect();
        while let Some(i) = stack.pop() {
            for &callee in &functions.defined[i].callees {
                let callee = callee as usize;
                if before[callee].is_none_or(|b| b < gate) {
                    before[callee] = Some(gate);
                    stack.push(callee);
                }
            }
        }
    }
    seeds
        .iter()
        .zip(before)
        .map(|(&seed, before)| {
            seed.filter(|&m| match before {
                None => true,
                Some(Before::FeaturesButApi) => MODULES[m].name == "api",
                Some(Before::Gate) => false,
            })
        })
        .collect()
}

/// The calls from one feature module into another that it doesn't
/// declare it needs ([`Module::needs`]): a map that loaded only the first
/// would call a placeholder. The API's module is exempt: it calls into a
/// feature's module only for a style that declares the feature.
fn undeclared(functions: &Functions, modules: &[Option<usize>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, function) in functions.defined.iter().enumerate() {
        let Some(from) = modules[i].filter(|&m| MODULES[m].name != "api") else {
            continue;
        };
        for &callee in &function.callees {
            let callee = callee as usize;
            if let Some(to) = modules[callee]
                && to != from
                && !MODULES[from].needs.contains(&MODULES[to].name)
            {
                out.push((i, callee));
            }
        }
    }
    out
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
/// wasm-opt wrote for `whole`, which names its functions by index, and
/// `map` the linker's map of the module it was made from, which bounds
/// its data objects ([`data`]). Returns where each module went, by
/// [`MODULES`]' order.
pub(crate) fn split(
    whole: &Path,
    symbols: &Path,
    map: &Path,
    primary: &Path,
    prefix: &str,
    features: &[&str],
) -> Result<(Split, Vec<PathBuf>), String> {
    let read = |path: &Path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
    let mut wasm = read(whole)?;
    let functions = functions(&wasm)?;
    let symbols = String::from_utf8(read(symbols)?).map_err(|e| e.to_string())?;
    // `index:name`, in the module's index space, imports first.
    let names: HashMap<u32, &str> = symbols
        .lines()
        .filter_map(|line| {
            let (index, name) = line.split_once(':')?;
            Some((index.parse().ok()?, name))
        })
        .collect();
    let name = |i: usize| names.get(&(functions.imported + i as u32)).copied();
    let seeds: Vec<Option<usize>> = (0..functions.defined.len())
        .map(|i| seed(name(i)?))
        .collect();
    let roots: Vec<Option<Before>> = (0..functions.defined.len())
        .map(|i| before_gate(name(i)?))
        .collect();
    let modules = assign(&functions, &ungated(&functions, &seeds, &roots));
    let undeclared = undeclared(&functions, &modules);
    if !undeclared.is_empty() {
        let named = |i: usize| {
            let module = MODULES[modules[i].expect("in a module")].name;
            let name = name(i).map_or_else(|| i.to_string(), unhashed);
            format!("{name} ({module})")
        };
        let calls: Vec<String> = (undeclared.iter().take(8))
            .map(|&(from, to)| format!("  {} calls {}", named(from), named(to)))
            .collect();
        return Err(format!(
            "{} calls from a feature module into another it doesn't list in its `needs`:\n{}",
            undeclared.len(),
            calls.join("\n"),
        ));
    }
    // The data only one module's code reaches goes with it.
    let linked = std::fs::read_to_string(map).map_err(|e| format!("{}: {e}", map.display()))?;
    let objects = data::objects(&linked)?;
    let pointers: Vec<(&[u32], Option<usize>)> = (functions.defined.iter())
        .zip(&modules)
        .map(|(f, &m)| (f.pointers.as_slice(), m))
        .collect();
    let homes = data::assign(
        &objects,
        &wasm,
        &functions.segments,
        &pointers,
        &functions.roots,
    );
    let held_data = |m: usize| -> Vec<data::Object> {
        (objects.iter().zip(&homes))
            .filter(|&(_, &home)| home == Some(m))
            .map(|(&o, _)| o)
            .collect()
    };
    let module_data: Vec<Vec<(u32, Vec<u8>)>> = (0..MODULES.len())
        .map(|m| data::bytes(&wasm, &functions.segments, &held_data(m)))
        .collect();
    let moved: Vec<data::Object> = (objects.iter().zip(&homes))
        .filter(|&(_, home)| home.is_some())
        .map(|(&o, _)| o)
        .collect();
    data::zero(&mut wasm, &functions.segments, &moved);
    // What wasm-split splits: `whole` without that data.
    let zeroed = whole.with_extension("zeroed.wasm");
    std::fs::write(&zeroed, &wasm).map_err(|e| format!("{}: {e}", zeroed.display()))?;
    let mut manifest = String::new();
    let mut split = Split {
        modules: Vec::new(),
    };
    for (m, module) in MODULES.iter().enumerate() {
        let held: Vec<usize> = (0..modules.len())
            .filter(|&i| modules[i] == Some(m))
            .collect();
        let bytes = held.iter().map(|&i| functions.defined[i].size).sum();
        let data = module_data[m].iter().map(|(_, b)| b.len()).sum();
        split.modules.push((module.name, held.len(), bytes, data));
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
            .arg(&zeroed)
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
    let _ = std::fs::remove_file(&zeroed);
    let placeholders = whole.with_extension("placeholders");
    std::fs::write(&placeholders, named).map_err(|e| format!("{}: {e}", placeholders.display()))?;
    // Each module writes its data into its memory as it's instantiated.
    for (path, segments) in paths.iter().zip(&module_data) {
        if path.exists() && !segments.is_empty() {
            let module = data::with_data(&read(path)?, segments)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            std::fs::write(path, module).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    // Functions with one body fold into one, within each module alone
    // (`whole` was optimized without), and core.wasm drops the data it
    // zeroed.
    for path in std::iter::once(primary).chain(paths.iter().map(PathBuf::as_path)) {
        if path.exists() {
            let mut opt = Command::new("wasm-opt");
            opt.arg("--duplicate-function-elimination");
            if path == primary {
                opt.args(["--zero-filled-memory", "--memory-packing"]);
            }
            crate::run(opt.args(features).arg(path).arg("-o").arg(path))?;
        }
    }
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
    /// The active data segments: where each starts in memory, and its
    /// bytes' range in the module.
    segments: Vec<(u32, Range<usize>)>,
    /// The constants of the globals' initial values that may be addresses.
    roots: Vec<u32>,
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
    /// Its constants that may be addresses of data: any at or past 1 MiB,
    /// where Rust lays data out, offsets of loads and stores included.
    pointers: Vec<u32>,
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
    let mut segments = Vec::new();
    let mut roots = Vec::new();
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
                        pointers: Vec::new(),
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
                    for op in global.init_expr.get_operators_reader() {
                        pointers(&op.map_err(|e| e.to_string())?, &mut roots);
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
                                ref_funcs(item.get_operators_reader(), &mut reached)?;
                            }
                        }
                    }
                }
            }
            Payload::DataSection(reader) => {
                for segment in reader {
                    let segment = segment.map_err(|e| e.to_string())?;
                    let DataKind::Active { offset_expr, .. } = segment.kind else {
                        continue;
                    };
                    let mut ops = offset_expr.get_operators_reader();
                    let Operator::I32Const { value } = ops.read().map_err(|e| e.to_string())?
                    else {
                        return Err("a data segment's offset isn't a constant".into());
                    };
                    let at = segment.data.as_ptr() as usize - wasm.as_ptr() as usize;
                    segments.push((value as u32, at..at + segment.data.len()));
                }
            }
            Payload::CodeSectionEntry(body) => {
                let mut callees = Vec::new();
                let mut addresses = Vec::new();
                for op in body.get_operators_reader().map_err(|e| e.to_string())? {
                    let op = op.map_err(|e| e.to_string())?;
                    pointers(&op, &mut addresses);
                    match op {
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
                addresses.sort_unstable();
                addresses.dedup();
                function.pointers = addresses;
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
    Ok(Functions {
        imported,
        defined,
        segments,
        roots,
    })
}

/// Pushes what in `op` may be an address of data onto `out`: its
/// constants at or past 1 MiB, where Rust lays data out (each half of an
/// `i64`'s, each lane of a vector's), and its memory offset.
fn pointers(op: &Operator<'_>, out: &mut Vec<u32>) {
    const DATA: u32 = 1 << 20;
    let mut push = |value: u32| {
        if value >= DATA {
            out.push(value);
        }
    };
    match op {
        Operator::I32Const { value } => push(*value as u32),
        Operator::I64Const { value } => {
            push(*value as u32);
            push((*value as u64 >> 32) as u32);
        }
        Operator::V128Const { value } => {
            for lane in value.bytes().as_chunks::<4>().0 {
                push(u32::from_le_bytes(*lane));
            }
        }
        _ => {
            if let Some(offset) = memory_offset(op) {
                push(u32::try_from(offset).unwrap_or(u32::MAX));
            }
        }
    }
}

/// The memory offset of a load or store.
fn memory_offset(op: &Operator<'_>) -> Option<u64> {
    use Operator::*;
    match op {
        I32Load { memarg }
        | I64Load { memarg }
        | F32Load { memarg }
        | F64Load { memarg }
        | I32Load8S { memarg }
        | I32Load8U { memarg }
        | I32Load16S { memarg }
        | I32Load16U { memarg }
        | I64Load8S { memarg }
        | I64Load8U { memarg }
        | I64Load16S { memarg }
        | I64Load16U { memarg }
        | I64Load32S { memarg }
        | I64Load32U { memarg }
        | I32Store { memarg }
        | I64Store { memarg }
        | F32Store { memarg }
        | F64Store { memarg }
        | I32Store8 { memarg }
        | I32Store16 { memarg }
        | I64Store8 { memarg }
        | I64Store16 { memarg }
        | I64Store32 { memarg }
        | V128Load { memarg }
        | V128Load8x8S { memarg }
        | V128Load8x8U { memarg }
        | V128Load16x4S { memarg }
        | V128Load16x4U { memarg }
        | V128Load32x2S { memarg }
        | V128Load32x2U { memarg }
        | V128Load8Splat { memarg }
        | V128Load16Splat { memarg }
        | V128Load32Splat { memarg }
        | V128Load64Splat { memarg }
        | V128Load32Zero { memarg }
        | V128Load64Zero { memarg }
        | V128Store { memarg }
        | V128Load8Lane { memarg, .. }
        | V128Load16Lane { memarg, .. }
        | V128Load32Lane { memarg, .. }
        | V128Load64Lane { memarg, .. }
        | V128Store8Lane { memarg, .. }
        | V128Store16Lane { memarg, .. }
        | V128Store32Lane { memarg, .. }
        | V128Store64Lane { memarg, .. } => Some(memarg.offset),
        _ => None,
    }
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
        // A path ends where a name does.
        assert_eq!(at("<orb_render::Renderer>::pad_dems"), Some("dem"));
        assert_eq!(at("<orb_render::Renderer>::pad_dems_later"), None);
        // A closure goes with the function that makes it, and so does
        // code generic over it, but not code generic over a module's type.
        assert_eq!(
            at("<<orb_tile::mlt::decode::{closure#0} as FnOnce<()>>::call_once"),
            Some("mlt")
        );
        assert_eq!(
            at(
                "js_sys::future_to_promise::<<orb_tile::mlt::Layer>::new::{closure#1}>::{closure#0}"
            ),
            Some("mlt")
        );
        assert_eq!(
            at(
                "<x::Iter<orb_tile::mlt::Layer> as Iterator>::fold::<usize, <usize as Sum>::sum<y::Map<orb_tile::mlt::Layer>>::{closure#0}>"
            ),
            None
        );
        // A body binaryen shares among similar functions is any module's.
        assert_eq!(
            at("byn$mgfn-shared$<orb_tile::mlt::decode::{closure#0}>"),
            None
        );
    }

    #[test]
    fn a_module_takes_what_only_its_code_calls() {
        let function = |callees: &[u32], reachable| Function {
            reachable,
            exported: false,
            callees: callees.to_vec(),
            size: 100,
            pointers: Vec::new(),
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
            segments: Vec::new(),
            roots: Vec::new(),
        };
        let seeds = [None, Some(0), None, None, None, None, Some(1), None];
        assert_eq!(
            assign(&functions, &seeds),
            [None, Some(0), None, Some(0), None, Some(0), Some(1), None]
        );
        // Code before the gate, 2 here, keeps what it reaches in core:
        // 7, and 1 when it calls it, seed or not, with what 1 calls.
        let mut before = vec![None; 8];
        before[2] = Some(Before::Gate);
        let mut functions = functions;
        functions.defined[2].callees.push(1);
        let modules = assign(&functions, &ungated(&functions, &seeds, &before));
        assert_eq!(modules, [None, None, None, None, None, None, Some(1), None]);
        // An edit, 2 now, keeps the API's seeds it reaches, but no other
        // module's.
        let at = |name| MODULES.iter().position(|m| m.name == name);
        let (api, dem) = (at("api"), at("dem"));
        before[2] = Some(Before::FeaturesButApi);
        let seeds = [None, api, None, None, None, None, dem, None];
        assert_eq!(ungated(&functions, &seeds, &before), seeds);
        let seeds = [None, dem, None, None, None, None, dem, None];
        assert_eq!(
            ungated(&functions, &seeds, &before),
            [None, None, None, None, None, None, dem, None]
        );
        assert_eq!(
            before_gate("<orb_style[0a1b]::v8::doc::StyleDoc>::from_json"),
            Some(Before::Gate)
        );
        assert_eq!(
            before_gate("<orb_style[0a1b]::v8::doc::StyleDoc>::apply"),
            Some(Before::FeaturesButApi)
        );
        assert_eq!(before_gate("orb_runtime::map::Map::frame"), None);
    }

    #[test]
    fn a_module_calls_only_the_modules_it_needs() {
        let function = |callees: &[u32]| Function {
            reachable: false,
            exported: false,
            callees: callees.to_vec(),
            size: 100,
            pointers: Vec::new(),
        };
        let functions = Functions {
            imported: 0,
            defined: vec![function(&[1]), function(&[]), function(&[1])],
            segments: Vec::new(),
            roots: Vec::new(),
        };
        let at = |name| MODULES.iter().position(|m| m.name == name);
        // sqlite's code may call archive's (gzip), geojson's may not, and
        // the API's calls whatever feature its style declares.
        let modules = [at("sqlite"), at("archive"), at("geojson")];
        assert_eq!(undeclared(&functions, &modules), [(2, 1)]);
        let modules = [at("api"), at("dem"), None];
        assert!(undeclared(&functions, &modules).is_empty());
    }
}
