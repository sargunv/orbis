//! Boundary rules, checked against `cargo metadata`.
//!
//! Crates are classified by name; a crate not named in any list is a core
//! crate: no threads, no I/O, no GPU handles, no `unsafe`, buildable for
//! wasm32.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// Crates at or above `orb-runtime`, and tools: they may own threads, I/O,
/// GPU handles and platform APIs.
const PLATFORM: &[&str] = &[
    "orb-runtime",
    "orb-host",
    "orb-host-native",
    "orb-hal-wgpu",
    "orb-api",
    "orb-android",
    "orb-cli",
    "orb-ffi",
    "orb-web",
    "orb-shaper",
    "orb-demo",
    "orb-fixtures",
    "orb-fuzz",
    "orb-bench",
    "xtask",
];

/// External crates that bring threads, I/O, GPU handles or platform APIs.
/// Core crates may not depend on them, directly or transitively.
const PLATFORM_DEPS: &[&str] = &[
    "wgpu",
    "wgpu-core",
    "wgpu-hal",
    "winit",
    "ureq",
    "tokio",
    "web-sys",
    "js-sys",
    "wasm-bindgen",
];

/// Crates allowed to use `unsafe`. Every other crate root must
/// carry `#![forbid(unsafe_code)]`.
const UNSAFE_ALLOWED: &[&str] = &[
    "orb-hal-wgpu",
    // System font files are mapped, not copied (`fonts.rs`).
    "orb-host-native",
    // The Android app's entry point, JNI and logcat.
    "orb-android",
    // maplibre-native-ffi's C API.
    "orb-ffi",
    "orb-web",
    // A Windows child's CPU time and peak memory, for `h2h` (`h2h.rs`).
    "xtask",
    // A thread's instructions retired, for the benches (`clock.rs`).
    "orb-bench",
];

/// Job-side crates are share-nothing and never depend on scene-side crates.
const JOB_SIDE: &[&str] = &["orb-tile", "orb-layout", "orb-text", "orb-symbol"];
const SCENE_SIDE: &[&str] = &[
    "orb-style",
    "orb-camera",
    "orb-render",
    "orb-hal",
    "orb-shaders",
];

/// orb-runtime's tile-job modules, which name no scene-side crate, so the
/// job side keeps apart within the crate.
const JOB_MODULES: &[&str] = &[
    "features", "geojson", "glyphs", "job_host", "line", "loader", "style", "symbol", "tile",
    "stages", "wire", "worker",
];

/// Crates that don't build for wasm32.
pub(crate) const NATIVE_ONLY: &[&str] = &[
    "orb-host-native",
    "orb-api",
    "orb-bench",
    "orb-android",
    "orb-cli",
    "orb-demo",
    "orb-ffi",
    "orb-fixtures",
    "orb-fuzz",
    "xtask",
];

/// What only the GPU-driven path has (compute, indirect draws, storage
/// buffers and textures), which the WebGL 2 path
/// (#140) lacks. Outside the HALs that define it, only the files of
/// [`GPU_ONLY_FILES`] may name it, each only what its entry allows.
const GPU_ONLY: &[&str] = &[
    "Pass::Compute",
    "ComputePass",
    "ComputeCommand",
    "create_compute_pipeline",
    "DrawIndirect",
    "DrawIndexedIndirect",
    "BufferUsages::INDIRECT",
    "BufferUsages::STORAGE",
    "E::storage(",
    "E::storage_rw(",
    "BindingType::Storage",
    "StorageTexture",
    "STORAGE_BINDING",
];

/// The crates that define and implement [`GPU_ONLY`] API.
const HALS: &[&str] = &["orb-hal", "orb-hal-wgpu"];

/// What a file of [`GPU_ONLY_FILES`] may name of [`GPU_ONLY`].
#[derive(Clone, Copy)]
enum Allows {
    All,
    Only(&'static [&'static str]),
}

/// The files, under `crates/`, that may name [`GPU_ONLY`] API: each with
/// the API it may name, and what stands in for that on WebGL 2 or the gap
/// it leaves. A feature born in compute names its GL formulation here, in
/// the same commit; an entry that allows what its file no longer names
/// fails too, so the list never grants more than is used.
const GPU_ONLY_FILES: &[(&str, Allows, &str)] = &[
    (
        "orb-render/src/driver/gpu.rs",
        Allows::All,
        "the GPU driver: the CPU driver's direct draws",
    ),
    (
        "orb-render/src/pipeline.rs",
        Allows::Only(&["create_compute_pipeline"]),
        "the compute pipelines of the files below, which WebGL 2 never makes",
    ),
    (
        "orb-render/src/downsample.rs",
        Allows::Only(&[
            "Pass::Compute",
            "ComputePass",
            "ComputeCommand",
            "BufferUsages::STORAGE",
            "BindingType::Storage",
            "StorageTexture",
            "STORAGE_BINDING",
        ]),
        "compute mips and edges over regions in storage: downsample_draw.wgsl's drawn ones over regions in a data texture",
    ),
    (
        "orb-render/src/image.rs",
        Allows::Only(&["Pass::Compute", "STORAGE_BINDING"]),
        "raster pages' compute mips, one pass for every page, and the views they store to: drawn ones, into render targets",
    ),
    (
        "orb-render/src/terrain.rs",
        Allows::Only(&["E::storage(", "STORAGE_BINDING"]),
        "the drape clear's regions in storage: fetch_drape.wgsl over a data texture; drape mips: drawn ones",
    ),
    (
        "orb-render/src/scene/tables.rs",
        Allows::Only(&["BufferUsages::STORAGE", "BindingType::Storage"]),
        "the tables' storage backing: their texture backing",
    ),
    (
        "orb-render/src/symbol.rs",
        Allows::Only(&[
            "Pass::Compute",
            "ComputePass",
            "ComputeCommand",
            "BufferUsages::STORAGE",
            "E::storage(",
            "BindingType::Storage",
            "StorageTexture",
            "STORAGE_BINDING",
        ]),
        "symbol placements in one dispatch into a storage texture, which reads every node's uniform and the draw arguments from storage: each corner placing its quad itself (symbol_place_inline.wgsl)",
    ),
];

/// [`GPU_ONLY`] API outside what [`GPU_ONLY_FILES`] allows, in every
/// crate's sources but the HALs'.
fn gpu_only_uses() -> Result<Vec<String>, String> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates");
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(&crates).map_err(|e| format!("{}: {e}", crates.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if !HALS.contains(&name.as_ref()) && path.join("src").is_dir() {
            dirs.push(path.join("src"));
        }
    }
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let mut errors = Vec::new();
    let mut listed = BTreeSet::new();
    for path in files {
        let name = path
            .strip_prefix(&crates)
            .expect("under crates")
            .to_string_lossy()
            .replace('\\', "/");
        let allows = GPU_ONLY_FILES
            .iter()
            .find(|(f, ..)| *f == name)
            .map_or(Allows::Only(&[]), |&(_, allows, _)| allows);
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut named = BTreeSet::new();
        for (i, line) in source.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for &api in GPU_ONLY.iter().filter(|api| code.contains(*api)) {
                named.insert(api);
                if let Allows::Only(allowed) = allows
                    && !allowed.contains(&api)
                {
                    errors.push(format!(
                        "{name}:{}: `{api}` is GPU-only; draw it another way, or allow it in xtask's GPU_ONLY_FILES with what stands in for it on WebGL 2 (#140)",
                        i + 1
                    ));
                }
            }
        }
        if let Allows::Only(allowed) = allows {
            for api in allowed.iter().filter(|api| !named.contains(*api)) {
                errors.push(format!(
                    "xtask's GPU_ONLY_FILES allows {name} `{api}`, which it no longer names"
                ));
            }
        }
        listed.insert(name);
    }
    for (file, ..) in GPU_ONLY_FILES.iter().filter(|(f, ..)| !listed.contains(*f)) {
        errors.push(format!(
            "xtask's GPU_ONLY_FILES lists {file}, which is gone"
        ));
    }
    Ok(errors)
}

pub(crate) fn check() -> Result<(), String> {
    let metadata = metadata()?;
    let graph = Graph::new(&metadata)?;
    let mut errors = Vec::new();

    for (name, package) in &graph.workspace {
        let deps = graph.normal_closure(name);
        if !PLATFORM.contains(&name.as_str()) {
            for dep in deps
                .iter()
                .filter(|d| PLATFORM.contains(&d.as_str()) || PLATFORM_DEPS.contains(&d.as_str()))
            {
                errors.push(format!(
                    "{name}: core crate depends on platform crate `{dep}`"
                ));
            }
        }
        if JOB_SIDE.contains(&name.as_str()) {
            for dep in deps.iter().filter(|d| SCENE_SIDE.contains(&d.as_str())) {
                errors.push(format!(
                    "{name}: job-side crate depends on scene-side crate `{dep}`"
                ));
            }
        }
        if !UNSAFE_ALLOWED.contains(&name.as_str()) {
            for root in &package.roots {
                let source = std::fs::read_to_string(root)
                    .map_err(|e| format!("{}: {e}", root.display()))?;
                if !source.contains("#![forbid(unsafe_code)]") {
                    errors.push(format!(
                        "{}: missing #![forbid(unsafe_code)]",
                        root.display()
                    ));
                }
            }
        }
    }

    let runtime = Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/orb-runtime/src");
    for module in JOB_MODULES {
        let path = runtime.join(format!("{module}.rs"));
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for scene in SCENE_SIDE.iter().map(|c| c.replace('-', "_")) {
            if scene != "orb_style" && scene != "orb_shaders" && source.contains(&scene) {
                errors.push(format!(
                    "orb-runtime's {module}.rs: job-side module names `{scene}`"
                ));
            }
        }
    }

    errors.extend(gpu_only_uses()?);

    if errors.is_empty() {
        eprintln!("boundaries: {} crates ok", graph.workspace.len());
        Ok(())
    } else {
        Err(format!("boundary violations:\n  {}", errors.join("\n  ")))
    }
}

fn metadata() -> Result<Value, String> {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1"])
        .output()
        .map_err(|e| format!("failed to run cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| format!("bad cargo metadata: {e}"))
}

struct Package {
    /// Crate root files of the package's lib and bin targets.
    roots: Vec<std::path::PathBuf>,
}

struct Graph {
    workspace: BTreeMap<String, Package>,
    /// Package name → names of its normal (non-dev, non-build) dependencies.
    normal_deps: BTreeMap<String, BTreeSet<String>>,
}

impl Graph {
    fn new(metadata: &Value) -> Result<Self, String> {
        let bad = || "unexpected cargo metadata shape".to_string();
        let packages = metadata["packages"].as_array().ok_or_else(bad)?;
        let members: BTreeSet<&str> = metadata["workspace_members"]
            .as_array()
            .ok_or_else(bad)?
            .iter()
            .filter_map(Value::as_str)
            .collect();

        let mut id_to_name = BTreeMap::new();
        let mut workspace = BTreeMap::new();
        for package in packages {
            let id = package["id"].as_str().ok_or_else(bad)?;
            let name = package["name"].as_str().ok_or_else(bad)?.to_string();
            id_to_name.insert(id, name.clone());
            if members.contains(id) {
                let roots = package["targets"]
                    .as_array()
                    .ok_or_else(bad)?
                    .iter()
                    .filter(|t| {
                        t["kind"].as_array().is_some_and(|k| {
                            k.iter().any(|k| matches!(k.as_str(), Some("lib" | "bin")))
                        })
                    })
                    .filter_map(|t| t["src_path"].as_str().map(|p| Path::new(p).to_path_buf()))
                    .collect();
                workspace.insert(name, Package { roots });
            }
        }

        let mut normal_deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for node in metadata["resolve"]["nodes"].as_array().ok_or_else(bad)? {
            let name = &id_to_name[node["id"].as_str().ok_or_else(bad)?];
            let entry = normal_deps.entry(name.clone()).or_default();
            for dep in node["deps"].as_array().ok_or_else(bad)? {
                let is_normal = dep["dep_kinds"]
                    .as_array()
                    .ok_or_else(bad)?
                    .iter()
                    .any(|k| k["kind"].is_null());
                if is_normal {
                    entry.insert(id_to_name[dep["pkg"].as_str().ok_or_else(bad)?].clone());
                }
            }
        }
        Ok(Self {
            workspace,
            normal_deps,
        })
    }

    /// Every package reachable from `name` through normal dependencies.
    fn normal_closure(&self, name: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![name.to_string()];
        while let Some(next) = stack.pop() {
            for dep in self.normal_deps.get(&next).into_iter().flatten() {
                if seen.insert(dep.clone()) {
                    stack.push(dep.clone());
                }
            }
        }
        seen
    }
}
