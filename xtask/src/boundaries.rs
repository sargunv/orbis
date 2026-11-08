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
    "orb-capi",
    "orb-cli",
    "orb-web",
    "orb-shaper",
    "orb-demo",
    "orb-fixtures",
    "orb-bench",
    "diplomat-swift-kotlin",
    "dsk-fixture",
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
    "orb-capi",
    "orb-web",
    // The binding emitters' test bridge: Diplomat's macro writes `extern "C"`.
    "dsk-fixture",
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
    "wire", "worker",
];

/// Crates that don't build for wasm32.
pub(crate) const NATIVE_ONLY: &[&str] = &[
    "orb-host-native",
    "orb-api",
    "orb-bench",
    "orb-cli",
    "diplomat-swift-kotlin",
    "dsk-fixture",
    "orb-capi",
    "orb-demo",
    "orb-fixtures",
    "xtask",
];

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
