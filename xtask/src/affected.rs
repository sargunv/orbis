//! `cargo xtask check --affected`: the same checks as `cargo xtask check`,
//! scoped to the workspace packages that changed since a base revision (its
//! merge base with `HEAD`, plus working-tree and untracked files) and to every
//! package that depends on them, transitively. Changes outside every package
//! (`web/`, `AGENTS.md`) select nothing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// The packages a change reaches: the ones with changed files, and the ones
/// that depend on those.
pub(crate) struct Selection {
    changed: BTreeSet<String>,
    dependents: BTreeSet<String>,
}

impl Selection {
    /// Every selected package, changed first, each group in name order.
    pub(crate) fn packages(&self) -> Vec<&String> {
        self.changed.iter().chain(&self.dependents).collect()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.changed.contains(name) || self.dependents.contains(name)
    }

    /// The two-line report of what was selected and why.
    pub(crate) fn report(&self) -> String {
        let list = |names: &BTreeSet<String>| {
            if names.is_empty() {
                "none".to_string()
            } else {
                names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        format!(
            "affected: {} changed, {} dependent\n  changed: {}\n  dependent: {}",
            self.changed.len(),
            self.dependents.len(),
            list(&self.changed),
            list(&self.dependents)
        )
    }
}

pub(crate) fn select(base: &str) -> Result<Selection, String> {
    let metadata = metadata()?;
    let bad = || "unexpected cargo metadata shape".to_string();
    let root = PathBuf::from(
        metadata["workspace_root"]
            .as_str()
            .ok_or_else(bad)?
            .to_string(),
    );
    let members: BTreeSet<&str> = metadata["workspace_members"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    // Workspace package name → its directory, from the repository root.
    let mut packages = BTreeMap::new();
    let mut id_to_name = BTreeMap::new();
    for package in metadata["packages"].as_array().ok_or_else(bad)? {
        let id = package["id"].as_str().ok_or_else(bad)?;
        let name = package["name"].as_str().ok_or_else(bad)?.to_string();
        id_to_name.insert(id, name.clone());
        if members.contains(id) {
            let manifest = package["manifest_path"].as_str().ok_or_else(bad)?;
            let dir = Path::new(manifest)
                .parent()
                .ok_or_else(bad)?
                .strip_prefix(&root)
                .map_err(|e| format!("{manifest} is outside the workspace: {e}"))?
                .to_path_buf();
            packages.insert(name, dir);
        }
    }

    let mut changed = BTreeSet::new();
    for file in changed_files(&root, base)? {
        if let Some(name) = owner(&packages, &file) {
            changed.insert(name);
        }
    }

    // Reverse dependency closure over the workspace, all dependency kinds:
    // a dev-dependency's change breaks its dependents' tests too.
    let mut deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for node in metadata["resolve"]["nodes"].as_array().ok_or_else(bad)? {
        let name = id_to_name
            .get(node["id"].as_str().ok_or_else(bad)?)
            .ok_or_else(bad)?
            .clone();
        if !packages.contains_key(&name) {
            continue;
        }
        let entry = deps.entry(name).or_default();
        for dep in node["deps"].as_array().ok_or_else(bad)? {
            let dep = id_to_name
                .get(dep["pkg"].as_str().ok_or_else(bad)?)
                .ok_or_else(bad)?;
            if packages.contains_key(dep) {
                entry.insert(dep.clone());
            }
        }
    }

    let mut dependents = BTreeSet::new();
    let mut stack: Vec<String> = changed.iter().cloned().collect();
    while let Some(next) = stack.pop() {
        for (name, deps) in &deps {
            if changed.contains(name) || dependents.contains(name) || !deps.contains(&next) {
                continue;
            }
            dependents.insert(name.clone());
            stack.push(name.clone());
        }
    }

    Ok(Selection {
        changed,
        dependents,
    })
}

/// The workspace package owning `file`: the deepest manifest directory that
/// contains it (`dsk-fixture` sits inside `diplomat-swift-kotlin`).
fn owner(packages: &BTreeMap<String, PathBuf>, file: &Path) -> Option<String> {
    packages
        .iter()
        .filter(|(_, dir)| file.starts_with(dir))
        .max_by_key(|(_, dir)| dir.components().count())
        .map(|(name, _)| name.clone())
}

/// Every file changed since the merge base of `base` and `HEAD`, including
/// working-tree edits, plus untracked, non-ignored files. Paths from the
/// repository root.
fn changed_files(root: &Path, base: &str) -> Result<BTreeSet<PathBuf>, String> {
    let merge_base = git(root, &["merge-base", "HEAD", base])?;
    let diff = git(root, &["diff", "--name-only", merge_base.trim()])?;
    let untracked = git(root, &["ls-files", "--others", "--exclude-standard"])?;
    Ok(diff
        .lines()
        .chain(untracked.lines())
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("git output is not UTF-8: {e}"))
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
