//! `cargo xtask check --affected`: the same checks as `cargo xtask check`,
//! scoped to the workspace packages whose files, or files their code reads,
//! changed since a base revision (its merge base with `HEAD`, plus working-tree
//! and untracked files), the packages building against those, transitively,
//! and the packages whose tests build against any of them (a dev-dependency),
//! though not those packages' own dependents.
//!
//! A package reads the files its directory holds, and the ones outside it that
//! its Rust names: through `include_str!`, `include_bytes!` or `#[path]`
//! (relative to the file naming them), or as a path built on
//! `env!("CARGO_MANIFEST_DIR")` (relative to its manifest), as tests read
//! `web/demo/data` at run time. A file compiled into a package's library or
//! binaries changes what its dependents build against, so it selects them too,
//! even from its own `tests/`; one read at run time, or compiled only into its
//! `tests/`, `benches/` or `examples/`, selects that package alone. The
//! workspace's own build files (`Cargo.lock`, the toolchain) reach every
//! package; any other change outside the packages (`AGENTS.md`, the web
//! package's TypeScript) selects nothing.
//!
//! Reads built any other way go unseen: xtask's `root().join(..)`, a path
//! relative to the working directory, or `orb-fixtures`' walk up from its
//! manifest. A test that reads a file outside its package builds the path on
//! `CARGO_MANIFEST_DIR`, starting it with the literal that names the file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The packages a change reaches: the ones whose files, or files they read,
/// changed, and the ones that build or test against those.
pub(crate) struct Selection {
    changed: BTreeSet<String>,
    dependents: BTreeSet<String>,
}

/// A path that a package's Rust names: outside the package, or inside it and
/// compiled into its library or binaries (as a file under its `tests/` can be).
#[derive(Debug, PartialEq)]
struct Read {
    /// From the repository root, `/`-separated; a prefix where a `concat!` or
    /// `format!` finishes the name.
    path: String,
    /// Compiled into the package's library or binaries, so what its dependents
    /// build against, rather than read at run time or by its own tests alone.
    compiled: bool,
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

pub(crate) fn select(base: Option<&str>) -> Result<Selection, String> {
    let metadata = crate::cargo_metadata()?;
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
    // Packages a change reaches only through files they read at run time or
    // compile into their own tests: selected, but their dependents aren't.
    let mut readers = BTreeSet::new();
    match base {
        Some(base) => {
            let files = changed_files(&root, base)?;
            if WORKSPACE_FILES
                .iter()
                .any(|file| files.contains(Path::new(file)))
            {
                changed.extend(packages.keys().cloned());
            }
            for file in &files {
                if let Some((name, dir)) = owner(&packages, file) {
                    // Only the package's own tests, benches and examples
                    // build what these hold.
                    let own = ["tests", "benches", "examples"]
                        .iter()
                        .any(|target| file.starts_with(dir.join(target)));
                    if own {
                        readers.insert(name.clone());
                    } else {
                        changed.insert(name.clone());
                    }
                }
            }
            for (name, dir) in &packages {
                let reads = named_reads(&root, dir)?;
                if reads.iter().any(|read| read.path.is_empty()) {
                    return Err(format!(
                        "{name} names the repository's root in its Rust, so every change \
                         would select it; build its path from the directory it reads"
                    ));
                }
                for read in &reads {
                    if files.iter().any(|file| reaches(&read.path, file)) {
                        if read.compiled {
                            changed.insert(name.clone());
                        } else {
                            readers.insert(name.clone());
                        }
                    }
                }
            }
        }
        None => changed.extend(packages.keys().cloned()),
    }

    // Each package's workspace dependencies, and whether its library or
    // binaries build against each (not only its tests: a dev-dependency).
    let mut deps: BTreeMap<String, Vec<(String, bool)>> = BTreeMap::new();
    for node in metadata["resolve"]["nodes"].as_array().ok_or_else(bad)? {
        let name = id_to_name
            .get(node["id"].as_str().ok_or_else(bad)?)
            .ok_or_else(bad)?;
        if !packages.contains_key(name) {
            continue;
        }
        let entry = deps.entry(name.clone()).or_default();
        for dep in node["deps"].as_array().ok_or_else(bad)? {
            let dep_name = id_to_name
                .get(dep["pkg"].as_str().ok_or_else(bad)?)
                .ok_or_else(bad)?;
            if packages.contains_key(dep_name) {
                let kinds = dep["dep_kinds"].as_array().ok_or_else(bad)?;
                let built = kinds.iter().any(|kind| kind["kind"] != "dev");
                entry.push((dep_name.clone(), built));
            }
        }
    }

    // What builds against a changed package changes too, transitively; what
    // only tests against one is checked, but its own dependents aren't.
    let mut built = changed.clone();
    let mut stack: Vec<String> = changed.iter().cloned().collect();
    while let Some(next) = stack.pop() {
        for (name, deps) in &deps {
            if !built.contains(name) && deps.iter().any(|(dep, b)| *b && *dep == next) {
                built.insert(name.clone());
                stack.push(name.clone());
            }
        }
    }
    let tested = (deps.iter())
        .filter(|(_, deps)| deps.iter().any(|(dep, _)| built.contains(dep)))
        .map(|(name, _)| name.clone());
    let dependents: BTreeSet<String> = (built.iter().cloned().chain(tested))
        .filter(|name| !changed.contains(name))
        .collect();
    changed.extend(
        readers
            .into_iter()
            .filter(|name| !dependents.contains(name)),
    );

    Ok(Selection {
        changed,
        dependents,
    })
}

/// The workspace package owning `file`, and its directory: the deepest
/// manifest directory that contains it.
fn owner<'a>(
    packages: &'a BTreeMap<String, PathBuf>,
    file: &Path,
) -> Option<(&'a String, &'a PathBuf)> {
    packages
        .iter()
        .filter(|(_, dir)| file.starts_with(dir))
        .max_by_key(|(_, dir)| dir.components().count())
}

/// Files every package's build reads, from the repository root: a change to
/// one selects every package.
const WORKSPACE_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    ".config/nextest.toml",
];

/// Whether `file` (from the repository root) is what `read` names: the file
/// itself, one in that directory, or one a `concat!` could finish the name of.
fn reaches(read: &str, file: &Path) -> bool {
    file.to_str().is_some_and(|file| file.starts_with(read))
}

/// The paths that the Rust sources of the package in `dir` (from `root`) name
/// outside it, and the ones its library or binaries compile in from inside it,
/// which a file's place alone (in `tests/`, say) doesn't tell.
fn named_reads(root: &Path, dir: &Path) -> Result<Vec<Read>, String> {
    let mut sources = Vec::new();
    rust_sources(&root.join(dir), &mut sources)?;
    let mut reads = Vec::new();
    for source in sources {
        let text =
            std::fs::read_to_string(&source).map_err(|e| format!("{}: {e}", source.display()))?;
        let file_dir = source
            .parent()
            .and_then(|parent| parent.strip_prefix(root).ok())
            .ok_or_else(|| format!("{} is outside the workspace", source.display()))?;
        // What only the package's own tests, benches and examples compile in
        // reaches no dependent.
        let own = ["tests", "benches", "examples"]
            .iter()
            .any(|target| file_dir.starts_with(dir.join(target)));
        reads.extend(
            named_paths(&text, file_dir, dir)
                .into_iter()
                .map(|read| Read {
                    compiled: read.compiled && !own,
                    ..read
                })
                .filter(|read| read.compiled || !Path::new(&read.path).starts_with(dir)),
        );
    }
    Ok(reads)
}

/// Every `.rs` file under `dir`, but in build output and hidden directories.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let kind = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if kind.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_sources(&path, out)?;
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The paths `source` names that a build or test reads: each
/// `include_str!`, `include_bytes!` and `#[path]` literal from `file_dir`, the
/// source's directory, and from `package_dir` the path built on
/// `env!("CARGO_MANIFEST_DIR")`, read at run time unless an include takes it:
/// the literal that continues it, as `concat!`'s next argument or a `.join(`
/// right after it, or the format string it is the first argument of, after its
/// placeholder. A name that a `concat!` or `format!` goes on to finish stays a
/// prefix (cut at a `{` placeholder); one leaving the repository is dropped.
fn named_paths(source: &str, file_dir: &Path, package_dir: &Path) -> Vec<Read> {
    let mut reads = Vec::new();
    for marker in ["include_str!(", "include_bytes!(", "#[path"] {
        for (at, _) in source.match_indices(marker) {
            let rest = source[at + marker.len()..].trim_start();
            let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();
            let rest = rest.strip_prefix("concat!(").unwrap_or(rest).trim_start();
            if let Some((name, _)) = literal(rest) {
                reads.extend(resolve(file_dir, prefix(name)).map(|path| Read {
                    path,
                    compiled: true,
                }));
            }
        }
    }
    const MANIFEST_DIR: &str = "env!(\"CARGO_MANIFEST_DIR\")";
    for (at, _) in source.match_indices(MANIFEST_DIR) {
        let after = source[at + MANIFEST_DIR.len()..].trim_start();
        let continued = after.strip_prefix(',').or_else(|| {
            let join = after
                .strip_prefix(')')?
                .trim_start()
                .strip_prefix(".join(")?;
            let join = join.trim_start();
            Some(join.strip_prefix("format!(").unwrap_or(join))
        });
        let following = continued.and_then(|rest| literal(rest.trim_start()));
        let before = &source[..at];
        let before = &before[before.rfind(';').map_or(0, |end| end + 1)..];
        let format = before.rfind("format!(").and_then(|open| {
            let (template, rest) = literal(before[open + "format!(".len()..].trim_start())?;
            let first_argument = rest.trim().strip_prefix(',')?.trim().is_empty();
            first_argument.then_some(template.split_once("{}")?.1)
        });
        let included = ["include_str!(", "include_bytes!("].iter().any(|include| {
            before
                .trim_end()
                .strip_suffix("concat!(")
                .is_some_and(|before| before.trim_end().ends_with(include))
        });
        if let Some(name) = following.map(|(name, _)| name).or(format) {
            reads.extend(
                resolve(package_dir, prefix(name).trim_start_matches('/')).map(|path| Read {
                    path,
                    compiled: included,
                }),
            );
        }
    }
    reads
}

/// The contents of the string literal, plain or raw, that `text` starts with,
/// and what follows its closing quote.
fn literal(text: &str) -> Option<(&str, &str)> {
    if let Some(raw) = text.strip_prefix('r') {
        let hashes = raw.len() - raw.trim_start_matches('#').len();
        let body = raw[hashes..].strip_prefix('"')?;
        let close = format!("\"{}", "#".repeat(hashes));
        let end = body.find(&close)?;
        return Some((&body[..end], &body[end + close.len()..]));
    }
    let body = text.strip_prefix('"')?;
    let end = body.find(['"', '\n'])?;
    body[end..]
        .starts_with('"')
        .then(|| (&body[..end], &body[end + 1..]))
}

/// `name` up to its first `{` placeholder.
fn prefix(name: &str) -> &str {
    name.split('{').next().unwrap_or(name)
}

/// `relative` joined to `dir` and normalised, `/`-separated; `None` when it
/// climbs out of the repository.
fn resolve(dir: &Path, relative: &str) -> Option<String> {
    let mut parts: Vec<&str> = dir.iter().filter_map(|part| part.to_str()).collect();
    for part in relative.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            part => parts.push(part),
        }
    }
    let mut path = parts.join("/");
    if relative.ends_with('/') && !path.is_empty() {
        path.push('/');
    }
    Some(path)
}

/// Every file changed since the merge base of `base` and `HEAD`, including
/// working-tree edits, plus untracked, non-ignored files. Paths from the
/// repository root.
fn changed_files(root: &Path, base: &str) -> Result<BTreeSet<PathBuf>, String> {
    let merge_base = crate::git(root, &["merge-base", "HEAD", base])?;
    let diff = crate::git(root, &["diff", "--name-only", &merge_base])?;
    let untracked = crate::git(root, &["ls-files", "--others", "--exclude-standard"])?;
    Ok(diff
        .lines()
        .chain(untracked.lines())
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reads(source: &str) -> Vec<Read> {
        named_paths(
            source,
            Path::new("crates/orb-bench/src"),
            Path::new("crates/orb-bench"),
        )
    }

    fn named(source: &str) -> Vec<String> {
        reads(source).into_iter().map(|read| read.path).collect()
    }

    #[test]
    fn includes_resolve_from_their_file() {
        assert_eq!(
            named(r#"const S: &str = include_str!("../../../web/scripts/sessions.json");"#),
            ["web/scripts/sessions.json"]
        );
        assert_eq!(
            named(r#"&include_bytes!("../tests/data/a.pbf")[..]"#),
            ["crates/orb-bench/tests/data/a.pbf"]
        );
        assert_eq!(
            named(r#"#[path = "../../../crates/orb-fixtures/src/status.rs"] mod status;"#),
            ["crates/orb-fixtures/src/status.rs"]
        );
        assert_eq!(
            named(r##"include_str!(r#"../../../web/demo/data/a.json"#)"##),
            ["web/demo/data/a.json"]
        );
    }

    #[test]
    fn manifest_paths_resolve_from_the_manifest() {
        assert_eq!(
            named(r#"concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data")"#),
            ["web/demo/data"]
        );
        assert_eq!(
            named(
                r#"PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../web/demo/glyphs")"#
            ),
            ["web/demo/glyphs"]
        );
    }

    /// What an include takes is compiled in, so it reaches the package's
    /// dependents; a path read at run time reaches only the package.
    #[test]
    fn includes_compile_in_and_manifest_paths_read_at_run_time() {
        let compiled = |source| {
            reads(source)
                .iter()
                .map(|read| read.compiled)
                .collect::<Vec<_>>()
        };
        assert_eq!(compiled(r#"include_str!("../../../web/a.json")"#), [true]);
        assert_eq!(
            compiled(r#"concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data")"#),
            [false]
        );
        assert_eq!(
            compiled(r#"include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/a.json"))"#),
            [true]
        );
    }

    /// A `concat!` or `format!` finishes the name later, so what's known is a
    /// prefix.
    #[test]
    fn a_name_finished_later_stays_a_prefix() {
        let prefix = &named(r#"include_str!(concat!("../tests/layout-", $name, ".json"))"#)[0];
        assert_eq!(prefix, "crates/orb-bench/tests/layout-");
        assert!(reaches(
            prefix,
            Path::new("crates/orb-bench/tests/layout-a.json")
        ));
        assert_eq!(
            named(r#"format!("file://{}/../../web/{name}.json", env!("CARGO_MANIFEST_DIR"));"#),
            ["web/"]
        );
        assert_eq!(
            named(r#"format!("{}: {}", path, env!("CARGO_MANIFEST_DIR"));"#),
            Vec::<String>::new(),
            "the directory isn't the format's first argument"
        );
        assert_eq!(
            named(r#"Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../web/{name}"))"#),
            ["web/"]
        );
    }

    /// Only a literal that continues the path names a file: not one later in
    /// the statement, and not one leaving the repository.
    #[test]
    fn literals_that_dont_continue_the_path_name_nothing() {
        assert_eq!(
            named(r#"let d = env!("CARGO_MANIFEST_DIR"); let s = "../../web";"#),
            Vec::<String>::new()
        );
        assert_eq!(
            named(
                r#"Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("../../the workspace root")"#
            ),
            Vec::<String>::new()
        );
        assert_eq!(
            named(r#"include_str!("../../../../outside.json")"#),
            Vec::<String>::new()
        );
    }
}
