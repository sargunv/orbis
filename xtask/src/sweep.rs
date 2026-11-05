//! `cargo xtask sweep`: prune stale build artifacts when the disk runs low.
//!
//! Cargo keeps every build of a crate under a fresh metadata hash, so a
//! `target/<profile>/deps` fills with duplicates: one worktree held 14 copies
//! of `liborb_runtime-<hash>.rlib` at about 60 MB each. Per `deps` directory
//! this keeps the two newest builds of each crate and deletes the rest; per
//! `incremental/` directory, the newest session of each crate. Nothing outside
//! the target directory is touched.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The newest builds kept per crate, to rebuild beside an older one.
const KEEP: usize = 2;

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let (mut dry_run, mut target) = (false, None);
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--target-dir" => {
                target = Some(
                    args.next()
                        .ok_or("sweep: --target-dir takes a path")?
                        .into(),
                );
            }
            other => return Err(format!("sweep: unknown argument `{other}`")),
        }
    }
    let target = target.unwrap_or_else(|| {
        std::env::var_os("CARGO_TARGET_DIR").map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../target"),
            PathBuf::from,
        )
    });
    let target =
        fs::canonicalize(&target).map_err(|e| format!("sweep: {}: {e}", target.display()))?;

    let (mut deps, mut incremental) = (Vec::new(), Vec::new());
    collect(&target, &mut deps, &mut incremental);
    let mut stale = Vec::new();
    for dir in &deps {
        stale.extend(sweep(dir, KEEP, deps_key)?);
    }
    for dir in &incremental {
        stale.extend(sweep(dir, 1, session_key)?);
    }

    let mut freed = 0;
    for path in &stale {
        let bytes = size(path);
        if dry_run {
            println!("{}", path.strip_prefix(&target).unwrap_or(path).display());
        } else if let Err(e) = remove(path) {
            eprintln!("sweep: {}: {e}", path.display());
            continue;
        }
        freed += bytes;
    }
    println!(
        "sweep: {} artifacts, {freed} bytes{}",
        stale.len(),
        if dry_run { " would be freed" } else { " freed" },
    );
    Ok(())
}

/// Every `deps` and `incremental` directory under `dir`; symlinks not entered.
fn collect(dir: &Path, deps: &mut Vec<PathBuf>, incremental: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let path = entry.path();
        match entry.file_name().to_str() {
            Some("deps") => deps.push(path),
            Some("incremental") => incremental.push(path),
            _ => collect(&path, deps, incremental),
        }
    }
}

/// In one directory, every path belonging to a build of a crate stem but its
/// `keep` newest, builds keyed by `key`.
fn sweep(
    dir: &Path,
    keep: usize,
    key: impl Fn(&str) -> Option<(String, String)>,
) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)
        .map_err(|e| format!("sweep: {}: {e}", dir.display()))?
        .flatten()
    {
        let name = entry.file_name().into_string().unwrap_or_default();
        let Some((stem, build)) = key(&name) else {
            continue;
        };
        let modified = entry
            .metadata()
            .ok()
            .and_then(|meta| meta.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        files.push((stem, build, modified, entry.path()));
    }
    let stale = doomed(
        files
            .iter()
            .map(|(stem, build, modified, _)| (stem.as_str(), build.as_str(), *modified)),
        keep,
    );
    Ok(files
        .iter()
        .filter(|(stem, build, ..)| stale.contains(&(stem.as_str(), build.as_str())))
        .map(|(.., path)| path.clone())
        .collect())
}

/// A `deps` file name's crate stem and build hash, from
/// `liborb_runtime-<hash>.rlib`, `orb_runtime-<hash>.d`, a test binary
/// `orb_runtime-<hash>` or an object `orb_runtime-<hash>.o`. Names like
/// `invoked.timestamp` don't parse.
fn deps_key(name: &str) -> Option<(String, String)> {
    let (stem, rest) = name.rsplit_once('-')?;
    let hash = rest.split('.').next()?;
    let stem = stem.strip_prefix("lib").unwrap_or(stem);
    let hex = hash.len() == 16 && hash.bytes().all(|b| b.is_ascii_hexdigit());
    let valid = hex && !stem.is_empty();
    valid.then(|| (stem.to_owned(), hash.to_owned()))
}

/// An `incremental` session's crate stem and its own name, `<crate>-<hash>`,
/// whose hash is not only hex digits.
fn session_key(name: &str) -> Option<(String, String)> {
    let stem = name.rsplit_once('-').map_or(name, |(stem, _)| stem);
    Some((stem.to_owned(), name.to_owned()))
}

/// The builds to delete: every build of each crate stem but its `keep`
/// newest, a build dated by its newest file.
fn doomed<'a>(
    files: impl Iterator<Item = (&'a str, &'a str, SystemTime)>,
    keep: usize,
) -> BTreeSet<(&'a str, &'a str)> {
    let mut files: Vec<_> = files.collect();
    files.sort_by_key(|&(_, _, modified)| Reverse(modified));
    let (mut kept, mut doomed) = (BTreeMap::new(), BTreeSet::new());
    for (stem, build, _) in files {
        let kept = kept.entry(stem).or_insert_with(BTreeSet::new);
        if !kept.contains(build) {
            if kept.len() == keep {
                doomed.insert((stem, build));
            } else {
                kept.insert(build);
            }
        }
    }
    doomed
}

/// The bytes under `path`: the file's, or a directory's whole tree.
fn size(path: &Path) -> u64 {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => {
            fs::read_dir(path).map_or(0, |e| e.flatten().map(|e| size(&e.path())).sum())
        }
        Ok(meta) => meta.len(),
        Err(_) => 0,
    }
}

fn remove(path: &Path) -> std::io::Result<()> {
    if fs::symlink_metadata(path)?.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Names group by crate stem and build; only the newest two survive.
    #[test]
    fn keeps_the_newest_two_builds() {
        let at = |secs| SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        let (a, b, c, d) = (
            "0123456789abcdef",
            "fedcba9876543210",
            "00000000000000ff",
            "1111111111111111",
        );
        let want = Some(("orb_runtime".to_owned(), a.to_owned()));
        for name in [
            format!("liborb_runtime-{a}.rlib"),
            format!("orb_runtime-{a}.d"),
            format!("orb_runtime-{a}"),
            format!("orb_runtime-{a}.o"),
        ] {
            assert_eq!(deps_key(&name), want);
        }
        assert_eq!(deps_key("invoked.timestamp"), None);

        // `a` was built twice: its newest file dates the build.
        let files = [
            ("orb_runtime", a, at(1)),
            ("orb_runtime", a, at(9)),
            ("orb_runtime", b, at(5)),
            ("orb_runtime", c, at(7)),
            ("orb_runtime", d, at(3)),
            // The same hash of another crate is another build.
            ("orb_scene", a, at(2)),
            ("orb_scene", b, at(8)),
        ];
        assert_eq!(
            doomed(files.into_iter(), KEEP),
            [("orb_runtime", b), ("orb_runtime", d)]
                .into_iter()
                .collect()
        );
    }
}
