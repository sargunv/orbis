//! `cargo xtask sweep`: prune stale build artifacts when the disk runs low.
//!
//! Cargo keeps every build of a crate under a fresh metadata hash, so a
//! `target/<profile>/deps` fills with duplicates: one worktree held 14 copies
//! of `liborb_runtime-<hash>.rlib` at about 60 MB each. One stem has many
//! builds live at once (lib, test harness, clippy's check metadata, feature
//! sets), so per `deps` directory this keeps every build compiled within
//! `--keep-hours` (default 6) and the two newest of each crate, and deletes
//! the rest; per `incremental/` directory, the newest session of each crate.
//!
//! A no-op rebuild touches nothing: neither a build's `.fingerprint` directory
//! nor its deps files move (measured), so a build's newest deps file dates when
//! it was last compiled. Nothing outside the target directory is touched.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The newest builds kept per crate, to rebuild beside an older one.
const KEEP: usize = 2;

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let (mut dry_run, mut target, mut hours) = (false, None, 6u64);
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
            "--keep-hours" => {
                let value = args.next().ok_or("sweep: --keep-hours takes a number")?;
                hours = value
                    .parse()
                    .map_err(|_| format!("sweep: --keep-hours `{value}` is not a number"))?;
            }
            other => return Err(format!("sweep: unknown argument `{other}`")),
        }
    }
    let window = Duration::from_secs(hours.saturating_mul(3600));
    let cutoff = SystemTime::now()
        .checked_sub(window)
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let target = target.unwrap_or_else(|| {
        std::env::var_os("CARGO_TARGET_DIR")
            .map_or_else(|| crate::root().join("target"), PathBuf::from)
    });
    let target =
        fs::canonicalize(&target).map_err(|e| format!("sweep: {}: {e}", target.display()))?;

    let (mut deps, mut incremental) = (Vec::new(), Vec::new());
    collect(&target, &mut deps, &mut incremental);
    let mut stale = Vec::new();
    for dir in &deps {
        stale.extend(sweep(dir, KEEP, cutoff, deps_key)?);
    }
    for dir in &incremental {
        stale.extend(sweep(dir, 1, cutoff, session_key)?);
    }

    let (mut freed, mut counted) = (0, BTreeSet::new());
    for path in &stale {
        let bytes = size(path, &mut counted);
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
    cutoff: SystemTime,
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
        cutoff,
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

/// The builds to delete: those of each crate stem that were neither compiled
/// since `cutoff` nor among its `keep` newest, a build dated by its newest file.
fn doomed<'a>(
    files: impl Iterator<Item = (&'a str, &'a str, SystemTime)>,
    keep: usize,
    cutoff: SystemTime,
) -> BTreeSet<(&'a str, &'a str)> {
    let mut files: Vec<_> = files.collect();
    files.sort_by_key(|&(.., modified)| Reverse(modified));
    let (mut newest, mut seen, mut doomed) = (BTreeMap::new(), BTreeSet::new(), BTreeSet::new());
    for (stem, build, modified) in files {
        // Newest first: a build's first sighting is its newest file.
        if !seen.insert((stem, build)) {
            continue;
        }
        let rank = newest.entry(stem).or_insert(0);
        if *rank >= keep && modified < cutoff {
            doomed.insert((stem, build));
        }
        *rank += 1;
    }
    doomed
}

/// The bytes under `path`: the file's, or a directory's whole tree, counting a
/// file's bytes once across hard links.
fn size(path: &Path, seen: &mut BTreeSet<(u64, u64)>) -> u64 {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_dir() {
        return fs::read_dir(path).map_or(0, |e| e.flatten().map(|e| size(&e.path(), seen)).sum());
    }
    let counted = inode(&meta).is_none_or(|id| seen.insert(id));
    if counted { meta.len() } else { 0 }
}

/// The device and inode a file's bytes live in, where the platform reports them.
#[cfg(unix)]
fn inode(meta: &fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Some((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn inode(_: &fs::Metadata) -> Option<(u64, u64)> {
    None
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

    /// Names group by crate stem and build; a build survives when it was
    /// compiled in the window or is one of its crate's newest two.
    #[test]
    fn keeps_recent_and_newest_builds() {
        let at = |hours: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(hours * 3600);
        let (a, b, c, d, e) = (
            "0123456789abcdef",
            "fedcba9876543210",
            "00000000000000ff",
            "1111111111111111",
            "2222222222222222",
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

        // `e` and `d` are the newest two of orb_runtime; `c` is past them but
        // within the 4 h window, so only the two older builds go.
        let mut files = vec![
            ("orb_runtime", e, at(5)),
            ("orb_runtime", d, at(4)),
            ("orb_runtime", c, at(3)),
            ("orb_runtime", b, at(1)),
            ("orb_runtime", a, at(0)),
            // orb_render is idle: its newest two stay anyway.
            ("orb_render", a, at(1)),
            ("orb_render", b, at(2)),
            // The same hash of another crate is another build.
            ("orb_style", a, at(5)),
        ];
        assert_eq!(
            doomed(files.clone().into_iter(), KEEP, at(2)),
            [("orb_runtime", a), ("orb_runtime", b)]
                .into_iter()
                .collect()
        );
        // `a` was built twice: its newest file dates the build, keeping it.
        files.push(("orb_runtime", b, at(6)));
        assert_eq!(
            doomed(files.into_iter(), KEEP, at(2)),
            [("orb_runtime", a)].into_iter().collect()
        );
    }
}
