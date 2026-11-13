//! A/B runs against another revision: `bench --compare <rev>` builds
//! `orb-bench` at `<rev>` and runs it interleaved with this tree's, and
//! `h2h --compare <rev>` adds that revision's `orbis` as an engine. The
//! revision is checked out into one worktree, `target/compare/src`, kept
//! between runs and built into `target/compare/target`: a checkout touches
//! only the files that differ, so the next comparison rebuilds only the
//! crates they're in.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::root;

/// A revision's build in `target/compare/`, which stays locked until this
/// drops.
pub(crate) struct Build {
    /// The revision as given.
    pub(crate) rev: String,
    pub(crate) commit: String,
    /// The build's binaries: `target/compare/target/<profile>`.
    pub(crate) bin: PathBuf,
    _lock: File,
}

impl Build {
    /// Checks `rev` out into the comparison worktree and builds `packages`
    /// there in `profile`.
    pub(crate) fn new(rev: &str, packages: &[&str], profile: &str) -> Result<Self, String> {
        let root = root();
        let commit = git(
            &root,
            &[
                "rev-parse",
                "--short",
                "--verify",
                &format!("{rev}^{{commit}}"),
            ],
        )?;
        let compare = root.join("target/compare");
        std::fs::create_dir_all(&compare).map_err(|e| format!("{}: {e}", compare.display()))?;
        // Another comparison in this tree checks out and builds in the same place.
        let lock_path = compare.join("lock");
        let lock = File::create(&lock_path).map_err(|e| format!("{}: {e}", lock_path.display()))?;
        if lock.try_lock().is_err() {
            eprintln!("compare: waiting for another comparison in this tree");
            lock.lock()
                .map_err(|e| format!("{}: {e}", lock_path.display()))?;
        }
        let worktree = compare.join("src");
        eprintln!("compare: {rev} ({commit}) in {}", worktree.display());
        if worktree.join(".git").exists() {
            git(
                &worktree,
                &["checkout", "--quiet", "--detach", "--force", &commit],
            )?;
        } else {
            // A worktree whose directory went with target/ is still registered.
            git(&root, &["worktree", "prune"])?;
            git(
                &root,
                &["worktree", "add", "--detach", path(&worktree)?, &commit],
            )?;
        }
        let manifest = worktree.join("Cargo.toml");
        let target = compare.join("target");
        let mut args = vec![
            "build",
            "--profile",
            profile,
            "--manifest-path",
            path(&manifest)?,
            "--target-dir",
            path(&target)?,
        ];
        for package in packages {
            args.extend(["-p", package]);
        }
        crate::cargo(&args)?;
        let bin = target.join(profile);
        // The DXC this tree's builds load (`cargo xtask dxc`), so both
        // sides of a D3D12 comparison compile with the same one.
        for dll in crate::dxc::DLLS {
            let from = root.join("target").join(profile).join(dll);
            if from.exists() {
                let to = bin.join(dll);
                std::fs::copy(&from, &to).map_err(|e| format!("{}: {e}", to.display()))?;
            }
        }
        Ok(Self {
            rev: rev.to_string(),
            commit,
            bin,
            _lock: lock,
        })
    }
}

/// `cargo xtask bench --compare <rev> [--runs N] [orb-bench args]`: `runs`
/// rounds of `orb-bench` at `rev` and at this tree, alternating which goes
/// first, then every table of the report with each number as the median
/// of `rev`'s runs → this tree's, and the change. A `rev` that names an
/// `orb-bench` binary runs that build instead, as built once and kept
/// aside, without a checkout of its own.
pub(crate) fn bench(args: &[String]) -> Result<(), String> {
    let mut rev = None;
    let mut runs = 3;
    let mut pass = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--compare" => rev = Some(args.next().ok_or("bench: --compare takes a revision")?),
            "--runs" => {
                let n = args.next().ok_or("bench: --runs takes a count")?;
                runs = n
                    .parse::<usize>()
                    .map_err(|e| format!("bench: --runs {n}: {e}"))?
                    .max(1);
            }
            _ => pass.push(arg.as_str()),
        }
    }
    let rev = rev.ok_or("bench: --compare takes a revision")?;
    let given = Path::new(rev);
    let (_build, label, before) = match given.is_file() {
        true => (None, rev.clone(), given.to_path_buf()),
        false => {
            let build = Build::new(rev, &["orb-bench"], "release")?;
            let label = format!("{} ({})", build.rev, build.commit);
            let bin = build.bin.join(crate::exe("orb-bench"));
            (Some(build), label, bin)
        }
    };
    crate::cargo(&["build", "--release", "-p", "orb-bench"])?;
    let after = root().join("target/release").join(crate::exe("orb-bench"));
    let mut outputs: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    for run in 0..runs {
        let order = if run % 2 == 0 { [0, 1] } else { [1, 0] };
        for side in order {
            let (name, bin) = match side {
                0 => (label.as_str(), &before),
                _ => ("working tree", &after),
            };
            eprintln!("compare: run {} of {runs}: {name}", run + 1);
            let out = Command::new(bin)
                .args(&pass)
                .output()
                .map_err(|e| format!("{}: {e}", bin.display()))?;
            if !out.status.success() {
                return Err(format!(
                    "{name}'s orb-bench failed:\n{}",
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            outputs[side].push(String::from_utf8_lossy(&out.stdout).into_owned());
        }
    }
    print!(
        "## Bench: {label} → working tree, {runs} runs interleaved\n\n{}",
        merge(&outputs[0], &outputs[1])
    );
    Ok(())
}

/// The last of `after`'s reports, each numeric table cell replaced by the
/// median over `before`'s reports → the median over `after`'s and the
/// change, and each `min–max` cell by the range over `before`'s → over
/// `after`'s. A row is matched by its table, its first cell and how many
/// rows before it in the table share that cell (the tile bench's mvt and
/// mlt rows of one tile).
fn merge(before: &[String], after: &[String]) -> String {
    let template = after.last().map_or("", String::as_str);
    let (before, after) = (cells(before), cells(after));
    let mut out = String::new();
    for (key, line) in keyed(template) {
        let (Some(key), Some(row)) = (key, row_cells(line)) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let mut merged = Vec::new();
        for (i, cell) in row.iter().enumerate() {
            // This cell's column over the runs.
            let column = |runs: &Rows| -> Vec<String> {
                (runs.get(&key).into_iter().flatten())
                    .map(|r| r.get(i).cloned().unwrap_or_default())
                    .collect()
            };
            let (b, a) = (column(&before), column(&after));
            merged.push(if i == 0 {
                cell.clone()
            } else if let (Some(b), Some(a)) = (medians(&b), medians(&a)) {
                change(b, a, cell)
            } else if let (Some(b), Some(a)) = (with_peaks(&b), with_peaks(&a)) {
                // A total and its peak, as the stutter bench's uploads.
                let (total, peak) = cell.split_once(" (").unwrap_or((cell, ""));
                let peak = peak.trim_end_matches(')');
                format!(
                    "{} (peak {})",
                    change(b.0, a.0, total),
                    change(b.1, a.1, peak)
                )
            } else if let (Some(b), Some(a)) = (ranges(&b), ranges(&a)) {
                let digits = precision(cell);
                format!(
                    "{:.digits$}–{:.digits$} → {:.digits$}–{:.digits$}",
                    b.0, b.1, a.0, a.1
                )
            } else {
                cell.clone()
            });
        }
        writeln!(out, "| {} |", merged.join(" | ")).expect("a String");
    }
    out
}

/// A table row's key: its table's index, its first cell and how many
/// rows before it in the table have that first cell.
type Key = (usize, String, usize);

/// Report table rows by [`Key`], each run's.
type Rows = HashMap<Key, Vec<Vec<String>>>;

/// The median of numeric cells, if every one is a number.
fn medians(cells: &[String]) -> Option<f64> {
    let mut values = cells
        .iter()
        .map(|c| number(c))
        .collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| median(&mut values))
}

/// The medians of `total (peak)` cells, if every one is one.
fn with_peaks(cells: &[String]) -> Option<(f64, f64)> {
    let (mut totals, mut peaks) = (Vec::new(), Vec::new());
    for cell in cells {
        let (total, peak) = cell.split_once(" (")?;
        totals.push(number(total)?);
        peaks.push(number(peak.strip_suffix(')')?)?);
    }
    (!totals.is_empty()).then(|| (median(&mut totals), median(&mut peaks)))
}

/// The range over `min–max` cells, if every one is one.
fn ranges(cells: &[String]) -> Option<(f64, f64)> {
    let bounds = cells.iter().map(|c| {
        let (lo, hi) = c.trim_matches('*').split_once('–')?;
        Some((number(lo)?, number(hi)?))
    });
    bounds
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .reduce(|(lo, hi), (a, b)| (lo.min(a), hi.max(b)))
}

/// The digits after the point in a numeric or `min–max` cell.
fn precision(cell: &str) -> usize {
    let first = cell.trim_matches('*').split('–').next().unwrap_or_default();
    first.split_once('.').map_or(0, |(_, f)| f.len())
}

/// `before → after (±x%)`, at the cell's own precision.
fn change(before: f64, after: f64, cell: &str) -> String {
    let digits = precision(cell);
    let percent = if before == 0.0 {
        String::new()
    } else {
        format!(" ({:+.1}%)", (after - before) / before * 100.0)
    };
    format!("{before:.digits$} → {after:.digits$}{percent}")
}

/// Each report's table rows by [`Key`].
fn cells(reports: &[String]) -> Rows {
    let mut rows: HashMap<_, Vec<_>> = HashMap::new();
    for report in reports {
        for (key, line) in keyed(report) {
            if let (Some(key), Some(row)) = (key, row_cells(line)) {
                rows.entry(key).or_default().push(row);
            }
        }
    }
    rows
}

/// The report's lines, each table data row with its [`Key`].
fn keyed(report: &str) -> impl Iterator<Item = (Option<Key>, &str)> {
    let mut seen: HashMap<(usize, String), usize> = HashMap::new();
    tables(report).map(move |(table, line)| {
        let first = row_cells(line).map(|row| row[0].clone());
        let key = table.zip(first).map(|(table, first)| {
            let repeat = seen.entry((table, first.clone())).or_default();
            *repeat += 1;
            (table, first, *repeat - 1)
        });
        (key, line)
    })
}

/// The report's lines, each table row with its table's index.
fn tables(report: &str) -> impl Iterator<Item = (Option<usize>, &str)> {
    let mut table = 0;
    let mut previous = false;
    report.lines().map(move |line| {
        let row = line.starts_with('|');
        if row && !previous {
            table += 1;
        }
        previous = row;
        (row.then_some(table), line)
    })
}

/// A table data row's cells; none for a separator.
fn row_cells(line: &str) -> Option<Vec<String>> {
    let inner = line.trim().strip_prefix('|')?.strip_suffix('|')?;
    if inner.chars().all(|c| matches!(c, '-' | '|' | ':' | ' ')) {
        return None;
    }
    Some(inner.split('|').map(|c| c.trim().to_string()).collect())
}

fn number(cell: &str) -> Option<f64> {
    cell.trim_matches('*')
        .parse()
        .ok()
        .filter(|n: &f64| n.is_finite())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn path(p: &Path) -> Result<&str, String> {
    p.to_str()
        .ok_or_else(|| format!("{}: not UTF-8", p.display()))
}

fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_tables_by_row() {
        let report = |gpu: &str, cpu: &str| {
            format!(
                "### Frame bench\n\n- data\n\n| view | GPU mean | draws |\n|---|---|---|\n| z12 | {gpu} | 40 |\n| z15 | {cpu} | n/a |\n\nTimes in ms.\n"
            )
        };
        let before = [
            report("2.000", "1.0"),
            report("4.000", "3.0"),
            report("3.000", "2.0"),
        ];
        let after = [
            report("1.000", "2.0"),
            report("1.500", "2.0"),
            report("2.000", "2.0"),
        ];
        let merged = merge(&before, &after);
        assert!(
            merged.contains("| z12 | 3.000 → 1.500 (-50.0%) | 40 → 40 (+0.0%) |"),
            "{merged}"
        );
        assert!(
            merged.contains("| z15 | 2.0 → 2.0 (+0.0%) | n/a |"),
            "{merged}"
        );
        assert!(merged.contains("|---|---|---|\n"));
        assert!(merged.starts_with("### Frame bench\n\n- data\n"));
    }

    /// Rows sharing a first cell, as the tile bench's mvt and mlt rows of
    /// one tile do, each merge with their own.
    #[test]
    fn keeps_rows_with_one_first_cell_apart() {
        let report = |mvt: &str, mlt: &str| {
            format!(
                "| tile | encoding | ms |\n|---|---|---|\n| 0/0/0 | mvt | {mvt} |\n| 0/0/0 | mlt | {mlt} |\n"
            )
        };
        let merged = merge(&[report("2.0", "8.0")], &[report("1.0", "4.0")]);
        assert!(
            merged.contains("| 0/0/0 | mvt | 2.0 → 1.0 (-50.0%) |"),
            "{merged}"
        );
        assert!(
            merged.contains("| 0/0/0 | mlt | 8.0 → 4.0 (-50.0%) |"),
            "{merged}"
        );
    }

    #[test]
    fn merges_totals_with_their_peaks() {
        let report = |cell: &str| format!("| path | MiB (peak) |\n|---|---|\n| a | {cell} |\n");
        let merged = merge(
            &[report("800.0 (20.0)"), report("900.0 (30.0)")],
            &[report("400.0 (10.0)"), report("400.0 (10.0)")],
        );
        assert!(
            merged.contains("| a | 850.0 → 400.0 (-52.9%) (peak 25.0 → 10.0 (-60.0%)) |"),
            "{merged}"
        );
    }

    #[test]
    fn merges_ranges_over_runs() {
        let report = |range: &str| format!("| view | min–max |\n|---|---|\n| z12 | {range} |\n");
        let merged = merge(
            &[report("1.0–3.0"), report("0.5–2.0")],
            &[report("2.0–2.5"), report("1.5–4.0")],
        );
        assert!(merged.contains("| z12 | 0.5–3.0 → 1.5–4.0 |"), "{merged}");
    }
}
