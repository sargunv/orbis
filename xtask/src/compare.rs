//! A/B runs against another revision: `bench --compare <rev>` builds
//! `orb-bench` at `<rev>` in a temporary worktree and runs it interleaved
//! with this tree's, and `h2h --compare <rev>` adds that revision's
//! `orbis` as an engine. Builds share `target/compare/target`, so a second
//! comparison rebuilds only what changed.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A revision's build in a worktree under `target/compare/`, removed when
/// this drops.
pub(crate) struct Build {
    /// The revision as given.
    pub(crate) rev: String,
    pub(crate) commit: String,
    worktree: PathBuf,
    /// The build's binaries: `target/compare/target/<profile>`.
    pub(crate) bin: PathBuf,
}

impl Build {
    /// Checks `rev` out into a worktree and builds `packages` there in `profile`.
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
        let worktree = compare.join(format!("src-{commit}-{}", std::process::id()));
        std::fs::create_dir_all(&compare).map_err(|e| format!("{}: {e}", compare.display()))?;
        eprintln!("compare: {rev} ({commit}) into {}", worktree.display());
        git(
            &root,
            &["worktree", "add", "--detach", path(&worktree)?, &commit],
        )?;
        let build = Self {
            rev: rev.to_string(),
            commit,
            worktree,
            bin: compare.join("target").join(profile),
        };
        let manifest = build.worktree.join("Cargo.toml");
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
        Ok(build)
    }
}

impl Drop for Build {
    fn drop(&mut self) {
        let root = root();
        if let Err(e) = git(
            &root,
            &[
                "worktree",
                "remove",
                "--force",
                &self.worktree.display().to_string(),
            ],
        ) {
            eprintln!("compare: {e}; remove {} by hand", self.worktree.display());
        }
    }
}

/// `cargo xtask bench --compare <rev> [--runs N] [orb-bench args]`: `runs`
/// rounds of `orb-bench` at `rev` and at this tree, alternating which goes
/// first, then every table of the report with each number as the median
/// of `rev`'s runs → this tree's, and the change.
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
    let before = Build::new(rev, &["orb-bench"], "release")?;
    crate::cargo(&["build", "--release", "-p", "orb-bench"])?;
    let after = root().join("target/release/orb-bench");
    let mut outputs: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    for run in 0..runs {
        let order = if run % 2 == 0 { [0, 1] } else { [1, 0] };
        for side in order {
            let (name, bin) = match side {
                0 => (before.rev.as_str(), before.bin.join("orb-bench")),
                _ => ("working tree", after.clone()),
            };
            eprintln!("compare: run {} of {runs}: {name}", run + 1);
            let out = Command::new(&bin)
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
        "## Bench: {} ({}) → working tree, {runs} runs interleaved\n\n{}",
        before.rev,
        before.commit,
        merge(&outputs[0], &outputs[1])
    );
    Ok(())
}

/// The last of `after`'s reports, each numeric table cell replaced by the
/// median over `before`'s reports → the median over `after`'s and the
/// change. A row is matched by its table and first cell.
fn merge(before: &[String], after: &[String]) -> String {
    let template = after.last().map_or("", String::as_str);
    let (before, after) = (cells(before), cells(after));
    let mut out = String::new();
    for (table, line) in tables(template) {
        let row = row_cells(line);
        let Some((table, first)) = table.zip(row.as_ref().and_then(|r| r.first())) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let key = (table, first.clone());
        let mut merged = Vec::new();
        for (i, cell) in row.as_ref().into_iter().flatten().enumerate() {
            let column = |runs: &HashMap<(usize, String), Vec<Vec<String>>>| {
                let values: Option<Vec<f64>> = (runs.get(&key).into_iter().flatten())
                    .map(|r| r.get(i).and_then(|c| number(c)))
                    .collect();
                values.filter(|v| !v.is_empty()).map(|mut v| median(&mut v))
            };
            merged.push(match (i, column(&before), column(&after)) {
                (1.., Some(b), Some(a)) => change(b, a, cell),
                _ => cell.clone(),
            });
        }
        writeln!(out, "| {} |", merged.join(" | ")).expect("a String");
    }
    out
}

/// `before → after (±x%)`, at the cell's own precision.
fn change(before: f64, after: f64, cell: &str) -> String {
    let digits = cell
        .trim_matches('*')
        .split_once('.')
        .map_or(0, |(_, f)| f.len());
    let percent = if before == 0.0 {
        String::new()
    } else {
        format!(" ({:+.1}%)", (after - before) / before * 100.0)
    };
    format!("{before:.digits$} → {after:.digits$}{percent}")
}

/// Each report's table rows by table index and first cell.
fn cells(reports: &[String]) -> HashMap<(usize, String), Vec<Vec<String>>> {
    let mut rows: HashMap<_, Vec<_>> = HashMap::new();
    for report in reports {
        for (table, line) in tables(report) {
            if let (Some(table), Some(row)) = (table, row_cells(line)) {
                rows.entry((table, row[0].clone())).or_default().push(row);
            }
        }
    }
    rows
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

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .to_path_buf()
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
}
