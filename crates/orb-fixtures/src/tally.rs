//! What every suite's driver shares: its report directory, emptied as the
//! suite starts, and the counts its tests add up to, written as its
//! `summary.json` and printed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};

use crate::write_json;

/// A suite's tests so far.
pub(crate) struct Tally {
    /// Where its reports go.
    pub(crate) output: PathBuf,
    started: Instant,
    pub(crate) totals: Counts,
    /// The counts by each test's prefix: its family or group.
    prefixes: BTreeMap<String, Counts>,
    /// How many tests skipped for each reason.
    skips: BTreeMap<String, usize>,
    /// The counts of the tests Native's manifest ignores.
    native_ignored: Counts,
    /// Tests whose frames asked for pipelines the style's set left out.
    late: usize,
    tests: Vec<Value>,
}

impl Tally {
    /// Starts a suite whose reports go to `output`, emptied of an earlier
    /// run's (the directory itself stays, for a container that mounts it).
    pub(crate) fn start(output: PathBuf) -> Result<Self, String> {
        if output.exists() {
            for entry in std::fs::read_dir(&output).map_err(|e| e.to_string())? {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)
                } else {
                    std::fs::remove_file(&path)
                }
                .map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        std::fs::create_dir_all(&output).map_err(|e| format!("{}: {e}", output.display()))?;
        Ok(Self {
            output,
            started: Instant::now(),
            totals: Counts::default(),
            prefixes: BTreeMap::new(),
            skips: BTreeMap::new(),
            native_ignored: Counts::default(),
            late: 0,
            tests: Vec::new(),
        })
    }

    /// Counts a test's `result` (its `status`, `skip_reasons` and
    /// `native_ignored`) under `prefix`, and keeps it for the summary.
    pub(crate) fn add(&mut self, prefix: &str, result: Value) {
        let status = result["status"].as_str().expect("our status");
        self.totals.add(status);
        self.prefixes.entry(prefix.into()).or_default().add(status);
        if result["native_ignored"].is_string() {
            self.native_ignored.add(status);
        }
        self.late += usize::from(result["pipelines_late"].is_u64());
        for reason in result["skip_reasons"].as_array().into_iter().flatten() {
            *self
                .skips
                .entry(reason.as_str().expect("our reason").into())
                .or_default() += 1;
        }
        self.tests.push(result);
    }

    /// Writes the summary: `fields` (an object), then the time taken, the
    /// counts, the skip reasons most common first, Native's ignored tests
    /// where the suite has any, and every test. Returns it.
    pub(crate) fn finish(self, mut fields: Value) -> Result<Value, String> {
        let mut skips: Vec<_> = self.skips.into_iter().collect();
        skips.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let summary = fields.as_object_mut().expect("an object");
        summary.insert(
            "total_seconds".into(),
            json!(self.started.elapsed().as_secs_f64()),
        );
        summary.insert("totals".into(), self.totals.json());
        summary.insert(
            "prefixes".into(),
            (self.prefixes.iter())
                .map(|(p, c)| (p.clone(), c.json()))
                .collect(),
        );
        summary.insert(
            "skip_reasons".into(),
            (skips.iter())
                .map(|(reason, count)| json!({"reason": reason, "count": count}))
                .collect(),
        );
        if self.native_ignored.any() {
            summary.insert("native_ignored".into(), self.native_ignored.json());
        }
        summary.insert("tests".into(), Value::Array(self.tests));
        write_json(&self.output.join("summary.json"), &fields)?;
        Ok(fields)
    }

    /// Prints the counts by prefix under `title`, then the totals and the
    /// tests Native ignores.
    pub(crate) fn print(&self, title: &str) {
        let indent = title.len() + 2;
        println!("\n{title}  prefix          pass  fail  skip  in-scope pass rate");
        for (prefix, counts) in &self.prefixes {
            print!("{:indent$}", "");
            counts.print(prefix);
        }
        print!("{:indent$}", "");
        self.totals.print("TOTAL");
        let ignored = &self.native_ignored;
        if ignored.any() {
            println!(
                "  Native ignores {} of them: {} pass, {} fail, {} skip",
                ignored.pass + ignored.fail + ignored.skip,
                ignored.pass,
                ignored.fail,
                ignored.skip
            );
        }
        if self.late > 0 {
            println!(
                "  {} asked for pipelines outside the style's set (LATE above)",
                self.late
            );
        }
    }

    /// Where the summary goes.
    pub(crate) fn summary_path(&self) -> PathBuf {
        self.output.join("summary.json")
    }
}

/// How many tests passed, failed and skipped.
#[derive(Default)]
pub(crate) struct Counts {
    pub(crate) pass: usize,
    pub(crate) fail: usize,
    pub(crate) skip: usize,
}

impl Counts {
    fn add(&mut self, status: &str) {
        match status {
            "pass" => self.pass += 1,
            "fail" => self.fail += 1,
            "skip" => self.skip += 1,
            _ => unreachable!(),
        }
    }

    fn any(&self) -> bool {
        self.pass + self.fail + self.skip > 0
    }

    fn rate(&self) -> Option<f64> {
        let scope = self.pass + self.fail;
        (scope != 0).then(|| self.pass as f64 / scope as f64)
    }

    fn json(&self) -> Value {
        json!({"pass":self.pass,"fail":self.fail,"skip":self.skip,"pass_rate":self.rate()})
    }

    /// The counts a summary gives.
    pub(crate) fn from_json(v: &Value) -> Self {
        let n = |key: &str| v[key].as_u64().unwrap_or(0) as usize;
        Self {
            pass: n("pass"),
            fail: n("fail"),
            skip: n("skip"),
        }
    }

    pub(crate) fn print(&self, name: &str) {
        let rate = self
            .rate()
            .map_or_else(|| "n/a".into(), |r| format!("{:.2}%", r * 100.0));
        println!(
            "{name:<15} {:4}  {:4}  {:4}  {rate:>8}",
            self.pass, self.fail, self.skip
        );
    }
}

/// The backend for `platform`: a browser (`chrome`, `chrome-webgl2`, on a
/// phone `android-chrome`, `android-chrome-webgl2`, and `firefox`,
/// `webkit`) or the native runtime, fetching `assets` as the suite's server
/// serves them.
pub(crate) fn backend(
    root: &Path,
    platform: &str,
    assets: &Path,
    kind: crate::fetch::Assets,
) -> Result<crate::Backend, String> {
    match platform {
        "chrome"
        | "chrome-webgl2"
        | "android-chrome"
        | "android-chrome-webgl2"
        | "firefox"
        | "webkit" => Ok(crate::Backend::Chrome(crate::chrome::Chrome::new(
            root,
            assets,
            kind,
            platform.ends_with("webgl2"),
            match platform {
                "firefox" | "webkit" => Some(platform),
                _ if platform.starts_with("android-") => Some("android"),
                _ => None,
            },
        )?)),
        _ => crate::runtime(assets, kind),
    }
}
