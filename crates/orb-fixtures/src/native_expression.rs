//! MapLibre Native's expression suite
//! (`fixtures/native/integration/expression-tests`): each `test.json`
//! through orb-expr's conformance runner with Native's expectations
//! (`orb_expr::conformance::Reference::Native`), honouring the manifest's
//! ignore lists as the render suites do. Native's runner also checks each
//! expression's serialization and a round trip through it; Orbis doesn't
//! serialize expressions, so those checks are left out, as they are for
//! the style-spec suite.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use orb_expr::conformance::{Reference, check};
use serde_json::json;

use crate::native::{Manifest, mirror};
use crate::{Counts, write_json};

pub(crate) fn run(root: &Path, filter: &str) -> Result<(), String> {
    let manifest = Manifest::read(root)?;
    let base = mirror(root).join("integration");
    let dir = base.join("expression-tests");
    if !dir.is_dir() {
        return Err(format!(
            "{} is missing; run `cargo xtask fixtures native` to fetch the Native fixture mirror",
            dir.display()
        ));
    }
    let started = Instant::now();
    let mut paths = Vec::new();
    find(&dir, &mut paths)?;
    paths.sort();
    let mut totals = Counts::default();
    let mut ignored = Counts::default();
    let mut groups: BTreeMap<String, Counts> = BTreeMap::new();
    let mut results = Vec::new();
    for path in paths {
        let id = path
            .parent()
            .expect("test parent")
            .strip_prefix(&base)
            .expect("under the mirror")
            .to_string_lossy()
            .replace('\\', "/");
        if !crate::matches(&id, filter) {
            continue;
        }
        let mut result = json!({"id": id});
        let reason = manifest.ignores.get(&id);
        if let Some(reason) = reason {
            result["native_ignored"] = json!(reason);
        }
        let status = if reason.is_some_and(|r| r.starts_with("skip")) {
            result["skip_reasons"] = json!(["native:skip"]);
            "skip"
        } else {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{id}: {e}"))?;
            let fixture = orb_expr::json::parse(&text).map_err(|e| format!("{id}: {e:?}"))?;
            // A fixture the runner can't read fails, rather than ending the
            // suite.
            let outcome = std::panic::catch_unwind(|| check(&id, &fixture, Reference::Native))
                .unwrap_or_else(|_| Err(format!("{id}: the runner panicked")));
            match outcome {
                Ok(()) => "pass",
                Err(message) => {
                    result["failure"] = json!(message);
                    "fail"
                }
            }
        };
        result["status"] = json!(status);
        totals.add(status);
        if reason.is_some() {
            ignored.add(status);
        }
        let rel = id.strip_prefix("expression-tests/").unwrap_or(&id);
        groups
            .entry(rel.split('/').next().unwrap_or(rel).to_string())
            .or_default()
            .add(status);
        results.push(result);
    }
    let output = root.join("docs/scratch/rt/native-expression");
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let summary = json!({
        "suite": "native-expression",
        "filter": filter,
        "total_seconds": started.elapsed().as_secs_f64(),
        "totals": totals.json(),
        "native_ignored": ignored.json(),
        "groups": groups.iter().map(|(g, c)| (g.clone(), c.json())).collect::<BTreeMap<_, _>>(),
        "tests": results,
    });
    write_json(&output.join("summary.json"), &summary)?;
    for test in summary["tests"].as_array().expect("tests") {
        if test["status"] == "fail" {
            println!("FAIL {}", test["failure"].as_str().unwrap_or_default());
        }
    }
    println!("\nnative-expression  group           pass  fail  skip  in-scope pass rate");
    for (group, counts) in &groups {
        print!("{:<18} ", "");
        counts.print(group);
    }
    print!("{:<18} ", "");
    totals.print("TOTAL");
    println!(
        "  Native ignores {} of them: {} pass, {} fail, {} skip",
        ignored.pass + ignored.fail + ignored.skip,
        ignored.pass,
        ignored.fail,
        ignored.skip
    );
    println!("\nReport: {}", output.join("summary.json").display());
    Ok(())
}

fn find(dir: &Path, paths: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            find(&entry.path(), paths)?;
        } else if entry.file_name() == "test.json" {
            paths.push(entry.path());
        }
    }
    Ok(())
}
