//! MapLibre Native's expression suite
//! (`fixtures/native/integration/expression-tests`): each `test.json`
//! through orb-expr's conformance runner with Native's expectations
//! (`orb_expr::conformance::Reference::Native`), honouring the manifest's
//! ignore lists as the render suites do. Native's runner also checks each
//! expression's serialization and a round trip through it; Orbis doesn't
//! serialize expressions, so those checks are left out, as they are for
//! the style-spec suite.

use std::path::Path;

use orb_expr::conformance::{Reference, check};
use serde_json::json;

use crate::native::{Manifest, mirror};
use crate::tally::Tally;

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
    let mut tally = Tally::start(root.join("docs/scratch/rt/native-expression"))?;
    let mut paths = Vec::new();
    find(&dir, &mut paths)?;
    paths.sort();
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
                    println!("FAIL {message}");
                    result["failure"] = json!(message);
                    "fail"
                }
            }
        };
        result["status"] = json!(status);
        let rel = id.strip_prefix("expression-tests/").unwrap_or(&id);
        tally.add(rel.split('/').next().unwrap_or(rel), result);
    }
    tally.print("native-expression");
    let report = tally.summary_path();
    tally.finish(json!({"suite": "native-expression", "filter": filter}))?;
    println!("\nReport: {}", report.display());
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
