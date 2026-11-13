//! The complete style-spec expression bundle, independent of render fixtures.
use std::collections::BTreeMap;
use std::io::Read;

use flate2::read::GzDecoder;
use tar::Archive;

// The version in the pinned GL JS package-lock.json, rather than style-spec's
// moving default branch. Keep SOURCE.md and the count assertion in sync.
const COMMIT: &str = "c25cc5da789337fca2df8d0eaaeeecd5eea7f634";
const TESTS: &str = "test/integration/expression/tests/";
pub(crate) fn bundle() -> Result<(), String> {
    let url = format!("https://codeload.github.com/maplibre/maplibre-style-spec/tar.gz/{COMMIT}");
    let mut response = ureq::get(&url).call().map_err(|e| e.to_string())?;
    let mut archive = Archive::new(GzDecoder::new(response.body_mut().as_reader()));
    let mut tests = BTreeMap::new();
    let mut license = None;
    for entry in archive.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = String::from_utf8(entry.path_bytes().into_owned()).map_err(|e| e.to_string())?;
        let Some((_, path)) = path.split_once('/') else {
            continue;
        };
        if path == "LICENSE.txt" {
            let mut text = String::new();
            entry.read_to_string(&mut text).map_err(|e| e.to_string())?;
            license = Some(text);
        } else if let Some(name) = path
            .strip_prefix(TESTS)
            .and_then(|p| p.strip_suffix("/test.json"))
        {
            let mut text = String::new();
            entry.read_to_string(&mut text).map_err(|e| e.to_string())?;
            let json: Box<serde_json::value::RawValue> =
                serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
            tests.insert(name.to_owned(), json);
        }
    }
    if tests.len() != 583 {
        return Err(format!(
            "expected 583 expression fixtures, found {}",
            tests.len()
        ));
    }
    let out = crate::root().join("crates/orb-expr/tests/data");
    let mut json = serde_json::to_string_pretty(&tests).map_err(|e| e.to_string())?;
    json.push('\n');
    std::fs::write(out.join("expression-fixtures.json"), json).map_err(|e| e.to_string())?;
    std::fs::write(
        out.join("LICENSE-style-spec.txt"),
        license.ok_or("missing upstream license")?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!("bundled {} expression fixtures at {COMMIT}", tests.len());
    Ok(())
}
