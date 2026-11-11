//! `cargo xtask fuzz`: runs one of orb-fuzz's targets under cargo-fuzz
//! (libFuzzer), with debug assertions and overflow checks on and no
//! sanitizer (the crates it reaches forbid `unsafe`). Its corpus grows in
//! `target/fuzz/corpus/<target>`, seeded from the fixture mirrors; inputs
//! that panic land in `target/fuzz/artifacts/<target>`.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::root;

/// Each target and the files under `fixtures/` (and `web/demo/data`) it is
/// seeded with: their extensions, and a byte put before each one for the
/// targets that read options from their first byte.
const TARGETS: &[(&str, &[&str], Option<u8>)] = &[
    ("style", &["json"], None),
    ("mvt", &["mvt", "pbf"], Some(14)),
    ("mlt", &["mlt"], Some(14)),
    ("glyphs", &["pbf"], None),
    ("geojson", &["geojson"], None),
    ("dem", &[], None),
];

pub(crate) fn run(args: &[String]) -> Result<(), String> {
    let usage = || {
        let names: Vec<_> = TARGETS.iter().map(|t| t.0).collect();
        format!(
            "usage: cargo xtask fuzz <{}> [--seconds n]",
            names.join("|")
        )
    };
    let name = args.first().ok_or_else(usage)?;
    let &(name, extensions, prefix) = TARGETS.iter().find(|t| t.0 == name).ok_or_else(usage)?;
    let seconds = match args.iter().position(|a| a == "--seconds") {
        Some(i) => args.get(i + 1).ok_or_else(usage)?.as_str(),
        None => "600",
    };
    crate::fixtures::ensure()?;
    crate::native_fixtures::ensure()?;
    let root = root();
    let dir = root.join("target/fuzz");
    let corpus = dir.join("corpus").join(name);
    let seeds = dir.join("seeds").join(name);
    let artifacts = dir.join("artifacts").join(name);
    if seeds.exists() {
        std::fs::remove_dir_all(&seeds).map_err(|e| format!("{}: {e}", seeds.display()))?;
    }
    for d in [&corpus, &seeds, &artifacts] {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut count = 0;
    for source in ["fixtures/gl-js", "fixtures/native", "web/demo/data"] {
        seed(
            &root.join(source),
            &seeds,
            (extensions, prefix, name),
            &mut count,
        )?;
    }
    eprintln!("fuzz {name}: {count} seeds in {}", seeds.display());
    crate::run(
        Command::new("cargo")
            .current_dir(&root)
            .args(["fuzz", "run", "--fuzz-dir", "crates/orb-fuzz"])
            // No `cfg(fuzzing)`: mlt-core compiles its own fuzzing module under
            // it, which needs its `arbitrary` feature.
            .args([
                "--sanitizer",
                "none",
                "--debug-assertions",
                "--no-cfg-fuzzing",
            ])
            .arg(name)
            .args([&corpus, &seeds])
            .arg("--")
            .arg(format!("-max_total_time={seconds}"))
            .arg(format!("-artifact_prefix={}/", artifacts.display())),
    )
}

/// Copies the files under `dir` with one of `extensions` into `seeds`,
/// after `prefix` when given, counting them in `count`. Glyph ranges sit
/// under a `glyphs` directory, and a vector tile's `.pbf` anywhere else.
fn seed(
    dir: &Path,
    seeds: &Path,
    (extensions, prefix, target): (&[&str], Option<u8>, &str),
    count: &mut usize,
) -> Result<(), String> {
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let glyphs = path.components().any(|c| c.as_os_str() == "glyphs");
            let wanted = extensions.contains(&extension)
                && match (target, extension) {
                    ("glyphs", _) => glyphs,
                    ("mvt", "pbf") => !glyphs,
                    _ => true,
                };
            if !wanted {
                continue;
            }
            let mut bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if let Some(byte) = prefix {
                bytes.insert(0, byte);
            }
            let to = seeds.join(format!("{count:05}.{extension}"));
            std::fs::write(&to, bytes).map_err(|e| format!("{}: {e}", to.display()))?;
            *count += 1;
        }
    }
    Ok(())
}
