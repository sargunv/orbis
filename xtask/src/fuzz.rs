//! `cargo xtask fuzz`: runs one of orb-fuzz's targets under cargo-fuzz
//! (libFuzzer), with debug assertions and overflow checks on and no
//! sanitizer (the crates it reaches forbid `unsafe`). Its corpus grows in
//! `target/fuzz/corpus/<target>`, seeded from the fixture mirrors; inputs
//! that panic land in `target/fuzz/artifacts/<target>`.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::root;

/// Each target and the files under `fixtures/` (and `web/demo/data` and
/// orb-text's test fonts) it is
/// seeded with: their extensions, and a byte put before each one for the
/// targets that read options from their first byte.
const TARGETS: &[(&str, &[&str], Option<u8>)] = &[
    ("style", &["json"], None),
    ("mvt", &["mvt", "pbf"], Some(14)),
    ("mlt", &["mlt"], Some(14)),
    ("glyphs", &["pbf"], None),
    ("font", &["ttf", "otf", "woff", "woff2"], None),
    ("geojson", &["geojson"], None),
    ("dem", &[], None),
    ("packed", &[], None),
    ("sqlite", &["mbtiles", "db", "sqlite"], None),
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
    for source in [
        "fixtures/gl-js",
        "fixtures/native",
        "fixtures/mln-ffi",
        "web/demo/data",
        "crates/orb-text/tests/data/fonts",
    ] {
        seed(
            &root.join(source),
            &seeds,
            (extensions, prefix, name),
            &mut count,
        )?;
    }
    if name == "sqlite" {
        sqlite_seeds(&seeds, &mut count)?;
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

/// Writes SQLite files the fixtures lack into `seeds`, on 1 KiB pages so
/// that their trees have interior pages and their tiles overflow: MBTiles
/// in both layouts, and a MapLibre Native offline database.
fn sqlite_seeds(seeds: &Path, count: &mut usize) -> Result<(), String> {
    let tile =
        |z: u32, x: u32| -> Vec<u8> { (0..3000).map(|i| (i * 7 + z * 31 + x) as u8).collect() };
    let schemas = [
        (
            "tiles.mbtiles",
            "CREATE TABLE metadata (name text, value text);
             CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
             CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);
             INSERT INTO metadata VALUES ('format', 'pbf'), ('minzoom', '0'), ('maxzoom', '3'),
               ('json', '{\"vector_layers\":[{\"id\":\"x\"}]}');",
            "INSERT INTO tiles VALUES (?1, ?2, ?3, ?4)",
        ),
        (
            "deduplicated.mbtiles",
            "CREATE TABLE metadata (name text, value text);
             CREATE TABLE map (zoom_level integer, tile_column integer, tile_row integer, tile_id text);
             CREATE TABLE images (tile_data blob, tile_id text);
             CREATE UNIQUE INDEX map_index ON map (zoom_level, tile_column, tile_row);
             CREATE UNIQUE INDEX images_id ON images (tile_id);
             CREATE VIEW tiles AS SELECT zoom_level, tile_column, tile_row, tile_data
               FROM map JOIN images ON images.tile_id = map.tile_id;
             INSERT INTO metadata VALUES ('format', 'png');",
            "INSERT INTO map VALUES (?1, ?2, ?3, ?1 || '/' || ?2 || '/' || ?3);
             INSERT INTO images VALUES (?4, ?1 || '/' || ?2 || '/' || ?3);",
        ),
        (
            "offline.db",
            "CREATE TABLE regions (id INTEGER PRIMARY KEY AUTOINCREMENT, definition TEXT NOT NULL, description BLOB);
             CREATE TABLE resources (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL UNIQUE,
               kind INTEGER NOT NULL, expires INTEGER, modified INTEGER, etag TEXT, data BLOB,
               compressed INTEGER NOT NULL DEFAULT 0, accessed INTEGER NOT NULL, must_revalidate INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE tiles (id INTEGER PRIMARY KEY AUTOINCREMENT, url_template TEXT NOT NULL,
               pixel_ratio INTEGER NOT NULL, z INTEGER NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,
               expires INTEGER, modified INTEGER, etag TEXT, data BLOB, compressed INTEGER NOT NULL DEFAULT 0,
               accessed INTEGER NOT NULL, must_revalidate INTEGER NOT NULL DEFAULT 0,
               UNIQUE (url_template, pixel_ratio, z, x, y));
             CREATE TABLE region_resources (region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
               resource_id INTEGER NOT NULL REFERENCES resources(id), UNIQUE (region_id, resource_id));
             CREATE TABLE region_tiles (region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
               tile_id INTEGER NOT NULL REFERENCES tiles(id), UNIQUE (region_id, tile_id));
             INSERT INTO regions VALUES (1, '{\"style_url\":\"https://x/s.json\",\"bounds\":[1,2,3,4],\"min_zoom\":0,\"max_zoom\":3,\"pixel_ratio\":1}', x'6d');
             INSERT INTO resources VALUES (1, 'https://x/s.json', 1, 0, 0, NULL, x'7b7d', 0, 0, 0);
             INSERT INTO region_resources VALUES (1, 1);",
            "INSERT INTO tiles (url_template, pixel_ratio, z, x, y, data, accessed) VALUES ('https://x/{z}/{x}/{y}.pbf', 1, ?1, ?2, ?3, ?4, 0);
             INSERT INTO region_tiles VALUES (1, last_insert_rowid());",
        ),
    ];
    for (name, schema, insert) in schemas {
        let path = seeds.join(name);
        let fail = |e: rusqlite::Error| format!("{}: {e}", path.display());
        let db = rusqlite::Connection::open(&path).map_err(fail)?;
        db.execute_batch("PRAGMA page_size = 1024;").map_err(fail)?;
        db.execute_batch(schema).map_err(fail)?;
        for z in 0..4u32 {
            for x in 0..1u32 << z {
                for y in 0..1u32 << z {
                    let data = tile(z, x);
                    let values: [&dyn rusqlite::ToSql; 4] = [&z, &x, &y, &data];
                    for statement in insert.split(';').filter(|s| !s.trim().is_empty()) {
                        let mut statement = db.prepare(statement).map_err(fail)?;
                        // Each statement takes the first of `?1`…`?4` it numbers.
                        for (i, value) in
                            values.iter().enumerate().take(statement.parameter_count())
                        {
                            statement.raw_bind_parameter(i + 1, value).map_err(fail)?;
                        }
                        statement.raw_execute().map_err(fail)?;
                    }
                }
            }
        }
        *count += 1;
    }
    Ok(())
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
