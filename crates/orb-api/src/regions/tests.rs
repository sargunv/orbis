use super::*;
use orb_host::{Abort, Fetch, ResourceKind};

/// A fresh directory under the system's temporary one.
fn temporary(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("orb-api-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// MapLibre Native's offline schema, with one region of a style, a glyph
/// range and two tiles; the glyphs compressed.
fn native_database(path: &std::path::Path) {
    let db = rusqlite::Connection::open(path).unwrap();
    db.execute_batch(
        "CREATE TABLE resources (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL,
           kind INTEGER NOT NULL, expires INTEGER, modified INTEGER, etag TEXT, data BLOB,
           compressed INTEGER NOT NULL DEFAULT 0, accessed INTEGER NOT NULL,
           must_revalidate INTEGER NOT NULL DEFAULT 0, UNIQUE (url));
         CREATE TABLE tiles (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, url_template TEXT NOT NULL,
           pixel_ratio INTEGER NOT NULL, z INTEGER NOT NULL, x INTEGER NOT NULL, y INTEGER NOT NULL,
           expires INTEGER, modified INTEGER, etag TEXT, data BLOB, compressed INTEGER NOT NULL DEFAULT 0,
           accessed INTEGER NOT NULL, must_revalidate INTEGER NOT NULL DEFAULT 0,
           UNIQUE (url_template, pixel_ratio, z, x, y));
         CREATE TABLE regions (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, definition TEXT NOT NULL,
           description BLOB);
         CREATE TABLE region_resources (region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
           resource_id INTEGER NOT NULL REFERENCES resources(id), UNIQUE (region_id, resource_id));
         CREATE TABLE region_tiles (region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
           tile_id INTEGER NOT NULL REFERENCES tiles(id), UNIQUE (region_id, tile_id));
         PRAGMA user_version = 6;",
    )
    .unwrap();
    let definition = r#"{"style_url":"https://x/style.json","bounds":[1.0,2.0,3.0,4.0],"min_zoom":0.0,"max_zoom":1.0,"pixel_ratio":2.0,"include_ideographs":true}"#;
    db.execute(
        "INSERT INTO regions (definition, description) VALUES (?, ?)",
        rusqlite::params![definition, vec![7u8, 8, 9]],
    )
    .unwrap();
    let style = br#"{"version":8,"sources":{},"layers":[]}"#.to_vec();
    db.execute(
        "INSERT INTO resources (url, kind, data, accessed) VALUES ('https://x/style.json', 1, ?, 0)",
        [style],
    )
    .unwrap();
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&[5u8; 2000], 6);
    db.execute(
        "INSERT INTO resources (url, kind, data, compressed, accessed) \
         VALUES ('https://x/0-255.pbf', 4, ?, 1, 0)",
        [compressed],
    )
    .unwrap();
    for (i, (template, z, x, y)) in [
        ("https://x/{z}/{x}/{y}.pbf", 1, 1, 0),
        ("https://x/{z}/{x}/{y}{ratio}.png", 1, 0, 1),
    ]
    .into_iter()
    .enumerate()
    {
        db.execute(
            "INSERT INTO tiles (url_template, pixel_ratio, z, x, y, data, accessed) \
             VALUES (?, 2, ?, ?, ?, ?, 0)",
            rusqlite::params![template, z, x, y, vec![i as u8 + 1; 300]],
        )
        .unwrap();
    }
    db.execute_batch(
        "INSERT INTO region_resources VALUES (1, 1), (1, 2);
         INSERT INTO region_tiles VALUES (1, 1), (1, 2);",
    )
    .unwrap();
}

fn fetched(regions: &Regions, url: &str) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    regions.fetch.get(
        ResourceKind::Tile,
        url,
        &Abort::new(),
        Box::new(move |r| tx.send(r).unwrap()),
    );
    rx.recv().unwrap().unwrap()
}

#[test]
fn merges_a_native_database_once() {
    let dir = temporary("merge");
    let side = dir.join("side.db");
    native_database(&side);
    let before = std::fs::read(&side).unwrap();
    let regions = Regions::new(&dir.join("state"), NativeFetch::new());
    let merged = regions.merge(&side).unwrap();
    assert_eq!(merged.len(), 1);
    let region = &merged[0];
    assert!(region.complete);
    let d = &region.definition;
    assert_eq!(d.style_url, "https://x/style.json");
    assert_eq!(d.bounds, [2.0, 1.0, 4.0, 3.0]);
    assert_eq!(
        (d.zooms, d.pixel_ratio, d.include_ideographs),
        ([0.0, 1.0], 2.0, true)
    );
    assert_eq!(d.metadata, [7, 8, 9]);
    assert_eq!(regions.list().unwrap(), merged);
    // Served offline: resources by URL, tiles from the extract or by URL.
    assert_eq!(fetched(&regions, "https://x/0-255.pbf"), vec![5u8; 2000]);
    assert_eq!(fetched(&regions, "https://x/1/1/0.pbf"), vec![1u8; 300]);
    assert_eq!(fetched(&regions, "https://x/1/0/1@2x.png"), vec![2u8; 300]);
    // Merged again, it's the same region.
    assert_eq!(regions.merge(&side).unwrap(), merged);
    assert_eq!(regions.list().unwrap().len(), 1);
    assert_eq!(std::fs::read(&side).unwrap(), before);
    // An Orbis state directory merges too.
    let other = Regions::new(&dir.join("other"), NativeFetch::new());
    let copied = other.merge(&dir.join("state")).unwrap();
    assert_eq!(copied[0].definition, *d);
    assert_eq!(fetched(&other, "https://x/1/1/0.pbf"), vec![1u8; 300]);
    assert!(regions.merge(&dir.join("missing.db")).is_err());
    std::fs::write(dir.join("bad.db"), b"SQLite format 3\0 not really").unwrap();
    assert!(regions.merge(&dir.join("bad.db")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
