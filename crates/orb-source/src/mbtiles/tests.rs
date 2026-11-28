use rusqlite::{Connection, params};

use super::*;

/// An archive `fill` writes, as SQLite wrote it to disk.
pub(crate) fn written(fill: impl FnOnce(&Connection)) -> Vec<u8> {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("orb-mbtiles-test-{}-{n}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    fill(&Connection::open(&path).unwrap());
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    bytes
}

fn body(z: u8, x: u32, y: u32) -> Vec<u8> {
    format!("{z}/{x}/{y}")
        .into_bytes()
        .repeat(1 + (x as usize % 5) * 300)
}

fn reader() -> crate::archive::Reader {
    crate::archive::Reader::Mbtiles(Reader::default())
}

fn tilejson(file: &[u8]) -> Result<TileJson, String> {
    reader().read_in(file, crate::archive::Reader::tilejson)
}

fn flat(gzip: bool, metadata: bool, index: bool) -> Vec<u8> {
    written(|db| {
        db.execute_batch(
            "CREATE TABLE metadata (name text, value text);
             CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);",
        )
        .unwrap();
        if index {
            db.execute_batch(
                "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);",
            )
            .unwrap();
        }
        if metadata {
            db.execute_batch(
                r#"INSERT INTO metadata VALUES ('format', 'pbf'), ('minzoom', '1'), ('maxzoom', '4'),
                   ('bounds', '-10.5,-20,30,40.25'), ('attribution', '<a href="x">"x"</a>'),
                   ('json', '{"vector_layers":[{"id":"water","fields":{}}]}');"#,
            )
            .unwrap();
        }
        for z in 2..5u8 {
            for x in 0..(1u32 << z) {
                for y in 0..(1u32 << z) {
                    let tms = (1 << z) - 1 - y;
                    let data = body(z, x, y);
                    let data = if gzip {
                        crate::gzip::encode(&data)
                    } else {
                        data
                    };
                    db.execute(
                        "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                        params![z, x, tms, data],
                    )
                    .unwrap();
                }
            }
        }
    })
}

#[test]
fn reads_an_archives_tilejson_and_tiles_by_xyz() {
    for gzip in [false, true] {
        let file = flat(gzip, true, true);
        assert_eq!(
            tilejson(&file).unwrap(),
            TileJson {
                minzoom: Some(1.0),
                maxzoom: Some(4.0),
                bounds: Some([-10.5, -20.0, 30.0, 40.25]),
                encoding: Some(Encoding::Mvt),
                ..TileJson::default()
            }
        );
        let mut reader = reader();
        for (z, x, y) in [(2, 0, 0), (3, 5, 1), (4, 15, 3), (4, 9, 14)] {
            let tile = reader.tile_in(&file, TileIndex::new(z, x, y)).unwrap();
            assert_eq!(
                tile,
                Some((body(z, x, y), Some(Encoding::Mvt))),
                "{z}/{x}/{y}"
            );
        }
        let missing = reader.tile_in(&file, TileIndex::new(1, 0, 0)).unwrap();
        assert_eq!(missing, None);
    }
}

#[test]
fn zooms_the_metadata_leaves_out_come_from_the_tiles() {
    let tilejson = tilejson(&flat(false, false, true)).unwrap();
    assert_eq!((tilejson.minzoom, tilejson.maxzoom), (Some(2.0), Some(4.0)));
}

#[test]
fn archives_without_a_tile_index_are_refused() {
    let error = tilejson(&flat(false, true, false)).unwrap_err();
    assert!(
        error.contains("MBTiles table tiles has no index"),
        "{error}"
    );
}

#[test]
fn reads_deduplicated_archives() {
    let file = written(|db| {
        db.execute_batch(
            "CREATE TABLE map (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_id TEXT);
             CREATE UNIQUE INDEX map_index ON map (zoom_level, tile_column, tile_row);
             CREATE TABLE images (tile_data blob, tile_id text);
             CREATE UNIQUE INDEX images_id ON images (tile_id);
             CREATE TABLE metadata (name text, value text);
             CREATE UNIQUE INDEX name ON metadata (name);
             CREATE VIEW tiles AS SELECT map.zoom_level AS zoom_level, map.tile_column AS tile_column,
               map.tile_row AS tile_row, images.tile_data AS tile_data
               FROM map JOIN images ON images.tile_id = map.tile_id;
             INSERT INTO metadata VALUES ('format', 'png');
             INSERT INTO images VALUES (x'89504e47', 'ocean'), (x'89504e48', 'land');",
        )
        .unwrap();
        for x in 0..8u32 {
            for y in 0..8u32 {
                let id = if (x + y) % 3 == 0 { "land" } else { "ocean" };
                db.execute("INSERT INTO map VALUES (3, ?, ?, ?)", params![x, 7 - y, id])
                    .unwrap();
            }
        }
    });
    let tilejson = tilejson(&file).unwrap();
    assert_eq!((tilejson.minzoom, tilejson.encoding), (Some(3.0), None));
    let mut reader = reader();
    let tile = reader.tile_in(&file, TileIndex::new(3, 1, 2)).unwrap();
    assert_eq!(tile, Some((vec![0x89, 0x50, 0x4e, 0x48], None)));
    let tile = reader.tile_in(&file, TileIndex::new(3, 1, 1)).unwrap();
    assert_eq!(tile, Some((vec![0x89, 0x50, 0x4e, 0x47], None)));
}

#[test]
fn what_isnt_an_archive_says_so() {
    assert!(tilejson(&[7u8; 8192]).is_err());
    let file = written(|db| db.execute_batch("CREATE TABLE other (a);").unwrap());
    assert!(
        tilejson(&file)
            .unwrap_err()
            .contains("no MBTiles tiles table")
    );
}
