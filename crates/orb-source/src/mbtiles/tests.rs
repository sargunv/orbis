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

/// Answers `step` from `file` until it's done, and counts its reads.
fn answer<T>(
    file: &[u8],
    reader: &mut Reader,
    mut step: impl FnMut(&mut Reader) -> Answer<T>,
) -> (T, usize) {
    for reads in 0..64 {
        match step(reader).unwrap() {
            Step::Done(value) => return (value, reads),
            Step::Read { offset, length } => {
                let (at, end) = (offset as usize, (offset + length) as usize);
                reader
                    .insert(offset, &file[at.min(file.len())..end.min(file.len())])
                    .unwrap();
            }
        }
    }
    panic!("the reader never answered");
}

fn flat(gzip: bool, metadata: bool) -> Vec<u8> {
    written(|db| {
        db.execute_batch(
            "CREATE TABLE metadata (name text, value text);
             CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
             CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);",
        )
        .unwrap();
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
        let file = flat(gzip, true);
        let mut reader = Reader::default();
        let (tilejson, _) = answer(&file, &mut reader, Reader::tilejson);
        let doc = orb_expr::json::parse(&tilejson).unwrap();
        assert_eq!(doc.get("minzoom").and_then(|v| v.as_f64()), Some(1.0));
        assert_eq!(doc.get("maxzoom").and_then(|v| v.as_f64()), Some(4.0));
        assert_eq!(doc.get("encoding").and_then(|v| v.as_str()), Some("mvt"));
        assert_eq!(
            doc.get("attribution").and_then(|v| v.as_str()),
            Some(r#"<a href="x">"x"</a>"#)
        );
        assert!(
            tilejson.contains(r#""bounds":[-10.5,-20,30,40.25]"#),
            "{tilejson}"
        );
        assert!(
            tilejson.contains(r#""vector_layers":[{"id":"water""#),
            "{tilejson}"
        );
        for (z, x, y) in [(2, 0, 0), (3, 5, 1), (4, 15, 3), (4, 9, 14)] {
            let (tile, reads) = answer(&file, &mut reader, |r| r.tile(TileIndex::new(z, x, y)));
            assert_eq!(
                tile,
                Some((body(z, x, y), Format::Vector(Encoding::Mvt))),
                "{z}/{x}/{y}"
            );
            assert!(reads <= 4, "{z}/{x}/{y}: {reads} reads");
        }
        let (missing, _) = answer(&file, &mut reader, |r| r.tile(TileIndex::new(1, 0, 0)));
        assert_eq!(missing, None);
    }
}

#[test]
fn zooms_the_metadata_leaves_out_come_from_the_tiles() {
    let file = flat(false, false);
    let (tilejson, _) = answer(&file, &mut Reader::default(), Reader::tilejson);
    assert_eq!(tilejson, r#"{"minzoom":2,"maxzoom":4,"encoding":"mvt"}"#);
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
    let mut reader = Reader::default();
    let (tilejson, _) = answer(&file, &mut reader, Reader::tilejson);
    assert_eq!(tilejson, r#"{"minzoom":3,"maxzoom":3}"#);
    let (tile, _) = answer(&file, &mut reader, |r| r.tile(TileIndex::new(3, 1, 2)));
    assert_eq!(tile, Some((vec![0x89, 0x50, 0x4e, 0x48], Format::Raster)));
    let (tile, _) = answer(&file, &mut reader, |r| r.tile(TileIndex::new(3, 1, 1)));
    assert_eq!(tile, Some((vec![0x89, 0x50, 0x4e, 0x47], Format::Raster)));
}

#[test]
fn what_isnt_an_archive_says_so() {
    let mut reader = Reader::default();
    let not = vec![7u8; 8192];
    let step = reader.tilejson();
    let Ok(Step::Read { offset, length }) = step else {
        panic!("{step:?}");
    };
    assert!(reader.insert(offset, &not[..length as usize]).is_err());
    let file = written(|db| db.execute_batch("CREATE TABLE other (a);").unwrap());
    let mut reader = Reader::default();
    let mut failed = None;
    for _ in 0..8 {
        match reader.tilejson() {
            Ok(Step::Read { offset, length }) => {
                let end = ((offset + length) as usize).min(file.len());
                reader.insert(offset, &file[offset as usize..end]).unwrap();
            }
            Ok(Step::Done(t)) => panic!("{t}"),
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    assert!(failed.unwrap().contains("no MBTiles tiles table"));
}
