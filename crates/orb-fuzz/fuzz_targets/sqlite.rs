//! A SQLite file: its schema, every table's rows and every index's
//! entries, then read as an MBTiles archive by byte ranges (the reader's
//! page cache and its reads) and as MapLibre Native's offline database.

#![no_main]
#![forbid(unsafe_code)]

use orb_runtime::orb_sqlite::{Database, Value};
use orb_source::mbtiles::{Reader, Step};

/// Answers `step` from `data` until it's done or fails.
fn answer<T>(
    data: &[u8],
    reader: &mut Reader,
    mut step: impl FnMut(&mut Reader) -> Result<Step<T>, String>,
) {
    for _ in 0..64 {
        match step(reader) {
            Ok(Step::Read { offset, length }) => {
                let at = (offset as usize).min(data.len());
                let end = (offset.saturating_add(length) as usize).min(data.len());
                if reader.insert(offset, &data[at..end]).is_err() {
                    return;
                }
            }
            _ => return,
        }
    }
}

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Ok(db) = Database::open(data) {
        for name in [
            "tiles",
            "map",
            "images",
            "metadata",
            "regions",
            "resources",
            "region_tiles",
        ] {
            if let Some(table) = db.table(name) {
                let mut rows = 0;
                let _ = db.rows(data, table, |_| {
                    rows += 1;
                    Ok(rows < 1000)
                });
                let _ = db.row(data, table, 1);
                if let Some(index) = table.columns.first().and_then(|c| db.index(name, &[c])) {
                    let _ = db.find_row(data, table, index, &[Value::Integer(3)]);
                    let _ = db.last(data, index);
                    let mut entries = 0;
                    let _ = db.entries(data, index, |_| {
                        entries += 1;
                        Ok(entries < 1000)
                    });
                }
            }
        }
        if let Ok(native) = orb_runtime::native_db::NativeDatabase::open(data) {
            let _ = native.regions();
        }
    }
    let mut reader = Reader::default();
    answer(data, &mut reader, Reader::tilejson);
    for tile in [(0, 0, 0), (3, 2, 5), (14, 9000, 6000)] {
        let tile = orb_crs::TileIndex::new(tile.0, tile.1, tile.2);
        answer(data, &mut reader, |r| r.tile(tile));
    }
});
