use std::cell::RefCell;

use rusqlite::{Connection, params};

use super::*;

/// A database `fill` writes, as SQLite wrote it to disk.
fn written(page_size: u32, fill: impl FnOnce(&Connection)) -> Vec<u8> {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("orb-sqlite-test-{}-{n}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    {
        let db = Connection::open(&path).unwrap();
        db.execute_batch(&format!("PRAGMA page_size = {page_size};"))
            .unwrap();
        fill(&db);
    }
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    bytes
}

/// A tile's body: big enough at some rows to overflow its page.
fn body(z: i64, x: i64, y: i64) -> Vec<u8> {
    let len = if (x + y) % 7 == 0 {
        3000 + 17 * x
    } else {
        20 + y
    };
    (0..len)
        .map(|i| (i * 31 + z * 7 + x * 3 + y) as u8)
        .collect()
}

fn tiles(page_size: u32) -> Vec<u8> {
    written(page_size, |db| {
        db.execute_batch(
            "CREATE TABLE metadata (name text, value text);
             CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
             CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);
             INSERT INTO metadata VALUES ('format', 'pbf'), ('name', 'test');",
        )
        .unwrap();
        let tx = db.unchecked_transaction().unwrap();
        for z in 0..6i64 {
            for x in 0..(1 << z) {
                for y in 0..(1 << z) {
                    tx.execute(
                        "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                        params![z, x, y, body(z, x, y)],
                    )
                    .unwrap();
                }
            }
        }
        tx.commit().unwrap();
    })
}

fn key(z: i64, x: i64, y: i64) -> [Value; 3] {
    [Value::Integer(z), Value::Integer(x), Value::Integer(y)]
}

#[test]
fn reads_tables_and_indexes_through_every_level_and_overflow() {
    for page_size in [512, 1024, 4096] {
        let file = tiles(page_size);
        let db = Database::open(&file[..]).unwrap();
        assert_eq!(db.header().page_size, page_size);
        let tiles = db.table("tiles").unwrap();
        let index = db.index("tiles", &["zoom_level", "tile_column"]).unwrap();
        assert_eq!(index.name, "tile_index");
        for z in 0..6 {
            for x in 0..(1 << z) {
                for y in 0..(1 << z) {
                    let row = db
                        .find_row(&file[..], tiles, index, &key(z, x, y))
                        .unwrap()
                        .unwrap_or_else(|| panic!("{z}/{x}/{y} at {page_size}"));
                    assert_eq!(row.values[3], Value::Blob(body(z, x, y)));
                }
            }
        }
        assert_eq!(db.find(&file[..], index, &key(6, 0, 0)).unwrap(), None);
        assert_eq!(
            db.last(&file[..], index).unwrap().unwrap()[..3],
            key(5, 31, 31)
        );
        assert_eq!(db.find(&file[..], index, &key(2, 1, 9)).unwrap(), None);
        // A prefix finds the first entry it starts.
        let first = db
            .find(&file[..], index, &[Value::Integer(3), Value::Integer(2)])
            .unwrap()
            .unwrap();
        assert_eq!(first[..3], key(3, 2, 0));
        let mut rows = 0;
        db.rows(&file[..], tiles, |row| {
            rows += 1;
            assert_eq!(row.rowid, rows);
            Ok(true)
        })
        .unwrap();
        assert_eq!(rows, (0..6).map(|z| 1 << (2 * z)).sum::<i64>());
        let mut entries = Vec::new();
        db.entries(&file[..], index, |entry| {
            entries.push(entry);
            Ok(true)
        })
        .unwrap();
        assert_eq!(entries.len() as i64, rows);
        assert!(entries.windows(2).all(|w| {
            w[0][..3]
                .iter()
                .zip(&w[1][..3])
                .map(|(a, b)| a.compare(b))
                .find(|o| o.is_ne())
                == Some(std::cmp::Ordering::Less)
        }));
    }
}

#[test]
fn reads_rowid_aliases_and_constraint_indexes() {
    let file = written(1024, |db| {
        db.execute_batch(
            "CREATE TABLE resources (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL, \
             data BLOB, UNIQUE (url));
             CREATE TABLE images (tile_data blob, tile_id text);
             CREATE UNIQUE INDEX images_id on images (tile_id);",
        )
        .unwrap();
        for i in 0..400 {
            db.execute(
                "INSERT INTO resources (url, data) VALUES (?, ?)",
                params![format!("https://x/{i}.json"), vec![i as u8; i]],
            )
            .unwrap();
            db.execute(
                "INSERT INTO images VALUES (?, ?)",
                params![
                    vec![i as u8; 3],
                    format!("{:08x}", i as u64 * 2654435761 % (1 << 32))
                ],
            )
            .unwrap();
        }
    });
    let db = Database::open(&file[..]).unwrap();
    let resources = db.table("resources").unwrap();
    assert_eq!(resources.rowid_alias, Some(0));
    let row = db.row(&file[..], resources, 37).unwrap().unwrap();
    assert_eq!(row.values[0], Value::Integer(37));
    assert_eq!(row.values[1].as_text(), Some("https://x/36.json"));
    let by_url = db.index("resources", &["url"]).unwrap();
    assert!(by_url.name.starts_with("sqlite_autoindex_resources_"));
    let found = db
        .find_row(
            &file[..],
            resources,
            by_url,
            &[Value::Text(b"https://x/123.json".to_vec())],
        )
        .unwrap()
        .unwrap();
    assert_eq!(found.values[2], Value::Blob(vec![123; 123]));
    let images = db.table("images").unwrap();
    let by_id = db.index("images", &["tile_id"]).unwrap();
    for i in [0u64, 1, 199, 399] {
        let id = format!("{:08x}", i * 2654435761u64 % (1 << 32));
        let row = db
            .find_row(&file[..], images, by_id, &[Value::Text(id.into_bytes())])
            .unwrap()
            .unwrap();
        assert_eq!(row.values[0], Value::Blob(vec![i as u8; 3]));
    }
    assert!(db.row(&file[..], resources, 401).unwrap().is_none());
}

/// Reads ranges of a file, counting them.
struct Ranges<'a> {
    file: &'a [u8],
    cache: RefCell<Cache>,
    reads: RefCell<Vec<(u64, u64)>>,
}

impl Ranges<'_> {
    /// `read` with the cache, reading what it needs until it answers.
    fn answer<T>(&self, mut read: impl FnMut(&Cache) -> Result<T>) -> T {
        for _ in 0..64 {
            let need = match read(&self.cache.borrow()) {
                Err(Error::Need { page, run }) => self.cache.borrow().range(page, run),
                other => return other.unwrap(),
            };
            self.reads.borrow_mut().push(need);
            let (at, len) = (need.0 as usize, need.1 as usize);
            let bytes = &self.file[at.min(self.file.len())..(at + len).min(self.file.len())];
            self.cache.borrow_mut().insert(need.0, bytes).unwrap();
        }
        panic!("the cache never answered");
    }
}

#[test]
fn a_cache_reads_what_lookups_need_and_little_more() {
    let file = tiles(1024);
    let ranges = Ranges {
        file: &file,
        cache: RefCell::default(),
        reads: RefCell::default(),
    };
    let db = ranges.answer(Database::open);
    assert_eq!(*ranges.reads.borrow(), [(0, 4096)]);
    let tiles = db.table("tiles").unwrap();
    let index = db.index("tiles", &["zoom_level"]).unwrap();
    let mut first = 0;
    for (z, x, y) in [(5, 14, 0), (5, 14, 1), (4, 3, 4), (5, 14, 0)] {
        ranges.reads.borrow_mut().clear();
        let row = ranges.answer(|cache| db.find_row(cache, tiles, index, &key(z, x, y)));
        assert_eq!(row.unwrap().values[3], Value::Blob(body(z, x, y)));
        let reads = ranges.reads.borrow().len();
        if first == 0 {
            first = reads;
        }
        // An overflowing body reads its chain at once: SQLite wrote it in
        // a run.
        assert!(reads <= 8, "{z}/{x}/{y}: {reads} reads");
    }
    // Read again, the tile reads nothing.
    assert!(ranges.reads.borrow().is_empty());
    assert!(first > 0);
}

#[test]
fn damaged_files_fail_without_panicking() {
    let file = tiles(512);
    assert!(Database::open(&file[..50]).is_err());
    assert!(Database::open(&b"SQLite format 2\0"[..]).is_err());
    let db = Database::open(&file[..]).unwrap();
    let tiles = db.table("tiles").unwrap();
    let index = db.index("tiles", &["zoom_level"]).unwrap();
    let truncated = &file[..file.len() / 2];
    let mut failed = 0;
    for x in 0..32 {
        if db.find_row(truncated, tiles, index, &key(5, x, 3)).is_err() {
            failed += 1;
        }
    }
    assert!(failed > 0);
    // Every byte flipped in turn, in the first pages: answers or errors.
    for at in (0..file.len().min(8 * 512)).step_by(3) {
        let mut bad = file.clone();
        bad[at] ^= 0xa5;
        if let Ok(db) = Database::open(&bad[..])
            && let (Some(tiles), Some(index)) =
                (db.table("tiles"), db.index("tiles", &["zoom_level"]))
        {
            let _ = db.find_row(&bad[..], tiles, index, &key(5, 7, 7));
            let _ = db.rows(&bad[..], tiles, |_| Ok(true));
        }
    }
}
