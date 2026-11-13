//! MBTiles archives: a SQLite file of tiles, read by byte ranges with no
//! I/O, as the PMTiles reader is. A [`Reader`] answers what it can from
//! the pages read so far and [`Step::Read`] for the next pages it needs;
//! the caller reads them, [`Reader::insert`]s them and asks again.
//!
//! It reads both layouts writers use: a `tiles` table, and mbutil's and
//! OpenMapTiles' deduplicated `map` and `images` tables behind a `tiles`
//! view. Rows count from the south (TMS), as the spec has them.

use orb_crs::TileIndex;
use orb_sqlite::{Cache, Database, Error, Index, Table, Value};
use orb_style::Encoding;

/// The most a tile decompresses to.
const MAX_TILE: usize = 64 << 20;

/// What a read of the archive needs or gives.
#[derive(Clone, Debug, PartialEq)]
pub enum Step<T> {
    /// Read `length` bytes at `offset`, insert them, and ask again.
    Read {
        offset: u64,
        length: u64,
    },
    Done(T),
}

/// What the tiles are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Vector(Encoding),
    /// Images, decoded by content.
    Raster,
}

/// An archive's tiles and how to find one.
#[derive(Debug)]
struct Archive {
    db: Database,
    tiles: Tiles,
    format: Format,
}

#[derive(Debug)]
enum Tiles {
    /// `tiles (zoom_level, tile_column, tile_row, tile_data)`, by an index
    /// on the first three if it has one.
    Table {
        table: Table,
        index: Option<Index>,
        columns: [usize; 4],
    },
    /// `map (zoom_level, tile_column, tile_row, tile_id)` and
    /// `images (tile_data, tile_id)`.
    Deduplicated {
        map: Table,
        map_index: Index,
        tile_id: usize,
        images: Table,
        images_index: Index,
        tile_data: usize,
    },
}

/// One archive's pages read so far, and its schema once read.
#[derive(Debug, Default)]
pub struct Reader {
    cache: Cache,
    archive: Option<Archive>,
    /// The first read failed as one past a short file's end does.
    short_head: bool,
}

/// The least a database file holds: one page of the smallest size.
const MIN_FILE: u64 = 512;

/// An error, or the pages to read.
type Answer<T> = Result<Step<T>, String>;

impl Reader {
    /// Takes bytes read at `offset` for a [`Step::Read`].
    pub fn insert(&mut self, offset: u64, bytes: &[u8]) -> Result<(), String> {
        self.cache.insert(offset, bytes).map_err(|e| e.to_string())
    }

    /// A read at `offset` failed as one past the file's end does: whether
    /// to read again, shorter.
    pub fn read_failed(&mut self, offset: u64) -> bool {
        let again = offset == 0 && self.cache.header().is_none() && !self.short_head;
        self.short_head |= again;
        again
    }

    fn answer<T>(&self, result: orb_sqlite::Result<T>) -> Answer<T> {
        match result {
            Ok(value) => Ok(Step::Done(value)),
            Err(Error::Need { page, run }) => {
                let (offset, mut length) = self.cache.range(page, run);
                if self.short_head && self.cache.header().is_none() {
                    length = MIN_FILE;
                }
                Ok(Step::Read { offset, length })
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn archive(&mut self) -> Answer<()> {
        if self.archive.is_some() {
            return Ok(Step::Done(()));
        }
        let opened = Database::open(&self.cache).and_then(|db| open(&self.cache, db));
        match self.answer(opened)? {
            Step::Done(archive) => {
                self.archive = Some(archive);
                Ok(Step::Done(()))
            }
            Step::Read { offset, length } => Ok(Step::Read { offset, length }),
        }
    }

    /// The archive's TileJSON, from its `metadata` table and, for zooms
    /// it doesn't name, its tiles.
    pub fn tilejson(&mut self) -> Answer<String> {
        if let Step::Read { offset, length } = self.archive()? {
            return Ok(Step::Read { offset, length });
        }
        let archive = self.archive.as_ref().expect("opened");
        let result = tilejson(&self.cache, archive);
        self.answer(result)
    }

    /// Tile `tile`'s bytes, decompressed if gzipped, and what they are;
    /// `None` for a tile the archive leaves out.
    pub fn tile(&mut self, tile: TileIndex) -> Answer<Option<(Vec<u8>, Format)>> {
        if let Step::Read { offset, length } = self.archive()? {
            return Ok(Step::Read { offset, length });
        }
        let archive = self.archive.as_ref().expect("opened");
        let found = find(&self.cache, archive, tile);
        match self.answer(found)? {
            Step::Done(Some(bytes)) => {
                let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
                    crate::gzip::decode(&bytes, MAX_TILE)
                        .map_err(|e| format!("MBTiles gzip: {e}"))?
                } else {
                    bytes
                };
                Ok(Step::Done(Some((bytes, archive.format))))
            }
            Step::Done(None) => Ok(Step::Done(None)),
            Step::Read { offset, length } => Ok(Step::Read { offset, length }),
        }
    }
}

fn columns<const N: usize>(table: &Table, names: [&str; N]) -> orb_sqlite::Result<[usize; N]> {
    let mut out = [0; N];
    for (out, name) in out.iter_mut().zip(names) {
        *out = table.column(name).ok_or_else(|| {
            Error::Unsupported(format!("MBTiles table {} has no {name}", table.name))
        })?;
    }
    Ok(out)
}

const KEY: [&str; 3] = ["zoom_level", "tile_column", "tile_row"];

fn open(pages: &Cache, db: Database) -> orb_sqlite::Result<Archive> {
    let tiles = match db.table("tiles") {
        Some(table) => Tiles::Table {
            columns: columns(
                table,
                ["zoom_level", "tile_column", "tile_row", "tile_data"],
            )?,
            index: db.index("tiles", &KEY).cloned(),
            table: table.clone(),
        },
        None => {
            let (Some(map), Some(images)) = (db.table("map"), db.table("images")) else {
                return Err(Error::Unsupported("no MBTiles tiles table".into()));
            };
            let [tile_id] = columns(map, ["tile_id"])?;
            let [tile_data] = columns(images, ["tile_data"])?;
            let indexed = |table, columns: &[&str]| {
                db.index(table, columns).cloned().ok_or_else(|| {
                    Error::Unsupported(format!("MBTiles table {table} has no index on {columns:?}"))
                })
            };
            Tiles::Deduplicated {
                map_index: indexed("map", &KEY)?,
                images_index: indexed("images", &["tile_id"])?,
                map: map.clone(),
                tile_id,
                images: images.clone(),
                tile_data,
            }
        }
    };
    let format = match metadata(pages, &db, "format")?.as_deref() {
        Some("png" | "jpg" | "jpeg" | "webp") => Format::Raster,
        Some("mlt") => Format::Vector(Encoding::Mlt),
        _ => Format::Vector(Encoding::Mvt),
    };
    Ok(Archive { db, tiles, format })
}

/// The `metadata` table's value for `name`.
fn metadata(pages: &Cache, db: &Database, name: &str) -> orb_sqlite::Result<Option<String>> {
    let Some(table) = db.table("metadata") else {
        return Ok(None);
    };
    let [key, value] = columns(table, ["name", "value"])?;
    let mut found = None;
    db.rows(pages, table, |row| {
        if row.values[key].as_text() == Some(name) {
            found = row.values[value].as_text().map(str::to_string);
            return Ok(false);
        }
        Ok(true)
    })?;
    Ok(found)
}

fn find(pages: &Cache, archive: &Archive, tile: TileIndex) -> orb_sqlite::Result<Option<Vec<u8>>> {
    let db = &archive.db;
    let row = (1i64 << tile.z) - 1 - i64::from(tile.row);
    let key = [
        Value::Integer(tile.z.into()),
        Value::Integer(tile.col.into()),
        Value::Integer(row),
    ];
    let found = match &archive.tiles {
        Tiles::Table {
            table,
            index: Some(index),
            columns,
        } => db
            .find_row(pages, table, index, &key)?
            .map(|r| (r, columns[3])),
        Tiles::Table {
            table,
            index: None,
            columns,
        } => {
            let mut found = None;
            db.rows(pages, table, |r| {
                let at = |i: usize| r.values[columns[i]].as_integer();
                if (0..3).all(|i| at(i) == key[i].as_integer()) {
                    found = Some(r);
                    return Ok(false);
                }
                Ok(true)
            })?;
            found.map(|r| (r, columns[3]))
        }
        Tiles::Deduplicated {
            map,
            map_index,
            tile_id,
            images,
            images_index,
            tile_data,
        } => match db.find_row(pages, map, map_index, &key)? {
            Some(mut entry) => {
                let id = std::mem::replace(&mut entry.values[*tile_id], Value::Null);
                db.find_row(pages, images, images_index, &[id])?
                    .map(|r| (r, *tile_data))
            }
            None => None,
        },
    };
    Ok(found.and_then(|(mut row, data)| row.values[data].take_bytes()))
}

fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn tilejson(pages: &Cache, archive: &Archive) -> orb_sqlite::Result<String> {
    let db = &archive.db;
    let mut members = Vec::new();
    let number = |text: Option<String>| text.and_then(|t| orb_expr::decimal::parse(t.trim()));
    let zoom_of =
        |entry: Option<Vec<Value>>| entry.and_then(|e| e.first().and_then(Value::as_integer));
    let index = match &archive.tiles {
        Tiles::Table { index, .. } => index.as_ref(),
        Tiles::Deduplicated { map_index, .. } => Some(map_index),
    };
    // Zooms the metadata leaves out are its tiles' first and last.
    let minzoom = match number(metadata(pages, db, "minzoom")?) {
        Some(z) => Some(z),
        None => match index {
            Some(index) => zoom_of(db.find(pages, index, &[])?).map(|z| z as f64),
            None => None,
        },
    };
    let maxzoom = match number(metadata(pages, db, "maxzoom")?) {
        Some(z) => Some(z),
        None => match index {
            Some(index) => zoom_of(db.last(pages, index)?).map(|z| z as f64),
            None => None,
        },
    };
    if let Some(z) = minzoom {
        members.push(format!("\"minzoom\":{z}"));
    }
    if let Some(z) = maxzoom {
        members.push(format!("\"maxzoom\":{z}"));
    }
    if let Some(bounds) = metadata(pages, db, "bounds")? {
        let values: Vec<f64> = bounds
            .split(',')
            .filter_map(|v| orb_expr::decimal::parse(v.trim()))
            .collect();
        if let [w, s, e, n] = values[..] {
            members.push(format!("\"bounds\":[{w},{s},{e},{n}]"));
        }
    }
    if let Format::Vector(encoding) = archive.format {
        let name = match encoding {
            Encoding::Mvt => "mvt",
            Encoding::Mlt => "mlt",
        };
        members.push(format!("\"encoding\":\"{name}\""));
    }
    if let Some(attribution) = metadata(pages, db, "attribution")? {
        members.push(format!("\"attribution\":{}", quote(&attribution)));
    }
    // Vector layers, from the JSON the spec keeps in `json`.
    if let Some(json) = metadata(pages, db, "json")?
        && let Ok(doc) = orb_expr::json::parse(&json)
        && let Some(layers) = doc.get("vector_layers")
    {
        members.push(format!("\"vector_layers\":{layers}"));
    }
    Ok(format!("{{{}}}", members.join(",")))
}

#[cfg(test)]
mod tests;
