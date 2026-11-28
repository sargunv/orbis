//! MBTiles archives: a SQLite file of tiles, read as [`crate::archive`]
//! reads any archive, a step at a time for the pages it needs.
//!
//! It reads both layouts writers use: a `tiles` table, and mbutil's and
//! OpenMapTiles' deduplicated `map` and `images` tables behind a `tiles`
//! view. Rows count from the south (TMS), as the spec has them.

use orb_crs::TileIndex;
use orb_sqlite::{Cache, Database, Error, Index, Pages, Table, Value};
use orb_style::Encoding;

use crate::archive::{Body, Range, Step, Tile};
use crate::endpoint::TileJson;

/// The most a tile decompresses to.
const MAX_TILE: usize = 64 << 20;

/// An archive's tiles and how to find one.
#[derive(Debug)]
struct Archive {
    db: Database,
    tiles: Tiles,
    /// A vector tile's encoding; `None` for images, decoded by content.
    encoding: Option<Encoding>,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant, reason = "one per archive")]
enum Tiles {
    /// `tiles (zoom_level, tile_column, tile_row, tile_data)`, by its index
    /// on the first three.
    Table {
        table: Table,
        index: Index,
        tile_data: usize,
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

impl Tiles {
    /// The index by tile.
    fn index(&self) -> &Index {
        match self {
            Self::Table { index, .. } => index,
            Self::Deduplicated { map_index, .. } => map_index,
        }
    }
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
    /// Takes bytes read for a [`Step::Read`] of `range`.
    pub fn insert(&mut self, range: Range, bytes: &[u8]) -> Result<(), String> {
        (self.cache.insert(range.offset, bytes)).map_err(|e| e.to_string())
    }

    /// Whether the read of `range` is still wanted: its first page isn't
    /// in yet.
    pub fn wants(&self, range: Range) -> bool {
        match self.cache.header() {
            None => range.offset == 0,
            Some(header) => {
                let page = range.offset / u64::from(header.page_size) + 1;
                u32::try_from(page).is_ok_and(|page| self.cache.page(page, 1).is_err())
            }
        }
    }

    /// A read of `range` failed as one past the file's end does: whether
    /// to read again, shorter.
    pub fn read_failed(&mut self, range: Range) -> bool {
        let again = range.offset == 0 && self.cache.header().is_none() && !self.short_head;
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
                Ok(Step::Read(Range::new(offset, length)?))
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Reads the archive's schema, then answers `then` of it.
    fn with<T>(
        &mut self,
        then: impl FnOnce(&Cache, &Archive) -> orb_sqlite::Result<T>,
    ) -> Answer<T> {
        if self.archive.is_none() {
            let opened = Database::open(&self.cache).and_then(|db| open(&self.cache, db));
            match self.answer(opened)? {
                Step::Done(archive) => self.archive = Some(archive),
                Step::Read(range) => return Ok(Step::Read(range)),
            }
        }
        let archive = self.archive.as_ref().expect("opened");
        self.answer(then(&self.cache, archive))
    }

    /// What the archive's `metadata` table and, for zooms it doesn't
    /// name, its tiles say of them.
    pub fn tilejson(&mut self) -> Answer<TileJson> {
        self.with(tilejson)
    }

    /// Tile `tile`, decompressed if gzipped; `None` for a tile the archive
    /// leaves out.
    pub fn tile(&mut self, tile: TileIndex) -> Answer<Option<Tile>> {
        let found = match self.with(|pages, archive| find(pages, archive, tile))? {
            Step::Done(found) => found,
            Step::Read(range) => return Ok(Step::Read(range)),
        };
        let Some(bytes) = found else {
            return Ok(Step::Done(None));
        };
        let bytes = match bytes.starts_with(&[0x1f, 0x8b]) {
            true => {
                crate::gzip::decode(&bytes, MAX_TILE).map_err(|e| format!("MBTiles gzip: {e}"))?
            }
            false => bytes,
        };
        let encoding = self.archive.as_ref().expect("opened").encoding;
        Ok(Step::Done(Some(Tile {
            body: Body::Bytes(bytes),
            encoding,
        })))
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
    let indexed = |table, columns: &[&str]| {
        db.index(table, columns).cloned().ok_or_else(|| {
            Error::Unsupported(format!("MBTiles table {table} has no index on {columns:?}"))
        })
    };
    let tiles = match db.table("tiles") {
        Some(table) => {
            let [_, _, _, tile_data] = columns(
                table,
                ["zoom_level", "tile_column", "tile_row", "tile_data"],
            )?;
            Tiles::Table {
                index: indexed("tiles", &KEY)?,
                table: table.clone(),
                tile_data,
            }
        }
        None => {
            let (Some(map), Some(images)) = (db.table("map"), db.table("images")) else {
                return Err(Error::Unsupported("no MBTiles tiles table".into()));
            };
            let [tile_id] = columns(map, ["tile_id"])?;
            let [tile_data] = columns(images, ["tile_data"])?;
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
    let encoding = match metadata(pages, &db, "format")?.as_deref() {
        Some("png" | "jpg" | "jpeg" | "webp") => None,
        Some("mlt") => Some(Encoding::Mlt),
        _ => Some(Encoding::Mvt),
    };
    Ok(Archive {
        db,
        tiles,
        encoding,
    })
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
            index,
            tile_data,
        } => (db.find_row(pages, table, index, &key)?).map(|r| (r, *tile_data)),
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

fn tilejson(pages: &Cache, archive: &Archive) -> orb_sqlite::Result<TileJson> {
    let db = &archive.db;
    let number = |text: Option<String>| text.and_then(|t| orb_expr::decimal::parse(t.trim()));
    let zoom_of = |entry: Option<Vec<Value>>| {
        entry
            .and_then(|e| e.first().and_then(Value::as_integer))
            .map(|z| z as f64)
    };
    // Zooms the metadata leaves out are its tiles' first and last.
    let index = archive.tiles.index();
    let minzoom = match number(metadata(pages, db, "minzoom")?) {
        Some(z) => Some(z),
        None => zoom_of(db.find(pages, index, &[])?),
    };
    let maxzoom = match number(metadata(pages, db, "maxzoom")?) {
        Some(z) => Some(z),
        None => zoom_of(db.last(pages, index)?),
    };
    let bounds = metadata(pages, db, "bounds")?.and_then(|bounds| {
        let values: Vec<f64> = (bounds.split(','))
            .filter_map(|v| orb_expr::decimal::parse(v.trim()))
            .collect();
        values.try_into().ok()
    });
    Ok(TileJson {
        minzoom,
        maxzoom,
        bounds,
        encoding: archive.encoding,
        ..TileJson::default()
    })
}

#[cfg(test)]
mod tests;
