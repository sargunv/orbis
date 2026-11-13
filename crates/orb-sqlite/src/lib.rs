//! A read-only reader of SQLite's file format, with no I/O: enough to read
//! MBTiles archives and MapLibre Native's offline databases.
//!
//! It reads the header, the schema, table B-trees by rowid and index
//! B-trees by key, with overflow pages; nothing writes. A [`Pages`] hands
//! it pages: a [`Cache`] of pages read so far, which answers a page it
//! lacks with [`Error::Need`] so the caller reads it (a byte range) and
//! tries again, as the PMTiles reader does, or whole files in memory.
//! Tables `WITHOUT ROWID`, UTF-16 text and collations but `BINARY` aren't
//! read; a write-ahead log isn't either, so what it read is the database
//! as last checkpointed.

#![forbid(unsafe_code)]

mod btree;
mod cache;
mod record;
mod schema;

use std::sync::Arc;

pub use cache::Cache;
pub use record::Value;
pub use schema::{Index, Table};

/// The database header's length, at the start of page 1.
pub const HEADER_LEN: usize = 100;

/// Why a read didn't answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Page `page` (counting from 1) isn't read yet: read it, with the
    /// `run - 1` pages after it, which the read will likely want too, and
    /// try again.
    Need { page: u32, run: u32 },
    /// The file isn't a database, or is damaged.
    Corrupt(String),
    /// The database uses what this reader doesn't read.
    Unsupported(String),
}

impl Error {
    pub(crate) fn corrupt(message: impl Into<String>) -> Self {
        Error::Corrupt(message.into())
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Error::Unsupported(message.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Need { page, run } => write!(f, "pages {page}..{} aren't read", page + run),
            Error::Corrupt(message) => write!(f, "not a valid SQLite database: {message}"),
            Error::Unsupported(message) => write!(f, "an unsupported SQLite database: {message}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Where the reader gets pages.
pub trait Pages {
    /// Page `number`, counting from 1, whole. `run` is how many pages from
    /// it on the reader expects to want, for a source that reads ahead.
    fn page(&self, number: u32, run: u32) -> Result<Arc<[u8]>>;
}

/// A whole database in memory.
impl Pages for [u8] {
    fn page(&self, number: u32, _: u32) -> Result<Arc<[u8]>> {
        let header = Header::parse(self)?;
        let size = header.page_size as usize;
        (number as usize)
            .checked_sub(1)
            .and_then(|i| i.checked_mul(size))
            .and_then(|at| self.get(at..at.checked_add(size)?))
            .map(Arc::from)
            .ok_or_else(|| Error::corrupt(format!("page {number} is past the file's end")))
    }
}

/// What the reader takes from the database header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub page_size: u32,
    /// Each page's bytes less those reserved at its end.
    pub usable: u32,
    /// The file's pages, when the header knows.
    pub pages: Option<u32>,
}

impl Header {
    /// Parses the header at the start of `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let bytes = bytes
            .get(..HEADER_LEN)
            .filter(|b| b.starts_with(b"SQLite format 3\0"))
            .ok_or_else(|| Error::corrupt("no SQLite header"))?;
        let u32_at = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().expect("4"));
        let page_size = match u16::from_be_bytes([bytes[16], bytes[17]]) {
            1 => 65536,
            size => u32::from(size),
        };
        if !page_size.is_power_of_two() || !(512..=65536).contains(&page_size) {
            return Err(Error::corrupt(format!("a page size of {page_size}")));
        }
        if bytes[19] > 2 {
            return Err(Error::unsupported(format!(
                "file format version {}",
                bytes[19]
            )));
        }
        let usable = page_size - u32::from(bytes[20]);
        if usable < 480 || bytes[21..24] != [64, 32, 32] {
            return Err(Error::corrupt("bad payload fractions or reserved space"));
        }
        // The page count holds when the change counter it was written at
        // is the file's.
        let pages = (bytes[24..28] == bytes[92..96])
            .then(|| u32_at(28))
            .filter(|&n| n > 0);
        match u32_at(56) {
            0 | 1 => {}
            encoding => {
                return Err(Error::unsupported(format!(
                    "text encoding {encoding} (only UTF-8 is read)"
                )));
            }
        }
        Ok(Self {
            page_size,
            usable,
            pages,
        })
    }
}

/// A row of a table.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub rowid: i64,
    /// One value per column of the table, the rowid in its alias.
    pub values: Vec<Value>,
}

/// A database's header and schema.
#[derive(Clone, Debug, PartialEq)]
pub struct Database {
    header: Header,
    tables: Vec<Table>,
    indexes: Vec<Index>,
}

impl Database {
    /// Reads the header and the schema.
    pub fn open(pages: &(impl Pages + ?Sized)) -> Result<Self> {
        let header = Header::parse(&pages.page(1, 1)?)?;
        let tree = btree::Tree::new(pages, &header);
        let mut entries = Vec::new();
        tree.rows(1, &mut |_, payload| {
            entries.push(record::decode(&payload, 5)?);
            Ok(true)
        })?;
        let mut db = Self {
            header,
            tables: Vec::new(),
            indexes: Vec::new(),
        };
        let text = |v: &Value| v.as_text().map(str::to_string);
        // `type, name, tbl_name, rootpage, sql`.
        let mut autoindexes = Vec::new();
        for entry in &entries {
            let [kind, name, table, root, sql] = &entry[..] else {
                return Err(Error::corrupt("a schema row without five columns"));
            };
            let (Some(kind), Some(name)) = (text(kind), text(name)) else {
                return Err(Error::corrupt("a schema row without a type or name"));
            };
            let root = root
                .as_integer()
                .and_then(|r| u32::try_from(r).ok())
                .unwrap_or(0);
            match (kind.as_str(), text(sql)) {
                ("table", Some(sql)) if root > 0 => match schema::table(&name, root, &sql) {
                    Ok(table) => db.tables.push(table),
                    // A table the reader can't read is no table to it.
                    Err(Error::Unsupported(_)) => {}
                    Err(e) => return Err(e),
                },
                ("index", Some(sql)) if root > 0 => {
                    let table = text(table).unwrap_or_default();
                    db.indexes.push(schema::index(&name, &table, root, &sql)?);
                }
                // A constraint's index, which has no SQL.
                ("index", None) if root > 0 => autoindexes.push((name, root)),
                _ => {}
            }
        }
        for (name, root) in autoindexes {
            let Some((table, n)) = name
                .strip_prefix("sqlite_autoindex_")
                .and_then(|rest| rest.rsplit_once('_'))
                .and_then(|(table, n)| Some((table, n.parse::<usize>().ok()?)))
            else {
                continue;
            };
            let Some(table) = db.table(table) else {
                continue;
            };
            let Some(columns) = n
                .checked_sub(1)
                .and_then(|i| table.constraint_indexes.get(i))
            else {
                continue;
            };
            db.indexes.push(Index {
                descending: vec![false; columns.len()],
                columns: columns.clone(),
                name,
                table: table.name.clone(),
                root,
            });
        }
        Ok(db)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Table `name` (ASCII case-insensitively), if the reader can read it.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
    }

    /// An index of `table` that starts with `columns`.
    pub fn index(&self, table: &str, columns: &[&str]) -> Option<&Index> {
        self.indexes.iter().find(|i| {
            i.table.eq_ignore_ascii_case(table)
                && i.columns.len() >= columns.len()
                && i.columns
                    .iter()
                    .zip(columns)
                    .all(|(a, b)| a.eq_ignore_ascii_case(b))
        })
    }

    fn row_of(table: &Table, rowid: i64, payload: &[u8]) -> Result<Row> {
        let mut values = record::decode(payload, table.columns.len())?;
        values.resize(table.columns.len(), Value::Null);
        if let Some(alias) = table.rowid_alias {
            values[alias] = Value::Integer(rowid);
        }
        Ok(Row { rowid, values })
    }

    /// Calls `each` with the rows of `table`, in rowid order, until it
    /// returns false.
    pub fn rows(
        &self,
        pages: &(impl Pages + ?Sized),
        table: &Table,
        mut each: impl FnMut(Row) -> Result<bool>,
    ) -> Result<()> {
        btree::Tree::new(pages, &self.header).rows(table.root, &mut |rowid, payload| {
            each(Self::row_of(table, rowid, &payload)?)
        })
    }

    /// Row `rowid` of `table`.
    pub fn row(
        &self,
        pages: &(impl Pages + ?Sized),
        table: &Table,
        rowid: i64,
    ) -> Result<Option<Row>> {
        btree::Tree::new(pages, &self.header)
            .row(table.root, rowid)?
            .map(|payload| Self::row_of(table, rowid, &payload))
            .transpose()
    }

    /// Calls `each` with the entries of `index` in order (its columns'
    /// values, then the rowid), until it returns false.
    pub fn entries(
        &self,
        pages: &(impl Pages + ?Sized),
        index: &Index,
        mut each: impl FnMut(Vec<Value>) -> Result<bool>,
    ) -> Result<()> {
        btree::Tree::new(pages, &self.header).entries(index.root, &mut |payload| {
            each(record::decode(&payload, usize::MAX)?)
        })
    }

    /// The first entry of `index` that starts with `key`: its columns'
    /// values, then the rowid.
    pub fn find(
        &self,
        pages: &(impl Pages + ?Sized),
        index: &Index,
        key: &[Value],
    ) -> Result<Option<Vec<Value>>> {
        btree::Tree::new(pages, &self.header).find(index.root, key, &index.descending)
    }

    /// The last entry of `index`.
    pub fn last(&self, pages: &(impl Pages + ?Sized), index: &Index) -> Result<Option<Vec<Value>>> {
        btree::Tree::new(pages, &self.header).last(index.root)
    }

    /// The row of `table` that `index` finds for `key`.
    pub fn find_row(
        &self,
        pages: &(impl Pages + ?Sized),
        table: &Table,
        index: &Index,
        key: &[Value],
    ) -> Result<Option<Row>> {
        let Some(entry) = self.find(pages, index, key)? else {
            return Ok(None);
        };
        let rowid = entry
            .last()
            .and_then(Value::as_integer)
            .ok_or_else(|| Error::corrupt("an index entry without a rowid"))?;
        self.row(pages, table, rowid)
    }
}

#[cfg(test)]
mod tests;
