//! Pages read so far, for a reader that reads byte ranges.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::{Error, HEADER_LEN, Header, Pages, Result};

/// What the first read takes: the header and, at the usual page size,
/// page 1 with the schema.
const HEAD: u64 = 4096;
/// The most bytes one read takes ahead.
const MAX_RUN_BYTES: u64 = 4 << 20;
/// Leaf and overflow pages kept, and interior pages: the trees' upper
/// levels, which every lookup walks, are kept longer.
const LEAF_BYTES: usize = 8 << 20;
const INTERIOR_BYTES: usize = 16 << 20;
/// The last inserts, which a lookup in progress may still want, are
/// never evicted.
const KEPT_INSERTS: u64 = 16;

/// The pages read so far of a database read by byte ranges. As a
/// [`Pages`] it answers a page it lacks with [`Error::Need`]; read
/// [`Cache::range`] of it, [`Cache::insert`] the bytes and try again.
#[derive(Debug, Default)]
pub struct Cache {
    header: Option<Header>,
    interior: HashMap<u32, Arc<[u8]>>,
    interior_bytes: usize,
    leaves: HashMap<u32, (u64, Arc<[u8]>)>,
    /// Leaf pages by insert, oldest first.
    order: VecDeque<(u64, u32)>,
    leaf_bytes: usize,
    inserts: u64,
}

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The header, once the first read is in.
    pub fn header(&self) -> Option<&Header> {
        self.header.as_ref()
    }

    /// The bytes to read, as offset and length, for [`Error::Need`] of
    /// `run` pages from `page`: the head until the page size is known.
    pub fn range(&self, page: u32, run: u32) -> (u64, u64) {
        match self.header {
            None => (0, HEAD),
            Some(header) => {
                let size = u64::from(header.page_size);
                let mut pages = u64::from(run.max(1)).min((MAX_RUN_BYTES / size).max(1));
                if let Some(count) = header.pages {
                    pages = pages.min(u64::from(count.saturating_sub(page)) + 1);
                }
                (u64::from(page.saturating_sub(1)) * size, pages * size)
            }
        }
    }

    /// Takes bytes read from `offset`, a range [`Cache::range`] gave: the
    /// whole pages in them.
    pub fn insert(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let head = self.header.is_none();
        let header = match self.header {
            Some(header) => header,
            None if offset == 0 => *self.header.insert(Header::parse(bytes)?),
            None => return Err(Error::corrupt("pages read before the header")),
        };
        let size = header.page_size as usize;
        if !offset.is_multiple_of(size as u64) {
            return Err(Error::corrupt("a read not on a page boundary"));
        }
        // A read short of the page it asked for means the file ends
        // inside it; the head's may be just short of a big page.
        if bytes.len() < size && !head {
            return Err(Error::corrupt("the file ends inside a page"));
        }
        self.inserts += 1;
        let first = u32::try_from(offset / size as u64 + 1)
            .map_err(|_| Error::corrupt("a page past the format's limit"))?;
        let count = (bytes.len() / size) as u32;
        for (i, page) in (first..).zip(bytes.chunks_exact(size)) {
            let page: Arc<[u8]> = Arc::from(page);
            let at = if i == 1 { HEADER_LEN } else { 0 };
            // Interior B-tree pages; overflow pages start with a page
            // number, whose first byte is 0 below 16 million pages.
            if matches!(page[at], 2 | 5) {
                if self.interior.insert(i, page).is_none() {
                    self.interior_bytes += size;
                }
            } else {
                self.order.push_back((self.inserts, i));
                if self.leaves.insert(i, (self.inserts, page)).is_none() {
                    self.leaf_bytes += size;
                }
            }
        }
        while self.leaf_bytes > LEAF_BYTES {
            let Some(&(insert, page)) = self.order.front() else {
                break;
            };
            if insert + KEPT_INSERTS > self.inserts {
                break;
            }
            self.order.pop_front();
            if self.leaves.get(&page).is_some_and(|(at, _)| *at == insert) {
                self.leaves.remove(&page);
                self.leaf_bytes -= size;
            }
        }
        if self.interior_bytes > INTERIOR_BYTES {
            self.interior
                .retain(|&i, _| i >= first && i - first < count);
            self.interior_bytes = self.interior.len() * size;
        }
        Ok(())
    }

    /// The bytes it holds.
    pub fn bytes(&self) -> usize {
        self.interior_bytes + self.leaf_bytes
    }
}

impl Pages for Cache {
    fn page(&self, number: u32, run: u32) -> Result<Arc<[u8]>> {
        if let Some(page) = self.interior.get(&number) {
            return Ok(page.clone());
        }
        match self.leaves.get(&number) {
            Some((_, page)) => Ok(page.clone()),
            None => Err(Error::Need { page: number, run }),
        }
    }
}
