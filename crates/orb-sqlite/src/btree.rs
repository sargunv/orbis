//! B-tree pages: table trees keyed by rowid, index trees keyed by record,
//! and the overflow chains of payloads too big for their page.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use crate::record::{Value, decode, varint};
use crate::{Error, Header, Pages, Result};

/// How deep a tree may go before it's taken as a cycle: a tree of 4 KiB
/// pages this deep would hold far more rows than a file can.
const MAX_DEPTH: usize = 40;
/// The biggest payload read: SQLite's own default limit on a value.
const MAX_PAYLOAD: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    InteriorIndex,
    InteriorTable,
    LeafIndex,
    LeafTable,
}

struct Node {
    data: Arc<[u8]>,
    kind: Kind,
    cells: usize,
    /// Where the cell pointers start.
    pointers: usize,
    right: Option<u32>,
}

pub(crate) struct Tree<'a, P: Pages + ?Sized> {
    pages: &'a P,
    usable: usize,
    count: Option<u32>,
}

impl<'a, P: Pages + ?Sized> Tree<'a, P> {
    pub(crate) fn new(pages: &'a P, header: &Header) -> Self {
        Self {
            pages,
            usable: header.usable as usize,
            count: header.pages,
        }
    }

    fn page(&self, number: u32, run: u32) -> Result<Arc<[u8]>> {
        if number == 0 || self.count.is_some_and(|count| number > count) {
            return Err(Error::corrupt(format!("a pointer to page {number}")));
        }
        self.pages.page(number, run.max(1))
    }

    fn node(&self, number: u32) -> Result<Node> {
        let data = self.page(number, 1)?;
        let at = if number == 1 { crate::HEADER_LEN } else { 0 };
        let bytes = data
            .get(at..at + 12)
            .ok_or_else(|| Error::corrupt("a short page"))?;
        let kind = match bytes[0] {
            2 => Kind::InteriorIndex,
            5 => Kind::InteriorTable,
            10 => Kind::LeafIndex,
            13 => Kind::LeafTable,
            other => return Err(Error::corrupt(format!("page {number} has type {other}"))),
        };
        let cells = usize::from(u16::from_be_bytes([bytes[3], bytes[4]]));
        let interior = matches!(kind, Kind::InteriorIndex | Kind::InteriorTable);
        let right = interior.then(|| u32::from_be_bytes(bytes[8..12].try_into().expect("4")));
        let pointers = at + if interior { 12 } else { 8 };
        if pointers + 2 * cells > data.len() {
            return Err(Error::corrupt(format!("page {number}'s cells overflow it")));
        }
        Ok(Node {
            data,
            kind,
            cells,
            pointers,
            right,
        })
    }

    fn cell(&self, node: &Node, i: usize) -> Result<usize> {
        let at = node.pointers + 2 * i;
        let offset = usize::from(u16::from_be_bytes([node.data[at], node.data[at + 1]]));
        if offset >= self.usable.min(node.data.len()) {
            return Err(Error::corrupt("a cell past its page"));
        }
        Ok(offset)
    }

    fn child(&self, node: &Node, at: usize) -> Result<u32> {
        let bytes = node
            .data
            .get(at..at + 4)
            .ok_or_else(|| Error::corrupt("a child pointer past its page"))?;
        Ok(u32::from_be_bytes(bytes.try_into().expect("4")))
    }

    /// The payload of `length` bytes at `at` in `node`, its overflow
    /// chain followed.
    fn payload(&self, node: &Node, at: usize, length: u64) -> Result<Vec<u8>> {
        if length > MAX_PAYLOAD {
            return Err(Error::corrupt("an oversized payload"));
        }
        let length = length as usize;
        let u = self.usable;
        let max_local = match node.kind {
            Kind::LeafTable => u - 35,
            _ => (u - 12) * 64 / 255 - 23,
        };
        let min_local = (u - 12) * 32 / 255 - 23;
        let local = if length <= max_local {
            length
        } else {
            let k = min_local + (length - min_local) % (u - 4);
            if k <= max_local { k } else { min_local }
        };
        let end = at + local;
        // Each overflow page holds `u - 4` bytes: a payload bigger than
        // the file's pages hold is damage, not a reason to allocate it.
        let per_page = u - 4;
        if self
            .count
            .is_some_and(|count| (length - local).div_ceil(per_page) > count as usize)
        {
            return Err(Error::corrupt("a payload longer than its file"));
        }
        let mut out = Vec::with_capacity(length.min(local + 64 * per_page));
        out.extend_from_slice(
            node.data
                .get(at..end)
                .ok_or_else(|| Error::corrupt("a payload past its page"))?,
        );
        if local == length {
            return Ok(out);
        }
        let mut next = self.child(node, end)?;
        // Each page adds at least a byte, so a chain that cycles ends by
        // its length, which the file's pages bound.
        while out.len() < length {
            let left = length - out.len();
            let pages = left.div_ceil(per_page);
            if next == 0 {
                return Err(Error::corrupt("an overflow chain ends early"));
            }
            let page = self.page(next, u32::try_from(pages).unwrap_or(u32::MAX))?;
            let body = page
                .get(4..4 + per_page.min(left))
                .ok_or_else(|| Error::corrupt("a short overflow page"))?;
            out.extend_from_slice(body);
            next = u32::from_be_bytes(page[..4].try_into().expect("4"));
        }
        Ok(out)
    }

    /// A table leaf's cell: its rowid and payload.
    fn table_row(&self, node: &Node, i: usize) -> Result<(i64, Vec<u8>)> {
        let mut at = self.cell(node, i)?;
        let length = varint(&node.data, &mut at)?;
        let rowid = varint(&node.data, &mut at)? as i64;
        Ok((rowid, self.payload(node, at, length)?))
    }

    /// A table interior's cell: its child and the greatest rowid in it.
    fn table_key(&self, node: &Node, i: usize) -> Result<(u32, i64)> {
        let at = self.cell(node, i)?;
        let child = self.child(node, at)?;
        let mut at = at + 4;
        Ok((child, varint(&node.data, &mut at)? as i64))
    }

    /// An index cell: its child, if interior, and its entry's payload.
    fn index_entry(&self, node: &Node, i: usize) -> Result<(Option<u32>, Vec<u8>)> {
        let mut at = self.cell(node, i)?;
        let child = match node.kind {
            Kind::InteriorIndex => {
                let child = self.child(node, at)?;
                at += 4;
                Some(child)
            }
            _ => None,
        };
        let length = varint(&node.data, &mut at)?;
        Ok((child, self.payload(node, at, length)?))
    }

    /// Calls `each` with every row of the table tree at `root`, in rowid
    /// order, until it returns false.
    pub(crate) fn rows(
        &self,
        root: u32,
        each: &mut dyn FnMut(i64, Vec<u8>) -> Result<bool>,
    ) -> Result<()> {
        self.walk(root, &mut HashSet::new(), 0, &mut |tree, node, i| {
            let (rowid, payload) = tree.table_row(node, i)?;
            each(rowid, payload)
        })
        .map(|_| ())
    }

    /// Calls `each` with every entry of the index tree at `root`, in
    /// order, until it returns false.
    pub(crate) fn entries(
        &self,
        root: u32,
        each: &mut dyn FnMut(Vec<u8>) -> Result<bool>,
    ) -> Result<()> {
        self.walk(root, &mut HashSet::new(), 0, &mut |tree, node, i| {
            let (_, payload) = tree.index_entry(node, i)?;
            each(payload)
        })
        .map(|_| ())
    }

    /// Visits the cells of the tree at `number` in order: leaves' cells,
    /// and index interiors' between their children. False stops it.
    fn walk(
        &self,
        number: u32,
        seen: &mut HashSet<u32>,
        depth: usize,
        visit: &mut dyn FnMut(&Self, &Node, usize) -> Result<bool>,
    ) -> Result<bool> {
        if depth > MAX_DEPTH || !seen.insert(number) {
            return Err(Error::corrupt("a B-tree cycle"));
        }
        let node = self.node(number)?;
        match node.kind {
            Kind::LeafTable | Kind::LeafIndex => {
                for i in 0..node.cells {
                    if !visit(self, &node, i)? {
                        return Ok(false);
                    }
                }
            }
            Kind::InteriorTable | Kind::InteriorIndex => {
                for i in 0..node.cells {
                    let at = self.cell(&node, i)?;
                    let child = self.child(&node, at)?;
                    if !self.walk(child, seen, depth + 1, visit)? {
                        return Ok(false);
                    }
                    if node.kind == Kind::InteriorIndex && !visit(self, &node, i)? {
                        return Ok(false);
                    }
                }
                let right = node.right.expect("interior pages point right");
                return self.walk(right, seen, depth + 1, visit);
            }
        }
        Ok(true)
    }

    /// The payload of row `rowid` of the table tree at `root`.
    pub(crate) fn row(&self, root: u32, rowid: i64) -> Result<Option<Vec<u8>>> {
        let mut number = root;
        for _ in 0..=MAX_DEPTH {
            let node = self.node(number)?;
            match node.kind {
                Kind::InteriorTable => {
                    // The first child whose greatest rowid is at least it.
                    let (mut low, mut high) = (0, node.cells);
                    while low < high {
                        let mid = (low + high) / 2;
                        if self.table_key(&node, mid)?.1 < rowid {
                            low = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                    number = match low < node.cells {
                        true => self.table_key(&node, low)?.0,
                        false => node.right.expect("interior pages point right"),
                    };
                }
                Kind::LeafTable => {
                    let (mut low, mut high) = (0, node.cells);
                    while low < high {
                        let mid = (low + high) / 2;
                        let mut at = self.cell(&node, mid)?;
                        varint(&node.data, &mut at)?;
                        match (varint(&node.data, &mut at)? as i64).cmp(&rowid) {
                            Ordering::Less => low = mid + 1,
                            Ordering::Greater => high = mid,
                            Ordering::Equal => return Ok(Some(self.table_row(&node, mid)?.1)),
                        }
                    }
                    return Ok(None);
                }
                _ => return Err(Error::corrupt("an index page in a table tree")),
            }
        }
        Err(Error::corrupt("a B-tree cycle"))
    }

    /// The last entry of the index tree at `root`.
    pub(crate) fn last(&self, root: u32) -> Result<Option<Vec<Value>>> {
        let mut number = root;
        for _ in 0..=MAX_DEPTH {
            let node = self.node(number)?;
            match node.kind {
                Kind::InteriorIndex => number = node.right.expect("interior pages point right"),
                Kind::LeafIndex if node.cells == 0 => return Ok(None),
                Kind::LeafIndex => {
                    let (_, payload) = self.index_entry(&node, node.cells - 1)?;
                    return Ok(Some(decode(&payload, usize::MAX)?));
                }
                _ => return Err(Error::corrupt("a table page in an index tree")),
            }
        }
        Err(Error::corrupt("a B-tree cycle"))
    }

    /// The first entry of the index tree at `root` whose leading values
    /// are `key`, compared with `descending` per column.
    pub(crate) fn find(
        &self,
        root: u32,
        key: &[Value],
        descending: &[bool],
    ) -> Result<Option<Vec<Value>>> {
        self.find_in(root, key, descending, 0)
    }

    fn find_in(
        &self,
        number: u32,
        key: &[Value],
        descending: &[bool],
        depth: usize,
    ) -> Result<Option<Vec<Value>>> {
        if depth > MAX_DEPTH {
            return Err(Error::corrupt("a B-tree cycle"));
        }
        let node = self.node(number)?;
        if !matches!(node.kind, Kind::InteriorIndex | Kind::LeafIndex) {
            return Err(Error::corrupt("a table page in an index tree"));
        }
        let entry = |i| -> Result<(Option<u32>, Vec<Value>)> {
            let (child, payload) = self.index_entry(&node, i)?;
            Ok((child, decode(&payload, usize::MAX)?))
        };
        let order = |values: &[Value]| {
            key.iter()
                .zip(values)
                .zip(descending.iter().chain(std::iter::repeat(&false)))
                .map(|((k, v), &desc)| {
                    let order = v.compare(k);
                    if desc { order.reverse() } else { order }
                })
                .find(|o| o.is_ne())
                .unwrap_or_else(|| values.len().min(key.len()).cmp(&key.len()))
        };
        // The first entry at least `key`.
        let (mut low, mut high) = (0, node.cells);
        while low < high {
            let mid = (low + high) / 2;
            if order(&entry(mid)?.1) == Ordering::Less {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == node.cells {
            return match node.right {
                Some(right) => self.find_in(right, key, descending, depth + 1),
                None => Ok(None),
            };
        }
        let (child, values) = entry(low)?;
        // An equal entry may sit before it, in its child.
        if let Some(child) = child
            && let Some(found) = self.find_in(child, key, descending, depth + 1)?
        {
            return Ok(Some(found));
        }
        Ok((order(&values) == Ordering::Equal).then_some(values))
    }
}
