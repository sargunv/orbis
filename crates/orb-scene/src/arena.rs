use std::collections::BTreeMap;
use std::ops::Range;

/// A first-fit allocator over a fixed range of words, coalescing on free.
///
/// Freed ranges may be reused at once: writes are queue operations ordered
/// after every previously submitted pass, and a freed range is no longer
/// referenced by anything submitted later.
#[derive(Clone, Debug)]
pub(crate) struct Arena {
    /// Free blocks by start.
    free: BTreeMap<u32, u32>,
    used: u32,
}

impl Arena {
    pub(crate) fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { free, used: 0 }
    }

    pub(crate) fn used(&self) -> u32 {
        self.used
    }

    pub(crate) fn alloc(&mut self, len: u32) -> Option<Range<u32>> {
        if len == 0 {
            return Some(0..0);
        }
        let (&start, &size) = self.free.iter().find(|(_, size)| **size >= len)?;
        self.free.remove(&start);
        if size > len {
            self.free.insert(start + len, size - len);
        }
        self.used += len;
        Some(start..start + len)
    }

    pub(crate) fn free(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        self.used -= range.len() as u32;
        let mut start = range.start;
        let mut len = range.len() as u32;
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back() {
            debug_assert!(prev + prev_len <= start, "double free");
            if prev + prev_len == start {
                self.free.remove(&prev);
                start = prev;
                len += prev_len;
            }
        }
        if let Some(next_len) = self.free.remove(&(start + len)) {
            len += next_len;
        }
        self.free.insert(start, len);
    }
}

/// Fixed-capacity slot indices with reuse.
#[derive(Debug)]
pub(crate) struct Slots {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
}

impl Slots {
    pub(crate) fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next: 0,
            free: Vec::new(),
        }
    }

    pub(crate) fn alloc(&mut self) -> Option<u32> {
        self.free.pop().or_else(|| {
            (self.next < self.capacity).then(|| {
                self.next += 1;
                self.next - 1
            })
        })
    }

    pub(crate) fn free(&mut self, slot: u32) {
        self.free.push(slot);
    }

    pub(crate) fn available(&self) -> u32 {
        self.capacity - self.next + self.free.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_first_fit_and_coalesces() {
        let mut arena = Arena::new(100);
        let a = arena.alloc(30).unwrap();
        let b = arena.alloc(30).unwrap();
        let c = arena.alloc(30).unwrap();
        assert_eq!((a.clone(), b.clone(), c.clone()), (0..30, 30..60, 60..90));
        assert!(arena.alloc(20).is_none());
        arena.free(a);
        arena.free(c);
        // c merged with the free tail into 60..100.
        assert_eq!(arena.alloc(41), None);
        arena.free(b);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.alloc(100), Some(0..100));
    }

    #[test]
    fn slots_reuse() {
        let mut slots = Slots::new(2);
        let a = slots.alloc().unwrap();
        let _b = slots.alloc().unwrap();
        assert_eq!(slots.alloc(), None);
        slots.free(a);
        assert_eq!(slots.alloc(), Some(a));
    }
}
