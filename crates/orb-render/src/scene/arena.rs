use std::collections::{BTreeMap, BTreeSet};
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
    capacity: u32,
    /// The words it would have needed for the allocations it refused since
    /// [`take_needed`](Self::take_needed) last asked.
    needed: u32,
}

impl Arena {
    pub(crate) fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            free,
            used: 0,
            capacity,
            needed: 0,
        }
    }

    /// Extends the arena from `old` to `new` words: a larger buffer took
    /// its place, with its contents at the same offsets.
    pub(crate) fn grow(&mut self, old: u32, new: u32) {
        assert!(new > old && old == self.capacity, "arenas only grow");
        self.used += new - old;
        self.free(old..new);
        self.capacity = new;
    }

    /// Where the free block reaching `capacity` starts: every word from
    /// there to `capacity` is free, and none above is held.
    pub(crate) fn free_from(&self, capacity: u32) -> u32 {
        match self.free.range(..capacity).next_back() {
            Some((&start, &len)) if start + len == capacity => start,
            _ => capacity,
        }
    }

    /// Gives back `new..old`, which must be free: a smaller buffer took
    /// its place, with the contents below `new` at the same offsets.
    pub(crate) fn shrink(&mut self, old: u32, new: u32) {
        let start = self.free_from(old);
        assert!(start <= new && new < old, "only a free tail goes");
        self.free.remove(&start);
        if start < new {
            self.free.insert(start, new - start);
        }
        self.capacity = new;
    }

    /// The largest block one allocation can take.
    #[cfg(test)]
    fn largest_free(&self) -> u32 {
        self.free.values().copied().max().unwrap_or(0)
    }

    pub(crate) fn used(&self) -> u32 {
        self.used
    }

    pub(crate) fn alloc(&mut self, len: u32) -> Option<Range<u32>> {
        if len == 0 {
            return Some(0..0);
        }
        let Some((&start, &size)) = self.free.iter().find(|(_, size)| **size >= len) else {
            // Growth extends the free tail.
            let needed = self.free_from(self.capacity).saturating_add(len);
            self.needed = self.needed.max(needed);
            return None;
        };
        self.free.remove(&start);
        if size > len {
            self.free.insert(start + len, size - len);
        }
        self.used += len;
        Some(start..start + len)
    }

    /// Whether allocations of `lens` would all succeed now, in order: what
    /// [`alloc`](Self::alloc) would do, found without copying the free
    /// list. First fit takes the first block with room, and what it leaves
    /// of a block stays where the block was in the list's order.
    pub(crate) fn fits_all(&self, lens: impl IntoIterator<Item = u32>) -> bool {
        // Words taken so far from each block they came from, by its start.
        let mut taken: Vec<(u32, u32)> = Vec::new();
        'next: for len in lens.into_iter().filter(|&len| len > 0) {
            for (&start, &size) in &self.free {
                let at = taken.iter().position(|&(s, _)| s == start);
                if size - at.map_or(0, |i| taken[i].1) >= len {
                    match at {
                        Some(i) => taken[i].1 += len,
                        None => taken.push((start, len)),
                    }
                    continue 'next;
                }
            }
            return false;
        }
        true
    }

    /// The size, in words, that would have held each allocation refused
    /// since the last call when it was refused, or 0. It counts what the
    /// arena held then, which an upload that failed may have given back
    /// since: retried, it takes that room again.
    pub(crate) fn take_needed(&mut self) -> u32 {
        std::mem::take(&mut self.needed)
    }

    /// What [`take_needed`](Self::take_needed) would return.
    pub(crate) fn needed(&self) -> u32 {
        self.needed
    }

    /// Records a refusal of `lens` found without allocating them
    /// ([`fits_all`](Self::fits_all)): the size that holds them all past
    /// what's held.
    pub(crate) fn refuse(&mut self, lens: impl IntoIterator<Item = u32>) {
        let words = lens.into_iter().fold(0u32, u32::saturating_add);
        let needed = self.free_from(self.capacity).saturating_add(words);
        self.needed = self.needed.max(needed);
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

/// Fixed-capacity slot indices with reuse, the lowest free one first, so
/// that held slots stay low and a table can shrink once they go.
#[derive(Debug)]
pub(crate) struct Slots {
    capacity: u32,
    /// Every slot from here up is free; the one below, if any, is held.
    next: u32,
    /// Free slots below `next`.
    free: BTreeSet<u32>,
    /// The capacity that would have held each reservation refused since
    /// [`take_needed`](Self::take_needed) last asked.
    needed: u32,
}

impl Slots {
    pub(crate) fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next: 0,
            free: BTreeSet::new(),
            needed: 0,
        }
    }

    pub(crate) fn alloc(&mut self) -> Option<u32> {
        self.free.pop_first().or_else(|| {
            (self.next < self.capacity).then(|| {
                self.next += 1;
                self.next - 1
            })
        })
    }

    pub(crate) fn free(&mut self, slot: u32) {
        debug_assert!(slot < self.next, "a held slot");
        self.free.insert(slot);
        while self.next > 0 && self.free.remove(&(self.next - 1)) {
            self.next -= 1;
        }
    }

    /// Where the free slots above every held one start.
    pub(crate) fn held_below(&self) -> u32 {
        self.next
    }

    /// Gives up the slots from `capacity` up, which must all be free.
    pub(crate) fn shrink(&mut self, capacity: u32) {
        assert!(self.next <= capacity && capacity < self.capacity);
        self.capacity = capacity;
    }

    /// Whether `count` slots are available; if not, the capacity that
    /// would have held them is kept for [`take_needed`](Self::take_needed).
    pub(crate) fn reserve(&mut self, count: u32) -> bool {
        let available = self.available();
        if count > available {
            let needed = self.capacity.saturating_add(count - available);
            self.needed = self.needed.max(needed);
        }
        count <= available
    }

    /// The capacity that would have held each reservation refused since
    /// the last call when it was refused, or 0.
    pub(crate) fn take_needed(&mut self) -> u32 {
        std::mem::take(&mut self.needed)
    }

    /// What [`take_needed`](Self::take_needed) would return.
    pub(crate) fn needed(&self) -> u32 {
        self.needed
    }

    pub(crate) fn available(&self) -> u32 {
        self.capacity - self.next + self.free.len() as u32
    }

    /// Slots handed out and not freed.
    pub(crate) fn used(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    /// Extends to `capacity` slots.
    pub(crate) fn grow(&mut self, capacity: u32) {
        assert!(capacity > self.capacity, "slots only grow");
        self.capacity = capacity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_first_fit_and_coalesces() {
        let mut arena = Arena::new(100);
        let a = arena.alloc(30).unwrap();
        assert_eq!(arena.largest_free(), 70);
        let b = arena.alloc(30).unwrap();
        let c = arena.alloc(30).unwrap();
        assert_eq!((a.clone(), b.clone(), c.clone()), (0..30, 30..60, 60..90));
        assert!(arena.alloc(20).is_none());
        arena.free(a);
        arena.free(c);
        // c merged with the free tail into 60..100.
        assert_eq!(arena.alloc(41), None);
        arena.free(b);
        // Freed down to nothing above 30: the tail from 30 can go.
        let mut tail = Arena::new(100);
        let low = tail.alloc(30).unwrap();
        let high = tail.alloc(30).unwrap();
        tail.free(high);
        assert_eq!(tail.free_from(100), 30);
        tail.shrink(100, 50);
        assert_eq!(
            (tail.free_from(50), tail.largest_free(), tail.used()),
            (30, 20, 30)
        );
        assert_eq!(tail.alloc(21), None);
        tail.free(low);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.alloc(100), Some(0..100));
    }

    /// `fits_all` answers as allocating would, over free lists of many
    /// shapes.
    #[test]
    fn fits_all_is_first_fit() {
        let mut seed = 7u64;
        let mut next = |n: u32| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        for _ in 0..200 {
            let mut arena = Arena::new(256);
            let mut held = Vec::new();
            for _ in 0..next(20) {
                if let Some(r) = arena.alloc(1 + next(40)) {
                    held.push(r);
                }
            }
            for r in held.into_iter().filter(|_| next(2) == 0) {
                arena.free(r);
            }
            let lens: Vec<u32> = (0..1 + next(6)).map(|_| next(60)).collect();
            let mut copy = arena.clone();
            let allocates = lens.iter().all(|&len| copy.alloc(len).is_some());
            assert_eq!(arena.fits_all(lens.iter().copied()), allocates, "{lens:?}");
        }
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

    /// The lowest free slot goes first, and freeing the highest held slots
    /// lowers where the free tail starts, so a table can shrink to it.
    #[test]
    fn slots_stay_low() {
        let mut slots = Slots::new(8);
        let held: Vec<u32> = (0..6).map(|_| slots.alloc().unwrap()).collect();
        for slot in [1, 4, 5] {
            slots.free(held[slot]);
        }
        assert_eq!(slots.held_below(), 4);
        assert_eq!((slots.alloc(), slots.alloc()), (Some(1), Some(4)));
        for slot in [0, 1, 2, 3, 4] {
            slots.free(slot);
        }
        assert_eq!(
            (slots.held_below(), slots.used(), slots.available()),
            (0, 0, 8)
        );
        slots.shrink(2);
        assert_eq!(slots.available(), 2);
    }

    /// A refusal keeps the size that would have held it then, so room
    /// given back after it doesn't hide what the retry needs.
    #[test]
    fn refusals_keep_the_size_they_needed() {
        let mut arena = Arena::new(100);
        let held = arena.alloc(60).unwrap();
        assert_eq!((arena.alloc(50), arena.alloc(70)), (None, None));
        arena.free(held);
        assert_eq!((arena.take_needed(), arena.take_needed()), (130, 0));
        let mut slots = Slots::new(4);
        slots.alloc().unwrap();
        assert!(slots.reserve(3) && !slots.reserve(5) && !slots.reserve(4));
        assert_eq!((slots.take_needed(), slots.take_needed()), (6, 0));
    }

    #[test]
    fn grows_into_a_block_joined_with_the_free_tail() {
        let mut arena = Arena::new(100);
        let a = arena.alloc(90).unwrap();
        assert!(arena.alloc(20).is_none());
        arena.grow(100, 200);
        assert_eq!(arena.used(), 90);
        assert_eq!(arena.largest_free(), 110);
        assert_eq!(arena.alloc(20), Some(90..110));
        arena.free(a);
        assert_eq!(arena.used(), 20);
    }
}
