//! The backends' resource tables.

use std::collections::HashMap;

/// Resources by id. Ids count up and are never reused, so a stale id finds
/// nothing instead of another resource; removed entries are dropped, so the
/// table holds only live resources.
#[derive(Debug)]
pub struct Slots<T> {
    items: HashMap<u32, T>,
    next: u32,
}

impl<T> Default for Slots<T> {
    fn default() -> Self {
        Self {
            items: HashMap::new(),
            next: 0,
        }
    }
}

impl<T> Slots<T> {
    pub fn insert(&mut self, item: T) -> u32 {
        let id = self.next;
        self.next = id.checked_add(1).expect("resource ids exhausted");
        self.items.insert(id, item);
        id
    }

    /// Replaces a live entry.
    pub fn set(&mut self, id: u32, item: T) {
        self[id] = item;
    }

    /// An id for an entry that [`fill`](Self::fill) adds later.
    pub fn reserve(&mut self) -> u32 {
        let id = self.next;
        self.next = id.checked_add(1).expect("resource ids exhausted");
        id
    }

    /// Adds the entry of a reserved id.
    pub fn fill(&mut self, id: u32, item: T) {
        assert!(id < self.next, "resource {id} was never reserved");
        assert!(
            self.items.insert(id, item).is_none(),
            "resource {id} filled twice"
        );
    }

    pub fn contains(&self, id: u32) -> bool {
        self.items.contains_key(&id)
    }

    /// Live resources.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Takes every live resource.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.items.drain().map(|(_, item)| item)
    }

    pub fn remove(&mut self, id: u32) -> T {
        self.items
            .remove(&id)
            .unwrap_or_else(|| panic!("resource {id} was destroyed or never created"))
    }
}

impl<T> std::ops::Index<u32> for Slots<T> {
    type Output = T;

    fn index(&self, id: u32) -> &T {
        self.items
            .get(&id)
            .unwrap_or_else(|| panic!("resource {id} was destroyed or never created"))
    }
}

impl<T> std::ops::IndexMut<u32> for Slots<T> {
    fn index_mut(&mut self, id: u32) -> &mut T {
        self.items
            .get_mut(&id)
            .unwrap_or_else(|| panic!("resource {id} was destroyed or never created"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_only_live_resources_and_never_reuses_ids() {
        let mut slots = Slots::default();
        for i in 0..1000 {
            let id = slots.insert(i);
            assert_eq!(id, i);
            assert_eq!(slots.remove(id), i);
        }
        assert!(slots.items.is_empty());
        assert_eq!(slots.insert(7), 1000);
    }
}
