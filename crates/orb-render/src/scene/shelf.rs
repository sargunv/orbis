//! Shelf packing for atlas pages: rows of rectangles filled left to right,
//! with freed rectangles reused by exact size.

#[derive(Debug, Default)]
pub(crate) struct Shelves {
    x: u32,
    y: u32,
    height: u32,
    /// Rectangles allocated and not yet freed.
    pub(crate) live: u32,
    /// Freed rectangles, reused by exact size.
    pub(crate) free: Vec<([u32; 2], [u32; 2])>,
}

impl Shelves {
    /// A `w` × `h` rectangle of a `size`² page, if it fits.
    pub(crate) fn allocate(&mut self, size: u32, w: u32, h: u32) -> Option<[u32; 2]> {
        if w > size || h > size {
            return None;
        }
        if let Some(i) = self.free.iter().position(|(_, size)| *size == [w, h]) {
            self.live += 1;
            return Some(self.free.swap_remove(i).0);
        }
        let (x, y, height) = if self.x + w > size {
            (0, self.y + self.height, 0)
        } else {
            (self.x, self.y, self.height)
        };
        if y + h > size {
            return None;
        }
        self.x = x + w;
        self.y = y;
        self.height = height.max(h);
        self.live += 1;
        Some([x, y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shelves_do_not_overlap_and_overflow_does_not_mutate() {
        let mut page = Shelves::default();
        assert_eq!(page.allocate(8, 5, 3), Some([0, 0]));
        assert_eq!(page.allocate(8, 4, 4), Some([0, 3]));
        assert_eq!(page.allocate(8, 9, 1), None);
        assert_eq!(page.allocate(8, 4, 4), Some([4, 3]));
        assert_eq!(page.allocate(8, 1, 2), None);
    }
}
