//! The glyph atlas (decision 0009): SDF bitmaps in R8 array-texture pages,
//! shelf packed, and a stable table of glyph rows that symbol shaders read.
//!
//! Rows are reserved before their bitmaps exist, so a tile's quads can name
//! glyphs by row at upload; a reserved row is transparent until its bitmap
//! arrives, and nothing that refers to it changes then. The atlas knows
//! rows only: which glyph a row holds is the runtime's business.

use orb_hal::{BufferId, Gpu, TextureFormat, TextureId, TextureRegion};

use crate::shelf::Shelves;
use crate::{OutOfSpace, PageTexture, TilePage};

/// Texels on a side of a glyph page.
pub const GLYPH_PAGE: u32 = 512;
/// Transparent texels around every bitmap, as GL JS's glyph atlas pads.
pub const GLYPH_PADDING: u32 = 1;
/// One more transparent texel around the padded rectangle: glyph quads
/// cover their padding, so filtering at their edge reaches one texel past
/// it, which must not be whichever glyph was packed next to it.
const GLYPH_MARGIN: u32 = 1;
/// Words per glyph row: padded origin (`x | y << 16`), padded size
/// (`w | h << 16`), page layer, and a zero.
pub const GLYPH_ROW_WORDS: usize = 4;

#[derive(Debug, Clone, Copy)]
struct Row {
    /// Page layer and allocated rectangle (the padded one and its
    /// margin), once the bitmap is in.
    placed: Option<(u32, [u32; 2], [u32; 2])>,
}

/// Glyph SDF pages and their row table. Row 0 is always empty: it stands
/// for a glyph nothing could draw.
#[derive(Debug)]
pub struct GlyphAtlas {
    page: TilePage,
    /// Per layer of `page`, its shelves.
    shelves: Vec<Shelves>,
    /// The row table: words of a scene buffer from `base`.
    table: BufferId,
    base: u32,
    capacity: u32,
    rows: Vec<Option<Row>>,
    free: Vec<u32>,
}

impl GlyphAtlas {
    /// An atlas of at most `capacity` rows, its row table in `table` from
    /// word `base` (`capacity` × [`GLYPH_ROW_WORDS`] words).
    pub fn new(gpu: &mut dyn Gpu, capacity: u32, table: BufferId, base: u32) -> Self {
        assert!(capacity > 1);
        let page = TilePage::new(
            gpu,
            &[PageTexture {
                label: "glyphs",
                width: GLYPH_PAGE,
                height: GLYPH_PAGE,
                format: TextureFormat::R8Unorm,
                usage: orb_hal::TextureUsages::TEXTURE_BINDING,
            }],
        );
        gpu.write_buffer(table, u64::from(base) * 4, &[0; GLYPH_ROW_WORDS * 4]);
        Self {
            page,
            shelves: Vec::new(),
            table,
            base,
            capacity,
            rows: vec![Some(Row { placed: None })],
            free: Vec::new(),
        }
    }

    /// The R8 page array, replaced when the atlas grows (see
    /// [`revision`](Self::revision)).
    pub fn texture(&self) -> TextureId {
        self.page.texture(0)
    }

    /// The row table's first word in its buffer: [`GLYPH_ROW_WORDS`] per
    /// row.
    pub fn table_base(&self) -> u32 {
        self.base
    }

    fn row_offset(&self, row: u32) -> u64 {
        (u64::from(self.base) + u64::from(row) * GLYPH_ROW_WORDS as u64) * 4
    }

    /// Changes whenever [`texture`](Self::texture) is replaced, so bind
    /// groups over it are rebuilt.
    pub fn revision(&self) -> u32 {
        self.page.revision()
    }

    /// Rows in use, the empty row included.
    pub fn len(&self) -> u32 {
        (self.rows.len() - self.free.len()) as u32
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 1
    }

    /// A new transparent row.
    pub fn reserve(&mut self) -> Result<u32, OutOfSpace> {
        let row = match self.free.pop() {
            Some(row) => row,
            None if (self.rows.len() as u32) < self.capacity => {
                self.rows.push(None);
                self.rows.len() as u32 - 1
            }
            None => return Err(OutOfSpace),
        };
        self.rows[row as usize] = Some(Row { placed: None });
        Ok(row)
    }

    /// Writes a `size[0]` × `size[1]` R8 bitmap into a reserved row, with
    /// [`GLYPH_PADDING`] transparent texels around it.
    pub fn upload(
        &mut self,
        gpu: &mut dyn Gpu,
        row: u32,
        size: [u32; 2],
        bitmap: &[u8],
    ) -> Result<(), OutOfSpace> {
        assert_eq!(bitmap.len() as u64, u64::from(size[0]) * u64::from(size[1]));
        let entry = self.rows[row as usize].as_ref().expect("a reserved row");
        assert!(entry.placed.is_none(), "glyph row {row} already uploaded");
        if size[0] == 0 || size[1] == 0 {
            return Ok(());
        }
        let padded = size.map(|s| s + 2 * GLYPH_PADDING);
        let allocated = padded.map(|s| s + 2 * GLYPH_MARGIN);
        let found = self.shelves.iter_mut().enumerate().find_map(|(layer, s)| {
            s.allocate(GLYPH_PAGE, allocated[0], allocated[1])
                .map(|at| (layer as u32, at))
        });
        let (layer, origin) = match found {
            Some(found) => found,
            None => {
                let layer = self.page.alloc(gpu)?;
                if self.shelves.len() <= layer as usize {
                    self.shelves
                        .resize_with(layer as usize + 1, Shelves::default);
                }
                let origin = self.shelves[layer as usize]
                    .allocate(GLYPH_PAGE, allocated[0], allocated[1])
                    .ok_or(OutOfSpace)?;
                (layer, origin)
            }
        };
        let (w, h) = (allocated[0] as usize, allocated[1] as usize);
        let mut texels = vec![0u8; w * h];
        let pad = (GLYPH_PADDING + GLYPH_MARGIN) as usize;
        for (y, line) in bitmap.chunks(size[0] as usize).enumerate() {
            texels[(y + pad) * w + pad..][..line.len()].copy_from_slice(line);
        }
        gpu.write_texture(
            &TextureRegion {
                texture: self.page.texture(0),
                x: origin[0],
                y: origin[1],
                layer,
                width: allocated[0],
                height: allocated[1],
            },
            &texels,
        );
        let inner = origin.map(|o| o + GLYPH_MARGIN);
        let words = [
            inner[0] | inner[1] << 16,
            padded[0] | padded[1] << 16,
            layer,
            0,
        ];
        gpu.write_buffer(
            self.table,
            self.row_offset(row),
            bytemuck::cast_slice(&words),
        );
        self.rows[row as usize].as_mut().expect("reserved").placed =
            Some((layer, origin, allocated));
        Ok(())
    }

    /// Frees a row and its texels.
    pub fn release(&mut self, gpu: &mut dyn Gpu, row: u32) {
        assert_ne!(row, 0, "the empty row is never released");
        let entry = self.rows[row as usize].take().expect("a reserved row");
        if let Some((layer, origin, size)) = entry.placed {
            let shelves = &mut self.shelves[layer as usize];
            shelves.live -= 1;
            shelves.free.push((origin, size));
            if shelves.live == 0 {
                *shelves = Shelves::default();
            }
            gpu.write_buffer(self.table, self.row_offset(row), &[0; GLYPH_ROW_WORDS * 4]);
        }
        self.free.push(row);
    }
}
