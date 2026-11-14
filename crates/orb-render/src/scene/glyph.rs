//! The glyph atlas: SDF bitmaps in R8 array-texture pages,
//! shelf packed, and a stable table of glyph rows that symbol shaders read.
//!
//! Rows are reserved before their bitmaps exist, so a tile's quads can name
//! glyphs by row at upload; a reserved row is transparent until its bitmap
//! arrives, and nothing that refers to it changes then. The atlas knows
//! rows only: which glyph a row holds is the runtime's business.
//!
//! The pages are a [`TilePage`] of [`PAGE_CHUNKS`] chunks, which the symbol
//! draws bind each of: the atlas grows by its next chunk, copying nothing,
//! and a row's layer word names the chunk its bitmap is in.

use orb_hal::{Gpu, TextureFormat, TextureId};

use crate::scene::shelf::Shelves;
use crate::scene::texels::AtlasTexels;
use crate::scene::{OutOfSpace, PAGE_CHUNKS, PageTexture, TableWriter, TilePage};

/// Texels on a side of a glyph page.
pub const GLYPH_PAGE: u32 = 512;
/// Transparent texels around every bitmap, as GL JS's glyph atlas pads.
pub const GLYPH_PADDING: u32 = 1;
/// One more transparent texel around the padded rectangle: glyph quads
/// cover their padding, so filtering at their edge reaches one texel past
/// it, which must not be whichever glyph was packed next to it.
const GLYPH_MARGIN: u32 = 1;
/// Glyph pages the atlas starts with: 2 MiB, which Latin labels of a
/// style's few fonts take a fraction of. It grows as bitmaps need, by a
/// chunk as large as those before it (8, 8, 16 and 32 pages).
const FIRST_PAGES: u32 = 8;
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
    /// Per layer word of `page` (in order: by chunk, then layer), its
    /// shelves.
    shelves: std::collections::BTreeMap<u32, Shelves>,
    /// The row table: the first words of the scene's property table.
    table: TableWriter,
    capacity: u32,
    rows: Vec<Option<Row>>,
    free: Vec<u32>,
    /// Per chunk of `page`, its texels, uploaded once a frame
    /// ([`flush`](Self::flush)).
    texels: Vec<AtlasTexels>,
}

impl GlyphAtlas {
    /// Destroys the glyph pages; the row table is the scene's.
    pub fn destroy(self, gpu: &mut dyn Gpu) {
        self.page.destroy(gpu);
    }

    /// GPU bytes of the glyph pages.
    pub fn bytes(&self) -> u64 {
        self.page.bytes()
    }

    /// What moving the glyph pages did since the last call.
    pub(crate) fn take_resized(&mut self) -> crate::scene::Resized {
        self.page.take_resized()
    }

    /// An atlas of at most `capacity` rows, its row table the first
    /// `capacity` × [`GLYPH_ROW_WORDS`] words of `table`.
    pub(crate) fn new(gpu: &mut dyn Gpu, capacity: u32, table: TableWriter) -> Self {
        assert!(capacity > 1);
        let page = TilePage::new(
            gpu,
            &[PageTexture {
                label: "glyphs",
                width: GLYPH_PAGE,
                height: GLYPH_PAGE,
                format: TextureFormat::R8Unorm,
                usage: orb_hal::TextureUsages::TEXTURE_BINDING,
                mip_levels: 1,
            }],
            (FIRST_PAGES, PAGE_CHUNKS),
        );
        table.write_words(gpu, 0, &[0; GLYPH_ROW_WORDS]);
        Self {
            page,
            shelves: Default::default(),
            table,
            capacity,
            rows: vec![Some(Row { placed: None })],
            free: Vec::new(),
            texels: Vec::new(),
        }
    }

    /// Uploads the bitmaps written since the last flush: one write a page
    /// layer, before the frame that draws them.
    pub(crate) fn flush(&mut self, gpu: &mut dyn Gpu) {
        for (k, texture, _) in self.page.chunks(0) {
            if let Some(texels) = self.texels.get_mut(k) {
                texels.flush(gpu, texture);
            }
        }
    }

    /// The R8 page arrays by chunk, `placeholder` where there's none yet:
    /// a chunk comes as the atlas grows (see [`revision`](Self::revision)).
    pub fn textures(&self, placeholder: TextureId) -> [TextureId; PAGE_CHUNKS] {
        self.page.textures(0, placeholder)
    }

    /// The row table moved, at the same offset, into a larger table.
    pub(crate) fn set_table(&mut self, table: TableWriter) {
        self.table = table;
    }

    fn row_offset(&self, row: u32) -> u32 {
        row * GLYPH_ROW_WORDS as u32
    }

    /// Changes whenever [`textures`](Self::textures) do, so bind groups
    /// over them are rebuilt.
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
    /// `GLYPH_PADDING` transparent texels around it.
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
        let found = self.shelves.iter_mut().find_map(|(&layer, s)| {
            s.allocate(GLYPH_PAGE, allocated[0], allocated[1])
                .map(|at| (layer, at))
        });
        let (layer, origin) = match found {
            Some(found) => found,
            None => {
                let layer = self.page.alloc(gpu, true)?;
                let origin = (self.shelves.entry(layer).or_default())
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
        let (k, at) = TilePage::split(layer);
        if self.texels.len() <= k {
            self.texels
                .resize_with(k + 1, || AtlasTexels::new(GLYPH_PAGE, GLYPH_PAGE, 1));
        }
        self.texels[k].write(at, origin, allocated, &texels);
        let inner = origin.map(|o| o + GLYPH_MARGIN);
        let words = [
            inner[0] | inner[1] << 16,
            padded[0] | padded[1] << 16,
            layer,
            0,
        ];
        self.table.write_words(gpu, self.row_offset(row), &words);
        self.rows[row as usize].as_mut().expect("reserved").placed =
            Some((layer, origin, allocated));
        Ok(())
    }

    /// Frees a row and its texels.
    pub fn release(&mut self, gpu: &mut dyn Gpu, row: u32) {
        assert_ne!(row, 0, "the empty row is never released");
        let entry = self.rows[row as usize].take().expect("a reserved row");
        if let Some((layer, origin, size)) = entry.placed {
            let shelves = self.shelves.get_mut(&layer).expect("a placed row's layer");
            shelves.live -= 1;
            shelves.free.push((origin, size));
            if shelves.live == 0 {
                *shelves = Shelves::default();
                let (k, at) = TilePage::split(layer);
                self.texels[k].forget(at);
            }
            self.table
                .write_words(gpu, self.row_offset(row), &[0; GLYPH_ROW_WORDS]);
        }
        self.free.push(row);
    }
}
