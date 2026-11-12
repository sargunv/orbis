//! Runtime-shared icon/pattern pages and their stable shader table. Layout
//! resolves names without allocating pixels; a later upload fills the same row.

use std::collections::HashMap;

use orb_hal::*;
use rustc_hash::FxHashSet;

use crate::scene::DataTable;
use crate::scene::shelf::Shelves;

/// Words of a row of the rect table (`types.wgsl`'s `ImageRect`).
const RECT_WORDS: u32 = 8;
pub use orb_layout::{ImageKey, ImageMetadata};

/// What a table row holds: an image as patterns and layers sample it, or
/// a sprite image as symbol icons sample it, with transparent padding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Slot {
    Image(ImageKey),
    Icon(String),
}

/// Straight-alpha host pixels and their sampling role.
#[derive(Debug)]
pub struct ImageUpload<'a> {
    pub size: [u32; 2],
    pub pixels: &'a [u8],
    pub metadata: ImageMetadata,
    pub pattern: bool,
}

/// All tiles bind these resources once per pass. The texture changes when
/// the atlas grows a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageBindings {
    /// The rects, a row of [`RECT_WORDS`] words each.
    pub(crate) table: DataTable,
    pub texture: TextureId,
    pub sampler: SamplerId,
    pub dashes: TextureId,
    /// Texels on a side of an image page.
    pub size: u32,
}

#[derive(Debug)]
struct Entry {
    namespace: u64,
    page: Option<usize>,
    metadata: Option<ImageMetadata>,
    size: Option<[u32; 2]>,
    dash_rows: Option<[u32; 2]>,
    /// The padded rectangle's origin.
    origin: [u32; 2],
    /// Transparent texels allocated around the padded rectangle.
    margin: u32,
    ramp_refs: Option<u32>,
}

/// An upload that found no room while a map held memory
/// ([`ImageAtlas::hold`]): its row reads as a missing image until a frame
/// no map holds in places it.
#[derive(Debug)]
struct Pending {
    id: u32,
    size: [u32; 2],
    pixels: Vec<u8>,
    pattern: bool,
    premultiplied: bool,
}

/// Shelf-packed RGBA8 array pages. Row zero is a transparent missing image.
/// Names and rows remain stable until their style namespace is released.
/// Pages are reset when their last image leaves; no camera work touches them.
///
/// The atlas is the runtime's, shared by its maps, so a renderer's gate
/// ([`Moves`](crate::Moves)) doesn't reach it: each map says whether it
/// holds memory ([`hold`](Self::hold)), and while any does no page is
/// added, which copies every page before it. An image that finds no room
/// then waits, its row's size and metadata set (layout reads them) but not
/// its texels and rect, until [`place_pending`](Self::place_pending) runs
/// in a frame no map holds.
#[derive(Debug)]
pub struct ImageAtlas {
    bindings: ImageBindings,
    size: u32,
    capacity: u32,
    pages: Vec<Shelves>,
    /// The most pages it grows to.
    max_pages: u32,
    names: HashMap<(u64, Slot), u32>,
    entries: Vec<Option<Entry>>,
    free: Vec<u32>,
    namespace: u64,
    lines: crate::scene::line::LineAtlas,
    /// Changes whenever an image's pixels do.
    revision: u64,
    /// The style namespaces of the maps that hold memory.
    holders: FxHashSet<u64>,
    pending: Vec<Pending>,
    /// Namespaces whose icons were placed after they waited: their maps lay
    /// their symbols out again ([`take_icons_placed`](Self::take_icons_placed)),
    /// as a symbol takes its icon's rect when it is laid out.
    icons_placed: FxHashSet<u64>,
}

impl ImageAtlas {
    /// GPU bytes of the icon and pattern pages.
    pub fn bytes(&self) -> u64 {
        u64::from(self.size) * u64::from(self.size) * 4 * self.pages.len() as u64
    }

    /// Image rows held: each style's images and gradient ramps.
    pub fn rows(&self) -> usize {
        self.entries.len() - 1 - self.free.len()
    }

    /// An atlas of `size`² pages, one at first and up to `max_pages` as
    /// images need, and `capacity` image rows.
    pub fn new(gpu: &mut dyn Gpu, size: u32, max_pages: u32, capacity: u32) -> Self {
        assert!(size > 2 && max_pages > 0 && capacity > 0);
        let texture = page_texture(gpu, size, 1);
        let table = DataTable::new(
            gpu,
            "image rects",
            u64::from(capacity) * u64::from(RECT_WORDS),
        );
        let sampler = gpu.create_sampler(&SamplerDesc {
            label: "images",
            filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        let lines = crate::scene::line::LineAtlas::new(gpu);
        Self {
            bindings: ImageBindings {
                table,
                texture,
                sampler,
                dashes: lines.texture,
                size,
            },
            size,
            capacity,
            pages: vec![Shelves::default()],
            max_pages,
            names: HashMap::new(),
            entries: vec![None],
            free: Vec::new(),
            namespace: 0,
            lines,
            revision: 0,
            holders: FxHashSet::default(),
            pending: Vec::new(),
            icons_placed: FxHashSet::default(),
        }
    }

    /// Whether the map of style `namespace` holds memory (its camera
    /// moves): while any map does, no page is added.
    pub fn hold(&mut self, namespace: u64, held: bool) {
        match held {
            true => self.holders.insert(namespace),
            false => self.holders.remove(&namespace),
        };
    }

    /// Whether an image of style `namespace` waits for a page.
    pub fn pending(&self, namespace: u64) -> bool {
        (self.pending.iter()).any(|p| {
            self.entries[p.id as usize]
                .as_ref()
                .is_some_and(|e| e.namespace == namespace)
        })
    }

    /// Places the images that waited for a page, adding the pages they
    /// need, unless a map holds memory. What doesn't fit with every page
    /// added is dropped, its row left missing, and its error returned.
    pub fn place_pending(&mut self, gpu: &mut dyn Gpu) -> Vec<&'static str> {
        if !self.holders.is_empty() {
            return Vec::new();
        }
        let mut errors = Vec::new();
        for p in std::mem::take(&mut self.pending) {
            let entry = self.entries[p.id as usize].as_mut().expect("a pending row");
            let metadata = entry.metadata.take().expect("a pending image's metadata");
            entry.size = None;
            let namespace = entry.namespace;
            let image = ImageUpload {
                size: p.size,
                pixels: &p.pixels,
                metadata,
                pattern: p.pattern,
            };
            match self.upload_pixels(gpu, p.id, image, p.premultiplied) {
                Ok(()) if !p.pattern => {
                    self.icons_placed.insert(namespace);
                }
                Ok(()) => {}
                Err(e) => errors.push(e),
            }
        }
        errors
    }

    /// Whether icons of style `namespace` were placed since it last asked,
    /// after they waited for a page.
    pub fn take_icons_placed(&mut self, namespace: u64) -> bool {
        self.icons_placed.remove(&namespace)
    }

    /// Changes whenever an image's pixels do (an upload, or an image
    /// cleared or released), so what draws them can be drawn again.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn bindings(&self) -> ImageBindings {
        self.bindings
    }

    /// Allocates a style identity, distinct even for maps sharing one runtime.
    pub fn namespace(&mut self) -> u64 {
        self.namespace += 1;
        self.namespace
    }

    pub fn resolve(&mut self, namespace: u64, name: &str) -> Result<u32, &'static str> {
        if name.is_empty() {
            return Ok(0);
        }
        self.reserve(namespace, Slot::Image(ImageKey::Name(name.into())))
    }

    /// A sprite image's icon row: the same pixels as its pattern row, but
    /// padded with transparency, since icon quads cover their padding.
    pub fn resolve_icon(&mut self, namespace: u64, name: &str) -> Result<u32, &'static str> {
        if name.is_empty() {
            return Ok(0);
        }
        self.reserve(namespace, Slot::Icon(name.into()))
    }

    fn reserve(&mut self, namespace: u64, slot: Slot) -> Result<u32, &'static str> {
        if let Some(&id) = self.names.get(&(namespace, slot.clone())) {
            return Ok(id);
        }
        let id = if let Some(id) = self.free.pop() {
            id
        } else {
            if self.entries.len() as u32 == self.capacity {
                return Err("image table is full");
            }
            self.entries.push(None);
            self.entries.len() as u32 - 1
        };
        self.entries[id as usize] = Some(Entry {
            namespace,
            page: None,
            metadata: None,
            size: None,
            dash_rows: None,
            origin: [0; 2],
            margin: 0,
            ramp_refs: matches!(slot, Slot::Image(ImageKey::Ramp { .. })).then_some(0),
        });
        self.names.insert((namespace, slot), id);
        Ok(id)
    }

    /// Materializes a generated key once, reserving a stable shared table id.
    pub fn resolve_key(
        &mut self,
        gpu: &mut dyn Gpu,
        namespace: u64,
        key: &ImageKey,
    ) -> Result<u32, &'static str> {
        if let ImageKey::Name(name) = key {
            return self.resolve(namespace, name);
        }
        let id = self.reserve(namespace, Slot::Image(key.clone()))?;
        if self.entries[id as usize].as_ref().unwrap().size.is_some() {
            return Ok(id);
        }
        match key {
            ImageKey::Dash { array, round } => {
                let (rows, length) = self.lines.insert(gpu, array, *round)?;
                let words = [
                    0,
                    rows[0] + rows[1] / 2,
                    crate::scene::line::WIDTH,
                    rows[1] - 1,
                    0,
                    length.to_bits(),
                    0,
                    0,
                ];
                self.bindings.table.write(gpu, id * RECT_WORDS, &words);
                let entry = self.entries[id as usize].as_mut().unwrap();
                entry.size = Some([crate::scene::line::WIDTH, rows[1]]);
                entry.dash_rows = Some(rows);
            }
            ImageKey::Ramp { colors, nearest } => {
                // Fold long ramps into 256-wide rectangles. Shader addressing
                // stays linear across rows, without filtering across neighbours.
                let pixels: Vec<u8> = colors.iter().flat_map(|c| c.to_le_bytes()).collect();
                self.upload_pixels(
                    gpu,
                    id,
                    ImageUpload {
                        size: [256, colors.len() as u32 / 256],
                        pixels: &pixels,
                        metadata: ImageMetadata {
                            sdf: *nearest,
                            ..Default::default()
                        },
                        pattern: false,
                    },
                    true,
                )?;
            }
            ImageKey::Name(_) => unreachable!(),
        }
        Ok(id)
    }

    pub fn metadata(&self, id: u32) -> Option<&ImageMetadata> {
        self.entries.get(id as usize)?.as_ref()?.metadata.as_ref()
    }

    /// An uploaded image's padded rectangle origin and page layer.
    pub fn rect(&self, id: u32) -> Option<[u32; 3]> {
        let entry = self.entries.get(id as usize)?.as_ref()?;
        let layer = entry.page?;
        Some([entry.origin[0], entry.origin[1], layer as u32])
    }

    pub fn size(&self, id: u32) -> Option<[u32; 2]> {
        self.entries.get(id as usize)?.as_ref()?.size
    }

    /// Uploads straight RGBA8, premultiplying once. Pattern padding wraps
    /// each edge, with transparent corners, as GL JS ImageAtlas does. Icons
    /// use transparent padding; one name can have separate icon/pattern rows.
    pub fn upload(
        &mut self,
        gpu: &mut dyn Gpu,
        id: u32,
        image: ImageUpload<'_>,
    ) -> Result<(), &'static str> {
        self.upload_pixels(gpu, id, image, false)
    }

    fn upload_pixels(
        &mut self,
        gpu: &mut dyn Gpu,
        id: u32,
        image: ImageUpload<'_>,
        premultiplied: bool,
    ) -> Result<(), &'static str> {
        self.revision += 1;
        let ImageUpload {
            size: [width, height],
            pixels: rgba,
            metadata,
            pattern,
        } = image;
        // Icon quads cover their padding, so filtering at their edge
        // reaches one texel past it: keep that texel transparent too, not
        // a neighbour's (a pattern's padding wraps its image).
        let margin = u32::from(!pattern);
        let border = 2 + 2 * margin;
        if width == 0
            || height == 0
            || width + border > self.size
            || height + border > self.size
            || rgba.len() as u64 != u64::from(width) * u64::from(height) * 4
            || !metadata.pixel_ratio.is_finite()
            || metadata.pixel_ratio <= 0.0
        {
            return Err("invalid atlas image");
        }
        let entry = self
            .entries
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or("unknown image")?;
        if entry.size.is_some() {
            return Err("image already uploaded");
        }
        let w = width + border;
        let h = height + border;
        let size = self.size;
        let place = |pages: &mut Vec<Shelves>| {
            pages
                .iter_mut()
                .enumerate()
                .find_map(|(i, p)| p.allocate(size, w, h).map(|pos| (i, pos)))
        };
        let (layer, [x0, y0]) = match place(&mut self.pages) {
            Some(found) => found,
            None if !self.holders.is_empty() && (self.pages.len() as u32) < self.max_pages => {
                self.pending.push(Pending {
                    id,
                    size: [width, height],
                    pixels: rgba.to_vec(),
                    pattern,
                    premultiplied,
                });
                let entry = self.entries[id as usize].as_mut().expect("checked above");
                entry.metadata = Some(metadata);
                entry.size = Some([width, height]);
                return Ok(());
            }
            None if self.add_page(gpu) => place(&mut self.pages).ok_or("image atlas is full")?,
            None => return Err("image atlas is full"),
        };
        let padded = padded(width, height, rgba, pattern, premultiplied);
        let pixels = if margin == 0 {
            padded
        } else {
            let row = (width as usize + 2) * 4;
            let mut out = vec![0; w as usize * h as usize * 4];
            for (y, src) in padded.chunks_exact(row).enumerate() {
                let at = ((y + 1) * w as usize + 1) * 4;
                out[at..at + row].copy_from_slice(src);
            }
            out
        };
        gpu.write_texture(
            &TextureRegion {
                texture: self.bindings.texture,
                x: x0,
                y: y0,
                layer: layer as u32,
                width: w,
                height: h,
                level: 0,
            },
            &pixels,
        );
        let (x, y) = (x0 + margin, y0 + margin);
        let words = [
            x + 1,
            y + 1,
            width,
            height,
            layer as u32,
            metadata.pixel_ratio.to_bits(),
            u32::from(metadata.sdf),
            0,
        ];
        self.bindings.table.write(gpu, id * RECT_WORDS, &words);
        let entry = self.entries[id as usize].as_mut().expect("checked above");
        entry.page = Some(layer);
        entry.origin = [x, y];
        entry.margin = margin;
        entry.metadata = Some(metadata);
        entry.size = Some([width, height]);
        Ok(())
    }

    /// Adds a page, its pages so far copied into a texture one larger;
    /// whether there was room for one.
    fn add_page(&mut self, gpu: &mut dyn Gpu) -> bool {
        let pages = self.pages.len() as u32;
        if pages >= self.max_pages {
            return false;
        }
        let grown = page_texture(gpu, self.size, pages + 1);
        gpu.submit(&[orb_hal::Pass::CopyTexture {
            source: self.bindings.texture,
            destination: grown,
            copies: vec![orb_hal::TextureCopy {
                from: [0; 3],
                to: [0; 3],
                size: [self.size, self.size, pages],
            }],
        }]);
        gpu.destroy_texture(std::mem::replace(&mut self.bindings.texture, grown));
        self.pages.push(Shelves::default());
        true
    }

    /// Holds a generated ramp while a resident tile references its row.
    pub fn retain_ramp(&mut self, id: u32) {
        *self.entries[id as usize]
            .as_mut()
            .unwrap()
            .ramp_refs
            .as_mut()
            .expect("gradient ramp") += 1;
    }

    /// Releases a tile's ramp leases. Its exact atlas rectangles and table ids
    /// become reusable when the last resident tile lets go.
    pub fn release_ramps(&mut self, gpu: &mut dyn Gpu, ids: impl IntoIterator<Item = u32>) {
        for id in ids {
            let Some(entry) = self.entries.get_mut(id as usize).and_then(Option::as_mut) else {
                continue;
            };
            let refs = entry.ramp_refs.as_mut().expect("gradient ramp");
            assert!(*refs > 0, "unbalanced ramp lease");
            *refs -= 1;
            if *refs == 0 {
                self.names.retain(|_, row| *row != id);
                self.erase(gpu, id);
            }
        }
    }

    fn erase(&mut self, gpu: &mut dyn Gpu, id: u32) {
        self.vacate(gpu, id);
        let entry = self.entries[id as usize].take().unwrap();
        if let Some(rows) = entry.dash_rows {
            self.lines.release(rows);
        }
        self.free.push(id);
    }

    /// Frees a row's texels and empties its table words; the id stays.
    fn vacate(&mut self, gpu: &mut dyn Gpu, id: u32) {
        self.revision += 1;
        self.pending.retain(|p| p.id != id);
        let entry = self.entries[id as usize].as_mut().unwrap();
        if let Some(page) = entry.page.take() {
            let p = &mut self.pages[page];
            p.live -= 1;
            let [w, h] = entry.size.unwrap();
            let m = entry.margin;
            p.free.push((
                [entry.origin[0] - m, entry.origin[1] - m],
                [w + 2 + 2 * m, h + 2 + 2 * m],
            ));
            if p.live == 0 {
                *p = Shelves::default();
            }
        }
        entry.size = None;
        entry.metadata = None;
        self.bindings
            .table
            .write(gpu, id * RECT_WORDS, &[0; RECT_WORDS as usize]);
    }

    /// Empties a style's named image, as a pattern and as an icon, for new
    /// pixels or none: its rows keep their ids, which tiles hold, and read
    /// as a missing image until uploaded again.
    pub fn clear(&mut self, gpu: &mut dyn Gpu, namespace: u64, name: &str) {
        for slot in [
            Slot::Image(ImageKey::Name(name.into())),
            Slot::Icon(name.into()),
        ] {
            if let Some(&id) = self.names.get(&(namespace, slot)) {
                self.vacate(gpu, id);
            }
        }
    }

    pub fn release(&mut self, gpu: &mut dyn Gpu, namespace: u64) {
        self.holders.remove(&namespace);
        self.icons_placed.remove(&namespace);
        self.names.retain(|(n, _), _| *n != namespace);
        for id in 1..self.entries.len() {
            if self.entries[id]
                .as_ref()
                .is_some_and(|e| e.namespace == namespace)
            {
                self.erase(gpu, id as u32);
            }
        }
    }
}

/// The pages' texture array.
fn page_texture(gpu: &mut dyn Gpu, size: u32, pages: u32) -> orb_hal::TextureId {
    gpu.create_texture(&TextureDesc {
        label: "images",
        width: size,
        height: size,
        array_layers: pages,
        dimension: orb_hal::TextureViewDimension::D2Array,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    })
}

fn padded(width: u32, height: u32, rgba: &[u8], pattern: bool, premultiplied: bool) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0; (w + 2) * (h + 2) * 4];
    for y in 0..h + 2 {
        for x in 0..w + 2 {
            if (x == 0 || x == w + 1) && (y == 0 || y == h + 1) {
                continue;
            }
            if !pattern && (x == 0 || x == w + 1 || y == 0 || y == h + 1) {
                continue;
            }
            let sx = (x + w - 1) % w;
            let sy = (y + h - 1) % h;
            let src = &rgba[(sy * w + sx) * 4..][..4];
            let dst = &mut out[(y * (w + 2) + x) * 4..][..4];
            for c in 0..3 {
                dst[c] = if premultiplied {
                    src[c]
                } else {
                    ((u16::from(src[c]) * u16::from(src[3]) + 127) / 255) as u8
                };
            }
            dst[3] = src[3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_wraps_edges_and_premultiplies_with_transparent_corners() {
        let p = padded(2, 1, &[255, 0, 0, 128, 0, 255, 0, 255], true, false);
        let pixel = |x: usize, y: usize| &p[(y * 4 + x) * 4..][..4];
        assert_eq!(pixel(1, 1), [128, 0, 0, 128]);
        assert_eq!(pixel(0, 1), [0, 255, 0, 255]);
        assert_eq!(pixel(3, 1), [128, 0, 0, 128]);
        assert_eq!(pixel(1, 0), pixel(1, 1));
        assert_eq!(pixel(0, 0), [0; 4]);
        let icon = padded(2, 1, &[255; 8], false, false);
        assert_eq!(&icon[..16], [0; 16]);
    }

    /// While a map holds memory the atlas adds no page, which would copy
    /// every page before it: an image that finds no room waits, its size
    /// and metadata known but not its rect, and goes in at the first frame
    /// no map holds, an icon asking for its symbols laid out again.
    #[test]
    fn an_image_waits_for_a_page_while_a_map_holds_memory() {
        let Ok(mut gpu) = pollster::block_on(orb_hal_wgpu::WgpuGpu::offscreen()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        // A page of 8² texels holds one 4² image and its padding.
        let mut atlas = ImageAtlas::new(&mut gpu, 8, 3, 8);
        let (mine, other) = (atlas.namespace(), atlas.namespace());
        let upload = |atlas: &mut ImageAtlas, gpu: &mut dyn Gpu, name: &str, pattern: bool| {
            let id = match pattern {
                true => atlas.resolve(mine, name),
                false => atlas.resolve_icon(mine, name),
            };
            let id = id.unwrap();
            let image = ImageUpload {
                size: [4, 4],
                pixels: &[255; 64],
                metadata: ImageMetadata {
                    pixel_ratio: 1.0,
                    ..Default::default()
                },
                pattern,
            };
            atlas.upload(gpu, id, image).unwrap();
            id
        };
        let first = upload(&mut atlas, &mut gpu, "first", true);
        assert_eq!(atlas.rect(first), Some([0, 0, 0]));
        // Another map's camera moves.
        atlas.hold(other, true);
        let before = gpu.counters();
        let pattern = upload(&mut atlas, &mut gpu, "pattern", true);
        let icon = upload(&mut atlas, &mut gpu, "icon", false);
        assert_eq!((atlas.rect(pattern), atlas.rect(icon)), (None, None));
        assert_eq!(atlas.size(icon), Some([4, 4]));
        assert!(atlas.pending(mine));
        assert!(atlas.place_pending(&mut gpu).is_empty());
        assert_eq!(atlas.rect(pattern), None);
        let moved = gpu.counters().since(&before);
        assert_eq!((moved.textures_created, moved.bytes_copied), (0, 0));
        assert!(!atlas.take_icons_placed(mine));
        // It rests.
        atlas.hold(other, false);
        assert!(atlas.place_pending(&mut gpu).is_empty());
        assert_eq!(atlas.rect(pattern), Some([0, 0, 1]));
        assert_eq!(atlas.rect(icon), Some([1, 1, 2]));
        assert!(!atlas.pending(mine));
        assert!(atlas.take_icons_placed(mine));
        assert!(!atlas.take_icons_placed(mine));
    }
}
