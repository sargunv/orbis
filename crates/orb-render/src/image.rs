//! Image families: raster, hillshade and color-relief layers, the tile
//! pages they sample, and the DEM prepare pass (`Prepare`).
//!
//! Each image tile is a one-chunk quad mesh whose property row
//! ([`IMAGE_ROW_WORDS`]) names its layer in the source's [`TilePage`], so
//! tiles go through the same cull, placements and fallback clipping as
//! vector tiles. A node's bind group holds its source's texture array;
//! it is rebuilt, and the bundle re-recorded, only when the node changes
//! pages or a page grows.
//!
//! DEM tiles are prepared when the neighbours the view draws beside them
//! change (`Renderer::pad_dems`): `pad` fills a two-texel border from those
//! neighbours (GL JS's `backfillBorder`), `slope` computes hillshade
//! slopes. Each is a draw into the tile's layer.

use crate::release::Release;
use crate::scene::{PageTexture, Pixels, TilePage};
use orb_hal::*;
use orb_shaders::{COLOR_RELIEF_SHADER, HILLSHADE_SHADER, HILLSHADE_SLOPES_SHADER, RASTER_SHADER};

use crate::pipeline::Draw;
use crate::{Family, OutOfSpace, TargetFormat};

/// Words per image tile property row: texture layer; the latitudes of the
/// tile's top and bottom edges as f32; then its fade ([`ImageFade::words`]):
/// when it arrived, and the ancestor it cross-fades from.
pub const IMAGE_ROW_WORDS: usize = 6;

/// How an image tile fades in (`raster-fade-duration`, GL JS's
/// `getFadeValues`): from the loaded ancestor that covered it, by
/// cross-fading, or without one by its opacity, over the duration from
/// when it arrived. Shaders measure the time against the frame's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageFade {
    /// When it arrived, in the frame's milliseconds; [`ImageFade::NONE`]'s
    /// is long past.
    pub arrived: f32,
    /// The ancestor's layer in the same page, how many zooms up it is, and
    /// the tile's column and row offsets within it at the tile's zoom.
    pub ancestor: Option<(u32, u8, [u16; 2])>,
}

impl ImageFade {
    /// Shown at once: a tile that replaced resident descendants, or one
    /// added to a node later.
    pub const NONE: Self = Self {
        arrived: f32::MIN,
        ancestor: None,
    };

    /// The row's last three words.
    pub fn words(self) -> [u32; 3] {
        let (layer, at) = match self.ancestor {
            Some((layer, depth, [x, y])) => (
                layer,
                u32::from(depth) | u32::from(x & 0xfff) << 8 | u32::from(y & 0xfff) << 20,
            ),
            None => (u32::MAX, 0),
        };
        [self.arrived.to_bits(), layer, at]
    }
}

/// Tile units per side of an image quad (GL JS's `EXTENT`).
pub const IMAGE_EXTENT: u32 = 8192;

/// A page of tile textures, from [`Renderer::create_page`](crate::Renderer::create_page).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageId(u32);

/// What a page holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageKind {
    /// Raster images, premultiplied.
    Raster,
    /// DEM tiles: raw elevations, a padded copy and hillshade slopes.
    Dem,
    /// DEM tiles a terrain stands on: raw elevations and a padded copy.
    /// Nothing shades from them, so they have no slopes: a third fewer
    /// bytes, and half the prepare draws.
    Terrain,
}

impl PageKind {
    /// Whether its tiles are DEM tiles, prepared on arrival.
    fn dem(self) -> bool {
        matches!(self, Self::Dem | Self::Terrain)
    }
}

/// A DEM tile to prepare: its layer, its resident neighbours' layers and
/// how to read its elevations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemPrepare {
    pub layer: u32,
    /// The tiles at (dx, dy) = (−1, −1), (0, −1), (1, −1), (−1, 0), (1, 0),
    /// (−1, 1), (0, 1), (1, 1), at the same zoom.
    pub neighbours: [Option<u32>; 8],
    /// `None` prepares the whole tile. `Some(mask)` prepares only the
    /// borders toward the neighbours whose bits are set, in
    /// [`neighbours`](Self::neighbours)' order: the layer was prepared
    /// before, and only those neighbours changed since. A border is a
    /// function of its own neighbour, so the rest stands.
    pub borders: Option<u8>,
    /// The tile's zoom level.
    pub zoom: f32,
    /// GL JS's unpack vector: metres = `r·x + g·y + b·z − w`.
    pub unpack: [f32; 4],
}

use orb_shaders::layout::prepare_job;

impl DemPrepare {
    /// `prepare.wgsl`'s `Job`.
    fn words(&self) -> Vec<u32> {
        use prepare_job as n;
        let neighbours = self.neighbours.map(|n| n.unwrap_or(u32::MAX));
        let mut b = orb_shaders::Block::new(n::WORDS);
        b.f(n::UNPACK, &self.unpack).u(n::NEIGHBOURS, &neighbours);
        b.u(n::LAYER, &[self.layer]).f(n::ZOOM, &[self.zoom]);
        b.0
    }
}

/// Texture array indices of a DEM page.
const DEM_RAW: usize = 0;
const DEM_PADDED: usize = 1;
const DEM_SLOPES: usize = 2;

#[derive(Debug)]
struct Page {
    kind: PageKind,
    page: TilePage,
    pending: Vec<DemPrepare>,
    /// Raster layers uploaded since their mip levels were last made.
    mip_layers: Vec<u32>,
    /// Of those, the ones whose tile reaches past its area, and the
    /// texels inside it, `[x0, y0, x1, y1)`: their edges extend first.
    edges: Vec<(u32, [u32; 4])>,
    mips: Option<Mips>,
    /// The prepare draws' targets.
    targets: Option<prepare::Targets>,
}

/// A raster page's mip bindings, for one revision of its
/// texture: views of each level, and the downsampler's chain over them
/// (whole layers).
#[derive(Debug)]
struct Mips {
    revision: u32,
    views: Vec<TextureId>,
    chain: crate::downsample::Chain,
}

fn destroy_mips(gpu: &mut dyn Gpu, mips: Mips) {
    (mips.chain, mips.views).release(gpu);
}

/// What an image node's bind group holds.
#[derive(Debug, Default)]
pub(crate) struct NodeImage {
    page: Option<PageId>,
    /// Color-relief ramp: colours in row 0, packed elevations in row 1.
    ramp: Option<TextureId>,
    /// The page revision the group was built for; `None` to rebuild.
    built: Option<u32>,
}

/// What a frame's prepare did.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Prepared {
    pub(crate) tiles: u32,
    pub(crate) draws: u32,
}

/// One target format's raster, hillshade and color-relief pipelines, each
/// made for the first node of its family.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ImagePipelines(pub(crate) [Option<RenderPipelineId>; 3]);

fn image_index(family: Family) -> usize {
    match family {
        Family::Raster => 0,
        Family::Hillshade => 1,
        Family::ColorRelief => 2,
        _ => unreachable!("not an image family"),
    }
}

impl ImagePipelines {
    pub(crate) fn get(&self, family: Family) -> Option<RenderPipelineId> {
        self.0[image_index(family)]
    }
}

#[derive(Debug)]
pub(crate) struct Images {
    target: TargetFormat,
    draw_layout: BindGroupLayoutId,
    image_layout: BindGroupLayoutId,
    relief_layout: BindGroupLayoutId,
    /// Raster, hillshade and color relief, made for their first node.
    pipelines: ImagePipelines,
    /// Shaded hillshades' slopes into the drape atlas, made for the first.
    slopes: Option<RenderPipelineId>,
    linear: SamplerId,
    nearest: SamplerId,
    /// Bound by nodes whose source has no page yet.
    placeholder: TextureId,
    placeholder_ramp: TextureId,
    /// The prepare draws, made for the first DEM page.
    prepare: Option<prepare::Draws>,
    pages: Vec<Option<Page>>,
    /// What the last frame's levels and edge extensions used (scratch
    /// layers, drawn levels' targets), released by the next mip pass.
    edges: crate::downsample::Transient,
    /// What removed pages' moves did since the last
    /// [`take_resized`](Self::take_resized).
    resized: crate::scene::Resized,
}

impl Images {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        draw_layout: BindGroupLayoutId,
    ) -> Self {
        use BindGroupLayoutEntry as E;
        let fs = ShaderStages::FRAGMENT;
        let image_entries = [
            E::uniform(0, ShaderStages::VERTEX | fs),
            E::texture(1, fs, true, TextureViewDimension::D2Array),
            E::sampler(2, fs, true),
            E::sampler(3, fs, true),
            // The texture again, for the nearest sampler (`image.wgsl`).
            E::texture(5, fs, true, TextureViewDimension::D2Array),
        ];
        let image_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "image node",
            entries: &image_entries,
        });
        let mut relief_entries = image_entries.to_vec();
        relief_entries.push(E::texture(4, fs, true, TextureViewDimension::D2));
        let relief_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "color relief node",
            entries: &relief_entries,
        });
        // Raster tiles are mipmapped (GL JS's `useMipmap`) and sampled
        // anisotropically, as GL JS does above 20° of pitch; pages without
        // mip levels sample their one level.
        let mut sampler = |label, filter, anisotropy| {
            gpu.create_sampler(&SamplerDesc {
                label,
                filter,
                mipmap_filter: filter,
                anisotropy,
            })
        };
        let linear = sampler("linear", FilterMode::Linear, 16);
        let nearest = sampler("nearest", FilterMode::Nearest, 1);
        let mut placeholder = |label, height, layers, dimension| {
            gpu.create_texture(&TextureDesc {
                label,
                width: 1,
                height,
                array_layers: layers,
                dimension,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_levels: 1,
            })
        };
        let placeholder_array = placeholder("no tiles", 1, 1, TextureViewDimension::D2Array);
        let placeholder_ramp = placeholder("no ramp", 2, 1, TextureViewDimension::D2);

        Self {
            target,
            draw_layout,
            image_layout,
            relief_layout,
            pipelines: ImagePipelines::default(),
            slopes: None,
            linear,
            nearest,
            placeholder: placeholder_array,
            placeholder_ramp,
            prepare: None,
            pages: Vec::new(),
            edges: Default::default(),
            resized: Default::default(),
        }
    }

    /// Makes `family`'s pipeline for the map's target if it isn't made.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, family: Family) -> RenderPipelineId {
        let (target, draw_layout) = (self.target, self.draw_layout);
        let mut pipelines = self.pipelines;
        let id = self.make_for(gpu, &mut pipelines, target, draw_layout, family);
        self.pipelines = pipelines;
        id
    }

    /// Makes `family`'s pipeline in `pipelines`, for another target (the
    /// drape atlas's format), if it isn't made.
    pub(crate) fn make_for(
        &self,
        gpu: &mut dyn Gpu,
        pipelines: &mut ImagePipelines,
        target: TargetFormat,
        draw_layout: BindGroupLayoutId,
        family: Family,
    ) -> RenderPipelineId {
        *pipelines.0[image_index(family)].get_or_insert_with(|| {
            let (shader, layout) = match family {
                Family::Raster => (&RASTER_SHADER, self.image_layout),
                Family::Hillshade => (&HILLSHADE_SHADER, self.image_layout),
                _ => (&COLOR_RELIEF_SHADER, self.relief_layout),
            };
            let blend = Some(BlendState::PremultipliedAlpha);
            Draw::new(shader, &[draw_layout, layout], target, blend).make(gpu)
        })
    }

    /// Makes the pipeline shaded hillshades drape their slopes with, if
    /// it isn't made.
    pub(crate) fn make_slopes(&mut self, gpu: &mut dyn Gpu) -> RenderPipelineId {
        let target = TargetFormat {
            color: crate::DRAPE_FORMAT,
            sample_count: 1,
        };
        let layouts = [self.draw_layout, self.image_layout];
        *self.slopes.get_or_insert_with(|| {
            let blend = Some(BlendState::PremultipliedAlpha);
            Draw::new(&HILLSHADE_SLOPES_SHADER, &layouts, target, blend).make(gpu)
        })
    }

    /// The pipeline shaded hillshades drape their slopes with, made or not.
    pub(crate) fn slopes_pipeline(&self) -> Option<RenderPipelineId> {
        self.slopes
    }

    /// The bind group layout of raster and hillshade nodes' groups.
    pub(crate) fn image_layout(&self) -> BindGroupLayoutId {
        self.image_layout
    }

    pub(crate) fn pipeline(&self, family: Family) -> Option<RenderPipelineId> {
        self.pipelines.get(family)
    }

    /// Makes the DEM prepare pipelines, unless they are made: the
    /// pipeline set's where the style reads DEM tiles, or for the first DEM
    /// page.
    pub(crate) fn make_prepare(&mut self, gpu: &mut dyn Gpu) {
        if self.prepare.is_none() {
            self.prepare = Some(prepare::Draws::new(gpu));
        }
    }

    /// Whether DEM tiles can be prepared: their pipelines compiled, or no
    /// DEM page needs them. What reads prepared DEM tiles (hillshade,
    /// color relief, terrain) waits for it.
    pub(crate) fn prepare_ready(&self, gpu: &mut dyn Gpu) -> bool {
        self.prepare.as_ref().is_none_or(|p| p.ready(gpu))
    }

    /// The prepare pipelines, for the first-visible order.
    pub(crate) fn prepare_pipelines(&self) -> Vec<PipelineId> {
        self.prepare
            .iter()
            .flat_map(prepare::Draws::pipelines)
            .collect()
    }

    /// A page of `kind` for tiles of `width × height` texels, of `layers`
    /// layers to start with.
    pub(crate) fn create_page(
        &mut self,
        gpu: &mut dyn Gpu,
        kind: PageKind,
        [width, height]: [u32; 2],
        layers: u32,
    ) -> PageId {
        let sampled = TextureUsages::TEXTURE_BINDING;
        // Host images are copied in as render targets, and the prepare
        // draws into its layers.
        let uploaded = sampled | TextureUsages::RENDER_ATTACHMENT;
        let texture = |label, border: u32, usage| PageTexture {
            label,
            width: width + 2 * border,
            height: height + 2 * border,
            format: TextureFormat::Rgba8Unorm,
            usage,
            mip_levels: 1,
        };
        // Raster pages are mipmapped: compute passes make the levels once
        // a raster node draws, or without compute render passes draw them.
        if kind.dem() {
            self.make_prepare(gpu);
        }
        let levels = |border: u32| 32 - (width.max(height) + 2 * border).max(1).leading_zeros();
        let written = if gpu.caps().compute {
            TextureUsages::STORAGE_BINDING
        } else {
            TextureUsages::empty()
        };
        let textures = match kind {
            PageKind::Raster => vec![PageTexture {
                usage: uploaded | written,
                mip_levels: levels(0),
                ..texture("raster tiles", 0, uploaded)
            }],
            PageKind::Dem => vec![
                texture("dem tiles", 0, uploaded),
                texture("padded dem tiles", 2, uploaded),
                texture("hillshade slopes", 1, uploaded),
            ],
            PageKind::Terrain => vec![
                texture("dem tiles", 0, uploaded),
                texture("padded dem tiles", 2, uploaded),
            ],
        };
        let page = Page {
            kind,
            page: TilePage::new(gpu, &textures, layers),
            pending: Vec::new(),
            mip_layers: Vec::new(),
            edges: Vec::new(),
            mips: None,
            targets: None,
        };
        let id = match self.pages.iter().position(Option::is_none) {
            Some(free) => {
                self.pages[free] = Some(page);
                free
            }
            None => {
                self.pages.push(Some(page));
                self.pages.len() - 1
            }
        };
        PageId(id as u32)
    }

    pub(crate) fn remove_page(&mut self, gpu: &mut dyn Gpu, id: PageId) {
        let mut page = self.pages[id.0 as usize]
            .take()
            .expect("page already removed");
        self.resized += page.page.take_resized();
        if let Some(mips) = page.mips {
            destroy_mips(gpu, mips);
        }
        if let Some(targets) = page.targets {
            targets.destroy(gpu);
        }
        page.page.destroy(gpu);
    }

    fn page(&self, id: PageId) -> &Page {
        self.pages[id.0 as usize].as_ref().expect("unknown page")
    }

    fn page_mut(&mut self, id: PageId) -> &mut Page {
        self.pages[id.0 as usize].as_mut().expect("unknown page")
    }

    /// A one-texel array, bound where no page is.
    pub(crate) fn placeholder(&self) -> TextureId {
        self.placeholder
    }

    /// The nearest-neighbour sampler, no mips.
    pub(crate) fn nearest(&self) -> SamplerId {
        self.nearest
    }

    /// The trilinear, anisotropic sampler.
    pub(crate) fn linear(&self) -> SamplerId {
        self.linear
    }

    /// A DEM page's padded elevations (2-texel borders from neighbours), and
    /// the revision of its textures.
    pub(crate) fn dem_texture(&self, id: PageId) -> (TextureId, u32) {
        let page = self.page(id);
        assert!(page.kind.dem(), "terrain reads DEM pages");
        (page.page.texture(DEM_PADDED), page.page.revision())
    }

    /// The size of a page's tiles.
    /// Bytes one layer of a page takes.
    pub(crate) fn layer_bytes(&self, id: PageId) -> u64 {
        self.page(id).page.layer_bytes()
    }

    pub(crate) fn page_size(&self, id: PageId) -> [u32; 2] {
        self.page(id).page.size(0)
    }

    /// What moving the pages did since the last call.
    pub(crate) fn take_resized(&mut self) -> crate::scene::Resized {
        let mut resized = std::mem::take(&mut self.resized);
        for page in self.pages.iter_mut().flatten() {
            resized += page.page.take_resized();
        }
        resized
    }

    /// GPU bytes of the raster and DEM pages.
    pub(crate) fn bytes(&self) -> u64 {
        self.pages.iter().flatten().map(|p| p.page.bytes()).sum()
    }

    /// A layer of the page, growing it if it's full.
    pub(crate) fn alloc_layer(&mut self, gpu: &mut dyn Gpu, id: PageId) -> Result<u32, OutOfSpace> {
        self.page_mut(id).page.alloc(gpu)
    }

    /// Layers in use across the pages.
    pub(crate) fn used_layers(&self) -> u64 {
        self.pages
            .iter()
            .flatten()
            .map(|p| u64::from(p.page.used()))
            .sum()
    }

    /// Shrinks the first page whose layers in use went down to a
    /// quarter: one move. Whether one did.
    pub(crate) fn shrink(&mut self, gpu: &mut dyn Gpu) -> bool {
        (self.pages.iter_mut().flatten()).any(|page| page.page.shrink(gpu))
    }

    /// Whether page `id` can grow.
    pub(crate) fn can_grow(&self, id: PageId) -> bool {
        self.pages[id.0 as usize]
            .as_ref()
            .is_some_and(|page| page.page.can_grow())
    }

    /// Grows page `id`: one move. Whether it grew.
    pub(crate) fn grow(&mut self, gpu: &mut dyn Gpu, id: PageId) -> bool {
        (self.pages[id.0 as usize].as_mut()).is_some_and(|page| page.page.grow(gpu).is_ok())
    }

    /// Changes whenever a page's textures are replaced.
    pub(crate) fn revisions(&self) -> u64 {
        (self.pages.iter().flatten())
            .map(|page| u64::from(page.page.revision()))
            .sum()
    }

    /// Whether [`shrink`](Self::shrink) would shrink a page.
    pub(crate) fn would_shrink(&self) -> bool {
        self.pages
            .iter()
            .flatten()
            .any(|page| page.page.would_shrink())
    }

    /// Whether a page holds more than three quarters of its layers, and
    /// growing it fits in `room` bytes.
    pub(crate) fn wants_headroom(&self, room: u64) -> bool {
        (self.pages.iter().flatten()).any(|page| page.page.headroom().is_some_and(|b| b <= room))
    }

    /// Grows the first page holding more than three quarters of its
    /// layers, within `room` bytes; whether one grew.
    pub(crate) fn grow_headroom(&mut self, gpu: &mut dyn Gpu, room: u64) -> bool {
        (self.pages.iter_mut().flatten())
            .find(|page| page.page.headroom().is_some_and(|b| b <= room))
            .is_some_and(|page| page.page.grow(gpu).is_ok())
    }

    /// A layer the page already has, without growing it.
    pub(crate) fn alloc_existing_layer(&mut self, id: PageId) -> Option<u32> {
        self.page_mut(id).page.alloc_existing()
    }

    pub(crate) fn free_layer(&mut self, id: PageId, layer: u32) {
        let page = self.page_mut(id);
        page.page.free(layer);
        page.edges.retain(|e| e.0 != layer);
    }

    pub(crate) fn write_layer(
        &mut self,
        gpu: &mut dyn Gpu,
        id: PageId,
        layer: u32,
        pixels: Pixels<'_>,
    ) {
        let page = self.page_mut(id);
        page.page.write(gpu, 0, layer, pixels);
        page.edges.retain(|e| e.0 != layer);
        if page.page.mip_levels(0) > 1 && !page.mip_layers.contains(&layer) {
            page.mip_layers.push(layer);
        }
    }

    /// Marks a raster layer just written as a tile that reaches past its
    /// area, `area` the texels inside, `[x0, y0, x1, y1)`: before its mip
    /// levels are made, each texel past the area takes the nearest one
    /// inside, so that filtering at the area's edge, at every level, reads
    /// the edge as at a tile's own.
    pub(crate) fn extend_edges(&mut self, id: PageId, layer: u32, area: [u32; 4]) {
        let page = self.page_mut(id);
        if page.mip_layers.contains(&layer) {
            page.edges.push((layer, area));
        }
    }

    /// The passes that make mip levels for raster layers uploaded since
    /// the last, if any, after the passes that extend their edges; and
    /// their dispatches or draws.
    pub(crate) fn mip_pass(
        &mut self,
        gpu: &mut dyn Gpu,
        downsampler: &crate::downsample::Downsampler,
    ) -> Option<(Vec<Pass>, u32)> {
        // Without layers, the pipelines aren't asked after: made ahead of
        // need, they compile in the background. Layers wait for them.
        let layers = (self.pages.iter().flatten()).any(|p| !p.mip_layers.is_empty());
        if layers && !downsampler.ready(gpu) {
            return None;
        }
        // The last frame, which used them, was submitted.
        std::mem::take(&mut self.edges).release(gpu);
        if !layers {
            return None;
        }
        let mut passes = Vec::new();
        let mut levels = Vec::new();
        let mut work = 0;
        for page in self.pages.iter_mut().flatten() {
            if page.mip_layers.is_empty() {
                continue;
            }
            let (revision, capacity) = (page.page.revision(), page.page.capacity());
            if page.mips.as_ref().is_some_and(|m| m.revision != revision) {
                destroy_mips(gpu, page.mips.take().expect("checked"));
            }
            let texture = page.page.texture(0);
            let mips = page.mips.get_or_insert_with(|| {
                let views: Vec<TextureId> = (0..page.page.mip_levels(0))
                    .map(|level| gpu.create_texture_view(texture, level..level + 1))
                    .collect();
                let chain =
                    downsampler.chain(gpu, (texture, &views), ("image mip regions", capacity));
                Mips {
                    revision,
                    views,
                    chain,
                }
            });
            let [side, _] = page.page.size(0);
            let edges = std::mem::take(&mut page.edges);
            if !edges.is_empty() {
                let extend = (texture, mips.views[0], side);
                let extended = downsampler.extend_passes(gpu, extend, &edges, &mut self.edges)?;
                work += extended.len() as u32 - 1;
                passes.extend(extended);
            }
            let layers = std::mem::take(&mut page.mip_layers);
            let regions: Vec<[u32; 4]> = layers.iter().map(|&layer| [layer, 0, 0, side]).collect();
            mips.chain.regions.write(gpu, &regions);
            let chain = ("image mips", &mips.chain);
            let (made, n) = downsampler.mip_passes(gpu, chain, &regions, &mut self.edges)?;
            work += n;
            levels.extend(made);
        }
        // One compute pass makes every page's levels.
        let mut merged: Vec<Pass> = Vec::new();
        for pass in levels {
            match (merged.last_mut(), pass) {
                (Some(Pass::Compute(last)), Pass::Compute(next)) => {
                    last.commands.extend(next.commands);
                }
                (_, pass) => merged.push(pass),
            }
        }
        (!merged.is_empty()).then(|| {
            passes.extend(merged);
            (passes, work)
        })
    }

    pub(crate) fn prepare_dem(&mut self, id: PageId, job: DemPrepare) {
        let page = self.page_mut(id);
        assert!(page.kind.dem(), "only DEM pages are prepared");
        // A job still waiting for the layer: this one takes its borders
        // too, or the whole tile if either does.
        let mut job = job;
        if let Some(at) = page.pending.iter().position(|j| j.layer == job.layer) {
            let waiting = page.pending.swap_remove(at);
            job.borders = waiting.borders.zip(job.borders).map(|(a, b)| a | b);
        }
        page.pending.push(job);
    }

    /// The draws that prepare pending DEM tiles, if any, and how many
    /// tiles they prepare. Jobs wait while the pipelines compile.
    pub(crate) fn prepare_pass(&mut self, gpu: &mut dyn Gpu) -> Option<(Vec<Pass>, Prepared)> {
        if !self.prepare_ready(gpu) {
            return None;
        }
        let draws = self.prepare.as_ref()?;
        let mut passes = Vec::new();
        let mut tiles = 0;
        for page in self.pages.iter_mut().flatten() {
            tiles += draws.passes(gpu, page, &mut passes);
        }
        let draws = passes.len() as u32;
        (tiles > 0).then_some((passes, Prepared { tiles, draws }))
    }

    /// Writes a color-relief ramp: `colors` (premultiplied rgba8) and the
    /// stop elevations packed as DEM texels, one stop per column.
    pub(crate) fn set_ramp(
        &mut self,
        gpu: &mut dyn Gpu,
        node: &mut NodeImage,
        colors: &[u32],
        elevations: &[u32],
    ) {
        assert_eq!(colors.len(), elevations.len(), "one colour per stop");
        node.ramp.take().release(gpu);
        let width = colors.len() as u32;
        let texture = gpu.create_texture(&TextureDesc {
            label: "color relief ramp",
            width,
            height: 2,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
        });
        let texels: Vec<u32> = colors.iter().chain(elevations).copied().collect();
        gpu.write_texture(
            &TextureRegion {
                texture,
                x: 0,
                y: 0,
                layer: 0,
                width,
                height: 2,
                level: 0,
            },
            bytemuck::cast_slice(&texels),
        );
        node.ramp = Some(texture);
        node.built = None;
    }

    pub(crate) fn bind_page(node: &mut NodeImage, page: PageId) {
        if node.page != Some(page) {
            node.page = Some(page);
            node.built = None;
        }
    }

    /// Whether `node`'s bind group is out of date.
    pub(crate) fn stale(&self, node: &NodeImage) -> bool {
        let revision = node.page.map(|id| self.page(id).page.revision());
        node.built.is_none() || (revision.is_some() && node.built != revision)
    }

    /// A new bind group for an image node with its uniform at
    /// `uniform`.
    pub(crate) fn node_group(
        &self,
        gpu: &mut dyn Gpu,
        family: Family,
        node: &mut NodeImage,
        uniform: BindingResource,
    ) -> BindGroupId {
        node.built = Some(node.page.map_or(0, |id| self.page(id).page.revision()));
        self.group(gpu, family, node, uniform)
    }

    /// A bind group of an image node's textures with a uniform at
    /// `uniform`.
    pub(crate) fn group(
        &self,
        gpu: &mut dyn Gpu,
        family: Family,
        node: &NodeImage,
        uniform: BindingResource,
    ) -> BindGroupId {
        let page = node.page.map(|id| self.page(id));
        let texture = match (page, family) {
            (None, _) => self.placeholder,
            (Some(page), Family::Raster) => page.page.texture(0),
            (Some(page), Family::Hillshade) => page.page.texture(DEM_SLOPES),
            (Some(page), Family::ColorRelief) => page.page.texture(DEM_PADDED),
            _ => unreachable!("not an image family"),
        };
        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: uniform,
            },
            BindGroupEntry::texture(1, texture, TextureViewDimension::D2Array),
            BindGroupEntry::sampler(2, self.linear),
            BindGroupEntry::sampler(3, self.nearest),
            BindGroupEntry::texture(5, texture, TextureViewDimension::D2Array),
        ];
        let layout = if family == Family::ColorRelief {
            let ramp = node.ramp.unwrap_or(self.placeholder_ramp);
            entries.push(BindGroupEntry::texture(4, ramp, TextureViewDimension::D2));
            self.relief_layout
        } else {
            self.image_layout
        };
        gpu.create_bind_group(&BindGroupDesc {
            label: "image node",
            layout,
            entries: &entries,
        })
    }
}

/// The prepare (render-to-texture `Prepare`): for each pending
/// job a draw into its padded layer, then one into its slope layer where
/// the page has slopes, the job read from a uniform slot of its own.
mod prepare {
    use orb_hal::*;
    use orb_shaders::{PREPARE_PAD_SHADER, PREPARE_SLOPE_SHADER};

    use super::{DEM_PADDED, DEM_RAW, DEM_SLOPES, Page, PageKind, prepare_job};
    use crate::TargetFormat;
    use crate::pipeline::Draw;
    use crate::release::Release;

    /// A job slot's bytes; jobs are 64 bytes.
    const SLOT: u64 = 256;

    /// A draw a job makes: its group's label, the texture it reads, the
    /// one it draws into and its borders' shape.
    type Stage = (&'static str, usize, usize, Border);

    /// The draws a job makes on a page of `kind`: a terrain's page pads
    /// its elevations alone.
    fn stages(kind: PageKind) -> &'static [Stage] {
        const STAGES: [Stage; 2] = [
            ("prepare pad", DEM_RAW, DEM_PADDED, Border::Padded),
            ("prepare slope", DEM_PADDED, DEM_SLOPES, Border::Slopes),
        ];
        match kind {
            PageKind::Terrain => &STAGES[..1],
            _ => &STAGES,
        }
    }

    /// A layer a job draws into, for its borders' rectangles.
    #[derive(Clone, Copy)]
    enum Border {
        /// (dim + 4)² texels: the tile's at 2..dim + 2, then a border of
        /// two texels from the neighbours.
        Padded,
        /// (dim + 2)² texels, texel i from the padded texels i..=i + 2.
        Slopes,
    }

    impl Border {
        /// The texels, `[x, y, width, height]`, that the neighbour `k`
        /// ([`DemPrepare::neighbours`](super::DemPrepare::neighbours)'
        /// order) of a `dim`-texel tile decides.
        fn rect(self, dim: u32, k: usize) -> [u32; 4] {
            const OFFSETS: [(i32, i32); 8] = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ];
            let (dx, dy) = OFFSETS[k];
            // The padded texels the neighbour fills along one axis, then
            // the texels of this layer that read them.
            let span = |d: i32| {
                let (lo, hi) = match d {
                    -1 => (0, 2),
                    0 => (2, dim + 2),
                    _ => (dim + 2, dim + 4),
                };
                match self {
                    Self::Padded => (lo, hi),
                    Self::Slopes => (lo.saturating_sub(2), hi.min(dim + 2)),
                }
            };
            let ((x0, x1), (y0, y1)) = (span(dx), span(dy));
            [x0, y0, x1 - x0, y1 - y0]
        }
    }

    #[derive(Debug)]
    pub(super) struct Draws {
        pub(super) layout: BindGroupLayoutId,
        pub(super) pad: RenderPipelineId,
        pub(super) slope: RenderPipelineId,
    }

    /// A page's layer views, job slots and groups, for one revision of its
    /// textures.
    #[derive(Debug)]
    pub(super) struct Targets {
        revision: u32,
        jobs: BufferId,
        slots: u32,
        /// Per layer: a view a [stage](stages).
        views: Vec<Vec<TextureId>>,
        /// Per slot: a group a [stage](stages).
        groups: Vec<Vec<BindGroupId>>,
    }

    impl Targets {
        pub(super) fn destroy(self, gpu: &mut dyn Gpu) {
            (self.views, self.groups, self.jobs).release(gpu);
        }
    }

    impl Draws {
        pub(super) fn new(gpu: &mut dyn Gpu) -> Self {
            let fs = ShaderStages::FRAGMENT;
            let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "prepare draw",
                entries: &[
                    BindGroupLayoutEntry::texture(0, fs, false, TextureViewDimension::D2Array),
                    BindGroupLayoutEntry::uniform(2, fs),
                ],
            });
            let target = TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            };
            let mut pipeline = |shader| Draw::new(shader, &[layout], target, None).make(gpu);
            Self {
                pad: pipeline(&PREPARE_PAD_SHADER),
                slope: pipeline(&PREPARE_SLOPE_SHADER),
                layout,
            }
        }

        pub(super) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
            gpu.pipeline_ready(self.pad.into()) && gpu.pipeline_ready(self.slope.into())
        }

        pub(super) fn pipelines(&self) -> [PipelineId; 2] {
            [self.pad.into(), self.slope.into()]
        }

        /// Appends a page's pending jobs' passes; returns the jobs.
        pub(super) fn passes(
            &self,
            gpu: &mut dyn Gpu,
            page: &mut Page,
            passes: &mut Vec<Pass>,
        ) -> u32 {
            if page.pending.is_empty() {
                return 0;
            }
            let revision = page.page.revision();
            let count = page.pending.len() as u32;
            if page
                .targets
                .as_ref()
                .is_some_and(|t| t.revision != revision || t.slots < count)
            {
                page.targets.take().expect("checked").destroy(gpu);
            }
            let targets = page.targets.get_or_insert_with(|| {
                let slots = count.max(page.page.capacity()).max(1);
                Targets {
                    revision,
                    jobs: gpu.create_buffer(&BufferDesc {
                        label: "dem jobs",
                        size: u64::from(slots) * SLOT,
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    }),
                    slots,
                    views: Vec::new(),
                    groups: Vec::new(),
                }
            });
            for (slot, job) in page.pending.drain(..).enumerate() {
                gpu.write_buffer(
                    targets.jobs,
                    slot as u64 * SLOT,
                    bytemuck::cast_slice(&job.words()),
                );
                let stages = stages(page.kind);
                while targets.groups.len() <= slot {
                    let at = targets.groups.len() as u64 * SLOT;
                    let groups = (stages.iter())
                        .map(|&(label, source, ..)| {
                            let texture = page.page.texture(source);
                            gpu.create_bind_group(&BindGroupDesc {
                                label,
                                layout: self.layout,
                                entries: &[
                                    BindGroupEntry::texture(
                                        0,
                                        texture,
                                        TextureViewDimension::D2Array,
                                    ),
                                    BindGroupEntry::slice(
                                        2,
                                        targets.jobs,
                                        at,
                                        prepare_job::WORDS as u64 * 4,
                                    ),
                                ],
                            })
                        })
                        .collect();
                    targets.groups.push(groups);
                }
                let layer = job.layer as usize;
                if targets.views.len() <= layer {
                    targets.views.resize(layer + 1, Vec::new());
                }
                let views = &mut targets.views[layer];
                if views.is_empty() {
                    *views = (stages.iter())
                        .map(|&(_, _, t, _)| {
                            gpu.create_layer_view(page.page.texture(t), job.layer, 0)
                        })
                        .collect();
                }
                let dim = page.page.size(DEM_RAW)[0];
                let pipelines = [self.pad, self.slope];
                let drawn =
                    (stages.iter().zip(pipelines)).zip(views.iter().zip(&targets.groups[slot]));
                for ((&(.., border), pipeline), (&target, &group)) in drawn {
                    let mut commands = vec![
                        RenderCommand::SetPipeline(pipeline),
                        RenderCommand::SetBindGroup { index: 0, group },
                    ];
                    let triangle = RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    };
                    match job.borders {
                        None => commands.push(triangle),
                        Some(mask) => {
                            for k in (0..8).filter(|k| mask & (1 << k) != 0) {
                                commands.push(RenderCommand::SetScissor(border.rect(dim, k)));
                                commands.push(triangle.clone());
                            }
                        }
                    }
                    passes.push(Pass::Render(RenderPass {
                        label: "prepare",
                        colors: vec![ColorAttachment {
                            target,
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Load,
                        }],
                        depth: None,
                        commands,
                    }));
                }
            }
            count
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Border;

        /// The neighbour, in `DemPrepare::neighbours`' order, whose texels
        /// fill padded texel `p` of a `dim`-texel tile (`prepare_pad.wgsl`),
        /// or `None` inside the tile.
        fn filled_by(dim: i32, p: [i32; 2]) -> Option<usize> {
            let d = p.map(|v| (v - 2).div_euclid(dim).clamp(-1, 1));
            let k = ((d[1] + 1) * 3 + d[0] + 1) as usize;
            (k != 4).then_some(if k > 4 { k - 1 } else { k })
        }

        fn inside(rect: [u32; 4], [x, y]: [i32; 2]) -> bool {
            let [rx, ry, w, h] = rect.map(|v| v as i32);
            (rx..rx + w).contains(&x) && (ry..ry + h).contains(&y)
        }

        #[test]
        fn borders_are_the_texels_their_neighbours_decide() {
            let dim = 5;
            for k in 0..8 {
                let padded = Border::Padded.rect(dim as u32, k);
                for y in 0..dim + 4 {
                    for x in 0..dim + 4 {
                        let ours = filled_by(dim, [x, y]) == Some(k);
                        assert_eq!(inside(padded, [x, y]), ours, "padded {k} at {x},{y}");
                    }
                }
                let slopes = Border::Slopes.rect(dim as u32, k);
                for y in 0..dim + 2 {
                    for x in 0..dim + 2 {
                        // Slope texel (x, y) reads padded texels x..=x + 2.
                        let reads = (0..3)
                            .any(|j| (0..3).any(|i| filled_by(dim, [x + i, y + j]) == Some(k)));
                        assert_eq!(inside(slopes, [x, y]), reads, "slopes {k} at {x},{y}");
                    }
                }
            }
        }
    }
}

impl crate::release::Release for Images {
    fn release(mut self, gpu: &mut dyn Gpu) {
        for id in 0..self.pages.len() {
            if self.pages[id].is_some() {
                self.remove_page(gpu, PageId(id as u32));
            }
        }
        let draws = self.prepare.map(|d| (d.layout, d.pad, d.slope));
        let fixed = (
            self.image_layout,
            self.relief_layout,
            self.pipelines.0,
            self.slopes,
        );
        (fixed, self.linear, self.nearest).release(gpu);
        (self.placeholder, self.placeholder_ramp, draws, self.edges).release(gpu);
    }
}

impl crate::release::Release for NodeImage {
    fn release(self, gpu: &mut dyn Gpu) {
        self.ramp.release(gpu);
    }
}
