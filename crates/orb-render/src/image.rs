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
use orb_shaders::{COLOR_RELIEF_SHADER, HILLSHADE_SHADER, RASTER_SHADER};

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
}

/// A DEM tile to prepare: its layer, its resident neighbours' layers and
/// how to read its elevations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemPrepare {
    pub layer: u32,
    /// The tiles at (dx, dy) = (−1, −1), (0, −1), (1, −1), (−1, 0), (1, 0),
    /// (−1, 1), (0, 1), (1, 1), at the same zoom.
    pub neighbours: [Option<u32>; 8],
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
    mips: Option<Mips>,
    /// The prepare draws' targets.
    targets: Option<prepare::Targets>,
}

/// A raster page's or DEM slopes' mip bindings, for one revision of its
/// texture: views of each level, the downsampler's group per level after
/// the first, and the regions they read (whole layers).
#[derive(Debug)]
struct Mips {
    revision: u32,
    regions: BufferId,
    views: Vec<TextureId>,
    groups: Vec<BindGroupId>,
}

fn destroy_mips(gpu: &mut dyn Gpu, mips: Mips) {
    (mips.groups, mips.views, mips.regions).release(gpu);
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
    linear: SamplerId,
    nearest: SamplerId,
    /// Bound by nodes whose source has no page yet.
    placeholder: TextureId,
    placeholder_ramp: TextureId,
    /// The prepare draws, made for the first DEM page.
    prepare: Option<prepare::Draws>,
    pages: Vec<Option<Page>>,
}

impl Images {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        draw_layout: BindGroupLayoutId,
    ) -> Self {
        let vf = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let fs = ShaderStages::FRAGMENT;
        let array = |binding| BindGroupLayoutEntry {
            binding,
            visibility: fs,
            ty: BindingType::Texture {
                sample: TextureSampleType::Float { filterable: true },
                dimension: TextureViewDimension::D2Array,
            },
        };
        let sampler = |binding| BindGroupLayoutEntry {
            binding,
            visibility: fs,
            ty: BindingType::Sampler { filtering: true },
        };
        let uniform = BindGroupLayoutEntry {
            binding: 0,
            visibility: vf,
            ty: BindingType::Uniform,
        };
        let image_entries = [uniform, array(1), sampler(2), sampler(3)];
        let image_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "image node",
            entries: &image_entries,
        });
        let mut relief_entries = image_entries.to_vec();
        relief_entries.push(BindGroupLayoutEntry {
            binding: 4,
            visibility: fs,
            ty: BindingType::Texture {
                sample: TextureSampleType::Float { filterable: true },
                dimension: TextureViewDimension::D2,
            },
        });
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
                address: AddressMode::ClampToEdge,
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
            linear,
            nearest,
            placeholder: placeholder_array,
            placeholder_ramp,
            prepare: None,
            pages: Vec::new(),
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
            crate::draw_pipeline(
                gpu,
                shader,
                PrimitiveTopology::TriangleList,
                target,
                &[draw_layout, layout],
            )
        })
    }

    /// The bind group layout of raster and hillshade nodes' groups.
    pub(crate) fn image_layout(&self) -> BindGroupLayoutId {
        self.image_layout
    }

    pub(crate) fn pipeline(&self, family: Family) -> Option<RenderPipelineId> {
        self.pipelines.get(family)
    }

    /// Makes the DEM prepare pipelines, for the first DEM page.
    fn make_prepare(&mut self, gpu: &mut dyn Gpu) {
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

    pub(crate) fn create_page(
        &mut self,
        gpu: &mut dyn Gpu,
        kind: PageKind,
        [width, height]: [u32; 2],
        downsampler: &mut crate::downsample::Downsampler,
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
        if kind == PageKind::Dem {
            self.make_prepare(gpu);
        }
        // Mipmapped (#3): a compute pass makes the levels.
        downsampler.make(gpu);
        let levels = |border: u32| 32 - (width.max(height) + 2 * border).max(1).leading_zeros();
        let textures = match kind {
            PageKind::Raster => vec![PageTexture {
                usage: uploaded | TextureUsages::STORAGE_BINDING,
                mip_levels: levels(0),
                ..texture("raster tiles", 0, uploaded)
            }],
            // Slopes are mipmapped too, for the terrain pass's shaded
            // hillshades in the distance.
            PageKind::Dem => vec![
                texture("dem tiles", 0, uploaded),
                texture("padded dem tiles", 2, uploaded),
                PageTexture {
                    usage: uploaded | TextureUsages::STORAGE_BINDING,
                    mip_levels: levels(1),
                    ..texture("hillshade slopes", 1, uploaded)
                },
            ],
        };
        let page = Page {
            kind,
            page: TilePage::new(gpu, &textures),
            pending: Vec::new(),
            mip_layers: Vec::new(),
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
        let page = self.pages[id.0 as usize]
            .take()
            .expect("page already removed");
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

    /// A DEM page's padded elevations (2-texel borders from neighbours), and
    /// the revision of its textures.
    pub(crate) fn dem_texture(&self, id: PageId) -> (TextureId, u32) {
        let page = self.page(id);
        assert_eq!(page.kind, PageKind::Dem, "terrain reads DEM pages");
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

    /// A layer the page already has, without growing it.
    pub(crate) fn alloc_existing_layer(&mut self, id: PageId) -> Option<u32> {
        self.page_mut(id).page.alloc_existing()
    }

    pub(crate) fn free_layer(&mut self, id: PageId, layer: u32) {
        self.page_mut(id).page.free(layer);
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
        if page.page.mip_levels(0) > 1 && !page.mip_layers.contains(&layer) {
            page.mip_layers.push(layer);
        }
    }

    /// The compute pass that makes mip levels for raster layers uploaded,
    /// and DEM slopes prepared, since the last, if any, and its
    /// dispatches.
    pub(crate) fn mip_pass(
        &mut self,
        gpu: &mut dyn Gpu,
        downsampler: &crate::downsample::Downsampler,
    ) -> Option<(Pass, u32)> {
        // Layers wait for the pipeline to compile.
        if !downsampler.ready(gpu) {
            return None;
        }
        let mut commands = Vec::new();
        let mut dispatches = 0;
        for page in self.pages.iter_mut().flatten() {
            if page.mip_layers.is_empty() {
                continue;
            }
            let (revision, capacity) = (page.page.revision(), page.page.capacity());
            if page.mips.as_ref().is_some_and(|m| m.revision != revision) {
                destroy_mips(gpu, page.mips.take().expect("checked"));
            }
            let index = match page.kind {
                PageKind::Raster => 0,
                PageKind::Dem => DEM_SLOPES,
            };
            let texture = page.page.texture(index);
            let levels = page.page.mip_levels(index);
            let mips = page.mips.get_or_insert_with(|| {
                let regions = gpu.create_buffer(&BufferDesc {
                    label: "image mip regions",
                    size: u64::from(capacity) * 16,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });
                let views: Vec<TextureId> = (0..levels)
                    .map(|level| gpu.create_texture_view(texture, level..level + 1))
                    .collect();
                let groups = downsampler.groups(gpu, &views, regions);
                Mips {
                    revision,
                    regions,
                    views,
                    groups,
                }
            });
            let [side, _] = page.page.size(index);
            let layers = std::mem::take(&mut page.mip_layers);
            let regions: Vec<[u32; 4]> = layers.iter().map(|&layer| [layer, 0, 0, side]).collect();
            gpu.write_buffer(mips.regions, 0, bytemuck::cast_slice(&regions));
            let count = layers.len() as u32;
            commands.extend(downsampler.dispatches(gpu, &mips.groups, side, count)?);
            dispatches += mips.groups.len() as u32;
        }
        (!commands.is_empty()).then(|| {
            (
                Pass::Compute(ComputePass {
                    label: "image mips".into(),
                    commands,
                }),
                dispatches,
            )
        })
    }

    pub(crate) fn prepare_dem(&mut self, id: PageId, job: DemPrepare) {
        let page = self.page_mut(id);
        assert_eq!(page.kind, PageKind::Dem, "only DEM pages are prepared");
        page.pending.retain(|j| j.layer != job.layer);
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
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Texture {
                    texture,
                    dimension: TextureViewDimension::D2Array,
                },
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(self.linear),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Sampler(self.nearest),
            },
        ];
        let layout = if family == Family::ColorRelief {
            entries.push(BindGroupEntry {
                binding: 4,
                resource: BindingResource::Texture {
                    texture: node.ramp.unwrap_or(self.placeholder_ramp),
                    dimension: TextureViewDimension::D2,
                },
            });
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
/// job a draw into its padded layer, then one into its slope layer, the job
/// read from a uniform slot of its own.
mod prepare {
    use orb_hal::*;
    use orb_shaders::{PREPARE_PAD_SHADER, PREPARE_SLOPE_SHADER, Shader};

    use super::{DEM_PADDED, DEM_RAW, DEM_SLOPES, Page, prepare_job};
    use crate::release::Release;

    /// A job slot's bytes; jobs are 64 bytes.
    const SLOT: u64 = 256;

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
        /// Per layer: its padded and slope views.
        views: Vec<Option<[TextureId; 2]>>,
        /// Per slot: its pad and slope groups.
        groups: Vec<[BindGroupId; 2]>,
    }

    impl Targets {
        pub(super) fn destroy(self, gpu: &mut dyn Gpu) {
            (self.views, self.groups, self.jobs).release(gpu);
        }
    }

    impl Draws {
        pub(super) fn new(gpu: &mut dyn Gpu) -> Self {
            let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
                label: "prepare draw",
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample: TextureSampleType::Float { filterable: false },
                            dimension: TextureViewDimension::D2Array,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Uniform,
                    },
                ],
            });
            let mut pipeline = |shader: &Shader| {
                let module = crate::shader_module(gpu, shader);
                let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
                    label: shader.label,
                    bind_group_layouts: &[layout],
                    vertex: ShaderStage {
                        module,
                        entry_point: shader.vertex,
                        constants: &[],
                    },
                    fragment: ShaderStage {
                        module,
                        entry_point: shader.fragment,
                        constants: &[],
                    },
                    targets: &[ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                    }],
                    depth: None,
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    sample_count: 1,
                });
                gpu.destroy_shader_module(module);
                pipeline
            };
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
            let mipmapped = page.page.mip_levels(DEM_SLOPES) > 1;
            for (slot, job) in page.pending.drain(..).enumerate() {
                if mipmapped && !page.mip_layers.contains(&job.layer) {
                    page.mip_layers.push(job.layer);
                }
                gpu.write_buffer(
                    targets.jobs,
                    slot as u64 * SLOT,
                    bytemuck::cast_slice(&job.words()),
                );
                while targets.groups.len() <= slot {
                    let at = targets.groups.len() as u64 * SLOT;
                    let mut group = |label, texture| {
                        gpu.create_bind_group(&BindGroupDesc {
                            label,
                            layout: self.layout,
                            entries: &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: BindingResource::Texture {
                                        texture,
                                        dimension: TextureViewDimension::D2Array,
                                    },
                                },
                                BindGroupEntry {
                                    binding: 2,
                                    resource: BindingResource::Buffer {
                                        buffer: targets.jobs,
                                        offset: at,
                                        size: Some(prepare_job::WORDS as u64 * 4),
                                    },
                                },
                            ],
                        })
                    };
                    let pad = group("prepare pad", page.page.texture(DEM_RAW));
                    let slope = group("prepare slope", page.page.texture(DEM_PADDED));
                    targets.groups.push([pad, slope]);
                }
                let layer = job.layer as usize;
                if targets.views.len() <= layer {
                    targets.views.resize(layer + 1, None);
                }
                let views = *targets.views[layer].get_or_insert_with(|| {
                    [DEM_PADDED, DEM_SLOPES]
                        .map(|t| gpu.create_layer_view(page.page.texture(t), job.layer))
                });
                for (pipeline, group, target) in [
                    (self.pad, targets.groups[slot][0], views[0]),
                    (self.slope, targets.groups[slot][1], views[1]),
                ] {
                    passes.push(Pass::Render(RenderPass {
                        label: "prepare".into(),
                        colors: vec![ColorAttachment {
                            target,
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Load,
                        }],
                        depth: None,
                        commands: vec![
                            RenderCommand::SetPipeline(pipeline),
                            RenderCommand::SetBindGroup { index: 0, group },
                            RenderCommand::Draw {
                                vertices: 0..3,
                                instances: 0..1,
                            },
                        ],
                    }));
                }
            }
            count
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
        let fixed = (self.image_layout, self.relief_layout, self.pipelines.0);
        (fixed, self.linear, self.nearest).release(gpu);
        (self.placeholder, self.placeholder_ramp, draws).release(gpu);
    }
}

impl crate::release::Release for NodeImage {
    fn release(self, gpu: &mut dyn Gpu) {
        self.ramp.release(gpu);
    }
}
