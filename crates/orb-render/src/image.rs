//! Image families: raster, hillshade and color-relief layers, the tile
//! pages they sample, and the DEM prepare pass (plan §8.5's `Prepare`).
//!
//! Each image tile is a one-chunk quad mesh whose property row
//! ([`IMAGE_ROW_WORDS`]) names its layer in the source's [`TilePage`], so
//! tiles go through the same cull, placements and fallback clipping as
//! vector tiles. A node's bind group holds its source's texture array;
//! it is rebuilt, and the bundle re-recorded, only when the node changes
//! pages or a page grows.
//!
//! DEM tiles are prepared once when they arrive, and again when a
//! neighbour arrives: `pad` fills a two-texel border from the neighbours
//! (GL JS's `backfillBorder`), `slope` computes hillshade slopes. Both are
//! one dispatch per page over every pending tile.

use orb_hal::*;
use orb_scene::{PageTexture, Pixels, TilePage};
use orb_shaders::{
    COLOR_RELIEF_SHADER, HILLSHADE_SHADER, PREPARE_PAD_SHADER, PREPARE_SLOPE_SHADER, RASTER_SHADER,
    Shader,
};

use crate::{Family, OutOfSpace, TargetFormat};

/// Words per image tile property row: texture layer, then the latitudes
/// of the tile's top and bottom edges as f32, then padding.
pub const IMAGE_ROW_WORDS: usize = 4;

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

const JOB_WORDS: usize = 16;
const NO_TILE: u32 = u32::MAX;

impl DemPrepare {
    fn words(&self) -> [u32; JOB_WORDS] {
        let mut words = [0; JOB_WORDS];
        for (word, v) in words.iter_mut().zip(self.unpack) {
            *word = v.to_bits();
        }
        words[4] = self.layer;
        for (word, n) in words[5..13].iter_mut().zip(self.neighbours) {
            *word = n.unwrap_or(NO_TILE);
        }
        words[13] = self.zoom.to_bits();
        words
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
    prepare: Option<Prepare>,
    /// Compat's prepare targets.
    #[cfg(feature = "compat")]
    targets: Option<compat_prepare::Targets>,
}

/// A DEM page's prepare bindings, for one revision of its textures.
#[derive(Debug)]
struct Prepare {
    revision: u32,
    jobs: BufferId,
    capacity: u32,
    pad: BindGroupId,
    slope: BindGroupId,
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
    pub(crate) dispatches: u32,
    pub(crate) draws: u32,
}

#[derive(Debug)]
struct PreparePipelines {
    pad_layout: BindGroupLayoutId,
    slope_layout: BindGroupLayoutId,
    pad: ComputePipelineId,
    slope: ComputePipelineId,
}

#[derive(Debug)]
pub(crate) struct Images {
    image_layout: BindGroupLayoutId,
    relief_layout: BindGroupLayoutId,
    raster: RenderPipelineId,
    hillshade: RenderPipelineId,
    color_relief: RenderPipelineId,
    linear: SamplerId,
    nearest: SamplerId,
    /// Bound by nodes whose source has no page yet.
    placeholder: TextureId,
    placeholder_ramp: TextureId,
    /// The prepare dispatch's layouts and pipelines (pad, slope); none on
    /// Compat, which has no compute.
    prepare: Option<PreparePipelines>,
    /// Compat's prepare draws.
    #[cfg(feature = "compat")]
    draws: Option<compat_prepare::Draws>,
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
        let [raster, hillshade, color_relief] =
            image_pipelines(gpu, target, draw_layout, image_layout, relief_layout);

        let mut sampler = |label, filter| {
            gpu.create_sampler(&SamplerDesc {
                label,
                filter,
                address: AddressMode::ClampToEdge,
                mipmap_filter: FilterMode::Nearest,
            })
        };
        let linear = sampler("linear", FilterMode::Linear);
        let nearest = sampler("nearest", FilterMode::Nearest);
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

        let cs = ShaderStages::COMPUTE;
        let prepare_layout = |gpu: &mut dyn Gpu, label| {
            gpu.create_bind_group_layout(&BindGroupLayoutDesc {
                label,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: cs,
                        ty: BindingType::Texture {
                            sample: TextureSampleType::Float { filterable: false },
                            dimension: TextureViewDimension::D2Array,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: cs,
                        ty: BindingType::StorageTexture {
                            format: TextureFormat::Rgba8Unorm,
                            dimension: TextureViewDimension::D2Array,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: cs,
                        ty: BindingType::Storage { read_only: true },
                    },
                ],
            })
        };
        let compute_tier = gpu.caps().tier == Tier::Modern;
        let pad_layout = prepare_layout(gpu, "prepare pad");
        let slope_layout = prepare_layout(gpu, "prepare slope");
        let mut compute = |shader: &Shader, layout| {
            let module = gpu.create_shader_module(&ShaderModuleDesc {
                label: shader.label,
                source: ShaderSource::Wgsl(shader.source().into()),
            });
            gpu.create_compute_pipeline(&ComputePipelineDesc {
                label: shader.label,
                bind_group_layouts: &[layout],
                compute: ShaderStage {
                    module,
                    entry_point: shader.compute,
                    constants: &[],
                },
            })
        };
        let prepare = compute_tier.then(|| PreparePipelines {
            pad_layout,
            slope_layout,
            pad: compute(&PREPARE_PAD_SHADER, pad_layout),
            slope: compute(&PREPARE_SLOPE_SHADER, slope_layout),
        });

        Self {
            image_layout,
            relief_layout,
            raster,
            hillshade,
            color_relief,
            linear,
            nearest,
            placeholder: placeholder_array,
            placeholder_ramp,
            prepare,
            #[cfg(feature = "compat")]
            draws: (!compute_tier).then(|| compat_prepare::Draws::new(gpu)),
            pages: Vec::new(),
        }
    }

    /// Raster, hillshade and color-relief pipelines for another target
    /// (the drape atlas's format).
    pub(crate) fn pipelines_for(
        &self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        draw_layout: BindGroupLayoutId,
    ) -> [RenderPipelineId; 3] {
        image_pipelines(
            gpu,
            target,
            draw_layout,
            self.image_layout,
            self.relief_layout,
        )
    }

    pub(crate) fn pipeline(&self, family: Family) -> Option<RenderPipelineId> {
        match family {
            Family::Raster => Some(self.raster),
            Family::Hillshade => Some(self.hillshade),
            Family::ColorRelief => Some(self.color_relief),
            _ => None,
        }
    }

    pub(crate) fn create_page(
        &mut self,
        gpu: &mut dyn Gpu,
        kind: PageKind,
        [width, height]: [u32; 2],
    ) -> PageId {
        let sampled = TextureUsages::TEXTURE_BINDING;
        // Host images are copied in as render targets.
        let uploaded = sampled | TextureUsages::RENDER_ATTACHMENT;
        let prepared = sampled | TextureUsages::STORAGE_BINDING;
        let texture = |label, border: u32, usage| PageTexture {
            label,
            width: width + 2 * border,
            height: height + 2 * border,
            format: TextureFormat::Rgba8Unorm,
            usage,
        };
        let textures = match kind {
            PageKind::Raster => vec![texture("raster tiles", 0, uploaded)],
            PageKind::Dem => vec![
                texture("dem tiles", 0, uploaded),
                texture("padded dem tiles", 2, prepared),
                texture("hillshade slopes", 1, prepared),
            ],
        };
        let page = Page {
            kind,
            page: TilePage::new(gpu, &textures),
            pending: Vec::new(),
            prepare: None,
            #[cfg(feature = "compat")]
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
        if let Some(prepare) = page.prepare {
            destroy_prepare(gpu, prepare);
        }
        #[cfg(feature = "compat")]
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

    /// A DEM page's padded elevations (2-texel borders from neighbours), and
    /// the revision of its textures.
    pub(crate) fn dem_texture(&self, id: PageId) -> (TextureId, u32) {
        let page = self.page(id);
        assert_eq!(page.kind, PageKind::Dem, "terrain reads DEM pages");
        (page.page.texture(DEM_PADDED), page.page.revision())
    }

    /// The size of a page's tiles.
    pub(crate) fn page_size(&self, id: PageId) -> [u32; 2] {
        self.page(id).page.size(0)
    }

    pub(crate) fn alloc_layer(&mut self, gpu: &mut dyn Gpu, id: PageId) -> Result<u32, OutOfSpace> {
        self.page_mut(id).page.alloc(gpu)
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
        self.page(id).page.write(gpu, 0, layer, pixels);
    }

    pub(crate) fn prepare_dem(&mut self, id: PageId, job: DemPrepare) {
        let page = self.page_mut(id);
        assert_eq!(page.kind, PageKind::Dem, "only DEM pages are prepared");
        page.pending.retain(|j| j.layer != job.layer);
        page.pending.push(job);
    }

    /// The compute pass that prepares pending DEM tiles, if any, and how
    /// many tiles it prepares.
    pub(crate) fn prepare_pass(&mut self, gpu: &mut dyn Gpu) -> Option<(Vec<Pass>, Prepared)> {
        #[cfg(feature = "compat")]
        if let Some(draws) = &self.draws {
            let mut passes = Vec::new();
            let mut prepared = 0;
            for page in self.pages.iter_mut().flatten() {
                prepared += draws.passes(gpu, page, &mut passes);
            }
            return (prepared > 0).then(|| {
                let draws = passes.len() as u32;
                (
                    passes,
                    Prepared {
                        tiles: prepared,
                        dispatches: 0,
                        draws,
                    },
                )
            });
        }
        let pipelines = self.prepare.as_ref()?;
        let mut commands = Vec::new();
        let mut prepared = 0;
        for page in self.pages.iter_mut().flatten() {
            if page.pending.is_empty() {
                continue;
            }
            let revision = page.page.revision();
            let capacity = page.page.capacity();
            if page
                .prepare
                .as_ref()
                .is_some_and(|p| p.revision != revision || p.capacity < capacity)
            {
                destroy_prepare(gpu, page.prepare.take().expect("checked"));
            }
            let prepare = page.prepare.get_or_insert_with(|| {
                let jobs = gpu.create_buffer(&BufferDesc {
                    label: "dem jobs",
                    size: u64::from(capacity) * JOB_WORDS as u64 * 4,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });
                let mut group = |label, layout, from, to| {
                    gpu.create_bind_group(&BindGroupDesc {
                        label,
                        layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: BindingResource::Texture {
                                    texture: page.page.texture(from),
                                    dimension: TextureViewDimension::D2Array,
                                },
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: BindingResource::Texture {
                                    texture: page.page.texture(to),
                                    dimension: TextureViewDimension::D2Array,
                                },
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: BindingResource::buffer(jobs),
                            },
                        ],
                    })
                };
                Prepare {
                    revision,
                    jobs,
                    capacity,
                    pad: group("prepare pad", pipelines.pad_layout, DEM_RAW, DEM_PADDED),
                    slope: group(
                        "prepare slope",
                        pipelines.slope_layout,
                        DEM_PADDED,
                        DEM_SLOPES,
                    ),
                }
            });
            let jobs: Vec<u32> = page.pending.drain(..).flat_map(|j| j.words()).collect();
            let count = (jobs.len() / JOB_WORDS) as u32;
            gpu.write_buffer(prepare.jobs, 0, bytemuck::cast_slice(&jobs));
            let groups = |size: [u32; 2]| size[0].div_ceil(8);
            for (pipeline, group, texture) in [
                (pipelines.pad, prepare.pad, DEM_PADDED),
                (pipelines.slope, prepare.slope, DEM_SLOPES),
            ] {
                let n = groups(page.page.size(texture));
                commands.extend([
                    ComputeCommand::SetPipeline(pipeline),
                    ComputeCommand::SetBindGroup { index: 0, group },
                    ComputeCommand::Dispatch {
                        x: n,
                        y: n,
                        z: count,
                    },
                ]);
            }
            prepared += count;
        }
        (prepared > 0).then(|| {
            let pass = ComputePass {
                label: "prepare".into(),
                timestamps: None,
                commands,
            };
            (
                vec![Pass::Compute(pass)],
                Prepared {
                    tiles: prepared,
                    dispatches: 2,
                    draws: 0,
                },
            )
        })
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
        if let Some(old) = node.ramp.take() {
            gpu.destroy_texture(old);
        }
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
        let page = node.page.map(|id| self.page(id));
        let texture = match (page, family) {
            (None, _) => self.placeholder,
            (Some(page), Family::Raster) => page.page.texture(0),
            (Some(page), Family::Hillshade) => page.page.texture(DEM_SLOPES),
            (Some(page), Family::ColorRelief) => page.page.texture(DEM_PADDED),
            _ => unreachable!("not an image family"),
        };
        node.built = Some(page.map_or(0, |p| p.page.revision()));
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

    pub(crate) fn release_node(gpu: &mut dyn Gpu, node: NodeImage) {
        if let Some(ramp) = node.ramp {
            gpu.destroy_texture(ramp);
        }
    }
}

/// The raster, hillshade and color-relief pipelines for `target`.
fn image_pipelines(
    gpu: &mut dyn Gpu,
    target: TargetFormat,
    draw_layout: BindGroupLayoutId,
    image_layout: BindGroupLayoutId,
    relief_layout: BindGroupLayoutId,
) -> [RenderPipelineId; 3] {
    let mut pipeline = |shader: &Shader, layout| {
        crate::draw_pipeline(
            gpu,
            shader,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[draw_layout, layout],
        )
    };
    [
        pipeline(&RASTER_SHADER, image_layout),
        pipeline(&HILLSHADE_SHADER, image_layout),
        pipeline(&COLOR_RELIEF_SHADER, relief_layout),
    ]
}

fn destroy_prepare(gpu: &mut dyn Gpu, prepare: Prepare) {
    gpu.destroy_bind_group(prepare.pad);
    gpu.destroy_bind_group(prepare.slope);
    gpu.destroy_buffer(prepare.jobs);
}

/// Compat's prepare (plan §8.5's render-to-texture `Prepare`): for each
/// pending job a draw into its padded layer, then one into its slope layer,
/// the job read from a uniform slot of its own.
#[cfg(feature = "compat")]
mod compat_prepare {
    use orb_hal::*;
    use orb_shaders::{PREPARE_PAD_DRAW_SHADER, PREPARE_SLOPE_DRAW_SHADER, Shader};

    use super::{DEM_PADDED, DEM_RAW, DEM_SLOPES, JOB_WORDS, Page};

    /// A job slot's bytes; jobs are 64 bytes.
    const SLOT: u64 = 256;

    #[derive(Debug)]
    pub(super) struct Draws {
        layout: BindGroupLayoutId,
        pad: RenderPipelineId,
        slope: RenderPipelineId,
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
            for view in self.views.into_iter().flatten().flatten() {
                gpu.destroy_texture(view);
            }
            for group in self.groups.into_iter().flatten() {
                gpu.destroy_bind_group(group);
            }
            gpu.destroy_buffer(self.jobs);
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
                let module = crate::shader_module(gpu, shader, false, 0);
                gpu.create_render_pipeline(&RenderPipelineDesc {
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
                })
            };
            Self {
                pad: pipeline(&PREPARE_PAD_DRAW_SHADER),
                slope: pipeline(&PREPARE_SLOPE_DRAW_SHADER),
                layout,
            }
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
                let words = job.words();
                // The uniform's layout: unpack, neighbours, layer, zoom.
                let mut uniform = [0u32; JOB_WORDS];
                uniform[..4].copy_from_slice(&words[..4]);
                uniform[4..12].copy_from_slice(&words[5..13]);
                uniform[12] = words[4];
                uniform[13] = words[13];
                gpu.write_buffer(
                    targets.jobs,
                    slot as u64 * SLOT,
                    bytemuck::cast_slice(&uniform),
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
                                        size: Some(JOB_WORDS as u64 * 4),
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
                        color: ColorAttachment {
                            target,
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Load,
                        },
                        depth: None,
                        timestamps: None,
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
