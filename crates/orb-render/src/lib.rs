//! The Modern GPU-driven path (plan §8.3, §8.5).
//!
//! A frame has a cull and main pass: one cull/compact dispatch, a workgroup per render
//! node that keeps its visible `(chunk, tile placement)` candidates in draw
//! order and writes its indirect arguments, and the main pass, which
//! replays a render bundle of one indirect draw per render node. The bundle
//! is recorded again only when the node list changes or an image page
//! grows; tile arrival changes buffers, not commands. Frames in which DEM
//! tiles arrived first run the prepare pass (see the `image` module). Heatmap
//! nodes prepare density targets after culling and composite in layer order.
//!
//! Fill, line and extrusion nodes draw compacted indices, which the cull
//! dispatch keeps in each candidate's slot of one index buffer, with one
//! `drawIndexedIndirect` per node; point families and image tiles pull
//! their vertices (decision 0002).

#![forbid(unsafe_code)]

mod extrusion;
mod heatmap;
mod image;
mod paint;
mod sky;
pub use sky::SkyPaint;

pub use image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, PageId, PageKind};
pub use orb_scene::Pixels;
pub use orb_scene::{
    ARGS_WORDS, ClipRect, FrameBlock, ImageAtlas, ImageBindings, ImageKey, ImageMetadata,
    ImageUpload, MeshData, MeshId, NodeCull, NodeDraw, OutOfSpace, Scene, SceneCapacity,
    SceneStats, TileId, TileRow, Words,
};
pub use orb_scene::{
    GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas, PLACED_WORDS, QUAD_WORDS, SYMBOL_HEADER_WORDS,
    SYMBOL_STATE_WORDS, SymbolMeshData, pack_image_rect, symbol_header,
};
pub use paint::{
    CIRCLE_ROW_WORDS, CirclePaint, ColorReliefPaint, Curve, EXTRUSION_ROW_WORDS, ExtrusionPaint,
    FILL_ROW_WORDS, FillPaint, GpuPaint, HEATMAP_ROW_WORDS, HeatmapPaint, HillshadeMethod,
    HillshadePaint, LINE_ROW_WORDS, LinePaint, MAX_LIGHTS, NodePaint, PatternPaint, RasterPaint,
    SYMBOL_ROW_WORDS, StateLookup, SymbolPaint, SymbolSize, Translate, pack_color,
};

use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, CULL_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_SHADER, SYMBOL_SHADER,
    Shader,
};

/// The layout family a render node draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    FillExtrusion,
    Fill,
    /// A fill's antialiasing outline: 1 px line segments, drawn with fill
    /// paint (outline colour and opacity).
    FillOutline,
    Line,
    Circle,
    Heatmap,
    /// Image tiles of a `raster` source.
    Raster,
    /// Slopes of a `raster-dem` source's tiles.
    Hillshade,
    /// Elevations of a `raster-dem` source's tiles through a colour ramp.
    ColorRelief,
    /// A symbol layer's icons.
    SymbolIcon,
    /// A symbol layer's text.
    SymbolText,
}

impl Family {
    /// Whether the family draws image tiles from a page.
    pub fn is_image(self) -> bool {
        matches!(self, Self::Raster | Self::Hillshade | Self::ColorRelief)
    }

    /// Whether the family draws symbol quads.
    pub fn is_symbol(self) -> bool {
        matches!(self, Self::SymbolIcon | Self::SymbolText)
    }

    /// How the family's nodes draw (decision 0002): meshes from compacted
    /// indices, instanced quads by vertex pulling.
    fn draw(self) -> NodeDraw {
        match self {
            Self::Fill | Self::FillOutline | Self::Line | Self::FillExtrusion => NodeDraw::Indexed,
            Self::Circle | Self::Heatmap | Self::SymbolIcon | Self::SymbolText => NodeDraw::Points,
            Self::Raster | Self::Hillshade | Self::ColorRelief => NodeDraw::Images,
        }
    }
}

/// Where node `index`'s draw arguments start in the scene's args buffer.
fn args_offset(index: u32) -> u64 {
    u64::from(index) * orb_scene::ARGS_WORDS as u64 * 4
}

/// The attachment a renderer draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetFormat {
    pub color: TextureFormat,
    pub sample_count: u32,
}

/// Counters for one frame (plan §16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub dispatches: u32,
    /// Draws issued: one indirect draw per node, and prepare passes'.
    pub draws: u32,
    /// Whether the bundle was re-recorded this frame.
    pub bundle_recorded: bool,
    pub candidates: u32,
    /// Objects the CPU wrote since the last frame (plan §16): tile
    /// placement rows and node uniforms. A camera-only frame rewrites only
    /// the placements in view.
    pub placements_written: u32,
    pub paints_written: u32,
    /// DEM tiles the prepare pass processed.
    pub prepared: u32,
}

#[derive(Debug)]
struct Node {
    family: Family,
    /// Draws with the bytecode-class pipeline.
    bytecode: bool,
    group: BindGroupId,
    cull: NodeCull,
    /// The uniform words last written, to skip unchanged writes.
    written: Vec<u32>,
    /// What an image family's group binds.
    image: Option<image::NodeImage>,
    heatmap: Option<heatmap::NodeHeatmap>,
    extrusion: Option<extrusion::NodeExtrusion>,
    /// A symbol node draws its halos in a pass of their own.
    separate_halo: bool,
}

#[derive(Debug)]
pub struct Renderer {
    target: TargetFormat,
    sky: Option<sky::Sky>,
    scene: Scene,
    draw_group: BindGroupId,
    cull_group: BindGroupId,
    draw_layout: BindGroupLayoutId,
    node_layout: BindGroupLayoutId,
    /// The vector families' table-class pipelines.
    paint: PaintPipelines,
    /// Their bytecode class, created for the first node that interprets
    /// paint (plan §8.4: only it compiles the interpreter).
    bytecode: Option<PaintPipelines>,
    /// Symbols with halo and fill together, halo only, fill only.
    symbol_pipelines: [RenderPipelineId; 3],
    cull_pipeline: ComputePipelineId,
    node_uniforms: BufferId,
    node_stride: u64,
    nodes: Vec<Node>,
    bundle: Option<RenderBundleId>,
    /// Node uniform writes since the last frame.
    paints_written: u32,
    images: image::Images,
    heatmaps: heatmap::Heatmaps,
    extrusions: extrusion::Extrusions,
    glyphs: GlyphAtlas,
    /// The glyph atlas revision the scene group binds.
    glyphs_bound: u32,
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
    nearest: SamplerId,
}

/// Bytes reserved per node uniform; every node struct fits.
const NODE_UNIFORM_SIZE: u64 = 512;

impl Renderer {
    pub fn new(gpu: &mut dyn Gpu, target: TargetFormat, capacity: SceneCapacity) -> Self {
        let images = ImageAtlas::new(gpu, 512, 4, 4096);
        Self::with_images(gpu, target, capacity, images.bindings())
    }

    /// Uses the Runtime's shared image array and rect table.
    pub fn with_images(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        capacity: SceneCapacity,
        images: ImageBindings,
    ) -> Self {
        let scene = Scene::new(gpu, capacity);
        let b = *scene.buffers();
        let storage = |binding, visibility, read_only| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Storage { read_only },
        };
        let uniform = |binding, visibility| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Uniform,
        };

        let vf = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let vs = ShaderStages::VERTEX;
        let vf_storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Storage { read_only: true },
        };
        let draw_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "scene",
            entries: &[
                uniform(0, vf),
                vf_storage(1),
                storage(2, vs, true),
                storage(3, vs, true),
                storage(4, vs, true),
                storage(5, vs, true),
                storage(6, vs, true),
                storage(7, vs, true),
                storage(8, ShaderStages::FRAGMENT, true),
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler { filtering: true },
                },
                BindGroupLayoutEntry {
                    binding: 11,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 12,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 13,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler { filtering: true },
                },
            ],
        });
        let node_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node",
            entries: &[uniform(0, vf)],
        });
        let cs = ShaderStages::COMPUTE;
        let cull_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "cull",
            entries: &[
                uniform(0, cs),
                storage(1, cs, true),
                storage(2, cs, true),
                storage(3, cs, false),
                storage(4, cs, true),
                storage(5, cs, false),
                storage(6, cs, false),
                storage(7, cs, true),
                storage(8, cs, false),
            ],
        });

        let whole = BindingResource::buffer;
        let entries = |buffers: &[BufferId]| -> Vec<BindGroupEntry> {
            buffers
                .iter()
                .enumerate()
                .map(|(i, &buffer)| BindGroupEntry {
                    binding: i as u32,
                    resource: whole(buffer),
                })
                .collect()
        };
        let glyphs = GlyphAtlas::new(gpu, capacity.glyph_rows, b.props, scene.glyph_table());
        let nearest = gpu.create_sampler(&SamplerDesc {
            label: "nearest",
            filter: FilterMode::Nearest,
            address: AddressMode::ClampToEdge,
        });
        let draw_group = scene_group(gpu, draw_layout, &b, &images, glyphs.texture(), nearest);
        let cull_group = gpu.create_bind_group(&BindGroupDesc {
            label: "cull",
            layout: cull_layout,
            entries: &entries(&[
                b.frame,
                b.tiles,
                b.chunks,
                b.candidates,
                b.node_cull,
                b.visible,
                b.args,
                b.indices,
                b.compacted,
            ]),
        });

        let layouts = [draw_layout, node_layout];
        let paint = PaintPipelines::new(gpu, target, &layouts, false);
        let symbol_pipelines =
            [0.0, 1.0, 2.0].map(|pass| symbol_pipeline(gpu, target, &layouts, pass));
        let cull_module = gpu.create_shader_module(&ShaderModuleDesc {
            label: CULL_SHADER.label,
            wgsl: CULL_SHADER.source().into(),
        });
        let cull_pipeline = gpu.create_compute_pipeline(&ComputePipelineDesc {
            label: CULL_SHADER.label,
            bind_group_layouts: &[cull_layout],
            compute: ShaderStage {
                module: cull_module,
                entry_point: CULL_SHADER.compute,
                constants: &[],
            },
        });

        let image_bindings = images;
        let images = image::Images::new(gpu, target, draw_layout);
        let heatmaps = heatmap::Heatmaps::new(gpu, target, draw_layout, node_layout);
        let extrusions = extrusion::Extrusions::new(gpu, target, draw_layout, node_layout);
        let node_stride = NODE_UNIFORM_SIZE.max(u64::from(
            gpu.caps().limits.min_uniform_buffer_offset_alignment,
        ));
        let node_uniforms = gpu.create_buffer(&BufferDesc {
            label: "node uniforms",
            size: node_stride * u64::from(capacity.nodes),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Self {
            target,
            sky: None,
            scene,
            draw_group,
            cull_group,
            draw_layout,
            node_layout,
            paint,
            bytecode: None,
            symbol_pipelines,
            cull_pipeline,
            node_uniforms,
            node_stride,
            nodes: Vec::new(),
            bundle: None,
            paints_written: 0,
            images,
            heatmaps,
            extrusions,
            glyphs_bound: glyphs.revision(),
            nearest,
            glyphs,
            image_bindings,
        }
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// The glyph atlas symbol quads sample.
    pub fn glyphs(&mut self) -> &mut GlyphAtlas {
        &mut self.glyphs
    }

    /// Uses four samples for 3D surfaces by default, or one when the host
    /// explicitly disables antialiasing (e.g. reference image comparisons).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        if self.extrusions.set_antialias(enabled) {
            self.invalidate_bundle(gpu);
        }
    }

    /// The scene's per-node draw arguments and cull counters, for
    /// diagnostics: [`ARGS_WORDS`] words per node, as
    /// `cull.wgsl` writes them, and the number of nodes.
    pub fn draw_args(&self) -> (BufferId, u32) {
        (self.scene.buffers().args, self.nodes.len() as u32)
    }

    /// Replaces the render nodes, in draw order. The scene must hold no
    /// meshes.
    pub fn set_nodes(&mut self, gpu: &mut dyn Gpu, families: &[Family]) {
        let retained = vec![None; families.len()];
        self.remap_nodes(gpu, families, &retained);
    }

    /// Replaces the render nodes, in draw order, keeping some with their
    /// meshes: new node `i` is old node `retained[i]`, of the same family,
    /// or a new node. Old nodes not retained must hold no meshes. Paint is
    /// written again for every node.
    pub fn remap_nodes(
        &mut self,
        gpu: &mut dyn Gpu,
        families: &[Family],
        retained: &[Option<u32>],
    ) {
        assert_eq!(families.len(), retained.len());
        assert!(
            families.len() as u32 <= self.scene.capacity().nodes,
            "{} render nodes exceed the scene's {}",
            families.len(),
            self.scene.capacity().nodes
        );
        let mut old: Vec<Option<Node>> = self.nodes.drain(..).map(Some).collect();
        let mut map = vec![None; old.len()];
        let mut kept: Vec<Option<Node>> = Vec::with_capacity(families.len());
        for (i, (&family, &from)) in families.iter().zip(retained).enumerate() {
            let node = from.map(|from| {
                let node = old[from as usize]
                    .take()
                    .expect("each old node is retained once");
                assert_eq!(node.family, family, "a retained node keeps its family");
                map[from as usize] = Some(i as u32);
                node
            });
            kept.push(node);
        }
        for node in old.into_iter().flatten() {
            gpu.destroy_bind_group(node.group);
            if let Some(extrusion) = node.extrusion {
                extrusion::Extrusions::release(gpu, extrusion);
            }
            if let Some(heatmap) = node.heatmap {
                heatmap::Heatmaps::release(gpu, heatmap);
            }
            if let Some(image) = node.image {
                image::Images::release_node(gpu, image);
            }
        }
        for (i, (&family, node)) in families.iter().zip(kept).enumerate() {
            let uniform = self.node_uniform(i as u32);
            let (mut image, cull) = match node {
                Some(node) => {
                    // Groups bind the uniform at the node's index.
                    gpu.destroy_bind_group(node.group);
                    if let Some(extrusion) = node.extrusion {
                        extrusion::Extrusions::release(gpu, extrusion);
                    }
                    if let Some(heatmap) = node.heatmap {
                        heatmap::Heatmaps::release(gpu, heatmap);
                    }
                    (node.image, node.cull)
                }
                None => (
                    family.is_image().then(image::NodeImage::default),
                    NodeCull {
                        margin_px: 0.0,
                        elevation: 0.0,
                        draw: family.draw(),
                        visible: true,
                    },
                ),
            };
            let group = match &mut image {
                Some(image) => self.images.node_group(gpu, family, image, uniform),
                None => gpu.create_bind_group(&BindGroupDesc {
                    label: "node",
                    layout: self.node_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform,
                    }],
                }),
            };
            self.nodes.push(Node {
                family,
                bytecode: false,
                group,
                extrusion: (family == Family::FillExtrusion)
                    .then(extrusion::NodeExtrusion::default),
                heatmap: (family == Family::Heatmap).then(|| heatmap::Heatmaps::node(gpu)),
                cull,
                written: Vec::new(),
                image,
                separate_halo: false,
            });
        }
        let cull: Vec<NodeCull> = self.nodes.iter().map(|n| n.cull).collect();
        self.scene.remap_nodes(&map, &cull);
        self.invalidate_bundle(gpu);
    }

    fn node_uniform(&self, node: u32) -> BindingResource {
        BindingResource::Buffer {
            buffer: self.node_uniforms,
            offset: u64::from(node) * self.node_stride,
            size: Some(NODE_UNIFORM_SIZE),
        }
    }

    fn invalidate_bundle(&mut self, gpu: &mut dyn Gpu) {
        if let Some(bundle) = self.bundle.take() {
            gpu.destroy_render_bundle(bundle);
        }
    }

    /// A page of tile textures, `size` texels per tile, for image nodes to
    /// bind with [`bind_page`](Self::bind_page).
    pub fn create_page(&mut self, gpu: &mut dyn Gpu, kind: PageKind, size: [u32; 2]) -> PageId {
        self.images.create_page(gpu, kind, size)
    }

    /// Destroys a page. Nodes that bind it must be rebound or replaced.
    pub fn remove_page(&mut self, gpu: &mut dyn Gpu, page: PageId) {
        self.images.remove_page(gpu, page);
    }

    /// The size of a page's tiles in texels.
    pub fn page_size(&self, page: PageId) -> [u32; 2] {
        self.images.page_size(page)
    }

    /// Makes an image node draw from `page`'s textures.
    pub fn bind_page(&mut self, node: u32, page: PageId) {
        let image = self.nodes[node as usize]
            .image
            .as_mut()
            .expect("pages bind to image nodes");
        image::Images::bind_page(image, page);
    }

    /// A free layer of `page` for one tile, growing the page if needed.
    pub fn alloc_layer(&mut self, gpu: &mut dyn Gpu, page: PageId) -> Result<u32, OutOfSpace> {
        self.images.alloc_layer(gpu, page)
    }

    pub fn free_layer(&mut self, page: PageId, layer: u32) {
        self.images.free_layer(page, layer);
    }

    /// Writes a tile's image into its layer: premultiplied colour for a
    /// raster page, the encoded elevations for a DEM page.
    pub fn write_layer(&mut self, gpu: &mut dyn Gpu, page: PageId, layer: u32, pixels: Pixels<'_>) {
        self.images.write_layer(gpu, page, layer, pixels);
    }

    /// Queues a DEM tile for the next frame's prepare pass; a later job for
    /// the same layer replaces an earlier one.
    pub fn prepare_dem(&mut self, page: PageId, job: DemPrepare) {
        self.images.prepare_dem(page, job);
    }

    /// Sets a color-relief node's ramp: per stop, its premultiplied rgba8
    /// colour and its elevation packed as a DEM texel.
    pub fn set_ramp(&mut self, gpu: &mut dyn Gpu, node: u32, colors: &[u32], elevations: &[u32]) {
        let image = self.nodes[node as usize]
            .image
            .as_mut()
            .expect("ramps belong to color-relief nodes");
        self.images.set_ramp(gpu, image, colors, elevations);
    }

    /// Updates a heatmap's 256-texel premultiplied RGBA ramp.
    pub fn set_heatmap_ramp(&mut self, gpu: &mut dyn Gpu, node: u32, colors: &[u32]) {
        heatmap::Heatmaps::set_ramp(
            gpu,
            self.nodes[node as usize]
                .heatmap
                .as_mut()
                .expect("heatmap node"),
            colors,
        );
    }

    /// Rebuilds image node groups whose pages or ramps changed.
    fn refresh_image_groups(&mut self, gpu: &mut dyn Gpu) {
        let mut rebuilt = false;
        for i in 0..self.nodes.len() {
            let uniform = self.node_uniform(i as u32);
            let node = &mut self.nodes[i];
            let Some(image) = &mut node.image else {
                continue;
            };
            if !self.images.stale(image) {
                continue;
            }
            gpu.destroy_bind_group(node.group);
            node.group = self.images.node_group(gpu, node.family, image, uniform);
            rebuilt = true;
        }
        if rebuilt {
            self.invalidate_bundle(gpu);
        }
    }

    /// Writes a node's paint values. The paint's family must match the
    /// node's.
    pub fn set_paint(&mut self, gpu: &mut dyn Gpu, node: u32, paint: &NodePaint) {
        let bytecode = paint.gpu().is_some_and(|g| g.mask != 0);
        if bytecode != self.nodes[node as usize].bytecode {
            if bytecode && self.bytecode.is_none() {
                let layouts = [self.draw_layout, self.node_layout];
                self.bytecode = Some(PaintPipelines::new(gpu, self.target, &layouts, true));
            }
            self.nodes[node as usize].bytecode = bytecode;
            self.invalidate_bundle(gpu);
        }
        let entry = &mut self.nodes[node as usize];
        assert!(
            paint.fits(entry.family),
            "paint family mismatch for node {node}"
        );
        let mut words = paint.uniform_words(node);
        if let NodePaint::Symbol(p) = paint {
            words[14] = self.glyphs.table_base();
            words[15] = (GLYPH_PAGE as f32).to_bits();
            words[18] = (self.image_bindings.size as f32).to_bits();
            if entry.separate_halo != p.separate_halo {
                entry.separate_halo = p.separate_halo;
                if let Some(bundle) = self.bundle.take() {
                    gpu.destroy_render_bundle(bundle);
                }
            }
        }
        if words != entry.written {
            gpu.write_buffer(
                self.node_uniforms,
                u64::from(node) * self.node_stride,
                bytemuck::cast_slice(&words),
            );
            entry.written = words;
            self.paints_written += 1;
        }
        let margin = paint.cull_margin_px();
        let elevation = match paint {
            NodePaint::Extrusion(p) | NodePaint::ExtrusionPattern(p, _) => {
                p.height.max(p.base).max(0.0)
            }
            _ => 0.0,
        };
        if margin != entry.cull.margin_px || elevation != entry.cull.elevation {
            entry.cull.margin_px = margin;
            entry.cull.elevation = elevation;
            self.sync_cull();
        }
    }

    /// Shows or hides a node without re-recording the bundle: hidden nodes
    /// contribute no cull candidates.
    pub fn set_visible(&mut self, node: u32, visible: bool) {
        let entry = &mut self.nodes[node as usize];
        if entry.cull.visible != visible {
            entry.cull.visible = visible;
            self.sync_cull();
        }
    }

    fn sync_cull(&mut self) {
        let cull: Vec<NodeCull> = self.nodes.iter().map(|n| n.cull).collect();
        self.scene.set_nodes(&cull);
    }

    /// Sets the style's sky, drawn before the map. An absent sky leaves the
    /// host's backing colour visible, matching GL JS's transparent default.
    pub fn set_sky(&mut self, gpu: &mut dyn Gpu, paint: Option<SkyPaint>) {
        if paint.is_some() && self.sky.is_none() {
            self.sky = Some(sky::Sky::new(gpu, self.target, self.scene.buffers().frame));
        }
        if let Some(sky) = &mut self.sky
            && sky.set(gpu, paint)
        {
            self.paints_written += 1;
        }
    }

    fn sky_commands(&self) -> Vec<RenderCommand> {
        self.sky.as_ref().map_or_else(Vec::new, sky::Sky::commands)
    }

    /// Draws a frame into `target`, cleared to `clear` (premultiplied RGBA).
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
    ) -> FrameStats {
        if self.glyphs.revision() != self.glyphs_bound {
            // The glyph pages grew into new textures.
            gpu.destroy_bind_group(self.draw_group);
            let b = *self.scene.buffers();
            self.draw_group = scene_group(
                gpu,
                self.draw_layout,
                &b,
                &self.image_bindings,
                self.glyphs.texture(),
                self.nearest,
            );
            self.glyphs_bound = self.glyphs.revision();
            self.invalidate_bundle(gpu);
            // Extrusion and heatmap nodes record the scene group too.
            for node in &mut self.nodes {
                if let Some(extrusion) = &mut node.extrusion {
                    extrusion.forget_bundle(gpu);
                }
                if let Some(heatmap) = &mut node.heatmap {
                    heatmap.forget_bundle(gpu);
                }
            }
        }
        let mut stats = self.render_modern(gpu, frame, target, clear);
        stats.draws += u32::from(self.sky.as_ref().is_some_and(|sky| sky.enabled));
        stats
    }

    fn render_modern(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
    ) -> FrameStats {
        let candidates = self.scene.prepare(gpu, frame);
        self.refresh_image_groups(gpu);
        let density_size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio / 4.0).floor().max(1.0) as u32);
        let size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio).max(1.0) as u32);
        let mut resized = self.extrusions.resize(
            gpu,
            size,
            self.nodes.iter().filter(|n| n.extrusion.is_some()).count(),
        );
        for i in 0..self.nodes.len() {
            let uniform = self.node_uniform(i as u32);
            if let Some(extrusion) = &mut self.nodes[i].extrusion {
                resized |= self.extrusions.resize_node(gpu, extrusion, uniform, size);
            }
            if let Some(heatmap) = &mut self.nodes[i].heatmap {
                resized |= self.heatmaps.resize(gpu, heatmap, uniform, density_size);
            }
        }
        if resized {
            self.invalidate_bundle(gpu);
        }
        let bundle_recorded = self.bundle.is_none();
        if bundle_recorded {
            let mut commands = Vec::with_capacity(self.nodes.len() * 4 + 1);
            if self.nodes.iter().any(|n| n.cull.draw == NodeDraw::Indexed) {
                commands.push(RenderCommand::SetIndexBuffer {
                    buffer: self.scene.buffers().compacted,
                    format: IndexFormat::Uint32,
                    offset: 0,
                });
            }
            for i in 0..self.nodes.len() {
                self.node_commands(i, &mut commands);
            }
            self.bundle = Some(gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "layers",
                    color_formats: &[self.target.color],
                    depth_format: None,
                    sample_count: self.target.sample_count,
                },
                &commands,
            ));
        }

        let mut passes = Vec::with_capacity(3);
        let mut stats = FrameStats {
            draws: self.nodes.len() as u32,
            bundle_recorded,
            candidates,
            placements_written: self.scene.take_placements_written(),
            paints_written: std::mem::take(&mut self.paints_written),
            ..FrameStats::default()
        };
        if let Some((prepare, prepared)) = self.images.prepare_pass(gpu) {
            stats.dispatches += 2;
            stats.prepared = prepared;
            passes.push(Pass::Compute(prepare));
        }
        // The dispatch writes every node's arguments, zeros included, so it
        // runs even when nothing is resident. Indexed nodes' batches add to
        // cleared arguments.
        if !self.nodes.is_empty() {
            if self.nodes.iter().any(|n| n.cull.draw == NodeDraw::Indexed) {
                passes.push(Pass::ClearBuffer {
                    buffer: self.scene.buffers().args,
                    offset: 0,
                    size: Some(args_offset(self.nodes.len() as u32)),
                });
            }
            let [x, y] = self.scene.cull_grid();
            stats.dispatches += 1;
            passes.push(Pass::Compute(ComputePass {
                label: "cull".into(),
                timestamps: None,
                commands: vec![
                    ComputeCommand::SetPipeline(self.cull_pipeline),
                    ComputeCommand::SetBindGroup {
                        index: 0,
                        group: self.cull_group,
                    },
                    ComputeCommand::Dispatch { x, y, z: 1 },
                ],
            }));
        }
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(heatmap) = &mut node.heatmap
                && let Some(pass) = self.heatmaps.prepare(
                    gpu,
                    heatmap,
                    self.draw_group,
                    node.group,
                    self.scene.buffers().args,
                    index as u32,
                )
            {
                stats.draws += 1;
                passes.push(Pass::Render(pass));
            }
        }
        let mut first = true;
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(extrusion) = &mut node.extrusion {
                passes.push(Pass::Render(self.extrusions.prepare(
                    gpu,
                    extrusion,
                    self.draw_group,
                    node.group,
                    self.scene.buffers(),
                    index as u32,
                    first,
                )));
                first = false;
                stats.draws += 1;
            }
        }
        let mut commands = self.sky_commands();
        let bundle = self.bundle.expect("recorded above");
        commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
        passes.push(Pass::Render(RenderPass {
            label: "main".into(),
            color: ColorAttachment {
                store: StoreOp::Store,
                target,
                resolve: None,
                load: LoadOp::Clear(clear),
            },
            depth: None,
            timestamps: None,
            commands,
        }));
        gpu.submit(&passes);
        stats
    }

    /// Node `i`'s commands in the main pass, as the bundle records them.
    fn node_commands(&self, i: usize, commands: &mut Vec<RenderCommand>) {
        let node = &self.nodes[i];
        let args = args_offset(i as u32);
        if let Some(extrusion) = &node.extrusion {
            if let Some(group) = extrusion.group {
                commands.extend([
                    RenderCommand::SetPipeline(self.extrusions.composite),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: self.draw_group,
                    },
                    RenderCommand::SetBindGroup { index: 1, group },
                    RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                ]);
            }
            return;
        }
        if node.family == Family::Heatmap {
            if let Some(group) = node.heatmap.as_ref().and_then(|h| h.group) {
                commands.extend([
                    RenderCommand::SetPipeline(self.heatmaps.color_pipeline),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: self.draw_group,
                    },
                    RenderCommand::SetBindGroup { index: 1, group },
                    RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                ]);
            }
            return;
        }
        if node.family.is_symbol() {
            let passes: &[RenderPipelineId] = if node.separate_halo {
                &self.symbol_pipelines[1..]
            } else {
                &self.symbol_pipelines[..1]
            };
            commands.extend([
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: self.draw_group,
                },
                RenderCommand::SetBindGroup {
                    index: 1,
                    group: node.group,
                },
            ]);
            for &pipeline in passes {
                commands.extend([
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::DrawIndirect {
                        buffer: self.scene.buffers().args,
                        offset: args,
                    },
                ]);
            }
            return;
        }
        let paint = match (node.bytecode, &self.bytecode) {
            (true, Some(bytecode)) => bytecode,
            _ => &self.paint,
        };
        let pipeline = paint.get(node.family).unwrap_or_else(|| {
            self.images
                .pipeline(node.family)
                .expect("image families have pipelines")
        });
        commands.extend([
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup {
                index: 0,
                group: self.draw_group,
            },
            RenderCommand::SetBindGroup {
                index: 1,
                group: node.group,
            },
            if node.cull.draw == NodeDraw::Indexed {
                RenderCommand::DrawIndexedIndirect {
                    buffer: self.scene.buffers().args,
                    offset: args,
                }
            } else {
                RenderCommand::DrawIndirect {
                    buffer: self.scene.buffers().args,
                    offset: args,
                }
            },
        ]);
    }
}

/// The vector families' pipelines of one paint class.
#[derive(Clone, Copy, Debug)]
struct PaintPipelines {
    fill: RenderPipelineId,
    fill_outline: RenderPipelineId,
    line: RenderPipelineId,
    circle: RenderPipelineId,
}

impl PaintPipelines {
    fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        layouts: &[BindGroupLayoutId],
        bytecode: bool,
    ) -> Self {
        let mut pipeline = |shader: &Shader, topology| {
            draw_pipeline(gpu, shader, bytecode, topology, target, layouts)
        };
        Self {
            fill: pipeline(&FILL_SHADER, PrimitiveTopology::TriangleList),
            fill_outline: pipeline(&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            line: pipeline(&LINE_SHADER, PrimitiveTopology::TriangleList),
            circle: pipeline(&CIRCLE_SHADER, PrimitiveTopology::TriangleList),
        }
    }

    fn get(&self, family: Family) -> Option<RenderPipelineId> {
        Some(match family {
            Family::Fill => self.fill,
            Family::FillOutline => self.fill_outline,
            Family::Line => self.line,
            Family::Circle => self.circle,
            _ => return None,
        })
    }
}

/// A draw pipeline over the scene's bindings: `layouts` starts with the
/// scene and node groups.
fn draw_pipeline(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    bytecode: bool,
    topology: PrimitiveTopology,
    target: TargetFormat,
    layouts: &[BindGroupLayoutId],
) -> RenderPipelineId {
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: shader.label,
        wgsl: if bytecode {
            shader.bytecode_source()
        } else {
            shader.source()
        }
        .into(),
    });
    gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: shader.label,
        bind_group_layouts: layouts,
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
            format: target.color,
            blend: Some(BlendState::PremultipliedAlpha),
        }],
        depth: None,
        topology,
        sample_count: target.sample_count,
    })
}

/// The scene's group: its buffers, the image pages and the glyph pages.
fn scene_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    b: &orb_scene::SceneBuffers,
    images: &ImageBindings,
    glyphs: TextureId,
    nearest: SamplerId,
) -> BindGroupId {
    let whole = BindingResource::buffer;
    let mut entries: Vec<BindGroupEntry> = [
        b.frame,
        b.tiles,
        b.chunks,
        b.visible,
        b.vertices,
        b.indices,
        b.props,
        b.node_cull,
        images.table,
    ]
    .iter()
    .enumerate()
    .map(|(i, &buffer)| BindGroupEntry {
        binding: i as u32,
        resource: whole(buffer),
    })
    .collect();
    let texture = |binding, texture, dimension| BindGroupEntry {
        binding,
        resource: BindingResource::Texture { texture, dimension },
    };
    entries.extend([
        texture(9, images.texture, TextureViewDimension::D2Array),
        BindGroupEntry {
            binding: 10,
            resource: BindingResource::Sampler(images.sampler),
        },
        texture(11, images.dashes, TextureViewDimension::D2),
        texture(12, glyphs, TextureViewDimension::D2Array),
        BindGroupEntry {
            binding: 13,
            resource: BindingResource::Sampler(nearest),
        },
    ]);
    gpu.create_bind_group(&BindGroupDesc {
        label: "scene",
        layout,
        entries: &entries,
    })
}

/// A symbol pipeline drawing halo and fill together (`pass` 0), or the
/// halo (1) or the fill (2) alone.
fn symbol_pipeline(
    gpu: &mut dyn Gpu,
    target: TargetFormat,
    layouts: &[BindGroupLayoutId],
    pass: f64,
) -> RenderPipelineId {
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: SYMBOL_SHADER.label,
        wgsl: SYMBOL_SHADER.source().into(),
    });
    let constants = [("PASS", pass)];
    gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: SYMBOL_SHADER.label,
        bind_group_layouts: layouts,
        vertex: ShaderStage {
            module,
            entry_point: SYMBOL_SHADER.vertex,
            constants: &constants,
        },
        fragment: ShaderStage {
            module,
            entry_point: SYMBOL_SHADER.fragment,
            constants: &constants,
        },
        targets: &[ColorTargetState {
            format: target.color,
            blend: Some(BlendState::PremultipliedAlpha),
        }],
        depth: None,
        topology: PrimitiveTopology::TriangleList,
        sample_count: target.sample_count,
    })
}
