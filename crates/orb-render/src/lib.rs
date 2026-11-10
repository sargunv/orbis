//! The GPU-driven render graph.
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
//! their vertices.
//!
//! The `scene` module holds what a frame reads on the GPU: the persistent
//! tables, the arenas of tile streams and the atlases, kept consistent as
//! meshes arrive and leave.

#![forbid(unsafe_code)]

mod atmosphere;
mod downsample;
mod extrusion;
mod heatmap;
mod image;
mod layer_composite;
mod offscreen;
mod paint;
mod pipeline;
mod release;
mod scene;
mod terrain;
mod tiles;
pub use atmosphere::{AtmospherePaint, AtmosphereView, GlobeSky};
pub use layer_composite::Blend;
pub use terrain::{
    DRAPE_FORMAT, DRAPE_SLOT_PX, DrapeLayout, TERRAIN_EXTENT, TERRAIN_GRID, TerrainDem,
    TerrainGlobe, TerrainPaint, TerrainTile,
};

pub use image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, ImageFade, PageId, PageKind};
pub use orb_layout::{Family, pack_color};
pub use paint::{
    CirclePaint, ColorReliefPaint, Curve, EXTRUSION_GRADIENT_ROWS, EXTRUSION_OPACITY_ROWS,
    ElevationPaint, ExtrusionPaint, FillPaint, HeatmapPaint, HillshadeMethod, HillshadePaint,
    LinePaint, MAX_LIGHTS, NodePaint, PatternPaint, RasterPaint, StateLookup, SymbolPaint,
    SymbolSize, Translate,
};
pub use scene::Pixels;
pub use scene::{
    ARGS_WORDS, ClipRect, DemRef, FrameBlock, FrameGlobe, FrameLight, FrameTerrain, ImageAtlas,
    ImageBindings, ImageKey, ImageMetadata, ImageUpload, MeshData, MeshId, MeshSize, NodeCull,
    NodeDraw, OutOfSpace, PlacementTarget, RowGlobe, SceneCapacity, SceneStats, TileId, TileRow,
    TileWarp, Words, warp_words,
};
pub use scene::{DESKTOP_BUDGET, MOBILE_BUDGET, PoolMemory, SceneMemory};
pub use scene::{
    GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas, PLACED_WORDS, QUAD_WORDS, SYMBOL_HEADER_WORDS,
    SYMBOL_STATE_WORDS, SymbolMeshData, pack_image_rect, symbol_header,
};
pub use tiles::{
    GpuTile, ImageBuild, ImageMesh, MeshBuild, MeshRows, TileBuild, TileKey, UploadError, WarpKey,
    changed_borders, dem_neighbours, tile_latitudes,
};

use crate::pipeline::Draw;
use crate::release::Release;
use std::collections::BTreeMap;

use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, CULL_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_SHADER, LINE_STYLED_SHADER,
    SYMBOL_SHADERS,
};
use scene::Scene;

/// How a family's nodes draw: meshes from compacted
/// indices, instanced quads by vertex pulling.
fn draw_of(family: Family) -> NodeDraw {
    match family {
        Family::Fill | Family::FillOutline | Family::Line | Family::FillExtrusion => {
            NodeDraw::Indexed
        }
        Family::Circle | Family::Heatmap | Family::SymbolIcon | Family::SymbolText => {
            NodeDraw::Points
        }
        Family::Raster | Family::Hillshade | Family::ColorRelief => NodeDraw::Images,
    }
}

/// Where node `index`'s draw arguments start in the scene's args buffer.
fn args_offset(index: u32) -> u64 {
    u64::from(index) * crate::scene::ARGS_WORDS as u64 * 4
}

/// A premultiplied sRGB-encoded colour as it mixes: itself, or when
/// `linear` its unpremultiplied colour decoded to linear light,
/// premultiplied (`color.wgsl`'s `premultiplied_to_linear`).
pub(crate) fn color_in(c: [f32; 4], linear: bool) -> [f32; 4] {
    let a = c[3];
    if !linear || a <= 0.0 {
        return if linear { [0.0; 4] } else { c };
    }
    let decode = |v: f32| {
        let v = (v / a).clamp(0.0, 1.0);
        let lin = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
        lin * a
    };
    [decode(c[0]), decode(c[1]), decode(c[2]), a]
}

/// The attachment a renderer draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetFormat {
    pub color: TextureFormat,
    pub sample_count: u32,
}

/// Counters for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub dispatches: u32,
    /// Draws issued: the main bundles' (a node with candidates, a
    /// layer's composite), and other passes'.
    pub draws: u32,
    /// Whether the bundle was re-recorded this frame.
    pub bundle_recorded: bool,
    pub candidates: u32,
    /// The cull's candidates were laid out again: tiles or placements came
    /// or went, or nodes changed how they draw.
    pub candidates_rebuilt: bool,
    /// Objects the CPU wrote since the last frame: tile
    /// placement rows and node uniforms. A camera-only frame rewrites only
    /// the placements in view.
    pub placements_written: u32,
    pub paints_written: u32,
    /// DEM tiles the prepare pass processed.
    pub prepared: u32,
    /// Pipelines the device is still compiling.
    pub pipelines_pending: u32,
    /// Nodes not drawn because a pipeline they need is still compiling.
    pub nodes_waiting: u32,
    /// Something in the frame waited for a pipeline (nodes, terrain, the
    /// DEM prepare, the sky): the frame isn't complete.
    pub waited: bool,
    pub times: RenderTimes,
}

/// Where [`Renderer::render`]'s CPU time went, in microseconds of the
/// caller's clock. What it doesn't name (binding, pipelines, targets) is
/// the rest of its time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderTimes {
    /// The cull's candidates laid out again and tile placement rows
    /// written.
    pub cull: u32,
    /// The main pass's bundles recorded again.
    pub bundles: u32,
    /// The drape passes of slots redrawn: their bundles and passes.
    pub drape: u32,
    /// The frame's other passes listed (wgpu records them in `submit`).
    pub encode: u32,
    /// The passes and the buffer writes staged for them recorded into
    /// wgpu's encoder and submitted.
    pub submit: u32,
}

/// Microseconds from `from` to `to`, two readings of a millisecond clock.
pub fn micros(from: f64, to: f64) -> u32 {
    ((to - from) * 1000.0).round().max(0.0) as u32
}

/// A render node: its family, and the source an image node draws the
/// tiles of (by style key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderNode {
    pub family: Family,
    pub source: Option<u32>,
}

#[derive(Debug)]
struct Node {
    family: Family,
    source: Option<u32>,
    group: BindGroupId,
    cull: NodeCull,
    /// The uniform words last written, to skip unchanged writes.
    written: Vec<u32>,
    /// What an image family's group binds.
    image: Option<image::NodeImage>,
    heatmap: Option<heatmap::NodeHeatmap>,
    /// An extrusion node's colour target.
    extrusion: Option<extrusion::NodeExtrusion>,
    /// A symbol node draws its halos in a pass of their own.
    separate_halo: bool,
    /// A symbol node places its labels along lines.
    along_line: bool,
    /// A line node has dashes, a pattern or a gradient: it draws with the
    /// styled line pipeline.
    styled: bool,
    /// What an extrusion node's pipelines are specialised for.
    elevation: paint::ElevationShape,
    /// What an extrusion node darkens on the ground: the frame's shadows,
    /// its paint's occlusion. Its paint asks for occlusion.
    ground: extrusion::Ground,
    occlusion: bool,
    /// What its recorded draw darkens: what of `ground` had compiled.
    ground_drawn: extrusion::Ground,
    /// On terrain, the drape stack a draped node draws into, and the
    /// terrain layer that composites it.
    stack: Option<u8>,
    terrain_layer: Option<u8>,
    /// A hillshade the terrain pass lights from the frame's bearing: it
    /// drapes its unlit slopes ([`Renderer::set_shaded_on_terrain`]).
    shaded: bool,
    /// Its style layer's key, layer opacity and blend mode
    /// (`*-layer-opacity`, `*-layer-blend`): below opacity 1 or blending
    /// other than normally, the layer composites once.
    composite: Option<(u32, f32, Blend)>,
    /// It has been shown since it was made: its pipelines were asked for,
    /// and the bundles record it. Until then it costs nothing, neither
    /// pipelines nor targets nor passes.
    drawn: bool,
    /// A bundle recorded since the last invalidation left it out: it had
    /// no candidates, so its draw would have drawn nothing. Candidates
    /// arriving for it record the bundles again.
    left_out: std::cell::Cell<bool>,
}

impl Node {
    /// A hillshade the terrain pass lights: its stack holds its slopes.
    fn shades_on_terrain(&self) -> bool {
        self.shaded && self.stack.is_some()
    }

    /// Its paint pipeline's family, and whether it's the styled line's.
    fn paint_kind(&self) -> (Family, bool) {
        (self.family, self.styled)
    }

    /// What the cull sees of it.
    fn scene_cull(&self) -> NodeCull {
        self.cull
    }
}

#[derive(Debug)]
pub struct Renderer {
    id: u64,
    /// What it draws into: the host's target, or in linear light an sRGB
    /// texture of its own ([`offscreen::LinearScene`]), encoded into the
    /// host's at the end of each frame.
    target: TargetFormat,
    /// The host's target.
    host: TargetFormat,
    /// Colours mix in linear light.
    linear: Option<offscreen::LinearScene>,
    atmosphere: atmosphere::Atmosphere,
    scene: Scene,
    draw_group: BindGroupId,
    /// The cull dispatch's pipeline and group.
    cull: (ComputePipelineId, BindGroupId),
    draw_layout: BindGroupLayoutId,
    node_layout: BindGroupLayoutId,
    /// The vector families' pipelines, each made for the family's first
    /// node.
    paint: PaintPipelines,
    /// Point and line labels' symbol pipelines, each made for the first
    /// node that draws with it.
    symbol_pipelines: [Option<RenderPipelineId>; 2],
    /// The symbol shader's pass (group 2): halo and fill together, halo
    /// only, fill only.
    symbol_passes: [BindGroupId; 3],
    symbol_pass_buffer: BufferId,
    symbol_pass_layout: BindGroupLayoutId,
    /// Nodes changed since pipelines were last made for them, or one
    /// was shown.
    pipelines_stale: bool,
    /// Pipelines the device had compiled when the bundles were recorded.
    made_seen: u32,
    /// Nodes the main bundle left out, waiting for a pipeline.
    waiting: u32,
    /// Whether a drape bundle has left out a draped node waiting for a
    /// pipeline since every slot was last redrawn. A slot keeps lacking
    /// the node whatever later recordings leave out (one whose slots the
    /// node has no candidates in leaves out nothing), so the pipeline's
    /// arrival redraws every slot.
    drape_waiting: bool,
    /// Nodes the main bundles record draws for.
    recorded: u32,
    node_uniforms: BufferId,
    node_stride: u64,
    /// The node uniforms as written, and the nodes changed since the last
    /// frame: one buffer write a frame, not one a node.
    node_words: Vec<u32>,
    nodes_dirty: Option<std::ops::Range<usize>>,
    nodes: Vec<Node>,
    tiles: tiles::Tiles,
    surface_warps: tiles::SurfaceWarps,
    /// The main pass's bundles: one, or one more than the groups that
    /// blend other than normally (empty until recorded).
    bundles: Vec<RenderBundleId>,
    /// Node uniform writes since the last frame.
    paints_written: u32,
    images: image::Images,
    /// Mip levels for raster pages, DEM slopes and the drape atlas.
    downsampler: downsample::Downsampler,
    heatmaps: heatmap::Heatmaps,
    extrusions: extrusion::Extrusions,
    /// Layers drawn whole and composited at their layer opacity.
    /// Runs of nodes composited once (`*-layer-opacity`, `-blend`).
    layers: layer_composite::LayerComposites,
    /// What draws apart from the map composites through these.
    composites: offscreen::Composites,
    /// Made for the first terrain.
    terrain: Option<terrain::Terrain>,
    /// The drape passes of each style zoom slots are drawn at.
    drape_zooms: BTreeMap<u8, DrapeZoom>,
    /// What each terrain draw composites, in style order.
    terrain_layers: Vec<terrain::TerrainLayer>,
    /// Colour and slope stacks of the terrain layers.
    drape_stacks: [u32; 2],
    /// Draped families' pipelines for the drape atlas, made for the first
    /// terrain when the target's format isn't the atlas's.
    drape_pipelines: Option<DrapePipelines>,
    /// Bound for elevated content without terrain.
    no_dem: TextureId,
    /// The DEM texture the scene groups bind.
    dem_bound: TextureId,
    /// Bound for symbols and circles without terrain.
    no_terrain_depth: TextureId,
    /// The terrain depth bound in the scene groups.
    terrain_depth_bound: TextureId,
    glyphs: GlyphAtlas,
    /// The glyph atlas revision the scene group binds.
    glyphs_bound: u32,
    /// The scene's buffer revision its groups bind.
    buffers_bound: u64,
    cull_layout: BindGroupLayoutId,
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
    /// The image texture the scene group binds.
    images_bound: TextureId,
    nearest: SamplerId,
}

/// Bytes reserved per node uniform; every node struct fits.
const NODE_UNIFORM_SIZE: u64 = 512;

impl Renderer {
    /// Frees everything the renderer holds on `gpu`, the device it was
    /// made on, so what a map's frames made goes with it. The images it
    /// was handed stay.
    pub fn destroy(self, gpu: &mut dyn Gpu) {
        use release::Release;
        self.scene.destroy(gpu);
        self.glyphs.destroy(gpu);
        let layouts = (
            self.draw_layout,
            self.node_layout,
            self.symbol_pass_layout,
            self.cull_layout,
        );
        let groups = (
            self.draw_group,
            self.cull,
            self.symbol_passes,
            self.symbol_pass_buffer,
        );
        let pipelines = (self.paint.pipelines, self.symbol_pipelines);
        (
            layouts,
            groups,
            pipelines,
            self.node_uniforms,
            self.bundles,
            self.nodes,
        )
            .release(gpu);
        (
            self.atmosphere,
            self.images,
            self.downsampler,
            self.heatmaps,
            self.extrusions,
        )
            .release(gpu);
        (self.layers, self.composites, self.linear).release(gpu);
        let drape = self
            .drape_pipelines
            .map(|d| (d.paint.pipelines, d.images.0));
        (
            self.terrain,
            self.drape_zooms,
            drape,
            self.no_dem,
            self.no_terrain_depth,
            self.nearest,
        )
            .release(gpu);
    }

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
        Self::with_color_space(gpu, target, capacity, images, false)
    }

    /// [`with_images`](Self::with_images), mixing colours in linear light
    /// when `linear` (`dev.sargunv.orbis:color-space`) rather than as
    /// gamma-encoded values, as GL JS and MapLibre Native do.
    pub fn with_color_space(
        gpu: &mut dyn Gpu,
        host: TargetFormat,
        capacity: SceneCapacity,
        images: ImageBindings,
        linear: bool,
    ) -> Self {
        let target = TargetFormat {
            color: if linear {
                TextureFormat::Rgba8UnormSrgb
            } else {
                host.color
            },
            ..host
        };
        let scene = Scene::new(gpu, capacity);
        let b = *scene.buffers();
        let (vs, fs, cs) = (
            ShaderStages::VERTEX,
            ShaderStages::FRAGMENT,
            ShaderStages::COMPUTE,
        );
        let vf = vs | fs;
        use BindGroupLayoutEntry as E;
        use TextureViewDimension::{D2, D2Array};
        let draw_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "scene",
            entries: &[
                E::uniform(0, vf),
                E::storage(1, vf),
                E::storage(2, vs),
                E::storage(3, vs),
                E::storage(4, vs),
                E::storage(5, vs),
                // Extrusions' fragments read colours over the elevation.
                E::storage(6, vf),
                E::storage(7, vs),
                E::storage(8, fs),
                E::texture(9, fs, true, D2Array),
                E::sampler(10, fs, true),
                E::texture(11, fs, true, D2),
                E::texture(12, fs, true, D2Array),
                E::sampler(13, fs, true),
                E::texture(14, vs, false, D2Array),
                E::texture(15, vf, false, D2),
            ],
        });
        let node_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node",
            entries: &[E::uniform(0, vf)],
        });
        let cull_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "cull",
            entries: &[
                E::uniform(0, cs),
                E::storage(1, cs),
                E::storage(2, cs),
                E::storage_rw(3, cs),
                E::storage(4, cs),
                E::storage_rw(5, cs),
                E::storage_rw(6, cs),
                E::storage(7, cs),
                E::storage_rw(8, cs),
            ],
        });

        let glyphs = GlyphAtlas::new(gpu, capacity.glyph_rows, b.props);
        let nearest = gpu.create_sampler(&SamplerDesc {
            label: "nearest",
            filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        let no_dem = gpu.create_texture(&TextureDesc {
            label: "no dem",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2Array,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        // Nothing hides symbols and circles without terrain: its depth is
        // the far plane, 0 in reversed-Z.
        let no_terrain_depth = gpu.create_texture(&TextureDesc {
            label: "no terrain depth",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        let draw_group = scene_group(
            gpu,
            draw_layout,
            &b,
            b.frame,
            &images,
            (glyphs.texture(), nearest, no_dem, no_terrain_depth),
        );
        let cull_group = cull_group(gpu, cull_layout, &b);

        let symbol_pass_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "symbol pass",
            entries: &[E::uniform(0, fs)],
        });
        let pass_stride = u64::from(gpu.caps().limits.min_uniform_buffer_offset_alignment).max(16);
        let pass_buffer = gpu.create_buffer(&BufferDesc {
            label: "symbol passes",
            size: 3 * pass_stride,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let symbol_passes = [0u32, 1, 2].map(|pass| {
            gpu.write_buffer(
                pass_buffer,
                u64::from(pass) * pass_stride,
                bytemuck::cast_slice(&[pass, 0, 0, 0]),
            );
            gpu.create_bind_group(&BindGroupDesc {
                label: "symbol pass",
                layout: symbol_pass_layout,
                entries: &[BindGroupEntry::slice(
                    0,
                    pass_buffer,
                    u64::from(pass) * pass_stride,
                    16,
                )],
            })
        });
        // The cull pipeline first: nothing draws without it.
        let cull = (
            pipeline::compute(gpu, &CULL_SHADER, &[cull_layout]),
            cull_group,
        );

        let image_bindings = images;
        let images = image::Images::new(gpu, target, draw_layout);
        let downsampler = downsample::Downsampler::new(gpu);
        let heatmaps = heatmap::Heatmaps::new(draw_layout, node_layout);
        let composites = offscreen::Composites::new(gpu, target);
        let linear = linear.then(|| offscreen::LinearScene::new(gpu, &composites, host));
        // 3D layers draw into RGBA8, in linear light its sRGB twin.
        let extrusion_format = if linear.is_some() {
            TextureFormat::Rgba8Unorm.srgb()
        } else {
            TextureFormat::Rgba8Unorm
        };
        let extrusions = extrusion::Extrusions::new(draw_layout, node_layout, extrusion_format);
        let node_stride = NODE_UNIFORM_SIZE.max(u64::from(
            gpu.caps().limits.min_uniform_buffer_offset_alignment,
        ));
        let node_uniforms = gpu.create_buffer(&BufferDesc {
            label: "node uniforms",
            size: node_stride * u64::from(capacity.nodes),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let atmosphere = atmosphere::Atmosphere::new(gpu);
        static RENDERERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            id: RENDERERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            target,
            host,
            linear,
            atmosphere,
            scene,
            draw_group,
            cull,
            draw_layout,
            node_layout,
            paint: PaintPipelines::default(),
            symbol_pipelines: [None; 2],
            symbol_passes,
            symbol_pass_buffer: pass_buffer,
            symbol_pass_layout,
            pipelines_stale: false,
            made_seen: 0,
            waiting: 0,
            drape_waiting: false,
            recorded: 0,
            node_uniforms,
            node_stride,
            node_words: vec![0; (node_stride / 4) as usize * capacity.nodes as usize],
            nodes_dirty: None,
            nodes: Vec::new(),
            tiles: Default::default(),
            surface_warps: Default::default(),
            bundles: Vec::new(),
            paints_written: 0,
            images,
            downsampler,
            heatmaps,
            layers: Default::default(),
            composites,
            extrusions,
            terrain: None,
            drape_zooms: BTreeMap::new(),
            terrain_layers: Vec::new(),
            drape_stacks: [0; 2],
            drape_pipelines: None,
            no_dem,
            dem_bound: no_dem,
            no_terrain_depth,
            terrain_depth_bound: no_terrain_depth,
            glyphs_bound: glyphs.revision(),
            buffers_bound: 0,
            cull_layout,
            nearest,
            glyphs,
            images_bound: image_bindings.texture,
            image_bindings,
        }
    }

    /// The image atlas's bindings as they are now: the atlas grows pages,
    /// which the next frame binds.
    pub fn set_images(&mut self, bindings: ImageBindings) {
        self.image_bindings = bindings;
    }

    /// The format of the targets it draws into.
    pub fn format(&self) -> TargetFormat {
        self.host
    }

    /// Whether colours mix in linear light.
    pub fn linear(&self) -> bool {
        self.linear.is_some()
    }

    /// The drape atlas's colour page's format, as layers draw into it.
    fn drape_format(&self) -> TextureFormat {
        if self.linear() {
            DRAPE_FORMAT.srgb()
        } else {
            DRAPE_FORMAT
        }
    }

    /// Identifies the renderer among those of the process: a map holds
    /// the tiles of one renderer, and forgets them for another.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The scene's counts, for the stats HUD and memory accounting.
    pub fn stats(&self) -> SceneStats {
        self.scene.stats()
    }

    /// The scene's GPU memory.
    pub fn memory(&self) -> crate::scene::SceneMemory {
        self.scene.memory()
    }

    /// The terrain's drape atlas, mips included.
    pub fn drape_bytes(&self) -> u64 {
        self.terrain
            .as_ref()
            .map_or(0, terrain::Terrain::atlas_bytes)
    }

    /// Changes when a pool moves into a larger buffer: room may have
    /// changed for tiles that didn't fit.
    pub fn pools_revision(&self) -> u64 {
        self.scene.buffers_revision()
    }

    /// The glyph atlas symbol quads sample.
    pub fn glyphs(&mut self) -> &mut GlyphAtlas {
        &mut self.glyphs
    }

    /// Uses four samples for 3D surfaces by default, or one when the host
    /// explicitly disables antialiasing (e.g. reference image comparisons).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        if self.extrusions.set_antialias(enabled) {
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// The scene's per-node draw arguments and cull counters, for
    /// diagnostics: [`ARGS_WORDS`] words per node, as
    /// `cull.wgsl` writes them, and the number of nodes.
    pub fn draw_args(&self) -> (BufferId, u32) {
        (self.scene.buffers().args, self.nodes.len() as u32)
    }

    /// Replaces the render nodes, in draw order, none of them drawing a
    /// source's image tiles. The scene must hold no meshes.
    pub fn set_nodes(&mut self, gpu: &mut dyn Gpu, families: &[Family]) {
        let nodes: Vec<RenderNode> = families
            .iter()
            .map(|&family| RenderNode {
                family,
                source: None,
            })
            .collect();
        self.remap_nodes(gpu, &nodes, &vec![None; nodes.len()]);
    }

    /// Replaces the render nodes, in draw order, keeping some with their
    /// meshes: new node `i` is old node `retained[i]`, of the same family
    /// and source, or a new node. Old nodes not retained must hold no
    /// vector meshes; their image tiles' quads go with them, and new image
    /// nodes draw the tiles their source has resident. Paint is written
    /// again for every node. Returns whether every resident image tile
    /// found room in the new nodes.
    pub fn remap_nodes(
        &mut self,
        gpu: &mut dyn Gpu,
        nodes: &[RenderNode],
        retained: &[Option<u32>],
    ) -> bool {
        assert_eq!(nodes.len(), retained.len());
        let families: Vec<Family> = nodes.iter().map(|n| n.family).collect();
        let families = &families[..];
        // A style with more nodes than ever drawn: the node tables grow.
        // Every node's group binds its uniform again below.
        let count = families.len() as u32;
        let mut retired_uniforms = None;
        if count > self.scene.capacity().nodes {
            self.scene.grow_nodes(gpu, count);
            let uniforms = gpu.create_buffer(&BufferDesc {
                label: "node uniforms",
                size: self.node_stride * u64::from(self.scene.capacity().nodes),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            retired_uniforms = Some(std::mem::replace(&mut self.node_uniforms, uniforms));
            // The uniforms' CPU copy grows with them.
            let words = (self.node_stride / 4) as usize * self.scene.capacity().nodes as usize;
            self.node_words.resize(words, 0);
        }
        let mut old: Vec<Option<Node>> = self.nodes.drain(..).map(Some).collect();
        let mut map = vec![None; old.len()];
        let mut kept: Vec<Option<Node>> = Vec::with_capacity(families.len());
        for (i, (new, &from)) in nodes.iter().zip(retained).enumerate() {
            let node = from.map(|from| {
                let node = old[from as usize]
                    .take()
                    .expect("each old node is retained once");
                assert_eq!(
                    (node.family, node.source),
                    (new.family, new.source),
                    "a retained node keeps its family and source"
                );
                map[from as usize] = Some(i as u32);
                node
            });
            kept.push(node);
        }
        // Image tiles' quads in nodes leaving go first.
        for (index, _) in map.iter().enumerate().filter(|(_, m)| m.is_none()) {
            let meshes: Vec<MeshId> = self.tiles.meshes_of(index as u32).collect();
            for mesh in meshes {
                self.scene.remove_mesh(mesh);
            }
        }
        self.tiles.remap(&map);
        old.release(gpu);
        let mut added = Vec::new();
        for (i, ((&family, node), desc)) in families.iter().zip(kept).zip(nodes).enumerate() {
            let uniform = self.node_uniform(i as u32);
            if node.is_none() && family.is_image() {
                added.push(i as u32);
            }
            let (mut image, cull) = match node {
                Some(node) => {
                    // Groups bind the uniform at the node's index.
                    (node.group, node.extrusion, node.heatmap).release(gpu);
                    (node.image, node.cull)
                }
                None => (
                    family.is_image().then(image::NodeImage::default),
                    NodeCull {
                        margin_px: 0.0,
                        elevation: 0.0,
                        draw: draw_of(family),
                        visible: true,
                        target: PlacementTarget::Map,
                        labels: family.is_symbol(),
                        surface: false,
                        unclipped: family == Family::FillExtrusion,
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
                source: desc.source,
                group,
                extrusion: (family == Family::FillExtrusion)
                    .then(extrusion::NodeExtrusion::default),
                heatmap: (family == Family::Heatmap)
                    .then(|| heatmap::Heatmaps::node(gpu, self.linear.is_some())),
                cull,
                written: Vec::new(),
                image,
                separate_halo: false,
                along_line: false,
                styled: false,
                elevation: paint::ElevationShape::default(),
                ground: extrusion::Ground::default(),
                occlusion: false,
                ground_drawn: extrusion::Ground::default(),
                stack: None,
                terrain_layer: None,
                shaded: false,
                composite: None,
                drawn: false,
                left_out: Default::default(),
            });
        }
        // Every group that bound the old uniforms is gone.
        retired_uniforms.release(gpu);
        self.assign_stacks();
        // Nodes moved: their drape paint is set again at their indices.
        for (_, level) in std::mem::take(&mut self.drape_zooms) {
            level.release(gpu);
        }
        let cull: Vec<NodeCull> = self.nodes.iter().map(Node::scene_cull).collect();
        self.scene.remap_nodes(&map, &cull);
        self.pipelines_stale = true;
        self.invalidate_bundle(gpu);
        let complete = self.add_image_nodes(gpu, &added);
        let sources: Vec<u32> = self.tiles.sources().map(|(source, _)| source).collect();
        for source in sources {
            self.drop_unused_page(source);
        }
        self.destroy_retired_pages(gpu);
        complete
    }

    /// Puts draped nodes in their drape stacks and terrain layers while
    /// terrain is on: each run of them between nodes that aren't draped is
    /// a terrain layer of one stack, and a layer that composites once
    /// (below opacity 1 or blending other than normally) has a terrain
    /// layer of its own, which composites as the layer's. A shaded
    /// hillshade ends its stack and draws its slopes into one of its own:
    /// the terrain layer lights them over the stack before and under the
    /// next, which draws the layers above it (one hillshade a terrain
    /// layer; another starts the next). Colour stacks come first, in
    /// order, then slope stacks, which the drape keeps on a page of their
    /// own ([`DrapeLayout`]).
    fn assign_stacks(&mut self) {
        let on = self.terrain.as_ref().is_some_and(|t| t.paint.is_some());
        let mut stacks = 0u8;
        let mut layers: Vec<terrain::TerrainLayer> = Vec::new();
        let mut previous: Option<Option<u32>> = None;
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let draped = on && node.family.is_draped();
            let group = composite_group(node.composite);
            let shaded = draped && node.shaded;
            let open = layers
                .last()
                .filter(|l| draped && previous == Some(group) && !(shaded && l.shade.is_some()));
            if draped && open.is_none() {
                layers.push(terrain::TerrainLayer::default());
            }
            previous = draped.then_some(group);
            node.stack = None;
            node.terrain_layer = draped.then(|| layers.len() as u8 - 1);
            if let Some(layer) = layers.last_mut().filter(|_| draped) {
                let mut next = || {
                    stacks = stacks.saturating_add(1);
                    stacks - 1
                };
                node.stack = Some(if shaded {
                    let slopes = next();
                    layer.shade = Some((i as u32, slopes));
                    slopes
                } else {
                    let part = match layer.shade {
                        Some(_) => &mut layer.above,
                        None => &mut layer.below,
                    };
                    *part.get_or_insert_with(next)
                });
            }
        }
        // Colour stacks keep their order ahead of the slope stacks.
        let slopes: Vec<u8> = layers.iter().filter_map(|l| Some(l.shade?.1)).collect();
        let colour = u32::from(stacks) - slopes.len() as u32;
        let renumber = |stack: u8| match slopes.iter().position(|&s| s == stack) {
            Some(rank) => colour as u8 + rank as u8,
            None => stack - slopes.iter().filter(|&&s| s < stack).count() as u8,
        };
        for layer in &mut layers {
            layer.below = layer.below.map(renumber);
            layer.above = layer.above.map(renumber);
            layer.shade = layer.shade.map(|(node, s)| (node, renumber(s)));
        }
        for node in &mut self.nodes {
            node.stack = node.stack.map(renumber);
            node.cull.target = match node.stack {
                Some(stack) => PlacementTarget::Drape(stack),
                None => PlacementTarget::Map,
            };
        }
        self.drape_stacks = [colour, slopes.len() as u32];
        self.terrain_layers = layers;
    }

    /// Makes node `node`, a hillshade, one the terrain pass lights from
    /// the frame's bearing between drape stacks, draping its slopes, or one
    /// it drapes lit: a viewport-anchored hillshade turns with the map,
    /// and no drape slot should.
    pub fn set_shaded_on_terrain(&mut self, gpu: &mut dyn Gpu, node: u32, shaded: bool) {
        let n = &mut self.nodes[node as usize];
        if n.shaded == shaded {
            return;
        }
        n.shaded = shaded;
        if n.terrain_layer.is_some() {
            self.assign_stacks();
            self.sync_cull();
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// Whether node `node` is a hillshade the terrain pass lights now,
    /// draping its slopes.
    pub fn shades_on_terrain(&self, node: u32) -> bool {
        self.nodes[node as usize].shades_on_terrain()
    }

    /// The drape stack node `node` draws into while terrain is on, if it
    /// is draped.
    pub fn drape_stack(&self, node: u32) -> Option<u8> {
        self.nodes[node as usize].stack
    }

    /// Drape stacks of the current nodes while terrain is on: colour
    /// stacks, then slope stacks.
    pub(crate) fn drape_stacks(&self) -> [u32; 2] {
        self.drape_stacks
    }

    /// Turns terrain on over a DEM page, or off. Draped nodes then draw
    /// only drape placements: their stack's ([`PlacementTarget::Drape`]).
    /// Turns terrain on with its DEM page, or flat before it has one, or
    /// off (`None`).
    pub fn set_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        terrain: Option<(Option<PageId>, TerrainPaint)>,
    ) {
        let was = self.terrain.as_ref().is_some_and(|t| t.paint.is_some());
        match terrain {
            Some((page, paint)) => {
                let target = self.target;
                let (dem, dim) = match page {
                    Some(page) => (
                        self.images.dem_texture(page),
                        self.images.page_size(page)[0],
                    ),
                    None => ((self.images.placeholder(), 0), 1),
                };
                let atmosphere = self.atmosphere.uniform;
                let t = self.terrain.get_or_insert_with(|| {
                    let shade_layout = self.images.image_layout();
                    terrain::Terrain::new(gpu, target, atmosphere, shade_layout)
                });
                t.set_paint(gpu, paint, dem, dim);
            }
            None => {
                if let Some(t) = &mut self.terrain {
                    t.disable(gpu);
                }
                for (_, level) in std::mem::take(&mut self.drape_zooms) {
                    level.release(gpu);
                }
            }
        }
        if was != self.terrain.as_ref().is_some_and(|t| t.paint.is_some()) {
            self.assign_stacks();
            self.sync_cull();
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// Whether terrain is on: draped nodes draw drape placements.
    pub fn terrain_on(&self) -> bool {
        self.terrain.as_ref().is_some_and(|t| t.paint.is_some())
    }

    /// How many terrain tiles the drape atlas can hold with this style's
    /// stacks: the most a terrain cover may have.
    pub fn drape_room(&self, gpu: &dyn Gpu) -> u32 {
        terrain::room(gpu, self.drape_stacks())
    }

    /// Makes room for `tile_slots` terrain tiles' drape slots, one per
    /// drape stack, keeping the texels of the tile slots in `kept`. Returns
    /// the layout and the kept slots that moved (from, to). A new layout
    /// generation means every slot is blank. Fails past the
    /// [`drape_room`](Self::drape_room).
    pub fn reserve_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, kept): (u32, &[u32]),
    ) -> Result<(DrapeLayout, Vec<(u32, u32)>), OutOfSpace> {
        let stacks = self.drape_stacks();
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let made = terrain.reserve(gpu, (tile_slots, kept), stacks, &mut self.downsampler)?;
        let layout = terrain.layout();
        if made.is_some() {
            // Stack layers and the main bundle's composites follow it.
            self.invalidate_bundle(gpu);
        }
        Ok((layout, made.unwrap_or_default()))
    }

    /// Redraws these terrain tiles' drape slots this frame, each at its
    /// drape zoom, in the stacks given: their drape placements are in the
    /// scene, and every draped node's paint at that zoom was set
    /// ([`set_drape_paint`](Self::set_drape_paint)).
    pub fn redraw_drape(&mut self, tile_slots: &[(u32, u8, &[u8])]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.redraw(tile_slots);
        }
    }

    /// Writes a draped node's paint for the slots drawn at style zoom
    /// `zoom`, or with `None` leaves it out of them (its layer doesn't
    /// draw at that zoom). Whether anything changed: slots drawn at that
    /// zoom then need drawing again.
    pub fn set_drape_paint(
        &mut self,
        gpu: &mut dyn Gpu,
        node: u32,
        zoom: u8,
        paint: Option<&NodePaint>,
    ) -> bool {
        let words = paint.map(|p| {
            assert!(
                p.fits(self.nodes[node as usize].family),
                "paint family mismatch for node {node}"
            );
            p.uniform_words(node, self.linear.is_some())
        });
        let nodes = self.nodes.len() as u64;
        let stride = self.node_stride;
        let level = self.drape_zooms.entry(zoom).or_insert_with(|| DrapeZoom {
            frame: [(); 2].map(|()| {
                gpu.create_buffer(&BufferDesc {
                    label: "drape frame",
                    size: crate::scene::FRAME_BYTES,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                })
            }),
            uniforms: gpu.create_buffer(&BufferDesc {
                label: "drape node uniforms",
                size: stride * nodes,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
            written: Vec::new(),
            scene: [None; 2],
            bundle: [None, None],
        });
        level.written.resize(self.nodes.len(), None);
        let before = &level.written[node as usize];
        if *before == words {
            return false;
        }
        if before.is_some() != words.is_some() {
            level.forget_bundle(gpu);
        }
        if let Some(words) = &words {
            gpu.write_buffer(
                level.uniforms,
                u64::from(node) * stride,
                bytemuck::cast_slice(words),
            );
            self.paints_written += 1;
        }
        level.written[node as usize] = words;
        true
    }

    /// This frame's terrain tiles.
    pub fn set_terrain_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.set_tiles(gpu, tiles);
        }
    }

    fn node_uniform(&self, node: u32) -> BindingResource {
        BindingResource::Buffer {
            buffer: self.node_uniforms,
            offset: u64::from(node) * self.node_stride,
            size: Some(NODE_UNIFORM_SIZE),
        }
    }

    /// Sets a node's layer composite: its style layer `layer` (nodes of
    /// one layer share it) draws whole and composites once at `opacity`
    /// in `blend` when the opacity is below 1 or the blend isn't normal.
    pub fn set_layer_composite(
        &mut self,
        gpu: &mut dyn Gpu,
        node: u32,
        layer: u32,
        opacity: f32,
        blend: Blend,
    ) {
        let n = &mut self.nodes[node as usize];
        let grouped = |c: Option<(u32, f32, Blend)>| {
            c.filter(|&(_, o, b)| o < 1.0 || b != Blend::Normal)
                .map(|(l, _, b)| (l, b))
        };
        let before = grouped(n.composite);
        n.composite = Some((layer, opacity, blend));
        if grouped(n.composite) != before {
            if n.terrain_layer.is_some() {
                // The layer gains or loses a terrain layer of its own.
                self.assign_stacks();
                self.sync_cull();
            }
            self.invalidate_bundle(gpu);
        } else if before.is_some() {
            self.layers.set_opacity(gpu, node as usize, opacity);
        }
    }

    fn drop_bundles(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.bundles).release(gpu);
    }

    fn invalidate_bundle(&mut self, gpu: &mut dyn Gpu) {
        self.drop_bundles(gpu);
        for node in &self.nodes {
            node.left_out.set(false);
        }
        for zoom in self.drape_zooms.values_mut() {
            zoom.forget_bundle(gpu);
        }
    }

    /// A page of tile textures, `size` texels per tile, for image nodes to
    /// bind.
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

    /// A free layer of `page` for one tile, growing the page if needed.
    /// GPU bytes of the raster and DEM pages and of the glyph pages.
    pub fn texture_bytes(&self) -> (u64, u64) {
        (self.images.bytes(), self.glyphs.bytes())
    }

    /// A layer of `page`, which grows to the device's layer limit.
    pub fn alloc_layer(&mut self, gpu: &mut dyn Gpu, page: PageId) -> Result<u32, OutOfSpace> {
        self.images.alloc_layer(gpu, page)
    }

    /// Tile page layers in use.
    pub fn used_layers(&self) -> u64 {
        self.images.used_layers()
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

    /// How far node `node`'s drawn pixels reach past its chunks' bounds,
    /// from its last paint.
    pub fn cull_margin_px(&self, node: u32) -> f32 {
        self.nodes[node as usize].cull.margin_px
    }

    /// How high node `node`'s content stands, style metres (extrusions'
    /// tallest; 0 for flat content), from its last paint.
    pub fn cull_elevation(&self, node: u32) -> f32 {
        self.nodes[node as usize].cull.elevation
    }

    /// Writes a node's paint values. The paint's family must match the
    /// node's.
    pub fn set_paint(&mut self, gpu: &mut dyn Gpu, node: u32, paint: &NodePaint) {
        let entry = &mut self.nodes[node as usize];
        assert!(
            paint.fits(entry.family),
            "paint family mismatch for node {node}"
        );
        let mut words = paint.uniform_words(node, self.linear.is_some());
        if let NodePaint::Symbol(p) = paint {
            use orb_shaders::layout::symbol_node as n;
            words[n::GLYPH_PAGE] = (GLYPH_PAGE as f32).to_bits();
            words[n::IMAGE_PAGE] = (self.image_bindings.size as f32).to_bits();
            if entry.separate_halo != p.separate_halo || entry.along_line != p.along_line {
                if entry.along_line != p.along_line {
                    self.pipelines_stale = true;
                }
                entry.separate_halo = p.separate_halo;
                entry.along_line = p.along_line;
                std::mem::take(&mut self.bundles).release(gpu);
            }
        }
        let styled = match paint {
            NodePaint::Line(p) => p.dash.is_some() || p.gradient,
            NodePaint::LinePattern(..) => true,
            _ => false,
        };
        if entry.styled != styled {
            entry.styled = styled;
            if entry.drawn {
                self.pipelines_stale = true;
                self.invalidate_bundle(gpu);
            }
        }
        let entry = &mut self.nodes[node as usize];
        if let NodePaint::Extrusion(p) | NodePaint::ExtrusionPattern(p, _) = paint {
            let shape = p.elevation_shape();
            if entry.elevation != shape {
                entry.elevation = shape;
                self.pipelines_stale |= entry.drawn;
                if let Some(e) = &mut entry.extrusion {
                    e.forget_bundle(gpu);
                }
            }
            entry.occlusion = p.ground_occlusion();
        }
        if words != entry.written {
            let (node, stride) = (node as usize, (self.node_stride / 4) as usize);
            self.node_words[node * stride..][..words.len()].copy_from_slice(&words);
            self.nodes_dirty = Some(match self.nodes_dirty.take() {
                Some(r) => r.start.min(node)..r.end.max(node + 1),
                None => node..node + 1,
            });
            entry.written = words;
            self.paints_written += 1;
        }
        // Layers drawn apart apply their opacity as they composite.
        if let (
            NodePaint::Extrusion(ExtrusionPaint { opacity, .. })
            | NodePaint::ExtrusionPattern(ExtrusionPaint { opacity, .. }, _),
            Some(e),
        ) = (paint, &mut entry.extrusion)
        {
            e.set_opacity(gpu, *opacity);
        }
        if let (NodePaint::Heatmap(p), Some(h)) = (paint, &mut entry.heatmap) {
            h.apart.opacity = p.opacity;
        }
        let margin = paint.cull_margin_px();
        let elevation = match paint {
            NodePaint::Extrusion(p) | NodePaint::ExtrusionPattern(p, _) => {
                p.height.max(p.base).max(0.0)
            }
            _ => 0.0,
        };
        let surface = paint.on_surface();
        if margin != entry.cull.margin_px
            || elevation != entry.cull.elevation
            || surface != entry.cull.surface
        {
            entry.cull.margin_px = margin;
            entry.cull.elevation = elevation;
            entry.cull.surface = surface;
            self.scene.set_node(node, entry.scene_cull());
        }
    }

    /// Shows or hides a node without re-recording the bundle or laying the
    /// cull's candidates out again.
    pub fn set_visible(&mut self, node: u32, visible: bool) {
        let entry = &mut self.nodes[node as usize];
        if entry.cull.visible != visible {
            entry.cull.visible = visible;
            self.pipelines_stale |= visible && !entry.drawn;
            self.scene.set_node(node, entry.scene_cull());
        }
    }

    fn sync_cull(&mut self) {
        let cull: Vec<NodeCull> = self.nodes.iter().map(Node::scene_cull).collect();
        self.scene.set_nodes(&cull);
    }

    /// Sets the frame's atmosphere: the style's sky, drawn before the map,
    /// and the fog on terrain. Without a sky the host's backing colour
    /// shows, matching GL JS's transparent default, and terrain has no fog.
    pub fn set_atmosphere(
        &mut self,
        gpu: &mut dyn Gpu,
        paint: Option<AtmospherePaint>,
        view: AtmosphereView,
    ) {
        let frame = self.scene.buffers().frame;
        self.atmosphere.set(gpu, self.target, frame, paint, view);
    }

    /// Whether a layer can composite at a layer opacity, asking for the
    /// pipeline it takes: a frame that would wait for it draws nothing,
    /// so what fades through it starts once it compiled.
    pub fn layer_composite_ready(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.composites.make(gpu, offscreen::Mode::Over);
        self.composites.ready(gpu, offscreen::Mode::Over)
    }

    /// Asks for the pipelines the nodes shown need and counts those
    /// compiling: a frame is complete only once none are.
    pub fn pipelines(&mut self, gpu: &mut dyn Gpu) -> PipelineCounts {
        if std::mem::take(&mut self.pipelines_stale) {
            self.make_pipelines(gpu);
        }
        gpu.poll_pipelines()
    }

    /// Makes the pipelines the nodes shown (their layers draw at this
    /// zoom) or that have drawn need and that aren't made yet, in draw
    /// order: the device compiles them in the order they are asked for
    /// (the cull's was first), and [`prioritize`](Self::prioritize) moves
    /// what the view shows ahead.
    /// A still compiles what its zoom shows and no more, while the
    /// compiles still overlap the tiles' loading. A node shown for the
    /// first time records the bundles again, which left it out.
    fn make_pipelines(&mut self, gpu: &mut dyn Gpu) {
        let layouts = [self.draw_layout, self.node_layout];
        let drape_target = TargetFormat {
            color: self.drape_format(),
            sample_count: 1,
        };
        let own_drape = self.target.color != drape_target.color;
        let (mut started, mut draped) = (false, false);
        for i in 0..self.nodes.len() {
            // A node that drew keeps its pipelines made for what it draws
            // into (the 3D sample count, the terrain's drape) while it
            // hides: the bundles record it, and it may show again.
            if !self.nodes[i].cull.visible && !self.nodes[i].drawn {
                continue;
            }
            let node = &mut self.nodes[i];
            if !std::mem::replace(&mut node.drawn, true) {
                started = true;
                draped |= node.stack.is_some();
            }
            let (family, kind) = (node.family, node.paint_kind());
            if node.shades_on_terrain() {
                // Its slopes into the drape atlas, which the terrain pass
                // lights.
                self.images.make_slopes(gpu);
                continue;
            }
            if node.stack.is_some() {
                // Into the drape atlas, with its format's pipelines if
                // the target's differs.
                if !own_drape {
                    self.make_node_pipeline(gpu, i);
                    continue;
                }
                let drape = self
                    .drape_pipelines
                    .get_or_insert_with(DrapePipelines::default);
                if family.is_image() {
                    self.images.make_for(
                        gpu,
                        &mut drape.images,
                        drape_target,
                        self.draw_layout,
                        family,
                    );
                    if family == Family::Raster {
                        self.downsampler.make(gpu);
                    }
                } else {
                    drape.paint.make(gpu, drape_target, &layouts, kind);
                }
                continue;
            }
            self.make_node_pipeline(gpu, i);
        }
        if started {
            self.invalidate_bundle(gpu);
        }
        if draped && let Some(terrain) = self.terrain.as_mut() {
            // Slots drawn without them lack those layers.
            terrain.redraw_all();
        }
    }

    fn make_node_pipeline(&mut self, gpu: &mut dyn Gpu, i: usize) {
        let layouts = [self.draw_layout, self.node_layout];
        let family = self.nodes[i].family;
        match family {
            Family::Fill | Family::FillOutline | Family::Line | Family::Circle => {
                let kind = self.nodes[i].paint_kind();
                self.paint.make(gpu, self.target, &layouts, kind);
            }
            Family::SymbolIcon | Family::SymbolText => {
                let layouts = [self.draw_layout, self.node_layout, self.symbol_pass_layout];
                let line = usize::from(self.nodes[i].along_line);
                if self.symbol_pipelines[line].is_none() {
                    let blend = Some(BlendState::PremultipliedAlpha);
                    let draw = Draw::new(&SYMBOL_SHADERS[line], &layouts, self.target, blend);
                    self.symbol_pipelines[line] = Some(draw.make(gpu));
                }
            }
            Family::Raster | Family::Hillshade | Family::ColorRelief => {
                self.images.make(gpu, family);
                if family == Family::Raster {
                    // Raster pages are mipmapped.
                    self.downsampler.make(gpu);
                }
            }
            Family::Heatmap => self.heatmaps.make(gpu),
            Family::FillExtrusion => {
                self.extrusions.make(gpu, self.nodes[i].elevation);
                self.extrusions.make_ground(gpu, self.nodes[i].ground);
            }
        }
    }

    /// The pipelines node `i` draws with, made or not.
    fn node_pipelines(&self, i: usize, out: &mut Vec<PipelineId>) {
        let node = &self.nodes[i];
        if node.terrain_layer.is_some()
            && let Some(terrain) = &self.terrain
        {
            out.extend(terrain.pipelines());
        }
        if node.shades_on_terrain() {
            out.extend(self.images.slopes_pipeline().map(PipelineId::from));
        } else if node.stack.is_some()
            && let Some(d) = self.drape_pipelines.as_ref()
        {
            out.extend(
                d.paint
                    .get(node.paint_kind())
                    .or_else(|| {
                        node.family
                            .is_image()
                            .then(|| d.images.get(node.family))
                            .flatten()
                    })
                    .map(PipelineId::from),
            );
        }
        if node.terrain_layer.is_some() || node.heatmap.is_some() || node.extrusion.is_some() {
            out.extend(self.composites.pipelines());
        }
        match node.family {
            Family::Heatmap => out.extend(self.heatmaps.pipelines()),
            Family::FillExtrusion => {
                out.extend(self.extrusions.pipelines(node.elevation));
                out.extend(self.extrusions.ground_pipelines(node.ground));
            }
            Family::SymbolIcon | Family::SymbolText => {
                out.extend(
                    self.symbol_pipelines[usize::from(node.along_line)].map(PipelineId::from),
                );
            }
            family if family.is_image() => {
                out.extend(self.images.pipeline(family).map(PipelineId::from));
                if family == Family::Raster {
                    out.extend(self.downsampler.pipelines());
                } else {
                    out.extend(self.images.prepare_pipelines());
                }
            }
            _ => out.extend(self.paint.get(node.paint_kind()).map(PipelineId::from)),
        }
    }

    /// Moves the pipelines of nodes with meshes in the scene ahead of the
    /// others still compiling: what the current view needs first. Nodes
    /// earlier in draw order go first (the bottom layers cover most of
    /// the view), the cull before everything.
    fn prioritize(&self, gpu: &mut dyn Gpu) {
        let mut wanted = Vec::new();
        wanted.push(self.cull.0.into());
        wanted.extend(self.linear.as_ref().map(offscreen::LinearScene::pipeline));
        for i in 0..self.nodes.len() {
            if self.scene.node_has_meshes(i as u32) {
                self.node_pipelines(i, &mut wanted);
            }
        }
        if self.atmosphere.enabled {
            wanted.extend(self.atmosphere.pipelines());
        }
        // Each moves to the front: the last asked for compiles first.
        let mut seen = std::collections::HashSet::new();
        wanted.retain(|&id| seen.insert(id) && !gpu.pipeline_ready(id));
        for &id in wanted.iter().rev() {
            gpu.prioritize_pipeline(id);
        }
    }

    /// Draws a frame into `target`, cleared to `clear` (premultiplied RGBA),
    /// timing its stages on `clock` (milliseconds).
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
        clock: &dyn Fn() -> f64,
    ) -> FrameStats {
        self.destroy_retired_pages(gpu);
        self.bind_scene(gpu);
        if let Some(nodes) = self.nodes_dirty.take() {
            let stride = (self.node_stride / 4) as usize;
            gpu.write_buffer(
                self.node_uniforms,
                nodes.start as u64 * self.node_stride,
                bytemuck::cast_slice(&self.node_words[nodes.start * stride..nodes.end * stride]),
            );
        }
        let generation = self.terrain.as_ref().map(|t| t.layout().generation);
        let mut stats = self.record(gpu, frame, target, clear, clock);
        // Drape slots blanked during the frame (pipelines a slot waited for
        // arrived, or terrain couldn't draw yet) draw again from the next:
        // this one shows them without what they lacked.
        stats.waited |= self.terrain.as_ref().map(|t| t.layout().generation) != generation;
        stats.draws += u32::from(self.atmosphere.enabled);
        self.end_surface_warps();
        stats
    }

    /// Binds what the scene groups bind anew when it changed: the glyph
    /// pages grew into new textures, the terrain changed, or a scene pool
    /// moved into a larger buffer.
    fn bind_scene(&mut self, gpu: &mut dyn Gpu) {
        let dem = self
            .terrain
            .as_ref()
            .and_then(terrain::Terrain::dem_texture)
            .unwrap_or(self.no_dem);
        // Symbols and circles read the terrain's depth.
        let terrain_depth = self
            .terrain
            .as_ref()
            .and_then(terrain::Terrain::occlusion_texture)
            .unwrap_or(self.no_terrain_depth);
        let buffers = self.scene.buffers_revision();
        if self.glyphs.revision() != self.glyphs_bound
            || dem != self.dem_bound
            || terrain_depth != self.terrain_depth_bound
            || buffers != self.buffers_bound
            || self.image_bindings.texture != self.images_bound
        {
            self.images_bound = self.image_bindings.texture;
            if buffers != self.buffers_bound {
                self.buffers_bound = buffers;
                gpu.destroy_bind_group(self.cull.1);
                self.cull.1 = cull_group(gpu, self.cull_layout, self.scene.buffers());
            }
            gpu.destroy_bind_group(self.draw_group);
            let b = *self.scene.buffers();
            self.draw_group = scene_group(
                gpu,
                self.draw_layout,
                &b,
                b.frame,
                &self.image_bindings,
                (self.glyphs.texture(), self.nearest, dem, terrain_depth),
            );
            self.glyphs_bound = self.glyphs.revision();
            self.dem_bound = dem;
            self.terrain_depth_bound = terrain_depth;
            for zoom in self.drape_zooms.values_mut() {
                for scene in &mut zoom.scene {
                    scene.take().release(gpu);
                }
            }
            self.invalidate_bundle(gpu);
            // Extrusion and heatmap nodes record the scene group too.
            for node in &mut self.nodes {
                if let Some(extrusion) = &mut node.extrusion {
                    extrusion.forget_bundle(gpu);
                }
                if let Some(heatmap) = &mut node.heatmap {
                    heatmap.apart.forget_bundle(gpu);
                }
            }
        }
    }

    fn record(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
        clock: &dyn Fn() -> f64,
    ) -> FrameStats {
        let mut times = RenderTimes::default();
        let started = clock();
        let (candidates, candidates_rebuilt) =
            self.scene.prepare(gpu, frame, self.linear.is_some());
        times.cull = micros(started, clock());
        // A node the bundles left out for want of candidates draws once
        // it has some.
        if candidates_rebuilt
            && (self.nodes.iter().enumerate())
                .any(|(i, n)| n.left_out.get() && self.scene.node_candidates(i as u32) > 0)
        {
            self.invalidate_bundle(gpu);
        }
        // A heatmap hidden, or with nothing to draw, composites nothing.
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if let Some(heatmap) = &mut node.heatmap
                && let Some(offscreen) = &mut heatmap.apart.offscreen
            {
                let shown = node.cull.visible && self.scene.node_candidates(i as u32) > 0;
                let opacity = if shown { heatmap.apart.opacity } else { 0.0 };
                offscreen.set_opacity(gpu, opacity);
            }
        }
        self.refresh_image_groups(gpu);
        // What extrusion nodes darken on the ground: their pipelines are
        // made with the rest.
        let shadows = frame.light.shadow().1 > 0.0;
        for node in self.nodes.iter_mut().filter(|n| n.extrusion.is_some()) {
            let ground = extrusion::Ground {
                shadows,
                occlusion: node.occlusion,
            };
            if node.ground != ground {
                node.ground = ground;
                self.pipelines_stale |= node.drawn;
            }
        }
        let compiling = self.pipelines(gpu);
        let made = compiling.made;
        if compiling.made != self.made_seen {
            // Pipelines arrived: record again what waited for them.
            self.made_seen = compiling.made;
            let drape_waiting = std::mem::take(&mut self.drape_waiting);
            if self.waiting > 0 || drape_waiting {
                self.invalidate_bundle(gpu);
            }
            if drape_waiting && let Some(terrain) = self.terrain.as_mut() {
                // Slots drawn without them lack those layers, at whichever
                // zoom: all draw again, each zoom's bundle recorded anew.
                terrain.redraw_all();
            }
        }
        if compiling.pending > 0 {
            self.prioritize(gpu);
        }
        let terrain_on = self.terrain.as_ref().is_some_and(terrain::Terrain::active);
        let terrain_layers = if terrain_on {
            self.terrain_layers.len()
        } else {
            0
        };
        let density_size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio / 4.0).floor().max(1.0) as u32);
        let size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio).max(1.0) as u32);
        let logical = frame.viewport.map(|v| v.max(1.0) as u32);
        // Only nodes that have drawn have targets.
        let mut resized = self.extrusions.resize(
            gpu,
            size,
            self.nodes
                .iter()
                .filter(|n| n.drawn && n.extrusion.is_some())
                .count()
                + terrain_layers,
        );
        if let Some(terrain) = self.terrain.as_mut().filter(|_| terrain_on) {
            terrain.set_layers(gpu, &self.terrain_layers, self.drape_stacks[0]);
            let sizes = (size, logical);
            resized |= terrain.resize(gpu, &mut self.composites, sizes, self.extrusions.samples);
        }
        for node in self.nodes.iter_mut().filter(|n| n.drawn) {
            if let Some(extrusion) = &mut node.extrusion {
                resized |= self
                    .extrusions
                    .resize_node(gpu, &mut self.composites, extrusion, size);
            }
            if let Some(heatmap) = &mut node.heatmap {
                resized |= self
                    .heatmaps
                    .resize(gpu, &mut self.composites, heatmap, density_size);
            }
        }
        if resized {
            self.invalidate_bundle(gpu);
        }
        // Bound once the targets are sized: a resize remade the terrain's
        // depth the scene group samples, and preparing may have grown the
        // compacted index buffer. Bound before, the frame's bundles would
        // sample the destroyed depth and the submit fail.
        self.bind_scene(gpu);
        let bundle_recorded = self.bundles.is_empty();
        let recording = clock();
        if bundle_recorded {
            let indexed = self.nodes.iter().any(|n| n.cull.draw == NodeDraw::Indexed);
            let index_buffer = RenderCommand::SetIndexBuffer {
                buffer: self.scene.buffers().compacted,
                offset: 0,
            };
            let mut ready = self.ready(gpu, terrain_on);
            let mut commands = Vec::with_capacity(self.nodes.len() * 4 + 1);
            if indexed {
                commands.push(index_buffer.clone());
            }
            // Runs of one layer's nodes below layer opacity 1 or blending
            // other than normally draw into a texture of their own; draped
            // nodes take the terrain's way.
            // On terrain, a draped layer's group is its terrain layer: the
            // texture it composites.
            let mut runs: Vec<layer_composite::Run> = Vec::new();
            for (i, node) in self.nodes.iter().enumerate() {
                let Some(layer) = composite_group(node.composite) else {
                    continue;
                };
                let (_, opacity, blend) = node.composite.expect("grouped");
                let terrain_layer = node.terrain_layer.filter(|_| terrain_on);
                match runs.last_mut() {
                    Some(run)
                        if run.nodes.end == i
                            && composite_group(self.nodes[run.nodes.start].composite)
                                == Some(layer) =>
                    {
                        run.nodes.end = i + 1;
                    }
                    _ => runs.push(layer_composite::Run {
                        nodes: i..i + 1,
                        opacity,
                        blend,
                        source: terrain_layer.and_then(|l| self.terrain.as_ref()?.layer_texture(l)),
                    }),
                }
            }
            self.layers
                .regroup(gpu, &mut self.composites, runs, (size, self.target));
            // Until the composites compile, nothing draws: a split main
            // pass needs its blend draw.
            if !self.layers.ready(gpu, &self.composites) {
                self.layers.clear(gpu);
                ready.cull = false;
            }
            (self.waiting, self.recorded) = (0, 0);
            for g in 0..self.layers.groups.len() {
                if self.layers.groups[g].drawn_elsewhere() {
                    continue;
                }
                let mut group = vec![RenderCommand::SetIndexBuffer {
                    buffer: self.scene.buffers().compacted,
                    offset: 0,
                }];
                for i in self.layers.groups[g].nodes.clone() {
                    let groups = (self.draw_group, self.nodes[i].group);
                    let before = group.len();
                    if !self.node_commands(gpu, &ready, i, groups, &mut group) {
                        self.waiting += 1;
                    }
                    self.recorded += u32::from(group.len() > before);
                }
                self.layers.groups[g].bundle = Some(gpu.create_render_bundle(
                    &RenderBundleDesc {
                        label: "layer composite",
                        color_formats: &[self.target.color],
                        depth_format: None,
                        sample_count: self.target.sample_count,
                    },
                    &group,
                ));
            }
            let layers = RenderBundleDesc {
                label: "layers",
                color_formats: &[self.target.color],
                depth_format: None,
                sample_count: self.target.sample_count,
            };
            for i in 0..self.nodes.len() {
                let (start, member) = self.layers.at(i);
                match start {
                    // The main pass splits here: its blend draw opens the
                    // next pass, before the next bundle.
                    Some(group) if group.blend != Blend::Normal => {
                        let segment = std::mem::take(&mut commands);
                        self.bundles
                            .push(gpu.create_render_bundle(&layers, &segment));
                        if indexed {
                            commands.push(index_buffer.clone());
                        }
                    }
                    Some(group) => {
                        commands.extend(self.layers.composite_commands(&self.composites, group));
                        // The composite changed the pipeline and bindings.
                        if indexed {
                            commands.push(index_buffer.clone());
                        }
                    }
                    None => {}
                }
                let groups = (self.draw_group, self.nodes[i].group);
                let before = commands.len();
                if !member && !self.node_commands(gpu, &ready, i, groups, &mut commands) {
                    self.waiting += 1;
                }
                self.recorded += u32::from(commands.len() > before);
            }
            self.bundles
                .push(gpu.create_render_bundle(&layers, &commands));
        }
        let draping = clock();
        times.bundles = micros(recording, draping);
        let draws = match self.terrain.as_ref().filter(|_| terrain_on) {
            Some(terrain) => terrain.pending_draws().to_vec(),
            None => Vec::new(),
        };
        let drape: Vec<terrain::DrapeBundle> = draws
            .into_iter()
            .filter_map(|at| Some((at.0, at.1, self.drape_bundle(gpu, frame, at)?)))
            .collect();
        let encoding = clock();
        let mut drape_passes = 0;

        let mut passes = Vec::with_capacity(3);
        let mut stats = FrameStats {
            draws: self.recorded,
            bundle_recorded,
            candidates,
            candidates_rebuilt,
            placements_written: self.scene.take_placements_written(),
            paints_written: std::mem::take(&mut self.paints_written),
            pipelines_pending: compiling.pending,
            nodes_waiting: self.waiting,
            ..FrameStats::default()
        };
        if let Some((prepare, prepared)) = self.images.prepare_pass(gpu) {
            let start = passes.len();
            stats.draws += prepared.draws;
            stats.prepared = prepared.tiles;
            passes.extend(prepare);
            Pass::group(&mut passes, start, "dem prepare");
        }
        if let Some((mips, dispatches)) = self.images.mip_pass(gpu, &self.downsampler) {
            let start = passes.len();
            stats.dispatches += dispatches;
            passes.extend(mips);
            Pass::group(&mut passes, start, "image mips");
        }
        // The dispatch writes every node's arguments, zeros included, so it
        // runs even when nothing is resident. Indexed nodes' batches add to
        // cleared arguments.
        let (pipeline, group) = self.cull;
        let cull_ready = gpu.pipeline_ready(pipeline.into());
        if !self.nodes.is_empty() && cull_ready {
            let start = passes.len();
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
                label: "cull",
                commands: vec![
                    ComputeCommand::SetPipeline(pipeline),
                    ComputeCommand::SetBindGroup { index: 0, group },
                    ComputeCommand::Dispatch { x, y, z: 1 },
                ],
            }));
            Pass::group(&mut passes, start, "cull");
        }
        let start = passes.len();
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(heatmap) = &mut node.heatmap
                && node.drawn
                && cull_ready
                && self.heatmaps.ready(gpu, &self.composites)
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
        Pass::group(&mut passes, start, "heatmaps");
        if let Some(terrain) = self.terrain.as_mut().filter(|_| terrain_on) {
            let passing = clock();
            let start = passes.len();
            let depth = self.extrusions.depth().expect("sized above");
            let b = self.scene.buffers();
            // Shaded hillshades light their slopes' slots: blank until
            // they draw there.
            let shades: Vec<Option<BindGroupId>> = (self.terrain_layers.iter())
                .map(|l| {
                    let node = &self.nodes[l.shade?.0 as usize];
                    node.cull.visible.then_some(node.group)
                })
                .collect();
            stats.draws += terrain.passes(
                gpu,
                (b.frame, b.props),
                (&drape, &shades),
                depth,
                &self.downsampler,
                &mut passes,
            );
            Pass::group(&mut passes, start, "terrain");
            drape_passes = micros(passing, clock());
        }
        let mut first = !terrain_on;
        let extrusions_ready = cull_ready && self.extrusions.ready(gpu, &self.composites);
        let (draw_group, buffers) = (self.draw_group, *self.scene.buffers());
        let start = passes.len();
        for (index, node) in self.nodes.iter_mut().enumerate() {
            let Some(target) = node
                .extrusion
                .as_mut()
                .filter(|_| extrusions_ready && node.drawn)
            else {
                continue;
            };
            // Ground passes join the recorded draw as they compile.
            let ground = self.extrusions.ground_ready(gpu, node.ground);
            if ground != node.ground_drawn {
                node.ground_drawn = ground;
                target.forget_bundle(gpu);
            }
            let pass = self.extrusions.prepare(
                gpu,
                target,
                (draw_group, node.group),
                &buffers,
                (index as u32, node.elevation, ground),
                first,
            );
            passes.push(Pass::Render(pass));
            first = false;
            stats.draws += 1;
        }
        Pass::group(&mut passes, start, "3d layers");
        let start = passes.len();
        passes.extend(self.layers.passes());
        Pass::group(&mut passes, start, "layers apart");
        // In linear light the map draws into its own texture, cleared to the
        // colour decoded, then encodes into the host's target.
        let (into, into_clear) = match &mut self.linear {
            Some(linear) => (
                linear.texture(gpu, &self.composites, size),
                color_in(clear.map(|c| c as f32), true).map(f64::from),
            ),
            None => (target, clear),
        };
        let start = passes.len();
        passes.extend(self.layers.main_passes(
            &self.composites,
            (
                self.atmosphere.commands(gpu),
                &self.bundles,
                self.atmosphere.commands_after(gpu),
            ),
            into,
            into_clear,
        ));
        if let Some(linear) = &self.linear {
            passes.push(linear.pass(gpu, target, clear));
        }
        Pass::group(&mut passes, start, "main");
        let submitting = clock();
        gpu.submit(&passes);
        times.submit = micros(submitting, clock());
        times.drape = micros(draping, encoding) + drape_passes;
        times.encode = micros(encoding, submitting).saturating_sub(drape_passes);
        stats.times = times;
        // Every check passed only if nothing was compiling or compiled
        // during the frame.
        let end = gpu.poll_pipelines();
        stats.pipelines_pending = end.pending;
        stats.waited =
            compiling.pending > 0 || end.pending > 0 || end.made != made || self.waiting > 0;
        stats
    }

    /// What a bundle record finds compiled besides the nodes' own
    /// pipelines.
    fn ready(&mut self, gpu: &mut dyn Gpu, terrain_on: bool) -> Ready {
        Ready {
            cull: gpu.pipeline_ready(self.cull.0.into()),
            terrain: self
                .terrain
                .as_mut()
                .filter(|_| terrain_on)
                .is_some_and(|t| t.ready(gpu))
                && self.composites.ready(gpu, offscreen::Mode::Over),
            prepare: self.images.prepare_ready(gpu),
            mips: self.downsampler.ready(gpu),
        }
    }

    /// The drape pass of style zoom `zoom` into page `page` for this
    /// frame: its frame written, and its bundle of the draped nodes drawn
    /// at it into that page's stacks, recorded if it isn't. `None` when no
    /// node's paint was set at it.
    fn drape_bundle(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        (zoom, page): (u8, usize),
    ) -> Option<RenderBundleId> {
        let terrain = self.terrain.as_ref()?;
        let drape_frame = terrain.drape_frame(frame, (zoom, page));
        let colour = self.drape_stacks[0];
        let level = self.drape_zooms.get_mut(&zoom)?;
        crate::scene::write_frame(gpu, level.frame[page], &drape_frame, self.linear.is_some());
        if level.scene[page].is_none() {
            let b = *self.scene.buffers();
            level.scene[page] = Some(scene_group(
                gpu,
                self.draw_layout,
                &b,
                level.frame[page],
                &self.image_bindings,
                // Draped content isn't hidden by the terrain it's on.
                (
                    self.glyphs.texture(),
                    self.nearest,
                    self.dem_bound,
                    self.no_terrain_depth,
                ),
            ));
            level.bundle[page].take().release(gpu);
        }
        if let Some((bundle, _)) = &level.bundle[page] {
            return Some(*bundle);
        }
        let ready = self.ready(gpu, true);
        let level = &self.drape_zooms[&zoom];
        let (scene, uniforms) = (level.scene[page].expect("made above"), level.uniforms);
        let mut commands = vec![RenderCommand::SetIndexBuffer {
            buffer: self.scene.buffers().compacted,
            offset: 0,
        }];
        let mut groups = Vec::new();
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let on_page = node
                .stack
                .is_some_and(|s| usize::from(u32::from(s) >= colour) == page);
            if !on_page || !level.written.get(i).is_some_and(Option::is_some) {
                continue;
            }
            let uniform = BindingResource::Buffer {
                buffer: uniforms,
                offset: i as u64 * self.node_stride,
                size: Some(NODE_UNIFORM_SIZE),
            };
            let group = match &node.image {
                Some(image) => self.images.group(gpu, node.family, image, uniform),
                None => gpu.create_bind_group(&BindGroupDesc {
                    label: "drape node",
                    layout: self.node_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform,
                    }],
                }),
            };
            groups.push(group);
            if !self.node_commands(gpu, &ready, i, (scene, group), &mut commands) {
                self.drape_waiting = true;
            }
        }
        // The slope page stays as it is in linear light.
        let format = if page == 0 {
            self.drape_format()
        } else {
            DRAPE_FORMAT
        };
        let bundle = gpu.create_render_bundle(
            &RenderBundleDesc {
                label: "drape",
                color_formats: &[format],
                depth_format: None,
                sample_count: 1,
            },
            &commands,
        );
        let level = self.drape_zooms.get_mut(&zoom).expect("found above");
        level.bundle[page] = Some((bundle, groups));
        Some(bundle)
    }

    /// Node `i`'s commands as a bundle records them, over the scene group
    /// `scene`: the map's, or the drape's for draped nodes on terrain. In
    /// the map, a terrain layer composites in its first node's place. `false` when the node waits for a pipeline and records
    /// nothing.
    fn node_commands(
        &self,
        gpu: &mut dyn Gpu,
        ready: &Ready,
        i: usize,
        (scene, group): (BindGroupId, BindGroupId),
        commands: &mut Vec<RenderCommand>,
    ) -> bool {
        if !ready.cull {
            return false;
        }
        let node = &self.nodes[i];
        let args = args_offset(i as u32);
        if let (Some(layer), true) = (node.terrain_layer, scene == self.draw_group) {
            let first = i == 0 || self.nodes[i - 1].terrain_layer != Some(layer);
            if !ready.terrain {
                return false;
            }
            if let Some(composite) = self
                .terrain
                .as_ref()
                .filter(|_| first)
                .and_then(|t| t.layer_composite(layer))
            {
                commands.extend(self.composites.commands(composite));
            }
            return true;
        }
        if !node.drawn {
            // Nothing to draw yet: drawing records the bundles again.
            return true;
        }
        if let Some(extrusion) = &node.extrusion {
            if !self.extrusions.ready(gpu, &self.composites) {
                return false;
            }
            if let Some(o) = &extrusion.offscreen {
                let composite = &o.composite;
                commands.extend(self.composites.commands(composite));
            }
            return true;
        }
        if let Some(heatmap) = &node.heatmap {
            if !self.heatmaps.ready(gpu, &self.composites) {
                return false;
            }
            if let Some(o) = &heatmap.apart.offscreen {
                let composite = &o.composite;
                commands.extend(self.composites.commands(composite));
            }
            return true;
        }
        // Without candidates its draw draws nothing: on sparse views most
        // of a style's nodes, each a pipeline, bindings and an indirect
        // draw the CPU encodes and the GPU reads every frame.
        if self.scene.node_candidates(i as u32) == 0 {
            node.left_out.set(true);
            return true;
        }
        if node.family.is_symbol() {
            let Some(pipeline) = self.symbol_pipelines[usize::from(node.along_line)]
                .filter(|&p| gpu.pipeline_ready(p.into()))
            else {
                return false;
            };
            let passes = if node.separate_halo {
                &self.symbol_passes[1..]
            } else {
                &self.symbol_passes[..1]
            };
            commands.extend([
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetBindGroup { index: 1, group },
            ]);
            for &group in passes {
                commands.extend([
                    RenderCommand::SetBindGroup { index: 2, group },
                    RenderCommand::DrawIndirect {
                        buffer: self.scene.buffers().args,
                        offset: args,
                    },
                ]);
            }
            return true;
        }
        if matches!(node.family, Family::Hillshade | Family::ColorRelief) && !ready.prepare {
            return false;
        }
        if node.family == Family::Raster && !ready.mips {
            return false;
        }
        // Into the drape atlas, with its format's pipelines if the
        // target's differs.
        let drape = self
            .drape_pipelines
            .as_ref()
            .filter(|_| scene != self.draw_group && self.target.color != self.drape_format());
        let pipeline = match (drape, node.family.is_image()) {
            _ if node.shades_on_terrain() => self.images.slopes_pipeline(),
            (Some(d), true) => d.images.get(node.family),
            (Some(d), false) => d.paint.get(node.paint_kind()),
            (None, true) => self.images.pipeline(node.family),
            (None, false) => self.paint.get(node.paint_kind()),
        };
        let Some(pipeline) = pipeline.filter(|&p| gpu.pipeline_ready(p.into())) else {
            return false;
        };
        commands.extend([
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::SetBindGroup { index: 1, group },
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
        true
    }
}

/// The style layer a node composites once with, when it does: below layer
/// opacity 1 or blending other than normally.
fn composite_group(composite: Option<(u32, f32, Blend)>) -> Option<u32> {
    composite
        .filter(|&(_, o, b)| o < 1.0 || b != Blend::Normal)
        .map(|(l, _, _)| l)
}

/// What one bundle record found compiled besides the nodes' own
/// pipelines.
#[derive(Clone, Copy)]
struct Ready {
    cull: bool,
    terrain: bool,
    /// The DEM prepare's, which hillshade and color relief wait for.
    prepare: bool,
    /// The raster mip levels', which raster nodes wait for.
    mips: bool,
}

/// Draped families' pipelines for the drape atlas's format, each made for
/// its family's first draped node.
#[derive(Debug, Default)]
struct DrapePipelines {
    paint: PaintPipelines,
    images: image::ImagePipelines,
}

/// The drape pass of one style zoom: a slot draws its
/// terrain tile's content at that zoom whatever the camera's, so draped
/// nodes' paint at it is written once, not per frame. Its bundle replays
/// the draped nodes over its frame and their paint at that zoom; their
/// drape placements of other zooms draw nothing in it (`drawn_in_pass`).
#[derive(Debug)]
struct DrapeZoom {
    /// The drape frame at this zoom, per drape page.
    frame: [BufferId; 2],
    /// Node uniforms at this zoom, at each node's offset.
    uniforms: BufferId,
    /// Per node, the words written, or `None` where it doesn't draw at
    /// this zoom (outside its layer's zooms, or not draped).
    written: Vec<Option<Vec<u32>>>,
    /// Per page, the scene group over its `frame`, and the bundle of the
    /// nodes drawing into its stacks with the node groups it binds.
    scene: [Option<BindGroupId>; 2],
    bundle: [Option<(RenderBundleId, Vec<BindGroupId>)>; 2],
}

impl release::Release for Node {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.group, self.image, self.heatmap, self.extrusion).release(gpu);
    }
}

impl release::Release for DrapeZoom {
    fn release(self, gpu: &mut dyn Gpu) {
        DrapeZoom::release(self, gpu);
    }
}

impl DrapeZoom {
    fn forget_bundle(&mut self, gpu: &mut dyn Gpu) {
        for bundle in &mut self.bundle {
            bundle.take().release(gpu);
        }
    }

    fn release(mut self, gpu: &mut dyn Gpu) {
        self.forget_bundle(gpu);
        let [a, b] = self.scene;
        let [f, g] = self.frame;
        ((a, b), (f, g), self.uniforms).release(gpu);
    }
}

/// The vector families' pipelines (fill, fill outline, line, circle), each
/// made for the family's first node.
#[derive(Clone, Copy, Debug, Default)]
struct PaintPipelines {
    pipelines: [Option<RenderPipelineId>; 5],
}

impl PaintPipelines {
    /// The slot of a node's [`Node::paint_kind`].
    fn index((family, styled): (Family, bool)) -> Option<usize> {
        Some(match family {
            Family::Fill => 0,
            Family::FillOutline => 1,
            Family::Line if styled => 4,
            Family::Line => 2,
            Family::Circle => 3,
            _ => return None,
        })
    }

    fn get(&self, kind: (Family, bool)) -> Option<RenderPipelineId> {
        self.pipelines[Self::index(kind)?]
    }

    fn make(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        layouts: &[BindGroupLayoutId],
        kind: (Family, bool),
    ) {
        let Some(index) = Self::index(kind) else {
            return;
        };
        let (shader, topology) = match index {
            0 => (&FILL_SHADER, PrimitiveTopology::TriangleList),
            1 => (&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            2 => (&LINE_SHADER, PrimitiveTopology::TriangleList),
            4 => (&LINE_STYLED_SHADER, PrimitiveTopology::TriangleList),
            _ => (&CIRCLE_SHADER, PrimitiveTopology::TriangleList),
        };
        self.pipelines[index].get_or_insert_with(|| {
            Draw {
                topology,
                ..Draw::new(
                    shader,
                    layouts,
                    target,
                    Some(BlendState::PremultipliedAlpha),
                )
            }
            .make(gpu)
        });
    }
}

/// The cull pass's group: the scene's tables and index buffers.
fn cull_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    b: &crate::scene::SceneBuffers,
) -> BindGroupId {
    let entries: Vec<BindGroupEntry> = [
        b.frame,
        b.tiles,
        b.chunks,
        b.candidates,
        b.node_cull,
        b.visible,
        b.args,
        b.indices,
        b.compacted,
    ]
    .iter()
    .enumerate()
    .map(|(i, &buffer)| BindGroupEntry::buffer(i as u32, buffer))
    .collect();
    gpu.create_bind_group(&BindGroupDesc {
        label: "cull",
        layout,
        entries: &entries,
    })
}

/// The scene's group: its buffers, the image pages and the glyph pages.
fn scene_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    b: &crate::scene::SceneBuffers,
    frame: BufferId,
    images: &ImageBindings,
    (glyphs, nearest, dem, terrain_depth): (TextureId, SamplerId, TextureId, TextureId),
) -> BindGroupId {
    use TextureViewDimension::{D2, D2Array};
    let mut entries: Vec<BindGroupEntry> = [
        frame,
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
    .map(|(i, &buffer)| BindGroupEntry::buffer(i as u32, buffer))
    .collect();
    entries.extend([
        BindGroupEntry::texture(9, images.texture, D2Array),
        BindGroupEntry::sampler(10, images.sampler),
        BindGroupEntry::texture(11, images.dashes, D2),
        BindGroupEntry::texture(12, glyphs, D2Array),
        BindGroupEntry::sampler(13, nearest),
        BindGroupEntry::texture(14, dem, D2Array),
        BindGroupEntry::texture(15, terrain_depth, D2),
    ]);
    gpu.create_bind_group(&BindGroupDesc {
        label: "scene",
        layout,
        entries: &entries,
    })
}
