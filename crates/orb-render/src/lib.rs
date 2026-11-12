//! The GPU-driven render graph.
//!
//! A frame has a cull and main pass: one flat cull/compact dispatch, which
//! keeps each render node's visible `(chunk, tile placement)` candidates in
//! draw order and writes its indirect arguments, and the main pass, which
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
mod driver;
mod extrusion;
mod heatmap;
mod image;
mod layer_composite;
mod offscreen;
mod paint;
mod pipeline;
mod release;
mod scene;
mod symbol;
mod terrain;
mod tiles;
pub use atmosphere::{AtmospherePaint, AtmosphereView, GlobeSky};
pub use layer_composite::Blend;
pub use terrain::{
    DRAPE_FORMAT, DRAPE_SLOT_PX, DrapeFull, DrapeLayout, TERRAIN_EXTENT, TERRAIN_GRID, TerrainDem,
    TerrainGlobe, TerrainPaint, TerrainTile,
};

pub use image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, ImageFade, PageId, PageKind};
pub use orb_layout::{Family, pack_color};
pub use paint::{
    CirclePaint, ColorReliefPaint, Curve, EXTRUSION_GRADIENT_ROWS, EXTRUSION_OPACITY_ROWS,
    ElevationPaint, ExtrusionPaint, FILL_ANTIALIAS_ROWS, FillPaint, HeatmapPaint, HillshadeMethod,
    HillshadePaint, LinePaint, MAX_LIGHTS, NodePaint, PatternPaint, RasterPaint, RowScale,
    StateLookup, SymbolPaint, SymbolSize, Translate, ZoomBlend,
};
pub use scene::Pixels;
pub use scene::{
    ARGS_WORDS, ClipRect, DemRef, FrameBlock, FrameGlobe, FrameLight, FrameTerrain, ImageAtlas,
    ImageBindings, ImageKey, ImageMetadata, ImageUpload, MeshData, MeshId, MeshSize, NodeCull,
    NodeDraw, NodeMeshes, OutOfSpace, PlacementTarget, RowGlobe, SceneCapacity, SceneStats, TileId,
    TileRow, TileWarp, Words, warp_words,
};
pub use scene::{DESKTOP_BUDGET, MOBILE_BUDGET, PoolMemory, SceneMemory};
pub use scene::{
    GLYPH_PADDING, GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas, PLACED_WORDS, QUAD_WORDS,
    SYMBOL_HEADER_WORDS, SYMBOL_STATE_WORDS, SymbolMeshData, pack_image_rect, symbol_header,
    symbol_state,
};
pub use tiles::{
    GpuTile, ImageBuild, ImageMesh, MeshBuild, MeshRows, TileBuild, TileKey, UploadError, WarpKey,
    changed_borders, dem_neighbours, tile_latitudes,
};

use crate::pipeline::{Draw, Variants};
use crate::release::Release;

#[cfg(cpu_driven)]
#[doc(hidden)]
pub use driver::cpu::{Compared, DriverCheck};
use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_SHADER, LINE_STYLED_SHADER,
};
#[cfg(cpu_driven)]
use scene::Backing;
use scene::{DataTable, Scene, Table, Tables};

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

/// What a renderer draws otherwise than on WebGPU, for want of something
/// its device lacks (WebGL 2, #140): reported, never silent. Each is
/// decided from the device's caps where the renderer picks what it draws
/// with; a map reports those its style uses (`MapEvent::Degraded`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Gap {
    /// No blendable `Rgba16Float` target: heatmap density accumulates in 8
    /// bits, which bands and saturates (GL JS's own fallback).
    HeatmapPrecision,
    /// Depth remapped from 0–1 into GL's −1–1, where the context has no
    /// clip control (Firefox): far 3D buildings may fight at high pitch.
    DepthPrecision,
}

impl Gap {
    /// Its name, as events and fixtures give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::HeatmapPrecision => "heatmap-precision",
            Self::DepthPrecision => "depth-precision",
        }
    }
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
    /// Terrain drape slots drawn again: each a pass's share over a slot of
    /// up to 1024² texels, and its mips.
    pub drape_slots: u32,
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
    /// An extrusion node's cut stream's group ([`Scene::cut_row`]): its
    /// uniform at the stream's row, the node's paint with that row's
    /// index.
    cut_group: Option<BindGroupId>,
    /// A symbol node draws its halos in a pass of their own.
    separate_halo: bool,
    /// A symbol node places its labels along lines.
    along_line: bool,
    /// A line node has dashes, a pattern or a gradient: it draws with the
    /// styled line pipeline.
    styled: bool,
    /// A paint or symbol node's paint reads a feature's property row or
    /// state: it draws with its pipeline's variant with rows.
    rows: bool,
    /// What an extrusion node's pipelines are specialised for.
    elevation: paint::ExtrusionShape,
    /// On terrain, the drape stack a draped node draws into, and the
    /// terrain layer that composites it.
    stack: Option<u8>,
    terrain_layer: Option<u8>,
    /// A shaded hillshade blending other than normally that its terrain
    /// layer blends into the colours beneath it, rather than one
    /// composited apart.
    blends_on_terrain: bool,
    /// A hillshade the terrain pass lights from the frame's bearing: it
    /// drapes its unlit slopes ([`Renderer::set_shaded_on_terrain`]).
    shaded: bool,
    /// What it drapes changes with the zoom
    /// ([`Renderer::set_drapes_by_zoom`]).
    by_zoom: bool,
    /// Its style layer's key, layer opacity and blend mode
    /// (`*-layer-opacity`, `*-layer-blend`): below opacity 1 or blending
    /// other than normally, the layer composites once.
    composite: Option<(u32, f32, Blend)>,
    /// A background's colour where it is solid (opaque, no pattern), as
    /// its fragments write it: what the main pass can clear to in place
    /// of drawing it ([`Renderer::set_background_fills`]).
    solid: Option<[f32; 4]>,
    /// It has been shown since it was made: its pipelines were asked for,
    /// and the bundles record it. Until then it costs nothing, neither
    /// pipelines nor targets nor passes.
    drawn: bool,
    /// A bundle recorded since the last invalidation left it out: it had
    /// no candidates, so its draw would have drawn nothing. Candidates
    /// arriving for it record the bundles again.
    left_out: std::cell::Cell<bool>,
}

/// Whether a node painted with `a` draws with the pipelines it does
/// painted with `b`.
pub fn same_pipelines(a: &NodePaint, b: &NodePaint) -> bool {
    PaintShape::of(a) == PaintShape::of(b)
}

/// What of a node's paint picks its pipelines.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PaintShape {
    styled: bool,
    rows: bool,
    along_line: Option<bool>,
    elevation: Option<paint::ExtrusionShape>,
}

impl PaintShape {
    fn of(paint: &NodePaint) -> Self {
        Self {
            styled: paint.styled(),
            rows: paint.reads_rows(),
            along_line: match paint {
                NodePaint::Symbol(p) => Some(p.along_line),
                _ => None,
            },
            elevation: match paint {
                NodePaint::Extrusion(_) | NodePaint::ExtrusionPattern(..) => {
                    Some(paint.extrusion_shape())
                }
                _ => None,
            },
        }
    }
}

impl Node {
    /// A hillshade the terrain pass lights: its stack holds its slopes.
    fn shades_on_terrain(&self) -> bool {
        self.shaded && self.stack.is_some()
    }

    /// Its paint pipeline.
    fn paint_kind(&self) -> PaintKind {
        PaintKind {
            family: self.family,
            styled: self.styled,
            rows: self.rows,
        }
    }

    /// What of it picks its pipelines, which [`Renderer::set_pipelines`]
    /// sets from a paint it may draw with and puts back.
    fn shape(&self) -> NodeShape {
        (self.styled, self.rows, self.along_line, self.elevation)
    }

    /// Its pipelines' shape as `paint` picks them.
    fn set_shape(&mut self, paint: &NodePaint) {
        (self.styled, self.rows) = (paint.styled(), paint.reads_rows());
        match paint {
            NodePaint::Symbol(p) => self.along_line = p.along_line,
            NodePaint::Extrusion(_) | NodePaint::ExtrusionPattern(..) => {
                self.elevation = paint.extrusion_shape();
            }
            _ => {}
        }
    }

    fn restore_shape(&mut self, shape: NodeShape) {
        (self.styled, self.rows, self.along_line, self.elevation) = shape;
    }
}

/// [`Node::shape`]: styled, rows, along lines and elevation.
type NodeShape = (bool, bool, bool, paint::ExtrusionShape);

/// What draws together at one zoom a style shows it at, of a style's
/// pipeline set ([`Renderer::set_pipelines`]): a node and its paint
/// there (`None` for a family whose pipelines its paint doesn't pick),
/// or a symbol layer's icon and text nodes, which draw in one list with
/// one pipeline (`None` for a half that draws nothing there).
#[derive(Clone, Debug)]
pub struct PipelineGroup {
    pub nodes: Vec<(u32, Option<NodePaint>)>,
    /// The blend the layer composites in there, when it composites once
    /// (below layer opacity 1, or blending other than normally).
    pub composite: Option<Blend>,
    /// It draws in the view the set was made for: its pipelines compile
    /// first, and frames wait for them.
    pub first: bool,
}

#[derive(Debug)]
pub struct Renderer {
    id: u64,
    /// The host's target.
    target: TargetFormat,
    /// What it draws otherwise than on WebGPU.
    gaps: Vec<Gap>,
    atmosphere: atmosphere::Atmosphere,
    scene: Scene,
    draw_group: BindGroupId,
    /// How the frame's draws learn their counts: on the GPU, its cull.
    driver: driver::Driver,
    draw_layout: BindGroupLayoutId,
    node_layout: BindGroupLayoutId,
    /// The vector families' pipelines, each made for the family's first
    /// node.
    paint: PaintPipelines,
    /// Symbol layers' pipelines, placement target and passes.
    symbols: symbol::Symbols,
    /// Whether the map's views may need the full projection
    /// ([`set_full_projection`](Self::set_full_projection)).
    full_projection: bool,
    /// The style has terrain, on or not yet
    /// ([`set_terrain_reachable`](Self::set_terrain_reachable)).
    terrain_reachable: bool,
    /// Whether the last frame needed the paint and symbol pipelines' full
    /// variants, not their plane ones (`Variants`): its bundles drew with
    /// them.
    full: bool,
    /// The frame's backgrounds cover every pixel
    /// ([`set_background_fills`](Self::set_background_fills)).
    background_fills: bool,
    /// The solid background node the first of `bundles` ends with, as
    /// they were recorded: the bundle a frame the backgrounds fill leaves
    /// out, clearing to its colour.
    beneath: Option<usize>,
    /// Nodes changed since frames last checked that the pipeline set
    /// holds theirs, or one was shown.
    pipelines_stale: bool,
    /// Every pipeline the style can reach ([`set_pipelines`](Self::set_pipelines)).
    pipeline_set: Vec<PipelineGroup>,
    /// The set is to be asked for again, in what the map draws it with
    /// now: other variants, another sample count, or into the drape.
    set_stale: bool,
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
    /// The node uniforms as written, and the nodes changed since the last
    /// frame: one buffer write a frame, not one a node, and one for the
    /// cut streams' rows past them.
    node_words: Vec<u32>,
    nodes_dirty: [Option<std::ops::Range<usize>>; 2],
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
    /// The DEM page the terrain reads, as `set_terrain` last gave it.
    terrain_page: Option<PageId>,
    /// The drape pass's paint, frames and bundles, made for the first
    /// draped paint.
    drape: Option<Drape>,
    /// What each terrain draw composites, in style order.
    terrain_layers: Vec<terrain::TerrainLayer>,
    /// Colour and slope stacks of the terrain layers.
    drape_stacks: [u32; 2],
    /// Draped families' pipelines for the drape atlas, made for the first
    /// terrain when the target's format isn't the atlas's.
    drape_pipelines: Option<DrapePipelines>,
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
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
    /// The image texture the scene group binds.
    images_bound: TextureId,
}

/// Bytes reserved per node uniform; every node struct fits. A symbol
/// layer's icon node binds its text node's after its own, this far apart
/// (`symbol.wgsl`'s `halves`).
const NODE_UNIFORM_SIZE: u64 = 512;
const _: () = assert!(orb_shaders::layout::symbol_node::WORDS as u64 * 4 == NODE_UNIFORM_SIZE);

/// The node uniforms' usage: the symbol placements read them all at once.
fn node_uniform_usage(caps: &Caps) -> BufferUsages {
    BufferUsages::UNIFORM | BufferUsages::COPY_DST | symbol::read_usage(caps)
}

impl Renderer {
    /// Frees everything the renderer holds on `gpu`, the device it was
    /// made on, so what a map's frames made goes with it. The images it
    /// was handed stay.
    pub fn destroy(self, gpu: &mut dyn Gpu) {
        use release::Release;
        self.scene.destroy(gpu);
        self.glyphs.destroy(gpu);
        let layouts = (self.draw_layout, self.node_layout);
        self.driver.destroy(gpu);
        let groups = self.draw_group;
        let pipelines = (self.paint.pipelines, self.symbols);
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
        (self.layers, self.composites).release(gpu);
        let drape = self
            .drape_pipelines
            .map(|d| (d.paint.pipelines, d.images.0));
        (self.terrain, self.drape, drape, self.no_terrain_depth).release(gpu);
    }

    /// What a renderer on a device of `caps` draws otherwise than on
    /// WebGPU, whether or not a style uses it: [`Gap`].
    pub fn gaps_on(caps: &Caps) -> Vec<Gap> {
        let mut gaps = Vec::new();
        if !caps.float_targets {
            gaps.push(Gap::HeatmapPrecision);
        }
        if !caps.zero_to_one_depth {
            gaps.push(Gap::DepthPrecision);
        }
        gaps
    }

    /// What this renderer draws otherwise than on WebGPU: its device's
    /// [`gaps_on`](Self::gaps_on).
    pub fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    /// A renderer into `target`'s formats, drawing icons and patterns from
    /// the image atlas `images` (the map's, which outlives renderers).
    pub fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        capacity: SceneCapacity,
        images: ImageBindings,
    ) -> Self {
        let kind = driver::Kind::of(gpu.caps());
        let scene = Scene::new(gpu, capacity, kind);
        let b = *scene.buffers();
        // The driver's pipeline first: nothing draws without it.
        let driver = driver::Driver::new(gpu, kind, &scene);
        let (vs, fs) = (ShaderStages::VERTEX, ShaderStages::FRAGMENT);
        let vf = vs | fs;
        use BindGroupLayoutEntry as E;
        use TextureViewDimension::{D2, D2Array};
        let backing = scene.tables().backing();
        let mut entries = vec![E::uniform(0, vf)];
        entries.extend(Tables::layout_entries(backing));
        entries.push(DataTable::layout_entry(backing, 8, fs));
        entries.extend([
            E::texture(9, fs, true, D2Array),
            E::sampler(10, fs, true),
            E::texture(11, fs, true, D2),
            E::texture(12, fs, true, D2Array),
            E::sampler(13, fs, true),
            // Extrusions' walls read the ground per fragment.
            E::texture(14, vf, false, D2Array),
            E::texture(15, vf, false, D2),
            // The image pages again, for the nearest sampler
            // (`symbol.wgsl`).
            E::texture(16, fs, true, D2Array),
        ]);
        let draw_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "scene",
            entries: &entries,
        });
        let node_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node",
            entries: &[E::uniform(0, vf)],
        });

        let glyphs = GlyphAtlas::new(
            gpu,
            capacity.glyph_rows,
            scene.tables().writer(Table::Props),
        );
        let image_bindings = images;
        // Its nearest sampler and blank array serve the scene too.
        let images = image::Images::new(gpu, target, draw_layout);
        let (nearest, no_dem) = (images.nearest(), images.placeholder());
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
        });
        let draw_group = scene_group(
            gpu,
            draw_layout,
            scene.tables(),
            b.frame,
            &image_bindings,
            (glyphs.texture(), nearest, no_dem, no_terrain_depth),
        );

        let symbols = symbol::Symbols::new(gpu, backing);
        let downsampler = downsample::Downsampler::new(gpu);
        let heatmaps = heatmap::Heatmaps::new(draw_layout, node_layout);
        let composites = offscreen::Composites::new(gpu, target);
        // 3D layers draw into RGBA8.
        let extrusions =
            extrusion::Extrusions::new(draw_layout, node_layout, TextureFormat::Rgba8Unorm);
        // WebGPU aligns uniform offsets to at most 256 bytes: nodes lie
        // NODE_UNIFORM_SIZE apart, as the symbol shader's halves do.
        let alignment = gpu.caps().limits.min_uniform_buffer_offset_alignment;
        assert!(
            NODE_UNIFORM_SIZE.is_multiple_of(u64::from(alignment)),
            "uniform offsets aligned to {alignment} bytes"
        );
        let node_uniforms = gpu.create_buffer(&BufferDesc {
            label: "node uniforms",
            size: NODE_UNIFORM_SIZE * u64::from(capacity.nodes),
            usage: node_uniform_usage(gpu.caps()),
        });

        let atmosphere = atmosphere::Atmosphere::new(gpu);
        static RENDERERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            id: RENDERERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            target,
            gaps: Self::gaps_on(gpu.caps()),
            atmosphere,
            scene,
            draw_group,
            driver,
            draw_layout,
            node_layout,
            paint: PaintPipelines::default(),
            symbols,
            full_projection: false,
            terrain_reachable: false,
            full: false,
            background_fills: false,
            beneath: None,
            pipelines_stale: false,
            pipeline_set: Vec::new(),
            set_stale: false,
            made_seen: 0,
            waiting: 0,
            drape_waiting: false,
            recorded: 0,
            node_uniforms,
            node_words: vec![0; (NODE_UNIFORM_SIZE / 4) as usize * capacity.nodes as usize],
            nodes_dirty: [None, None],
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
            terrain_page: None,
            drape: None,
            terrain_layers: Vec::new(),
            drape_stacks: [0; 2],
            drape_pipelines: None,
            dem_bound: no_dem,
            no_terrain_depth,
            terrain_depth_bound: no_terrain_depth,
            glyphs_bound: glyphs.revision(),
            buffers_bound: 0,
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
        self.target
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
        let mut memory = self.scene.memory();
        let copies = self.driver.compacted_copies();
        memory.compacted.allocated *= copies;
        memory.compacted.ceiling *= copies;
        memory.cpu_copies += self.driver.heap_bytes();
        memory
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

    /// Draws 3D surfaces at four samples when `enabled`, or at one (the
    /// default).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        if self.extrusions.set_antialias(enabled) {
            self.set_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// Has the cull keep every candidate when `on` (`cull.wgsl`'s
    /// `keep_all`): it culls nothing, as a driver without it draws. For
    /// tests and measurements: what the cull saves, and that it only drops
    /// what draws nothing.
    pub fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) {
        if self.driver.set_keep_all(gpu, on) {
            self.invalidate_bundle(gpu);
        }
    }

    /// Runs the GPU's cull beside the CPU driver's when `on`, over the same
    /// scene and frames into buffers of its own, for a test to hold the
    /// two to the same words ([`driver_check`](Self::driver_check)); the
    /// candidates are laid out again, so both start alike. Only a
    /// CPU-driven renderer on a device with compute and the tables in
    /// storage buffers (orb-hal-wgpu's `force_cpu_driver(false)`) has a
    /// twin; returns whether this one does.
    #[cfg(cpu_driven)]
    #[doc(hidden)]
    pub fn check_driver(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        let driver::Driver::Cpu(d) = &mut self.driver else {
            return false;
        };
        if on && (!gpu.caps().compute || self.scene.tables().backing() != Backing::Storage) {
            return false;
        }
        d.set_twin(gpu, &self.scene, on);
        self.scene.lay_out_again();
        on
    }

    /// What the CPU driver and its GPU twin wrote, to read back after a
    /// frame and compare ([`check_driver`](Self::check_driver)).
    #[cfg(cpu_driven)]
    #[doc(hidden)]
    pub fn driver_check(&self) -> Option<DriverCheck> {
        match &self.driver {
            driver::Driver::Cpu(d) => DriverCheck::new(d, &self.scene),
            driver::Driver::Gpu(_) => None,
        }
    }

    /// The scene's per-node draw arguments and cull counters, for
    /// diagnostics: [`ARGS_WORDS`] words per node, as
    /// `cull.wgsl` writes them, and the number of nodes.
    pub fn draw_args(&self) -> (BufferId, u32) {
        (self.scene.buffers().args, self.nodes.len() as u32)
    }

    /// Per node, its family, the candidates its cull considers (as last
    /// laid out, none for a hidden node, which draws nothing) and those
    /// the cull kept, from the bytes of [`draw_args`](Self::draw_args)'
    /// buffer read back after a frame.
    pub fn cull_counts(&self, args: &[u8]) -> Vec<(Family, u32, u32)> {
        let word = |i: usize| {
            let at = 4 * (i * ARGS_WORDS + 5);
            args.get(at..at + 4)
                .map_or(0, |w| u32::from_le_bytes(w.try_into().expect("a word")))
        };
        // An extrusion node's counts take in its cut stream's.
        (self.nodes.iter().enumerate())
            .map(|(i, n)| {
                let rows = std::iter::once(i as u32).chain(self.scene.cut_row(i as u32));
                let (candidates, kept) = rows.fold((0, 0), |(c, k), row| {
                    (c + self.scene.node_candidates(row), k + word(row as usize))
                });
                (
                    n.family,
                    if self.scene_cull(i).visible {
                        candidates
                    } else {
                        0
                    },
                    kept,
                )
            })
            .collect()
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
        // A style with more nodes than ever drawn: the node tables grow,
        // with a row for each extrusion node's cut stream after them.
        // Every node's group binds its uniform again below.
        let cut = families
            .iter()
            .filter(|&&f| f == Family::FillExtrusion)
            .count();
        let count = (families.len() + cut) as u32;
        let mut retired_uniforms = None;
        if count > self.scene.capacity().nodes {
            self.scene.grow_nodes(gpu, count);
            let uniforms = gpu.create_buffer(&BufferDesc {
                label: "node uniforms",
                size: NODE_UNIFORM_SIZE * u64::from(self.scene.capacity().nodes),
                usage: node_uniform_usage(gpu.caps()),
            });
            retired_uniforms = Some(std::mem::replace(&mut self.node_uniforms, uniforms));
            // The uniforms' CPU copy grows with them.
            let words = (NODE_UNIFORM_SIZE / 4) as usize * self.scene.capacity().nodes as usize;
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
            // A symbol layer's icon node draws its text too, and binds the
            // text node's uniform after its own.
            let text = families.get(i + 1) == Some(&Family::SymbolText);
            assert_eq!(
                family == Family::SymbolIcon,
                text,
                "a symbol layer's icon node comes right before its text node"
            );
            let uniform = self.node_uniform(i as u32, text);
            if node.is_none() && family.is_image() {
                added.push(i as u32);
            }
            // A retained node keeps what decides its drape stack.
            let (mut image, cull, (shaded, by_zoom)) = match node {
                Some(node) => {
                    // Groups bind the uniform at the node's index.
                    ((node.group, node.cut_group), node.extrusion, node.heatmap).release(gpu);
                    (node.image, node.cull, (node.shaded, node.by_zoom))
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
                        meshes: NodeMeshes::Own,
                    },
                    (false, false),
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
                cut_group: None,
                heatmap: (family == Family::Heatmap).then(|| heatmap::Heatmaps::node(gpu)),
                cull,
                written: Vec::new(),
                image,
                separate_halo: false,
                along_line: false,
                styled: false,
                // A symbol half adds to the pipeline its layer draws with
                // only once painted: unpainted (its layer not shown yet, or
                // its part's opacity 0), it draws nothing.
                rows: !family.is_symbol(),
                elevation: paint::ExtrusionShape::default(),
                stack: None,
                terrain_layer: None,
                blends_on_terrain: false,
                shaded,
                by_zoom,
                composite: None,
                solid: None,
                drawn: false,
                left_out: Default::default(),
            });
        }
        // A symbol layer's draw finds its list by its icon node's index,
        // which its uniform holds even while no paint shows its icons.
        let stride = (NODE_UNIFORM_SIZE / 4) as usize;
        for i in 0..self.nodes.len() {
            if self.nodes[i].family == Family::SymbolIcon {
                let at = i * stride + orb_shaders::layout::symbol_node::INDEX;
                self.node_words[at] = i as u32;
                let dirty = &mut self.nodes_dirty[0];
                *dirty = Some(match dirty.take() {
                    Some(r) => r.start.min(i)..r.end.max(i + 1),
                    None => i..i + 1,
                });
            }
        }
        // Every group that bound the old uniforms is gone.
        retired_uniforms.release(gpu);
        self.assign_stacks();
        // Nodes moved: their drape paint is set again at their indices.
        self.drape.take().release(gpu);
        let cull: Vec<NodeCull> = (0..self.nodes.len()).map(|i| self.scene_cull(i)).collect();
        self.scene.remap_nodes(&map, &cull);
        for i in 0..self.nodes.len() {
            let Some(row) = self.scene.cut_row(i as u32) else {
                continue;
            };
            let uniform = self.node_uniform(row, false);
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "node cut stream",
                layout: self.node_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: uniform,
                }],
            });
            self.nodes[i].cut_group = Some(group);
        }
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
    /// layer; another starts the next). A terrain layer without one whose
    /// first nodes draw alike at every zoom and whose later ones don't
    /// ([`set_drapes_by_zoom`](Self::set_drapes_by_zoom)) splits there
    /// into a stack below and one above, so that zooming redraws only the
    /// one above. Colour stacks come first, in order, then slope stacks,
    /// which the drape keeps on a page of their own ([`DrapeLayout`]).
    fn assign_stacks(&mut self) {
        self.assign_stacks_as(self.terrain_on());
    }

    /// Puts the draped nodes in stacks as on terrain (`on`), or in none.
    fn assign_stacks_as(&mut self, on: bool) {
        let inputs: Vec<StackInput> = (self.nodes.iter())
            .map(|node| StackInput {
                draped: on && node.family.is_draped(),
                group: composite_group(node.composite),
                blend: (node.composite).map_or(Blend::Normal, |(_, _, blend)| blend),
                shaded: node.shaded,
                by_zoom: node.by_zoom,
            })
            .collect();
        let stacks = drape_stacks(&inputs);
        for (i, (node, (stack, layer))) in (self.nodes.iter_mut()).zip(stacks.nodes).enumerate() {
            node.stack = stack;
            node.terrain_layer = layer;
            node.blends_on_terrain = layer.is_some_and(|l| {
                let l = &stacks.layers[usize::from(l)];
                l.shade_blend != Blend::Normal && l.shade.is_some_and(|(n, _)| n as usize == i)
            });
            node.cull.target = match stack {
                Some(stack) => PlacementTarget::Drape(stack),
                None => PlacementTarget::Map,
            };
        }
        self.drape_stacks = stacks.counts;
        self.terrain_layers = stacks.layers;
        // A layer's own terrain layer composites in its blend mode.
        for node in self.nodes.iter().filter(|n| !n.blends_on_terrain) {
            if let (Some(layer), Some((_, _, blend))) = (node.terrain_layer, node.composite)
                && blend != Blend::Normal
            {
                self.terrain_layers[usize::from(layer)].blended = true;
            }
        }
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
            self.pipelines_stale = true;
            self.restack(gpu);
        }
    }

    /// Says whether what draped node `node` draws changes with the zoom:
    /// sizes in pixels, or paint that follows the zoom. Draped nodes that
    /// don't, under ones that do, keep a stack of their own, which zooming
    /// doesn't redraw.
    pub fn set_drapes_by_zoom(&mut self, gpu: &mut dyn Gpu, node: u32, by_zoom: bool) {
        let n = &mut self.nodes[node as usize];
        if n.by_zoom == by_zoom {
            return;
        }
        n.by_zoom = by_zoom;
        if n.terrain_layer.is_some() {
            self.restack(gpu);
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

    /// Turns terrain on with its DEM page, or flat before it has one, or
    /// off (`None`). On terrain, draped nodes draw only drape placements:
    /// their stack's ([`PlacementTarget::Drape`]).
    pub fn set_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        terrain: Option<(Option<PageId>, TerrainPaint)>,
    ) {
        let was = self.terrain_on();
        self.terrain_page = terrain.and_then(|(page, _)| page);
        match terrain {
            Some((page, paint)) => {
                let (dem, dim) = match page {
                    Some(page) => (
                        self.images.dem_texture(page),
                        self.images.page_size(page)[0],
                    ),
                    None => ((self.images.placeholder(), 0), 1),
                };
                self.terrain(gpu).set_paint(gpu, paint, dem, dim);
            }
            None => {
                if let Some(t) = &mut self.terrain {
                    t.disable(gpu);
                }
                self.drape.take().release(gpu);
            }
        }
        if was != self.terrain_on() {
            // Draped nodes draw into the drape, with its pipelines, and
            // in the full variants.
            self.set_stale = true;
            self.restack(gpu);
        }
    }

    /// Stacks the draped nodes again, after what puts them in stacks
    /// changed: their cull and bundles follow.
    fn restack(&mut self, gpu: &mut dyn Gpu) {
        self.assign_stacks();
        self.sync_cull();
        self.invalidate_bundle(gpu);
    }

    /// The terrain, made for the first that turns on or that the set asks
    /// for ahead.
    fn terrain(&mut self, gpu: &mut dyn Gpu) -> &mut terrain::Terrain {
        let (target, atmosphere) = (self.target, self.atmosphere.uniform);
        let images = (self.images.image_layout(), self.images.linear());
        (self.terrain).get_or_insert_with(|| terrain::Terrain::new(gpu, target, atmosphere, images))
    }

    /// Sets whether the style has terrain, on or not yet (its DEM source
    /// may still resolve): the pipeline set then holds the terrain's and
    /// the draped nodes' drape pipelines from the first.
    pub fn set_terrain_reachable(&mut self, on: bool) {
        if self.terrain_reachable != on {
            self.terrain_reachable = on;
            self.set_stale = true;
        }
    }

    /// Whether the style's terrain is still to turn on: its DEM source
    /// resolves.
    fn terrain_ahead(&self) -> bool {
        self.terrain_reachable && !self.terrain_on()
    }

    /// Whether terrain is on: draped nodes draw drape placements.
    pub fn terrain_on(&self) -> bool {
        self.terrain.as_ref().is_some_and(|t| t.paint.is_some())
    }

    /// Sets whether the map's views may need the full projection: it
    /// draws on a low-zoom surface (the style's globe, or a projection's
    /// low-zoom plane) at some zoom, or a source's tiles warp onto its
    /// surface, or terrain may be on. Those views need the paint and
    /// symbol pipelines' full variants, which are then the set's, so a
    /// zoom onto the surface or a warped tile's arrival doesn't wait for
    /// them.
    pub fn set_full_projection(&mut self, on: bool) {
        if self.full_projection != on {
            self.full_projection = on;
            self.set_stale = true;
        }
    }

    /// Whether a frame needs the paint and symbol pipelines' full
    /// variants: it is on its low-zoom surface, terrain is on, or a tile
    /// draws through planar patches.
    fn needs_full(&self, frame: &FrameBlock) -> bool {
        frame.globe.is_some_and(|g| g.amount > 0.0) || self.terrain_on() || self.tiles.warped() > 0
    }

    /// Which variants of the paint and symbol pipelines the set takes,
    /// `[plane, full]`: the plane's for a map that never needs the full
    /// ones, and otherwise the full ones, which draw every view. The map
    /// says which ([`set_full_projection`](Self::set_full_projection)),
    /// not its frames.
    fn variants(&self) -> [bool; 2] {
        let full = self.full_projection || self.terrain_on();
        [!full, full]
    }

    /// How many terrain tiles the drape atlas can hold with this style's
    /// stacks: the most a terrain cover may have.
    pub fn drape_room(&self, gpu: &dyn Gpu) -> u32 {
        terrain::room(gpu, self.drape_stacks())
    }

    /// The tile slots a map at rest with `tiles` terrain tiles reserves,
    /// when that resizes the drape pages: one past their room when the
    /// tiles take more than three quarters of it (within
    /// [`drape_room`](Self::drape_room)), so that they grow before a
    /// gesture needs them to, or the tiles when the pages would shrink to
    /// them. `None` when the pages stay.
    pub fn drape_at_rest(&self, gpu: &dyn Gpu, tiles: u32) -> Option<u32> {
        let terrain = self.terrain.as_ref()?;
        let held = terrain.layout().tile_slots;
        let stacks = self.drape_stacks();
        let ahead = (held + 1).min(self.drape_room(gpu));
        if held > 0 && u64::from(tiles) * 4 > u64::from(held) * 3 && ahead > held {
            return Some(ahead);
        }
        (held > 0 && terrain.would_reserve(gpu, tiles, stacks)).then_some(tiles)
    }

    /// Makes room for `tile_slots` terrain tiles' drape slots, one per
    /// drape stack, keeping the texels of the tile slots in `kept`. Returns
    /// the layout and the kept slots that moved (from, to). A new layout
    /// generation means every slot is blank. Fails past the
    /// [`drape_room`](Self::drape_room), or where the device refuses a
    /// page, which the next call makes smaller. Pages that hold the slots stay
    /// unless `may_shrink`: a continuous map shrinks them at rest.
    pub fn reserve_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, kept): (u32, &[u32]),
        may_shrink: bool,
    ) -> Result<(DrapeLayout, Vec<(u32, u32)>), DrapeFull> {
        let stacks = self.drape_stacks();
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let made = terrain.reserve(
            gpu,
            (tile_slots, kept),
            (stacks, may_shrink),
            &mut self.downsampler,
        )?;
        let layout = terrain.layout();
        if made.is_some() {
            // Stack layers and the main bundle's composites follow it.
            self.invalidate_bundle(gpu);
        }
        Ok((layout, made.unwrap_or_default()))
    }

    /// Redraws these terrain tiles' drape slots this frame, in the stacks
    /// given: their drape placements are in the scene, and every draped
    /// node's paint was set ([`set_drape_paint`](Self::set_drape_paint)).
    pub fn redraw_drape(&mut self, tile_slots: &[(u32, &[u8])]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.redraw(tile_slots);
        }
    }

    /// Writes a draped node's paint for the slots redrawn from now on, or
    /// with `None` leaves it out of them (its layer doesn't draw at the
    /// zoom). Whether anything changed: the slots of its stack then need
    /// drawing again.
    pub fn set_drape_paint(
        &mut self,
        gpu: &mut dyn Gpu,
        node: u32,
        paint: Option<&NodePaint>,
    ) -> bool {
        let words = paint.map(|p| {
            assert!(
                p.fits(self.nodes[node as usize].family),
                "paint family mismatch for node {node}"
            );
            p.uniform_words(node)
        });
        let nodes = self.nodes.len() as u64;
        let stride = NODE_UNIFORM_SIZE;
        let level = self.drape.get_or_insert_with(|| Drape {
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
            bundle: Vec::new(),
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

    /// Node `node`'s uniform, and with `next` the next node's after it.
    fn node_uniform(&self, node: u32, next: bool) -> BindingResource {
        BindingResource::Buffer {
            buffer: self.node_uniforms,
            offset: u64::from(node) * NODE_UNIFORM_SIZE,
            size: Some(NODE_UNIFORM_SIZE + if next { NODE_UNIFORM_SIZE } else { 0 }),
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
            self.pipelines_stale = true;
            if n.terrain_layer.is_some() {
                // The layer gains or loses a terrain layer of its own.
                self.restack(gpu);
            } else {
                self.invalidate_bundle(gpu);
            }
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
        if let Some(drape) = &mut self.drape {
            drape.forget_bundle(gpu);
        }
    }

    /// Records every bundle that draws nodes again, extrusions' and
    /// heatmaps' too: after their bindings or their draws changed.
    fn forget_draws(&mut self, gpu: &mut dyn Gpu) {
        self.invalidate_bundle(gpu);
        for node in &mut self.nodes {
            if let Some(extrusion) = &mut node.extrusion {
                extrusion.forget_bundle(gpu);
            }
            if let Some(heatmap) = &mut node.heatmap {
                heatmap.apart.forget_bundle(gpu);
            }
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
            let uniform = self.node_uniform(i as u32, false);
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
        let cut_row = self.scene.cut_row(node);
        let entry = &mut self.nodes[node as usize];
        assert!(
            paint.fits(entry.family),
            "paint family mismatch for node {node}"
        );
        let mut words = paint.uniform_words(node);
        entry.solid = match paint {
            NodePaint::Background(p)
                if p.table_mask == 0 && p.state.mask == 0 && p.color[3] * p.opacity == 1.0 =>
            {
                Some(p.color.map(|c| c * p.opacity))
            }
            _ => None,
        };
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
        let (styled, rows) = (paint.styled(), paint.reads_rows());
        if (entry.styled, entry.rows) != (styled, rows) {
            entry.styled = styled;
            entry.rows = rows;
            if self.nodes[self.drawer(node as usize)].drawn {
                self.pipelines_stale = true;
                self.invalidate_bundle(gpu);
            }
        }
        let entry = &mut self.nodes[node as usize];
        if let NodePaint::Extrusion(_) | NodePaint::ExtrusionPattern(..) = paint {
            let shape = paint.extrusion_shape();
            if entry.elevation != shape {
                entry.elevation = shape;
                self.pipelines_stale |= entry.drawn;
                if let Some(e) = &mut entry.extrusion {
                    e.forget_bundle(gpu);
                }
            }
        }
        if words != entry.written {
            let stride = (NODE_UNIFORM_SIZE / 4) as usize;
            // An extrusion node's cut stream draws with its paint, from
            // its own row, past the render nodes'.
            let cut = cut_row.map(|row| {
                let mut cut = words.clone();
                cut[orb_shaders::layout::extrusion_node::INDEX] = row;
                (row as usize, cut)
            });
            let rows = std::iter::once((0, node as usize, &words));
            for (part, row, words) in rows.chain(cut.as_ref().map(|(row, cut)| (1, *row, cut))) {
                self.node_words[row * stride..][..words.len()].copy_from_slice(words);
                let dirty = &mut self.nodes_dirty[part];
                *dirty = Some(match dirty.take() {
                    Some(r) => r.start.min(row)..r.end.max(row + 1),
                    None => row..row + 1,
                });
            }
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
            self.sync_node(node as usize);
        }
    }

    /// Shows or hides a node without re-recording the bundle or laying the
    /// cull's candidates out again (but for one half of a symbol layer).
    pub fn set_visible(&mut self, node: u32, visible: bool) {
        let entry = &mut self.nodes[node as usize];
        if entry.cull.visible != visible {
            entry.cull.visible = visible;
            let drawer = self.drawer(node as usize);
            self.pipelines_stale |= visible && !self.nodes[drawer].drawn;
            self.sync_node(node as usize);
        }
    }

    /// The node whose draw draws node `i`'s meshes: a symbol layer's icon
    /// node draws its text's.
    fn drawer(&self, i: usize) -> usize {
        match self.nodes[i].family {
            Family::SymbolText => i - 1,
            _ => i,
        }
    }

    /// What the cull sees of node `i`: its own, but that a symbol layer's
    /// icon node takes its text node's meshes after its own, in one list,
    /// and shows while either does. Its margin is the wider of the two, and
    /// the text node draws nothing itself.
    fn scene_cull(&self, i: usize) -> NodeCull {
        let node = &self.nodes[i];
        match node.family {
            Family::SymbolIcon => {
                let (icons, text) = (node.cull, self.nodes[i + 1].cull);
                NodeCull {
                    margin_px: icons.margin_px.max(text.margin_px),
                    visible: icons.visible || text.visible,
                    // A half hidden alone leaves the list; the two hidden
                    // keep it as it is, for when they show again.
                    meshes: NodeMeshes::WithNext {
                        own: icons.visible || !text.visible,
                        next: text.visible || !icons.visible,
                    },
                    ..icons
                }
            }
            Family::SymbolText => NodeCull {
                visible: false,
                meshes: NodeMeshes::Previous,
                ..node.cull
            },
            _ => node.cull,
        }
    }

    /// Gives the cull node `i` as it is now, with its symbol layer's other
    /// node.
    fn sync_node(&mut self, i: usize) {
        let drawer = self.drawer(i);
        let nodes = match self.nodes[drawer].family {
            Family::SymbolIcon => drawer..drawer + 2,
            _ => drawer..drawer + 1,
        };
        for n in nodes {
            self.scene.set_node(n as u32, self.scene_cull(n));
        }
    }

    /// The symbol pipeline a symbol layer's icon node `i` draws both its
    /// halves with: along lines if either places along them, with property
    /// rows if either reads them.
    fn symbol_kind(&self, i: usize) -> usize {
        let [icons, text] = [&self.nodes[i], &self.nodes[i + 1]];
        2 * usize::from(icons.along_line || text.along_line) + usize::from(icons.rows || text.rows)
    }

    /// Icon node `i`'s [`symbol_kind`](Self::symbol_kind), and the
    /// [`placings`](Self::placings) that place every symbol layer's quads.
    fn symbol_kinds(&self, i: usize) -> (usize, symbol::Placings) {
        (self.symbol_kind(i), self.placings())
    }

    /// The placement dispatches' kinds, by whether they place along lines:
    /// the point labels' and the line labels' (`None` where no symbol
    /// layer is of it), each with property rows if any of its layers
    /// reads them.
    fn placings(&self) -> symbol::Placings {
        let mut placings = [None; 2];
        let icons = (self.nodes.iter().enumerate()).filter(|(_, n)| n.family == Family::SymbolIcon);
        for (i, _) in icons {
            let kind = self.symbol_kind(i);
            let placing = placings[kind / 2].get_or_insert(kind);
            *placing |= kind;
        }
        placings
    }

    /// Sets whether the frame's backgrounds cover every pixel: drawn flat
    /// on tiles that fill the view. Then the main pass clears to the
    /// topmost solid one's colour, as MapLibre Native's
    /// `backgroundLayerAsColor` does, rather than shade it and the layers
    /// it hides: on the moto g a full-screen background cost 2 ms of the
    /// main pass at every view (#180). Pixels stay as they were: an opaque
    /// colour blends to itself over anything.
    pub fn set_background_fills(&mut self, fills: bool) {
        self.background_fills = fills;
    }

    /// The node the main pass can clear to in place of drawing it and
    /// what lies beneath it (`beneath`): the topmost solid background,
    /// when nothing beneath it composites apart, which could split the
    /// pass. Only paint chooses it, so neither a camera move nor showing
    /// or hiding a layer records the bundles again.
    fn solid_beneath(&self) -> Option<usize> {
        let i = (self.nodes.iter()).rposition(|n| {
            n.solid.is_some() && n.drawn && composite_group(n.composite).is_none()
        })?;
        (self.nodes[..i].iter())
            .all(|n| composite_group(n.composite).is_none())
            .then_some(i)
    }

    fn sync_cull(&mut self) {
        let cull: Vec<NodeCull> = (0..self.nodes.len()).map(|i| self.scene_cull(i)).collect();
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
        let target = (self.target, self.full_projection);
        self.atmosphere.set(gpu, target, frame, paint, view);
    }

    /// Whether a layer can composite at a layer opacity, asking for the
    /// pipeline it takes: a frame that would wait for it draws nothing,
    /// so what fades through it starts once it compiled.
    pub fn layer_composite_ready(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.composites.make(gpu, offscreen::Mode::Over);
        self.composites.ready(gpu, offscreen::Mode::Over)
    }

    /// Sets the style's pipeline set: every pipeline it can reach, by
    /// what draws together ([`PipelineGroup`]). Asks for them at once: the
    /// first view's groups as frames wait for them, in draw order, then
    /// the rest in the background, after every other compile. Frames then
    /// choose among them and make none: a node shown before its pipeline
    /// compiled waits for that pipeline alone, or draws with a variant
    /// compiled that draws it alike (`Variants::drawn`).
    pub fn set_pipelines(&mut self, gpu: &mut dyn Gpu, set: Vec<PipelineGroup>) {
        self.pipeline_set = set;
        self.set_stale = false;
        self.request_set(gpu);
    }

    /// What comes before a frame: asks for the set's pipelines again in
    /// what the map now draws them with (other variants, sample count,
    /// terrain), once that changed, and for the pipelines of nodes shown
    /// that the set left out. Returns how many of those it asked for: none
    /// for a complete set, which the path tests hold.
    pub fn request_pipelines(&mut self, gpu: &mut dyn Gpu) -> u32 {
        if std::mem::take(&mut self.set_stale) {
            self.request_set(gpu);
        }
        if !self.pipelines_stale {
            return 0;
        }
        let requested = gpu.counters().pipelines_requested;
        for i in 0..self.nodes.len() {
            if !self.scene_cull(i).visible && !self.nodes[i].drawn {
                continue;
            }
            self.make_for_node(gpu, i, false);
            let composite = self.nodes[i].composite;
            if let (Some(_), Some((.., blend))) = (composite_group(composite), composite) {
                self.composites.make(gpu, offscreen::Mode::Blend(blend));
            }
        }
        gpu.counters().pipelines_requested - requested
    }

    /// Asks for the set's pipelines that aren't made: the first view's,
    /// then the rest in the background. On a style whose terrain isn't on
    /// yet, they are the drape's and the terrain's: its draped nodes wait
    /// for it.
    fn request_set(&mut self, gpu: &mut dyn Gpu) {
        // Extrusions take the paint pipelines' variant.
        self.extrusions.set_full(self.variants()[1]);
        let set = std::mem::take(&mut self.pipeline_set);
        let ahead = self.terrain_ahead();
        if ahead {
            self.assign_stacks_as(true);
        }
        for first in [true, false] {
            gpu.set_background_compiles(!first);
            for group in set.iter().filter(|g| g.first == first) {
                self.request_group(gpu, &group.nodes, !first);
                if let Some(blend) = group.composite {
                    self.composites.make(gpu, offscreen::Mode::Blend(blend));
                }
            }
        }
        gpu.set_background_compiles(false);
        if ahead || self.terrain_on() {
            self.request_terrain(gpu);
        }
        if ahead {
            self.assign_stacks();
        }
        self.pipeline_set = set;
    }

    /// Asks for the terrain's pipelines at the sample count 3D layers draw
    /// at, a shaded hillshade's where the terrain layers shade one, and
    /// the drape atlas's mips.
    fn request_terrain(&mut self, gpu: &mut dyn Gpu) {
        let samples = self.extrusions.samples;
        let shaded = self.terrain_layers.iter().any(|l| l.shade.is_some());
        self.terrain(gpu).request_pipelines(gpu, samples, shaded);
        self.downsampler.make(gpu);
        self.images.make_prepare(gpu);
        // Terrain layers composite into the map.
        self.composites.make(gpu, offscreen::Mode::Over);
    }

    /// Asks for the pipelines one group draws with: each node shaped as
    /// its paint there picks them, then put back.
    fn request_group(&mut self, gpu: &mut dyn Gpu, group: &[(u32, Option<NodePaint>)], warm: bool) {
        let fits = |&(i, ref paint): &(u32, Option<NodePaint>)| {
            let node = self.nodes.get(i as usize);
            node.is_some_and(|n| paint.as_ref().is_none_or(|p| p.fits(n.family)))
        };
        if !group.iter().all(fits) {
            return;
        }
        let kept: Vec<_> = (group.iter())
            .map(|(i, paint)| {
                let node = &mut self.nodes[*i as usize];
                let kept = node.shape();
                match paint {
                    Some(paint) => node.set_shape(paint),
                    // A symbol half that draws nothing reads nothing.
                    None if node.family.is_symbol() => {
                        (node.rows, node.along_line) = (false, false)
                    }
                    None => {}
                }
                (*i as usize, kept)
            })
            .collect();
        let mut drawers: Vec<usize> = kept.iter().map(|&(i, _)| self.drawer(i)).collect();
        drawers.dedup();
        for i in drawers {
            self.make_for_node(gpu, i, warm);
        }
        for (i, kept) in kept {
            self.nodes[i].restore_shape(kept);
        }
    }

    /// Marks the nodes shown (their layers draw at this zoom) as drawn: a
    /// node shown for the first time records the bundles again, which
    /// left it out, and a draped one redraws the drape slots.
    fn show_nodes(&mut self, gpu: &mut dyn Gpu) {
        let (mut started, mut draped) = (false, false);
        for i in 0..self.nodes.len() {
            // A node that drew keeps its pipelines made for what it draws
            // into (the 3D sample count, the terrain's drape) while it
            // hides: the bundles record it, and it may show again.
            if !self.scene_cull(i).visible {
                continue;
            }
            let node = &mut self.nodes[i];
            if !std::mem::replace(&mut node.drawn, true) {
                started = true;
                draped |= node.stack.is_some();
            }
        }
        if started {
            self.invalidate_bundle(gpu);
        }
        if draped && let Some(terrain) = self.terrain.as_mut() {
            // Slots drawn without them lack those layers.
            terrain.redraw_all();
        }
    }

    /// Makes node `i`'s pipelines as it draws now, or (`warm`) ahead of
    /// its drawing.
    fn make_for_node(&mut self, gpu: &mut dyn Gpu, i: usize, warm: bool) {
        let layouts = [self.draw_layout, self.node_layout];
        let drape_target = TargetFormat {
            color: DRAPE_FORMAT,
            sample_count: 1,
        };
        let own_drape = self.target.color != drape_target.color;
        let variants = self.variants();
        let node = &self.nodes[i];
        let (family, kind) = (node.family, node.paint_kind());
        if matches!(family, Family::Hillshade | Family::ColorRelief) {
            // Its DEM tiles are prepared.
            self.images.make_prepare(gpu);
        }
        if node.shades_on_terrain() {
            // Its slopes into the drape atlas, which the terrain pass
            // lights.
            self.images.make_slopes(gpu);
            return;
        }
        if node.stack.is_none() && family.is_draped() && own_drape && self.terrain_ahead() {
            // It drapes once the style's terrain is on: until then it waits
            // for it, rather than for pipelines it would draw flat with.
            return;
        }
        if node.stack.is_some() && own_drape {
            // Into the drape atlas, with its format's pipelines.
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
                drape
                    .paint
                    .make(gpu, drape_target, &layouts, kind, variants);
            }
            return;
        }
        self.make_node_pipeline(gpu, i, warm);
    }

    fn make_node_pipeline(&mut self, gpu: &mut dyn Gpu, i: usize, warm: bool) {
        let layouts = [self.draw_layout, self.node_layout];
        let family = self.nodes[i].family;
        let variants = self.variants();
        match family {
            Family::Fill | Family::FillOutline | Family::Line | Family::Circle => {
                let kind = self.nodes[i].paint_kind();
                self.paint.make(gpu, self.target, &layouts, kind, variants);
            }
            Family::SymbolIcon | Family::SymbolText => {
                let kinds = self.symbol_kinds(self.drawer(i));
                let layouts = (self.draw_layout, self.node_layout);
                (self.symbols).make(gpu, layouts, self.target, kinds, variants);
            }
            Family::Raster | Family::Hillshade | Family::ColorRelief => {
                self.images.make(gpu, family);
                if family == Family::Raster {
                    // Raster pages are mipmapped.
                    self.downsampler.make(gpu);
                }
            }
            Family::Heatmap => {
                self.heatmaps.make(gpu);
                // Its target's composite through the ramp.
                self.composites.make(gpu, offscreen::Mode::Ramp);
            }
            Family::FillExtrusion => {
                self.extrusions.make(gpu, self.nodes[i].elevation, warm);
                // Its target's composite.
                self.composites.make(gpu, offscreen::Mode::Over);
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
                    .wanted(node.paint_kind(), self.full)
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
            Family::FillExtrusion => out.extend(self.extrusions.pipelines(node.elevation)),
            Family::SymbolIcon | Family::SymbolText => {
                let kinds = self.symbol_kinds(self.drawer(i));
                out.extend(self.symbols.wanted(kinds, self.full));
            }
            family if family.is_image() => {
                out.extend(self.images.pipeline(family).map(PipelineId::from));
                if family == Family::Raster {
                    out.extend(self.downsampler.pipelines());
                } else {
                    out.extend(self.images.prepare_pipelines());
                }
            }
            _ => {
                out.extend((self.paint.wanted(node.paint_kind(), self.full)).map(PipelineId::from))
            }
        }
    }

    /// Moves the pipelines of nodes with meshes in the scene ahead of the
    /// others still compiling: what the current view needs first. Nodes
    /// earlier in draw order go first (the bottom layers cover most of
    /// the view), the cull before everything.
    fn prioritize(&self, gpu: &mut dyn Gpu) {
        let mut wanted = Vec::new();
        wanted.extend(self.driver.pipelines());
        for i in 0..self.nodes.len() {
            if self.scene.node_has_meshes(i as u32) {
                self.node_pipelines(i, &mut wanted);
            }
        }
        wanted.extend(self.atmosphere.pipelines());
        // Each moves to the front: the last asked for compiles first.
        let mut seen = std::collections::HashSet::new();
        wanted.retain(|&id| seen.insert(id) && !gpu.pipeline_ready(id));
        for &id in wanted.iter().rev() {
            gpu.prioritize_pipeline(id);
        }
    }

    /// Draws a frame into `target`, cleared to `clear` (premultiplied RGBA),
    /// timing its stages on `clock` (milliseconds). It asks for no
    /// pipeline: [`request_pipelines`](Self::request_pipelines) comes
    /// before it.
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
        clock: &dyn Fn() -> f64,
    ) -> FrameStats {
        self.destroy_retired_pages(gpu);
        // A DEM tile that arrived since `set_terrain` may have grown its
        // page into new textures, the one it gave destroyed.
        if let (Some(t), Some(page)) = (&mut self.terrain, self.terrain_page) {
            t.set_dem(gpu, self.images.dem_texture(page));
        }
        self.bind_scene(gpu);
        for nodes in self
            .nodes_dirty
            .each_mut()
            .map(Option::take)
            .into_iter()
            .flatten()
        {
            let stride = (NODE_UNIFORM_SIZE / 4) as usize;
            gpu.write_buffer(
                self.node_uniforms,
                nodes.start as u64 * NODE_UNIFORM_SIZE,
                bytemuck::cast_slice(&self.node_words[nodes.start * stride..nodes.end * stride]),
            );
        }
        let generation = self.terrain.as_ref().map(|t| t.layout().generation);
        let requested = gpu.counters().pipelines_requested;
        let mut stats = self.record(gpu, frame, target, clear, clock);
        debug_assert_eq!(
            gpu.counters().pipelines_requested,
            requested,
            "a frame asked for a pipeline"
        );
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
            .unwrap_or(self.images.placeholder());
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
                self.driver.rebind(gpu, &self.scene);
            }
            gpu.destroy_bind_group(self.draw_group);
            self.draw_group = scene_group(
                gpu,
                self.draw_layout,
                self.scene.tables(),
                self.scene.buffers().frame,
                &self.image_bindings,
                (
                    self.glyphs.texture(),
                    self.images.nearest(),
                    dem,
                    terrain_depth,
                ),
            );
            self.glyphs_bound = self.glyphs.revision();
            self.dem_bound = dem;
            self.terrain_depth_bound = terrain_depth;
            for scene in self.drape.iter_mut().flat_map(|d| &mut d.scene) {
                scene.take().release(gpu);
            }
            // Extrusion and heatmap nodes record the scene group too.
            self.forget_draws(gpu);
        }
    }

    /// What the symbol placements bind of the scene, as
    /// `symbol::compute_layout` has it: the frame, the tables but the
    /// indices, the terrain as the scene group binds it, the node
    /// uniforms and the draw arguments.
    fn symbol_scene(&self) -> Vec<BindGroupEntry> {
        use TextureViewDimension::{D2, D2Array};
        let mut entries = vec![BindGroupEntry::buffer(0, self.scene.buffers().frame)];
        entries.extend((self.scene.tables().entries().into_iter()).filter(|e| e.binding != 5));
        entries.extend([
            BindGroupEntry::texture(14, self.dem_bound, D2Array),
            BindGroupEntry::texture(15, self.terrain_depth_bound, D2),
            BindGroupEntry::buffer(17, self.node_uniforms),
            BindGroupEntry::buffer(18, self.scene.buffers().args),
        ]);
        entries
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
        let (candidates, candidates_rebuilt) = self.scene.prepare(gpu, frame);
        // The CPU driver's cull: draws that changed record again.
        if self
            .driver
            .prepare(gpu, &mut self.scene, candidates_rebuilt)
        {
            self.forget_draws(gpu);
        }
        times.cull = micros(started, clock());
        self.follow_candidates(gpu, candidates_rebuilt);
        self.refresh_image_groups(gpu);
        let compiling = self.settle_pipelines(gpu, frame);
        let made = compiling.made;
        let terrain_on = self.terrain.as_ref().is_some_and(terrain::Terrain::active);
        let size = self.resize_targets(gpu, frame, terrain_on);
        // Bound once the targets are sized: a resize remade the terrain's
        // depth the scene group samples, and preparing may have grown the
        // compacted index buffer. Bound before, the frame's bundles would
        // sample the destroyed depth and the submit fail.
        self.bind_scene(gpu);
        let beneath = self.solid_beneath();
        if beneath != self.beneath {
            self.beneath = beneath;
            self.invalidate_bundle(gpu);
        }
        // The symbol draws bind the placement target.
        if self.symbols.fit(gpu, self.scene.placed_rows()) {
            self.invalidate_bundle(gpu);
        }
        self.symbols.bind(gpu, self.symbol_scene());
        let bundle_recorded = self.bundles.is_empty();
        let recording = clock();
        if bundle_recorded {
            self.record_bundles(gpu, size, terrain_on);
        }
        let draping = clock();
        times.bundles = micros(recording, draping);
        let stacks = match self.terrain.as_ref().filter(|_| terrain_on) {
            Some(terrain) => terrain.pending_stacks(),
            None => Vec::new(),
        };
        let drape: Vec<(u32, RenderBundleId)> = (stacks.into_iter())
            .filter_map(|stack| Some((stack, self.drape_bundle(gpu, frame, stack)?)))
            .collect();
        let encoding = clock();
        let mut drape_passes = 0;

        let mut passes = Vec::with_capacity(3);
        self.driver.moves(gpu, &mut self.scene, &mut passes);
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
        let cull_ready = self.cull_passes(gpu, &mut passes, &mut stats);
        let extrusions_ready = cull_ready && self.extrusions.ready(gpu, &self.composites);
        let (terrain_above, depth_cleared) = match terrain_on {
            true => {
                let passing = clock();
                let (above, cleared, drawn) =
                    self.terrain_passes(gpu, &drape, extrusions_ready, &mut passes);
                stats.draws += drawn.draws;
                stats.drape_slots = drawn.slots;
                drape_passes = micros(passing, clock());
                (above, cleared)
            }
            false => (Vec::new(), false),
        };
        let placings = (self.placings(), self.full);
        passes.extend(self.symbols.pass(gpu, placings, self.scene.placed_quads()));
        stats.draws += self.extrusion_passes(
            gpu,
            extrusions_ready,
            (terrain_above, depth_cleared),
            &mut passes,
        );
        let start = passes.len();
        passes.extend(self.layers.passes());
        Pass::group(&mut passes, start, "layers apart");
        self.main_passes(gpu, (target, clear), terrain_on, &mut passes);
        // What the frame wrote into data textures lands before it draws.
        #[cfg(cpu_driven)]
        self.scene.tables_mut().flush(gpu);
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

    /// Follows the cull's candidates as laid out again: a node the bundles
    /// left out for want of candidates draws once it has some, and a
    /// heatmap hidden, or with nothing to draw, composites nothing.
    fn follow_candidates(&mut self, gpu: &mut dyn Gpu, candidates_rebuilt: bool) {
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
    }

    /// Settles the pipelines the frame draws with: the variant it needs,
    /// and whatever waited for those that arrived recorded again.
    fn settle_pipelines(&mut self, gpu: &mut dyn Gpu, frame: &FrameBlock) -> PipelineCounts {
        // The bundles draw with the variants the frame needs.
        let full = self.needs_full(frame);
        debug_assert!(
            !full || self.variants()[1],
            "a frame needs the full projection, which the map didn't declare"
        );
        if full != self.full {
            self.full = full;
            self.invalidate_bundle(gpu);
        }
        if std::mem::take(&mut self.pipelines_stale) {
            self.show_nodes(gpu);
        }
        let compiling = gpu.poll_pipelines();
        if self.extrusions.settle(gpu) {
            for node in &mut self.nodes {
                if let Some(extrusion) = &mut node.extrusion {
                    extrusion.forget_bundle(gpu);
                }
            }
        }
        if compiling.made != self.made_seen {
            // Pipelines arrived: record again what waited for them.
            self.made_seen = compiling.made;
            let drape_waiting = std::mem::take(&mut self.drape_waiting);
            if self.waiting > 0 || drape_waiting {
                self.invalidate_bundle(gpu);
            }
            if drape_waiting && let Some(terrain) = self.terrain.as_mut() {
                // Slots drawn without them lack those layers: all draw
                // again, the drape's bundles recorded anew.
                terrain.redraw_all();
            }
        }
        if compiling.pending > 0 {
            self.prioritize(gpu);
        }
        compiling
    }

    /// Sizes the frame's targets to its viewport: the 3D layers', the
    /// terrain's, the heatmaps' densities. The bundles record again for
    /// targets made anew. Returns the frame's size in pixels.
    fn resize_targets(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        terrain_on: bool,
    ) -> [u32; 2] {
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
        size
    }

    /// Records the main pass's bundles and the composited layers' own,
    /// for a frame of `size` pixels.
    fn record_bundles(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], terrain_on: bool) {
        let mut ready = self.ready(gpu, terrain_on);
        let mut commands = Vec::with_capacity(self.nodes.len() * 5);
        // Runs of one layer's nodes below layer opacity 1 or blending
        // other than normally draw into a texture of their own; draped
        // nodes take the terrain's way.
        // On terrain, a draped layer's group is its terrain layer: the
        // texture it composites.
        // A hillshade its terrain layer blends composites with it.
        let mut runs: Vec<layer_composite::Run> = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let Some(layer) = composite_group(node.composite).filter(|_| !node.blends_on_terrain)
            else {
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
            let mut group = Vec::new();
            for i in self.layers.groups[g].nodes.clone() {
                let groups = (self.draw_group, self.nodes[i].group);
                let before = group.len();
                if !self.node_commands(gpu, ready, i, groups, &mut group) {
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
                &bound_once(group),
            ));
        }
        let layers = RenderBundleDesc {
            label: "layers",
            color_formats: &[self.target.color],
            depth_format: None,
            sample_count: self.target.sample_count,
        };
        for i in 0..=self.nodes.len() {
            // The solid background and what it hides go in a bundle
            // of their own, which the clear can stand for.
            if self.beneath.is_some_and(|b| i == b + 1) {
                let segment = bound_once(std::mem::take(&mut commands));
                self.bundles
                    .push(gpu.create_render_bundle(&layers, &segment));
            }
            if i == self.nodes.len() {
                break;
            }
            let (start, member) = self.layers.at(i);
            match start {
                // The main pass splits here: its blend draw opens the
                // next pass, before the next bundle.
                Some(group) if group.blend != Blend::Normal => {
                    let segment = bound_once(std::mem::take(&mut commands));
                    self.bundles
                        .push(gpu.create_render_bundle(&layers, &segment));
                    self.recorded += 1;
                }
                Some(group) => {
                    commands.extend(self.layers.composite_commands(&self.composites, group));
                    self.recorded += 1;
                }
                None => {}
            }
            let groups = (self.draw_group, self.nodes[i].group);
            let before = commands.len();
            if !member && !self.node_commands(gpu, ready, i, groups, &mut commands) {
                self.waiting += 1;
            }
            self.recorded += u32::from(commands.len() > before);
        }
        self.bundles
            .push(gpu.create_render_bundle(&layers, &bound_once(commands)));
    }

    /// The DEM prepare, image mip and cull passes and the heatmaps'
    /// density passes; whether the cull is ready to draw from.
    fn cull_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        passes: &mut Vec<Pass>,
        stats: &mut FrameStats,
    ) -> bool {
        if let Some((prepare, prepared)) = self.images.prepare_pass(gpu) {
            let start = passes.len();
            stats.draws += prepared.draws;
            stats.prepared = prepared.tiles;
            passes.extend(prepare);
            Pass::group(passes, start, "dem prepare");
        }
        if let Some((mips, dispatches)) = self.images.mip_pass(gpu, &self.downsampler) {
            let start = passes.len();
            stats.dispatches += dispatches;
            passes.extend(mips);
            Pass::group(passes, start, "image mips");
        }
        let cull_ready = self.driver.ready(gpu);
        let dispatches = self.driver.cull(gpu, &self.scene, passes);
        if self.driver.culled(dispatches) {
            self.scene.culled();
        }
        stats.dispatches += dispatches;
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
                    self.driver
                        .draw(self.scene.buffers(), index as u32, NodeDraw::Points),
                )
            {
                stats.draws += 1;
                passes.push(Pass::Render(pass));
            }
        }
        Pass::group(passes, start, "heatmaps");
        cull_ready
    }

    /// The terrain's passes: its drape slots (`drape`'s bundles) and its
    /// layers. Those above a 3D layer are returned by their first node,
    /// to draw after it; the rest join `passes`. Also returns whether a
    /// pass cleared the depth the 3D passes share, and the draws.
    fn terrain_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        drape: &[(u32, RenderBundleId)],
        extrusions_ready: bool,
        passes: &mut Vec<Pass>,
    ) -> (Vec<(usize, RenderPass)>, bool, terrain::TerrainDrawn) {
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let mut terrain_above: Vec<(usize, RenderPass)> = Vec::new();
        let mut depth_cleared = false;
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
        let mut layers = Vec::with_capacity(self.terrain_layers.len());
        let drawn = terrain.passes(
            gpu,
            (b.frame, self.scene.tables().entry(Table::Props, 8)),
            (drape, &shades),
            depth,
            &self.downsampler,
            (passes, &mut layers),
        );
        // A terrain layer draws after the 3D layers below it, over
        // their depth, so that buildings hide what they stand in front
        // of: draped layers above a fill-extrusion layer, and their
        // fog, would otherwise composite over its buildings. Those
        // below every 3D layer draw here.
        let first_3d =
            (self.nodes.iter()).position(|n| n.extrusion.is_some() && n.drawn && extrusions_ready);
        for (layer, pass) in layers.into_iter().enumerate() {
            let at = (self.nodes.iter())
                .position(|n| n.terrain_layer == Some(layer as u8))
                .unwrap_or(0);
            match first_3d {
                Some(first) if first < at => terrain_above.push((at, pass)),
                _ => {
                    depth_cleared = true;
                    passes.push(Pass::Render(pass));
                }
            }
        }
        Pass::group(passes, start, "terrain");
        (terrain_above, depth_cleared, drawn)
    }

    /// The 3D layers' passes, in order with the terrain layers above them
    /// (`terrain_above`, after a pass cleared their shared depth when
    /// `depth_cleared`); returns their draws.
    fn extrusion_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        extrusions_ready: bool,
        (terrain_above, mut depth_cleared): (Vec<(usize, RenderPass)>, bool),
        passes: &mut Vec<Pass>,
    ) -> u32 {
        let mut draws = 0;
        let (draw_group, buffers) = (self.draw_group, self.driver.buffers(self.scene.buffers()));
        let start = passes.len();
        let mut terrain_above = terrain_above.into_iter().peekable();
        // The first pass on the shared depth clears it, the rest load it.
        fn push_terrain(passes: &mut Vec<Pass>, mut pass: RenderPass, cleared: &mut bool) {
            if let Some(depth) = &mut pass.depth {
                depth.load = if *cleared {
                    LoadOp::Load
                } else {
                    LoadOp::Clear(0.0)
                };
            }
            *cleared = true;
            passes.push(Pass::Render(pass));
        }
        for (index, node) in self.nodes.iter_mut().enumerate() {
            let Some(target) = node
                .extrusion
                .as_mut()
                .filter(|_| extrusions_ready && node.drawn)
            else {
                continue;
            };
            while let Some((_, pass)) = terrain_above.next_if(|(at, _)| *at < index) {
                push_terrain(passes, pass, &mut depth_cleared);
            }
            let first = !depth_cleared;
            depth_cleared = true;
            let draw = |row| self.driver.draw(&buffers, row, NodeDraw::Indexed);
            let cut = self.scene.cut_row(index as u32).zip(node.cut_group);
            let (cut_row, cut_group) = cut.expect("an extrusion node draws a cut stream");
            let streams = extrusion::Streams {
                whole: (node.group, draw(index as u32)),
                cut: (cut_group, draw(cut_row)),
            };
            let pass = self.extrusions.prepare(
                gpu,
                target,
                draw_group,
                &buffers,
                (&streams, node.elevation),
                first,
            );
            passes.push(Pass::Render(pass));
            draws += 1;
        }
        for (_, pass) in terrain_above {
            push_terrain(passes, pass, &mut depth_cleared);
        }
        Pass::group(passes, start, "3d layers");
        draws
    }

    /// The main pass into `target`, cleared to `clear`: the sky, the
    /// bundles and the composited layers.
    fn main_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (target, mut clear): (TextureId, [f64; 4]),
        terrain_on: bool,
        passes: &mut Vec<Pass>,
    ) {
        // A solid background filling the frame is the clear: its bundle
        // and the sky drawn before it, which it hides, are left out.
        let (mut first, mut segments) = (Vec::new(), &self.bundles[..]);
        if let Some(b) = self.beneath {
            let (below, above) = segments.split_first().expect("recorded above");
            segments = above;
            let shown = self.nodes[b].cull.visible && self.scene.node_candidates(b as u32) > 0;
            match self.nodes[b].solid {
                Some(color) if shown && self.background_fills && !terrain_on => {
                    clear = color.map(f64::from);
                }
                _ => {
                    first = self.atmosphere.commands(gpu);
                    first.push(RenderCommand::ExecuteBundles(vec![*below]));
                }
            }
        } else {
            first = self.atmosphere.commands(gpu);
        }
        let start = passes.len();
        passes.extend(self.layers.main_passes(
            &self.composites,
            (first, segments, self.atmosphere.commands_after(gpu)),
            target,
            clear,
        ));
        Pass::group(passes, start, "main");
    }

    /// What a bundle record finds compiled besides the nodes' own
    /// pipelines. Those only a family needs are asked after once a node of
    /// it drew: made ahead of need ([`set_pipelines`](Self::set_pipelines)), they
    /// compile in the background until then.
    fn ready(&mut self, gpu: &mut dyn Gpu, terrain_on: bool) -> Ready {
        let drawn = |families: &[Family]| {
            (self.nodes.iter()).any(|n| n.drawn && families.contains(&n.family))
        };
        let (rasters, dems) = (
            drawn(&[Family::Raster]),
            drawn(&[Family::Hillshade, Family::ColorRelief]),
        );
        Ready {
            cull: self.driver.ready(gpu),
            terrain: self
                .terrain
                .as_mut()
                .filter(|_| terrain_on)
                .is_some_and(|t| t.ready(gpu))
                && self.composites.ready(gpu, offscreen::Mode::Over),
            prepare: !dems || self.images.prepare_ready(gpu),
            mips: !rasters || self.downsampler.ready(gpu),
        }
    }

    /// The drape pass into stack `stack` for this frame: its page's frame
    /// written, and its bundle of the draped nodes drawn into it, recorded
    /// if it isn't. `None` when no node's drape paint was set.
    fn drape_bundle(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        stack: u32,
    ) -> Option<RenderBundleId> {
        let terrain = self.terrain.as_ref()?;
        let colour = self.drape_stacks[0];
        let page = usize::from(stack >= colour);
        let drape_frame = terrain.drape_frame(frame, page);
        let level = self.drape.as_mut()?;
        crate::scene::write_frame(gpu, level.frame[page], &drape_frame);
        let at = stack as usize;
        if level.bundle.len() <= at {
            level.bundle.resize_with(at + 1, || None);
        }
        if level.scene[page].is_none() {
            level.scene[page] = Some(scene_group(
                gpu,
                self.draw_layout,
                self.scene.tables(),
                level.frame[page],
                &self.image_bindings,
                // Draped content isn't hidden by the terrain it's on.
                (
                    self.glyphs.texture(),
                    self.images.nearest(),
                    self.dem_bound,
                    self.no_terrain_depth,
                ),
            ));
            // The bundles of the page's stacks bound the group it replaces.
            for (s, bundle) in level.bundle.iter_mut().enumerate() {
                if usize::from(s as u32 >= colour) == page {
                    bundle.take().release(gpu);
                }
            }
        }
        if let Some((bundle, _)) = &level.bundle[at] {
            return Some(*bundle);
        }
        let ready = self.ready(gpu, true);
        let level = self.drape.as_ref().expect("found above");
        let (scene, uniforms) = (level.scene[page].expect("made above"), level.uniforms);
        let mut commands = Vec::new();
        let mut groups = Vec::new();
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let in_stack = node.stack.is_some_and(|s| u32::from(s) == stack);
            if !in_stack || !level.written.get(i).is_some_and(Option::is_some) {
                continue;
            }
            let uniform = BindingResource::Buffer {
                buffer: uniforms,
                offset: i as u64 * NODE_UNIFORM_SIZE,
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
            if !self.node_commands(gpu, ready, i, (scene, group), &mut commands) {
                self.drape_waiting = true;
            }
        }
        let bundle = gpu.create_render_bundle(
            &RenderBundleDesc {
                label: "drape",
                color_formats: &[DRAPE_FORMAT],
                depth_format: None,
                sample_count: 1,
            },
            &bound_once(commands),
        );
        let level = self.drape.as_mut().expect("found above");
        level.bundle[at] = Some((bundle, groups));
        Some(bundle)
    }

    /// Node `i`'s commands as a bundle records them, over the scene group
    /// `scene`: the map's, or the drape's for draped nodes on terrain. In
    /// the map, a terrain layer composites in its first node's place.
    /// `false` when the node waits for a pipeline and records nothing.
    fn node_commands(
        &self,
        gpu: &mut dyn Gpu,
        ready: Ready,
        i: usize,
        (scene, group): (BindGroupId, BindGroupId),
        commands: &mut Vec<RenderCommand>,
    ) -> bool {
        if !ready.cull {
            return false;
        }
        let node = &self.nodes[i];
        let draw = self
            .driver
            .draw(self.scene.buffers(), i as u32, node.cull.draw);
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
        // A direct draw of nothing records nothing: the bundles record
        // again when the node's draw changes.
        if self.driver.draws_nothing(i as u32) {
            return true;
        }
        if node.family.is_symbol() {
            // Its icons and its text, in one list. The cull's point
            // arguments read as indexed ones too: 6 indices into the quad
            // indices, and the first index, base vertex and first instance
            // all 0, as points always have them.
            let quads = (self.driver).draw(self.scene.buffers(), i as u32, NodeDraw::Indexed);
            let apart = [node.separate_halo, self.nodes[i + 1].separate_halo];
            let (kind, placings) = self.symbol_kinds(i);
            return self.symbols.commands(
                gpu,
                (kind, placings, self.full),
                (scene, group),
                apart,
                quads,
                commands,
            );
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
            .filter(|_| scene != self.draw_group && self.target.color != DRAPE_FORMAT);
        let pipeline = match (drape, node.family.is_image()) {
            _ if node.shades_on_terrain() => self.images.slopes_pipeline(),
            (Some(d), true) => d.images.get(node.family),
            (Some(d), false) => d.paint.drawn(gpu, node.paint_kind(), self.full),
            (None, true) => self.images.pipeline(node.family),
            (None, false) => self.paint.drawn(gpu, node.paint_kind(), self.full),
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
        ]);
        if node.cull.draw == NodeDraw::Indexed {
            commands.push(RenderCommand::SetIndexBuffer {
                buffer: self.driver.buffers(self.scene.buffers()).compacted,
                offset: 0,
            });
        }
        commands.push(draw);
        true
    }
}

/// What puts a node in a drape stack ([`Renderer::assign_stacks`]).
#[derive(Clone, Copy, Debug, Default)]
struct StackInput {
    /// Terrain is on and the node's family drapes.
    draped: bool,
    /// The layer it composites with, if its layer composites once.
    group: Option<u32>,
    /// How that layer blends.
    blend: Blend,
    /// A hillshade the terrain lights between stacks.
    shaded: bool,
    /// What it draws changes with the zoom.
    by_zoom: bool,
}

/// The nodes' drape stacks and terrain layers.
#[derive(Debug, PartialEq, Eq)]
struct DrapeStacks {
    /// Per node, its stack and terrain layer.
    nodes: Vec<(Option<u8>, Option<u8>)>,
    layers: Vec<terrain::TerrainLayer>,
    /// Colour stacks, then slope stacks.
    counts: [u32; 2],
}

/// The drape stacks and terrain layers of nodes `inputs`, as
/// [`Renderer::assign_stacks`] tells.
fn drape_stacks(inputs: &[StackInput]) -> DrapeStacks {
    let mut stacks = 0u8;
    let mut layers: Vec<terrain::TerrainLayer> = Vec::new();
    let mut nodes: Vec<(Option<u8>, Option<u8>)> = Vec::with_capacity(inputs.len());
    let mut previous: Option<Option<u32>> = None;
    for (i, node) in inputs.iter().enumerate() {
        let draped = node.draped;
        let shaded = draped && node.shaded;
        // A shaded hillshade blending other than normally blends into the
        // open terrain layer's colours beneath it, which are what it lies
        // on, where they draw normally and the layer shades no other: it
        // joins the layer, which goes on above it. Elsewhere its layer
        // composites apart, a terrain layer of its own.
        let blends = shaded && node.blend != Blend::Normal && previous == Some(None);
        let joins = blends
            && layers
                .last()
                .is_some_and(|l| l.shade.is_none() && l.below.is_some());
        // Where that layer shades another, it starts the next, whose
        // hillshade blends into what the layer before drew.
        let backdrop = blends && layers.last().is_some_and(|l| l.shade.is_some());
        let inline = joins || backdrop;
        let group = if inline { None } else { node.group };
        let open = layers
            .last()
            .filter(|l| draped && previous == Some(group) && !(shaded && l.shade.is_some()));
        if draped && open.is_none() {
            layers.push(terrain::TerrainLayer::default());
        }
        previous = draped.then_some(group);
        let terrain_layer = draped.then(|| layers.len() as u8 - 1);
        let mut stack = None;
        if let Some(layer) = layers.last_mut().filter(|_| draped) {
            let mut next = || {
                stacks = stacks.saturating_add(1);
                stacks - 1
            };
            stack = Some(if shaded {
                let slopes = next();
                layer.shade = Some((i as u32, slopes));
                if inline {
                    layer.shade_blend = node.blend;
                    layer.backdrop = backdrop;
                }
                slopes
            } else {
                let part = match layer.shade {
                    Some(_) => &mut layer.above,
                    None => &mut layer.below,
                };
                *part.get_or_insert_with(next)
            });
        }
        nodes.push((stack, terrain_layer));
    }
    // A layer without a shade whose zoom-free nodes come first splits
    // above them: the stack below keeps its slots while the camera
    // zooms.
    for (index, layer) in layers.iter_mut().enumerate() {
        if layer.shade.is_some() {
            continue;
        }
        let members: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].1 == Some(index as u8))
            .collect();
        let Some(split) =
            (members.iter().position(|&i| inputs[i].by_zoom)).filter(|&split| split > 0)
        else {
            continue;
        };
        stacks = stacks.saturating_add(1);
        layer.above = Some(stacks - 1);
        for &i in &members[split..] {
            nodes[i].0 = layer.above;
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
    for node in &mut nodes {
        node.0 = node.0.map(renumber);
    }
    DrapeStacks {
        nodes,
        layers,
        counts: [colour, slopes.len() as u32],
    }
}

/// A bundle's `commands` without the index buffer bindings that bind what
/// is bound already: each indexed draw binds its own (the compacted
/// indices, or symbols' [`QUAD_INDICES`]), and runs of one kind need it
/// bound once.
fn bound_once(mut commands: Vec<RenderCommand>) -> Vec<RenderCommand> {
    let mut bound = None;
    commands.retain(|c| match c {
        &RenderCommand::SetIndexBuffer { buffer, offset } => {
            bound.replace((buffer, offset)) != Some((buffer, offset))
        }
        _ => true,
    });
    commands
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

/// The drape pass: draped nodes' paint for the slots it redraws, and per
/// page its frame, scene group and bundle.
#[derive(Debug)]
struct Drape {
    /// The drape frame, per drape page.
    frame: [BufferId; 2],
    /// Node uniforms, at each node's offset.
    uniforms: BufferId,
    /// Per node, the words written, or `None` where it doesn't draw (outside
    /// its layer's zooms, or not draped).
    written: Vec<Option<Vec<u32>>>,
    /// Per page, the scene group over its `frame`.
    scene: [Option<BindGroupId>; 2],
    /// Per stack, the bundle of the nodes drawing into it, with the node
    /// groups it binds: a redraw replays each over its own slots.
    bundle: Vec<Option<(RenderBundleId, Vec<BindGroupId>)>>,
}

impl release::Release for Node {
    fn release(self, gpu: &mut dyn Gpu) {
        let groups = (self.group, self.cut_group);
        (groups, self.image, self.heatmap, self.extrusion).release(gpu);
    }
}

impl release::Release for Drape {
    fn release(self, gpu: &mut dyn Gpu) {
        Drape::release(self, gpu);
    }
}

impl Drape {
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

/// A paint node's pipeline ([`Node::paint_kind`]).
#[derive(Clone, Copy, Debug)]
struct PaintKind {
    family: Family,
    /// The styled line's.
    styled: bool,
    /// Its variant with property rows ([`orb_shaders::FEATURE_ROWS`]).
    rows: bool,
}

/// The vector families' pipelines (fill, fill outline, line, circle), each
/// made for the family's first node, with property rows and without.
#[derive(Clone, Copy, Debug, Default)]
struct PaintPipelines {
    pipelines: [Variants; 10],
}

impl PaintPipelines {
    /// The slot of a node's [`Node::paint_kind`].
    fn index(kind: PaintKind) -> Option<usize> {
        let shader = match kind.family {
            Family::Fill => 0,
            Family::FillOutline => 1,
            Family::Line if kind.styled => 4,
            Family::Line => 2,
            Family::Circle => 3,
            _ => return None,
        };
        Some(2 * shader + usize::from(kind.rows))
    }

    /// What a frame draws a node of `kind` with ([`Variants::drawn`]).
    fn drawn(&self, gpu: &mut dyn Gpu, kind: PaintKind, full: bool) -> Option<RenderPipelineId> {
        self.pipelines[Self::index(kind)?].drawn(gpu, full)
    }

    /// What a frame waits for to draw a node of `kind`
    /// ([`Variants::wanted`]).
    fn wanted(&self, kind: PaintKind, full: bool) -> Option<RenderPipelineId> {
        self.pipelines[Self::index(kind)?].wanted(full)
    }

    /// Makes the pipeline of a node of `kind` in each of the variants
    /// `[plane, full]` asked for.
    fn make(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        layouts: &[BindGroupLayoutId],
        kind: PaintKind,
        variants: [bool; 2],
    ) {
        let Some(index) = Self::index(kind) else {
            return;
        };
        let (shader, topology) = match index / 2 {
            0 => (&FILL_SHADER, PrimitiveTopology::TriangleList),
            1 => (&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            2 => (&LINE_SHADER, PrimitiveTopology::TriangleList),
            4 => (&LINE_STYLED_SHADER, PrimitiveTopology::TriangleList),
            _ => (&CIRCLE_SHADER, PrimitiveTopology::TriangleList),
        };
        let draw = Draw {
            topology,
            rows: kind.rows,
            ..Draw::new(
                shader,
                layouts,
                target,
                Some(BlendState::PremultipliedAlpha),
            )
        };
        self.pipelines[index].make(gpu, variants, draw);
    }
}

/// The scene's group: its buffers, the image pages and the glyph pages.
fn scene_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    tables: &Tables,
    frame: BufferId,
    images: &ImageBindings,
    (glyphs, nearest, dem, terrain_depth): (TextureId, SamplerId, TextureId, TextureId),
) -> BindGroupId {
    use TextureViewDimension::{D2, D2Array};
    let mut entries = vec![BindGroupEntry::buffer(0, frame)];
    entries.extend(tables.entries());
    entries.push(images.table.entry(8));
    entries.extend([
        BindGroupEntry::texture(9, images.texture, D2Array),
        BindGroupEntry::sampler(10, images.sampler),
        BindGroupEntry::texture(11, images.dashes, D2),
        BindGroupEntry::texture(12, glyphs, D2Array),
        BindGroupEntry::sampler(13, nearest),
        BindGroupEntry::texture(14, dem, D2Array),
        BindGroupEntry::texture(15, terrain_depth, D2),
        BindGroupEntry::texture(16, images.texture, D2Array),
    ]);
    gpu.create_bind_group(&BindGroupDesc {
        label: "scene",
        layout,
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain::TerrainLayer;

    /// Drape stacks by case: per node, whether it drapes, its composite
    /// group, whether it's a shaded hillshade and whether it draws by zoom;
    /// then each node's stack and terrain layer, the layers and the counts.
    #[test]
    fn drape_stacks_by_case() {
        let node = |draped, group, shaded, by_zoom| StackInput {
            draped,
            group,
            blend: match group {
                Some(_) => Blend::Multiply,
                None => Blend::Normal,
            },
            shaded,
            by_zoom,
        };
        let plain = node(true, None, false, false);
        let layer = |below, shade, above| TerrainLayer {
            below,
            shade,
            above,
            blended: false,
            shade_blend: Blend::Normal,
            backdrop: false,
        };
        type Case = (&'static str, Vec<StackInput>, DrapeStacks);
        let cases: Vec<Case> = vec![
            (
                "terrain off",
                vec![node(false, None, false, false); 2],
                DrapeStacks {
                    nodes: vec![(None, None); 2],
                    layers: vec![],
                    counts: [0, 0],
                },
            ),
            (
                "a run is one stack",
                vec![plain, plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)); 2],
                    layers: vec![layer(Some(0), None, None)],
                    counts: [1, 0],
                },
            ),
            (
                "a node that doesn't drape ends the run",
                vec![plain, node(false, None, false, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (None, None), (Some(1), Some(1))],
                    layers: vec![layer(Some(0), None, None), layer(Some(1), None, None)],
                    counts: [2, 0],
                },
            ),
            (
                "a layer that composites once has a terrain layer of its own",
                vec![plain, node(true, Some(5), false, false)],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(1))],
                    layers: vec![layer(Some(0), None, None), layer(Some(1), None, None)],
                    counts: [2, 0],
                },
            ),
            (
                "a shaded hillshade's slopes stack after the colour stacks",
                vec![plain, node(true, None, true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(2), Some(0)), (Some(1), Some(0))],
                    layers: vec![layer(Some(0), Some((1, 2)), Some(1))],
                    counts: [2, 1],
                },
            ),
            (
                "a second shaded hillshade starts a terrain layer",
                vec![node(true, None, true, false), node(true, None, true, false)],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(1))],
                    layers: vec![
                        layer(None, Some((0, 0)), None),
                        layer(None, Some((1, 1)), None),
                    ],
                    counts: [0, 2],
                },
            ),
            (
                "a blended shaded hillshade blends into the layer it lies on",
                vec![plain, node(true, Some(5), true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(2), Some(0)), (Some(1), Some(0))],
                    layers: vec![TerrainLayer {
                        shade_blend: Blend::Multiply,
                        ..layer(Some(0), Some((1, 2)), Some(1))
                    }],
                    counts: [2, 1],
                },
            ),
            (
                "after another shaded hillshade, it blends into its layer",
                vec![
                    plain,
                    node(true, None, true, false),
                    node(true, Some(5), true, false),
                    plain,
                ],
                DrapeStacks {
                    nodes: vec![
                        (Some(0), Some(0)),
                        (Some(2), Some(0)),
                        (Some(3), Some(1)),
                        (Some(1), Some(1)),
                    ],
                    layers: vec![
                        layer(Some(0), Some((1, 2)), None),
                        TerrainLayer {
                            shade_blend: Blend::Multiply,
                            backdrop: true,
                            ..layer(None, Some((2, 3)), Some(1))
                        },
                    ],
                    counts: [2, 2],
                },
            ),
            (
                "with nothing beneath it, it composites apart",
                vec![node(true, Some(5), true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(1), Some(0)), (Some(0), Some(1))],
                    layers: vec![layer(None, Some((0, 1)), None), layer(Some(0), None, None)],
                    counts: [1, 1],
                },
            ),
            (
                "zoom-free nodes first split from those drawn by zoom",
                vec![plain, node(true, None, false, true), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(0)), (Some(1), Some(0))],
                    layers: vec![layer(Some(0), None, Some(1))],
                    counts: [2, 0],
                },
            ),
            (
                "nodes drawn by zoom first don't split",
                vec![node(true, None, false, true), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)); 2],
                    layers: vec![layer(Some(0), None, None)],
                    counts: [1, 0],
                },
            ),
        ];
        for (name, inputs, expected) in cases {
            assert_eq!(drape_stacks(&inputs), expected, "{name}");
        }
    }
}
