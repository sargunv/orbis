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
    DRAPE_FORMAT, DRAPE_SLOT_PX, DrapeFull, DrapeLayout, SHEET_SLOTS, TERRAIN_EXTENT, TERRAIN_GRID,
    TerrainDem, TerrainGlobe, TerrainPaint, TerrainTile,
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
    TileRow, TileWarp, Untested, Words, warp_words,
};
pub use scene::{
    GLYPH_PADDING, GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas, PLACED_WORDS, QUAD_WORDS,
    SYMBOL_HEADER_WORDS, SYMBOL_STATE_WORDS, SymbolMeshData, pack_image_rect, symbol_header,
    symbol_state,
};
pub use scene::{PoolMemory, Resized, SceneMemory};
pub use tiles::{
    GpuTile, ImageBuild, ImageMesh, MeshBuild, MeshRows, Reserved, TileBuild, TileKey, UploadError,
    WarpKey, changed_borders, dem_neighbours, tile_extents, tile_latitudes, tile_size,
};

use crate::pipeline::{Attachments, Draw, Variants};
use crate::release::Release;

#[cfg(cpu_driven)]
#[doc(hidden)]
pub use driver::cpu::{Compared, DriverCheck};
use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_PLANES_SHADER, LINE_SHADER,
    LINE_STYLED_PLANES_SHADER, LINE_STYLED_SHADER,
};
#[cfg(cpu_driven)]
use scene::Backing;
use scene::{DataTable, Scene, Table, Tables};

/// Whether a family's nodes draw in two rows by their placements' clips
/// (`scene`'s `Stream::Untested`): fills, lines and image tiles, on a
/// device that makes their untested pipelines ([`untested_variants`]).
fn splits(family: Family, untested: bool) -> bool {
    untested && (matches!(family, Family::Fill | Family::Line) || family.is_image())
}

/// Whether the device draws placements that need no clip test in
/// pipelines that test none ([`Untested`]): not one whose compiles hold
/// its presents (Firefox's WebGPU), which compiles a style's whole set
/// before its first frame. There Liberty's first view compiled cold in
/// 2.10 s, and 2.86 s with the untested pipelines, for 0.1–0.4 ms of GPU
/// time a frame at 5120×2880; its paint draws with rows for fewer
/// compiles too ([`Renderer::rowed`]).
fn untested_variants(caps: &Caps) -> bool {
    !caps.compiles_hold_presents
}

/// Which of a split node's placements draw untested with `paint`, on a
/// device with clip distances or not (`planes`).
fn untested_of(paint: &NodePaint, planes: bool) -> Untested {
    match paint {
        _ if paint.translates() => Untested::None,
        NodePaint::Fill(_)
        | NodePaint::Background(_)
        | NodePaint::FillPattern(..)
        | NodePaint::Raster(_)
        | NodePaint::Hillshade(_)
        | NodePaint::ColorRelief(_) => Untested::Square,
        NodePaint::Line(_) | NodePaint::LinePattern(..) if planes => Untested::Rect,
        _ => Untested::None,
    }
}

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
    /// Bytes of compacted index slots the frame's candidate layout moved
    /// (natively through the scratch buffer, so copied twice).
    pub slot_bytes_moved: u64,
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
    /// The group of an extrusion node's cut stream, or a split node's
    /// tested placements ([`Scene::second_row`]): its uniform at that
    /// row, the node's paint with the row's index.
    second_group: Option<BindGroupId>,
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
    /// A fill or line node draws a pattern: it draws with its pipeline's
    /// variant with patterns.
    pattern: bool,
    /// What an extrusion node's pipelines are specialised for.
    elevation: paint::ExtrusionShape,
    /// On terrain, the drape stack a draped node draws into, and the
    /// terrain layer that draws it.
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
    /// A bundle recorded since the last invalidation left out its row, or
    /// its second row: it had no candidates, so its draw would have drawn
    /// nothing. Candidates arriving for it record the bundles again.
    left_out: std::cell::Cell<[bool; 2]>,
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
    pattern: bool,
    along_line: Option<bool>,
    elevation: Option<paint::ExtrusionShape>,
}

impl PaintShape {
    fn of(paint: &NodePaint) -> Self {
        Self {
            styled: paint.styled(),
            rows: paint.reads_rows(),
            pattern: paint.patterned(),
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

    /// Its paint pipeline, `patterned` where another node of its family
    /// draws a pattern ([`Renderer::patterned`]), with rows where another
    /// reads them and `rowed` says so ([`Renderer::rowed`]).
    fn paint_kind(&self, patterned: &[Family], rowed: &[Family]) -> PaintKind {
        PaintKind {
            family: self.family,
            styled: self.styled,
            rows: self.rows || rowed.contains(&self.family),
            pattern: self.pattern || patterned.contains(&self.family),
        }
    }

    /// What of it picks its pipelines, which [`Renderer::set_pipelines`]
    /// sets from a paint it may draw with and puts back.
    fn shape(&self) -> NodeShape {
        (
            self.styled,
            self.rows,
            self.pattern,
            self.along_line,
            self.elevation,
        )
    }

    /// Its pipelines' shape as `paint` picks them.
    fn set_shape(&mut self, paint: &NodePaint) {
        (self.styled, self.rows) = (paint.styled(), paint.reads_rows());
        self.pattern = paint.patterned();
        match paint {
            NodePaint::Symbol(p) => self.along_line = p.along_line,
            NodePaint::Extrusion(_) | NodePaint::ExtrusionPattern(..) => {
                self.elevation = paint.extrusion_shape();
            }
            _ => {}
        }
    }

    fn restore_shape(&mut self, shape: NodeShape) {
        (
            self.styled,
            self.rows,
            self.pattern,
            self.along_line,
            self.elevation,
        ) = shape;
    }
}

/// [`Node::shape`]: styled, rows, pattern, along lines and elevation.
type NodeShape = (bool, bool, bool, bool, paint::ExtrusionShape);

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
    /// How soon a view may need it: 0 where it draws in the view the set
    /// was made for (its pipelines compile first, and frames wait for
    /// them), else the place of the nearest zoom it draws at among the
    /// whole zooms nearest that view's. Background compiles go by it, so
    /// what a zoom reaches next compiles next.
    pub rank: u32,
}

/// Every pipeline a style can reach on a renderer
/// ([`Renderer::set_pipelines`]): its nodes' by what draws together, and
/// the sky's.
#[derive(Clone, Debug, Default)]
pub struct PipelineSet {
    pub groups: Vec<PipelineGroup>,
    /// The style has a sky: its pipeline, and the globe's scattering
    /// where views may be on a low-zoom surface.
    pub sky: bool,
    /// A hillshade may shade on terrain at some zoom: its slopes' and the
    /// shaded terrain's pipelines.
    pub shaded: bool,
}

#[derive(Debug)]
pub struct Renderer {
    id: u64,
    /// The host's target, and whether the map's passes have a depth
    /// attachment: while the style has terrain, which draws in them.
    target: Attachments,
    /// That depth attachment, made for the frame's size: never stored.
    map_depth: Option<(TextureId, [u32; 2])>,
    /// Whether 3D surfaces draw at four samples
    /// ([`set_antialias_3d`](Self::set_antialias_3d)).
    antialias_3d: bool,
    /// What it draws otherwise than on WebGPU.
    gaps: Vec<Gap>,
    atmosphere: atmosphere::Atmosphere,
    scene: Scene,
    /// The scene's groups, by block slot: each block's frame and pools
    /// with the tables they share; `None` for a free slot.
    draw_groups: Vec<Option<BindGroupId>>,
    /// How the frame's draws learn their counts: on the GPU, its cull.
    driver: driver::Driver,
    draw_layout: BindGroupLayoutId,
    node_layout: BindGroupLayoutId,
    /// The vector families' pipelines, each made for the family's first
    /// node.
    paint: PaintPipelines,
    /// The vector families' pipelines for the map's other kind of pass,
    /// with or without its depth ([`fit_depth`](Self::fit_depth)).
    paint_kept: PaintPipelines,
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
    pipeline_set: PipelineSet,
    /// The paint families a node of the set draws a pattern in: each of
    /// their nodes draws with the variant with patterns. Its uniform
    /// branch costs a pattern-less node nothing that shows, and a style
    /// with both kinds then compiles one pipeline, not two.
    patterned: Vec<Family>,
    /// Where compiles hold presents ([`Caps::compiles_hold_presents`]),
    /// the families a node of the set reads property rows in: each of
    /// their nodes draws with the variant with rows, one pipeline, not
    /// two, as every compile there is a moment the map shows nothing.
    rowed: Vec<Family>,
    /// The placement kind of the set's symbol layers at all their zooms
    /// ([`placings`](Self::placings)).
    set_placings: symbol::Placings,
    /// The set is to be asked for again, in what the map draws it with
    /// now: other variants, another sample count, or into the drape.
    set_stale: bool,
    /// Pipelines the set asked for since [`take_set_asked`](Self::take_set_asked).
    set_asked: u32,
    /// Pipelines the device had compiled when the bundles were recorded.
    made_seen: u32,
    /// Nodes the main bundle left out, waiting for a pipeline.
    waiting: u32,
    /// Whether a drape bundle has left out a draped node waiting for a
    /// pipeline, or drawn a split node's untested placements in its tested
    /// pipeline, since every slot was last redrawn. A slot keeps lacking
    /// the node, or what it drew, whatever later recordings leave out (one
    /// whose slots the node has no candidates in leaves out nothing), so
    /// the pipeline's arrival redraws every slot.
    drape_waiting: bool,
    /// Whether a bundle drew a split node's untested placements in its
    /// tested pipeline for want of the untested one, which arriving
    /// records the bundles again.
    fell_back: std::cell::Cell<bool>,
    /// The device has clip distances: lines draw untested from placements
    /// with one clip rectangle ([`Untested::Rect`]).
    planes: bool,
    /// Nodes the main bundles record draws for.
    recorded: u32,
    node_uniforms: BufferId,
    /// The node uniforms as written, and the nodes changed since the last
    /// frame: one buffer write a frame, not one a node, and one for the
    /// second streams' rows past them.
    node_words: Vec<u32>,
    nodes_dirty: [Option<std::ops::Range<usize>>; 2],
    nodes: Vec<Node>,
    tiles: tiles::Tiles,
    surface_warps: tiles::SurfaceWarps,
    /// The main pass's bundles (empty until recorded), and what the map's
    /// passes draw in order, the bundles among it.
    bundles: Vec<RenderBundleId>,
    steps: Vec<Step>,
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
    /// Since the device last refused the drape a sheet: the tile slots it
    /// held then, which the drape's room keeps to, and the bytes of the
    /// sheet refused ([`reserve_drape`](Self::reserve_drape)).
    drape_ceiling: Option<(u32, u64)>,
    /// The drape pass's paint, frames and bundles, made for the first
    /// draped paint.
    drape: Option<Drape>,
    /// What each draw of the terrain draws, in style order.
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
    /// The scene's segments revision the bundles drew
    /// ([`Scene::segments`]).
    segments_drawn: u64,
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
    /// The image texture the scene group binds.
    images_bound: TextureId,
    /// What new tile pages start from ([`size_for`](Self::size_for)).
    sizing: Option<Sizing>,
    /// What the symbol placement target did made again at another size,
    /// since the last [`take_resized`](Self::take_resized).
    resized: Resized,
    /// Whether this frame leaves the map nothing to do after it
    /// ([`set_settling`](Self::set_settling)): only then do pools, lists
    /// and pages give back what they hold free.
    settling: bool,
}

/// What a map's new tile pages start at ([`Renderer::size_for`]): the
/// tiles of 512 logical pixels its view covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sizing {
    /// The cover's tiles at the steepest pitch: the viewport's, in tiles,
    /// and those up to the horizon ([`Sizing::of_viewport`]).
    pub tiles: f64,
}

impl Sizing {
    /// Tiles a tilted view covers beyond those of its viewport: GL JS and
    /// Orbis cover a view at pitch 60 to 85 with about four times its
    /// flat tiles, the far ones coarser.
    const PITCHED: f64 = 4.0;

    /// Sizing for a viewport of `width × height` logical pixels.
    pub fn of_viewport([width, height]: [f64; 2]) -> Self {
        Self {
            tiles: width * height / (512.0 * 512.0) * Self::PITCHED,
        }
    }

    /// Layers a page of `size`-texel tiles starts with: one for each tile
    /// the cover holds at that size, and no fewer than eight. A view that
    /// holds more (stand-ins while tiles load, the coarser tiles a
    /// terrain's cover and culling read) grows it.
    fn page_layers(self, size: [u32; 2]) -> u32 {
        let side = f64::from(size[0].max(size[1]).max(1));
        let tiles = self.tiles * (512.0 / side).powi(2);
        (tiles.ceil() as u32).max(8)
    }
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
        let groups = self.draw_groups;
        let pipelines = (
            (self.paint.pipelines, self.paint.untested),
            (self.paint_kept.pipelines, self.paint_kept.untested),
            self.symbols,
        );
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
            .map(|d| ((d.paint.pipelines, d.paint.untested), d.images.0));
        (self.terrain, self.drape, drape, self.no_terrain_depth).release(gpu);
        self.map_depth.map(|(depth, _)| depth).release(gpu);
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
        let target = Attachments::from(target);
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
        let draw_groups = scene_groups(
            gpu,
            draw_layout,
            &scene,
            &image_bindings,
            (glyphs.texture(), nearest, no_dem, no_terrain_depth),
            None,
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
            map_depth: None,
            antialias_3d: false,
            gaps: Self::gaps_on(gpu.caps()),
            atmosphere,
            scene,
            draw_groups,
            driver,
            draw_layout,
            node_layout,
            paint: PaintPipelines::default(),
            paint_kept: PaintPipelines::default(),
            symbols,
            full_projection: false,
            terrain_reachable: false,
            full: false,
            background_fills: false,
            beneath: None,
            pipelines_stale: false,
            pipeline_set: PipelineSet::default(),
            patterned: Vec::new(),
            rowed: Vec::new(),
            set_placings: None,
            set_stale: false,
            set_asked: 0,
            made_seen: 0,
            waiting: 0,
            drape_waiting: false,
            fell_back: Default::default(),
            planes: gpu.caps().clip_distances,
            recorded: 0,
            node_uniforms,
            node_words: vec![0; (NODE_UNIFORM_SIZE / 4) as usize * capacity.nodes as usize],
            nodes_dirty: [None, None],
            nodes: Vec::new(),
            tiles: Default::default(),
            surface_warps: Default::default(),
            bundles: Vec::new(),
            steps: Vec::new(),
            paints_written: 0,
            images,
            downsampler,
            heatmaps,
            layers: Default::default(),
            composites,
            extrusions,
            terrain: None,
            terrain_page: None,
            drape_ceiling: None,
            drape: None,
            terrain_layers: Vec::new(),
            drape_stacks: [0; 2],
            drape_pipelines: None,
            dem_bound: no_dem,
            no_terrain_depth,
            terrain_depth_bound: no_terrain_depth,
            glyphs_bound: glyphs.revision(),
            buffers_bound: 0,
            segments_drawn: 0,
            glyphs,
            images_bound: image_bindings.texture,
            image_bindings,
            sizing: None,
            resized: Resized::default(),
            settling: true,
        }
    }

    /// Sizes the tile pages made from then on for `sizing`. The pools,
    /// tables and lists start small and grow to what the tiles in view
    /// need, the vertex and index pools a block at a time.
    pub fn size_for(&mut self, sizing: Sizing) {
        self.sizing = Some(sizing);
    }

    /// What the tile pages were last sized for ([`size_for`](Self::size_for)).
    pub fn sizing(&self) -> Option<Sizing> {
        self.sizing
    }

    /// The image atlas's bindings as they are now: the atlas grows pages,
    /// which the next frame binds.
    pub fn set_images(&mut self, bindings: ImageBindings) {
        self.image_bindings = bindings;
    }

    /// The format of the targets it draws into.
    pub fn format(&self) -> TargetFormat {
        self.target.format()
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

    /// Says whether the frame drawn next leaves the map nothing to do
    /// after it: nothing loading, deferred or fading. Pools, lists and
    /// pages grow in any frame that needs them to, and give back what they
    /// hold free only in such a frame, the last before the map idles: a
    /// view whose needs swing from frame to frame (a drape redrawing slots
    /// and then none, a cover taking tiles and dropping them) would
    /// otherwise make and let go of the same memory frame after frame.
    /// What a frame shows doesn't depend on it.
    pub fn set_settling(&mut self, settling: bool) {
        self.settling = settling;
    }

    /// What pools, tables, tile pages and the drape atlas did moving into
    /// memory of another size since the last call.
    pub fn take_resized(&mut self) -> Resized {
        let mut resized = std::mem::take(&mut self.resized);
        resized += self.scene.take_resized();
        resized += self.images.take_resized();
        resized += self.glyphs.take_resized();
        if let Some(terrain) = &mut self.terrain {
            resized += terrain.take_resized();
        }
        resized
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
        self.scene.buffers_revision() + self.images.revisions()
    }

    /// The glyph atlas symbol quads sample, its rows written where the
    /// property rows are now.
    pub fn glyphs(&mut self) -> &mut GlyphAtlas {
        self.glyphs
            .set_table(self.scene.tables().writer(Table::Props));
        &mut self.glyphs
    }

    /// Draws 3D surfaces at four samples when `enabled`, or at one (the
    /// default).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        self.antialias_3d = enabled;
        self.fit_samples_3d(gpu);
    }

    /// Draws 3D surfaces at the sample count they take: four where asked
    /// ([`set_antialias_3d`](Self::set_antialias_3d)), but the map's
    /// where its passes take their depth (`splits_at_3d`).
    fn fit_samples_3d(&mut self, gpu: &mut dyn Gpu) {
        let four = self.antialias_3d && !self.splits_at_3d();
        if self.extrusions.set_antialias(four) {
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

    /// The scene's per-row draw arguments and cull counters, for
    /// diagnostics: [`ARGS_WORDS`] words per cull row, as `cull.wgsl`
    /// writes them, block by block, and the number of rows.
    pub fn draw_args(&self) -> (BufferId, u32) {
        let rows = self.scene.rows() * self.scene.block_slots();
        (self.scene.buffers().args, rows)
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
        // A node's counts take in its second stream's, and every block's.
        let blocks = self.scene.block_slots();
        (self.nodes.iter().enumerate())
            .map(|(i, n)| {
                let rows = std::iter::once(i as u32).chain(self.scene.second_row(i as u32));
                let (candidates, kept) = rows.fold((0, 0), |(c, k), row| {
                    let kept: u32 = (0..blocks)
                        .map(|b| word(self.scene.row(row, b) as usize))
                        .sum();
                    (c + self.scene.node_candidates(row), k + kept)
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
        // with a row for each extrusion node's cut stream and each split
        // node's tested placements after them. Every node's group binds
        // its uniform again below.
        let untested = untested_variants(gpu.caps());
        let seconds = families
            .iter()
            .filter(|&&f| f == Family::FillExtrusion || splits(f, untested))
            .count();
        let count = (families.len() + seconds) as u32;
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
                    (
                        (node.group, node.second_group),
                        node.extrusion,
                        node.heatmap,
                    )
                        .release(gpu);
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
                        untested: splits(family, untested).then_some(Untested::None),
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
                second_group: None,
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
                pattern: false,
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
            let Some(row) = self.scene.second_row(i as u32) else {
                continue;
            };
            let uniform = self.node_uniform(row, false);
            let node = &self.nodes[i];
            let group = match &node.image {
                Some(image) => self.images.group(gpu, node.family, image, uniform),
                None => gpu.create_bind_group(&BindGroupDesc {
                    label: "node second stream",
                    layout: self.node_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform,
                    }],
                }),
            };
            self.nodes[i].second_group = Some(group);
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
    /// layer of its own, at the layer's opacity and in its mode. A shaded
    /// hillshade ends its stack and draws its slopes into one of its own:
    /// the terrain layer lights them over the stack before and under the
    /// next, which draws the layers above it (up to
    /// [`terrain::MAX_SHADES`] hillshades a terrain layer; another starts
    /// the next). A terrain layer without one whose
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
                (stacks.layers[usize::from(l)].shades.iter().flatten())
                    .any(|s| s.node as usize == i && s.blend != Blend::Normal)
            });
            node.cull.target = match stack {
                Some(stack) => PlacementTarget::Drape(stack),
                None => PlacementTarget::Map,
            };
        }
        self.drape_stacks = stacks.counts;
        self.terrain_layers = stacks.layers;
        // A layer's own terrain layer blends in its blend mode.
        for node in self.nodes.iter().filter(|n| !n.blends_on_terrain) {
            if let (Some(layer), Some((_, _, blend))) = (node.terrain_layer, node.composite)
                && blend != Blend::Normal
            {
                self.terrain_layers[usize::from(layer)].blend = blend;
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
            self.fit_depth(gpu);
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
        if self.terrain_on() {
            // The terrain layers may draw with pipelines the set's didn't.
            self.request_terrain(gpu, self.pipeline_set.shaded);
        }
    }

    /// The terrain, made for the first that turns on or that the set asks
    /// for ahead.
    fn terrain(&mut self, gpu: &mut dyn Gpu) -> &mut terrain::Terrain {
        // It draws in the map's passes while they have their depth.
        let target = Attachments {
            depth: true,
            ..self.target
        };
        let atmosphere = self.atmosphere.uniform;
        let linear = self.images.linear();
        (self.terrain).get_or_insert_with(|| terrain::Terrain::new(gpu, target, atmosphere, linear))
    }

    /// Sets whether the style has terrain, on or not yet (its DEM source
    /// may still resolve): the pipeline set then holds the terrain's and
    /// the draped nodes' drape pipelines from the first. A map that has
    /// drawn without terrain takes it only once what it draws on terrain
    /// is made ([`terrain_ready_as`](Self::terrain_ready_as)), so no frame
    /// between draws without its layers. Returns whether the terrain is
    /// reachable now: until it is, the map stays as it was.
    pub fn set_terrain_reachable(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        let drew_flat = !self.target.depth && self.nodes.iter().any(|n| n.drawn);
        let on = on && (self.terrain_reachable || !drew_flat || self.terrain_ready_as(gpu, true));
        if self.terrain_reachable != on {
            self.terrain_reachable = on;
            self.set_stale = true;
        }
        self.fit_depth(gpu);
        on
    }

    /// Gives the map's passes a depth attachment while the style has
    /// terrain or terrain is on, for the terrain to draw in, or none:
    /// every pipeline they draw with declares it, so the set and the
    /// nodes ask for those of the new kind, made ahead where the terrain
    /// waited for them.
    fn fit_depth(&mut self, gpu: &mut dyn Gpu) {
        let depth = self.terrain_reachable || self.terrain_on();
        if self.target.depth == depth {
            return;
        }
        self.swap_kind(depth);
        self.map_depth.take().map(|(depth, _)| depth).release(gpu);
        self.pipelines_stale = true;
        self.set_stale = true;
        self.invalidate_bundle(gpu);
        self.layers.clear(gpu);
    }

    /// Draws into the map's passes with (`depth`) or without the
    /// terrain's depth from now on: the pipelines for that kind, kept
    /// since the passes last were of it, take the place of these, which
    /// are kept in turn.
    fn swap_kind(&mut self, depth: bool) {
        self.target.depth = depth;
        let target = self.target;
        std::mem::swap(&mut self.paint, &mut self.paint_kept);
        self.symbols.swap_draws();
        self.images.retarget(target);
        self.composites.retarget(target);
        self.atmosphere.swap_target();
    }

    /// Whether every pipeline the map's frames draw with once its terrain
    /// is on (`on`), or once it is gone, is made, asking for those that
    /// aren't: those of its nodes shown so far, its sky and its
    /// composites, as they draw then. The map turns its terrain on or
    /// drops it only then, so that no frame between draws without the
    /// layers whose pipelines compile.
    pub fn terrain_ready_as(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        // The renderer as it is once terrain is on, its DEM still to come,
        // or gone.
        let paint = self.terrain.as_mut().and_then(|t| t.paint.take());
        let (reachable, full, depth) = (self.terrain_reachable, self.full, self.target.depth);
        self.terrain_reachable = on;
        if depth != on {
            self.swap_kind(on);
        }
        self.assign_stacks_as(on);
        self.full = on;
        self.request_set(gpu);
        let mut wanted = Vec::new();
        for i in 0..self.nodes.len() {
            if !self.scene_cull(i).visible && !self.nodes[i].drawn {
                continue;
            }
            self.make_for_node(gpu, i, false);
            if let Some((_, opacity, blend)) = self.nodes[i].composite
                && self.nodes[i].terrain_layer.is_none()
                && composite_group(self.nodes[i].composite).is_some()
            {
                let mode = match blend {
                    Blend::Normal if opacity < 1.0 => offscreen::Mode::Over,
                    blend => offscreen::Mode::Blend(blend),
                };
                self.composites.make(gpu, mode);
            }
            self.node_pipelines(i, &mut wanted);
        }
        wanted.extend(self.composites.pipelines());
        wanted.extend(self.atmosphere.pipelines());
        // Back as it was.
        self.full = full;
        self.terrain_reachable = reachable;
        if let (Some(terrain), Some(paint)) = (self.terrain.as_mut(), paint) {
            terrain.paint = Some(paint);
        }
        self.assign_stacks();
        if depth != on {
            self.swap_kind(depth);
        }
        self.extrusions.set_full(self.variants()[1]);
        wanted.into_iter().all(|id| gpu.pipeline_ready(id))
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

    /// The terrain tiles the drape pages hold, 0 before they're made.
    pub fn drape_slots(&self) -> u32 {
        (self.terrain.as_ref()).map_or(0, |t| t.layout().tile_slots)
    }

    /// How many terrain tiles the drape atlas can hold with this style's
    /// stacks: the most a terrain cover may have. Since the device refused
    /// a sheet, the slots the drape held then
    /// ([`lift_drape_ceiling`](Self::lift_drape_ceiling)).
    pub fn drape_room(&self, gpu: &dyn Gpu) -> u32 {
        let room = terrain::room(gpu, self.drape_stacks());
        self.drape_ceiling
            .map_or(room, |(slots, _)| room.min(slots))
    }

    /// Lets the drape ask the device for sheets past those it held when
    /// the device refused one, as a map whose memory settled does:
    /// whether it kept to fewer.
    pub fn lift_drape_ceiling(&mut self) -> bool {
        self.drape_ceiling.take().is_some()
    }

    /// Makes the drape pages hold `tile_slots` terrain tiles' slots, one
    /// per drape stack, in the fewest sheets that do ([`SHEET_SLOTS`]
    /// slots each): sheets are added blank and the last let go, the rest
    /// staying, slots and all, where they are. Pages of other stacks are
    /// new and blank (a new layout generation). Fails past the
    /// [`drape_room`](Self::drape_room), or where the device refuses a
    /// sheet (which lets go of them all): the room then keeps to the slots
    /// held before, so that the next frame's cover fits what the device
    /// gave, until [`lift_drape_ceiling`](Self::lift_drape_ceiling).
    pub fn reserve_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        tile_slots: u32,
    ) -> Result<DrapeLayout, DrapeFull> {
        if let Some((0, bytes)) = self.drape_ceiling {
            return Err(DrapeFull::Refused(bytes));
        }
        let stacks = self.drape_stacks();
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let held = terrain.layout().tile_slots;
        let made = terrain
            .reserve(gpu, (tile_slots, stacks), &mut self.downsampler)
            .inspect_err(|full| {
                if let DrapeFull::Refused(bytes) = *full {
                    self.drape_ceiling = Some((held.min(tile_slots), bytes));
                }
            })?;
        let layout = terrain.layout();
        if made {
            // Stack layers and the main bundle's composites follow it.
            self.invalidate_bundle(gpu);
        }
        Ok(layout)
    }

    /// Redraws these terrain tiles' drape slots this frame, at the side
    /// each shrinks to ([`TerrainTile::shrink`]), in the stacks given, as
    /// a map at style zoom `zoom` draws: their drape placements are in the
    /// scene, and every draped node's paint was set at that zoom
    /// ([`set_drape_paint`](Self::set_drape_paint)). The drape pass's
    /// frame takes it, for what its shaders blend by the zoom.
    pub fn redraw_drape(&mut self, zoom: f64, tile_slots: &[(u32, u8, &[u8])]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.redraw(zoom, tile_slots);
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
        // A split node's tested placements draw with the paint at their
        // row's offset, their row's index in it.
        let second = (self.scene.second_row(node))
            .filter(|_| self.nodes[node as usize].extrusion.is_none())
            .zip(paint)
            .map(|(row, p)| (row, p.uniform_words(row)));
        let rows = u64::from(self.scene.capacity().nodes);
        let stride = NODE_UNIFORM_SIZE;
        let level = self.drape.get_or_insert_with(|| Drape {
            frames: Default::default(),
            uniforms: gpu.create_buffer(&BufferDesc {
                label: "drape node uniforms",
                size: stride * rows,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
            written: Vec::new(),
            scene: Default::default(),
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
            let at = std::iter::once((node, words));
            for (row, words) in at.chain(second.as_ref().map(|(row, w)| (*row, w))) {
                gpu.write_buffer(
                    level.uniforms,
                    u64::from(row) * stride,
                    bytemuck::cast_slice(words),
                );
            }
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
        self.steps.clear();
    }

    fn invalidate_bundle(&mut self, gpu: &mut dyn Gpu) {
        self.drop_bundles(gpu);
        for node in &self.nodes {
            node.left_out.set([false; 2]);
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
        let layers = self.page_layers(size);
        self.images.create_page(gpu, kind, size, layers)
    }

    /// Whether image tiles of style source `source` have a page
    /// ([`upload_image`](Self::upload_image) makes its first).
    pub fn has_page(&self, source: u32) -> bool {
        self.tiles.page(source).is_some()
    }

    /// Layers a new page starts with: what the view's cover needs of its
    /// tiles ([`Sizing`]), or eight before the map sized them.
    pub(crate) fn page_layers(&self, size: [u32; 2]) -> u32 {
        self.sizing.map_or(8, |sizing| sizing.page_layers(size))
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

    /// A layer of `page`: a free one, or with `grow` one the page grows
    /// to make, to the device's limits. The terrain's page, one chunk,
    /// grows by copying the layers it holds, so a map frees what it keeps
    /// out of view first.
    pub fn alloc_layer(
        &mut self,
        gpu: &mut dyn Gpu,
        page: PageId,
        grow: bool,
    ) -> Result<u32, OutOfSpace> {
        self.images.alloc_layer(gpu, page, grow)
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

    /// Writes new elevations of the same size over DEM layer `layer` of
    /// `page`, unseen until [`publish_dems`](Self::publish_dems) shows
    /// them: memory doesn't move, and no prepare reads the layer before.
    pub fn stage_dem(&mut self, gpu: &mut dyn Gpu, page: PageId, layer: u32, pixels: Pixels<'_>) {
        self.images.stage_dem(gpu, page, layer, pixels);
    }

    /// Shows the elevations staged in `layers` of `page` in the next
    /// prepare pass, together: each layer prepared again, and the borders
    /// its neighbours padded from it.
    pub fn publish_dems(&mut self, page: PageId, layers: &[u32]) {
        self.images.publish_dems(page, layers);
    }

    /// The page and layer of source `source`'s resident image tile `tile`.
    pub fn image_layer(&self, source: u32, tile: orb_crs::TileIndex) -> Option<(PageId, u32)> {
        Some((self.tiles.page(source)?, self.tiles.layer_of(source, tile)?))
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
            let second = (self.scene.second_row(i as u32)).map(|row| self.node_uniform(row, false));
            let node = &mut self.nodes[i];
            let Some(image) = &mut node.image else {
                continue;
            };
            if !self.images.stale(image) {
                continue;
            }
            gpu.destroy_bind_group(node.group);
            node.group = self.images.node_group(gpu, node.family, image, uniform);
            // Its tested placements' group binds the page too.
            if let Some(uniform) = second {
                node.second_group.take().release(gpu);
                node.second_group = Some(self.images.group(gpu, node.family, image, uniform));
            }
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
        let second_row = self.scene.second_row(node);
        let planes = self.planes;
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
        let shape = (paint.styled(), paint.reads_rows(), paint.patterned());
        if (entry.styled, entry.rows, entry.pattern) != shape {
            (entry.styled, entry.rows, entry.pattern) = shape;
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
            // An extrusion node's cut stream, and a split node's tested
            // placements, draw with its paint, from their own row, past
            // the render nodes', which it says; the cut stream says it is.
            let second = second_row.map(|row| {
                if entry.extrusion.is_none() {
                    return (row as usize, paint.uniform_words(row));
                }
                let mut cut = words.clone();
                cut[orb_shaders::layout::extrusion_node::INDEX] = row;
                cut[orb_shaders::layout::extrusion_node::CUT] = 1;
                (row as usize, cut)
            });
            let rows = std::iter::once((0, node as usize, &words));
            for (part, row, words) in rows.chain(second.as_ref().map(|(row, w)| (1, *row, w))) {
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
        let untested = entry.cull.untested.map(|_| untested_of(paint, planes));
        if margin != entry.cull.margin_px
            || elevation != entry.cull.elevation
            || surface != entry.cull.surface
            || untested != entry.cull.untested
        {
            entry.cull.margin_px = margin;
            entry.cull.elevation = elevation;
            entry.cull.surface = surface;
            entry.cull.untested = untested;
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
        let rowed = self.rowed.iter().any(|f| f.is_symbol());
        2 * usize::from(icons.along_line || text.along_line)
            + usize::from(icons.rows || text.rows || rowed)
    }

    /// Icon node `i`'s [`symbol_kind`](Self::symbol_kind), and the
    /// [`placings`](Self::placings) that place every symbol layer's quads.
    fn symbol_kinds(&self, i: usize) -> (usize, symbol::Placings) {
        (self.symbol_kind(i), self.placings())
    }

    /// The placement dispatch's kind: with line placement if any symbol
    /// layer places along lines, and property rows if any reads them, at
    /// any zoom of the style's set or as its nodes are now (`None` without
    /// symbol layers): one pipeline, whatever the zoom.
    fn placings(&self) -> symbol::Placings {
        (self.nodes.iter().enumerate())
            .filter(|(_, n)| n.family == Family::SymbolIcon)
            .map(|(i, _)| self.symbol_kind(i))
            .chain(self.set_placings)
            .reduce(|a, b| a | b)
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
    /// what draws together ([`PipelineGroup`]), and the sky's. Asks for
    /// them at once: the
    /// first view's groups as frames wait for them, in draw order, then
    /// the rest in the background, after every other compile. Frames then
    /// choose among them and make none: a node shown before its pipeline
    /// compiled waits for that pipeline alone, or draws with a variant
    /// compiled that draws it alike (`Variants::drawn`).
    pub fn set_pipelines(&mut self, gpu: &mut dyn Gpu, set: PipelineSet) {
        let mut patterned: Vec<Family> = (set.groups.iter())
            .flat_map(|g| &g.nodes)
            .filter(|(_, paint)| paint.as_ref().is_some_and(NodePaint::patterned))
            .filter_map(|&(i, _)| self.nodes.get(i as usize).map(|n| n.family))
            .collect();
        patterned.sort_by_key(|f| *f as u8);
        patterned.dedup();
        let mut rowed: Vec<Family> = Vec::new();
        if gpu.caps().compiles_hold_presents {
            rowed = (set.groups.iter())
                .flat_map(|g| &g.nodes)
                .filter(|(_, paint)| paint.as_ref().is_some_and(NodePaint::reads_rows))
                .filter_map(|&(i, _)| self.nodes.get(i as usize).map(|n| n.family))
                .collect();
            rowed.sort_by_key(|f| *f as u8);
            rowed.dedup();
        }
        if (&patterned, &rowed) != (&self.patterned, &self.rowed) {
            (self.patterned, self.rowed) = (patterned, rowed);
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
        self.pipeline_set = set;
        self.set_stale = false;
        self.request_set(gpu);
    }

    /// Asks for the set's pipelines again in what the map now draws them
    /// with (other variants, sample count, terrain), once that changed
    /// since they were asked for.
    pub fn renew_pipelines(&mut self, gpu: &mut dyn Gpu) {
        if std::mem::take(&mut self.set_stale) {
            self.request_set(gpu);
        }
    }

    /// How many pipelines the set asked for since the last call: those
    /// the map asked for besides are what the set left out.
    pub fn take_set_asked(&mut self) -> u32 {
        std::mem::take(&mut self.set_asked)
    }

    /// What comes before a frame: renews the set
    /// ([`renew_pipelines`](Self::renew_pipelines)) if what the frame
    /// draws changed it, and asks for the pipelines of nodes shown that
    /// the set left out: none for a complete set, which the path tests
    /// hold.
    pub fn request_pipelines(&mut self, gpu: &mut dyn Gpu) {
        self.renew_pipelines(gpu);
        if !self.pipelines_stale {
            return;
        }
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
    }

    /// Asks for the set's pipelines that aren't made: the first view's,
    /// then the rest in the background. On a style whose terrain isn't on
    /// yet, they are the drape's and the terrain's: its draped nodes wait
    /// for it.
    fn request_set(&mut self, gpu: &mut dyn Gpu) {
        let asked = gpu.counters().pipelines_requested;
        // Extrusions take the paint pipelines' variant.
        self.extrusions.set_full(self.variants()[1]);
        let set = std::mem::take(&mut self.pipeline_set);
        let ahead = self.terrain_ahead();
        if ahead {
            self.assign_stacks_as(true);
        }
        // One placement pipeline places every symbol layer, at every zoom.
        self.set_placings = None;
        for group in &set.groups {
            let kept = self.shape_as(&group.nodes);
            let icons = kept.iter().map(|&(i, _)| self.drawer(i));
            let icons = icons.filter(|&i| self.nodes[i].family == Family::SymbolIcon);
            let kinds = icons.map(|i| self.symbol_kind(i)).reduce(|a, b| a | b);
            self.set_placings = [self.set_placings, kinds]
                .into_iter()
                .flatten()
                .reduce(|a, b| a | b);
            self.restore_shapes(kept);
        }
        let mut order: Vec<&PipelineGroup> = set.groups.iter().collect();
        order.sort_by_key(|g| g.rank);
        for group in order {
            let first = group.rank == 0;
            gpu.set_background_compiles(!first);
            self.request_group(gpu, &group.nodes, !first);
            if let Some(blend) = group.composite {
                self.composites.make(gpu, offscreen::Mode::Blend(blend));
            }
        }
        gpu.set_background_compiles(false);
        if set.sky {
            // With the first view's: a tilt that brings the horizon in
            // shows the sky at once.
            let frame = self.scene.buffers().frame;
            let target = (self.target, self.full_projection);
            self.atmosphere.request(gpu, target, frame, false);
        }
        if ahead || self.terrain_on() {
            self.request_terrain(gpu, set.shaded);
        }
        if ahead {
            self.assign_stacks();
        }
        self.pipeline_set = set;
        self.set_asked += gpu.counters().pipelines_requested - asked;
    }

    /// Asks for the terrain's pipelines, a shaded hillshade's where a
    /// hillshade may shade, its depth's in the 3D layers' passes where the
    /// style has a 3D layer, and the drape atlas's mips.
    fn request_terrain(&mut self, gpu: &mut dyn Gpu, shaded: bool) {
        let three_d = (self.nodes.iter())
            .any(|n| n.family == Family::FillExtrusion)
            .then(|| self.extrusions.format());
        let layers = &self.terrain_layers;
        let shaded = shaded || layers.iter().any(terrain::TerrainLayer::shaded);
        let beneath = layers.iter().any(terrain::TerrainLayer::reads_beneath);
        if shaded {
            self.images.make_slopes(gpu);
        }
        if beneath {
            // Its group's composite carries the map over to the pass after
            // the split.
            self.composites
                .make(gpu, offscreen::Mode::Blend(Blend::Multiply));
        }
        (self.terrain(gpu)).request_pipelines(gpu, three_d, (shaded, beneath));
        self.downsampler.make(gpu);
        self.images.make_prepare(gpu);
    }

    /// Asks for the pipelines one group draws with: each node shaped as
    /// its paint there picks them, then put back.
    fn request_group(&mut self, gpu: &mut dyn Gpu, group: &[(u32, Option<NodePaint>)], warm: bool) {
        let kept = self.shape_as(group);
        let mut drawers: Vec<usize> = kept.iter().map(|&(i, _)| self.drawer(i)).collect();
        drawers.dedup();
        for i in drawers {
            self.make_for_node(gpu, i, warm);
        }
        self.restore_shapes(kept);
    }

    /// Shapes a group's nodes as their paint there picks their pipelines;
    /// what they were, for [`restore_shapes`](Self::restore_shapes); none
    /// where a node isn't of its paint's family, and the group is left out.
    fn shape_as(&mut self, group: &[(u32, Option<NodePaint>)]) -> Vec<(usize, NodeShape)> {
        let fits = |&(i, ref paint): &(u32, Option<NodePaint>)| {
            let node = self.nodes.get(i as usize);
            node.is_some_and(|n| paint.as_ref().is_none_or(|p| p.fits(n.family)))
        };
        if !group.iter().all(fits) {
            return Vec::new();
        }
        (group.iter())
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
            .collect()
    }

    fn restore_shapes(&mut self, kept: Vec<(usize, NodeShape)>) {
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
        let drape_target = drape_attachments();
        let own_drape = self.target != drape_target;
        let variants = self.variants();
        let node = &self.nodes[i];
        let (family, kind) = (node.family, node.paint_kind(&self.patterned, &self.rowed));
        if matches!(family, Family::Hillshade | Family::ColorRelief) {
            // Its DEM tiles are prepared.
            self.images.make_prepare(gpu);
        }
        if family == Family::Hillshade {
            // Its table, which its draws and the terrain's shading sample.
            self.images.make_tables(gpu);
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
                let kind = self.nodes[i].paint_kind(&self.patterned, &self.rowed);
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
                    .wanted(node.paint_kind(&self.patterned, &self.rowed), self.full)
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
                if family == Family::Hillshade {
                    out.extend(self.images.tables_pipeline().map(PipelineId::from));
                }
            }
            _ => out.extend(
                (self
                    .paint
                    .wanted(node.paint_kind(&self.patterned, &self.rowed), self.full))
                .map(PipelineId::from),
            ),
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
        // Glyphs that arrived since the last frame, one write a page layer.
        self.glyphs.flush(gpu);
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
            std::mem::take(&mut self.draw_groups).release(gpu);
            self.draw_groups = scene_groups(
                gpu,
                self.draw_layout,
                &self.scene,
                &self.image_bindings,
                (
                    self.glyphs.texture(),
                    self.images.nearest(),
                    dem,
                    terrain_depth,
                ),
                None,
            );
            self.glyphs_bound = self.glyphs.revision();
            self.dem_bound = dem;
            self.terrain_depth_bound = terrain_depth;
            for scene in self.drape.iter_mut().flat_map(|d| &mut d.scene) {
                std::mem::take(scene).release(gpu);
            }
            // Extrusion and heatmap nodes record the scene group too.
            self.forget_draws(gpu);
        }
    }

    /// What the symbol placements bind of the scene, as
    /// `symbol::compute_layout` has it, by block slot: the block's frame,
    /// the tables but the indices, the terrain as the scene group binds
    /// it, the node uniforms and the draw arguments.
    fn symbol_scene(&self) -> Vec<Option<Vec<BindGroupEntry>>> {
        use TextureViewDimension::{D2, D2Array};
        let held: Vec<u32> = self.scene.blocks().collect();
        (0..self.scene.block_slots())
            .map(|block| {
                if !held.contains(&block) {
                    return None;
                }
                let frame = self.scene.block_frame(block);
                let mut entries = vec![BindGroupEntry::buffer(0, frame)];
                let tables = self.scene.tables().entries(block as usize);
                entries.extend(tables.into_iter().filter(|e| e.binding != 5));
                entries.extend([
                    BindGroupEntry::texture(14, self.dem_bound, D2Array),
                    BindGroupEntry::texture(15, self.terrain_depth_bound, D2),
                    BindGroupEntry::buffer(17, self.node_uniforms),
                    BindGroupEntry::buffer(18, self.scene.buffers().args),
                ]);
                Some(entries)
            })
            .collect()
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
        // A settling frame gives back what tables and pages a view left
        // mostly free.
        if self.settling {
            self.shrink(gpu);
        }
        let (candidates, candidates_rebuilt) = self.scene.prepare(gpu, frame, self.settling);
        // Nodes that draw in stretches of one block's slots record them.
        let segments = self.scene.segments_revision();
        if segments != self.segments_drawn {
            self.segments_drawn = segments;
            self.invalidate_bundle(gpu);
        }
        // The symbol draws bind the placement target, which the placements
        // write whole each frame: it is made again for the symbol nodes'
        // quads, with room for twice them, so that labels rarely outgrow
        // it, nothing to copy.
        let (rows, needed) = (self.symbols.rows(), self.scene.placed_rows());
        let fit = match needed * 2 > rows || needed * 8 <= rows {
            true => needed * 2,
            false => rows,
        };
        let before = gpu.counters();
        if self.symbols.fit(gpu, fit) {
            if rows > 0 {
                self.resized.count(gpu, &before);
            }
            self.invalidate_bundle(gpu);
        }
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
            slot_bytes_moved: self.scene.moved_bytes(),
            ..FrameStats::default()
        };
        let cull_ready = self.cull_passes(gpu, &mut passes, &mut stats);
        if terrain_on {
            let passing = clock();
            let drawn = self.terrain_passes(gpu, &drape, &mut passes);
            stats.draws += drawn.draws;
            stats.drape_slots = drawn.slots;
            drape_passes = micros(passing, clock());
        }
        // Over the terrain's depth, where it draws this frame.
        let terrain_depth = match self.terrain.as_ref().filter(|t| terrain_on && t.drawing()) {
            Some(t) => t.depth_ready(gpu, terrain::THREE_D),
            None => true,
        };
        let extrusions_ready =
            cull_ready && terrain_depth && self.extrusions.ready(gpu, &self.composites);
        let placings = (self.placings(), self.full);
        let quads: Vec<(u32, std::ops::Range<u32>)> = (self.scene.blocks())
            .map(|block| (block, self.scene.block_quads(block)))
            .collect();
        passes.extend(self.symbols.pass(gpu, placings, &quads));
        // Before the map's passes, or among them where they split at the
        // 3D layers.
        let split = self.steps.iter().any(|s| matches!(s, Step::ThreeD(_)));
        let (mut three_d, draws) = self.extrusion_passes(gpu, extrusions_ready, split);
        stats.draws += draws;
        if !split {
            let start = passes.len();
            passes.extend(three_d.drain(..).map(|(_, pass)| Pass::Render(pass)));
            Pass::group(&mut passes, start, "3d layers");
        }
        let start = passes.len();
        let depth = self.map_depth.filter(|_| self.target.depth);
        passes.extend(self.layers.passes(depth.map(|(depth, _)| depth)));
        Pass::group(&mut passes, start, "layers apart");
        stats.draws += self.main_passes(gpu, (target, clear), terrain_on, three_d, &mut passes);
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
        let rows = |i: usize| [Some(i as u32), self.scene.second_row(i as u32)];
        if candidates_rebuilt
            && (self.nodes.iter().enumerate()).any(|(i, n)| {
                (n.left_out.get().into_iter().zip(rows(i))).any(|(left, row)| {
                    left && row.is_some_and(|r| self.scene.node_candidates(r) > 0)
                })
            })
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
            let fell_back = self.fell_back.take();
            if self.waiting > 0 || drape_waiting || fell_back {
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
    /// map's depth, the terrain's, the heatmaps' densities. The bundles
    /// record again for targets made anew. Returns the frame's size in
    /// pixels.
    fn resize_targets(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        terrain_on: bool,
    ) -> [u32; 2] {
        let density_size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio / 4.0).floor().max(1.0) as u32);
        let size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio).max(1.0) as u32);
        let logical = frame.viewport.map(|v| v.max(1.0) as u32);
        self.fit_samples_3d(gpu);
        // Every 3D and heatmap node has its targets from the first frame,
        // whether or not its layer draws yet, so that none is made for a
        // first use in a gesture: a frame makes them only for a new
        // viewport, which the screen made anew too. The map's passes share
        // the 3D layers' depth where they split at them, which a pass of
        // its own then doesn't keep to itself.
        let extruded = self.nodes.iter().filter(|n| n.extrusion.is_some()).count();
        let shared = usize::from(extruded > 0 && terrain_on && self.splits_at_3d());
        let mut resized = self.extrusions.resize(gpu, size, extruded + shared);
        if self.target.depth && self.map_depth.is_none_or(|(_, made)| made != size) {
            self.map_depth.take().map(|(depth, _)| depth).release(gpu);
            let depth = gpu.create_texture(&TextureDesc {
                label: "map depth",
                width: size[0],
                height: size[1],
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                sample_count: self.target.sample_count,
                mip_levels: 1,
            });
            self.map_depth = Some((depth, size));
        }
        if terrain_on {
            // A terrain layer of a layer composited once draws at its
            // layer's opacity.
            let opacities: Vec<f32> = (0..self.terrain_layers.len())
                .map(|layer| {
                    let first = (self.nodes.iter())
                        .find(|n| n.terrain_layer == Some(layer as u8) && !n.blends_on_terrain);
                    first
                        .and_then(|n| {
                            n.composite
                                .filter(|_| composite_group(n.composite).is_some())
                        })
                        .map_or(1.0, |(_, opacity, _)| opacity)
                })
                .collect();
            let layers = (&self.terrain_layers[..], self.drape_stacks[0]);
            if let Some(terrain) = self.terrain.as_mut() {
                terrain.set_layers(gpu, layers, &opacities);
                resized |= terrain.resize(gpu, logical);
            }
        }
        for node in &mut self.nodes {
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

    /// Whether the map's passes split at each 3D layer, which then draws
    /// its pass between theirs: on terrain, where a terrain layer lies
    /// above a 3D layer, whose buildings hide what of it they stand in
    /// front of. The map's passes then take the 3D layers' depth, which
    /// holds the terrain's and their buildings', and draw at its sample
    /// count.
    fn splits_at_3d(&self) -> bool {
        let first = self.nodes.iter().position(|n| n.extrusion.is_some());
        self.terrain_on()
            && first.is_some_and(|first| {
                (self.nodes[first..].iter()).any(|n| n.terrain_layer.is_some())
            })
    }

    /// Records the map's bundles and the composited layers' own, for a
    /// frame of `size` pixels: what the map's passes draw ([`Step`]).
    fn record_bundles(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], terrain_on: bool) {
        let mut ready = self.ready(gpu, terrain_on);
        let mut commands = Vec::with_capacity(self.nodes.len() * 5);
        // Runs of one layer's nodes below layer opacity 1 or blending
        // other than normally draw into a texture of their own. On
        // terrain a draped layer's nodes draw in its terrain layer, at the
        // layer's opacity: one that reads the map beneath it (blending
        // other than normally, or a hillshade's blending into the map) is
        // a group that splits the pass, with nothing of its own to
        // composite.
        #[derive(Clone, Copy, PartialEq)]
        enum Key {
            Layer(u32),
            Terrain(u8),
        }
        let blank = self.composites.blank();
        let mut runs: Vec<(Key, layer_composite::Run)> = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let (key, opacity, blend, source) = match node.terrain_layer.filter(|_| terrain_on) {
                Some(l) => {
                    let layer = &self.terrain_layers[usize::from(l)];
                    if !layer.reads_beneath() {
                        continue;
                    }
                    let shade = layer.shades[0].map_or(Blend::Normal, |s| s.blend);
                    let blend = if layer.backdrop { shade } else { layer.blend };
                    (Key::Terrain(l), 1.0, blend, Some(blank))
                }
                None => {
                    let Some(layer) = composite_group(node.composite) else {
                        continue;
                    };
                    let (_, opacity, blend) = node.composite.expect("grouped");
                    (Key::Layer(layer), opacity, blend, None)
                }
            };
            match runs.last_mut() {
                Some((k, run)) if run.nodes.end == i && *k == key => run.nodes.end = i + 1,
                _ => runs.push((
                    key,
                    layer_composite::Run {
                        nodes: i..i + 1,
                        opacity,
                        blend,
                        source,
                    },
                )),
            }
        }
        let runs = runs.into_iter().map(|(_, run)| run).collect();
        let format = self.target.format();
        (self.layers).regroup(gpu, &mut self.composites, runs, (size, format));
        // Until the composites compile, nothing draws: a split main
        // pass needs its blend draw.
        if !self.layers.ready(gpu, &self.composites) {
            self.layers.clear(gpu);
            ready.cull = false;
        }
        (self.waiting, self.recorded) = (0, 0);
        let layers = RenderBundleDesc {
            label: "layers",
            color_formats: &[self.target.color],
            depth_format: self.target.depth.then_some(TextureFormat::Depth32Float),
            sample_count: self.target.sample_count,
        };
        for g in 0..self.layers.groups.len() {
            if self.layers.groups[g].drawn_elsewhere() {
                continue;
            }
            let mut group = Vec::new();
            for i in self.layers.groups[g].nodes.clone() {
                let node = &self.nodes[i];
                let groups = (&self.draw_groups[..], (node.group, node.second_group), true);
                let before = group.len();
                if !self.node_commands(gpu, ready, i, groups, &mut group) {
                    self.waiting += 1;
                }
                self.recorded += u32::from(group.len() > before);
            }
            self.layers.groups[g].bundle = Some(gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "layer composite",
                    ..layers
                },
                &bound_once(group),
            ));
        }
        let split = terrain_on && self.splits_at_3d();
        let mut steps = Vec::new();
        let mut bundles = Vec::new();
        let mut flush = |gpu: &mut dyn Gpu, commands: &mut Vec<RenderCommand>, beneath: bool| {
            if commands.is_empty() && !beneath {
                return None;
            }
            let bundle = gpu.create_render_bundle(&layers, &bound_once(std::mem::take(commands)));
            bundles.push(bundle);
            Some(if beneath {
                Step::Beneath(bundle)
            } else {
                Step::Bundle(bundle)
            })
        };
        for i in 0..=self.nodes.len() {
            // The solid background and what it hides go in a bundle
            // of their own, which the clear can stand for.
            if self.beneath.is_some_and(|b| i == b + 1) {
                steps.extend(flush(gpu, &mut commands, true));
            }
            if i == self.nodes.len() {
                break;
            }
            let (start, member) = self.layers.at(i);
            match start.map(|g| (g, &self.layers.groups[g])) {
                // The pass splits here: its blend draw opens the next
                // pass, before what follows.
                Some((g, group)) if group.blend != Blend::Normal => {
                    steps.extend(flush(gpu, &mut commands, false));
                    steps.push(Step::Blend(g));
                    self.recorded += 1;
                }
                Some((_, group)) => {
                    commands.extend(self.layers.composite_commands(&self.composites, group));
                    self.recorded += 1;
                }
                None => {}
            }
            let node = &self.nodes[i];
            if let Some(layer) = node.terrain_layer.filter(|_| terrain_on) {
                // A terrain layer draws in its first node's place.
                if i == 0 || self.nodes[i - 1].terrain_layer != Some(layer) {
                    if ready.cull && ready.terrain {
                        steps.extend(flush(gpu, &mut commands, false));
                        steps.push(Step::Terrain(layer));
                    } else {
                        self.waiting += 1;
                    }
                }
                continue;
            }
            if split && node.extrusion.is_some() {
                steps.extend(flush(gpu, &mut commands, false));
                steps.push(Step::ThreeD(i));
            }
            let groups = (&self.draw_groups[..], (node.group, node.second_group), true);
            let before = commands.len();
            if !member && !self.node_commands(gpu, ready, i, groups, &mut commands) {
                self.waiting += 1;
            }
            self.recorded += u32::from(commands.len() > before);
        }
        // The last bundle, empty or not: recorded bundles are never none.
        steps.extend(flush(gpu, &mut commands, true).map(|s| match s {
            Step::Beneath(b) => Step::Bundle(b),
            s => s,
        }));
        self.bundles = bundles;
        self.steps = steps;
    }

    /// The DEM prepare, hillshade table, image mip and cull passes and the
    /// heatmaps' density passes; whether the cull is ready to draw from.
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
        let start = passes.len();
        self.table_passes(gpu, passes);
        stats.draws += (passes.len() - start) as u32;
        Pass::group(passes, start, "hillshade tables");
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
            let draws: Vec<(BindGroupId, RenderCommand)> = (self.scene.blocks())
                .filter_map(|block| {
                    let row = self.scene.row(index as u32, block);
                    let group = self.draw_groups.get(block as usize).copied().flatten()?;
                    let draw = self
                        .driver
                        .draw(self.scene.buffers(), row, NodeDraw::Points);
                    Some((group, draw))
                })
                .collect();
            if let Some(heatmap) = &mut node.heatmap
                && node.drawn
                && cull_ready
                && self.heatmaps.ready(gpu, &self.composites)
                && let Some(pass) = self.heatmaps.prepare(gpu, heatmap, node.group, &draws)
            {
                stats.draws += 1;
                passes.push(Pass::Render(pass));
            }
        }
        Pass::group(passes, start, "heatmaps");
        cull_ready
    }

    /// The terrain's passes before the map: its drape slots (`drape`'s
    /// bundles) and its depth for what it hides, and what its draws in
    /// the map's passes draw this frame. Returns its draws.
    fn terrain_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        drape: &[(u32, RenderBundleId)],
        passes: &mut Vec<Pass>,
    ) -> terrain::TerrainDrawn {
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let start = passes.len();
        let b = self.scene.buffers();
        let scene = (b.frame, self.scene.tables().entry(Table::Props, 8));
        let drawn = terrain.passes(gpu, scene, drape, &self.downsampler, passes);
        Pass::group(passes, start, "terrain");
        drawn
    }

    /// Each terrain layer's hillshades' tables: those shown, which light
    /// their slopes' slots (blank until they draw there).
    fn shade_tables(&self) -> Vec<terrain::ShadeTables> {
        (self.terrain_layers.iter())
            .map(|l| {
                l.shades.map(|shade| {
                    let node = &self.nodes[shade?.node as usize];
                    node.cull
                        .visible
                        .then(|| node.image.as_ref()?.table())
                        .flatten()
                })
            })
            .collect()
    }

    /// The 3D layers' passes, by their nodes, each over the depth they
    /// share. The first clears it, and on terrain draws the terrain's
    /// depth into it first, unless the map's passes split at them
    /// (`split`), whose first pass did both. Also returns their draws.
    fn extrusion_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        extrusions_ready: bool,
        split: bool,
    ) -> (Vec<(usize, RenderPass)>, u32) {
        let mut out = Vec::new();
        let buffers = self.driver.buffers(self.scene.buffers());
        let terrain = (self.terrain.as_ref()).filter(|t| t.active());
        let mut terrain_depth = terrain.map(|t| t.depth_commands(terrain::THREE_D));
        let mut first = !split;
        for (index, node) in self.nodes.iter_mut().enumerate() {
            let Some(target) = node
                .extrusion
                .as_mut()
                .filter(|_| extrusions_ready && node.drawn)
            else {
                continue;
            };
            let draw = |row| self.driver.draw(&buffers, row, NodeDraw::Indexed);
            let cut = self.scene.second_row(index as u32).zip(node.second_group);
            let (cut_row, cut_group) = cut.expect("an extrusion node draws a cut stream");
            // Each block's buildings, over its group.
            let streams: Vec<(BindGroupId, extrusion::Streams)> = (self.scene.blocks())
                .filter_map(|block| {
                    let group = self.draw_groups.get(block as usize).copied().flatten()?;
                    let [whole, cut] = [index as u32, cut_row].map(|r| self.scene.row(r, block));
                    let streams = extrusion::Streams {
                        whole: (node.group, draw(whole)),
                        cut: (cut_group, draw(cut)),
                    };
                    Some((group, streams))
                })
                .collect();
            let before = if first {
                terrain_depth.take().unwrap_or_default()
            } else {
                Vec::new()
            };
            let pass = (self.extrusions).prepare(
                gpu,
                target,
                &buffers,
                (&streams, node.elevation),
                (first, before),
            );
            first = false;
            out.push((index, pass));
        }
        let draws = out.len() as u32;
        (out, draws)
    }

    /// The map's passes into `target`, cleared to `clear`: the sky, then
    /// what the steps draw (the bundles, the terrain's draws, the
    /// composited layers), then the scattering. The passes split where a
    /// group blends other than normally, and where a 3D layer of
    /// `three_d` (its pass, by its node) draws between them. Their depth
    /// where they have one: the map's own, the terrain's drawn into it
    /// first in each, or where they split at 3D layers the depth those
    /// share, which the first clears and draws the terrain's into.
    /// Returns the terrain's draws.
    fn main_passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (target, mut clear): (TextureId, [f64; 4]),
        terrain_on: bool,
        mut three_d: Vec<(usize, RenderPass)>,
        passes: &mut Vec<Pass>,
    ) -> u32 {
        let start = passes.len();
        let steps = std::mem::take(&mut self.steps);
        let split = steps.iter().any(|s| matches!(s, Step::ThreeD(_)));
        let depth = match split {
            true => self.extrusions.depth().map(|(depth, _)| depth),
            false => self.map_depth.map(|(depth, _)| depth),
        };
        let depth = depth.filter(|_| self.target.depth);
        let terrain_depth = (self.terrain.as_ref())
            .filter(|_| terrain_on)
            .map_or(Vec::new(), |t| t.depth_commands(terrain::MAP));
        let shades = self.shade_tables();
        let blends = steps.iter().filter(|s| matches!(s, Step::Blend(_))).count();
        let (mut draws, mut predraws) = (0, 0);
        // Each pass's first draw, with a depth of its own.
        let mut predraw = |commands: &mut Vec<RenderCommand>| {
            predraws += u32::from(!terrain_depth.is_empty());
            commands.extend(terrain_depth.iter().cloned());
        };
        let mut commands = Vec::new();
        predraw(&mut commands);
        let mut rest = &steps[..];
        // A solid background filling the frame is the clear: its bundle
        // and the sky drawn before it, which it hides, are left out.
        let mut sky = true;
        if let (Some(&Step::Beneath(below)), Some(b)) = (steps.first(), self.beneath) {
            rest = &steps[1..];
            let shown = self.nodes[b].cull.visible && self.scene.node_candidates(b as u32) > 0;
            match self.nodes[b].solid {
                Some(color) if shown && self.background_fills && !terrain_on => {
                    clear = color.map(f64::from);
                    sky = false;
                }
                _ => {
                    commands.extend(self.atmosphere.commands(gpu));
                    commands.push(RenderCommand::ExecuteBundles(vec![below]));
                    sky = false;
                }
            }
        }
        if sky {
            commands.extend(self.atmosphere.commands(gpu));
        }
        // The canvas of the passes, by the groups that split them so far,
        // and the passes into it so far.
        let (mut canvas, mut into) = (0usize, 0);
        let mut first = true;
        let mut pass = |this: &Self, commands: Vec<RenderCommand>, (canvas, into), last: bool| {
            let load = match into {
                0 => LoadOp::Clear(if canvas == 0 { clear } else { [0.0; 4] }),
                _ => LoadOp::Load,
            };
            let color = if canvas == blends {
                ColorAttachment {
                    store: StoreOp::Store,
                    target,
                    resolve: None,
                    load,
                }
            } else {
                this.layers.canvas(canvas).attachment(0, load)
            };
            let depth = depth.map(|target| DepthAttachment {
                target,
                load: if split && !first {
                    LoadOp::Load
                } else {
                    LoadOp::Clear(0.0)
                },
                store: if split && !last {
                    StoreOp::Store
                } else {
                    StoreOp::Discard
                },
            });
            first = false;
            Pass::Render(RenderPass {
                label: "main",
                colors: vec![color],
                depth,
                commands,
            })
        };
        for &step in rest {
            match step {
                Step::Beneath(bundle) | Step::Bundle(bundle) => {
                    commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
                }
                Step::Terrain(layer) => {
                    // A layer that reads the map beneath reads the canvas
                    // the pass before drew.
                    let beneath = canvas
                        .checked_sub(1)
                        .map(|k| self.layers.canvas(k).texture(0));
                    let tables = shades.get(usize::from(layer)).copied().unwrap_or_default();
                    let terrain = self.terrain.as_mut().expect("terrain is on");
                    let drawn = terrain.layer_commands(gpu, layer, tables, beneath);
                    draws += u32::from(!drawn.is_empty());
                    commands.extend(drawn);
                }
                Step::Blend(group) => {
                    let done = std::mem::take(&mut commands);
                    passes.push(pass(self, done, (canvas, into), false));
                    (canvas, into) = (canvas + 1, 0);
                    // The blend draw covers every pixel.
                    if !split {
                        predraw(&mut commands);
                    }
                    let group = &self.layers.groups[group];
                    commands.extend(self.layers.composite_commands(&self.composites, group));
                }
                Step::ThreeD(node) => {
                    let done = std::mem::take(&mut commands);
                    passes.push(pass(self, done, (canvas, into), false));
                    into += 1;
                    if let Some(at) = three_d.iter().position(|(n, _)| *n == node) {
                        passes.push(Pass::Render(three_d.swap_remove(at).1));
                    }
                }
            }
        }
        commands.extend(self.atmosphere.commands_after(gpu));
        passes.push(pass(self, commands, (canvas, into), true));
        self.steps = steps;
        Pass::group(passes, start, "main");
        draws + predraws
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
            terrain: (self.terrain.as_mut())
                .filter(|_| terrain_on)
                .is_some_and(|t| t.ready(gpu)),
            prepare: !dems || self.images.prepare_ready(gpu),
            tables: self.images.tables_ready(gpu),
            mips: !rasters || self.downsampler.ready(gpu),
        }
    }

    /// The passes that draw hillshade tables from paint they weren't drawn
    /// from: the drape's paint for a hillshade draped into its slots, which
    /// draw it at their zoom, else the map's.
    fn table_passes(&mut self, gpu: &mut dyn Gpu, passes: &mut Vec<Pass>) {
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if node.family != Family::Hillshade || !node.drawn {
                continue;
            }
            let draped = (self.drape.as_ref())
                .filter(|_| node.stack.is_some() && !node.shades_on_terrain())
                .and_then(|d| Some((d.uniforms, d.written.get(i)?.as_deref()?)));
            let (buffer, words) = draped.unwrap_or((self.node_uniforms, &node.written));
            let uniform = BindingResource::Buffer {
                buffer,
                offset: i as u64 * NODE_UNIFORM_SIZE,
                size: Some(NODE_UNIFORM_SIZE),
            };
            let words = words.to_vec();
            let Some(image) = self.nodes[i].image.as_mut() else {
                continue;
            };
            passes.extend(
                self.images
                    .table_pass(gpu, image, &words, uniform)
                    .map(Pass::Render),
            );
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
        let at = stack as usize;
        if level.bundle.len() <= at {
            level.bundle.resize_with(at + 1, || None);
        }
        if level.scene[page].is_empty() {
            // A frame and a group for each block, the frame telling the
            // block's rows.
            let held: Vec<u32> = self.scene.blocks().collect();
            std::mem::take(&mut level.frames[page]).release(gpu);
            level.frames[page] = (0..self.scene.block_slots())
                .map(|block| {
                    held.contains(&block).then(|| {
                        gpu.create_buffer(&BufferDesc {
                            label: "drape frame",
                            size: crate::scene::FRAME_BYTES,
                            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        })
                    })
                })
                .collect();
            level.scene[page] = scene_groups(
                gpu,
                self.draw_layout,
                &self.scene,
                &self.image_bindings,
                // Draped content isn't hidden by the terrain it's on.
                (
                    self.glyphs.texture(),
                    self.images.nearest(),
                    self.dem_bound,
                    self.no_terrain_depth,
                ),
                Some(&level.frames[page]),
            );
            // The bundles of the page's stacks bound the group it replaces.
            for (s, bundle) in level.bundle.iter_mut().enumerate() {
                if usize::from(s as u32 >= colour) == page {
                    bundle.take().release(gpu);
                }
            }
        }
        let words = crate::scene::frame_words(&drape_frame);
        for (block, buffer) in level.frames[page].iter().enumerate() {
            if let Some(buffer) = buffer {
                let mut words = words.clone();
                self.scene.block_words(block as u32, &mut words);
                gpu.write_buffer(*buffer, 0, bytemuck::cast_slice(&words));
            }
        }
        if let Some((bundle, _)) = &level.bundle[at] {
            return Some(*bundle);
        }
        let ready = self.ready(gpu, true);
        let level = self.drape.as_ref().expect("found above");
        let (scenes, uniforms) = (level.scene[page].clone(), level.uniforms);
        let mut commands = Vec::new();
        let mut groups = Vec::new();
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let in_stack = node.stack.is_some_and(|s| u32::from(s) == stack);
            if !in_stack || !level.written.get(i).is_some_and(Option::is_some) {
                continue;
            }
            let uniform = |row: u32| BindingResource::Buffer {
                buffer: uniforms,
                offset: u64::from(row) * NODE_UNIFORM_SIZE,
                size: Some(NODE_UNIFORM_SIZE),
            };
            let group = match &node.image {
                Some(image) => self
                    .images
                    .group(gpu, node.family, image, uniform(i as u32)),
                None => gpu.create_bind_group(&BindGroupDesc {
                    label: "drape node",
                    layout: self.node_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform(i as u32),
                    }],
                }),
            };
            groups.push(group);
            let second = (self.scene.second_row(i as u32))
                .filter(|_| node.extrusion.is_none())
                .map(|row| match &node.image {
                    Some(image) => self.images.group(gpu, node.family, image, uniform(row)),
                    None => gpu.create_bind_group(&BindGroupDesc {
                        label: "drape node second stream",
                        layout: self.node_layout,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: uniform(row),
                        }],
                    }),
                });
            groups.extend(second);
            let node_groups = (&scenes[..], (group, second), false);
            let fell_back = self.fell_back.replace(false);
            let recorded = self.node_commands(gpu, ready, i, node_groups, &mut commands);
            // A slot keeps what it drew: one drawn in the tested pipeline
            // for want of the untested one (which lines draw otherwise, by
            // rounding) is redrawn once it arrives, as for a pipeline a
            // node waited for, so it draws alike whenever that compiled.
            if !recorded || self.fell_back.get() {
                self.drape_waiting = true;
            }
            self.fell_back.set(fell_back || self.fell_back.get());
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

    /// Node `i`'s commands as a bundle records them, over the scene groups
    /// `scenes`, one per block slot: the map's (`in_map`), or the drape's
    /// for draped nodes on terrain, with its group and its second
    /// stream's. The node draws once in each block it may have candidates
    /// in, that block's group bound, or in its segments, in its tiles'
    /// order ([`Scene::segments`]); a split node's untested placements
    /// first, in their pipeline, then its tested ones. In the map, a
    /// terrain layer draws its nodes ([`Step::Terrain`]). `false` when
    /// the node waits for a pipeline: it records nothing, or draws with a
    /// stand-in for the variant it wants ([`Variants::drawn`]) until that
    /// one compiles and the bundles record again.
    fn node_commands(
        &self,
        gpu: &mut dyn Gpu,
        ready: Ready,
        i: usize,
        (scenes, (group, second_group), in_map): (
            &[Option<BindGroupId>],
            (BindGroupId, Option<BindGroupId>),
            bool,
        ),
        commands: &mut Vec<RenderCommand>,
    ) -> bool {
        if !ready.cull {
            return false;
        }
        let node = &self.nodes[i];
        if node.terrain_layer.is_some() && in_map {
            // Its terrain layer draws it ([`Step::Terrain`]).
            return ready.terrain;
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
        // Without candidates a row's draw draws nothing: on sparse views
        // most of a style's nodes, each a pipeline, bindings and an
        // indirect draw the CPU encodes and the GPU reads every frame. A
        // split node's second row holds its tested placements.
        let second = self.scene.second_row(i as u32);
        let streams = [Some(i as u32), second]
            .map(|row| row.filter(|&row| self.scene.node_candidates(row) > 0));
        node.left_out.set(streams.map(|row| row.is_none()));
        // A stream's rows in each block, with the block's group; a direct
        // draw of nothing records nothing: the bundles record again when
        // the node's draw changes.
        let rows = |stream: u32| -> Vec<(BindGroupId, u32)> {
            (scenes.iter().enumerate())
                .filter_map(|(block, scene)| {
                    let row = self.scene.row(stream, block as u32);
                    Some(((*scene)?, row)).filter(|&(_, row)| !self.driver.draws_nothing(row))
                })
                .collect()
        };
        let streams = streams.map(|row| row.map(rows).filter(|rows| !rows.is_empty()));
        if streams.iter().all(Option::is_none) {
            return true;
        }
        let buffers = self.scene.buffers();
        if node.family.is_symbol() {
            let rows = streams[0].clone().unwrap_or_default();
            // Its icons and its text, in one list. The cull's point
            // arguments read as indexed ones too: 6 indices into the quad
            // indices, and the first index, base vertex and first instance
            // all 0, as points always have them.
            let apart = [node.separate_halo, self.nodes[i + 1].separate_halo];
            let (kind, placings) = self.symbol_kinds(i);
            for (scene, row) in rows {
                let quads = (self.driver).draw(buffers, row, NodeDraw::Indexed);
                let recorded = self.symbols.commands(
                    gpu,
                    (kind, placings, self.full),
                    (scene, group),
                    apart,
                    quads,
                    commands,
                );
                if !recorded {
                    return false;
                }
            }
            return true;
        }
        if matches!(node.family, Family::Hillshade | Family::ColorRelief) && !ready.prepare {
            return false;
        }
        if node.family == Family::Hillshade && !ready.tables {
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
            .filter(|_| !in_map && self.target != drape_attachments());
        let kind = node.paint_kind(&self.patterned, &self.rowed);
        // A split node's untested placements draw in a pipeline that tests
        // no fragment once it compiled, else in the tested one.
        let split = second.is_some() && streams[0].is_some() && !node.shades_on_terrain();
        let untested = split.then(|| {
            let pipeline = match (drape, node.family.is_image()) {
                (Some(d), true) => d.images.untested(node.family).map(|p| (p, true)),
                (Some(d), false) => d.paint.drawn_untested(gpu, kind, self.full),
                (None, true) => self.images.untested(node.family).map(|p| (p, true)),
                (None, false) => self.paint.drawn_untested(gpu, kind, self.full),
            };
            let pipeline = pipeline.filter(|&(p, _)| gpu.pipeline_ready(p.into()));
            self.fell_back
                .set(self.fell_back.get() || pipeline.is_none());
            pipeline
        });
        let pipeline = match (drape, node.family.is_image()) {
            _ if node.shades_on_terrain() => self.images.slopes_pipeline().map(|p| (p, true)),
            (Some(d), true) => d.images.get(node.family).map(|p| (p, true)),
            (Some(d), false) => d.paint.drawn(gpu, kind, self.full),
            (None, true) => self.images.pipeline(node.family).map(|p| (p, true)),
            (None, false) => self.paint.drawn(gpu, kind, self.full),
        };
        let Some((pipeline, wanted)) = pipeline.filter(|&(p, _)| gpu.pipeline_ready(p.into()))
        else {
            return false;
        };
        let (untested, wanted) = match untested.flatten() {
            Some((untested, also)) => (untested, wanted && also),
            None => (pipeline, wanted),
        };
        let pipelines = [untested, pipeline];
        let groups = [Some(group), second_group];
        let ids = [Some(i as u32), second];
        for (((rows, pipeline), group), id) in
            (streams.into_iter().zip(pipelines)).zip(groups).zip(ids)
        {
            let (Some(rows), Some(id)) = (rows, id) else {
                continue;
            };
            let group = group.expect("a split node has its second stream's group");
            commands.extend([
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup { index: 1, group },
            ]);
            if node.cull.draw == NodeDraw::Indexed {
                commands.push(RenderCommand::SetIndexBuffer {
                    buffer: self.driver.buffers(buffers).compacted,
                    offset: 0,
                });
            }
            let Some(segments) = self.scene.segments(id) else {
                for (scene, row) in rows {
                    commands.extend([
                        RenderCommand::SetBindGroup {
                            index: 0,
                            group: scene,
                        },
                        self.driver.draw(buffers, row, node.cull.draw),
                    ]);
                }
                continue;
            };
            // Each stretch whole: the cull wrote every slot of its runs,
            // hidden indices for what is out of view, and holes between
            // runs hold hidden ones.
            for segment in segments {
                let Some(Some(scene)) = scenes.get(segment.block as usize) else {
                    continue;
                };
                if self.driver.draws_nothing(self.scene.row(id, segment.block)) {
                    continue;
                }
                commands.extend([
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: *scene,
                    },
                    RenderCommand::DrawIndexed {
                        indices: segment.indices.clone(),
                        base_vertex: 0,
                        instances: 0..1,
                    },
                ]);
            }
        }
        wanted
    }
}

/// The drape atlas's passes: draped nodes draw into them with the map's
/// pipelines where the map's passes are of the same attachments.
fn drape_attachments() -> Attachments {
    TargetFormat {
        color: DRAPE_FORMAT,
        sample_count: 1,
    }
    .into()
}

/// What the map's passes draw, in order ([`Renderer::record_bundles`]).
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The solid background and what it hides, after the sky: a frame the
    /// backgrounds fill clears to its colour in their place.
    Beneath(RenderBundleId),
    /// Nodes' draws.
    Bundle(RenderBundleId),
    /// Terrain layer `l`'s draw of the terrain, made each frame.
    Terrain(u8),
    /// Layer composite group `g` blends other than normally: the pass
    /// splits, and the next opens with its composite over the canvas the
    /// pass before drew.
    Blend(usize),
    /// 3D layer node `n` draws its pass between two of the map's, adding
    /// its buildings to the depth they share, which hide what of the
    /// terrain layers above it they stand in front of
    /// ([`Renderer::splits_at_3d`]).
    ThreeD(usize),
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
        // on, where they draw normally: it joins the layer, which goes on
        // above it, while the layer takes another hillshade. Where the
        // layer's hillshades fill it, it starts the next, blending into
        // what the layer before drew. Elsewhere its layer composites
        // apart, a terrain layer of its own.
        let blends = shaded && node.blend != Blend::Normal && previous == Some(None);
        let last = layers.last();
        let joins = blends && last.is_some_and(|l| !l.full() && (l.below.is_some() || l.shaded()));
        let backdrop = blends && last.is_some_and(terrain::TerrainLayer::full);
        let inline = joins || backdrop;
        let group = if inline { None } else { node.group };
        let open = layers
            .last()
            .filter(|l| draped && previous == Some(group) && !(shaded && l.full()));
        if draped && open.is_none() {
            layers.push(terrain::TerrainLayer {
                backdrop,
                ..Default::default()
            });
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
                let free = (layer.shades.iter_mut()).find(|s| s.is_none());
                *free.expect("a layer with room") = Some(terrain::Shade {
                    node: i as u32,
                    slopes,
                    above: None,
                    blend: if inline { node.blend } else { Blend::Normal },
                });
                slopes
            } else {
                let part = match layer.shades.iter_mut().rev().flatten().next() {
                    Some(shade) => &mut shade.above,
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
        if layer.shaded() {
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
    let slopes: Vec<u8> = (layers.iter())
        .flat_map(|l| l.shades.iter().flatten().map(|s| s.slopes))
        .collect();
    let colour = u32::from(stacks) - slopes.len() as u32;
    let renumber = |stack: u8| match slopes.iter().position(|&s| s == stack) {
        Some(rank) => colour as u8 + rank as u8,
        None => stack - slopes.iter().filter(|&&s| s < stack).count() as u8,
    };
    for layer in &mut layers {
        layer.below = layer.below.map(renumber);
        layer.above = layer.above.map(renumber);
        for shade in layer.shades.iter_mut().flatten() {
            shade.slopes = renumber(shade.slopes);
            shade.above = shade.above.map(renumber);
        }
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
    /// The hillshade tables', which hillshades wait for.
    tables: bool,
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
    /// The drape frame, per drape page and block slot: each block's tells
    /// its rows.
    frames: [Vec<Option<BufferId>>; 2],
    /// Node uniforms, at each node's offset.
    uniforms: BufferId,
    /// Per node, the words written, or `None` where it doesn't draw (outside
    /// its layer's zooms, or not draped).
    written: Vec<Option<Vec<u32>>>,
    /// Per page, the scene groups over its `frames`, by block slot; empty
    /// until made.
    scene: [Vec<Option<BindGroupId>>; 2],
    /// Per stack, the bundle of the nodes drawing into it, with the node
    /// groups it binds: a redraw replays each over its own slots.
    bundle: Vec<Option<(RenderBundleId, Vec<BindGroupId>)>>,
}

impl release::Release for Node {
    fn release(self, gpu: &mut dyn Gpu) {
        let groups = (self.group, self.second_group);
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
        let [f, g] = self.frames;
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
    /// Its variant with patterns ([`orb_shaders::PATTERNS`]).
    pattern: bool,
}

/// The vector families' pipelines (fill, fill outline, line, styled line,
/// circle), each made for the family's first node, with property rows and
/// without, and those that draw patterns with them and without; a fill's
/// and a line's also untested ([`Untested`]): a fill's fragments testing
/// nothing, a line's, where the device has clip distances
/// (`LINE_PLANES_SHADER`), clipped by their placement's planes instead.
#[derive(Clone, Copy, Debug, Default)]
struct PaintPipelines {
    pipelines: [Variants; 20],
    untested: [Variants; 20],
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
        Some(4 * shader + 2 * usize::from(kind.pattern) + usize::from(kind.rows))
    }

    /// What a frame draws a node of `kind` with, and whether it is the one
    /// it wants ([`Variants::drawn`]).
    fn drawn(
        &self,
        gpu: &mut dyn Gpu,
        kind: PaintKind,
        full: bool,
    ) -> Option<(RenderPipelineId, bool)> {
        self.pipelines[Self::index(kind)?].drawn(gpu, full)
    }

    /// What a frame draws a split node's untested placements of `kind`
    /// with, once compiled, and whether it is the variant it wants.
    fn drawn_untested(
        &self,
        gpu: &mut dyn Gpu,
        kind: PaintKind,
        full: bool,
    ) -> Option<(RenderPipelineId, bool)> {
        self.untested[Self::index(kind)?].drawn(gpu, full)
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
        target: Attachments,
        layouts: &[BindGroupLayoutId],
        kind: PaintKind,
        variants: [bool; 2],
    ) {
        let Some(index) = Self::index(kind) else {
            return;
        };
        let (shader, topology) = match index / 4 {
            0 => (&FILL_SHADER, PrimitiveTopology::TriangleList),
            1 => (&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            2 => (&LINE_SHADER, PrimitiveTopology::TriangleList),
            4 => (&LINE_STYLED_SHADER, PrimitiveTopology::TriangleList),
            _ => (&CIRCLE_SHADER, PrimitiveTopology::TriangleList),
        };
        let draw = Draw {
            topology,
            rows: kind.rows,
            patterns: kind.pattern,
            ..Draw::new(
                shader,
                layouts,
                target,
                Some(BlendState::PremultipliedAlpha),
            )
        };
        self.pipelines[index].make(gpu, variants, draw);
        let planes = gpu.caps().clip_distances;
        let vertex = match index / 4 {
            _ if !untested_variants(gpu.caps()) => None,
            0 => Some(&FILL_SHADER),
            2 if planes => Some(&LINE_PLANES_SHADER),
            4 if planes => Some(&LINE_STYLED_PLANES_SHADER),
            _ => None,
        };
        if let Some(vertex) = vertex {
            let untested = Draw {
                label: if index / 4 == 0 {
                    "fill untested"
                } else {
                    vertex.label
                },
                vertex: (vertex, vertex.vertex),
                constants: [&[], &[(orb_shaders::CLIP_FRAGMENTS, 0.0)]],
                ..draw
            };
            self.untested[index].make(gpu, variants, untested);
        }
    }
}

/// The scene's groups, by block slot (`None` for a free one): each
/// block's frame (`frames`' if given, else the block's own) and pools,
/// the tables every block shares, the image pages and the glyph pages.
fn scene_groups(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    scene: &Scene,
    images: &ImageBindings,
    (glyphs, nearest, dem, terrain_depth): (TextureId, SamplerId, TextureId, TextureId),
    frames: Option<&[Option<BufferId>]>,
) -> Vec<Option<BindGroupId>> {
    use TextureViewDimension::{D2, D2Array};
    let held: Vec<u32> = scene.blocks().collect();
    (0..scene.block_slots())
        .map(|block| {
            if !held.contains(&block) {
                return None;
            }
            let frame = match frames {
                Some(frames) => frames.get(block as usize).copied().flatten()?,
                None => scene.block_frame(block),
            };
            let mut entries = vec![BindGroupEntry::buffer(0, frame)];
            entries.extend(scene.tables().entries(block as usize));
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
            Some(gpu.create_bind_group(&BindGroupDesc {
                label: "scene",
                layout,
                entries: &entries,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain::{Shade, TerrainLayer};

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
        let layer = |below, above| TerrainLayer {
            below,
            above,
            ..TerrainLayer::default()
        };
        // Hillshade `node` lit from slopes stack `slopes`, under `above`.
        let shade = |node, slopes, above| {
            Some(Shade {
                node,
                slopes,
                above,
                blend: Blend::Normal,
            })
        };
        let multiplied = |node, slopes, above| {
            Some(Shade {
                blend: Blend::Multiply,
                ..shade(node, slopes, above).unwrap()
            })
        };
        let shaded = |below, shades: &[Option<Shade>]| TerrainLayer {
            below,
            shades: std::array::from_fn(|k| shades.get(k).copied().flatten()),
            ..TerrainLayer::default()
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
                    layers: vec![layer(Some(0), None)],
                    counts: [1, 0],
                },
            ),
            (
                "a node that doesn't drape ends the run",
                vec![plain, node(false, None, false, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (None, None), (Some(1), Some(1))],
                    layers: vec![layer(Some(0), None), layer(Some(1), None)],
                    counts: [2, 0],
                },
            ),
            (
                "a layer that composites once has a terrain layer of its own",
                vec![plain, node(true, Some(5), false, false)],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(1))],
                    layers: vec![layer(Some(0), None), layer(Some(1), None)],
                    counts: [2, 0],
                },
            ),
            (
                "a shaded hillshade's slopes stack after the colour stacks",
                vec![plain, node(true, None, true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(2), Some(0)), (Some(1), Some(0))],
                    layers: vec![shaded(Some(0), &[shade(1, 2, Some(1))])],
                    counts: [2, 1],
                },
            ),
            (
                "a second shaded hillshade shades the same terrain layer",
                vec![
                    node(true, None, true, false),
                    plain,
                    node(true, None, true, false),
                ],
                DrapeStacks {
                    nodes: vec![(Some(1), Some(0)), (Some(0), Some(0)), (Some(2), Some(0))],
                    layers: vec![shaded(None, &[shade(0, 1, Some(0)), shade(2, 2, None)])],
                    counts: [1, 2],
                },
            ),
            (
                "a third starts a terrain layer",
                vec![node(true, None, true, false); 3],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(0)), (Some(2), Some(1))],
                    layers: vec![
                        shaded(None, &[shade(0, 0, None), shade(1, 1, None)]),
                        shaded(None, &[shade(2, 2, None)]),
                    ],
                    counts: [0, 3],
                },
            ),
            (
                "a blended shaded hillshade blends into the layer it lies on",
                vec![plain, node(true, Some(5), true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(2), Some(0)), (Some(1), Some(0))],
                    layers: vec![shaded(Some(0), &[multiplied(1, 2, Some(1))])],
                    counts: [2, 1],
                },
            ),
            (
                "after another shaded hillshade, it blends into its layer too",
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
                        (Some(3), Some(0)),
                        (Some(1), Some(0)),
                    ],
                    layers: vec![shaded(
                        Some(0),
                        &[shade(1, 2, None), multiplied(2, 3, Some(1))],
                    )],
                    counts: [2, 2],
                },
            ),
            (
                "past a full layer, it blends into what that layer drew",
                vec![
                    plain,
                    node(true, None, true, false),
                    node(true, None, true, false),
                    node(true, Some(5), true, false),
                    plain,
                ],
                DrapeStacks {
                    nodes: vec![
                        (Some(0), Some(0)),
                        (Some(2), Some(0)),
                        (Some(3), Some(0)),
                        (Some(4), Some(1)),
                        (Some(1), Some(1)),
                    ],
                    layers: vec![
                        shaded(Some(0), &[shade(1, 2, None), shade(2, 3, None)]),
                        TerrainLayer {
                            backdrop: true,
                            ..shaded(None, &[multiplied(3, 4, Some(1))])
                        },
                    ],
                    counts: [2, 3],
                },
            ),
            (
                "with nothing beneath it, it composites apart",
                vec![node(true, Some(5), true, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(1), Some(0)), (Some(0), Some(1))],
                    layers: vec![shaded(None, &[shade(0, 1, None)]), layer(Some(0), None)],
                    counts: [1, 1],
                },
            ),
            (
                "zoom-free nodes first split from those drawn by zoom",
                vec![plain, node(true, None, false, true), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(0)), (Some(1), Some(0))],
                    layers: vec![layer(Some(0), Some(1))],
                    counts: [2, 0],
                },
            ),
            (
                "nodes drawn by zoom first don't split",
                vec![node(true, None, false, true), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)); 2],
                    layers: vec![layer(Some(0), None)],
                    counts: [1, 0],
                },
            ),
        ];
        for (name, inputs, expected) in cases {
            assert_eq!(drape_stacks(&inputs), expected, "{name}");
        }
    }
}
