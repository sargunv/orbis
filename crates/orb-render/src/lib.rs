//! The renderer: a frame's passes over the scene the map's tiles fill.
//!
//! The cull runs first, a dispatch per scene block on the GPU driver (or
//! on the CPU, WebGL 2's driver, `driver`), keeping each render node's
//! visible `(chunk, tile placement)` candidates in draw order and writing
//! their draw arguments; fill, line and extrusion nodes draw compacted
//! indices from their candidates' slots, point families and image tiles
//! pull their vertices. Before the map's passes come the DEM prepare, the
//! hillshade tables, image mips, heatmap densities, symbol placements, the
//! terrain's drape slots and the 3D layers' own passes.
//!
//! The map's passes replay the main recording (`MainRecording`): bundles
//! of the nodes' draws (a node a block, or its segments' stretches in tile
//! order), split into steps where a terrain layer, a layer blending other
//! than normally or a 3D layer draws between them. It records again only
//! when something it draws or binds changes: what it waited for arrived
//! (pipelines, or candidates of rows it left out), nodes or their
//! pipelines changed, the scene's bindings or segments moved, targets were
//! made anew. Pipelines come from one cache, looked up by what they draw
//! into the passes at hand (`pipeline`).
//!
//! The `scene` module holds what a frame reads on the GPU: the persistent
//! tables, the arenas of tile streams and the atlases, kept consistent as
//! meshes arrive and leave.

#![forbid(unsafe_code)]

mod atmosphere;
mod awaits;
mod downsample;
mod driver;
mod extrusion;
mod flip;
mod heatmap;
mod image;
mod layer_composite;
mod offscreen;
mod paint;
mod pipeline;
mod release;
mod scene;
mod symbol;
mod targets;
mod terrain;
mod tiles;
mod uniforms;
pub use atmosphere::{AtmospherePaint, AtmosphereView, GlobeSky};
pub use layer_composite::Blend;
pub use targets::{CombineMode, MAX_STOPS, NoiseAnchor, ProcessOp, ProcessPlan, TargetPlan};
pub use terrain::{
    DRAPE_FORMAT, DRAPE_SLOT_PX, DrapeFull, DrapeLayout, MAX_SHRINK, SlotRedraw, TERRAIN_EXTENT,
    TERRAIN_GRID, TerrainDem, TerrainGlobe, TerrainPaint, TerrainTile,
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
    ImageBindings, ImageKey, ImageMetadata, ImageUpload, MeshData, MeshId, MeshSize, OutOfSpace,
    PlacementTarget, RowGlobe, RowSphere, SceneCapacity, SceneStats, TileId, TileRow, TileWarp,
    Words, warp_words,
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

use crate::awaits::{Awaits, Lacks};
use crate::pipeline::{Attachments, Cache, Draw};
use crate::release::Release;
use crate::uniforms::{NODE_UNIFORM_SIZE, NodeUniforms};

#[cfg(cpu_driven)]
#[doc(hidden)]
pub use driver::cpu::{Compared, DriverCheck};
use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_PLANES_SHADER, LINE_SHADER,
    LINE_STYLED_PLANES_SHADER, LINE_STYLED_SHADER, Shader,
};
use rustc_hash::FxHashMap;
#[cfg(cpu_driven)]
use scene::Backing;
use scene::{
    DataTable, NodeCull, NodeDraw, NodeMeshes, Scene, Second, Split, Table, Tables, Untested,
};

/// Which placements of a node draw in pipelines that test no clip
/// ([`Untested`]) on a device: the one rule a node's scene rows, its
/// paint's routing and its pipelines all follow.
#[derive(Clone, Copy, Debug)]
struct Untesting {
    /// Nodes draw untested placements apart: not on a device whose
    /// compiles hold its presents (Firefox's WebGPU), which compiles a
    /// style's whole set before its first frame. There Liberty's first
    /// view compiled cold in 2.10 s, and 2.86 s with the untested
    /// pipelines, for 0.1–0.4 ms of GPU time a frame at 5120×2880; its
    /// paint draws with rows for fewer compiles too ([`Renderer::rowed`]).
    apart: bool,
    /// The device has clip distances, which clip a line to its
    /// placement's rectangle ([`Untested::Rect`]).
    planes: bool,
}

impl Untesting {
    fn of(caps: &Caps) -> Self {
        Self {
            apart: !caps.compiles_hold_presents,
            planes: caps.clip_distances,
        }
    }

    /// Which placements of a `family` node may draw untested, `None` where
    /// it draws them all tested, in one row: a fill's and an image tile's
    /// that draw their whole square, and a line's with one clip rectangle
    /// where the device has clip distances. A node with some splits its
    /// placements into two rows (`scene`'s `Stream::Untested`), and has
    /// an untested pipeline beside its tested one.
    fn of_family(self, family: Family) -> Option<Untested> {
        match family {
            _ if !self.apart => None,
            Family::Fill => Some(Untested::Square),
            Family::Line if self.planes => Some(Untested::Rect),
            family if family.is_image() => Some(Untested::Square),
            _ => None,
        }
    }

    /// How a `family` node draws in two rows, if it does: an extrusion's
    /// cut stream, or its untested placements apart, none of them until
    /// painted ([`split_painted`](Self::split_painted)).
    fn split(self, family: Family) -> Option<Split> {
        match family {
            Family::FillExtrusion => Some(Split::Cut),
            family => (self.of_family(family)).map(|_| Split::Untested(Untested::None)),
        }
    }

    /// A node's `split` once painted with `paint`: a split node's
    /// placements draw untested but while it translates, which moves its
    /// fragments off the placement's clip.
    fn split_painted(self, family: Family, split: Split, paint: &NodePaint) -> Split {
        match (split, self.of_family(family)) {
            (Split::Untested(_), Some(untested)) if !paint.translates() => {
                Split::Untested(untested)
            }
            (Split::Untested(_), _) => Split::Untested(Untested::None),
            (Split::Cut, _) => Split::Cut,
        }
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
    /// No renderable `Rgba16Float`: high-precision render targets, and the
    /// image processes through them, hold 8 bits, which band and round
    /// fine cuts.
    TargetPrecision,
}

impl Gap {
    /// Its name, as events and fixtures give it.
    pub fn name(self) -> &'static str {
        match self {
            Self::HeatmapPrecision => "heatmap-precision",
            Self::DepthPrecision => "depth-precision",
            Self::TargetPrecision => "target-precision",
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
    /// Draws issued: the main recording's (a node with candidates, a
    /// composite, a stack, a blend), and other passes'.
    pub draws: u32,
    /// Whether the main pass was recorded again this frame.
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
    /// The drape passes of slots redrawn: their draws and passes.
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
    /// What of its paint picks its pipelines.
    shape: Shape,
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
}

/// Whether a node painted with `a` draws with the pipelines it does
/// painted with `b`, both paint of its family.
pub fn same_pipelines(a: &NodePaint, b: &NodePaint) -> bool {
    Shape::of(a) == Shape::of(b)
}

/// What of a node's paint picks its pipelines.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Shape {
    /// A line node has dashes, a pattern or a gradient: it draws with the
    /// styled line pipeline.
    styled: bool,
    /// A paint or symbol node's paint reads a feature's property row or
    /// state: it draws with its pipeline's variant with rows.
    rows: bool,
    /// A fill or line node draws a pattern: it draws with its pipeline's
    /// variant with patterns.
    pattern: bool,
    /// A symbol node places its labels along lines.
    along_line: bool,
    /// What an extrusion node's pipelines are specialised for.
    elevation: paint::ExtrusionShape,
}

impl Shape {
    fn of(paint: &NodePaint) -> Self {
        Self {
            styled: paint.styled(),
            rows: paint.reads_rows(),
            pattern: paint.patterned(),
            along_line: matches!(paint, NodePaint::Symbol(p) if p.along_line),
            elevation: paint.extrusion_shape(),
        }
    }
}

impl Node {
    /// A hillshade the terrain pass lights: its stack holds its slopes.
    fn shades_on_terrain(&self) -> bool {
        self.shaded && self.stack.is_some()
    }
}

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
    /// The host's target's format.
    host: TargetFormat,
    /// What the map's passes draw into: the host's target, and while the
    /// style has terrain, which draws in them, a depth attachment, and
    /// with antialiasing on, four samples a pixel (`fit_target`).
    target: Attachments,
    /// The render pipelines that vary with the passes they draw in (the
    /// map's, the 3D layers', the drape's), in every kind of pass they
    /// drew in, looked up by what they draw into the passes at hand.
    pipeline_cache: pipeline::Cache,
    /// That depth attachment, made for the frame's size: never stored.
    map_depth: Option<(TextureId, [u32; 2])>,
    /// The multisampled colour the map's last canvas draws into, where
    /// they take more samples than the host's target, resolved into it:
    /// its size, and whether it's kept between the passes, where they
    /// split at 3D layers.
    map_samples: Option<(TextureId, [u32; 2], bool)>,
    /// The format of the depth the map's and the 3D layers' passes have.
    depth_format: TextureFormat,
    /// Whether 3D surfaces draw at four samples
    /// ([`set_antialias_3d`](Self::set_antialias_3d)).
    antialias_3d: bool,
    /// While the map draws upside down ([`set_flip_y`](Self::set_flip_y)),
    /// the canvas its passes draw into and the draw that turns it over.
    flip: Option<flip::Flip>,
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
    /// The vector families' draws, into the map's passes and the drape's.
    paint: PaintPipelines,
    /// Symbol layers' pipelines, placement target and passes.
    symbols: symbol::Symbols,
    /// Whether the map's views may need the full projection
    /// ([`set_full_projection`](Self::set_full_projection)).
    full_projection: bool,
    /// The style has terrain, on or not yet
    /// ([`set_terrain_reachable`](Self::set_terrain_reachable)).
    terrain_reachable: bool,
    /// The driver failed to make a pipeline the terrain draws with
    /// ([`terrain_failed`](Self::terrain_failed)): kept for the renderer's
    /// life, as its pipelines go with the terrain.
    terrain_failed: bool,
    /// Whether the last frame needed the paint and symbol pipelines' full
    /// variants, not their plane ones (`Variants`): its bundles drew with
    /// them.
    full: bool,
    /// The frame's backgrounds cover every pixel
    /// ([`set_background_fills`](Self::set_background_fills)).
    background_fills: bool,
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
    /// Pipelines the device had compiled when frames last looked for
    /// what arrived.
    made_seen: u32,
    /// What the drape's recorded draws waited for since every slot was
    /// last redrawn. A slot keeps what it drew, whatever later recordings
    /// lack (one made while a node has no candidates lacks nothing for
    /// it), so what of it arrives redraws every slot.
    drape_awaits: Awaits,
    /// The nodes' paint, and their second streams'.
    uniforms: NodeUniforms,
    nodes: Vec<Node>,
    tiles: tiles::Tiles,
    surface_warps: tiles::SurfaceWarps,
    /// The main pass's recording, until something it draws or binds
    /// changes.
    main: Option<MainRecording>,
    /// Node uniform writes since the last frame.
    paints_written: u32,
    images: image::Images,
    /// Mip levels for raster pages, DEM slopes and the drape atlas.
    downsampler: downsample::Downsampler,
    heatmaps: heatmap::Heatmaps,
    extrusions: extrusion::Extrusions,
    /// Runs of nodes composited once (`*-layer-opacity`, `-blend`).
    layers: layer_composite::LayerComposites,
    /// What draws apart from the map composites through these.
    composites: offscreen::Composites,
    /// Render targets, which layers draw into and fills through.
    targets: targets::Targets,
    /// Made for the first terrain.
    terrain: Option<terrain::Terrain>,
    /// The DEM page the terrain reads, as `set_terrain` last gave it.
    terrain_page: Option<PageId>,
    /// Since the device last refused the drape a texture: the tile slots it
    /// held then, which the drape's room keeps to, and the bytes of the
    /// texture refused ([`reserve_drape`](Self::reserve_drape)).
    drape_ceiling: Option<(u32, u64)>,
    /// The drape pass's paint, frames and recorded draws, made for the
    /// first draped paint.
    drape: Option<Drape>,
    /// What each draw of the terrain draws, in style order.
    terrain_layers: Vec<terrain::TerrainLayer>,
    /// Colour and slope stacks of the terrain layers.
    drape_stacks: [u32; 2],
    /// What the scene groups bind.
    scene_bound: SceneKey,
    /// Bound for symbols and circles without terrain.
    no_terrain_depth: TextureId,
    glyphs: GlyphAtlas,
    /// The glyph atlas revision the symbol draws bind.
    glyphs_bound: u32,
    /// Changes whenever what the nodes' draws bind or draw changes: the
    /// bundles of nodes drawn apart record again for another.
    draws_revision: u64,
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
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
        (
            layouts,
            groups,
            self.symbols,
            self.uniforms,
            self.main,
            self.nodes,
        )
            .release(gpu);
        (
            self.pipeline_cache,
            self.atmosphere,
            self.images,
            self.downsampler,
            self.heatmaps,
            self.extrusions,
        )
            .release(gpu);
        (self.layers, self.composites, self.targets).release(gpu);
        (self.terrain, self.drape, self.no_terrain_depth).release(gpu);
        self.flip.release(gpu);
        let samples = self.map_samples.map(|(texture, ..)| texture);
        (self.map_depth.map(|(depth, _)| depth), samples).release(gpu);
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
        if !caps.float_targets {
            gaps.push(Gap::TargetPrecision);
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
        let depth_format = pipeline::depth_format(gpu.caps());
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
        let host = target;
        let target = Attachments::from(target);
        let images = image::Images::new(gpu, draw_layout);
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
            (nearest, no_dem, no_terrain_depth),
            None,
        );

        let symbols = symbol::Symbols::new(
            gpu,
            backing,
            glyphs.textures(no_dem),
            (draw_layout, node_layout),
        );
        let downsampler = downsample::Downsampler::new(gpu);
        let heatmaps = heatmap::Heatmaps::new(draw_layout, node_layout);
        let composites = offscreen::Composites::new(gpu);
        // 3D layers draw into RGBA8, over a depth with a stencil while
        // the map's passes have one.
        let extrusions = extrusion::Extrusions::new(
            gpu.caps(),
            (draw_layout, node_layout),
            (TextureFormat::Rgba8Unorm, depth_format),
        );
        // WebGPU aligns uniform offsets to at most 256 bytes: nodes lie
        // NODE_UNIFORM_SIZE apart, as the symbol shader's halves do.
        let alignment = gpu.caps().limits.min_uniform_buffer_offset_alignment;
        assert!(
            NODE_UNIFORM_SIZE.is_multiple_of(u64::from(alignment)),
            "uniform offsets aligned to {alignment} bytes"
        );
        let usage = node_uniform_usage(gpu.caps());
        let uniforms = NodeUniforms::new(gpu, "node uniforms", capacity.nodes, usage);

        let atmosphere = atmosphere::Atmosphere::new(gpu);
        static RENDERERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            id: RENDERERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            host,
            target,
            pipeline_cache: pipeline::Cache::default(),
            map_depth: None,
            map_samples: None,
            depth_format,
            antialias_3d: false,
            flip: None,
            gaps: Self::gaps_on(gpu.caps()),
            atmosphere,
            scene,
            draw_groups,
            driver,
            draw_layout,
            node_layout,
            paint: PaintPipelines {
                layouts: [draw_layout, node_layout],
                untesting: Untesting::of(gpu.caps()),
            },
            symbols,
            full_projection: false,
            terrain_reachable: false,
            terrain_failed: false,
            full: false,
            background_fills: false,
            pipelines_stale: false,
            pipeline_set: PipelineSet::default(),
            patterned: Vec::new(),
            rowed: Vec::new(),
            set_placings: None,
            set_stale: false,
            set_asked: 0,
            made_seen: 0,
            drape_awaits: Awaits::default(),
            uniforms,
            nodes: Vec::new(),
            tiles: Default::default(),
            surface_warps: Default::default(),
            main: None,
            paints_written: 0,
            images,
            downsampler,
            heatmaps,
            layers: Default::default(),
            targets: targets::Targets::new(gpu),
            composites,
            extrusions,
            terrain: None,
            terrain_page: None,
            drape_ceiling: None,
            drape: None,
            terrain_layers: Vec::new(),
            drape_stacks: [0; 2],
            scene_bound: SceneKey {
                dem: no_dem,
                terrain_depth: no_terrain_depth,
                buffers: 0,
                images: image_bindings.texture,
            },
            no_terrain_depth,
            glyphs_bound: glyphs.revision(),
            draws_revision: 0,
            glyphs,
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

    /// The image atlas's bindings as they are now: the atlas grows pages,
    /// which the next frame binds.
    pub fn set_images(&mut self, bindings: ImageBindings) {
        self.image_bindings = bindings;
    }

    /// The format of the targets it draws into
    /// ([`set_format`](Self::set_format)).
    pub fn format(&self) -> TargetFormat {
        self.host
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
    /// default): 3D layers in their passes, and on terrain, which draws in
    /// the map's passes, those (as GL JS's `antialias` draws its whole
    /// canvas).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        if self.antialias_3d == enabled {
            return;
        }
        self.antialias_3d = enabled;
        // The map's passes where they split at 3D layers take their
        // sample count: four with terrain, as these.
        if self.extrusions.set_antialias(enabled) {
            self.set_stale = true;
            self.forget_recordings(gpu);
        }
        self.fit_target(gpu);
        // Nothing draws at the other sample count now: what draws at one
        // sample (terrain's depth for what it hides) draws there either
        // way.
        let samples = [1, self.host.sample_count, self.extrusions.samples];
        (self.pipeline_cache).retain(gpu, |a| samples.contains(&a.sample_count));
    }

    /// Draws into host targets of `format` from now on, keeping its tiles
    /// and everything else it holds: what draws into the host's target
    /// takes pipelines of the new format, made as the set and the nodes
    /// ask for them, as for antialiasing. Those of the old format stay in
    /// the cache, for a host that goes back to it.
    pub fn set_format(&mut self, gpu: &mut dyn Gpu, format: TargetFormat) {
        if self.host == format {
            return;
        }
        self.host = format;
        if let Some(flip) = self.flip.take() {
            flip.release(gpu);
            self.flip = Some(flip::Flip::new(gpu, &self.composites, format.into()));
        }
        self.fit_target(gpu);
    }

    /// Draws the map upside down into the host's target when `flipped`, or
    /// the right way up (the default): the picture drawn upright, mirrored
    /// top to bottom ([`flip`]). It costs a canvas of the frame's size and
    /// a full-screen draw while on.
    pub fn set_flip_y(&mut self, gpu: &mut dyn Gpu, flipped: bool) {
        match (flipped, self.flip.is_some()) {
            (true, false) => {
                let host = Attachments::from(self.host);
                self.flip = Some(flip::Flip::new(gpu, &self.composites, host));
            }
            (false, true) => release::Release::release(self.flip.take(), gpu),
            _ => {}
        }
    }

    /// What the map's passes draw into while they have their depth, which
    /// the terrain draws in.
    fn terrain_target(&self) -> Attachments {
        Attachments {
            depth: Some(self.depth_format),
            sample_count: match self.antialias_3d {
                true => 4,
                false => self.host.sample_count,
            },
            ..self.host.into()
        }
    }

    /// Has the cull keep every candidate when `on` (`cull.wgsl`'s
    /// `keep_all`): it culls nothing, as a driver without it draws. For
    /// tests that the cull only drops what draws nothing.
    pub fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) {
        if self.driver.set_keep_all(gpu, on) {
            self.forget_recordings(gpu);
        }
    }

    /// Runs the GPU's cull beside the CPU driver's when `on`, over the same
    /// scene and frames into buffers of its own, for a test to hold the
    /// two to the same words ([`driver_check`](Self::driver_check)); the
    /// candidates are laid out again, so both start alike. Only a
    /// CPU-driven renderer on a device with compute and the tables in
    /// storage buffers (`FramePath::Cpu`) has a twin; returns whether this
    /// one does.
    #[cfg(cpu_driven)]
    #[doc(hidden)]
    pub fn check_driver(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        let driver::Driver::Cpu(d) = &mut self.driver else {
            return false;
        };
        if on && (!gpu.caps().compute() || self.scene.tables().backing() != Backing::Storage) {
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

    /// The scene's per-row draw arguments and cull counters, for tests:
    /// [`ARGS_WORDS`] words per cull row, as `cull.wgsl` writes them,
    /// block by block, and the number of rows.
    #[doc(hidden)]
    pub fn draw_args(&self) -> (BufferId, u32) {
        let rows = self.scene.rows() * self.scene.block_slots();
        (self.scene.buffers().args, rows)
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
        let untesting = self.paint.untesting;
        let seconds = families.iter().filter(|&&f| untesting.split(f).is_some());
        let count = (families.len() + seconds.count()) as u32;
        let mut retired_uniforms = None;
        if count > self.scene.capacity().nodes {
            self.scene.grow_nodes(gpu, count);
            retired_uniforms = Some(self.uniforms.grow(gpu, self.scene.capacity().nodes));
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
            let uniform = self.uniforms.binding(i as u32, text);
            if node.is_none() && family.is_image() {
                added.push(i as u32);
            }
            // A retained node keeps what decides its drape stack.
            let (mut image, cull, shaded) = match node {
                Some(node) => {
                    // Groups bind the uniform at the node's index.
                    (
                        (node.group, node.second_group),
                        node.extrusion,
                        node.heatmap,
                    )
                        .release(gpu);
                    (node.image, node.cull, node.shaded)
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
                        split: untesting.split(family),
                        meshes: NodeMeshes::Own,
                    },
                    false,
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
                image,
                separate_halo: false,
                shape: Shape {
                    // A symbol half adds to the pipeline its layer draws
                    // with only once painted: unpainted (its layer not
                    // shown yet, or its part's opacity 0), it draws nothing.
                    rows: !family.is_symbol(),
                    ..Shape::default()
                },
                stack: None,
                terrain_layer: None,
                blends_on_terrain: false,
                shaded,
                composite: None,
                solid: None,
                drawn: false,
            });
        }
        // A symbol layer's draw finds its list by its icon node's index,
        // which its uniform holds even while no paint shows its icons.
        for i in 0..self.nodes.len() {
            if self.nodes[i].family == Family::SymbolIcon {
                let index = orb_shaders::layout::symbol_node::INDEX;
                self.uniforms.set_word(i as u32, index, i as u32);
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
            if let Some(second) = self.scene.second_row(i as u32) {
                let uniform = self.uniforms.binding(second.row(), false);
                self.nodes[i].second_group = Some(self.node_group(gpu, i, uniform));
            }
        }
        self.pipelines_stale = true;
        self.forget_recordings(gpu);
        let complete = self.add_image_nodes(gpu, &added);
        let sources: Vec<u32> = self.tiles.sources().map(|(source, _)| source).collect();
        for source in sources {
            self.drop_unused_page(source);
        }
        self.destroy_retired_pages(gpu);
        complete
    }

    /// A group binding node `i`'s `uniform` and, for an image node, its
    /// pages: one over its uniform at another row, or the drape's.
    fn node_group(&self, gpu: &mut dyn Gpu, i: usize, uniform: BindingResource) -> BindGroupId {
        let node = &self.nodes[i];
        match &node.image {
            Some(image) => self.images.group(gpu, node.family, image, uniform),
            None => gpu.create_bind_group(&BindGroupDesc {
                label: "node",
                layout: self.node_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: uniform,
                }],
            }),
        }
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
    /// the next). Colour stacks come first, in order, then slope stacks,
    /// which the drape keeps on a page of their own ([`DrapeLayout`]).
    fn assign_stacks(&mut self) {
        let stacks = self.stacks_as(self.terrain_on());
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let (stack, layer) = stacks.nodes[i];
            node.stack = stack;
            node.terrain_layer = layer;
            node.blends_on_terrain = stacks.blends_on_terrain(i);
            node.cull.target = match stack {
                Some(stack) => PlacementTarget::Drape(stack),
                None => PlacementTarget::Map,
            };
        }
        self.drape_stacks = stacks.counts;
        self.terrain_layers = stacks.layers;
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

    /// The drape stacks and terrain layers of the current nodes as on
    /// terrain (`on`), or none, as [`assign_stacks`](Self::assign_stacks)
    /// puts them.
    fn stacks_as(&self, on: bool) -> DrapeStacks {
        let inputs: Vec<StackInput> = (self.nodes.iter().enumerate())
            .map(|(i, node)| StackInput {
                draped: on && self.drapes(i),
                group: composite_group(node.composite),
                blend: (node.composite).map_or(Blend::Normal, |(_, _, blend)| blend),
                shaded: node.shaded,
            })
            .collect();
        let mut stacks = drape_stacks(&inputs);
        // A layer's own terrain layer blends in its blend mode.
        for (i, node) in self.nodes.iter().enumerate() {
            if let ((_, Some(layer)), Some((_, _, blend))) = (stacks.nodes[i], node.composite)
                && blend != Blend::Normal
                && !stacks.blends_on_terrain(i)
            {
                stacks.layers[usize::from(layer)].blend = blend;
            }
        }
        stacks
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
            self.fit_target(gpu);
            // Draped nodes draw into the drape, with its pipelines, and
            // in the full variants: the nodes shown take theirs, made
            // ahead into the cache, before the next frame draws.
            self.set_stale = true;
            self.pipelines_stale = true;
            self.restack(gpu);
        }
    }

    /// Stacks the draped nodes again, after what puts them in stacks
    /// changed: their cull and bundles follow.
    fn restack(&mut self, gpu: &mut dyn Gpu) {
        self.assign_stacks();
        self.sync_cull();
        self.forget_recordings(gpu);
        if self.terrain_on() {
            // The terrain layers may draw with pipelines the set's didn't.
            let reads = terrain_reads(self.pipeline_set.shaded, &self.terrain_layers);
            self.request_terrain(gpu, self.target, reads);
        }
    }

    /// The terrain, made for the first that turns on or that the set asks
    /// for ahead.
    fn terrain(&mut self, gpu: &mut dyn Gpu) -> &mut terrain::Terrain {
        let atmosphere = self.atmosphere.uniform;
        let linear = self.images.linear();
        (self.terrain).get_or_insert_with(|| terrain::Terrain::new(gpu, atmosphere, linear))
    }

    /// Sets whether the style has terrain, on or not yet (its DEM source
    /// may still resolve): the pipeline set then holds the terrain's and
    /// the draped nodes' drape pipelines from the first. A map that has
    /// drawn without terrain takes it only once what it draws on terrain
    /// is made ([`terrain_ready_as`](Self::terrain_ready_as)), so no frame
    /// between draws without its layers. Returns whether the terrain is
    /// reachable now: until it is, the map stays as it was.
    pub fn set_terrain_reachable(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        let drew_flat = self.target.depth.is_none() && self.nodes.iter().any(|n| n.drawn);
        let on = on && (self.terrain_reachable || !drew_flat || self.terrain_ready_as(gpu, true));
        if self.terrain_reachable != on {
            self.terrain_reachable = on;
            self.set_stale = true;
        }
        self.fit_target(gpu);
        on
    }

    /// What the map's passes draw into with terrain (`terrain`) or
    /// without: the host's target, and with terrain its depth, at four
    /// samples where antialiasing is on.
    fn kind_target(&self, terrain: bool) -> Attachments {
        match terrain {
            true => self.terrain_target(),
            false => self.host.into(),
        }
    }

    /// Gives the map's passes a depth attachment while the style has
    /// terrain or terrain is on, for the terrain to draw in, at four
    /// samples where antialiasing is on, or none at the host's: every
    /// pipeline they draw with declares them, so the set and the nodes
    /// ask for those of the new kind, made ahead where the terrain waited
    /// for them.
    fn fit_target(&mut self, gpu: &mut dyn Gpu) {
        let target = self.kind_target(self.terrain_reachable || self.terrain_on());
        if self.target == target {
            return;
        }
        self.target = target;
        self.map_depth.take().map(|(depth, _)| depth).release(gpu);
        self.map_samples
            .take()
            .map(|(texture, ..)| texture)
            .release(gpu);
        self.pipelines_stale = true;
        self.set_stale = true;
        if self.targets.any() {
            self.make_composites(gpu, target, offscreen::Mode::Over);
            self.make_stack(gpu, target);
        }
        self.forget_recordings(gpu);
    }

    /// Makes the pipeline of composites in `mode` into the map's passes
    /// as `target`, unless it is made.
    fn make_composites(&mut self, gpu: &mut dyn Gpu, target: Attachments, mode: offscreen::Mode) {
        (self.composites).make((gpu, &mut self.pipeline_cache), target, mode);
    }

    /// Makes the pipeline node `i` (as `view` has it) composites its
    /// style layer once with into the map's passes as `target`, where it
    /// does (below layer opacity 1, or blending other than normally) and
    /// no terrain layer draws it, unless it is made.
    fn make_layer_composite(
        &mut self,
        gpu: &mut dyn Gpu,
        (i, view): (usize, &NodeView),
        target: Attachments,
    ) {
        let node = &self.nodes[i];
        if let (None, Some(_), Some((.., blend))) = (
            view.terrain_layer,
            composite_group(node.composite),
            node.composite,
        ) {
            self.make_composites(gpu, target, offscreen::Mode::Blend(blend));
        }
    }

    /// Makes the pipelines of stacks into the map's passes as `target`,
    /// unless they are made.
    fn make_stack(&mut self, gpu: &mut dyn Gpu, target: Attachments) {
        (self.composites).make_stack((gpu, &mut self.pipeline_cache), target);
    }

    /// Whether every pipeline the map's frames draw with once its terrain
    /// is on (`on`), or once it is gone, is made, asking for those that
    /// aren't: those of its nodes shown so far, its sky and its
    /// composites, as they draw then ([`PassConfig`]), and the pipeline
    /// set's. The map turns its terrain on or drops it only then, so that
    /// no frame between draws without the layers whose pipelines compile.
    pub fn terrain_ready_as(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        // The renderer as it is once terrain is on, its DEM still to come,
        // or gone: on terrain the draped nodes draw into the drape, which
        // the terrain draws, and wait for both.
        let full = self.full_projection || on;
        let config = PassConfig {
            target: self.kind_target(on),
            terrain_on: false,
            terrain_ahead: on,
            full: on,
            variants: [!full, full],
        };
        let stacks = self.stacks_as(on);
        self.request_set_as(gpu, &config, &stacks);
        let mut wanted = Vec::new();
        for i in 0..self.nodes.len() {
            if !self.scene_cull(i).visible && !self.nodes[i].drawn {
                continue;
            }
            let view = self.view_in(i, |n| self.nodes[n].shape, &stacks);
            self.make_for_node(gpu, (i, &view), &config);
            self.make_layer_composite(gpu, (i, &view), config.target);
            self.node_pipelines((i, &view), &config, &mut wanted);
        }
        if self.targets.any() {
            self.make_composites(gpu, config.target, offscreen::Mode::Over);
            self.make_stack(gpu, config.target);
            wanted.extend(self.targets.pipelines());
        }
        let cache = &self.pipeline_cache;
        wanted.extend(self.composites.pipelines(cache, config.target));
        wanted.extend(self.atmosphere.pipelines(cache, config.target));
        wanted.into_iter().all(|id| gpu.pipeline_ready(id))
    }

    /// Whether the driver failed to make a pipeline the terrain draws
    /// with, where the device lives on (the web's): the terrain can't
    /// draw on this renderer's device, nor what drapes on it, so the map
    /// draws flat. Once it has, it stays so for the renderer's life.
    pub fn terrain_failed(&mut self, gpu: &dyn Gpu) -> bool {
        if !self.terrain_failed {
            let failed = self.terrain_pipelines().any(|p| gpu.pipeline_failed(p));
            self.terrain_failed = failed;
        }
        self.terrain_failed
    }

    /// The pipelines the terrain draws with, made so far: its own, its
    /// drape's mips' and its DEM tiles' preparation.
    fn terrain_pipelines(&self) -> impl Iterator<Item = PipelineId> + '_ {
        let at = (self.terrain_target(), self.extrusions.format());
        (self.terrain.iter())
            .flat_map(move |t| t.pipelines(&self.pipeline_cache, at))
            .chain(self.downsampler.pipelines())
            .chain(self.images.prepare_pipelines())
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

    /// Whether the terrain can draw this frame: its pipelines and the
    /// drape mips' compiled, those asked for here if they weren't (the
    /// drape's first texture would, after the plan). Until then it draws
    /// nothing, so the map plans no drape: one planned would lay out and
    /// place the whole cover each frame for passes that drop it. Only a
    /// frame whose drape was planned ([`reserve_drape`](Self::reserve_drape))
    /// draws the terrain.
    pub fn terrain_drawable(&mut self, gpu: &mut dyn Gpu) -> bool {
        let Some(terrain) = &self.terrain else {
            return false;
        };
        self.downsampler.make(gpu);
        terrain.ready(gpu, (&self.pipeline_cache, self.target)) && self.downsampler.ready(gpu)
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
    /// stacks: the most a terrain cover may have. Since the device refused
    /// a texture, the tiles the drape held then
    /// ([`lift_drape_ceiling`](Self::lift_drape_ceiling)).
    pub fn drape_room(&self, gpu: &dyn Gpu) -> u32 {
        let room = terrain::room(gpu, self.drape_stacks());
        self.drape_ceiling
            .map_or(room, |(slots, _)| room.min(slots))
    }

    /// The tile slots each drape texture holds with this style's stacks
    /// ([`DrapeLayout::shared`]).
    pub fn drape_shared(&self, gpu: &dyn Gpu) -> u32 {
        terrain::shared(gpu, self.drape_stacks())
    }

    /// Lets the drape ask the device for textures past those it held when
    /// the device refused one, as a map whose memory settled does:
    /// whether it kept to fewer.
    pub fn lift_drape_ceiling(&mut self) -> bool {
        self.drape_ceiling.take().is_some()
    }

    /// Makes the drape pages for this style's stacks, a slot per terrain
    /// tile and stack, their textures made as slots need them
    /// ([`redraw_drape`](Self::redraw_drape)). Pages of other stacks are
    /// new and blank (a new layout generation). Fails where the stacks
    /// leave no [`drape_room`](Self::drape_room), or since the device
    /// refused a texture with none held.
    pub fn reserve_drape(&mut self, gpu: &mut dyn Gpu) -> Result<DrapeLayout, DrapeFull> {
        if let Some((0, bytes)) = self.drape_ceiling {
            return Err(DrapeFull::Refused(bytes));
        }
        let stacks = self.drape_stacks();
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let made = terrain.reserve(gpu, stacks)?;
        let layout = terrain.layout();
        if made {
            // Stack layers and the main bundle's composites follow it.
            self.forget_recordings(gpu);
        }
        Ok(layout)
    }

    /// Redraws these terrain tiles' drape slots this frame, each `(tile
    /// slot, shrink, stacks)` at its side, [`DRAPE_SLOT_PX`] halved
    /// `shrink` times, in the stacks given, as
    /// a map at style zoom `zoom` draws: their drape placements are in the
    /// scene, and every draped node's paint was set at that zoom
    /// ([`set_drape_paint`](Self::set_drape_paint)). The drape pass's
    /// frame takes it, for what its shaders blend by the zoom. With the
    /// frame's terrain `tiles`, whose slots alone keep their texels, and
    /// before [`set_terrain_tiles`](Self::set_terrain_tiles) gives them,
    /// which draw from the texels slots take here. Fails where the device
    /// refuses a texture, which leaves every slot blank: the room then
    /// keeps to the tiles held before, so that the next frame's cover fits
    /// what the device gave, until
    /// [`lift_drape_ceiling`](Self::lift_drape_ceiling).
    pub fn redraw_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        zoom: f64,
        (tiles, tile_slots): (&[TerrainTile], &[SlotRedraw<'_>]),
    ) -> Result<(), DrapeFull> {
        let Some(terrain) = &mut self.terrain else {
            return Ok(());
        };
        let held = terrain.held_slots();
        let slots = (tiles, tile_slots);
        let drawn = terrain.redraw(gpu, zoom, slots, &mut self.downsampler);
        if let Err(DrapeFull::Refused(bytes)) = drawn {
            self.drape_ceiling = Some((held, bytes));
        }
        drawn
    }

    /// Lets go of the drape's textures no slot holds: in a frame that settles
    /// the map, which then holds what a fresh one there does.
    pub fn trim_drape(&mut self, gpu: &mut dyn Gpu) {
        if let Some(terrain) = &mut self.terrain {
            terrain.trim(gpu);
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
        let rows = self.scene.capacity().nodes;
        let second = self.scene.second_row(node);
        let nodes = self.nodes.len();
        let family = self.nodes[node as usize].family;
        let drape = self.drape.get_or_insert_with(|| Drape {
            uniforms: NodeUniforms::new(
                gpu,
                "drape node uniforms",
                rows,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            drawn: Vec::new(),
            slot_scenes: Default::default(),
            recorded: Vec::new(),
        });
        drape.drawn.resize(nodes, false);
        let was = std::mem::replace(&mut drape.drawn[node as usize], paint.is_some());
        if was != paint.is_some() {
            drape.forget_recorded(gpu);
        }
        let Some(paint) = paint else {
            return was;
        };
        assert!(paint.fits(family), "paint family mismatch for node {node}");
        let words = paint.uniform_words(node);
        let changed = paint_rows(&mut drape.uniforms, (node, second), (paint, words));
        self.paints_written += u32::from(changed);
        changed || !was
    }

    /// This frame's terrain tiles.
    pub fn set_terrain_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.set_tiles(gpu, tiles);
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
                self.forget_recordings(gpu);
            }
        } else if before.is_some() {
            self.layers.set_opacity(gpu, node as usize, opacity);
        }
    }

    /// Records the main pass again: the map's passes', the composited
    /// layers' and the render targets' bundles.
    fn forget_main(&mut self, gpu: &mut dyn Gpu) {
        self.main.take().release(gpu);
    }

    /// Records the main pass and the drape's draws again: after what both
    /// draw or bind changed.
    fn forget_recordings(&mut self, gpu: &mut dyn Gpu) {
        self.forget_main(gpu);
        if let Some(drape) = &mut self.drape {
            drape.forget_recorded(gpu);
        }
    }

    /// Records the drape's draws again and redraws every slot (stale) with
    /// them: what the slots waited for arrived, or they lack a node.
    fn stale_drape(&mut self, gpu: &mut dyn Gpu) {
        if let Some(drape) = &mut self.drape {
            drape.forget_recorded(gpu);
        }
        if let Some(terrain) = self.terrain.as_mut() {
            terrain.redraw_all();
        }
        self.drape_awaits = Awaits::default();
    }

    /// Records every bundle that draws nodes again, extrusions' and
    /// heatmaps' too (by [`draws_revision`](Self::draws_revision)): after
    /// their bindings or their draws changed.
    fn forget_draws(&mut self, gpu: &mut dyn Gpu) {
        self.forget_recordings(gpu);
        self.draws_revision += 1;
    }

    /// Sets what the style draws through render targets (the
    /// `dev.sargunv.orbis:targets` preview), by node index: the nodes'
    /// targets, the fills drawn through them and the image-process layers.
    /// The bundles record again for a new plan.
    pub fn set_targets(&mut self, gpu: &mut dyn Gpu, plan: TargetPlan) {
        if self.targets.set_plan(gpu, &self.composites, plan) {
            if self.targets.any() {
                self.make_composites(gpu, self.target, offscreen::Mode::Over);
                self.make_stack(gpu, self.target);
            }
            self.restack(gpu);
        }
    }

    /// Sets an image-process layer's operations as the frame's zoom has
    /// them.
    pub fn set_process_ops(&mut self, process: u32, ops: Vec<ProcessOp>) {
        self.targets.set_ops(process as usize, ops);
    }

    /// Shows or hides an image-process layer of the target plan: a hidden
    /// one draws nothing.
    pub fn set_process_visible(&mut self, process: u32, visible: bool) {
        self.targets.set_visible(process as usize, visible);
    }

    /// A page of tile textures, `size` texels per tile, for image nodes to
    /// bind.
    pub fn create_page(&mut self, gpu: &mut dyn Gpu, kind: PageKind, size: [u32; 2]) -> PageId {
        let layers = self.page_layers(size);
        self.images.create_page(gpu, kind, size, layers)
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
            let uniform = self.uniforms.binding(i as u32, false);
            let second = (self.scene.second_row(i as u32))
                .map(|second| self.uniforms.binding(second.row(), false));
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
            self.forget_recordings(gpu);
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
        let i = node as usize;
        let family = self.nodes[i].family;
        assert!(paint.fits(family), "paint family mismatch for node {node}");
        // A fill drawn through a target in a colour alone composites that
        // colour through it, drawing nothing of its own.
        let color = paint
            .solid_color()
            .filter(|_| matches!(paint, NodePaint::Fill(_)));
        if self.targets.set_color(gpu, i, color) {
            self.forget_recordings(gpu);
        } else if let Some(color) = color {
            self.layers.set_color(gpu, i, color);
            let layer = (self.targets.solid(i)).and_then(offscreen::Composite::layer);
            let stacks = self.main.iter_mut().flat_map(|m| &mut m.stacks);
            for (stack, fills) in stacks {
                for (k, _) in fills.iter().enumerate().filter(|(_, f)| **f == Some(i)) {
                    let layer = layer.expect("a fill a stack draws composites a colour");
                    offscreen::Composites::set_layer(gpu, stack, k, layer);
                }
            }
        }
        let background = matches!(paint, NodePaint::Background(_));
        self.nodes[i].solid = (paint.solid_color()).filter(|c| background && c[3] == 1.0);
        let shape = Shape::of(paint);
        if shape != self.nodes[i].shape {
            self.nodes[i].shape = shape;
            if self.nodes[self.drawer(i)].drawn {
                self.pipelines_stale = true;
                self.forget_recordings(gpu);
            }
        }
        let mut words = paint.uniform_words(node);
        if let NodePaint::Symbol(p) = paint {
            use orb_shaders::layout::symbol_node as n;
            words[n::GLYPH_PAGE] = (GLYPH_PAGE as f32).to_bits();
            words[n::IMAGE_PAGE] = (self.image_bindings.size as f32).to_bits();
            // Its halos draw in a pass of their own, which the bundles
            // record.
            if std::mem::replace(&mut self.nodes[i].separate_halo, p.separate_halo)
                != p.separate_halo
            {
                self.forget_main(gpu);
            }
        }
        let second = self.scene.second_row(node);
        let changed = paint_rows(&mut self.uniforms, (node, second), (paint, words));
        self.paints_written += u32::from(changed);
        let entry = &mut self.nodes[i];
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
        let cull = NodeCull {
            margin_px: paint.cull_margin_px(),
            elevation: match paint {
                NodePaint::Extrusion(p) | NodePaint::ExtrusionPattern(p, _) => {
                    p.height.max(p.base).max(0.0)
                }
                _ => 0.0,
            },
            surface: paint.on_surface(),
            split: (entry.cull.split).map(|s| self.paint.untesting.split_painted(family, s, paint)),
            ..entry.cull
        };
        if cull != entry.cull {
            entry.cull = cull;
            self.sync_node(i);
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

    /// The symbol pipeline a symbol layer of `shape`
    /// ([`layer_shape`](Self::layer_shape)) draws both its halves with:
    /// with property rows where it reads them or another does and `rowed`
    /// says so ([`Renderer::rowed`]).
    fn symbol_kind(&self, shape: Shape) -> symbol::SymbolKind {
        let rowed = self.rowed.iter().any(|f| f.is_symbol());
        symbol::SymbolKind {
            along_line: shape.along_line,
            rows: shape.rows || rowed,
        }
    }

    /// The paint pipeline a node of `family` and `shape` draws with,
    /// `patterned` where another node of its family draws a pattern
    /// ([`Renderer::patterned`]), with rows where another reads them and
    /// `rowed` says so ([`Renderer::rowed`]).
    fn paint_kind(&self, family: Family, shape: Shape) -> PaintKind {
        PaintKind {
            family,
            styled: shape.styled,
            rows: shape.rows || self.rowed.contains(&family),
            pattern: shape.pattern || self.patterned.contains(&family),
        }
    }

    /// The placement dispatch's kind: with line placement if any symbol
    /// layer places along lines, and property rows if any reads them, at
    /// any zoom of the style's set or as its nodes are now (`None` without
    /// symbol layers): one pipeline, whatever the zoom.
    fn placings(&self) -> symbol::Placings {
        (self.nodes.iter().enumerate())
            .filter(|(_, n)| n.family == Family::SymbolIcon)
            .map(|(i, _)| self.symbol_kind(self.shape(i)))
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

    /// The composite of fill `i` in its colour through its render target,
    /// where it draws a colour alone, normally blended at layer opacity 1:
    /// otherwise its layer composites it, at its opacity, in its mode.
    fn solid_through(&self, i: usize) -> Option<&offscreen::Composite> {
        let grouped = composite_group(self.nodes[i].composite).is_some();
        self.targets.solid(i).filter(|_| !grouped)
    }

    /// The node the main pass can clear to in place of drawing it and
    /// what lies beneath it ([`MainRecording::beneath`]): the topmost
    /// solid background, when nothing beneath it composites apart, which
    /// could split the pass. Only paint chooses it, so neither a camera
    /// move nor showing or hiding a layer records the bundles again.
    fn solid_beneath(&self) -> Option<usize> {
        // Not one drawn into a render target, which never reaches the map.
        let i = (self.nodes.iter().enumerate()).rposition(|(i, n)| {
            n.solid.is_some()
                && n.drawn
                && composite_group(n.composite).is_none()
                && !self.targets.routes(i)
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
        let cache = &mut self.pipeline_cache;
        self.atmosphere
            .set((gpu, cache), target, frame, paint, view);
    }

    /// Whether a layer can composite at a layer opacity, asking for the
    /// pipeline it takes: a frame that would wait for it draws nothing,
    /// so what fades through it starts once it compiled.
    pub fn layer_composite_ready(&mut self, gpu: &mut dyn Gpu) -> bool {
        let (mode, target) = (offscreen::Mode::Over, self.target);
        self.make_composites(gpu, target, mode);
        let needs = self.composites.needs(&self.pipeline_cache, target, mode);
        awaits::compiled(gpu, [needs])
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
            self.forget_recordings(gpu);
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
        let config = self.config();
        for i in 0..self.nodes.len() {
            if !self.scene_cull(i).visible && !self.nodes[i].drawn {
                continue;
            }
            let view = self.view(i);
            self.make_for_node(gpu, (i, &view), &config);
            self.make_layer_composite(gpu, (i, &view), config.target);
        }
    }

    /// Asks for the set's pipelines that aren't made, as the map draws
    /// now: the first view's, its labels' first ([`labels_first`]), then
    /// the rest in the background. On a style whose terrain isn't on yet,
    /// they are the drape's and the terrain's: its draped nodes wait for
    /// it.
    fn request_set(&mut self, gpu: &mut dyn Gpu) {
        let config = self.config();
        let stacks = self.stacks_as(self.terrain_on() || self.terrain_ahead());
        // One placement pipeline places every symbol layer, at every zoom.
        let placings = (self.pipeline_set.groups.iter())
            .flat_map(|group| self.group_views(&group.nodes, &stacks))
            .filter(|&(i, _)| self.nodes[i].family == Family::SymbolIcon)
            .map(|(_, view)| self.symbol_kind(view.shape))
            .reduce(|a, b| a | b);
        self.set_placings = placings;
        self.request_set_as(gpu, &config, &stacks);
    }

    /// Asks for the set's pipelines that aren't made, as the map draws
    /// as `config` and its nodes drape in `stacks`, as
    /// [`request_set`](Self::request_set) orders them.
    fn request_set_as(&mut self, gpu: &mut dyn Gpu, config: &PassConfig, stacks: &DrapeStacks) {
        let asked = gpu.counters().pipelines_requested;
        let set = std::mem::take(&mut self.pipeline_set);
        let mut order: Vec<&PipelineGroup> = set.groups.iter().collect();
        // The first view's labels first ([`labels_first`]).
        let labels = |g: &PipelineGroup| {
            (g.nodes.iter()).any(|&(i, _)| self.nodes.get(i as usize).is_some_and(labels_first))
        };
        order.sort_by_key(|g| (g.rank, !labels(g)));
        for group in order {
            let first = group.rank == 0;
            gpu.set_background_compiles(!first);
            for (i, view) in self.group_views(&group.nodes, stacks) {
                self.make_for_node(gpu, (i, &view), config);
            }
            if let Some(blend) = group.composite {
                self.make_composites(gpu, config.target, offscreen::Mode::Blend(blend));
            }
        }
        gpu.set_background_compiles(false);
        if set.sky {
            // With the first view's: a tilt that brings the horizon in
            // shows the sky at once.
            let frame = self.scene.buffers().frame;
            let target = (config.target, self.full_projection);
            let cache = &mut self.pipeline_cache;
            self.atmosphere.request((gpu, cache), target, frame, false);
        }
        if config.terrain_ahead || config.terrain_on {
            let reads = terrain_reads(set.shaded, &stacks.layers);
            self.request_terrain(gpu, config.target, reads);
        }
        self.pipeline_set = set;
        self.set_asked += gpu.counters().pipelines_requested - asked;
    }

    /// Asks for the terrain's pipelines, a shaded hillshade's where a
    /// hillshade may shade (`shaded`), the composite of a terrain layer
    /// that reads the map beneath it into the map's passes as `target`
    /// (`beneath`, as [`terrain_reads`] tells both), its depth's in the 3D
    /// layers' passes where the style has a 3D layer, and the drape
    /// atlas's mips.
    fn request_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        target: Attachments,
        (shaded, beneath): (bool, bool),
    ) {
        let three_d = (self.nodes.iter())
            .any(|n| n.family == Family::FillExtrusion)
            .then(|| self.extrusions.format());
        if shaded {
            self.images.make_slopes(gpu);
        }
        if beneath {
            // Its group's composite carries the map over to the pass after
            // the split.
            self.make_composites(gpu, target, offscreen::Mode::Blend(Blend::Multiply));
        }
        self.terrain(gpu);
        let terrain = self.terrain.as_ref().expect("made");
        let cache = &mut self.pipeline_cache;
        terrain.request_pipelines((gpu, cache), (target, three_d), (shaded, beneath));
        self.downsampler.make(gpu);
        self.images.make_prepare(gpu);
    }

    /// The nodes that draw `group` of a pipeline set (a symbol layer's
    /// icon node draws both halves) as its paint there picks their
    /// pipelines and they drape in `stacks`: none where a node isn't
    /// this renderer's or of its paint's family, and the group is left
    /// out.
    fn group_views(
        &self,
        group: &[(u32, Option<NodePaint>)],
        stacks: &DrapeStacks,
    ) -> Vec<(usize, NodeView)> {
        let fits = |&(i, ref paint): &(u32, Option<NodePaint>)| {
            let node = self.nodes.get(i as usize);
            node.is_some_and(|n| paint.as_ref().is_none_or(|p| p.fits(n.family)))
        };
        if !group.iter().all(fits) {
            return Vec::new();
        }
        let shape = |n: usize| {
            let shape = self.nodes[n].shape;
            match group.iter().find(|&&(i, _)| i as usize == n) {
                Some((_, Some(paint))) => Shape::of(paint),
                // A symbol half that draws nothing reads nothing.
                Some((_, None)) if self.nodes[n].family.is_symbol() => Shape {
                    rows: false,
                    along_line: false,
                    ..shape
                },
                _ => shape,
            }
        };
        let mut drawers: Vec<usize> = (group.iter())
            .map(|&(i, _)| self.drawer(i as usize))
            .collect();
        drawers.dedup();
        (drawers.into_iter())
            .map(|i| (i, self.view_in(i, shape, stacks)))
            .collect()
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
            self.forget_recordings(gpu);
        }
        if draped {
            // Slots drawn without them lack those layers.
            self.stale_drape(gpu);
        }
    }

    /// How the map draws now, besides its nodes' paint and stacks.
    fn config(&self) -> PassConfig {
        PassConfig {
            target: self.target,
            terrain_on: self.terrain_on(),
            terrain_ahead: self.terrain_ahead(),
            full: self.full,
            variants: self.variants(),
        }
    }

    /// Node `i` as it is now.
    fn view(&self, i: usize) -> NodeView {
        let node = &self.nodes[i];
        NodeView {
            shape: self.shape(i),
            stack: node.stack,
            terrain_layer: node.terrain_layer,
        }
    }

    /// Node `i` with each node's shape as `shape` gives it, draping as
    /// `stacks` puts it.
    fn view_in(&self, i: usize, shape: impl Fn(usize) -> Shape, stacks: &DrapeStacks) -> NodeView {
        let (stack, terrain_layer) = stacks.nodes[i];
        NodeView {
            shape: self.layer_shape(i, shape),
            stack,
            terrain_layer,
        }
    }

    /// What of node `i`'s paint picks its pipelines now
    /// ([`layer_shape`](Self::layer_shape)).
    fn shape(&self, i: usize) -> Shape {
        self.layer_shape(i, |n| self.nodes[n].shape)
    }

    /// What of node `i`'s paint picks its pipelines, with each node's as
    /// `shape` gives it: a symbol node's is its layer's, whose two halves
    /// draw with one pipeline, along lines if either places along them,
    /// with property rows if either reads them.
    fn layer_shape(&self, i: usize, shape: impl Fn(usize) -> Shape) -> Shape {
        if !self.nodes[i].family.is_symbol() {
            return shape(i);
        }
        let icons = self.drawer(i);
        let [icons, text] = [shape(icons), shape(icons + 1)];
        Shape {
            rows: icons.rows || text.rows,
            along_line: icons.along_line || text.along_line,
            ..icons
        }
    }

    /// What node `i` draws with, as `view` has it and the map draws as
    /// `config`: the one answer that making its pipelines
    /// ([`make_for_node`](Self::make_for_node)), waiting for and
    /// prioritising them ([`node_pipelines`](Self::node_pipelines)) and
    /// recording its draws ([`node_commands`](Self::node_commands)) all
    /// read.
    fn node_draws(&self, (i, view): (usize, &NodeView), config: &PassConfig) -> NodeDraws {
        let family = self.nodes[i].family;
        let drape = view.stack.is_some();
        let with = match family {
            // A hillshade the terrain pass lights: its stack holds its
            // slopes.
            _ if drape && self.nodes[i].shaded => DrawsWith::Slopes,
            // It drapes once the style's terrain is on: until then it waits
            // for that, rather than for pipelines it would draw flat with.
            _ if !drape && config.terrain_ahead && self.drapes(i) => DrawsWith::Terrain,
            Family::Heatmap => DrawsWith::Heatmap,
            Family::FillExtrusion => DrawsWith::Extrusion(view.shape.elevation),
            Family::SymbolIcon | Family::SymbolText => {
                DrawsWith::Symbols(self.symbol_kind(view.shape), self.placings())
            }
            family if family.is_image() => DrawsWith::Image { family, drape },
            _ => DrawsWith::Paint {
                kind: self.paint_kind(family, view.shape),
                drape,
            },
        };
        NodeDraws {
            with,
            prepare: matches!(family, Family::Hillshade | Family::ColorRelief),
            tables: family == Family::Hillshade,
            mips: family == Family::Raster,
        }
    }

    /// Whether node `i` drapes on terrain: a draped family's, but not
    /// what draws into or through a render target, in screen space on the
    /// plane.
    fn drapes(&self, i: usize) -> bool {
        self.nodes[i].family.is_draped() && !self.targets.routes(i)
    }

    /// Makes node `i`'s pipelines as `view` and `config` have it draw
    /// ([`node_draws`](Self::node_draws)).
    fn make_for_node(&mut self, gpu: &mut dyn Gpu, node: (usize, &NodeView), config: &PassConfig) {
        let draws = self.node_draws(node, config);
        if draws.prepare {
            self.images.make_prepare(gpu);
        }
        if draws.tables {
            // Which its draws and the terrain's shading sample.
            self.images.make_tables(gpu);
        }
        if draws.mips {
            self.downsampler.make(gpu);
        }
        let (target, variants) = (config.target, config.variants);
        let cache = &mut self.pipeline_cache;
        match draws.with {
            DrawsWith::Terrain => {}
            DrawsWith::Slopes => {
                self.images.make_slopes(gpu);
            }
            DrawsWith::Paint { kind, drape } => {
                let target = drape_or(drape, target);
                self.paint.make((gpu, cache), target, kind, variants);
            }
            DrawsWith::Image { family, drape } => {
                self.images
                    .make((gpu, cache), drape_or(drape, target), family);
            }
            DrawsWith::Symbols(kind, placings) => {
                (self.symbols).make((gpu, cache), target, (kind, placings), variants);
            }
            DrawsWith::Heatmap => {
                self.heatmaps.make(gpu);
                self.make_composites(gpu, target, offscreen::Mode::Ramp);
            }
            DrawsWith::Extrusion(shape) => {
                let stencil = target.depth.is_some();
                (self.extrusions).make((gpu, cache), (shape, stencil), variants);
                self.make_composites(gpu, target, offscreen::Mode::Over);
            }
        }
    }

    /// The pipelines node `i` draws with as `view` and `config` have it
    /// ([`node_draws`](Self::node_draws)), made or not.
    fn node_pipelines(
        &self,
        node: (usize, &NodeView),
        config: &PassConfig,
        out: &mut Vec<PipelineId>,
    ) {
        let draws = self.node_draws(node, config);
        if draws.drapes() && self.terrain.is_some() {
            // Its terrain layer draws it, which draws nothing until its
            // drape's mips and its DEM tiles' preparation can run.
            out.extend(self.terrain_pipelines());
        }
        let (cache, target, full) = (&self.pipeline_cache, config.target, config.full);
        match draws.with {
            DrawsWith::Terrain => {}
            DrawsWith::Slopes => out.extend(self.images.slopes_pipeline().map(PipelineId::from)),
            DrawsWith::Paint { kind, drape } => {
                let at = (cache, drape_or(drape, target));
                out.extend(self.paint.wanted(at, kind, full).map(PipelineId::from));
            }
            DrawsWith::Image { family, drape } => {
                let image = self
                    .images
                    .pipeline(cache, drape_or(drape, target), family, false);
                out.extend(image.map(PipelineId::from));
            }
            DrawsWith::Symbols(kind, placings) => {
                out.extend((self.symbols).wanted((cache, target), (kind, placings), full));
            }
            DrawsWith::Heatmap => out.extend(self.heatmaps.pipelines()),
            DrawsWith::Extrusion(shape) => {
                let stencil = target.depth.is_some();
                out.extend(self.extrusions.wanted(cache, (shape, stencil), full));
            }
        }
        if draws.drapes() || matches!(draws.with, DrawsWith::Heatmap | DrawsWith::Extrusion(_)) {
            out.extend(self.composites.pipelines(cache, target));
        }
        if draws.prepare {
            out.extend(self.images.prepare_pipelines());
        }
        if draws.tables {
            out.extend(self.images.tables_pipeline().map(PipelineId::from));
        }
        if draws.mips {
            out.extend(self.downsampler.pipelines());
        }
    }

    /// Moves the pipelines of nodes with meshes in the scene ahead of the
    /// others still compiling: what the current view needs first. The
    /// cull goes before everything, then the draw that turns an upside
    /// down map over (without it nothing shows), then labels
    /// ([`labels_first`]), then the other nodes in draw order (the bottom
    /// layers cover most of the view).
    fn prioritize(&self, gpu: &mut dyn Gpu) {
        let mut wanted = Vec::new();
        wanted.extend(self.driver.pipelines());
        wanted.extend(self.flip.as_ref().map(flip::Flip::pipeline));
        let config = self.config();
        for labels in [true, false] {
            for (i, node) in self.nodes.iter().enumerate() {
                if labels_first(node) == labels && self.scene.node_has_meshes(i as u32) {
                    self.node_pipelines((i, &self.view(i)), &config, &mut wanted);
                }
            }
        }
        wanted.extend(
            self.atmosphere
                .pipelines(&self.pipeline_cache, config.target),
        );
        // Each moves to the front: the last asked for compiles first.
        let mut seen = rustc_hash::FxHashSet::default();
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
        self.uniforms.flush(gpu);
        if let Some(drape) = &mut self.drape {
            drape.uniforms.flush(gpu);
        }
        let layout = self.terrain.as_ref().map(terrain::Terrain::layout);
        let requested = gpu.counters().pipelines_requested;
        let mut stats = self.record(gpu, frame, target, clear, clock);
        debug_assert_eq!(
            gpu.counters().pipelines_requested,
            requested,
            "a frame asked for a pipeline"
        );
        // Drape slots made stale (pipelines a slot waited for arrived) or
        // blank (terrain couldn't draw yet) during the frame draw again
        // from the next: this one shows them without what they lacked.
        stats.waited |= self.terrain.as_ref().map(terrain::Terrain::layout) != layout;
        stats.draws += u32::from(self.atmosphere.enabled);
        self.end_surface_warps();
        stats
    }

    /// Binds what the scene groups bind anew when it changed
    /// ([`SceneKey`]), once a frame, after everything that can change it;
    /// and the glyph pages' chunks anew in the symbol draws' groups when
    /// the pages grew.
    fn bind_scene(&mut self, gpu: &mut dyn Gpu) {
        if self.glyphs.revision() != self.glyphs_bound {
            self.glyphs_bound = self.glyphs.revision();
            let glyphs = self.glyphs.textures(self.images.placeholder());
            if self.symbols.set_glyphs(gpu, glyphs) {
                self.forget_main(gpu);
            }
        }
        let terrain = self.terrain.as_ref();
        let key = SceneKey {
            dem: (terrain.and_then(terrain::Terrain::dem_texture))
                .unwrap_or(self.images.placeholder()),
            // Symbols and circles read the terrain's depth.
            terrain_depth: (terrain.and_then(terrain::Terrain::occlusion_texture))
                .unwrap_or(self.no_terrain_depth),
            buffers: self.scene.buffers_revision(),
            images: self.image_bindings.texture,
        };
        if key == self.scene_bound {
            return;
        }
        if key.buffers != self.scene_bound.buffers {
            self.driver.rebind(gpu, &self.scene);
        }
        self.scene_bound = key;
        std::mem::take(&mut self.draw_groups).release(gpu);
        self.draw_groups = scene_groups(
            gpu,
            self.draw_layout,
            &self.scene,
            &self.image_bindings,
            (self.images.nearest(), key.dem, key.terrain_depth),
            None,
        );
        if let Some(drape) = self.drape.as_mut() {
            std::mem::take(&mut drape.slot_scenes).release(gpu);
        }
        // Extrusion and heatmap nodes record the scene group too.
        self.forget_draws(gpu);
    }

    /// What the symbol placements bind of the scene, as
    /// `symbol::compute_layout` has it, by block slot: the block's frame,
    /// the tables, the terrain as the scene group binds it, the node
    /// uniforms and the draw arguments.
    fn symbol_scene(&self) -> Vec<Option<Vec<BindGroupEntry>>> {
        let held: Vec<u32> = self.scene.blocks().collect();
        (0..self.scene.block_slots())
            .map(|block| {
                if !held.contains(&block) {
                    return None;
                }
                let frame = self.scene.block_frame(block);
                let mut entries = vec![BindGroupEntry::buffer(0, frame)];
                entries.extend(self.scene.tables().entries(block as usize));
                entries.extend([
                    BindGroupEntry::texture(14, self.scene_bound.dem),
                    BindGroupEntry::texture(15, self.scene_bound.terrain_depth),
                    BindGroupEntry::buffer(17, self.uniforms.buffer),
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
        // Nodes that draw in stretches of one block's slots record them,
        // extrusions' passes too (by the revision they record over). The
        // drape's recorded draws take each slot's stretches as its pass
        // draws ([`Renderer::slot_commands`]): they, the nodes they left
        // out and the slots' bundles stay.
        let [segments, _] = self.scene.segments_revision();
        if self.main.as_ref().is_some_and(|m| m.segments != segments) {
            self.forget_main(gpu);
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
            self.forget_main(gpu);
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
        let size = self.resize_targets(gpu, frame);
        // Bound once the targets are sized: a resize remade the terrain's
        // depth the scene group samples, and preparing may have grown the
        // compacted index buffer. Bound before, the frame's bundles would
        // sample the destroyed depth and the submit fail.
        self.bind_scene(gpu);
        let beneath = self.solid_beneath();
        if self.main.as_ref().is_some_and(|m| m.beneath != beneath) {
            self.forget_main(gpu);
        }
        self.symbols.bind(gpu, self.symbol_scene());
        let bundle_recorded = self.main.is_none();
        let recording = clock();
        if bundle_recorded {
            self.main = Some(self.record_main(gpu, beneath));
        }
        let draping = clock();
        times.bundles = micros(recording, draping);
        let slots = match self.terrain.as_ref().filter(|_| self.terrain_on()) {
            Some(terrain) => terrain.pending_slots(),
            None => Vec::new(),
        };
        let mut stacks: Vec<u32> = slots.iter().map(|&(stack, _)| stack).collect();
        stacks.dedup();
        for stack in stacks {
            self.record_drape(gpu, stack);
        }
        self.slot_frames(gpu, frame, &slots);
        let drape = self.slot_commands(&slots);
        let drape = self.slot_draws(gpu, drape);
        let encoding = clock();
        let mut drape_passes = 0;

        let mut passes = Vec::with_capacity(3);
        self.driver.moves(gpu, &mut self.scene, &mut passes);
        let main = self.main.as_ref().expect("recorded");
        let (split, waiting) = (main.splits(), main.awaits.waiting());
        let mut stats = FrameStats {
            draws: main.draws,
            bundle_recorded,
            candidates,
            candidates_rebuilt,
            placements_written: self.scene.take_placements_written(),
            paints_written: std::mem::take(&mut self.paints_written),
            pipelines_pending: compiling.pending,
            nodes_waiting: waiting,
            slot_bytes_moved: self.scene.moved_bytes(),
            ..FrameStats::default()
        };
        let cull_ready = self.cull_passes(gpu, &mut passes, &mut stats);
        let mut terrain_waited = false;
        if self.terrain_on() {
            let passing = clock();
            let drawn = self.terrain_passes(gpu, &drape, &mut passes);
            stats.draws += drawn.draws;
            stats.drape_slots = drawn.slots;
            terrain_waited = drawn.waited;
            drape_passes = micros(passing, clock());
        }
        // Over the terrain's depth, where it draws this frame.
        let terrain_depth = match self.terrain.as_ref().filter(|t| t.drawing()) {
            Some(t) => t.depth_ready(gpu, (&self.pipeline_cache, self.three_d_target())),
            None => true,
        };
        let extrusions_ready = cull_ready && terrain_depth;
        let placings = (self.placings(), self.full);
        let quads: Vec<(u32, std::ops::Range<u32>)> = (self.scene.blocks())
            .map(|block| (block, self.scene.block_quads(block)))
            .collect();
        passes.extend(self.symbols.pass(gpu, placings, &quads));
        // Before the map's passes, or among them where they split at the
        // 3D layers.
        let (mut three_d, draws) = self.extrusion_passes(gpu, extrusions_ready, split);
        stats.draws += draws;
        if !split {
            let start = passes.len();
            passes.extend(three_d.drain(..).map(|(_, pass)| Pass::Render(pass)));
            Pass::group(&mut passes, start, "3d layers");
        }
        let depth = self.map_depth.filter(|_| self.target.depth.is_some());
        if self.targets.any() && cull_ready {
            let start = passes.len();
            stats.draws += self.targets.passes(gpu, frame, &mut passes);
            Pass::group(&mut passes, start, "render targets");
        }
        let start = passes.len();
        let main = self.main.as_ref().expect("recorded");
        passes.extend(
            self.layers
                .passes(&main.layers, depth.map(|(depth, _)| depth)),
        );
        Pass::group(&mut passes, start, "layers apart");
        // Upside down, the map's passes draw into the canvas, which the
        // last pass turns over into the target.
        let canvas = (self.flip.as_mut())
            .map(|f| f.canvas(gpu, &mut self.composites, self.host.color, size));
        let map = (canvas.unwrap_or(target), clear, size);
        stats.draws += self.main_passes(gpu, map, three_d, &mut passes);
        let mut flip_waited = false;
        if let Some(flip) = &self.flip {
            let (pass, drew) = flip.pass(gpu, target);
            passes.push(pass);
            stats.draws += u32::from(drew);
            flip_waited = !drew;
        }
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
        stats.waited = compiling.pending > 0
            || !self.targets.ready(gpu)
            || end.pending > 0
            || end.made != made
            || self.main.as_ref().is_some_and(|m| m.awaits.lacking())
            || terrain_waited
            || flip_waited;
        // What a recording awaits arrives by a compile (a pipeline let go
        // counts as arrived): one that never compiles would leave it as
        // it is.
        debug_assert!(
            end.pending > 0
                || (self.main.iter().map(|m| &m.awaits))
                    .chain([&self.drape_awaits])
                    .all(|a| !a.stuck(gpu)),
            "a recording awaits a pipeline nothing compiles"
        );
        stats
    }

    /// Follows the cull's candidates as laid out again: a node the bundles
    /// left out for want of candidates draws once it has some, and a
    /// heatmap hidden, or with nothing to draw, composites nothing.
    fn follow_candidates(&mut self, gpu: &mut dyn Gpu, candidates_rebuilt: bool) {
        if candidates_rebuilt {
            let gained = |rows: &[u32]| rows.iter().any(|&r| self.scene.node_candidates(r) > 0);
            let main = (self.main.as_ref()).is_some_and(|m| gained(m.awaits.rows()));
            let drape = (self.drape.iter().flat_map(|d| d.recorded.iter().flatten()))
                .any(|r| gained(&r.left_out));
            if main {
                self.forget_main(gpu);
            }
            if drape && let Some(drape) = &mut self.drape {
                drape.forget_recorded(gpu);
            }
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
            self.forget_recordings(gpu);
        }
        if std::mem::take(&mut self.pipelines_stale) {
            self.show_nodes(gpu);
        }
        let compiling = gpu.poll_pipelines();
        if compiling.made != self.made_seen {
            // Pipelines arrived: record again what waited for them.
            self.made_seen = compiling.made;
            if self.main.as_ref().is_some_and(|m| m.awaits.arrived(gpu)) {
                self.forget_main(gpu);
            }
            if self.drape_awaits.arrived(gpu) {
                self.stale_drape(gpu);
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
    fn resize_targets(&mut self, gpu: &mut dyn Gpu, frame: &FrameBlock) -> [u32; 2] {
        let terrain_on = self.terrain_on();
        let density_size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio / 4.0).floor().max(1.0) as u32);
        let size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio).max(1.0) as u32);
        let logical = frame.viewport.map(|v| v.max(1.0) as u32);
        // Every 3D and heatmap node has its targets from the first frame,
        // whether or not its layer draws yet, so that none is made for a
        // first use in a gesture: a frame makes them only for a new
        // viewport, which the screen made anew too. The map's passes share
        // the 3D layers' depth where they split at them, which a pass of
        // its own then doesn't keep to itself.
        let extruded = self.nodes.iter().filter(|n| n.extrusion.is_some()).count();
        let shared = usize::from(extruded > 0 && terrain_on && self.splits_at_3d());
        let stencil = self.target.depth.is_some();
        let mut resized = (self.extrusions).resize(gpu, size, (extruded + shared, stencil));
        if let Some(format) = self.target.depth
            && self.map_depth.is_none_or(|(_, made)| made != size)
        {
            self.map_depth.take().map(|(depth, _)| depth).release(gpu);
            let depth = gpu.create_texture(&TextureDesc {
                label: "map depth",
                width: size[0],
                height: size[1],
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format,
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
        resized |= (self.targets).fit(gpu, &mut self.composites, logical, self.target);
        // The map's passes split at its 3D layers, which share their depth.
        let made = (
            size,
            self.target.format(),
            terrain_on && self.splits_at_3d(),
        );
        if resized {
            self.forget_main(gpu);
        }
        // What the groups draw changes only with what forgets the main
        // recording.
        if self.main.is_none() || !self.layers.made_for(made) {
            let runs = self.layer_runs();
            if (self.layers).fit(gpu, &mut self.composites, runs, made) {
                self.forget_main(gpu);
            }
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

    /// The runs of nodes that composite once ([`layer_composite`]): runs
    /// of one layer's nodes below layer opacity 1 or blending other than
    /// normally draw into a texture of their own. On terrain a draped
    /// layer's nodes draw in its terrain layer, at the layer's opacity:
    /// one that reads the map beneath it (blending other than normally, or
    /// a hillshade's blending into the map) is a group that splits the
    /// pass, with nothing of its own to composite.
    fn layer_runs(&self) -> Vec<layer_composite::Run> {
        let terrain_on = self.terrain_on();
        #[derive(Clone, Copy, PartialEq)]
        enum Key {
            Layer(u32),
            Terrain(u8),
            Mask(usize),
        }
        let blank = self.composites.blank();
        let mut runs: Vec<(Key, layer_composite::Run)> = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            // A node drawn into a render target draws in its pass; a fill
            // drawn through one composites through its coverage, at its
            // layer opacity.
            if self.targets.off_map(i) || self.solid_through(i).is_some() {
                continue;
            }
            let mask = (self.targets.mask_of(i)).and_then(|t| self.targets.texture(t));
            // One in a colour alone blends that colour through the target,
            // drawing nothing of its own: its draws lie on the map's plane,
            // short of the sky the target may cover.
            let color = self.targets.solid(i).map(offscreen::Composite::color);
            let (key, opacity, blend, source) = match node.terrain_layer.filter(|_| terrain_on) {
                _ if mask.is_some() => {
                    let (opacity, blend) = (node.composite)
                        .map_or((1.0, Blend::Normal), |(_, opacity, blend)| (opacity, blend));
                    (Key::Mask(i), opacity, blend, color.map(|_| blank))
                }
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
                        mask,
                        color,
                    },
                )),
            }
        }
        runs.into_iter().map(|(_, run)| run).collect()
    }

    /// Records the map's bundles and the composited layers' own, with
    /// `beneath` the solid background the main pass can clear to: what
    /// the map's passes draw ([`Step`]).
    fn record_main(&mut self, gpu: &mut dyn Gpu, beneath: Option<usize>) -> MainRecording {
        let terrain_on = self.terrain_on();
        let split = terrain_on && self.splits_at_3d();
        let mut ready = self.ready(gpu);
        let mut commands = Vec::with_capacity(self.nodes.len() * 5);
        // Until the composites compile, nothing draws: a split main
        // pass needs its blend draw.
        let at = (&self.pipeline_cache, self.target);
        let layers = Lacks::of(gpu, self.layers.needs(&self.composites, at));
        let composited = layers.is_empty();
        ready.cull.extend(&layers);
        let mut main = MainRecording {
            beneath,
            segments: self.scene.segments_revision()[0],
            ..MainRecording::default()
        };
        let layers = RenderBundleDesc {
            label: "layers",
            color_formats: &[self.target.color],
            depth_format: self.target.depth,
            sample_count: self.target.sample_count,
        };
        for g in &self.layers.groups {
            if !composited || g.drawn_elsewhere() {
                main.layers.push(None);
                continue;
            }
            let mut group = Vec::new();
            for i in g.nodes() {
                let node = &self.nodes[i];
                let groups = (&self.draw_groups[..], (node.group, node.second_group), true);
                let before = group.len();
                self.node_commands(gpu, &ready, i, groups, (&mut group, None), &mut main.awaits);
                main.draws += u32::from(group.len() > before);
            }
            let desc = RenderBundleDesc {
                label: "layer composite",
                ..layers
            };
            main.layers
                .push(Some(gpu.create_render_bundle(&desc, &bound_once(group))));
        }
        // Each drawn render target's writers, in style order: a float one's
        // only processes, through the composite made for it.
        let over = self.composites.needs(at.0, at.1, offscreen::Mode::Over);
        let over = Lacks::of(gpu, [over]);
        let mut float_over = Lacks::default();
        let float_over = (float_over.need(gpu, self.targets.float_over()), float_over);
        for t in self.targets.drawn().collect::<Vec<_>>() {
            let float = self.targets.holds_float(t);
            let mut draws = Vec::new();
            for writer in self.targets.writers(t) {
                match writer {
                    targets::Writer::Node(i) => {
                        let node = &self.nodes[i];
                        let groups = (&self.draw_groups[..], (node.group, node.second_group), true);
                        let before = draws.len();
                        let awaits = &mut main.awaits;
                        self.node_commands(gpu, &ready, i, groups, (&mut draws, None), awaits);
                        main.draws += u32::from(draws.len() > before);
                    }
                    targets::Writer::Process(p) => match self.targets.composite(p) {
                        Some(composite) if float => match &float_over {
                            (Some(pipeline), _) => draws.extend(
                                offscreen::Composites::commands_through(composite, *pipeline),
                            ),
                            (None, lacks) => main.awaits.wait(lacks.clone(), false),
                        },
                        Some(composite) if over.is_empty() => {
                            let composite = self.composites.commands(at, composite);
                            draws.extend(composite.into_iter().flatten());
                        }
                        Some(_) => main.awaits.wait(over.clone(), false),
                        None => main.awaits.wait(Lacks::unmade(), false),
                    },
                }
            }
            let desc = match float {
                true => RenderBundleDesc {
                    label: "render target",
                    color_formats: &[TextureFormat::Rgba16Float],
                    depth_format: None,
                    sample_count: 1,
                },
                false => RenderBundleDesc {
                    label: "render target",
                    ..layers
                },
            };
            let bundle = gpu.create_render_bundle(&desc, &bound_once(draws));
            self.targets.set_bundle(gpu, t, bundle);
        }
        // Image-process layers drawing into the map and fills drawn
        // through targets in a colour, in a row with nothing else between
        // them in style order (layers drawn into targets aside), draw as
        // stacks: a full-screen draw each would cost the map's pass its
        // every pixel again.
        let mut stacking = over.clone();
        stacking.all(gpu, self.composites.stack_needs(at.0, at.1));
        let stacked = stacking.is_empty();
        let mut row = Vec::new();
        for i in 0..=self.nodes.len() {
            // The solid background and what it hides go in a bundle
            // of their own, which the clear can stand for.
            if beneath.is_some_and(|b| i == b + 1) {
                self.stack_row(gpu, &mut row, (&mut commands, &mut main));
                let below = main.bundle(gpu, &layers, &mut commands);
                main.steps.push(Step::Beneath(below));
            }
            // Image-process layers drawing into the map, in their place.
            for p in self.targets.drawing_on_map_at(i) {
                match self
                    .targets
                    .composite(p)
                    .and_then(offscreen::Composite::layer)
                {
                    Some(layer) if stacked => row.push((layer, Composited::Process(p))),
                    Some(_) => main.awaits.wait(stacking.clone(), false),
                    None => main.awaits.wait(Lacks::unmade(), false),
                }
            }
            if i == self.nodes.len() {
                break;
            }
            if self.targets.off_map(i) {
                continue;
            }
            if let Some(solid) = self.solid_through(i) {
                match solid.layer() {
                    Some(layer) if stacked => row.push((layer, Composited::Fill(i))),
                    Some(_) => main.awaits.wait(stacking.clone(), false),
                    None => main.awaits.wait(Lacks::unmade(), false),
                }
                continue;
            }
            // Whatever else the node draws ends the row.
            self.stack_row(gpu, &mut row, (&mut commands, &mut main));
            let (start, member) = self.layers.at(i);
            match start
                .filter(|_| composited)
                .map(|g| (g, &self.layers.groups[g]))
            {
                // The pass splits here: its blend draw opens the next
                // pass, before what follows.
                Some((g, group)) if group.blend() != Blend::Normal => {
                    main.flush(gpu, &layers, &mut commands);
                    main.steps.push(Step::Blend(g));
                    main.draws += 1;
                }
                Some((_, group)) => {
                    let at = (&self.pipeline_cache, self.target);
                    commands.extend(group.composite_commands(&self.composites, at));
                    main.draws += 1;
                }
                None => {}
            }
            let node = &self.nodes[i];
            if let Some(layer) = node.terrain_layer.filter(|_| terrain_on) {
                // A terrain layer draws in its first node's place.
                if i == 0 || self.nodes[i - 1].terrain_layer != Some(layer) {
                    if ready.cull.is_empty() && ready.terrain.is_empty() {
                        main.flush(gpu, &layers, &mut commands);
                        main.steps.push(Step::Terrain(layer));
                    } else {
                        let mut lacks = ready.cull.clone();
                        lacks.extend(&ready.terrain);
                        main.awaits.wait(lacks, false);
                    }
                }
                continue;
            }
            if split && node.extrusion.is_some() {
                main.flush(gpu, &layers, &mut commands);
                main.steps.push(Step::ThreeD(i));
            }
            let groups = (&self.draw_groups[..], (node.group, node.second_group), true);
            let before = commands.len();
            if !member {
                let draws = (&mut commands, None);
                self.node_commands(gpu, &ready, i, groups, draws, &mut main.awaits);
            }
            main.draws += u32::from(commands.len() > before);
        }
        self.stack_row(gpu, &mut row, (&mut commands, &mut main));
        main.flush(gpu, &layers, &mut commands);
        main
    }

    /// Draws `row`, composites in a row as layers and what each
    /// composites, as stacks into `commands` (one alone as itself), and
    /// empties it.
    fn stack_row(
        &self,
        gpu: &mut dyn Gpu,
        row: &mut Vec<(offscreen::Layer, Composited)>,
        (commands, main): (&mut Vec<RenderCommand>, &mut MainRecording),
    ) {
        for chunk in row.chunks(offscreen::STACK) {
            main.draws += 1;
            if let [(_, alone)] = *chunk {
                let composite = match alone {
                    Composited::Process(p) => self.targets.composite(p),
                    Composited::Fill(i) => self.targets.solid(i),
                };
                let at = (&self.pipeline_cache, self.target);
                let draw = self.composites.commands(at, composite.expect("in the row"));
                commands.extend(draw.expect("stacked once its pipelines compiled"));
                continue;
            }
            let layers: Vec<offscreen::Layer> = chunk.iter().map(|&(layer, _)| layer).collect();
            let stack = self.composites.stack(gpu, &layers);
            let at = (&self.pipeline_cache, self.target);
            commands.extend(self.composites.stack_commands(at, &stack));
            let fills = chunk.iter().map(|&(_, c)| match c {
                Composited::Fill(i) => Some(i),
                Composited::Process(_) => None,
            });
            main.stacks.push((stack, fills.collect()));
        }
        row.clear();
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
            let draws = || {
                (self.scene.blocks())
                    .filter_map(|block| {
                        let row = self.scene.row(index as u32, block);
                        let group = self.draw_groups.get(block as usize).copied().flatten()?;
                        let draw = self
                            .driver
                            .draw(self.scene.buffers(), row, NodeDraw::Points);
                        Some((group, draw))
                    })
                    .collect()
            };
            let needs = (&self.composites, &self.pipeline_cache, self.target);
            if let Some(heatmap) = &mut node.heatmap
                && node.drawn
                && cull_ready
                && awaits::compiled(gpu, self.heatmaps.needs(needs))
                && let Some(pass) =
                    (self.heatmaps).prepare(gpu, heatmap, node.group, (draws, self.draws_revision))
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
        drape: &FxHashMap<(u32, u32), Vec<RenderCommand>>,
        passes: &mut Vec<Pass>,
    ) -> terrain::TerrainDrawn {
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let start = passes.len();
        let b = self.scene.buffers();
        let scene = (b.frame, self.scene.tables().entry(Table::Props, 8));
        let at = (&self.pipeline_cache, self.target);
        let drawn = terrain.passes(gpu, at, scene, drape, &self.downsampler, passes);
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
        let cache = &self.pipeline_cache;
        let terrain = (self.terrain.as_ref()).filter(|_| self.terrain_on());
        let mut terrain_depth = terrain.map(|t| t.depth_commands((cache, self.three_d_target())));
        let mut first = !split;
        let stencil = self.target.depth.is_some();
        let draws = [self.draws_revision, self.scene.segments_revision()[1]];
        for index in 0..self.nodes.len() {
            let node = &self.nodes[index];
            let Some(extrusion) = node.extrusion.as_ref() else {
                continue;
            };
            if !(extrusions_ready && node.drawn) {
                continue;
            }
            // What it draws with; nothing while that compiles, which the
            // main bundles wait for.
            let shape = (node.shape.elevation, stencil);
            let lacks = &mut Lacks::default();
            let Some(pipelines) = self.extrusions.drawn((gpu, cache), shape, self.full, lacks)
            else {
                continue;
            };
            let recorded = self.extrusions.recorded(pipelines, draws);
            // Its draws, where its bundle records them again.
            let streams = (!extrusion.records(recorded)).then(|| {
                let cut =
                    (self.scene.second_row(index as u32).map(Second::row)).zip(node.second_group);
                let (cut_row, cut_group) = cut.expect("an extrusion node draws a cut stream");
                let draws = |row| {
                    let mut commands = Vec::new();
                    self.whole_draws(&self.draw_groups, row, &mut commands);
                    commands
                };
                extrusion::Streams {
                    whole: (node.group, draws(index as u32)),
                    cut: (cut_group, draws(cut_row)),
                }
            });
            let before = if first {
                terrain_depth.take().unwrap_or_default()
            } else {
                Vec::new()
            };
            let target = (self.nodes[index].extrusion.as_mut()).expect("an extrusion node");
            let pass = (self.extrusions).prepare(
                gpu,
                target,
                &buffers,
                (streams, recorded),
                (first, before),
            );
            first = false;
            out.push((index, pass));
        }
        let draws = out.len() as u32;
        (out, draws)
    }

    /// The 3D layers' passes' attachments: their targets' over the map's
    /// depth.
    fn three_d_target(&self) -> Attachments {
        terrain::three_d_target(self.target, self.extrusions.format())
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
        (target, mut clear, size): (TextureId, [f64; 4], [u32; 2]),
        mut three_d: Vec<(usize, RenderPass)>,
        passes: &mut Vec<Pass>,
    ) -> u32 {
        let start = passes.len();
        let terrain_on = self.terrain_on();
        let main = self.main.as_mut().expect("recorded");
        let (steps, beneath) = (std::mem::take(&mut main.steps), main.beneath);
        let split = steps.iter().any(|s| matches!(s, Step::ThreeD(_)));
        let samples = self.map_samples(gpu, size, split);
        let depth = match split {
            true => self.extrusions.depth().map(|(depth, _)| depth),
            false => self.map_depth.map(|(depth, _)| depth),
        };
        let depth = depth.filter(|_| self.target.depth.is_some());
        let at = (&self.pipeline_cache, self.target);
        let terrain_depth = (self.terrain.as_ref())
            .filter(|_| terrain_on)
            .map_or(Vec::new(), |t| t.depth_commands(at));
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
        if let (Some(&Step::Beneath(below)), Some(b)) = (steps.first(), beneath) {
            rest = &steps[1..];
            let shown = self.nodes[b].cull.visible && self.scene.node_candidates(b as u32) > 0;
            match self.nodes[b].solid {
                Some(color) if shown && self.background_fills && !terrain_on => {
                    clear = color.map(f64::from);
                    sky = false;
                }
                _ => {
                    commands.extend(self.atmosphere.commands(gpu, at));
                    commands.extend(below.map(|b| RenderCommand::ExecuteBundles(vec![b])));
                    sky = false;
                }
            }
        }
        if sky {
            commands.extend(self.atmosphere.commands(gpu, at));
        }
        // The canvas of the passes, by the groups that split them so far,
        // and the passes into it so far.
        let (mut canvas, mut into) = (0usize, 0);
        let mut first = true;
        // A pass the next goes on from, after a 3D layer's (`keep`), stores
        // its samples, and only the canvas's last resolves them.
        let mut pass = |this: &Self, commands, (canvas, into), (last, keep): (bool, bool)| {
            let load = match into {
                0 => LoadOp::Clear(if canvas == 0 { clear } else { [0.0; 4] }),
                _ => LoadOp::Load,
            };
            let color = match samples {
                _ if canvas < blends => this.layers.canvas(canvas).attachment_kept(load, keep),
                Some(samples) => ColorAttachment {
                    target: samples,
                    resolve: (!keep).then_some(target),
                    store: if keep {
                        StoreOp::Store
                    } else {
                        StoreOp::Discard
                    },
                    load,
                },
                None => ColorAttachment {
                    store: StoreOp::Store,
                    target,
                    resolve: None,
                    load,
                },
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
                Step::Beneath(bundle) => {
                    commands.extend(bundle.map(|b| RenderCommand::ExecuteBundles(vec![b])));
                }
                Step::Bundle(bundle) => {
                    commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
                }
                Step::Terrain(layer) => {
                    // A layer that reads the map beneath reads the canvas
                    // the pass before drew.
                    let beneath = canvas
                        .checked_sub(1)
                        .map(|k| self.layers.canvas(k).texture());
                    let tables = shades.get(usize::from(layer)).copied().unwrap_or_default();
                    let terrain = self.terrain.as_mut().expect("terrain is on");
                    let at = (&self.pipeline_cache, self.target);
                    let drawn = terrain.layer_commands(gpu, at, layer, tables, beneath);
                    draws += u32::from(!drawn.is_empty());
                    commands.extend(drawn);
                }
                Step::Blend(group) => {
                    let done = std::mem::take(&mut commands);
                    passes.push(pass(self, done, (canvas, into), (false, false)));
                    (canvas, into) = (canvas + 1, 0);
                    // The blend draw covers every pixel.
                    if !split {
                        predraw(&mut commands);
                    }
                    let group = &self.layers.groups[group];
                    let at = (&self.pipeline_cache, self.target);
                    commands.extend(group.composite_commands(&self.composites, at));
                }
                Step::ThreeD(node) => {
                    let done = std::mem::take(&mut commands);
                    passes.push(pass(self, done, (canvas, into), (false, true)));
                    into += 1;
                    if let Some(at) = three_d.iter().position(|(n, _)| *n == node) {
                        passes.push(Pass::Render(three_d.swap_remove(at).1));
                    }
                }
            }
        }
        let at = (&self.pipeline_cache, self.target);
        commands.extend(self.atmosphere.commands_after(gpu, at));
        passes.push(pass(self, commands, (canvas, into), (true, false)));
        Pass::group(passes, start, "main");
        self.main.as_mut().expect("recorded").steps = steps;
        draws + predraws
    }

    /// The multisampled colour the map's last canvas draws into at
    /// `size`, where the map's passes take more samples than the host's
    /// target, its samples kept between passes where they `split` at 3D
    /// layers: made unless it is.
    fn map_samples(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], split: bool) -> Option<TextureId> {
        if self.target.sample_count == self.host.sample_count {
            return None;
        }
        if let Some((texture, made, kept)) = self.map_samples
            && (made, kept) == (size, split)
        {
            return Some(texture);
        }
        self.map_samples
            .take()
            .map(|(texture, ..)| texture)
            .release(gpu);
        let texture = gpu.create_texture(&TextureDesc {
            label: "map samples",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: self.target.color,
            usage: match split {
                true => TextureUsages::RENDER_ATTACHMENT,
                false => TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
            },
            sample_count: self.target.sample_count,
            mip_levels: 1,
        });
        self.map_samples = Some((texture, size, split));
        Some(texture)
    }

    /// What a bundle record finds lacking besides the nodes' own
    /// pipelines. Those only a family needs are asked after once a node of
    /// it drew: made ahead of need ([`set_pipelines`](Self::set_pipelines)), they
    /// compile in the background until then.
    fn ready(&mut self, gpu: &mut dyn Gpu) -> Ready {
        let drawn = |families: &[Family]| {
            (self.nodes.iter()).any(|n| n.drawn && families.contains(&n.family))
        };
        let (rasters, dems) = (
            drawn(&[Family::Raster]),
            drawn(&[Family::Hillshade, Family::ColorRelief]),
        );
        let terrain = match self.terrain.as_ref().filter(|_| self.terrain_on()) {
            Some(t) => Lacks::of(gpu, t.needs((&self.pipeline_cache, self.target))),
            None => Lacks::unmade(),
        };
        let prepare = match dems {
            true => self.images.prepare_pipelines(),
            false => Vec::new(),
        };
        let mips: Vec<PipelineId> = match rasters {
            true => self.downsampler.pipelines().collect(),
            false => Vec::new(),
        };
        let tables = self.images.tables_pipeline().map(PipelineId::from);
        Ready {
            cull: Lacks::of(gpu, self.driver.needs().map(Some)),
            terrain,
            prepare: Lacks::of(gpu, prepare.into_iter().map(Some)),
            tables: Lacks::of(gpu, tables.map(Some)),
            mips: Lacks::of(gpu, mips.into_iter().map(Some)),
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
                .filter(|d| node.stack.is_some() && !node.shades_on_terrain() && d.draws(i));
            let uniforms = draped.map_or(&self.uniforms, |d| &d.uniforms);
            let (words, uniform) = (
                uniforms.row(i as u32).to_vec(),
                uniforms.binding(i as u32, false),
            );
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

    /// Readies the drape passes into stack `stack` for this frame: its
    /// draws of the draped nodes recorded if they aren't ([`Recorded`]),
    /// once any node's drape paint was set.
    fn record_drape(&mut self, gpu: &mut dyn Gpu, stack: u32) {
        if self.terrain.is_none() {
            return;
        }
        let Some(level) = self.drape.as_mut() else {
            return;
        };
        let at = stack as usize;
        if level.recorded.len() <= at {
            level.recorded.resize_with(at + 1, || None);
        }
        if level.recorded[at].is_some() {
            return;
        }
        let ready = self.ready(gpu);
        let level = self.drape.as_ref().expect("found above");
        // Recorded over the map's scene groups, which each slot's pass
        // binds its own in place of ([`Renderer::slot_commands`]).
        let scenes = self.draw_groups.clone();
        let mut commands = Vec::new();
        let mut splices = Vec::new();
        let mut groups = Vec::new();
        let mut awaits = Awaits::default();
        for i in 0..self.nodes.len() {
            let in_stack = self.nodes[i].stack.is_some_and(|s| u32::from(s) == stack);
            if !in_stack || !level.draws(i) {
                continue;
            }
            let group = self.node_group(gpu, i, level.uniforms.binding(i as u32, false));
            groups.push(group);
            // Draped families split their placements, never cut them.
            let second = match self.scene.second_row(i as u32) {
                Some(Second::Tested(row)) => {
                    Some(self.node_group(gpu, i, level.uniforms.binding(row, false)))
                }
                _ => None,
            };
            groups.extend(second);
            let node_groups = (&scenes[..], (group, second), false);
            let draws = (&mut commands, Some(&mut splices));
            self.node_commands(gpu, &ready, i, node_groups, draws, &mut awaits);
        }
        let left_out = awaits.take_rows();
        // A slot keeps what it drew: one drawn while a node waited, or with
        // a stand-in (which draws otherwise, by rounding), is redrawn once
        // what it lacked arrives, so it draws alike whenever that compiled.
        self.drape_awaits.absorb(awaits);
        let level = self.drape.as_mut().expect("found above");
        level.recorded[at] = Some(Recorded {
            commands,
            splices,
            groups,
            left_out,
            slots: FxHashMap::default(),
        });
    }

    /// Readies the frames of each redrawn slot of `slots` (`(stack, tile
    /// slot)`s), one a block, and their scene groups: the drape frame with
    /// its camera at the slot's corner, so that what a slot draws of its
    /// own lies alike in clip space whatever its place in the grid, and
    /// what other slots hold whole slots off its layer. Those of slots the
    /// drape no longer holds go.
    fn slot_frames(&mut self, gpu: &mut dyn Gpu, frame: &FrameBlock, slots: &[(u32, u32)]) {
        let (Some(terrain), Some(level)) = (self.terrain.as_ref(), self.drape.as_mut()) else {
            return;
        };
        let layout = terrain.layout();
        let colour = self.drape_stacks[0];
        let mut done = rustc_hash::FxHashSet::default();
        for &(stack, slot) in slots {
            let page = usize::from(stack >= colour);
            if !done.insert((page, slot)) {
                continue;
            }
            let corner = layout.corner(slot, stack);
            let words = crate::scene::frame_words(&terrain.drape_frame(frame, page, corner));
            let (frames, _) = level.slot_scenes[page].entry(slot).or_insert_with(|| {
                let held: Vec<u32> = self.scene.blocks().collect();
                let frames: Vec<Option<BufferId>> = (0..self.scene.block_slots())
                    .map(|block| {
                        held.contains(&block).then(|| {
                            gpu.create_buffer(&BufferDesc {
                                label: "drape slot frame",
                                size: crate::scene::FRAME_BYTES,
                                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                            })
                        })
                    })
                    .collect();
                let groups = scene_groups(
                    gpu,
                    self.draw_layout,
                    &self.scene,
                    &self.image_bindings,
                    (
                        self.images.nearest(),
                        self.scene_bound.dem,
                        self.no_terrain_depth,
                    ),
                    Some(&frames),
                );
                (frames, groups)
            });
            for (block, buffer) in frames.iter().enumerate() {
                if let Some(buffer) = buffer {
                    let mut words = words.clone();
                    self.scene.block_words(block as u32, &mut words);
                    gpu.write_buffer(*buffer, 0, bytemuck::cast_slice(&words));
                }
            }
        }
        for page in &mut level.slot_scenes {
            page.retain(|&slot, scenes| {
                let held = terrain.holds(slot);
                if !held {
                    std::mem::take(scenes).release(gpu);
                }
                held
            });
        }
    }

    /// The drape passes' commands this frame, by stack and tile slot: each
    /// stack's [recorded](Recorded) draws, the draws of a node's runs
    /// those of the slot's runs alone. `slots` are the frame's redrawn
    /// `(stack, tile slot)`s.
    fn slot_commands(&self, slots: &[(u32, u32)]) -> FxHashMap<(u32, u32), Vec<RenderCommand>> {
        let Some(level) = &self.drape else {
            return FxHashMap::default();
        };
        let colour = self.drape_stacks[0];
        let mut by_slot = FxHashMap::default();
        for in_stack in slots.chunk_by(|a, b| a.0 == b.0) {
            let stack = in_stack[0].0;
            let Some(Some(recorded)) = level.recorded.get(stack as usize) else {
                continue;
            };
            let scenes = &self.draw_groups;
            // Each spliced stream's runs by slot, worked out once a frame.
            let streams: Vec<_> = (recorded.splices.iter())
                .map(|&(_, id)| self.scene.slot_segments(id))
                .collect();
            for &(_, slot) in in_stack {
                let mut commands = Vec::with_capacity(recorded.commands.len());
                let mut from = 0;
                for (&(at, id), runs) in recorded.splices.iter().zip(&streams) {
                    commands.extend_from_slice(&recorded.commands[from..at]);
                    from = at;
                    match runs {
                        Some(runs) => {
                            let segments = runs.get(&slot).map_or(&[][..], Vec::as_slice);
                            self.segment_draws(scenes, id, segments, &mut commands);
                        }
                        // A sorted stream's slots don't say whose runs
                        // they hold: it draws whole.
                        None => self.whole_draws(scenes, id, &mut commands),
                    }
                }
                commands.extend_from_slice(&recorded.commands[from..]);
                // The slot's own frames in place of the page's.
                if let Some((_, own)) = level.slot_scenes[usize::from(stack >= colour)].get(&slot) {
                    for command in &mut commands {
                        if let RenderCommand::SetBindGroup { index: 0, group } = command
                            && let Some(block) = scenes.iter().position(|s| *s == Some(*group))
                            && let Some(Some(mine)) = own.get(block)
                        {
                            *group = *mine;
                        }
                    }
                }
                by_slot.insert((stack, slot), bound_once(commands));
            }
        }
        by_slot
    }

    /// Each redrawn slot's pass's draws (`by_slot`, from
    /// [`slot_commands`](Self::slot_commands)): the commands themselves
    /// the first time the slot draws them, then a bundle of them its
    /// stack's [recorded](Recorded) draws keep for the slot while it draws
    /// the same. A slot redraws whole as its tile's footprint or the
    /// drape's zoom step changes, its content unchanged: a pass replaying
    /// a bundle costs the CPU two calls, where issuing its draws cost one
    /// a command. New content draws direct, as recording a bundle costs
    /// what issuing the commands does and more (on a moto g, a frame's 40
    /// slots took 54 ms recorded against 22 ms for 55 issued).
    fn slot_draws(
        &mut self,
        gpu: &mut dyn Gpu,
        by_slot: FxHashMap<(u32, u32), Vec<RenderCommand>>,
    ) -> FxHashMap<(u32, u32), Vec<RenderCommand>> {
        let Some(level) = &mut self.drape else {
            return FxHashMap::default();
        };
        let desc = RenderBundleDesc {
            label: "drape slot",
            color_formats: &[DRAPE_FORMAT],
            depth_format: None,
            sample_count: 1,
        };
        let mut draws = FxHashMap::default();
        for ((stack, slot), commands) in by_slot {
            let Some(Some(recorded)) = level.recorded.get_mut(stack as usize) else {
                continue;
            };
            let kept = (recorded.slots.get_mut(&slot)).filter(|k| k.0 == commands);
            let drawn = match kept {
                // Set once, state no draw follows is gone: the slot draws
                // nothing, and what it drew before goes.
                _ if commands.is_empty() => {
                    recorded
                        .slots
                        .remove(&slot)
                        .and_then(|(_, b)| b)
                        .release(gpu);
                    Vec::new()
                }
                Some((commands, bundle)) => {
                    let bundle =
                        *bundle.get_or_insert_with(|| gpu.create_render_bundle(&desc, commands));
                    vec![RenderCommand::ExecuteBundles(vec![bundle])]
                }
                None => {
                    let old = recorded.slots.insert(slot, (commands.clone(), None));
                    old.and_then(|(_, b)| b).release(gpu);
                    commands
                }
            };
            draws.insert((stack, slot), drawn);
        }
        draws
    }

    /// Indexed stream `id`'s draws of everything it holds: in segments, or
    /// a row's arguments a block.
    fn whole_draws(
        &self,
        scenes: &[Option<BindGroupId>],
        id: u32,
        commands: &mut Vec<RenderCommand>,
    ) {
        if let Some(segments) = self.scene.segments(id) {
            return self.segment_draws(scenes, id, segments, commands);
        }
        let buffers = self.scene.buffers();
        for (block, scene) in scenes.iter().enumerate() {
            let row = self.scene.row(id, block as u32);
            if let Some(scene) = scene
                && !self.driver.draws_nothing(row)
            {
                commands.extend([
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: *scene,
                    },
                    self.driver.draw(buffers, row, NodeDraw::Indexed),
                ]);
            }
        }
    }

    /// Node `i`'s commands as a bundle records them, over the scene groups
    /// `scenes`, one per block slot: the map's (`in_map`), or the drape's
    /// for draped nodes on terrain, with its group and its second
    /// stream's. The node draws once in each block it may have candidates
    /// in, that block's group bound, or in its segments, in its tiles'
    /// order ([`Scene::segments`]); a split node's untested placements
    /// first, in their pipeline, then its tested ones. In the map, a
    /// terrain layer draws its nodes ([`Step::Terrain`]). `awaits` gains
    /// the rows it left out and what it lacked: while a pipeline it needs
    /// compiles it records nothing, or draws with a stand-in for the one
    /// it wants ([`Variants::drawn`](pipeline::Variants::drawn)), until
    /// that one compiles and the recording records again.
    fn node_commands(
        &self,
        gpu: &mut dyn Gpu,
        ready: &Ready,
        i: usize,
        groups: (
            &[Option<BindGroupId>],
            (BindGroupId, Option<BindGroupId>),
            bool,
        ),
        (commands, splices): Recording<'_>,
        awaits: &mut Awaits,
    ) {
        let before = commands.len();
        let lacks = self.record_node(gpu, ready, i, groups, (commands, splices), awaits);
        awaits.wait(lacks, commands.len() > before);
    }

    /// [`node_commands`](Self::node_commands)' draws, and what they lack.
    fn record_node(
        &self,
        gpu: &mut dyn Gpu,
        ready: &Ready,
        i: usize,
        (scenes, (group, second_group), in_map): (
            &[Option<BindGroupId>],
            (BindGroupId, Option<BindGroupId>),
            bool,
        ),
        (commands, mut splices): Recording<'_>,
        awaits: &mut Awaits,
    ) -> Lacks {
        if !ready.cull.is_empty() {
            return ready.cull.clone();
        }
        let node = &self.nodes[i];
        if node.terrain_layer.is_some() && in_map {
            // Its terrain layer draws it ([`Step::Terrain`]).
            return ready.terrain.clone();
        }
        if !node.drawn {
            // Nothing to draw yet: drawing records the bundles again.
            return Lacks::default();
        }
        let draws = self.node_draws((i, &self.view(i)), &self.config());
        let (cache, target) = (&self.pipeline_cache, self.target);
        // Drawn apart, into a target of their own: here its composite, an
        // extrusion's with a stand-in too
        // ([`Variants::drawn`](pipeline::Variants::drawn)).
        let apart = match draws.with {
            DrawsWith::Extrusion(shape) => {
                let mut lacks = Lacks::default();
                let shape = (shape, target.depth.is_some());
                let drawn = (self.extrusions).drawn((gpu, cache), shape, self.full, &mut lacks);
                let over = self.composites.needs(cache, target, offscreen::Mode::Over);
                let composited = lacks.need(gpu, over).is_some();
                let offscreen = node.extrusion.as_ref().and_then(|a| a.offscreen.as_ref());
                Some((lacks, drawn.is_some() && composited, offscreen))
            }
            DrawsWith::Heatmap => {
                let lacks = Lacks::of(gpu, self.heatmaps.needs((&self.composites, cache, target)));
                let drawn = lacks.is_empty();
                let offscreen = node
                    .heatmap
                    .as_ref()
                    .and_then(|h| h.apart.offscreen.as_ref());
                Some((lacks, drawn, offscreen))
            }
            _ => None,
        };
        if let Some((lacks, drawn, offscreen)) = apart {
            if drawn && let Some(o) = offscreen {
                let composite = self.composites.commands((cache, target), &o.composite);
                commands.extend(composite.into_iter().flatten());
            }
            return lacks;
        }
        // Without candidates a row's draw draws nothing: on sparse views
        // most of a style's nodes, each a pipeline, bindings and an
        // indirect draw the CPU encodes and the GPU reads every frame. A
        // split node's second row holds its tested placements.
        let second = self.scene.second_row(i as u32).map(Second::row);
        let streams = [Some(i as u32), second]
            .map(|row| row.filter(|&row| self.scene.node_candidates(row) > 0));
        for (row, kept) in [Some(i as u32), second].into_iter().zip(streams) {
            if let (Some(row), None) = (row, kept) {
                awaits.leave_out(row);
            }
        }
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
            return Lacks::default();
        }
        let mut lacks = Lacks::default();
        let buffers = self.scene.buffers();
        if let DrawsWith::Symbols(kind, placings) = draws.with {
            let rows = streams[0].clone().unwrap_or_default();
            // Its icons and its text, in one list. The cull's point
            // arguments read as indexed ones too: 6 indices into the quad
            // indices, and the first index, base vertex and first instance
            // all 0, as points always have them.
            let apart = [node.separate_halo, self.nodes[i + 1].separate_halo];
            // Every block's draw, a stand-in's too: the node waits for the
            // variant it wants all the same.
            for (scene, row) in rows {
                let quads = (self.driver).draw(buffers, row, NodeDraw::Indexed);
                if !self.symbols.commands(
                    gpu,
                    (cache, target),
                    (kind, placings, self.full),
                    (scene, group),
                    apart,
                    quads,
                    commands,
                    &mut lacks,
                ) {
                    break;
                }
            }
            return lacks;
        }
        // What prepares and samples images' pages.
        for (reads, ready) in [
            (draws.prepare, &ready.prepare),
            (draws.tables, &ready.tables),
            (draws.mips, &ready.mips),
        ] {
            if reads {
                lacks.extend(ready);
            }
        }
        if !lacks.is_empty() {
            return lacks;
        }
        // A split node's untested placements draw in a pipeline that tests
        // no fragment once it compiled, else in the tested one, which
        // stands in for it: lacked until it compiles.
        let split = second.is_some() && streams[0].is_some();
        let full = self.full;
        let (untested, pipeline) = match draws.with {
            DrawsWith::Slopes => (None, lacks.need(gpu, self.images.slopes_pipeline())),
            DrawsWith::Image { family, drape } => {
                let target = drape_or(drape, target);
                let image = |untested| self.images.pipeline(cache, target, family, untested);
                let untested = split.then(|| lacks.need(gpu, image(true)));
                (untested.flatten(), lacks.need(gpu, image(false)))
            }
            DrawsWith::Paint { kind, drape } => {
                let target = drape_or(drape, target);
                let mut drawn = |untested, lacks: &mut Lacks| {
                    (self.paint).drawn((gpu, cache, target), (kind, untested), full, lacks)
                };
                let untested = split.then(|| drawn(true, &mut lacks));
                (untested.flatten(), drawn(false, &mut lacks))
            }
            // Waiting for the terrain, or drawn above.
            _ => return lacks,
        };
        let Some(pipeline) = pipeline else {
            return lacks;
        };
        let pipelines = [untested.unwrap_or(pipeline), pipeline];
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
                // A drape slot's pass draws its own runs alone, spliced in
                // here ([`Renderer::slot_commands`]).
                if let Some(splices) = splices.as_deref_mut() {
                    splices.push((commands.len(), id));
                    continue;
                }
            }
            match self.scene.segments(id) {
                Some(segments) => self.segment_draws(scenes, id, segments, commands),
                None => {
                    for (scene, row) in rows {
                        commands.extend([
                            RenderCommand::SetBindGroup {
                                index: 0,
                                group: scene,
                            },
                            self.driver.draw(buffers, row, node.cull.draw),
                        ]);
                    }
                }
            }
        }
        lacks
    }

    /// Indexed stream `id`'s draws of `segments`, each stretch whole: the
    /// cull wrote every slot of its runs, hidden indices for what is out
    /// of view, and holes between runs hold hidden ones.
    fn segment_draws(
        &self,
        scenes: &[Option<BindGroupId>],
        id: u32,
        segments: &[scene::Segment],
        commands: &mut Vec<RenderCommand>,
    ) {
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
}

/// Writes node `node`'s paint into `uniforms`: `words` at its row, and at
/// its second stream's (`second`), where an extrusion's cut stream takes
/// them saying it is one, and a split node's tested placements take
/// `paint` at their row's index. Whether any changed.
fn paint_rows(
    uniforms: &mut NodeUniforms,
    (node, second): (u32, Option<Second>),
    (paint, words): (&NodePaint, Vec<u32>),
) -> bool {
    let changed = uniforms.set(node, &words);
    changed
        | match second {
            Some(Second::Cut(row)) => {
                use orb_shaders::layout::extrusion_node as n;
                let mut cut = words;
                (cut[n::INDEX], cut[n::CUT]) = (row, 1);
                uniforms.set(row, &cut)
            }
            Some(Second::Tested(row)) => uniforms.set(row, &paint.uniform_words(row)),
            None => false,
        }
}

/// The drape atlas's passes where `drape`, else `target`.
fn drape_or(drape: bool, target: Attachments) -> Attachments {
    match drape {
        true => TargetFormat {
            color: DRAPE_FORMAT,
            sample_count: 1,
        }
        .into(),
        false => target,
    }
}

/// The main pass's recording ([`Renderer::record_main`]), as one value
/// that goes once anything it draws or binds changes.
#[derive(Debug, Default)]
struct MainRecording {
    /// The map's passes' bundles, and what the passes draw in order, the
    /// bundles among it.
    bundles: Vec<RenderBundleId>,
    steps: Vec<Step>,
    /// Each layer composite group's bundle, by group, `None` for one that
    /// records none.
    layers: Vec<Option<RenderBundleId>>,
    /// The stacks the bundles draw rows of composites with, and the fill
    /// whose colour each layer draws, if one does.
    stacks: Vec<(offscreen::Stack, Vec<Option<usize>>)>,
    /// What it waited for ([`Awaits`]): what of it arrives records it
    /// again.
    awaits: Awaits,
    /// Draws its bundles hold: those of nodes with candidates, and
    /// composites, stacks and blend draws.
    draws: u32,
    /// The solid background node its first bundle ends with: the bundle
    /// a frame the backgrounds fill leaves out, clearing to its colour.
    beneath: Option<usize>,
    /// The scene's segments revision it drew ([`Scene::segments`]).
    segments: u64,
}

impl MainRecording {
    /// Ends the bundle `commands` hold, if they hold any.
    fn bundle(
        &mut self,
        gpu: &mut dyn Gpu,
        desc: &RenderBundleDesc<'_>,
        commands: &mut Vec<RenderCommand>,
    ) -> Option<RenderBundleId> {
        if commands.is_empty() {
            return None;
        }
        let bundle = gpu.create_render_bundle(desc, &bound_once(std::mem::take(commands)));
        self.bundles.push(bundle);
        Some(bundle)
    }

    /// Ends the bundle `commands` hold as a step of nodes' draws, if they
    /// hold any.
    fn flush(
        &mut self,
        gpu: &mut dyn Gpu,
        desc: &RenderBundleDesc<'_>,
        commands: &mut Vec<RenderCommand>,
    ) {
        let bundle = self.bundle(gpu, desc, commands);
        self.steps.extend(bundle.map(Step::Bundle));
    }

    /// Whether the map's passes split at 3D layers.
    fn splits(&self) -> bool {
        self.steps.iter().any(|s| matches!(s, Step::ThreeD(_)))
    }
}

impl release::Release for MainRecording {
    fn release(self, gpu: &mut dyn Gpu) {
        let stacks: Vec<offscreen::Stack> = self.stacks.into_iter().map(|(s, _)| s).collect();
        (self.bundles, self.layers, stacks).release(gpu);
    }
}

/// What the scene groups bind besides the scene's tables: they are made
/// again when any of it changes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SceneKey {
    dem: TextureId,
    terrain_depth: TextureId,
    /// The scene's buffers' revision.
    buffers: u64,
    images: TextureId,
}

/// What a composite of a row drawn as stacks composites
/// ([`Renderer::stack_row`]).
#[derive(Clone, Copy, Debug)]
enum Composited {
    /// An image-process layer's output, into the map.
    Process(usize),
    /// A fill drawn through a target in its colour, by node.
    Fill(usize),
}

/// What the map's passes draw, in order ([`Renderer::record_bundles`]).
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The solid background and what it hides, after the sky: a frame the
    /// backgrounds fill clears to its colour in their place. Its bundle,
    /// `None` while they draw nothing yet (their pipelines compiling).
    Beneath(Option<RenderBundleId>),
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

impl DrapeStacks {
    /// Whether node `i` is a shaded hillshade that blends into its terrain
    /// layer, in a mode other than normal.
    fn blends_on_terrain(&self, i: usize) -> bool {
        self.nodes[i].1.is_some_and(|l| {
            (self.layers[usize::from(l)].shades.iter().flatten())
                .any(|s| s.node as usize == i && s.blend != Blend::Normal)
        })
    }
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

/// `commands` with each state set once: a pipeline, bind group or index
/// buffer set only where it changes what is set and a draw follows before
/// it is set again. Each node's draws set its pipeline, its groups and
/// its index buffer (the compacted indices, or symbols'
/// [`QUAD_INDICES`]), and each stretch its block's group, which nodes of
/// one kind and stretches of one block share; a node with nothing to draw
/// in a drape slot sets its state for no draw. Each call costs the CPU,
/// the web's a call into the browser.
fn bound_once(commands: Vec<RenderCommand>) -> Vec<RenderCommand> {
    /// What a state command sets: the pipeline, a group's index, or the
    /// index buffer.
    fn slot(command: &RenderCommand) -> Option<u32> {
        match command {
            RenderCommand::SetPipeline(_) => Some(u32::MAX),
            RenderCommand::SetIndexBuffer { .. } => Some(u32::MAX - 1),
            RenderCommand::SetBindGroup { index, .. } => Some(*index),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(commands.len());
    // What the commands kept leave set, and what is to be set before the
    // next draw.
    let mut set: Vec<RenderCommand> = Vec::new();
    let mut pending: Vec<RenderCommand> = Vec::new();
    for command in commands {
        if let Some(at) = slot(&command) {
            match pending.iter_mut().find(|c| slot(c) == Some(at)) {
                Some(waiting) => *waiting = command,
                None => pending.push(command),
            }
            continue;
        }
        for command in pending.drain(..) {
            let at = slot(&command);
            match set.iter_mut().find(|c| slot(c) == at) {
                Some(was) if *was == command => continue,
                Some(was) => *was = command.clone(),
                None => set.push(command.clone()),
            }
            out.push(command);
        }
        // Bundles leave nothing set after them.
        if matches!(command, RenderCommand::ExecuteBundles(_)) {
            set.clear();
        }
        out.push(command);
    }
    out
}

/// Whether `node`'s pipelines compile ahead of the other nodes' of a
/// view: a label's. A label shows only once its pipelines and the
/// placement pipeline that places every symbol layer have compiled, and
/// then a placement has run and committed, frames later; and that one
/// placement pipeline is a view's costliest compile (650 ms cold on an M6
/// where its fills' and lines' take 150 to 430). So the labels' chain is
/// the longest of a view: started first, it runs beside the other
/// layers' compiles, rather than after them, which on a loaded machine
/// kept a spun globe's labels (GL JS's `loaded()`) a second behind its
/// tiles.
fn labels_first(node: &Node) -> bool {
    node.family.is_symbol()
}

/// The style layer a node composites once with, when it does: below layer
/// opacity 1 or blending other than normally.
fn composite_group(composite: Option<(u32, f32, Blend)>) -> Option<u32> {
    composite
        .filter(|&(_, o, b)| o < 1.0 || b != Blend::Normal)
        .map(|(l, _, _)| l)
}

/// Where a node's commands record: a bundle's commands, and a drape
/// recording's splices ([`Recorded::splices`]).
type Recording<'a> = (
    &'a mut Vec<RenderCommand>,
    Option<&'a mut Vec<(usize, u32)>>,
);

/// What one bundle record found lacking besides the nodes' own
/// pipelines: nothing, where it can draw.
struct Ready {
    cull: Lacks,
    terrain: Lacks,
    /// The DEM prepare's, which hillshade and color relief wait for.
    prepare: Lacks,
    /// The hillshade tables', which hillshades wait for.
    tables: Lacks,
    /// The raster mip levels', which raster nodes wait for.
    mips: Lacks,
}

/// The drape pass: draped nodes' paint for the slots it redraws, the
/// slots' frames and the stacks' recorded draws.
#[derive(Debug)]
struct Drape {
    /// Draped nodes' paint at the zoom its slots draw at.
    uniforms: NodeUniforms,
    /// Per node, whether it draws: not outside its layer's zooms, or where
    /// it doesn't drape.
    drawn: Vec<bool>,
    /// Per page, by tile slot the drape holds, the frames its passes draw
    /// with, one a block slot, and the scene groups over them, which take
    /// the place of the map's in the slot's commands
    /// ([`Renderer::slot_frames`]).
    slot_scenes: [SlotScenes; 2],
    /// Per stack, the draws of the nodes drawing into it.
    recorded: Vec<Option<Recorded>>,
}

/// A drape page's frames and scene groups by tile slot, each by block
/// slot.
type SlotScenes = FxHashMap<u32, (Vec<Option<BufferId>>, Vec<Option<BindGroupId>>)>;

/// A drape stack's draws, recorded once for every slot's pass into it: its
/// draped nodes' commands, but for their runs, whose draws differ by slot.
/// Each pass draws its slot's runs alone, where replaying a bundle of
/// every redrawn slot's runs in each slot's pass cost the GPU every one of
/// them a pass (passes times slots), as it clipped all but the slot's own.
#[derive(Debug)]
struct Recorded {
    commands: Vec<RenderCommand>,
    /// Where an indexed stream's draws of a slot's runs go among the
    /// commands, with its row ([`Scene::slot_segments`]).
    splices: Vec<(usize, u32)>,
    /// The node groups the commands bind.
    groups: Vec<BindGroupId>,
    /// The rows it left out for want of candidates: those gaining some
    /// record it again.
    left_out: Vec<u32>,
    /// By tile slot, the commands its pass last drew and their bundle
    /// once it drew them again ([`Renderer::slot_draws`]): what they bind
    /// lives as long as these draws.
    slots: FxHashMap<u32, (Vec<RenderCommand>, Option<RenderBundleId>)>,
}

impl release::Release for Recorded {
    fn release(self, gpu: &mut dyn Gpu) {
        let bundles: Vec<RenderBundleId> =
            self.slots.into_values().filter_map(|(_, b)| b).collect();
        (self.groups, bundles).release(gpu);
    }
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
    /// Whether node `i` draws in the drape.
    fn draws(&self, i: usize) -> bool {
        self.drawn.get(i).copied().unwrap_or(false)
    }

    fn forget_recorded(&mut self, gpu: &mut dyn Gpu) {
        for recorded in &mut self.recorded {
            recorded.take().release(gpu);
        }
    }

    fn release(mut self, gpu: &mut dyn Gpu) {
        self.forget_recorded(gpu);
        (self.slot_scenes, self.uniforms).release(gpu);
    }
}

/// How the map draws, besides its nodes' paint and stacks: as it does
/// now, or as it would once its terrain turns on or goes
/// ([`Renderer::terrain_ready_as`]). What a node draws with follows from
/// it and the node's [`NodeView`] ([`Renderer::node_draws`]), so the
/// renderer asks after another way of drawing without becoming it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PassConfig {
    /// What the map's passes draw into.
    target: Attachments,
    /// Terrain is on.
    terrain_on: bool,
    /// The style's terrain is still to turn on: what drapes on it waits.
    terrain_ahead: bool,
    /// Frames draw with the paint and symbol pipelines' full variants,
    /// not their plane ones ([`Variants::wanted`](pipeline::Variants::wanted)).
    full: bool,
    /// The variants `[plane, full]` the pipeline set makes.
    variants: [bool; 2],
}

/// A node as what it draws with is asked after: what of its paint picks
/// its pipelines and where it drapes, as it is now
/// ([`Renderer::view`]), or as a pipeline set's paint at another zoom or
/// terrain turning on or going has it.
#[derive(Clone, Copy, Debug)]
struct NodeView {
    /// Its paint's shape: a symbol node's is its layer's
    /// ([`Renderer::layer_shape`]).
    shape: Shape,
    /// Its drape stack, where it drapes.
    stack: Option<u8>,
    /// The terrain layer that draws it, where one does.
    terrain_layer: Option<u8>,
}

/// Whether the terrain shades a hillshade, which the set says one may
/// (`shaded`) or a terrain layer of `layers` does, and whether a terrain
/// layer reads the map beneath it: the terrain's pipelines
/// ([`Renderer::request_terrain`]).
fn terrain_reads(shaded: bool, layers: &[terrain::TerrainLayer]) -> (bool, bool) {
    (
        shaded || layers.iter().any(terrain::TerrainLayer::shaded),
        layers.iter().any(terrain::TerrainLayer::reads_beneath),
    )
}

/// What a node draws with as the renderer is now
/// ([`Renderer::node_draws`]): its own draws' pipelines and the passes
/// before them that it reads.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NodeDraws {
    with: DrawsWith,
    /// It reads its DEM tiles prepared: a hillshade's or a color relief's.
    prepare: bool,
    /// It reads its table: a hillshade's.
    tables: bool,
    /// It samples its pages' mip levels: a raster's.
    mips: bool,
}

impl NodeDraws {
    /// Whether it draws into the drape atlas, which its terrain layer
    /// draws.
    fn drapes(&self) -> bool {
        matches!(
            self.with,
            DrawsWith::Slopes
                | DrawsWith::Paint { drape: true, .. }
                | DrawsWith::Image { drape: true, .. }
        )
    }
}

/// What a node's own draws draw with ([`NodeDraws`]).
#[derive(Clone, Copy, Debug, PartialEq)]
enum DrawsWith {
    /// Nothing until the style's terrain turns on, when it drapes.
    Terrain,
    /// A shaded hillshade's slopes, into the drape atlas, which the
    /// terrain pass lights.
    Slopes,
    /// The paint pipelines of a kind, into the map's passes or the drape
    /// atlas's (`drape`).
    Paint { kind: PaintKind, drape: bool },
    /// An image family's pipelines, into the map's passes or the drape
    /// atlas's.
    Image { family: Family, drape: bool },
    /// The symbol pipeline its layer's two halves draw with, and the
    /// placement dispatch's kind ([`Renderer::symbol_kind`],
    /// [`Renderer::placings`]).
    Symbols(symbol::SymbolKind, symbol::Placings),
    /// The heatmap's density, composited through its ramp.
    Heatmap,
    /// The 3D pipelines of a shape, composited over the map.
    Extrusion(paint::ExtrusionShape),
}

/// A paint node's pipeline ([`Renderer::paint_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PaintKind {
    family: Family,
    /// The styled line's.
    styled: bool,
    /// Its variant with property rows ([`orb_shaders::FEATURE_ROWS`]).
    rows: bool,
    /// Its variant with patterns ([`orb_shaders::PATTERNS`]).
    pattern: bool,
}

impl PaintKind {
    /// Its shader and what it draws, `None` for a family not of
    /// [`PaintPipelines`].
    fn shader(self) -> Option<(&'static Shader, PrimitiveTopology)> {
        let triangles = PrimitiveTopology::TriangleList;
        Some(match self.family {
            Family::Fill => (&FILL_SHADER, triangles),
            Family::FillOutline => (&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            Family::Line if self.styled => (&LINE_STYLED_SHADER, triangles),
            Family::Line => (&LINE_SHADER, triangles),
            Family::Circle => (&CIRCLE_SHADER, triangles),
            _ => return None,
        })
    }
}

/// The vector families' pipelines (fill, fill outline, line, styled line,
/// circle) by [`PaintKind`], into the map's passes or the drape atlas's,
/// tested and untested where its family has untested placements
/// ([`Untesting::of_family`]): a fill's fragments testing nothing, a
/// line's clipped by their placement's planes instead
/// (`LINE_PLANES_SHADER`).
#[derive(Debug)]
struct PaintPipelines {
    layouts: [BindGroupLayoutId; 2],
    /// Which placements of the map's nodes draw untested.
    untesting: Untesting,
}

impl PaintPipelines {
    /// What a frame draws a node of `kind` with into passes of `target`,
    /// its untested placements where `untested`, `lacks` gaining what it
    /// waits for ([`Variants::drawn`](pipeline::Variants::drawn)).
    fn drawn(
        &self,
        (gpu, cache, target): (&mut dyn Gpu, &Cache, Attachments),
        (kind, untested): (PaintKind, bool),
        full: bool,
        lacks: &mut Lacks,
    ) -> Option<RenderPipelineId> {
        let Some(draw) = self.draw(target, kind, untested) else {
            lacks.push(None);
            return None;
        };
        cache.variants(draw).drawn(gpu, full, lacks)
    }

    /// What a frame into passes of `target` waits for to draw a node of
    /// `kind` ([`Variants::wanted`](pipeline::Variants::wanted)).
    fn wanted(
        &self,
        (cache, target): (&Cache, Attachments),
        kind: PaintKind,
        full: bool,
    ) -> Option<RenderPipelineId> {
        cache.variants(self.draw(target, kind, false)?).wanted(full)
    }

    /// Makes the pipelines of a node of `kind` into passes of `target` in
    /// each of the variants `[plane, full]` asked for.
    fn make(
        &self,
        (gpu, cache): (&mut dyn Gpu, &mut Cache),
        target: Attachments,
        kind: PaintKind,
        variants: [bool; 2],
    ) {
        for untested in [false, true] {
            if let Some(draw) = self.draw(target, kind, untested) {
                cache.make_variants(gpu, variants, draw);
            }
        }
    }

    /// The tested or untested draw of a node of `kind` into passes of
    /// `target`, the untested one where its family has untested
    /// placements.
    fn draw(&self, target: Attachments, kind: PaintKind, untested: bool) -> Option<Draw<'_>> {
        let (shader, topology) = kind.shader()?;
        let tested = Draw {
            topology,
            rows: kind.rows,
            patterns: kind.pattern,
            ..Draw::new(
                shader,
                &self.layouts,
                target,
                Some(BlendState::PremultipliedAlpha),
            )
        };
        if !untested {
            return Some(tested);
        }
        let vertex = match self.untesting.of_family(kind.family)? {
            // Its whole square: its own vertices, no fragment tested.
            Untested::Square => shader,
            // Within its clip rectangle, which its planes clip.
            Untested::Rect if kind.styled => &LINE_STYLED_PLANES_SHADER,
            Untested::Rect => &LINE_PLANES_SHADER,
            Untested::None => return None,
        };
        Some(Draw {
            label: match kind.family {
                Family::Fill => "fill untested",
                _ => vertex.label,
            },
            vertex: (vertex, vertex.vertex),
            constants: [&[], &[(orb_shaders::CLIP_FRAGMENTS, 0.0)]],
            ..tested
        })
    }
}

/// The scene's groups, by block slot (`None` for a free one): each
/// block's frame (`frames`' if given, else the block's own) and pools,
/// the tables every block shares, the image pages and the terrain.
fn scene_groups(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    scene: &Scene,
    images: &ImageBindings,
    (nearest, dem, terrain_depth): (SamplerId, TextureId, TextureId),
    frames: Option<&[Option<BufferId>]>,
) -> Vec<Option<BindGroupId>> {
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
                BindGroupEntry::texture(9, images.texture),
                BindGroupEntry::sampler(10, images.sampler),
                BindGroupEntry::texture(11, images.dashes),
                BindGroupEntry::sampler(13, nearest),
                BindGroupEntry::texture(14, dem),
                BindGroupEntry::texture(15, terrain_depth),
                BindGroupEntry::texture(16, images.texture),
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
    /// group and whether it's a shaded hillshade; then each node's stack and
    /// terrain layer, the layers and the counts.
    #[test]
    fn drape_stacks_by_case() {
        let node = |draped, group, shaded| StackInput {
            draped,
            group,
            blend: match group {
                Some(_) => Blend::Multiply,
                None => Blend::Normal,
            },
            shaded,
        };
        let plain = node(true, None, false);
        let layer = |below| TerrainLayer {
            below,
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
                vec![node(false, None, false); 2],
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
                    layers: vec![layer(Some(0))],
                    counts: [1, 0],
                },
            ),
            (
                "a node that doesn't drape ends the run",
                vec![plain, node(false, None, false), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (None, None), (Some(1), Some(1))],
                    layers: vec![layer(Some(0)), layer(Some(1))],
                    counts: [2, 0],
                },
            ),
            (
                "a layer that composites once has a terrain layer of its own",
                vec![plain, node(true, Some(5), false)],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(1), Some(1))],
                    layers: vec![layer(Some(0)), layer(Some(1))],
                    counts: [2, 0],
                },
            ),
            (
                "a shaded hillshade's slopes stack after the colour stacks",
                vec![plain, node(true, None, true), plain],
                DrapeStacks {
                    nodes: vec![(Some(0), Some(0)), (Some(2), Some(0)), (Some(1), Some(0))],
                    layers: vec![shaded(Some(0), &[shade(1, 2, Some(1))])],
                    counts: [2, 1],
                },
            ),
            (
                "a second shaded hillshade shades the same terrain layer",
                vec![node(true, None, true), plain, node(true, None, true)],
                DrapeStacks {
                    nodes: vec![(Some(1), Some(0)), (Some(0), Some(0)), (Some(2), Some(0))],
                    layers: vec![shaded(None, &[shade(0, 1, Some(0)), shade(2, 2, None)])],
                    counts: [1, 2],
                },
            ),
            (
                "a third starts a terrain layer",
                vec![node(true, None, true); 3],
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
                vec![plain, node(true, Some(5), true), plain],
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
                    node(true, None, true),
                    node(true, Some(5), true),
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
                    node(true, None, true),
                    node(true, None, true),
                    node(true, Some(5), true),
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
                vec![node(true, Some(5), true), plain],
                DrapeStacks {
                    nodes: vec![(Some(1), Some(0)), (Some(0), Some(1))],
                    layers: vec![shaded(None, &[shade(0, 1, None)]), layer(Some(0))],
                    counts: [1, 1],
                },
            ),
        ];
        for (name, inputs, expected) in cases {
            assert_eq!(drape_stacks(&inputs), expected, "{name}");
        }
    }

    /// Commands bound once draw each draw with the state the commands
    /// gave it, setting nothing that is set already or that no draw uses.
    #[test]
    fn bound_once_draws_alike_with_each_state_set_once() {
        use RenderCommand::*;
        let pipeline = |n| SetPipeline(RenderPipelineId(n));
        let group = |index, n| SetBindGroup {
            index,
            group: BindGroupId(n),
        };
        let indices = SetIndexBuffer {
            buffer: BufferId(1),
            offset: 0,
        };
        let draw = |n| DrawIndexed {
            indices: n..n + 3,
            base_vertex: 0,
            instances: 0..1,
        };
        // Each draw with the state set before it.
        let drawn = |commands: &[RenderCommand]| {
            let mut set: Vec<RenderCommand> = Vec::new();
            let mut draws = Vec::new();
            for c in commands {
                match c {
                    DrawIndexed { .. } => draws.push((set.clone(), c.clone())),
                    ExecuteBundles(_) => set.clear(),
                    c => {
                        let same = |s: &RenderCommand| {
                            std::mem::discriminant(s) == std::mem::discriminant(c)
                                && match (s, c) {
                                    (
                                        SetBindGroup { index: a, .. },
                                        SetBindGroup { index: b, .. },
                                    ) => a == b,
                                    _ => true,
                                }
                        };
                        set.retain(|s| !same(s));
                        set.push(c.clone());
                        set.sort_by_key(|s| format!("{s:?}"));
                    }
                }
            }
            draws
        };
        let commands = vec![
            // A node with nothing to draw.
            pipeline(1),
            group(1, 10),
            indices.clone(),
            // A node drawing two stretches of one block.
            pipeline(1),
            group(1, 11),
            indices.clone(),
            group(0, 5),
            draw(0),
            group(0, 5),
            draw(3),
            // One of another kind, the same block.
            pipeline(2),
            group(1, 12),
            indices.clone(),
            group(0, 5),
            draw(6),
            ExecuteBundles(vec![RenderBundleId(1)]),
            // Its state again, after a bundle left none.
            pipeline(2),
            group(1, 12),
            indices.clone(),
            group(0, 5),
            draw(9),
            // State no draw follows.
            pipeline(3),
        ];
        let once = bound_once(commands.clone());
        assert_eq!(drawn(&once), drawn(&commands));
        assert_eq!(
            once,
            vec![
                pipeline(1),
                group(1, 11),
                indices.clone(),
                group(0, 5),
                draw(0),
                draw(3),
                pipeline(2),
                group(1, 12),
                draw(6),
                ExecuteBundles(vec![RenderBundleId(1)]),
                pipeline(2),
                group(1, 12),
                indices,
                group(0, 5),
                draw(9),
            ]
        );
    }
}
