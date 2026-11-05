use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_camera::{CameraState, View, Viewport};
use orb_crs::{TileIndex, TileMatrixSet};

use crate::surface::{Plan, Surface};
use orb_hal::{Gpu, HostBytesId, TextureId};
use orb_host::{Fetch, FetchError, ImageDecode};
use orb_render::GpuTile;
use orb_render::{
    ClipRect, Curve, DemRef, Family, FillPaint, FrameBlock, FrameStats, LinePaint, MeshData,
    MeshId, MeshSize, NodePaint, OutOfSpace, PageId, PatternPaint, PlacementTarget, RenderNode,
    Renderer, SceneCapacity, SceneStats, TargetFormat, TileKey, TileRow, TileWarp, Translate,
    Words,
};
use orb_render::{ImageAtlas, ImageUpload};
use orb_source::{Draw, TilePyramid};
use orb_style::{LayerType, SourceKind, TranslateAnchor};
use orb_v8::StyleDoc;

use crate::MapEvent;
use crate::image::{self, ResidentImage};
use crate::jobs::{Jobs, ResultMesh, Stream, StyleUpdate, TileRequest, TileResult, TileWorker};
use crate::state::FeatureStates;
use crate::style::{PaintProperty, StylePlan};
use orb_source::endpoint::TileEndpoint;

/// Logical pixel size of vector/background tiles and pattern origins.
const TILE_PX: f64 = 512.0;

/// Milliseconds a frame spends applying arrived tiles before it leaves the
/// rest for the next frame (plan §7.4's upload budget at 60 Hz).
const TILE_APPLY_MS: f64 = 1.5;

/// GPU images shared by the maps in one Runtime / host-service set. The
/// first render initializes them on that Runtime's device.
#[derive(Clone, Debug, Default)]
pub struct Images(pub(crate) Arc<Mutex<ImageState>>);

impl Images {
    /// Forgets the atlas with its device: call when the device is replaced,
    /// before maps draw with renderers on the new one. Each map adds its
    /// images again as it lays its style out again.
    pub fn reset(&self) {
        *self.0.lock().expect("images poisoned") = ImageState::default();
    }
}

#[derive(Debug, Default)]
pub(crate) struct ImageState {
    pub(crate) atlas: Option<ImageAtlas>,
    retired: Vec<u64>,
    retired_ramps: Vec<u32>,
}

/// What a map needs from its host.
#[derive(Clone)]
pub struct Services {
    /// Loads styles and TileJSON.
    pub fetch: Arc<dyn Fetch>,
    pub clock: Arc<dyn orb_host::Clock>,
    pub image_decode: Arc<dyn ImageDecode>,
    pub images: Images,
    /// Loads and lays out tiles.
    pub jobs: Arc<dyn Jobs>,
    /// Called from any thread when new data arrived and a frame should be
    /// drawn.
    pub wake: Arc<dyn Fn() + Send + Sync>,
}

impl std::fmt::Debug for Services {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Services")
    }
}

/// What a frame did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameReport {
    pub frame: FrameStats,
    pub scene: SceneStats,
    /// Tiles requested and not yet arrived.
    pub tiles_loading: u32,
    /// Tiles drawn, counting world copies and fallbacks.
    pub tiles_drawn: u32,
    /// The style generation the frame shows in full: every edit up to it
    /// has all its jobs applied (plan §7.1).
    pub generation: u64,
    /// Nothing is loading: the style, its sources and every tile the view
    /// needs have arrived, so the frame is complete.
    pub idle: bool,
    /// As `idle`, but fades may still run: labels are placed for this
    /// view and cross-fades need only time (GL JS `loaded()`).
    pub loaded: bool,
    /// GPU memory by pool, and the tile cache (plan §8's budget).
    pub memory: Memory,
    /// Microseconds this thread spent on placement: building the frame the
    /// placement lane places and applying its commits (plan §6.1 step 10),
    /// not the placement itself, which is the lane's (inline on a host
    /// without a placement thread or worker).
    pub placement_us: u32,
    /// Only compiling pipelines keep the frame from being complete: a
    /// host whose GPU wakes it when a pipeline compiles needs no frames
    /// until then (plan §8.4).
    pub compiling: bool,
}

/// The map's memory against its budget (plan §8), in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    /// The GPU memory budget the scene's pools are sized from.
    pub budget: u64,
    pub scene: orb_render::SceneMemory,
    /// Raster and DEM tile pages, and glyph pages.
    pub tile_textures: u64,
    pub glyph_textures: u64,
    /// Icons and patterns.
    pub image_atlas: u64,
    /// Tiles kept beyond those in view, and their cost: GPU memory and
    /// picking data.
    pub cached_tiles: u32,
    pub cached: u64,
    /// What the cache may hold: a quarter of the budget.
    pub cache_budget: u64,
}

impl Memory {
    /// GPU bytes allocated.
    pub fn allocated(&self) -> u64 {
        let s = &self.scene;
        s.vertices.allocated
            + s.indices.allocated
            + s.compacted.allocated
            + s.props.allocated
            + s.tables
            + self.tile_textures
            + self.glyph_textures
            + self.image_atlas
    }
}

/// What the map's tile jobs and uploads have done since it was created,
/// for frame-sequence tests and the HUD.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobCounts {
    /// Tile loads: fetch (or cache), decode and lay out every node.
    pub loads: u32,
    /// Jobs that lay out some nodes of a resident tile again.
    pub relayouts: u32,
    /// Jobs that evaluate only the property rows of resident meshes.
    pub columns: u32,
    /// Meshes uploaded, geometry and rows.
    pub meshes_uploaded: u32,
    /// Property rows replaced without geometry.
    pub rows_uploaded: u32,
    /// Feature-state table rows written.
    pub state_rows: u64,
}

enum Message {
    /// A placement committed, for the labels of `epoch`.
    Placed {
        epoch: u64,
        committed: orb_symbol::Committed,
    },
    /// A glyph raster job finished, for these code points and rows of the
    /// labels of `epoch`.
    Raster {
        epoch: u64,
        rows: Vec<crate::labels::RasterRow>,
        result: Result<Vec<crate::GlyphBitmap>, String>,
    },
    Style {
        request: u64,
        /// Where the document came from, for a scene document's relative
        /// style URL.
        url: String,
        /// A scene document's style: keep its surface.
        scene: bool,
        result: Result<String, FetchError>,
    },
    Sprite {
        generation: u64,
        result: Result<Vec<crate::sprite::SpriteImage>, String>,
    },
    TileJson {
        generation: u64,
        /// The source's key.
        source: u32,
        result: Result<String, FetchError>,
    },
    Tile {
        generation: u64,
        /// The source's key.
        source: u32,
        tile: TileIndex,
        /// The style generation of the edit a relayout or column job is
        /// for; `None` for a load.
        edit: Option<u64>,
        /// A DEM tile for the terrain rather than for the source's layers.
        terrain: bool,
        result: Result<TileResult, String>,
    },
}

/// The drape slots a frame redraws: terrain tiles with their tile slots,
/// and where slots are.
#[derive(Debug)]
pub(crate) struct DrapePlan {
    /// The surface's matrix set, which terrain tiles are of.
    pub(crate) surface: TileMatrixSet,
    pub(crate) layout: orb_render::DrapeLayout,
    pub(crate) redraw: Vec<(orb_crs::CoveredTile, u32)>,
}

/// A tile in the scene: one scene tile per layer extent holding a mesh per
/// render node with features, and an image tile's texture layer.
#[derive(Debug)]
pub(crate) struct ResidentTile {
    /// What the renderer holds of it: its meshes in scene tiles, its
    /// patches, an image tile's layer.
    pub(crate) gpu: GpuTile,
    /// What the map keeps of each mesh, symbol meshes included.
    pub(crate) meshes: Vec<Resident>,
    pub(crate) image: Option<ResidentImage>,
    /// A tile drawn through a warp's patches: its series.
    pub(crate) series: Option<Arc<orb_crs::Series>>,
    /// The style zoom it laid out at (decision 0022).
    pub(crate) zoom: f64,
    /// Its symbol layers' buckets; their meshes are among `meshes`.
    pub(crate) symbols: Vec<crate::labels::ResidentSymbols>,
    /// What it costs to keep: GPU bytes of its meshes and texture layer,
    /// and its picking data (see [`tile_cost`]).
    pub(crate) cost: u64,
}

impl ResidentTile {
    pub(crate) fn new(gpu: GpuTile, zoom: f64) -> Self {
        Self {
            gpu,
            meshes: Vec::new(),
            image: None,
            series: None,
            zoom,
            symbols: Vec::new(),
            cost: 0,
        }
    }
}

/// One node's mesh of a resident tile.
#[derive(Debug)]
pub(crate) struct Resident {
    /// The node's key.
    pub(crate) node: u32,
    pub(crate) mesh: MeshId,
    pub(crate) hits: Arc<crate::MeshHits>,
    /// The style revisions its geometry and its rows were laid out with.
    pub(crate) layout_rev: u64,
    pub(crate) rows_rev: u64,
    /// Gradient ramps its rows retain.
    pub(crate) ramps: Vec<u32>,
    /// The feature-state indices its features refer to.
    pub(crate) states: Vec<u32>,
}

#[derive(Debug)]
pub(crate) enum SourceState {
    Resolving,
    Ready {
        endpoint: Arc<TileEndpoint>,
        pyramid: TilePyramid<ResidentTile>,
        loading: u32,
        /// How its tiles reach the surface when they are on another plane.
        warp: Option<Box<WarpState>>,
    },
    Failed,
}

/// A `RasterWarp` source's plan and the footprints its tiles have had.
#[derive(Debug)]
pub(crate) struct WarpState {
    warp: orb_crs::Warp,
    footprints: HashMap<TileIndex, Option<orb_crs::Footprint>>,
    /// Vector tiles' quadratic patches.
    series: HashMap<TileIndex, Option<Arc<orb_crs::Series>>>,
}

impl WarpState {
    /// Tiles whose footprints and series are kept before the caches start
    /// over.
    const CACHED: usize = 4096;
    /// A quarter pixel at 512 pixels per tile side.
    const TOLERANCE: f64 = 1.0 / 2048.0;

    fn new(warp: orb_crs::Warp) -> Self {
        Self {
            warp,
            footprints: HashMap::new(),
            series: HashMap::new(),
        }
    }

    pub(crate) fn footprint(&mut self, tile: TileIndex) -> Option<orb_crs::Footprint> {
        if self.footprints.len() >= Self::CACHED {
            self.footprints.clear();
        }
        let warp = &self.warp;
        *self
            .footprints
            .entry(tile)
            .or_insert_with(|| warp.footprint(tile))
    }

    /// A vector tile's patches, or `None` if it can't be drawn through
    /// them (it crosses a wrapping plane's seam, or bends too much).
    pub(crate) fn series(&mut self, tile: TileIndex) -> Option<Arc<orb_crs::Series>> {
        if self.series.len() >= Self::CACHED {
            self.series.clear();
        }
        let warp = &self.warp;
        self.series
            .entry(tile)
            .or_insert_with(|| warp.series(tile, Self::TOLERANCE).map(Arc::new))
            .clone()
    }
}

/// A render node's state across style revisions, by node key.
#[derive(Debug, Default)]
pub(crate) struct NodeState {
    /// Per paint property, the largest value in any resident property row.
    pub(crate) maxima: Vec<f32>,
    /// The stops of a color-relief node's ramp, once written.
    pub(crate) ramp: Option<u32>,
    /// The revisions since which tiles laid out, or with rows evaluated,
    /// before are stale for the node.
    pub(crate) layout_epoch: u64,
    pub(crate) rows_epoch: u64,
}

#[derive(Debug)]
pub(crate) struct LoadedStyle {
    /// Identifies the style to the jobs.
    pub(crate) id: u64,
    pub(crate) doc: StyleDoc,
    pub(crate) worker: Arc<TileWorker>,
    /// Parallel to the plan's sources.
    pub(crate) sources: Vec<SourceState>,
    /// Sources an edit removed, whose tiles leave the scene next frame.
    pub(crate) retired: Vec<SourceState>,
    pub(crate) nodes: HashMap<u32, NodeState>,
    /// The node keys the renderer's nodes are of, in order: the plan's,
    /// except between an edit and the next frame.
    pub(crate) drawn: Vec<u32>,
    /// The zoom and bearing node paint was last written for; `None` to
    /// rewrite.
    pub(crate) painted: Option<(f64, f64)>,
    /// The draped nodes' paint at drape zooms.
    drape: DrapePaint,
    painted_fade: Option<Crossfade>,
    zoom_history: ZoomHistory,
    pub(crate) image_namespace: Option<u64>,
    sprites_started: bool,
    pub(crate) sprites_loading: u32,
    /// Sprite metadata loaded so far, and the images the host added; the
    /// jobs get it once every sprite has resolved, before tiles with symbol
    /// layers are requested.
    pub(crate) icons: crate::SpriteIcons,
    /// Image edits for the next frame's atlas.
    pub(crate) image_edits: Vec<crate::style_image::AtlasEdit>,
    /// Image names reported missing since they were last added.
    pub(crate) reported_missing: std::collections::HashSet<String>,
    /// The scene tile of the background layers' quads.
    background: Option<GpuTile>,
    /// Tiles with placements in publication order, parallel to sources.
    pub(crate) placed: Vec<Vec<TileIndex>>,
    /// Tiles that left the draw list but whose labels still draw until a
    /// placement covers what replaced them (`Labels::hold`), parallel to
    /// sources.
    pub(crate) held: Vec<Vec<TileIndex>>,
    pub(crate) states: FeatureStates,
    /// This frame's drawn symbol buckets, for placement.
    pub(crate) symbol_tiles: Vec<crate::labels::SymbolTile>,
    /// This frame's tiles to draw, per source, between their update and
    /// their placement.
    pub(crate) draws: Vec<Option<Vec<Draw>>>,
    /// The terrain, once its DEM source is ready.
    pub(crate) terrain: Option<crate::terrain::Terrain>,
    /// The terrain's DEM is on another tile matrix set than the surface;
    /// warned once.
    terrain_unusable: bool,
    /// Whether the background and image tiles' meshes are grids for a
    /// globe to bend.
    globe_meshes: bool,
}

/// The host data a tile result holds until it's uploaded or dropped.
fn host_data(result: &TileResult) -> (Option<HostBytesId>, Option<orb_hal::HostImageId>) {
    let image = match &result.image {
        Some(crate::TileImage {
            pixels: crate::ImagePixels::Host(image),
            ..
        }) => Some(*image),
        _ => None,
    };
    (result.host_bytes, image)
}

/// What tiles and styles let go of between frames, released from the
/// renderer at the next.
#[derive(Debug, Default)]
struct Retired {
    /// Tiles of sources and the background.
    tiles: Vec<GpuTile>,
    /// A replaced style's feature-state table.
    tables: Vec<std::ops::Range<u32>>,
}

impl Retired {
    fn release(&mut self, renderer: &mut Renderer) {
        for tile in self.tiles.drain(..) {
            renderer.release_tile(tile);
        }
        for table in self.tables.drain(..) {
            renderer.free_table(table);
        }
    }
}

/// A tile result waiting for room in the scene (`Map::apply_tile`).
#[derive(Debug)]
struct WaitingTile {
    generation: u64,
    /// The plan source's key.
    source: u32,
    tile: TileIndex,
    build: TileResult,
    /// [`Map::room_state`] when it last failed to fit.
    tried: Option<[u64; 6]>,
}

/// A map: a style, a camera and the tiles they need, drawn by
/// [`render`](Self::render).
/// How a map renders (plan §14.2): interactively, or the still output of
/// servers and snapshots.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum RenderMode {
    /// Interactive: labels place against the view and fade, and raster
    /// tiles fade in.
    #[default]
    Continuous,
    /// One picture: nothing fades and labels place as if nothing had been
    /// placed before, so that the same inputs draw the same pixels.
    Still,
    /// One tile of a grid, drawn with its neighbours around it: `Still`,
    /// with labels placed in tile scope over the grid's tiles listed, so
    /// that tiles drawn apart agree on the labels that cross their edges.
    Tile(Vec<GridTile>),
}

/// A tile of the grid [`RenderMode::Tile`] places labels over.
#[derive(Clone, Debug, PartialEq)]
pub struct GridTile {
    /// Its column and row: neighbours differ by one.
    pub col: i64,
    pub row: i64,
    /// A camera centred on it, looking straight down and unrotated, whose
    /// viewport is the tile; the same size for every tile.
    pub camera: CameraState,
    pub viewport: Viewport,
    /// Its centre in world units, unwrapped: on a plane that repeats, a
    /// tile past the edge stays beside its neighbours.
    pub center: glam::DVec2,
}

pub struct Map {
    /// Interactive, a still picture or a tile ([`Map::set_mode`]).
    pub(crate) mode: RenderMode,
    /// Placement and the scene side of labels.
    pub(crate) labels: crate::labels::Labels,
    /// Counts the [`Labels`](crate::labels::Labels) a GPU reset replaced.
    label_epoch: u64,
    /// The last frame's picking state; queries project from it.
    pub(crate) picking: std::cell::RefCell<crate::query::Picking>,
    /// This frame's placement time so far, in milliseconds.
    placement_ms: f64,
    /// Arrived tiles wait for the next frame: this one spent its budget.
    tiles_deferred: bool,
    /// Tiles that arrived when the scene had no room for them, while other
    /// tiles' jobs were in flight: those tiles' arrivals free the stand-ins
    /// they replace, so these try again on the next frames.
    waiting: Vec<WaitingTile>,
    /// Chunks the last frame's cull left out for want of room.
    candidates_dropped: u32,
    pub(crate) services: Services,
    videos: crate::video::Videos,
    /// The camera's only writer: the app's moves, the style's camera and
    /// gestures all go through it (decision 0042).
    pub(crate) camera: orb_camera::Camera,
    pub(crate) input: orb_interact::Interaction,
    /// The camera and terrain revisions the last frame saw.
    pub(crate) camera_seen: u64,
    pub(crate) terrain_seen: u64,
    /// Terrain tiles hang skirts ([`Map::set_terrain_skirts`]).
    terrain_skirts: bool,
    antialias_3d: bool,
    /// The plane the map draws on and its sources' tile matrix sets: the
    /// scene's, planned for its tiles.
    pub(crate) surface: Surface,
    /// A scene document whose style is loading, which the style parses
    /// under (models resolved against the scene).
    pending_scene: Option<orb_v8::SceneDocument>,
    /// When the map began, by its clock: frames' time counts from it.
    born: f64,
    /// When the last raster tile to arrive finishes fading in.
    image_fades_until: f64,
    /// Paint edits easing in (decision 0031).
    pub(crate) transitions: crate::transitions::Transitions,
    /// The latest `load_style_url` request; earlier ones are ignored.
    style_request: u64,
    generation: u64,
    tile_generation: u64,
    pub(crate) style: Option<LoadedStyle>,
    /// The renderer whose tiles the map holds, by its id: a frame with
    /// another one forgets them.
    renderer: Option<u64>,
    /// What tiles and styles let go of between frames, for the next frame
    /// to release from the renderer.
    retired: Retired,
    /// The renderer's nodes are of a replaced style.
    nodes_stale: bool,
    capacity: SceneCapacity,
    /// The GPU memory budget (plan §8).
    budget: u64,
    /// The last style id given to the jobs.
    style_id: u64,
    /// Counts style revisions: every new style and every edit.
    pub(crate) version: u64,
    /// Outstanding relayout and column jobs, per edit generation.
    pub(crate) pending: BTreeMap<u64, u32>,
    pub(crate) counts: JobCounts,
    clear_color: [f64; 4],
    sender: Sender<Message>,
    receiver: Receiver<Message>,
    pub(crate) warnings: Vec<String>,
    pub(crate) events: crate::events::EventQueue,
    /// Terrain pages of a replaced style, to destroy with the next
    /// frame's GPU.
    retired_pages: Vec<PageId>,
    retired_images: Vec<u64>,
    pub(crate) retired_ramps: Vec<u32>,
    /// This frame's pattern and dash cross-fade.
    crossfade: Crossfade,
    /// What elevated content reads of the terrain this frame.
    frame_terrain: Option<orb_render::FrameTerrain>,
    /// Tiles' patches on the globe.
    globe_warps: crate::globe::GlobeWarps,
    /// Terrains made so far: each new one's id.
    terrains: u64,
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("camera", self.camera.state())
            .finish_non_exhaustive()
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        let mut images = self.services.images.0.lock().expect("images poisoned");
        images.retired.append(&mut self.retired_images);
        images.retired_ramps.append(&mut self.retired_ramps);
        if let Some(namespace) = self.style.as_ref().and_then(|s| s.image_namespace) {
            images.retired.push(namespace);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Crossfade {
    pub(crate) from_scale: f32,
    pub(crate) mix: f32,
}

/// The draped nodes' paint at the drape zooms slots are drawn at
/// (decision 0019), and what tells a slot's content changed with it.
#[derive(Debug, Default)]
struct DrapePaint {
    /// Zooms written since the paint last changed.
    painted: HashSet<u8>,
    /// Per zoom, a revision that changes with what is written there.
    revisions: HashMap<u8, u64>,
    /// Zooms where something drawn turns with the bearing, and the
    /// bearing they were written at.
    turning: HashSet<u8>,
    bearing: f64,
    /// The drape layout generation written for.
    generation: Option<u64>,
}

#[derive(Debug, Default)]
struct ZoomHistory {
    zoom: Option<f64>,
    integer: f64,
    changed: Option<f64>,
}

impl ZoomHistory {
    fn update(&mut self, zoom: f64, now: f64) -> Crossfade {
        match self.zoom {
            None => self.integer = zoom.floor(),
            Some(old) if old.floor() < zoom.floor() => {
                self.integer = zoom.floor();
                self.changed = Some(now);
            }
            Some(old) if old.floor() > zoom.floor() => {
                self.integer = zoom.floor() + 1.0;
                self.changed = Some(now);
            }
            _ => {}
        }
        self.zoom = Some(zoom);
        let t = self
            .changed
            .map_or(1.0, |at| ((now - at) / 300.0).clamp(0.0, 1.0));
        let fraction = zoom - zoom.floor();
        if zoom > self.integer {
            Crossfade {
                from_scale: 2.0,
                mix: (fraction + (1.0 - fraction) * t) as f32,
            }
        } else {
            Crossfade {
                from_scale: 0.5,
                mix: (1.0 - (1.0 - t) * fraction) as f32,
            }
        }
    }
}

/// Placement's lane: a thread of its own where there are threads.
fn default_lane() -> Arc<dyn crate::placement::PlacementLane> {
    if cfg!(target_arch = "wasm32") {
        Arc::new(crate::placement::InlinePlacement::default())
    } else {
        crate::placement::ThreadPlacement::new()
    }
}

/// The jobs' view of a whole document: every layer and source.
pub(crate) fn full_update(
    doc: &StyleDoc,
    revision: u64,
    local_ideographs: Option<String>,
    surface: crate::SurfaceTiles,
) -> StyleUpdate {
    StyleUpdate {
        revision,
        layers: doc.layer_keys().map(|k| k.0).collect(),
        sources: doc.source_keys().map(|k| k.0).collect(),
        layer_json: doc
            .layer_keys()
            .enumerate()
            .map(|(i, k)| (k.0, Arc::new(doc.layer_json(i).clone())))
            .collect(),
        source_json: doc
            .source_keys()
            .enumerate()
            .map(|(i, k)| {
                let id = doc.style().sources[i].id.clone();
                (k.0, id, Arc::new(doc.source_json(i).clone()))
            })
            .collect(),
        state: doc.global_state(),
        glyphs: doc.style().glyphs.clone(),
        font_faces: doc.style().font_faces.clone(),
        local_ideographs,
        surface,
        models: StyleUpdate::models_of(doc.style()),
    }
}

/// A revision of `doc`'s plan, keeping `previous`'s source data.
pub(crate) fn revise(
    previous: Option<&TileWorker>,
    doc: &StyleDoc,
    new_data: &[u32],
    revision: u64,
    local_ideographs: Option<String>,
    surface: crate::SurfaceTiles,
) -> TileWorker {
    TileWorker::with_style(
        previous,
        Arc::clone(doc.style()),
        doc.layer_keys().map(|k| k.0).collect(),
        doc.source_keys().map(|k| k.0).collect(),
        new_data,
        revision,
        local_ideographs,
        surface,
    )
}

impl Map {
    /// A map with no style. With `camera: None`, the first style's camera
    /// is used, as GL JS does when a map is created without one.
    pub fn new(services: Services, camera: Option<CameraState>) -> Self {
        let (sender, receiver) = channel();
        let surface = Surface::default();
        let start = CameraState::new(orb_crs::Geodetic::new(0.0, 0.0), 0.0, surface.projection());
        let viewport = Viewport {
            width: 512.0,
            height: 512.0,
        };
        let mut store = orb_camera::Camera::new(start, surface.view_surface(), viewport);
        // A camera given is the app's: the style's doesn't replace it.
        if let Some(camera) = camera {
            store.jump_to_state(camera);
            store.take_events();
        }
        let born = services.clock.now();
        Self {
            mode: RenderMode::Continuous,
            born,
            image_fades_until: born,
            picking: Default::default(),
            placement_ms: 0.0,
            tiles_deferred: false,
            waiting: Vec::new(),
            candidates_dropped: 0,
            services,
            videos: Default::default(),
            camera: store,
            input: orb_interact::Interaction::default(),
            camera_seen: 0,
            terrain_seen: 0,
            surface,
            pending_scene: None,
            transitions: Default::default(),
            style_request: 0,
            generation: 0,
            tile_generation: 0,
            style: None,
            renderer: None,
            retired: Retired::default(),
            terrain_skirts: true,
            antialias_3d: true,
            nodes_stale: false,
            capacity: SceneCapacity::default(),
            budget: orb_render::DESKTOP_BUDGET,
            style_id: 0,
            version: 0,
            pending: BTreeMap::new(),
            counts: JobCounts::default(),
            clear_color: [0.0; 4],
            sender,
            receiver,
            warnings: Vec::new(),
            events: Default::default(),
            retired_pages: Vec::new(),
            retired_images: Vec::new(),
            retired_ramps: Vec::new(),
            crossfade: Crossfade {
                from_scale: 1.0,
                mix: 1.0,
            },
            frame_terrain: None,
            globe_warps: Default::default(),
            terrains: 0,
            labels: crate::labels::Labels::new(default_lane()),
            label_epoch: 0,
        }
    }

    /// Sets how the map renders. `Still` and `Tile` frames wait for nothing
    /// to fade, so a host renders until [`FrameReport::idle`] and keeps
    /// that frame.
    ///
    /// Hit tests and rendered queries answer for the last `Continuous`
    /// frame meanwhile: a snapshot leaves the live view's as they were.
    pub fn set_mode(&mut self, mode: RenderMode) {
        if self.mode == RenderMode::Continuous && mode != RenderMode::Continuous {
            // The labels shown now, before still placement replaces them.
            self.label_hits();
        }
        self.mode = mode;
    }

    /// Sets how long labels fade in and out, milliseconds (GL JS
    /// `fadeDuration`, 300 by default; 0 for still output).
    pub fn set_fade_duration(&mut self, ms: f64) {
        self.labels.fade_duration = ms;
    }

    /// Whether labels of different sources collide with each other (GL JS
    /// `crossSourceCollisions`, on by default).
    pub fn set_cross_source_collisions(&mut self, collide: bool) {
        self.labels.cross_source_collisions = collide;
    }

    /// Draws ideographs (CJK and the like) of the default font stack with
    /// this CSS font family instead of the style's glyph PBFs, as GL JS's
    /// `localIdeographFontFamily` does; `sans-serif` by default, `None` to
    /// use the PBFs. Applies to styles loaded afterwards.
    pub fn set_local_ideograph_font_family(&mut self, family: Option<&str>) {
        self.labels.ideographs = family.map(str::to_string);
    }

    /// Places labels on `lane` instead of the host's default (a thread
    /// natively, inline on the web). Call it before loading a style.
    pub fn set_placement_lane(&mut self, lane: Arc<dyn crate::placement::PlacementLane>) {
        assert!(
            self.style.is_none(),
            "set the placement lane before a style"
        );
        // Without a style, no bucket is on the old lane.
        self.labels.lane = lane;
    }

    /// Whether terrain tiles hang skirts below their edges, hiding the
    /// cracks between tiles of different detail (GL JS
    /// `terrainSkirtLength`: on is `auto`, the default; off is `none`, for
    /// translucent terrain, where skirts would show).
    pub fn set_terrain_skirts(&mut self, skirts: bool) {
        self.terrain_skirts = skirts;
    }

    /// Enables 4× MSAA for 3D surfaces (the default). Disabling it uses
    /// single-sample rasterization, matching a GL JS map with antialias off.
    pub fn set_antialias_3d(&mut self, enabled: bool) {
        self.antialias_3d = enabled;
    }

    /// Sets how much the GPU scene holds, for renderers made after
    /// ([`Map::create_renderer`]).
    pub fn set_scene_capacity(&mut self, capacity: SceneCapacity) {
        self.capacity = capacity;
    }

    /// Sets the GPU memory the map aims to stay within (plan §8:
    /// [`DESKTOP_BUDGET`](orb_render::DESKTOP_BUDGET), the default, or
    /// [`MOBILE_BUDGET`](orb_render::MOBILE_BUDGET)). Tiles out of view stay
    /// cached within a quarter of it and give way first: the scene's pools
    /// and tile pages grow only as the tiles in view need, to the device's
    /// limits, so a view that needs more than the budget still draws.
    pub fn set_memory_budget(&mut self, bytes: u64) {
        self.budget = bytes;
    }

    /// Requests every tile that failed (the network, say) again now, as
    /// when the host learns the network is back; until then each waits out
    /// its backoff, retried while frames are drawn.
    pub fn retry_failed_tiles(&mut self) {
        let Some(style) = &mut self.style else {
            return;
        };
        for source in &mut style.sources {
            if let SourceState::Ready { pyramid, .. } = source {
                pyramid.retry_failed(f64::INFINITY);
            }
        }
        if let Some(terrain) = &mut style.terrain {
            terrain.pyramid.retry_failed(f64::INFINITY);
        }
        (self.services.wake)();
    }

    /// The last placement commit's semantic snapshot (plan §13): the
    /// labels and icons it shows, a reading order and a region summary.
    /// `None` before labels are first placed.
    pub fn semantic_snapshot(&self) -> Option<orb_a11y::Snapshot> {
        let style = self.style.as_ref()?;
        let target = self.camera.state().target;
        self.labels.semantic_snapshot(
            style.worker.plan(),
            [target.lon, target.lat],
            self.zoom(),
            &self.bucket_planes(style),
        )
    }

    /// Per label text, the greatest opacity it drew with in the last
    /// frame, now: for benches that watch labels across gestures.
    pub fn shown_labels(&self) -> HashMap<Box<str>, f32> {
        self.labels.shown(self.services.clock.now())
    }

    /// The view's style zoom (decision 0022): GL JS's in Web Mercator,
    /// measured at the plane's reference latitude on other planes.
    pub fn zoom(&self) -> f64 {
        self.camera.state().zoom(self.surface.projection())
    }

    /// The surface's projection, for camera moves. It changes when a scene
    /// document loads (with [`MapEvent::StyleLoad`]) or with
    /// [`set_projection`](Self::set_projection).
    pub fn projection(&self) -> orb_crs::Projection {
        *self.surface.projection()
    }

    /// What views draw on: the plane, or the globe the style's `projection`
    /// wraps it around. Camera stores follow it on
    /// [`MapEvent::StyleData`].
    pub fn surface(&self) -> orb_crs::Surface {
        self.surface.view_surface()
    }

    /// Draws the map on another plane, keeping the style, the sources'
    /// tile matrix sets and the camera's target and resolution. Tiles load
    /// again.
    pub fn set_projection(&mut self, projection: orb_crs::Projection) {
        let mut tx = orb_v8::Transaction::new();
        tx.push(orb_v8::Op::SetPlane(Box::new(projection)));
        // Without a style there is no scene to draw on another plane.
        let _ = self.edit(&tx);
    }

    /// The view the camera gives for a viewport.
    pub fn view(&self, viewport: Viewport) -> View {
        match self.elevation() {
            Some(terrain) => View::over(
                self.camera.state(),
                viewport,
                &self.surface.view_surface(),
                &*terrain,
            ),
            None => View::new(self.camera.state(), viewport, &self.surface.view_surface()),
        }
    }

    /// Whether a paint transition is running (decision 0031).
    pub fn transitioning(&self) -> bool {
        self.transitions.running()
    }

    /// What the map did since the last call, in order: style and source
    /// data, load, idle, errors, the camera's moves and clicks, then the
    /// warnings since.
    pub fn take_events(&mut self) -> Vec<MapEvent> {
        let warnings = self.warnings.drain(..);
        let mut events = std::mem::take(&mut self.events.queue);
        events.extend(warnings.map(|message| MapEvent::Warning { message }));
        events
    }

    /// Reports a failed load as an error event.
    pub(crate) fn fail(&mut self, message: String) {
        self.events.push(MapEvent::Error { message });
    }

    /// Sets the host's backing colour (premultiplied RGBA). Transparent by
    /// default, like GL JS's canvas. Opaque hosts can supply their window's
    /// background so transparent sky does not become black at presentation.
    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.clear_color = color;
    }

    /// What the map's jobs and uploads have done so far.
    pub fn job_counts(&self) -> JobCounts {
        JobCounts {
            state_rows: self.style.as_ref().map_or(0, |s| s.states.rows_written),
            ..self.counts
        }
    }

    /// The style document, as edits have left it.
    pub fn style_doc(&self) -> Option<&StyleDoc> {
        self.style.as_ref().map(|s| &s.doc)
    }

    /// The style generation in full view: every edit up to it has its jobs
    /// applied.
    pub fn generation(&self) -> u64 {
        match self.pending.keys().next() {
            Some(&first) => first - 1,
            None => self.version,
        }
    }

    /// Loads a v8 style or a scene document from a URL; it replaces the
    /// current one when it arrives.
    pub fn load_style_url(&mut self, url: &str) {
        self.fetch_style(url, false);
    }

    fn fetch_style(&mut self, url: &str, scene: bool) {
        self.style_request += 1;
        let request = self.style_request;
        let owned = url.to_string();
        self.fetch_text(url, move |result| Message::Style {
            request,
            url: owned,
            scene,
            result,
        });
    }

    /// Replaces the style with a v8 document on the default surface (Web
    /// Mercator), or with a scene document and its style. Its tiles load
    /// as frames render.
    pub fn set_style_json(&mut self, json: &str) -> Result<(), String> {
        self.set_document(json, None, false)
    }

    /// Installs a v8 style or a scene document; `base`, the URL it came
    /// from, resolves its relative URLs, and `scene` keeps the surface for
    /// a scene's own style.
    fn set_document(&mut self, json: &str, base: Option<&str>, scene: bool) -> Result<(), String> {
        if !scene && orb_v8::is_scene(json) {
            let parsed = orb_v8::parse_scene(json).map_err(|d| d.to_string())?;
            self.warnings
                .extend(parsed.warnings.iter().map(ToString::to_string));
            let models = parsed
                .models
                .iter()
                .map(|(id, url)| {
                    let url = match base {
                        Some(base) => orb_source::endpoint::join_url(base, url),
                        None => url.clone(),
                    };
                    (id.clone(), url)
                })
                .collect();
            let style = parsed.style.clone();
            self.pending_scene = Some(orb_v8::SceneDocument { models, ..parsed });
            return match style {
                orb_v8::StyleRef::Inline(style) => self.set_document(&style, base, true),
                orb_v8::StyleRef::Url(url) => {
                    let url = match base {
                        Some(base) => orb_source::endpoint::join_url(base, &url),
                        None => url,
                    };
                    self.fetch_style(&url, true);
                    Ok(())
                }
            };
        }
        // A fetched style's relative URLs are relative to it.
        let resolved = base.map(|base| orb_source::endpoint::resolve_style_urls(json, base));
        let text = resolved.as_deref().unwrap_or(json);
        let pending = std::mem::take(&mut self.pending_scene).filter(|_| scene);
        let doc = match &pending {
            Some(scene) => StyleDoc::parse_extended(text, scene),
            None => StyleDoc::parse(text),
        }
        .map_err(|d| d.to_string())?;
        // A style URL still loading is superseded, and the old style's
        // paint transitions with it.
        self.style_request += 1;
        self.transitions = Default::default();
        self.install(doc);
        Ok(())
    }

    /// A renderer for this map's frames into targets of `format`, on the
    /// images its runtime shares.
    pub fn create_renderer(&self, gpu: &mut dyn Gpu, format: TargetFormat) -> Renderer {
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let images = images
            .atlas
            .get_or_insert_with(|| ImageAtlas::new(gpu, 1024, 8, 8192));
        Renderer::with_images(gpu, format, self.capacity, images.bindings())
    }

    /// Forgets the GPU state of another renderer: after a device loss, or
    /// to draw on another device (plan §8.8). This frame lays the style's
    /// tiles out again, from the jobs' cached tile frames where they have
    /// them (refetched where not); labels place afresh, and sprites and
    /// glyphs load again.
    fn forget_gpu(&mut self) {
        self.globe_warps = Default::default();
        self.retired_pages.clear();
        self.retired_images.clear();
        self.retired_ramps.clear();
        let old = &self.labels;
        let mut labels = crate::labels::Labels::new(Arc::clone(&old.lane));
        labels.fade_duration = old.fade_duration;
        labels.cross_source_collisions = old.cross_source_collisions;
        labels.ideographs = old.ideographs.clone();
        self.labels = labels;
        self.label_epoch += 1;
        *self.picking.get_mut() = Default::default();
        if let Some(style) = self.style.take() {
            self.generation += 1;
            self.install_revision(style.doc, Some((style.id, &style.worker)));
        }
        // What the old renderer held went with it.
        self.retired = Retired::default();
    }

    /// Replaces the style with a document, every tile loaded fresh.
    pub(crate) fn install(&mut self, doc: StyleDoc) {
        self.install_revision(doc, None);
    }

    /// Installs a document: a new style, or with `previous` the same
    /// style again as a new revision of its jobs' style, which keeps their
    /// cached tile frames.
    fn install_revision(&mut self, doc: StyleDoc, previous: Option<(u64, &Arc<TileWorker>)>) {
        self.version += 1;
        self.labels.invalidate();
        self.surface = Surface::of(doc.style());
        let worker = Arc::new(revise(
            previous.map(|(_, worker)| &**worker),
            &doc,
            &[],
            self.version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        ));
        let plan = worker.plan();
        self.warnings
            .extend(plan.style.warnings.iter().map(ToString::to_string));
        self.warnings.extend(plan.warnings.iter().cloned());
        // Until the app or the user moves the camera, it takes the style's.
        self.camera.set_surface(self.surface.view_surface());
        if previous.is_none() && !self.camera.modified() {
            let style = &plan.style;
            self.camera.jump_to(&orb_camera::CameraTarget {
                center: style
                    .center
                    .map(|[lon, lat]| orb_crs::Geodetic::new(lon, lat)),
                zoom: style.zoom,
                bearing: Some(style.bearing),
                pitch: Some(style.pitch),
                roll: Some(style.roll),
                ..Default::default()
            });
        }
        self.events.wake();
        if previous.is_none() {
            self.events.push(MapEvent::StyleLoad);
        }
        self.events.push(MapEvent::StyleData {
            generation: self.version,
        });
        self.clear_style();
        self.pending.clear();
        match previous {
            Some((id, _)) => self.style_id = id,
            None => self.style_id += 1,
        }
        self.services.jobs.set_style(
            self.style_id,
            &full_update(
                &doc,
                self.version,
                self.labels.ideographs.clone(),
                self.surface.tiles(),
            ),
            &worker,
        );

        let sources = (0..plan.sources.len())
            .map(|index| self.start_source(plan, index))
            .collect();
        self.style = Some(LoadedStyle {
            nodes: plan
                .nodes
                .iter()
                .map(|n| {
                    let state = NodeState {
                        maxima: vec![0.0; plan.paint_properties(n).len()],
                        ..NodeState::default()
                    };
                    (n.key, state)
                })
                .collect(),
            drawn: Vec::new(),
            placed: vec![Vec::new(); plan.sources.len()],
            held: vec![Vec::new(); plan.sources.len()],
            id: self.style_id,
            doc,
            worker: Arc::clone(&worker),
            sources,
            retired: Vec::new(),
            painted: None,
            drape: DrapePaint::default(),
            painted_fade: None,
            zoom_history: ZoomHistory::default(),
            image_namespace: None,
            sprites_started: false,
            sprites_loading: 0,
            icons: crate::SpriteIcons::default(),
            image_edits: Vec::new(),
            reported_missing: Default::default(),
            background: None,
            states: FeatureStates::default(),
            symbol_tiles: Vec::new(),
            draws: Vec::new(),
            terrain: None,
            terrain_unusable: false,
            globe_meshes: self.surface.globe().is_some(),
        });
        self.nodes_stale = true;
    }

    /// Starts resolving a source of the plan: its TileJSON or archive
    /// header, or a ready endpoint.
    pub(crate) fn start_source(&mut self, plan: &StylePlan, index: usize) -> SourceState {
        let key = plan.sources[index].key;
        let source = plan.source_def(index);
        let tileset = match &source.kind {
            SourceKind::Vector(v) => Some(&v.tileset),
            SourceKind::Raster(r) => Some(&r.tileset),
            SourceKind::RasterDem(d) => Some(&d.raster.tileset),
            _ => None,
        };
        match (&source.kind, tileset) {
            (_, Some(tileset)) => match &tileset.url {
                Some(url) => {
                    let generation = self.generation;
                    if let Some(archive) = url.strip_prefix("pmtiles://") {
                        let (sender, wake) = (self.sender.clone(), Arc::clone(&self.services.wake));
                        self.services.fetch.get_range(
                            archive,
                            0,
                            orb_source::pmtiles::HEADER_LEN,
                            Box::new(move |result| {
                                let result = result.and_then(|bytes| {
                                    orb_source::endpoint::archive_tilejson(&bytes)
                                        .map_err(FetchError::Range)
                                });
                                let _ = sender.send(Message::TileJson {
                                    generation,
                                    source: key,
                                    result,
                                });
                                wake();
                            }),
                        );
                    } else {
                        self.fetch_text(url, move |result| Message::TileJson {
                            generation,
                            source: key,
                            result,
                        });
                    }
                    SourceState::Resolving
                }
                None => ready_source(
                    self.planned(&source.id, false, plan.image_kind(index))
                        .and_then(|(tms, warp)| {
                            Ok((TileEndpoint::resolve(&source.kind, None, tms)?, warp))
                        }),
                    &source.id,
                    plan.image_kind(index).is_some(),
                    &mut self.events,
                ),
            },
            (SourceKind::Image(image), _) => ready_source(
                Ok((
                    TileEndpoint::image(image, self.surface.projection()),
                    Plan::Affine,
                )),
                &source.id,
                true,
                &mut self.events,
            ),
            (SourceKind::GeoJson(geojson), _) => ready_source(
                self.planned(&source.id, true, None)
                    .map(|(tms, warp)| (TileEndpoint::geojson(geojson, tms), warp)),
                &source.id,
                false,
                &mut self.events,
            ),
            _ => unreachable!("plan sources are drawable"),
        }
    }

    /// The tile matrix sets of this frame's symbol buckets.
    pub(crate) fn bucket_planes(&self, style: &LoadedStyle) -> crate::labels::BucketPlanes {
        crate::labels::BucketPlanes {
            to_world: style
                .symbol_tiles
                .iter()
                .map(|t| (t.bucket, t.to_world))
                .collect(),
            projection: *self.surface.projection(),
            period: self.surface.period(),
        }
    }

    /// A source's tile matrix set, if the planner can bring it to the
    /// surface.
    fn planned(
        &self,
        id: &str,
        geojson: bool,
        image: Option<crate::style::ImageKind>,
    ) -> Result<(TileMatrixSet, Plan), String> {
        let tms = self.surface.source_tms(id, geojson);
        self.surface.plan(&tms, image).map(|plan| (tms, plan))
    }

    /// Draws the frame shown at `frame_time` (the host's frame timestamp,
    /// on the clock its input times use, in milliseconds) into `target`
    /// with `renderer`: applies arrived data, takes the style's camera if
    /// nothing moved it, advances gestures and animations to it, loads what
    /// the view needs and renders. A renderer the map hasn't drawn with
    /// before (one on a new device, say) gets the map's tiles laid out
    /// again; the map holds tiles of one renderer at a time.
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        target: TextureId,
        viewport: Viewport,
        pixel_ratio: f32,
        frame_time: f64,
    ) -> FrameReport {
        if self.renderer != Some(renderer.id()) {
            if self.renderer.is_some() {
                self.forget_gpu();
            }
            self.renderer = Some(renderer.id());
            self.nodes_stale = true;
            self.events.push(MapEvent::Capability {
                tier: gpu.caps().tier,
            });
        }
        self.retired.release(renderer);
        renderer.set_antialias_3d(gpu, self.antialias_3d);
        self.start_sprites(gpu, pixel_ratio);
        self.apply_image_edits(gpu);
        // Bring the scene's nodes to the plan before results for it apply.
        self.sync_nodes(gpu, renderer);
        self.placement_ms = 0.0;
        self.apply_messages(gpu, renderer);
        // A URL style may have arrived while applying messages.
        self.start_sprites(gpu, pixel_ratio);
        self.sync_nodes(gpu, renderer);
        self.apply_videos(gpu, renderer);
        // The camera moves for this frame on what has arrived.
        let moving = self.advance_camera(viewport, frame_time);

        // Tiles' patches are on the low-zoom surface the style draws on.
        let plane = self.surface.globe().and_then(|g| g.plane);
        self.globe_warps.set_plane(renderer, plane);
        let view = self.view(viewport);
        // What this frame draws, for picking.
        let mut drawn = Vec::new();
        let zoom = self.zoom();
        let mut report = FrameReport::default();
        let clear = self.clear_color;
        // The atmosphere: the sky above the surface's horizon (a globe's
        // limb), fog on terrain, scattering over a globe.
        let (fog_near, fog_far) = view.fog_range_px();
        let (horizon, horizon_normal) = view.sky_horizon_line();
        let scene = self
            .style
            .as_ref()
            .map(|s| Arc::clone(&s.worker.plan().style));
        let lights = scene.as_deref().map(|s| &s.lights);
        // The sky over a sphere; a low-zoom plane's is a plane's.
        let globe = view
            .globe()
            .filter(|g| g.plane.is_none())
            .map(|g| crate::lights::globe_sky(g, &view, self.camera.state(), lights, zoom));
        renderer.set_atmosphere(
            gpu,
            scene
                .as_deref()
                .and_then(|s| s.atmosphere.as_ref())
                .map(|atmosphere| orb_render::AtmospherePaint {
                    sky_color: atmosphere.sky_color.evaluate(zoom).premultiplied(),
                    horizon_color: atmosphere.horizon_color.evaluate(zoom).premultiplied(),
                    fog_color: atmosphere.fog_color.evaluate(zoom).premultiplied(),
                    sky_horizon_blend: atmosphere.sky_horizon_blend.evaluate(zoom),
                    horizon_fog_blend: atmosphere.horizon_fog_blend.evaluate(zoom),
                    fog_ground_blend: atmosphere.fog_ground_blend.evaluate(zoom),
                    atmosphere_blend: atmosphere.atmosphere_blend.evaluate(zoom),
                    backdrop: atmosphere.backdrop.evaluate(zoom).premultiplied(),
                    stars: atmosphere.stars.evaluate(zoom),
                }),
            orb_render::AtmosphereView {
                horizon: horizon.as_vec2().to_array(),
                horizon_normal: horizon_normal.as_vec2().to_array(),
                height: viewport.height as f32,
                fog_range: [fog_near as f32, fog_far as f32],
                fog_opacity: fog_blend_opacity(self.camera.state().pitch),
                ray_inverse: crate::lights::ray_inverse(&view, self.camera.state().target),
                star_rotation: crate::lights::star_rotation(lights),
                globe,
            },
        );

        let now = self.services.clock.now();
        let mut fading = false;
        if let Some(style) = &mut self.style {
            let plan = style.worker.plan();
            if style.states.prepare(gpu, renderer, plan, zoom) {
                style.painted = None;
            }
            let crossfade = style.zoom_history.update(zoom, now);
            fading = crossfade.mix < 1.0
                && plan.nodes.iter().any(|n| {
                    plan.pattern(n).is_some() || {
                        let l = &plan.style.layers[n.layer];
                        l.ty == LayerType::Line
                            && l.get(orb_style::LINE_DASHARRAY).constant() != Some(None)
                    }
                });
            // Paint transitions write from the style overlaid with their
            // eased values, and once more when they end.
            let worker = Arc::clone(&style.worker);
            let overlay = self.transitions.overlay(worker.plan(), zoom, now);
            if overlay.is_some() || std::mem::take(&mut self.transitions.ran) {
                style.painted = None;
            }
            let mut images = self.services.images.0.lock().expect("images poisoned");
            let missing = write_paint(
                gpu,
                renderer,
                (style, overlay.as_ref().unwrap_or(worker.plan())),
                (zoom, self.camera.state().bearing),
                crossfade,
                images.atlas.as_mut().expect("initialized"),
            );
            drop(images);
            for name in missing {
                self.events.push(MapEvent::StyleImageMissing { name });
            }
            style.symbol_tiles.clear();
            self.crossfade = crossfade;
        }
        if self.style.is_some() {
            // Terrain's DEM and cover first: the sources cover over its
            // elevations, and drape over its tiles.
            self.update_terrain(gpu, renderer, &view, &mut report);
            let elevation = self.terrain_elevation();
            for index in 0..self.style.as_ref().map_or(0, |s| s.sources.len()) {
                self.update_source(renderer, &view, index, elevation.as_deref(), &mut report);
            }
            let drape = self.plan_drape(gpu, renderer, &view);
            self.place_background(
                gpu,
                renderer,
                &view,
                elevation.as_deref(),
                drape.as_ref(),
                &mut report,
            );
            for index in 0..self.style.as_ref().map_or(0, |s| s.sources.len()) {
                self.place_source(
                    gpu,
                    renderer,
                    &view,
                    index,
                    drape.as_ref(),
                    &mut report,
                    &mut drawn,
                );
            }
            self.trim_cache(renderer);
        }
        let labels_settled = self.place_labels(gpu, renderer, &view, now);
        report.placement_us = (self.placement_ms * 1000.0).round() as u32;

        if !self.retired_ramps.is_empty() {
            let mut images = self.services.images.0.lock().expect("images poisoned");
            images
                .atlas
                .as_mut()
                .unwrap()
                .release_ramps(gpu, self.retired_ramps.drain(..));
        }
        report.generation = self.generation();
        let picking = self.picking.get_mut();
        if self.mode == RenderMode::Continuous {
            if picking.layers_version != Some(self.version)
                && let Some(style) = &self.style
            {
                picking.layers = Arc::new(crate::query::HitLayers::of(style.worker.plan()));
                picking.layers_version = Some(self.version);
            }
            picking.drew(view.clone(), *self.camera.state(), drawn, report.generation);
        }
        report.loaded = self.style.as_ref().is_some_and(|s| {
            s.sprites_loading == 0
                && s.sources
                    .iter()
                    .all(|source| !matches!(source, SourceState::Resolving))
        }) && report.tiles_loading == 0
            && !self.tiles_deferred
            && self.waiting.is_empty()
            && self.pending.is_empty()
            && labels_settled;
        report.idle = report.loaded
            && !fading
            && !self.labels.fading(now)
            && !self.transitions.running()
            && now >= self.image_fades_until;
        if let Some(style) = &self.style {
            let plan = style.worker.plan();
            for index in std::mem::take(&mut self.events.sources) {
                let loaded = match &style.sources[index] {
                    SourceState::Ready { loading, .. } => *loading == 0,
                    SourceState::Resolving => false,
                    _ => true,
                };
                let source = plan.source_def(index).id.clone();
                self.events.push(MapEvent::SourceData { source, loaded });
            }
        }

        let globe = view.globe().map(|globe| frame_globe(globe, &view));
        let frame = FrameBlock {
            camera: view.camera_world().to_array(),
            world_span: self
                .surface
                .projection()
                .world_bounds()
                .size()
                .max_element(),
            view_proj: view.view_proj().as_mat4().to_cols_array_2d(),
            viewport: [viewport.width as f32, viewport.height as f32],
            world_per_px: view.world_per_px() as f32,
            pixel_ratio,
            zoom,
            bearing: self.camera.state().bearing.to_radians() as f32,
            extrusion_scale: self.extrusion_scale() as f32,
            ground_horizon: view.ground_horizon() as f32,
            symbol_fade_change: self.labels.fade_change(now),
            time: (now - self.born) as f32,
            pitch: self.camera.state().pitch.to_radians() as f32,
            roll: self.camera.state().roll.to_radians() as f32,
            align: match globe {
                Some(_) => [0.0; 2],
                None => pixel_alignment(
                    &view,
                    self.surface.projection(),
                    self.camera.state().bearing,
                ),
            },
            terrain: self.frame_terrain,
            lights: lights.map(|lights| {
                crate::lights::frame_lights(
                    lights,
                    zoom,
                    self.camera.state().bearing,
                    self.camera.state().target,
                )
            }),
            globe,
        };
        report.frame = renderer.render(gpu, &frame, target, clear);
        // A frame that left something out for a compiling pipeline isn't
        // complete (plan §8.4).
        report.compiling = report.idle && report.frame.waited;
        report.loaded &= !report.frame.waited;
        report.idle &= !report.frame.waited;
        // Idle once the camera rests too, and once per rest.
        let held = self.camera.is_held() || self.input.is_active();
        self.events
            .frame(report.loaded, report.idle && !moving && !held);
        report.idle &= !moving;
        self.globe_warps.end_frame();
        report.scene = renderer.stats();
        report.memory = self.memory(renderer);
        // Chunks the cull has no room for don't draw: say so when it starts.
        let dropped = report.scene.candidates_dropped;
        if dropped > 0 && self.candidates_dropped == 0 {
            let message = format!(
                "{dropped} chunks are not drawn: the scene's candidate or compacted-index \
                 memory is full"
            );
            self.events.push(MapEvent::Error { message });
        }
        self.candidates_dropped = dropped;
        report
    }

    /// Makes the renderer's nodes the plan's: removes the meshes of nodes
    /// edits removed and the tiles of removed sources, keeps the others'
    /// meshes in their new places, and gives new image nodes their quads.
    fn sync_nodes(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        let Some(style) = &mut self.style else {
            return;
        };
        for source in style.retired.drain(..) {
            if let SourceState::Ready { mut pyramid, .. } = source {
                for tile in pyramid.drain() {
                    self.retired_ramps
                        .extend(remove_resident(renderer, tile, &mut style.states));
                }
            }
        }
        let plan = style.worker.plan();
        // A projection edit put the map on or off a globe: the background
        // and image tiles take the meshes it bends.
        let globe = self.surface.globe().is_some();
        if style.globe_meshes != globe {
            style.globe_meshes = globe;
            if let Some(tile) = style.background.take() {
                renderer.release_tile(tile);
                style.background = make_background(gpu, renderer, plan, globe);
            }
            for source in &mut style.sources {
                let SourceState::Ready {
                    pyramid,
                    endpoint,
                    warp,
                    ..
                } = source
                else {
                    continue;
                };
                for (tile, resident) in pyramid.ready_mut() {
                    let handle = resident.gpu;
                    let Some(image) = resident.image.as_mut() else {
                        continue;
                    };
                    let Ok(grid) = image_mesh(
                        &mut self.globe_warps,
                        globe,
                        warp.as_deref_mut(),
                        endpoint,
                        tile,
                    ) else {
                        continue;
                    };
                    if image.warped == grid {
                        continue;
                    }
                    image.warped = grid;
                    // A warped image draws through planar patches only on a
                    // globe surface; elsewhere through its warp's mesh.
                    let series = warp
                        .as_deref_mut()
                        .filter(|_| globe)
                        .and_then(|w| w.series(tile));
                    let patches = series.as_deref().map(series_patches);
                    let mesh = image.warped.as_ref().map(|w| Arc::clone(&w.mesh));
                    resident.series = series;
                    let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
                    if renderer.set_image_mesh(gpu, handle, mesh, patches).is_err() {
                        self.warnings
                            .push("out of scene memory; an image layer misses tiles".into());
                    }
                }
            }
        }
        let keys: Vec<u32> = plan.nodes.iter().map(|n| n.key).collect();
        let fresh = std::mem::take(&mut self.nodes_stale);
        if !fresh && keys == style.drawn {
            return;
        }
        if let Some(tile) = style.background.take() {
            renderer.release_tile(tile);
        }
        for source in &mut style.sources {
            let SourceState::Ready { pyramid, .. } = source else {
                continue;
            };
            for (_, resident) in pyramid.ready_mut() {
                resident.symbols.retain(|s| keys.contains(&(s.layer * 2)));
                // The renderer drops the meshes of nodes that leave.
                resident.meshes.retain_mut(|mesh| {
                    if keys.contains(&mesh.node) {
                        return true;
                    }
                    for &index in &mesh.states {
                        style.states.release(index);
                    }
                    self.retired_ramps.append(&mut mesh.ramps);
                    false
                });
            }
        }
        let retained: Vec<Option<u32>> = keys
            .iter()
            .map(|k| {
                (!fresh)
                    .then(|| style.drawn.iter().position(|d| d == k))
                    .flatten()
                    .map(|i| i as u32)
            })
            .collect();
        let nodes: Vec<RenderNode> = plan
            .nodes
            .iter()
            .map(|n| RenderNode {
                family: n.family,
                source: n.family.is_image().then(|| plan.sources[n.source].key),
            })
            .collect();
        if !gpu.caps().rgba16float_blendable
            && plan.nodes.iter().any(|n| n.family == Family::Heatmap)
        {
            self.warnings.push(crate::COMPAT_NO_HEATMAP.into());
        }
        // New nodes of image sources draw the tiles already resident.
        if !renderer.remap_nodes(gpu, &nodes, &retained) {
            self.warnings
                .push("out of scene memory; an image layer misses tiles".into());
        }
        if !self.retired_pages.is_empty() {
            // The terrain may bind a retired terrain page; it turns on
            // again next.
            renderer.set_terrain(gpu, None);
        }
        for page in self.retired_pages.drain(..) {
            renderer.remove_page(gpu, page);
        }
        style.background = make_background(gpu, renderer, plan, style.globe_meshes);
        style.drawn = keys;
        style.painted = None;
        for placed in &mut style.placed {
            placed.clear();
        }
    }

    fn start_sprites(&mut self, gpu: &mut dyn Gpu, pixel_ratio: f32) {
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let retired = std::mem::take(&mut images.retired);
        let retired_ramps = std::mem::take(&mut images.retired_ramps);
        let images = images.atlas.as_mut().expect("renderer initializes images");
        images.release_ramps(
            gpu,
            retired_ramps
                .into_iter()
                .chain(self.retired_ramps.drain(..)),
        );
        for namespace in retired.into_iter().chain(self.retired_images.drain(..)) {
            images.release(gpu, namespace);
        }
        let Some(style) = &mut self.style else { return };
        if style.sprites_started {
            return;
        }
        style.image_namespace = Some(images.namespace());
        style.sprites_started = true;
        let sprites = &style.worker.plan().style.sprites;
        style.sprites_loading = sprites.len() as u32;
        style.icons.pixel_ratio = f64::from(pixel_ratio);
        for sprite in sprites {
            let (sender, wake, generation) = (
                self.sender.clone(),
                self.services.wake.clone(),
                self.generation,
            );
            crate::sprite::load(
                self.services.fetch.as_ref(),
                self.services.image_decode.clone(),
                sprite,
                pixel_ratio,
                Box::new(move |result| {
                    let _ = sender.send(Message::Sprite { generation, result });
                    wake();
                }),
            );
        }
    }

    // Pinned GL JS v8 heights use the IUGG mean sphere (6371008.8 m),
    // whereas projected XY uses the WGS84 semi-major axis. Retain v8's
    // vertical scale without changing the camera's true ground resolution.
    fn extrusion_scale(&self) -> f64 {
        self.surface
            .projection()
            .style_height_scale(self.camera.state().target)
    }

    /// Sends a tile job for source `index` of the current plan.
    pub(crate) fn send_job(
        &mut self,
        index: usize,
        tile: TileIndex,
        nodes: Option<Vec<u32>>,
        rows_only: bool,
        edit: Option<u64>,
    ) {
        let style = self.style.as_mut().expect("jobs need a style");
        // A video's tile waits for a frame from the host.
        let key = style.worker.plan().sources[index].key;
        if edit.is_none() && self.videos.want(key, tile) {
            return;
        }
        let vector = style.worker.plan().image_kind(index).is_none();
        let SourceState::Ready { endpoint, warp, .. } = &mut style.sources[index] else {
            return;
        };
        let mut bend = warp
            .as_mut()
            .filter(|_| vector)
            .and_then(|warp| warp.series(tile))
            .map(|series| bend_of(series.grid(), [false; 2]));
        let footprint = warp.as_mut().and_then(|warp| warp.footprint(tile));
        let zoom = self.surface.tile_zoom(&endpoint.tms, tile, footprint);
        // On a low-zoom surface, geometry bends with it, and on the sphere
        // the world's edge rows reach the poles: in tiles it can show, those
        // laid out below the zoom it hands over by.
        if let Some(globe) = self.surface.globe()
            && vector
            && globe.handover.is_none_or(|[_, end]| zoom < end)
        {
            let grid = self.globe_warps.grid(&endpoint.tms, tile);
            let poles = match globe.plane {
                Some(_) => [false; 2],
                None => orb_crs::sphere::poles(&endpoint.tms, tile),
            };
            if grid > 1 || poles.contains(&true) {
                let cells = bend.map_or(grid, |b| b.cells.max(grid));
                bend = Some(bend_of(cells, poles));
            }
        }
        let key = style.worker.plan().sources[index].key;
        let request = TileRequest {
            style: style.id,
            source: key,
            tile,
            zoom,
            bend,
            data_zoom: tile.z.min(endpoint.maxzoom),
            location: endpoint.location(endpoint.data_tile(tile)),
            encoding: endpoint.encoding,
            nodes,
            rows_only,
            dem_unpack: None,
        };
        match (edit, rows_only) {
            (None, _) => self.counts.loads += 1,
            (Some(_), false) => self.counts.relayouts += 1,
            (Some(_), true) => self.counts.columns += 1,
        }
        if let Some(edit) = edit {
            *self.pending.entry(edit).or_default() += 1;
        }
        self.load(request, edit, false);
    }

    /// Sends a request to the jobs; its result comes back as a message.
    fn load(&mut self, request: TileRequest, edit: Option<u64>, terrain: bool) {
        let (sender, wake, generation) = (
            self.sender.clone(),
            Arc::clone(&self.services.wake),
            self.tile_generation,
        );
        let (source, tile) = (request.source, request.tile);
        self.services.jobs.load(
            request,
            Box::new(move |result| {
                let _ = sender.send(Message::Tile {
                    generation,
                    source,
                    tile,
                    edit,
                    terrain,
                    result,
                });
                wake();
            }),
        );
    }

    /// Retires dropped buckets, rasterizes new glyphs and asks the lane to
    /// place when the drawn buckets or the camera changed. Returns whether
    /// labels have settled.
    fn place_labels(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        now: f64,
    ) -> bool {
        let extrusion_scale = self.extrusion_scale();
        let Some(style) = &self.style else {
            return true;
        };
        let (sender, wake) = (self.sender.clone(), Arc::clone(&self.services.wake));
        let epoch = self.label_epoch;
        let raster_done = move |rows: Vec<crate::labels::RasterRow>| -> crate::RasterDone {
            let (sender, wake) = (sender.clone(), Arc::clone(&wake));
            Box::new(move |result| {
                let _ = sender.send(Message::Raster {
                    epoch,
                    rows,
                    result,
                });
                wake();
            })
        };
        self.labels.maintain(
            gpu,
            renderer.glyphs(),
            self.services.jobs.as_ref(),
            &raster_done,
            &style.worker.plan().style.font_faces,
        );
        // Glyph rasterization above is the jobs'; placement starts here.
        let started = self.services.clock.now();
        let mut frame = self.labels.frame(
            style.worker.plan(),
            &style.symbol_tiles,
            view,
            view.camera_world(),
            self.surface.period(),
            self.camera.state(),
            self.camera.state().zoom(self.surface.projection()),
            extrusion_scale,
            now,
        );
        if self.mode != RenderMode::Continuous {
            frame.fade_duration = 0.0;
        }
        frame.scope = match &self.mode {
            RenderMode::Continuous => orb_symbol::Scope::Viewport,
            RenderMode::Still => orb_symbol::Scope::Still,
            RenderMode::Tile(tiles) => orb_symbol::Scope::Tiles(
                tiles
                    .iter()
                    .map(|t| {
                        let view = View::new(&t.camera, t.viewport, &self.surface.view_surface());
                        let f = self.labels.frame(
                            style.worker.plan(),
                            &style.symbol_tiles,
                            &view,
                            t.center,
                            self.surface.period(),
                            &t.camera,
                            t.camera.zoom(self.surface.projection()),
                            extrusion_scale,
                            now,
                        );
                        orb_symbol::ScopeTile {
                            col: t.col,
                            row: t.row,
                            camera: f.camera,
                            layers: f.layers,
                        }
                    })
                    .collect(),
            ),
        };
        self.labels.inherit(gpu, renderer, &frame);
        // Symbols stand on the terrain as drawn.
        frame.ground = style
            .terrain
            .as_ref()
            .map(|t| orb_symbol::GroundRef(t.snapshot() as Arc<dyn orb_symbol::Ground>));
        // Without symbol tiles there is nothing to place.
        if frame.layers.iter().all(|l| l.tiles.is_empty()) {
            self.placement_ms += self.services.clock.now() - started;
            return !self.labels.settling();
        }
        let settled = self.labels.is_placed(&frame);
        if self.labels.wants_placement(&frame) {
            self.labels.placing(&frame);
            let (sender, wake) = (self.sender.clone(), Arc::clone(&self.services.wake));
            let placing = self.services.clock.now();
            self.labels.lane.place(
                frame,
                Box::new(move |committed| {
                    let _ = sender.send(Message::Placed { epoch, committed });
                    wake();
                }),
            );
            self.placement_ms -= self.services.clock.now() - placing;
        }
        self.placement_ms += self.services.clock.now() - started;
        settled && !self.labels.settling()
    }

    /// Covers the view with one source's tiles: requests missing ones,
    /// places the drawable ones and evicts the rest.
    /// Covers the view for source `index` over the terrain's
    /// `elevation`, requests its missing tiles and keeps what to draw for
    /// [`place_source`](Self::place_source).
    fn update_source(
        &mut self,
        renderer: &Renderer,
        view: &View,
        index: usize,
        elevation: Option<&crate::terrain::TerrainElevation>,
        report: &mut FrameReport,
    ) {
        let elevation_scale = self.extrusion_scale();
        let style = self.style.as_mut().expect("called with a style");
        style.draws.resize_with(style.sources.len(), || None);
        style.draws[index] = None;
        if !style.worker.plan().nodes.iter().any(|n| n.source == index) {
            // A source only the terrain reads.
            return;
        }
        let zoom = self.camera.state().zoom(self.surface.projection());
        let plan = style.worker.plan();
        // Symbol layout needs the sprites' metadata, so a source with symbol
        // layers waits for them (GL JS's worker awaits `getImages` alike).
        if style.sprites_loading > 0
            && plan
                .nodes
                .iter()
                .any(|n| n.source == index && n.family == Family::SymbolIcon)
        {
            return;
        }
        let height = plan
            .nodes
            .iter()
            .filter(|node| node.source == index && node.family == Family::FillExtrusion)
            .map(|node| {
                let e = &plan.style.layers[node.layer];
                let maxima = style.nodes.get(&node.key).map_or(&[][..], |n| &n.maxima);
                let max = |i: usize| maxima.get(i).copied().unwrap_or(0.0);
                e.get(orb_style::EXTRUSION_HEIGHT)
                    .evaluate(zoom)
                    .max(e.get(orb_style::EXTRUSION_BASE).evaluate(zoom))
                    .max(max(1))
                    .max(max(2))
            })
            .fold(0.0f32, f32::max);
        // Circles and heatmaps draw from the tile their point is in alone,
        // where GL JS draws the copies in its neighbours' buffers too: a
        // neighbour out of view still reaches into it by their reach.
        // Labels, as in GL JS, show from their anchor's tile only.
        let reach = plan
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.source == index && matches!(n.family, Family::Circle | Family::Heatmap)
            })
            .map(|(i, _)| renderer.cull_margin_px(i as u32))
            .fold(0.0f32, f32::max);
        let SourceState::Ready {
            endpoint,
            pyramid,
            loading,
            warp,
            ..
        } = &mut style.sources[index]
        else {
            return;
        };
        let endpoint = Arc::clone(endpoint);
        let tms = &endpoint.tms;
        pyramid.retry_failed(self.services.clock.now());
        // Where the source's visible layers are shown, their tiles hold
        // their data: the cover keeps to the least zoom all of them have
        // data at, wherever that detail still shows (orb_camera's
        // `DATA_TILE_PX`), and picks finer by distance above it.
        let data_zoom = plan
            .nodes
            .iter()
            .filter(|n| n.source == index)
            .map(|n| &plan.style.layers[n.layer])
            .filter(|l| l.is_visible_at(zoom))
            .map(|l| l.minzoom.floor().max(0.0) as u8)
            .max()
            .unwrap_or(0)
            .clamp(pyramid.min_zoom(), pyramid.max_zoom());
        let cover = match (view.low_plane(), warp.as_mut()) {
            // On a low-zoom plane every set's tiles land through its warp.
            (Some(low), _) => {
                low.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                    self.globe_warps.footprint(tms, tile)
                })
            }
            (None, None) => view.cover_volume(
                tms,
                endpoint.cover_px,
                data_zoom..=pyramid.max_zoom(),
                f64::from(height) * elevation_scale,
                elevation.map(|e| e as &dyn orb_camera::Elevation),
                f64::from(reach),
            ),
            (None, Some(warp)) => {
                view.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                    warp.footprint(tile)
                })
            }
        };
        let update = pyramid.update(&cover);
        let mut requests = Vec::new();
        let vector = plan.image_kind(index).is_none();
        for &tile in &update.request {
            // Vector tiles on another plane need a series to draw through:
            // planar patches, or the sphere's on a globe.
            let unwarped = vector
                && warp.as_mut().is_some_and(|w| w.series(tile).is_none())
                && (self.surface.globe().is_none() || self.globe_warps.series(tms, tile).is_none());
            // Outside the source's bounds there is no data, as in GL JS:
            // nothing draws there, not even an ancestor.
            if !endpoint.contains(tile) || unwarped {
                pyramid.insert_empty(tile);
                continue;
            }
            *loading += 1;
            requests.push(tile);
        }
        report.tiles_loading += *loading;
        style.draws[index] = Some(update.draw);
        for tile in requests {
            self.send_job(index, tile, None, false, None);
        }
    }

    /// Places source `index`'s tiles for this frame: map placements
    /// standing on the terrain, and drape placements into the drape slots
    /// being redrawn.
    #[allow(clippy::too_many_arguments)]
    fn place_source(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        index: usize,
        drape: Option<&DrapePlan>,
        report: &mut FrameReport,
        drawn: &mut Vec<crate::query::DrawnTile>,
    ) {
        let style = self.style.as_mut().expect("called with a style");
        let Some(draws) = style.draws[index].take() else {
            return;
        };
        let draped = drape.filter(|_| {
            let plan = style.worker.plan();
            plan.nodes
                .iter()
                .any(|n| n.source == index && n.family.is_draped())
        });
        let terrain = style.terrain.as_ref();
        let source_key = style.worker.plan().sources[index].key;
        let SourceState::Ready {
            pyramid,
            endpoint,
            warp: source_warp,
            ..
        } = &mut style.sources[index]
        else {
            return;
        };
        let endpoint = Arc::clone(endpoint);
        let tms = &endpoint.tms;
        let mut rows: HashMap<TileIndex, Vec<&Draw>> = HashMap::new();
        // Picking reads tiles with features in drawing order, ancestors
        // before descendants.
        let mut order = Vec::new();
        for draw in &draws {
            let copies = rows.entry(draw.tile).or_default();
            if copies.is_empty() {
                order.push(draw.tile);
            }
            copies.push(draw);
        }
        for tile in order {
            let resident = pyramid.get(tile).expect("drawn tiles are ready");
            if resident.meshes.iter().any(|m| !m.hits.is_empty()) {
                drawn.push(crate::query::DrawnTile::new(
                    source_key,
                    tile,
                    &rows[&tile],
                    tms,
                    resident.series.as_deref(),
                ));
            }
        }
        let period = self.surface.period();
        let placed = &mut style.placed[index];
        // Labels past a Web Mercator tile go on exactly on a globe.
        let on_mercator = view.globe().is_some() && *tms.projection() == mercator_projection();
        for (&tile, draws) in &rows {
            let resident = pyramid.get(tile).expect("drawn tiles are ready");
            let globe = view
                .globe()
                .and_then(|_| self.globe_warps.series(tms, tile));
            // Each world copy places its labels, as GL JS's wrapped tiles do.
            for symbols in &resident.symbols {
                let mut wraps: Vec<i32> = draws.iter().map(|d| d.wrap).collect();
                wraps.sort_unstable();
                wraps.dedup();
                for wrap in wraps {
                    let to_world = match &resident.series {
                        Some(series) => series.linear(orb_symbol::EXTENT),
                        None => tms.data_to_world(tile, orb_symbol::EXTENT as u32),
                    };
                    style.symbol_tiles.push(crate::labels::SymbolTile {
                        to_world,
                        layer: symbols.layer,
                        bucket: symbols.bucket,
                        tile,
                        wrap,
                        mercator: on_mercator && resident.series.is_none(),
                        globe: globe.clone(),
                    });
                }
            }
            let frame = resident
                .image
                .as_ref()
                .and_then(|image| image.warped.as_ref())
                .map(|warped| warped.frame);
            let ground = terrain.map(|t| t.ground(tile));
            let series = resident.series.clone();
            // A tile on another plane has a place on the surface's plane
            // through its planar patches; without them (one at a pole, say)
            // it draws on the globe alone.
            let on_plane = source_warp.is_none() || series.is_some();
            // On a globe, a tile draws through its patches on the sphere.
            let globe = match view.globe() {
                Some(_) => self
                    .globe_warps
                    .warp(gpu, renderer, tms, tile)
                    .map(|(warp, anchor)| Some((warp, anchor))),
                None => on_plane.then_some(None),
            };
            // Nowhere to draw it this frame: no placements.
            let Some(globe) = globe else {
                renderer.unplace(resident.gpu);
                continue;
            };
            for extent in renderer.tile_extents(resident.gpu) {
                let at = Placement {
                    extent,
                    period,
                    zoom: self.camera.state().zoom(self.surface.projection()),
                    frame,
                    series: series.as_deref(),
                    globe,
                    on_plane,
                    tile_zoom: resident.zoom,
                };
                let map_rows = |draw: &Draw| -> Vec<TileRow> {
                    placements(tms, draw, at)
                        .map(|mut row| {
                            if let Some((dem, range)) = ground {
                                row.dem = dem;
                                row.terrain_range = range;
                            }
                            row
                        })
                        .collect()
                };
                let mut rows: Vec<TileRow> = draws.iter().flat_map(|d| map_rows(d)).collect();
                if let Some(drape) = draped {
                    for draw in draws {
                        let drawn = crate::terrain::Drawn::of(tms, draw, frame, period);
                        for &(terrain, slot) in &drape.redraw {
                            if drawn.overlaps(&drape.surface, terrain) {
                                let zoom =
                                    crate::terrain::drape_zoom(&drape.surface, terrain.index);
                                let at = Placement {
                                    zoom: f64::from(zoom),
                                    ..at
                                };
                                crate::terrain::drape_rows(
                                    &drape.surface,
                                    drawn,
                                    extent,
                                    (terrain, slot),
                                    &drape.layout,
                                    placements(tms, draw, at),
                                    &mut rows,
                                );
                            }
                        }
                    }
                }
                report.tiles_drawn += rows.len() as u32;
                if renderer.place(resident.gpu, extent, &rows).is_err() {
                    self.warnings
                        .push("out of tile placements; some tiles are not drawn".into());
                }
            }
        }
        // Tiles held for their labels: drawn again, gone, or done.
        let held = &mut style.held[index];
        let now = self.services.clock.now();
        held.retain(|tile| {
            let resident = pyramid.get(*tile);
            let buckets = || {
                resident
                    .into_iter()
                    .flat_map(|r| &r.symbols)
                    .map(|s| s.bucket)
            };
            let done = rows.contains_key(tile)
                || resident.is_none()
                || buckets().all(|b| self.labels.released(b, now));
            if done {
                for bucket in buckets() {
                    self.labels.unhold(bucket);
                }
                if let Some(r) = resident.filter(|_| !rows.contains_key(tile)) {
                    renderer.unplace(r.gpu);
                }
            }
            !done
        });
        // Tiles leaving the draw list: those with labels keep drawing them
        // until a placement covers what replaced them. Still output waits
        // for placement anyway, and fades nothing.
        let holds = self.mode == RenderMode::Continuous;
        for tile in placed.iter().filter(|t| !rows.contains_key(t)) {
            if let Some(resident) = pyramid.get(*tile) {
                if resident.symbols.is_empty() || !holds {
                    renderer.unplace(resident.gpu);
                } else {
                    renderer.hold_symbols(resident.gpu);
                    for s in &resident.symbols {
                        self.labels.hold(s.bucket);
                    }
                    held.push(*tile);
                }
            }
        }
        let mut ordered = Vec::new();
        for draw in &draws {
            // Multiple wraps of a tile share its meshes and placements.
            if ordered.last() != Some(&draw.tile) {
                ordered.push(draw.tile);
            }
        }
        if *placed != ordered {
            let tiles: Vec<_> = ordered
                .iter()
                .map(|tile| pyramid.get(*tile).expect("drawn tiles are ready").gpu)
                .collect();
            renderer.set_tile_order(&tiles);
            *placed = ordered;
        }
    }

    /// Keeps the tiles out of view within the cache budget, a quarter of the
    /// memory budget: the least recently used go first, from any source.
    /// The scene's pools then hold the tiles in view and what's left of the
    /// cache, and arrivals that need room evict more (`evict_unused`).
    fn trim_cache(&mut self, renderer: &mut Renderer) {
        let Some(style) = &mut self.style else {
            return;
        };
        let budget = self.budget / 4;
        // (age, source, tile, cost), oldest first.
        let mut unused: Vec<(u64, usize, TileIndex, u64)> = Vec::new();
        for (index, source) in style.sources.iter().enumerate() {
            if let SourceState::Ready { pyramid, .. } = source {
                unused.extend(
                    pyramid
                        .unused()
                        .map(|(tile, age, r)| (age, index, tile, r.cost)),
                );
            }
        }
        let mut total: u64 = unused.iter().map(|u| u.3).sum();
        if total <= budget {
            return;
        }
        unused.sort_unstable_by_key(|u| std::cmp::Reverse(u.0));
        for (_, index, tile, cost) in unused {
            if total <= budget {
                break;
            }
            let SourceState::Ready { pyramid, .. } = &mut style.sources[index] else {
                continue;
            };
            if let Some(resident) = pyramid.evict_unused(tile) {
                total -= cost;
                self.retired_ramps
                    .extend(remove_resident(renderer, resident, &mut style.states));
            }
        }
    }

    /// The map's memory against its budget.
    fn memory(&self, renderer: &Renderer) -> Memory {
        let (tile_textures, glyph_textures) = renderer.texture_bytes();
        let (mut cached_tiles, mut cached) = (0, 0);
        for source in self.style.iter().flat_map(|s| &s.sources) {
            if let SourceState::Ready { pyramid, .. } = source {
                for (_, _, r) in pyramid.unused() {
                    cached_tiles += 1;
                    cached += r.cost;
                }
            }
        }
        let images = self.services.images.0.lock().expect("images poisoned");
        Memory {
            budget: self.budget,
            scene: renderer.memory(),
            tile_textures,
            glyph_textures,
            image_atlas: images.atlas.as_ref().map_or(0, ImageAtlas::bytes),
            cached_tiles,
            cached,
            cache_budget: self.budget / 4,
        }
    }

    /// Keeps the terrain to the style's `terrain`: made once its DEM
    /// source is ready, its DEM tiles requested for this view, and the
    /// renderer told, once a DEM tile has arrived.
    fn update_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        report: &mut FrameReport,
    ) {
        let Some(style) = &mut self.style else {
            return;
        };
        let plan = style.worker.plan();
        let mut wanted = plan.terrain();
        if wanted.is_some() && gpu.caps().tier == orb_hal::Tier::Compat {
            if !std::mem::replace(&mut style.terrain_unusable, true) {
                self.warnings.push(crate::COMPAT_NO_TERRAIN.into());
            }
            wanted = None;
        }
        let same = match (&style.terrain, wanted) {
            (Some(t), Some((index, exaggeration))) => {
                t.source == index
                    && t.key == plan.sources[index].key
                    && t.exaggeration == exaggeration
            }
            (None, None) => true,
            _ => false,
        };
        if !same {
            if let Some(old) = style.terrain.take() {
                renderer.set_terrain(gpu, None);
                remove_terrain(gpu, renderer, old);
            }
            if let Some((index, _)) = wanted
                && let SourceState::Ready { endpoint, .. } = &style.sources[index]
                && endpoint.tms != *self.surface.tms()
            {
                // Terrain tiles are the surface's; their DEM must be too.
                if !std::mem::replace(&mut style.terrain_unusable, true) {
                    self.warnings.push(format!(
                        "terrain: the DEM source's tiles ({}) are not the map's ({}); the map is flat",
                        endpoint.tms.id(),
                        self.surface.tms().id()
                    ));
                }
            } else if let Some((index, exaggeration)) = wanted
                && let SourceState::Ready { endpoint, .. } = &style.sources[index]
            {
                // Raster sources cover with `tileSize / √2` (GL JS's
                // rounded raster cover).
                let tile_px = endpoint.cover_px * std::f64::consts::SQRT_2;
                self.terrains += 1;
                style.terrain = Some(crate::terrain::Terrain::new(
                    self.terrains,
                    index,
                    plan.sources[index].key,
                    exaggeration,
                    self.surface.tms(),
                    (tile_px, endpoint.minzoom, endpoint.maxzoom),
                ));
            }
        }
        let Some(terrain) = &mut style.terrain else {
            self.frame_terrain = None;
            renderer.set_terrain(gpu, None);
            return;
        };
        let SourceState::Ready { endpoint, .. } = &style.sources[terrain.source] else {
            return;
        };
        let endpoint = Arc::clone(endpoint);
        terrain.pyramid.retry_failed(self.services.clock.now());
        let requests = terrain.update(view, self.surface.tms());
        for retired in terrain.evict() {
            if let Some(page) = terrain.page {
                renderer.free_layer(page, retired.layer);
            }
        }
        let paint = terrain.page.map(|page| {
            let zoom = self.camera.state().zoom(self.surface.projection());
            let height_scale = view.height_scale();
            // GL JS's `getSkirtLength`: a fifth of a tile's width at the
            // camera's zoom, in metres at the equator.
            let skirt = if self.terrain_skirts {
                2.0 * std::f64::consts::PI * 6_371_008.8 / zoom.max(0.0).exp2() / 5.0
            } else {
                0.0
            };
            (
                page,
                orb_render::TerrainPaint {
                    unpack: endpoint.dem_unpack.expect("raster-dem").map(|v| v as f32),
                    exaggeration: terrain.exaggeration as f32,
                    skirt: (skirt * height_scale) as f32,
                },
            )
        });
        self.frame_terrain = paint.map(|(page, p)| orb_render::FrameTerrain {
            unpack: p.unpack,
            exaggeration: p.exaggeration,
            dem_dim: renderer.page_size(page)[0] as f32,
        });
        renderer.set_terrain(gpu, paint);
        let mut loading = 0;
        for tile in requests {
            if !endpoint.contains(tile) {
                terrain.pyramid.insert_empty(tile);
                continue;
            }
            loading += 1;
            let request = TileRequest {
                style: style.id,
                source: terrain.key,
                tile,
                zoom: endpoint.tms.style_zoom(tile),
                bend: None,
                data_zoom: tile.z,
                location: endpoint.location(tile),
                encoding: endpoint.encoding,
                nodes: None,
                rows_only: false,
                dem_unpack: endpoint.dem_unpack.map(crate::DemUnpack),
            };
            self.counts.loads += 1;
            let (sender, wake, generation) = (
                self.sender.clone(),
                Arc::clone(&self.services.wake),
                self.tile_generation,
            );
            let key = terrain.key;
            self.services.jobs.load(
                request,
                Box::new(move |result| {
                    let _ = sender.send(Message::Tile {
                        generation,
                        source: key,
                        tile,
                        edit: None,
                        terrain: true,
                        result,
                    });
                    wake();
                }),
            );
        }
        terrain.loading += loading;
        report.tiles_loading += terrain.loading;
    }

    /// The terrain's CPU elevations, if the style has terrain.
    fn terrain_elevation(&mut self) -> Option<Arc<crate::terrain::TerrainElevation>> {
        let tms = self.surface.tms();
        self.style
            .as_mut()?
            .terrain
            .as_mut()
            .map(|t| Arc::clone(t.elevation(tms)))
    }

    /// The terrain's elevations as drawn, for the camera
    /// ([`orb_camera::Camera::set_terrain`]) and queries; `None` without
    /// terrain. A new snapshot comes with each
    /// [`elevation_revision`](Self::elevation_revision).
    pub fn elevation(&self) -> Option<Arc<dyn orb_camera::Elevation>> {
        let terrain = self.style.as_ref()?.terrain.as_ref()?;
        Some(terrain.snapshot() as Arc<dyn orb_camera::Elevation>)
    }

    /// [`elevation`](Self::elevation) at reduced resolution, as bytes for a
    /// camera on another thread (plan §7.1's elevation pyramid; read with
    /// [`TerrainElevation::from_pyramid`](crate::TerrainElevation::from_pyramid));
    /// empty without terrain.
    pub fn elevation_pyramid(&self) -> Arc<[u8]> {
        self.style
            .as_ref()
            .and_then(|s| s.terrain.as_ref())
            .map_or_else(Default::default, |t| t.snapshot().pyramid())
    }

    /// Changes whenever [`elevation`](Self::elevation) has new data.
    pub fn elevation_revision(&self) -> u64 {
        self.style
            .as_ref()
            .and_then(|s| s.terrain.as_ref())
            .map_or(0, |t| t.id << 32 | t.revision)
    }

    /// A terrain DEM tile arrived: into the terrain's page, prepared with
    /// its neighbours, its elevations kept.
    fn apply_terrain_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
    ) {
        let Some(style) = &mut self.style else {
            return;
        };
        let Some(terrain) = style.terrain.as_mut().filter(|t| t.key == source) else {
            return;
        };
        terrain.loading = terrain.loading.saturating_sub(1);
        if !terrain.pyramid.is_loading(tile) {
            return;
        }
        let result = result.and_then(|r| match r {
            TileResult { missing: true, .. } => Ok(None),
            TileResult {
                image: Some(image),
                dem: Some(dem),
                ..
            } => Ok(Some((image, dem))),
            _ => Err("a terrain tile without elevations".to_string()),
        });
        let (image, grid) = match result {
            Ok(Some(found)) => found,
            Ok(None) => {
                terrain.pyramid.insert_missing(tile);
                return;
            }
            Err(e) => {
                self.warnings.push(format!("terrain tile {tile:?}: {e}"));
                terrain
                    .pyramid
                    .insert_failed(tile, self.services.clock.now());
                return;
            }
        };
        let SourceState::Ready { endpoint, .. } = &style.sources[terrain.source] else {
            return;
        };
        let unpack = endpoint.dem_unpack.expect("raster-dem").map(|v| v as f32);
        let size = [image.width, image.height];
        let page = *terrain
            .page
            .get_or_insert_with(|| renderer.create_page(gpu, orb_render::PageKind::Dem, size));
        if renderer.page_size(page) != size {
            self.warnings.push(format!(
                "terrain tile {tile:?}: {}×{} among {:?} tiles",
                size[0],
                size[1],
                renderer.page_size(page)
            ));
            terrain.pyramid.insert_empty(tile);
            return;
        }
        let Ok(layer) = renderer.alloc_layer(gpu, page) else {
            self.warnings
                .push("out of terrain texture layers; a DEM tile is not drawn".into());
            terrain.pyramid.insert_empty(tile);
            return;
        };
        let pixels = match &image.pixels {
            crate::ImagePixels::Core(bytes) => orb_render::Pixels::Core(bytes),
            &crate::ImagePixels::Host(image) => orb_render::Pixels::Host {
                image,
                premultiply: false,
            },
        };
        renderer.write_layer(gpu, page, layer, pixels);
        let layer_of = |t: TileIndex| terrain.pyramid.get(t).map(|d| d.layer);
        for job in orb_render::dem_prepares(tile, layer, unpack, &layer_of) {
            renderer.prepare_dem(page, job);
        }
        if let Some(old) = terrain.insert(
            tile,
            crate::terrain::DemTile {
                layer,
                dem: Arc::new(crate::terrain::Dem::new(grid)),
            },
        ) {
            renderer.free_layer(page, old.layer);
        }
    }

    /// Plans this frame's drape: makes room for the terrain tiles' slots,
    /// writes the draped paint at their drape zooms and picks the slots to
    /// redraw, by what is draped under each tile (decision 0019). Nothing
    /// here depends on the camera but which tiles there are, and the
    /// bearing under layers that turn with it.
    fn plan_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
    ) -> Option<DrapePlan> {
        let Some(style) = &mut self.style else {
            return None;
        };
        let terrain = style.terrain.as_ref()?;
        if !renderer.terrain_on() {
            return None;
        }
        let layout = match renderer.reserve_drape(gpu, terrain.slots_needed()) {
            Ok(layout) => layout,
            Err(_) => {
                self.warnings
                    .push("too many terrain tiles for the drape atlas; terrain is flat".into());
                renderer.set_terrain(gpu, None);
                return None;
            }
        };
        let surface = self.surface.tms().clone();
        let zooms: BTreeSet<u8> = terrain
            .tiles
            .iter()
            .map(|t| crate::terrain::drape_zoom(&surface, t.index))
            .collect();
        let worker = Arc::clone(&style.worker);
        let plan = worker.plan();
        let bearing = self.camera.state().bearing;
        let drape = &mut style.drape;
        if drape.generation != Some(layout.generation) {
            // A new atlas, or one whose slots all draw again.
            drape.generation = Some(layout.generation);
            drape.painted.clear();
        }
        if drape.bearing != bearing {
            drape.bearing = bearing;
            let turning = &drape.turning;
            drape.painted.retain(|z| !turning.contains(z));
        }
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        for &zoom in &zooms {
            if style.drape.painted.contains(&zoom) {
                continue;
            }
            let (changed, turns) =
                write_drape_paint(gpu, renderer, (style, plan), zoom, bearing, atlas);
            let drape = &mut style.drape;
            if changed {
                *drape.revisions.entry(zoom).or_default() += 1;
            }
            if turns {
                drape.turning.insert(zoom);
            } else {
                drape.turning.remove(&zoom);
            }
            drape.painted.insert(zoom);
        }
        let images_revision = atlas.revision();
        drop(images);
        let draped: Vec<usize> = (0..style.sources.len())
            .filter(|&i| {
                plan.nodes
                    .iter()
                    .any(|n| n.source == i && n.family.is_draped())
            })
            .collect();
        let period = self.surface.period();
        let sources = &style.sources;
        let draws = &style.draws;
        let drape = &style.drape;
        let revision = (
            self.version,
            style.states.writes(),
            style.states.zoom_key(),
            images_revision,
        );
        let content = |t: orb_crs::CoveredTile| {
            let zoom = crate::terrain::drape_zoom(&surface, t.index);
            let mut parts = Vec::new();
            for &i in &draped {
                let (
                    Some(draws),
                    SourceState::Ready {
                        pyramid, endpoint, ..
                    },
                ) = (&draws[i], &sources[i])
                else {
                    continue;
                };
                for draw in draws {
                    let resident = pyramid.get(draw.tile);
                    let frame = resident
                        .and_then(|r| r.image.as_ref())
                        .and_then(|image| image.warped.as_ref())
                        .map(|warped| warped.frame);
                    let drawn = crate::terrain::Drawn::of(&endpoint.tms, draw, frame, period);
                    if !drawn.overlaps(&surface, t) {
                        continue;
                    }
                    // The renderer's tiles stand for their meshes.
                    let meshes: Vec<_> = resident
                        .map(|r| r.meshes.iter().map(|m| m.mesh).collect())
                        .unwrap_or_default();
                    let held = resident.map(|r| r.gpu);
                    // A DEM tile's borders come from its neighbours.
                    let borders = endpoint.dem_unpack.map(|_| {
                        orb_render::resident_neighbours(draw.tile, &|n| pyramid.get(n).is_some())
                    });
                    let regions = draw.regions.clone();
                    parts.push((i, draw.tile, draw.wrap, regions, meshes, held, borders));
                }
            }
            let paint = drape.revisions.get(&zoom);
            let turned = drape.turning.contains(&zoom).then_some(bearing.to_bits());
            crate::terrain::hash((revision, parts, zoom, paint, turned))
        };
        let globe_warps = &mut self.globe_warps;
        let mut globe = |tile: TileIndex| {
            let target = view.globe()?.target;
            let (warp, anchor) = globe_warps.warp(gpu, renderer, &surface, tile)?;
            Some(orb_render::TerrainGlobe {
                warp,
                origin: (anchor - target).as_vec3().to_array(),
                poles: orb_crs::sphere::poles(&surface, tile),
            })
        };
        let terrain = style.terrain.as_mut().expect("checked above");
        let (tiles, redraw) = terrain.plan(&surface, view, layout, &content, &mut globe);
        renderer.set_terrain_tiles(gpu, &tiles);
        let slots: Vec<(u32, u8)> = redraw
            .iter()
            .map(|&(tile, slot)| (slot, crate::terrain::drape_zoom(&surface, tile.index)))
            .collect();
        renderer.redraw_drape(&slots);
        Some(DrapePlan {
            surface,
            layout,
            redraw,
        })
    }

    /// Places the background layers' quads on every covered tile, and on
    /// the drape slots being redrawn.
    fn place_background(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        elevation: Option<&crate::terrain::TerrainElevation>,
        drape: Option<&DrapePlan>,
        report: &mut FrameReport,
    ) {
        let Some(style) = &mut self.style else {
            return;
        };
        // Backgrounds draw on every covered tile, as GL JS's do: never
        // beyond the world's edge.
        let Some(tile) = style.background else {
            return;
        };
        let zoom = self.camera.state().zoom(self.surface.projection());
        let tms = self.surface.tms();
        let cover = match view.low_plane() {
            Some(low) => low.cover_warped(tms, TILE_PX, 24, |tile| {
                self.globe_warps.footprint(tms, tile)
            }),
            None => view.cover_volume(
                tms,
                TILE_PX,
                0..=24,
                0.0,
                elevation.map(|e| e as &dyn orb_camera::Elevation),
                0.0,
            ),
        };
        let period = self.surface.period();
        let at = Placement {
            extent: 8192,
            period,
            zoom,
            frame: None,
            series: None,
            globe: None,
            on_plane: true,
            tile_zoom: 0.0,
        };
        let globe_warps = &mut self.globe_warps;
        let mut rows: Vec<TileRow> = cover
            .iter()
            .filter_map(|covered| {
                let draw = Draw {
                    tile: covered.index,
                    wrap: covered.wrap,
                    regions: vec![covered.index],
                };
                let globe = match view.globe() {
                    Some(_) => Some(globe_warps.warp(gpu, renderer, tms, covered.index)?),
                    None => None,
                };
                let at = Placement {
                    tile_zoom: tms.style_zoom(covered.index),
                    globe,
                    ..at
                };
                placements(tms, &draw, at).next()
            })
            .collect();
        report.tiles_drawn += rows.len() as u32;
        if let Some(drape) = drape {
            for &(terrain, slot) in &drape.redraw {
                let draw = Draw {
                    tile: terrain.index,
                    wrap: terrain.wrap,
                    regions: vec![terrain.index],
                };
                let at = Placement {
                    tile_zoom: tms.style_zoom(terrain.index),
                    zoom: f64::from(crate::terrain::drape_zoom(tms, terrain.index)),
                    ..at
                };
                let map = placements(tms, &draw, at);
                crate::terrain::drape_rows(
                    tms,
                    crate::terrain::Drawn::of(tms, &draw, None, period),
                    8192,
                    (terrain, slot),
                    &drape.layout,
                    map,
                    &mut rows,
                );
            }
        }
        if renderer.place(tile, 8192, &rows).is_err() {
            self.warnings
                .push("out of tile placements; the background is not drawn".into());
        }
    }

    /// Installs a style that `load_style_url` fetched, unless a later
    /// request or `set_style_json` superseded it.
    fn apply_style(
        &mut self,
        request: u64,
        url: &str,
        scene: bool,
        result: Result<String, FetchError>,
    ) {
        if request != self.style_request {
            return;
        }
        let installed = result
            .map_err(|e| e.to_string())
            .and_then(|json| self.set_document(&json, Some(url), scene));
        if let Err(e) = installed {
            self.fail(format!("style: {e}"));
        }
    }

    /// Applies arrived results in order. New tiles apply within
    /// [`TILE_APPLY_MS`], at least one a frame: tiles that arrive together
    /// spread their uploads over frames rather than stalling one. Edits'
    /// results apply whole.
    fn apply_messages(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        let started = self.services.clock.now();
        self.tiles_deferred = false;
        while let Ok(message) = self.receiver.try_recv() {
            let new_tile = matches!(message, Message::Tile { edit: None, .. });
            match message {
                // Replies to the labels a GPU reset replaced are dropped.
                Message::Placed { epoch, committed } => {
                    if epoch == self.label_epoch {
                        let started = self.services.clock.now();
                        self.labels.committed(gpu, renderer, committed);
                        self.placement_ms += self.services.clock.now() - started;
                    }
                }
                Message::Raster {
                    epoch,
                    rows,
                    result,
                } => {
                    if epoch == self.label_epoch {
                        self.labels.rastered(
                            gpu,
                            renderer.glyphs(),
                            &rows,
                            result,
                            &mut self.warnings,
                        );
                    }
                }
                Message::Style {
                    request,
                    url,
                    scene,
                    result,
                } => self.apply_style(request, &url, scene, result),
                Message::Sprite { generation, result } => {
                    if let Some(style) = self
                        .style
                        .as_mut()
                        .filter(|_| generation == self.generation)
                    {
                        style.sprites_loading -= 1;
                        match result {
                            Err(e) => {
                                let message = format!("sprite: {e}");
                                self.events.push(MapEvent::Error { message });
                            }
                            Ok(sprites) => {
                                let mut atlas =
                                    self.services.images.0.lock().expect("images poisoned");
                                let atlas = atlas.atlas.as_mut().expect("initialized");
                                for image in sprites {
                                    style.icons.add(
                                        &image.name,
                                        [image.width, image.height],
                                        &image.metadata,
                                    );
                                    let namespace =
                                        style.image_namespace.expect("sprite namespace");
                                    // Patterns and icons pad differently.
                                    let mut upload = |pattern: bool| {
                                        let id = if pattern {
                                            atlas.resolve(namespace, &image.name)?
                                        } else {
                                            atlas.resolve_icon(namespace, &image.name)?
                                        };
                                        atlas.upload(
                                            gpu,
                                            id,
                                            ImageUpload {
                                                size: [image.width, image.height],
                                                pixels: &image.pixels,
                                                metadata: image.metadata.clone(),
                                                pattern,
                                            },
                                        )
                                    };
                                    let uploaded = upload(true).and_then(|()| upload(false));
                                    if let Err(e) = uploaded {
                                        self.warnings.push(format!("sprite {}: {e}", image.name));
                                    }
                                }
                            }
                        }
                        if style.sprites_loading == 0 {
                            self.services.jobs.set_icons(
                                style.id,
                                &Arc::new(style.icons.clone()),
                                &style.worker,
                            );
                        }
                    }
                }
                Message::TileJson {
                    generation,
                    source,
                    result,
                } => {
                    let Some(style) = self
                        .style
                        .as_mut()
                        .filter(|_| generation == self.generation)
                    else {
                        continue;
                    };
                    let plan = style.worker.plan();
                    let Some(index) = plan.source(source) else {
                        continue;
                    };
                    if !matches!(style.sources[index], SourceState::Resolving) {
                        continue;
                    }
                    let def = plan.source_def(index);
                    let tms = self.surface.source_tms(&def.id, false);
                    let endpoint = match (result, self.surface.plan(&tms, plan.image_kind(index))) {
                        (Err(e), _) => Err(format!("TileJSON: {e}")),
                        (Ok(_), Err(e)) => Err(e),
                        (Ok(text), Ok(warp)) => TileEndpoint::resolve(&def.kind, Some(&text), tms)
                            .map(|endpoint| (endpoint, warp)),
                    };
                    let image = plan.image_kind(index).is_some();
                    style.sources[index] = ready_source(endpoint, &def.id, image, &mut self.events);
                    self.events.sources.insert(index);
                    // Color-relief ramps wait for their source's encoding.
                    style.painted = None;
                    for node in plan.nodes.iter().filter(|n| n.source == index) {
                        style.nodes.entry(node.key).or_default().ramp = None;
                    }
                }
                Message::Tile {
                    generation,
                    source,
                    tile,
                    edit,
                    terrain,
                    result,
                } => {
                    let (host_bytes, host_image) =
                        result.as_ref().ok().map_or((None, None), host_data);
                    if let Some(edit) = edit {
                        self.finish_job(edit);
                    }
                    let mut kept = false;
                    if generation == self.tile_generation {
                        match edit {
                            None if terrain => {
                                self.apply_terrain_tile(gpu, renderer, source, tile, result)
                            }
                            None => {
                                if let Some(build) =
                                    self.apply_tile(gpu, renderer, source, tile, result, true)
                                {
                                    kept = true;
                                    let tried = Some(self.room_state(renderer));
                                    self.waiting.push(WaitingTile {
                                        generation,
                                        source,
                                        tile,
                                        build,
                                        tried,
                                    });
                                }
                            }
                            Some(_) => self.apply_edit(gpu, renderer, source, tile, result),
                        }
                    }
                    // Uploaded or dropped, the result's host data is done.
                    if let Some(bytes) = host_bytes.filter(|_| !kept) {
                        gpu.release_host_bytes(bytes);
                    }
                    if let Some(image) = host_image.filter(|_| !kept) {
                        gpu.release_host_image(image);
                    }
                }
            }
            if new_tile && self.services.clock.now() - started > TILE_APPLY_MS {
                self.tiles_deferred = true;
                break;
            }
        }
        self.retry_waiting(gpu, renderer, started);
    }

    /// Tries the tiles waiting for room again, in the frame's budget left,
    /// each only if room may have changed since it last failed: memory
    /// freed or grown, or other tiles' jobs done.
    fn retry_waiting(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer, started: f64) {
        let mut state = self.room_state(renderer);
        for mut waiting in std::mem::take(&mut self.waiting) {
            let (bytes, image) = host_data(&waiting.build);
            let release = |gpu: &mut dyn Gpu| {
                if let Some(bytes) = bytes {
                    gpu.release_host_bytes(bytes);
                }
                if let Some(image) = image {
                    gpu.release_host_image(image);
                }
            };
            if waiting.generation != self.tile_generation {
                release(gpu);
                continue;
            }
            if self.tiles_deferred || waiting.tried == Some(state) {
                self.waiting.push(waiting);
                continue;
            }
            match self.apply_tile(
                gpu,
                renderer,
                waiting.source,
                waiting.tile,
                Ok(waiting.build),
                false,
            ) {
                Some(build) => {
                    state = self.room_state(renderer);
                    waiting.build = build;
                    waiting.tried = Some(state);
                    self.waiting.push(waiting);
                }
                None => {
                    release(gpu);
                    state = self.room_state(renderer);
                }
            }
            self.tiles_deferred = self.services.clock.now() - started > TILE_APPLY_MS;
        }
    }

    /// What a waiting tile's room depends on: the scene's pools in use and
    /// their buffers, tile page layers in use, and tile jobs in flight.
    fn room_state(&self, renderer: &Renderer) -> [u64; 6] {
        let stats = renderer.stats();
        let loading: u64 = self
            .style
            .iter()
            .flat_map(|s| &s.sources)
            .map(|s| match s {
                SourceState::Ready { loading, .. } => u64::from(*loading),
                _ => 0,
            })
            .sum();
        [
            u64::from(stats.vertex_words),
            u64::from(stats.index_words),
            u64::from(stats.prop_words),
            renderer.pools_revision(),
            renderer.used_layers(),
            loading,
        ]
    }

    fn finish_job(&mut self, edit: u64) {
        if let Some(count) = self.pending.get_mut(&edit) {
            *count -= 1;
            if *count == 0 {
                self.pending.remove(&edit);
            }
        }
    }

    /// Asks the host to open and close video sources' players, and puts
    /// their frames in their tiles.
    fn apply_videos(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        let mut videos = std::mem::take(&mut self.videos);
        let plan = self.style.as_ref().map(|s| s.worker.plan());
        videos.sync(plan, &mut self.events.queue);
        for image in videos.released() {
            gpu.release_host_image(image);
        }
        let mut retries = Vec::new();
        for work in videos.work() {
            if let Some(frame) = work.fresh {
                self.write_video_frame(gpu, renderer, work.key, frame);
            }
            for (tile, counted) in work.wanted {
                let result = match (work.failed, work.frame) {
                    (Some(e), _) => Err(format!("video: {e}")),
                    (None, frame) => Ok(TileResult {
                        image: frame.map(crate::video::tile_image),
                        ..TileResult::default()
                    }),
                };
                // The frame stays the source's: a tile without room drops
                // it and tries again.
                if self
                    .apply_tile(gpu, renderer, work.key, tile, result, !counted)
                    .is_some()
                {
                    retries.push((work.key, tile));
                }
            }
        }
        for (key, tile) in retries {
            videos.retry(key, tile);
        }
        self.videos = videos;
    }

    /// Writes a video's new frame over its resident tile's layer; a frame
    /// of another size waits for the tile to load again.
    fn write_video_frame(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        key: u32,
        frame: &crate::TileImage,
    ) {
        let Some(style) = &self.style else {
            return;
        };
        let Some(index) = style.worker.plan().source(key) else {
            return;
        };
        let SourceState::Ready { pyramid, .. } = &style.sources[index] else {
            return;
        };
        let pixels = match &frame.pixels {
            crate::ImagePixels::Core(pixels) => orb_render::Pixels::Core(pixels),
            &crate::ImagePixels::Host(image) => orb_render::Pixels::Host {
                image,
                premultiply: true,
            },
        };
        for (tile, resident) in pyramid.ready() {
            if resident.image.is_some() {
                let key = orb_render::TileKey { source: key, tile };
                renderer.rewrite_image(gpu, key, [frame.width, frame.height], pixels);
            }
        }
    }

    /// Whether the host plays `video` sources: it then gets
    /// [`MapEvent::Video`] requests and hands frames to
    /// [`video_frame`](Self::video_frame). Without it, video sources draw
    /// nothing, with a warning.
    pub fn set_plays_video(&mut self, plays: bool) {
        self.videos.enabled = plays;
    }

    /// The frame source `id`'s video shows now, straight-alpha RGBA rows,
    /// top first, or an image the host registered with the GPU; it
    /// replaces any frame not yet drawn.
    pub fn video_frame(&mut self, id: &str, frame: crate::TileImage) -> Result<(), String> {
        self.videos.frame(id, frame)?;
        self.events.wake();
        Ok(())
    }

    /// Source `id`'s video can't play here: its tile draws nothing.
    pub fn video_failed(&mut self, id: &str, message: &str) -> Result<(), String> {
        self.videos.failed(id, message.to_owned())?;
        self.events.wake();
        Ok(())
    }

    /// Asks the host to play, pause or seek the video of source `id`
    /// (GL JS's `VideoSource` `play`, `pause` and `seek`).
    pub fn control_video(&mut self, id: &str, request: crate::VideoRequest) -> Result<(), String> {
        let plan = self.style.as_ref().map(|s| s.worker.plan());
        self.videos.sync(plan, &mut self.events.queue);
        self.videos.request(id, request, &mut self.events.queue)?;
        // The request goes out with the next frame's events.
        self.events.wake();
        Ok(())
    }

    fn apply_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
        arrived: bool,
    ) -> Option<TileResult> {
        let Some(style) = &mut self.style else {
            return None;
        };
        let plan = style.worker.plan();
        let source = plan.source(source)?;
        let SourceState::Ready {
            pyramid,
            loading,
            endpoint,
            warp,
            ..
        } = &mut style.sources[source]
        else {
            return None;
        };
        if arrived {
            *loading = loading.saturating_sub(1);
            self.events.sources.insert(source);
        }
        let mut build = match result {
            Ok(build) => build,
            Err(e) => {
                let message = format!("tile {tile:?}: {e}");
                self.events.push(MapEvent::Error { message });
                pyramid.insert_failed(tile, self.services.clock.now());
                return None;
            }
        };
        self.warnings.append(&mut build.warnings);
        crate::style_image::report_missing(
            &mut style.reported_missing,
            &style.icons,
            style.sprites_loading,
            &build,
            &mut self.events,
        );
        if build.missing {
            pyramid.insert_missing(tile);
            return None;
        }
        if let Some(tile_image) = &build.image {
            // Neighbours' prepares refer to the tile's layer, so only a
            // tile the pyramid still wants is uploaded.
            if !pyramid.is_loading(tile) {
                return None;
            }
            let kind = plan
                .image_kind(source)
                .expect("image results are for image sources");
            let tile_zoom = self.surface.tile_zoom(
                &endpoint.tms,
                tile,
                warp.as_mut().and_then(|w| w.footprint(tile)),
            );
            let globe = self.surface.globe().is_some();
            let Ok(warped) = image_mesh(
                &mut self.globe_warps,
                globe,
                warp.as_deref_mut(),
                endpoint,
                tile,
            ) else {
                // Nothing of the tile reaches the surface.
                pyramid.insert_empty(tile);
                return None;
            };
            // Raster tiles fade in over their layers' longest duration.
            let fade_ms = plan
                .nodes
                .iter()
                .filter(|n| n.source == source)
                .filter_map(|n| {
                    let r = &plan.style.layers[n.layer];
                    (r.ty == LayerType::Raster
                        && !matches!(
                            plan.source_def(source).kind,
                            orb_style::SourceKind::Image(_)
                        ))
                    .then(|| f64::from(r.get(orb_style::RASTER_FADE_DURATION).evaluate(tile_zoom)))
                })
                .fold(0.0, f64::max);
            // Still output shows tiles at once.
            let fade_ms = if self.mode == RenderMode::Continuous {
                fade_ms
            } else {
                0.0
            };
            let now = self.services.clock.now();
            let series = warp
                .as_deref_mut()
                .filter(|_| globe)
                .and_then(|w| w.series(tile));
            let patches = series.as_deref().map(series_patches);
            let image_build = orb_render::ImageBuild {
                key: TileKey {
                    source: plan.sources[source].key,
                    tile,
                },
                kind: match kind {
                    crate::style::ImageKind::Raster => orb_render::PageKind::Raster,
                    crate::style::ImageKind::Dem => orb_render::PageKind::Dem,
                },
                size: [tile_image.width, tile_image.height],
                pixels: match &tile_image.pixels {
                    crate::ImagePixels::Core(bytes) => orb_render::Pixels::Core(bytes),
                    &crate::ImagePixels::Host(image) => orb_render::Pixels::Host {
                        image,
                        premultiply: kind == crate::style::ImageKind::Raster,
                    },
                },
                mesh: warped.as_ref().map(|w| Arc::clone(&w.mesh)),
                patches: patches.as_ref().map(|(cells, p)| (*cells, &p[..])),
                fade_from: (fade_ms > 0.0).then_some((now - self.born) as f32),
                unpack: endpoint.dem_unpack.map(|u| u.map(|v| v as f32)),
            };
            // Room as for vector tiles: what's free first, then unused
            // tiles of any source go, then the page and pools grow; while
            // other tiles load, this one waits for theirs to go; then it
            // gives up for what stands in.
            let others_loading = style
                .sources
                .iter()
                .any(|s| matches!(s, SourceState::Ready { loading, .. } if *loading > 0));
            let mut grow = false;
            let uploaded = loop {
                match renderer.upload_image(gpu, &image_build, grow) {
                    Ok(handle) => break Ok(handle),
                    Err(orb_render::UploadError::Invalid(e)) => break Err(Some(e)),
                    Err(orb_render::UploadError::NoRoom) => {}
                }
                if let Some(victim) = evict_unused(&mut style.sources, source) {
                    self.retired_ramps
                        .extend(remove_resident(renderer, victim, &mut style.states));
                } else if !grow {
                    grow = true;
                } else {
                    break Err(None);
                }
            };
            let SourceState::Ready { pyramid, .. } = &mut style.sources[source] else {
                unreachable!("checked above");
            };
            match uploaded {
                Ok(handle) => {
                    if fade_ms > 0.0 {
                        self.image_fades_until = self.image_fades_until.max(now + fade_ms);
                    }
                    let mut resident = ResidentTile {
                        image: Some(ResidentImage { warped }),
                        series,
                        ..ResidentTile::new(handle, tile_zoom)
                    };
                    resident.cost = tile_cost(renderer, &resident);
                    if let Some(unwanted) = pyramid.insert(tile, resident) {
                        self.retired_ramps.extend(remove_resident(
                            renderer,
                            unwanted,
                            &mut style.states,
                        ));
                    }
                }
                Err(Some(e)) => {
                    self.warnings.push(format!("tile {tile:?}: {e}"));
                    pyramid.insert_empty(tile);
                }
                Err(None) if others_loading => return Some(build),
                Err(None) => {
                    let message = format!(
                        "tile {tile:?} doesn't fit in scene memory with the tiles in view; \
                         a coarser tile stands in"
                    );
                    self.events.push(MapEvent::Error { message });
                    pyramid.insert_missing(tile);
                }
            }
            return None;
        }
        if !pyramid.is_loading(tile) {
            return None;
        }
        // Nodes the tile was laid out for before an edit changed them are
        // laid out again from the job's cached tile.
        let mut stale = Vec::new();
        for node in plan.nodes.iter().filter(|n| n.source == source) {
            let state = style.nodes.entry(node.key).or_default();
            if build.revision < state.layout_epoch.max(state.rows_epoch) {
                stale.push(node.key);
            }
        }
        drop_stale(&mut build, plan, &stale);
        let footprint = warp.as_mut().and_then(|warp| warp.footprint(tile));
        let tile_zoom = self.surface.tile_zoom(&endpoint.tms, tile, footprint);
        // A tile on another plane draws through its series' patches.
        let series = warp.as_mut().and_then(|warp| warp.series(tile));
        let patches = series.as_deref().map(series_patches);
        let key = TileKey {
            source: plan.sources[source].key,
            tile,
        };
        let mut atlas = self.services.images.0.lock().expect("images poisoned");
        // Make room for the whole tile before writing any of it: an upload
        // that runs out part way starts over, so making room one eviction
        // at a time would write the tile again for every tile evicted.
        let sizes = mesh_sizes(plan, &build);
        let mut fits = renderer.fits(&sizes);
        while !fits {
            let Some(victim) = evict_unused(&mut style.sources, source) else {
                break;
            };
            self.retired_ramps
                .extend(remove_resident(renderer, victim, &mut style.states));
            fits = renderer.fits(&sizes);
        }
        // With nothing more to evict, the scene grows within its budget.
        if !fits {
            fits = renderer.grow_for(gpu, &sizes);
        }
        // Other tiles still loading will free the stand-ins they replace:
        // until they have, this one waits rather than give up its own, and
        // writes nothing meanwhile.
        let others_loading = style
            .sources
            .iter()
            .any(|s| matches!(s, SourceState::Ready { loading, .. } if *loading > 0));
        if !fits && others_loading {
            return Some(build);
        }
        // Then evict cached tiles the view doesn't use, least recently used
        // first, this source's before others'; then the tiles this one
        // replaces as soon as it's ready.
        let resident = loop {
            atlas
                .atlas
                .as_mut()
                .unwrap()
                .release_ramps(gpu, self.retired_ramps.drain(..));
            let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
            match upload(
                gpu,
                renderer,
                plan,
                (key, &build, patches),
                atlas.atlas.as_mut().expect("initialized"),
                style.image_namespace.expect("namespace"),
                &mut style.states,
                &mut self.labels,
                tile_zoom,
                &mut self.warnings,
            ) {
                Ok(resident) => break Some(resident),
                Err(OutOfSpace) => {
                    let unused = evict_unused(&mut style.sources, source);
                    // Labels' sizes aren't known ahead: grow both pools.
                    if unused.is_none() && renderer.grow(gpu) {
                        continue;
                    }
                    let victim = unused.or_else(|| match &mut style.sources[source] {
                        SourceState::Ready { pyramid, .. } if !others_loading => {
                            pyramid.evict_within(tile)
                        }
                        _ => None,
                    });
                    match victim {
                        Some(victim) => self.retired_ramps.extend(remove_resident(
                            renderer,
                            victim,
                            &mut style.states,
                        )),
                        None => break None,
                    }
                }
            }
        };
        drop(atlas);
        let SourceState::Ready { pyramid, .. } = &mut style.sources[source] else {
            unreachable!("checked above");
        };
        let Some(mut resident) = resident else {
            if others_loading {
                return Some(build);
            }
            // Even the tiles in view don't leave room: draw what stands in,
            // and say so.
            let message = format!(
                "tile {tile:?} doesn't fit in scene memory with the tiles in view; \
                 a coarser tile stands in"
            );
            self.events.push(MapEvent::Error { message });
            pyramid.insert_missing(tile);
            return None;
        };
        resident.series = series;
        self.counts.meshes_uploaded += build.meshes.len() as u32;
        record_maxima(&mut style.nodes, &build.meshes, &mut style.painted);
        resident.cost = tile_cost(renderer, &resident);
        if let Some(unwanted) = pyramid.insert(tile, resident) {
            self.retired_ramps
                .extend(remove_resident(renderer, unwanted, &mut style.states));
        }
        if !stale.is_empty() {
            let version = self.version;
            self.send_job(source, tile, Some(stale), false, Some(version));
        }
        None
    }

    fn fetch_text(
        &self,
        url: &str,
        message: impl FnOnce(Result<String, FetchError>) -> Message + Send + 'static,
    ) {
        let sender = self.sender.clone();
        let wake = Arc::clone(&self.services.wake);
        self.services.fetch.get(
            url,
            Box::new(move |result| {
                let result = result.map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
                let _ = sender.send(message(result));
                wake();
            }),
        );
    }

    /// Drops the style and its tiles.
    fn clear_style(&mut self) {
        if let Some(namespace) = self.style.as_ref().and_then(|s| s.image_namespace) {
            self.retired_images.push(namespace);
        }
        self.generation += 1;
        self.clear_style_tiles();
        if let Some(style) = &mut self.style {
            self.retired.tiles.extend(style.background.take());
            self.retired.tables.extend(style.states.take_table());
        }
        self.style = None;
    }

    /// Removes the style's tiles from the scene; they reload as needed.
    fn clear_style_tiles(&mut self) {
        let Some(style) = &mut self.style else { return };
        for source in style.sources.iter_mut().chain(&mut style.retired) {
            if let SourceState::Ready {
                pyramid, loading, ..
            } = source
            {
                *loading = 0;
                for tile in pyramid.drain() {
                    self.retired.tiles.push(tile.gpu);
                    self.retired_ramps
                        .extend(release_resident(tile, &mut style.states));
                }
            }
        }
        for placed in style.placed.iter_mut().chain(&mut style.held) {
            placed.clear();
        }
        if let Some(terrain) = style.terrain.take() {
            // Destroyed once the renderer's terrain is off.
            self.retired_pages.extend(terrain.page);
        }
        // In-flight tiles belong to the old scene.
        self.tile_generation += 1;
    }
}

/// Raises the nodes' row maxima to those of new meshes; paint is written
/// again when they grow.
pub(crate) fn record_maxima(
    nodes: &mut HashMap<u32, NodeState>,
    meshes: &[ResultMesh],
    painted: &mut Option<(f64, f64)>,
) {
    for mesh in meshes {
        let Some(node) = nodes.get_mut(&mesh.node) else {
            continue;
        };
        if node.maxima.len() < mesh.maxima.len() {
            node.maxima.resize(mesh.maxima.len(), 0.0);
        }
        for (max, &value) in node.maxima.iter_mut().zip(&mesh.maxima) {
            if value > *max {
                *max = value;
                *painted = None;
            }
        }
    }
}

/// A source whose endpoint resolved; `image` sources cache fewer tiles.
fn ready_source(
    endpoint: Result<(TileEndpoint, Plan), String>,
    id: &str,
    image: bool,
    events: &mut crate::events::EventQueue,
) -> SourceState {
    match endpoint {
        Ok((endpoint, plan)) => SourceState::Ready {
            warp: match plan {
                Plan::Affine => None,
                Plan::Warp(warp) => Some(Box::new(WarpState::new(*warp))),
            },
            pyramid: TilePyramid::new(
                endpoint.minzoom,
                if image {
                    endpoint.maxzoom
                } else {
                    endpoint.cover_zoom()
                },
                // Map::trim_cache bounds the cache by memory.
                usize::MAX,
            ),
            endpoint: Arc::new(endpoint),
            loading: 0,
        },
        Err(e) => {
            events.push(MapEvent::Error {
                message: format!("source {id}: {e}"),
            });
            SourceState::Failed
        }
    }
}

/// Drops a result's meshes and symbol buckets of nodes no longer planned
/// or laid out since (`stale`, by key); a symbol layer's bucket goes with
/// its nodes.
fn drop_stale(build: &mut TileResult, plan: &StylePlan, stale: &[u32]) {
    build
        .meshes
        .retain(|m| plan.node(m.node).is_some() && !stale.contains(&m.node));
    build.symbols.retain(|s| {
        let keys = [s.layer * 2, s.layer * 2 + 1];
        plan.node(keys[0]).is_some() && !keys.iter().any(|k| stale.contains(k))
    });
}

pub(crate) fn words(stream: &Stream, bytes: Option<HostBytesId>) -> Words<'_> {
    match stream {
        Stream::Core(words) => Words::Core(words),
        Stream::Host { offset, len } => Words::Host {
            bytes: bytes.expect("host streams come with host bytes"),
            offset: *offset,
            len: *len,
        },
    }
}

/// What a result mesh's rows refer to besides their words: the images
/// patched into them, the ramps among those (retained), and the
/// feature-state indices its features refer to (interned).
pub(crate) struct ResolvedRows {
    pub(crate) images: Vec<(u32, u32)>,
    pub(crate) ramps: Vec<u32>,
    pub(crate) states: Vec<u32>,
}

impl ResolvedRows {
    pub(crate) fn rows(&self) -> orb_render::MeshRows<'_> {
        orb_render::MeshRows {
            images: &self.images,
            states: &self.states,
        }
    }

    /// Hands back what [`resolve_rows`] retained, for rows that didn't
    /// upload.
    pub(crate) fn release(
        self,
        gpu: &mut dyn Gpu,
        atlas: &mut ImageAtlas,
        states: &mut FeatureStates,
    ) {
        for index in self.states {
            states.release(index);
        }
        atlas.release_ramps(gpu, self.ramps);
    }
}

/// Resolves a result mesh's images in the atlas and interns its features'
/// states.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_rows(
    gpu: &mut dyn Gpu,
    plan: &StylePlan,
    result: &TileResult,
    mesh: &ResultMesh,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    warnings: &mut Vec<String>,
) -> ResolvedRows {
    let mut images = Vec::new();
    let mut ramps = Vec::new();
    for image in &mesh.images {
        match atlas.resolve_key(gpu, namespace, &image.key) {
            Ok(image_id) => {
                if matches!(image.key, orb_render::ImageKey::Ramp { .. }) {
                    atlas.retain_ramp(image_id);
                    ramps.push(image_id);
                }
                images.push((image.word, image_id));
            }
            Err(e) => warnings.push(e.into()),
        }
    }
    let mut indices = Vec::new();
    if let (Some(ids), Some(node)) = (&mesh.state_ids, plan.node(mesh.node)) {
        let node = &plan.nodes[node];
        let source = plan.sources[node.source].key;
        let layer = plan.source_layer(node);
        indices = ids
            .iter()
            .enumerate()
            .map(|(slot, &i)| match result.ids.get(i as usize) {
                Some(id) => states.intern(source, layer, id, mesh.data.get(slot)),
                None => 0,
            })
            .collect();
    }
    ResolvedRows {
        images,
        ramps,
        states: indices,
    }
}

/// Uploads one result mesh into a resident tile, in the part of its
/// extent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_mesh(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    result: &TileResult,
    mesh: &ResultMesh,
    resident: &mut ResidentTile,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    warnings: &mut Vec<String>,
) -> Result<(), OutOfSpace> {
    let node = plan.node(mesh.node).expect("meshes of planned nodes") as u32;
    let data = MeshData {
        vertices: words(&mesh.vertices, result.host_bytes),
        indices: words(&mesh.indices, result.host_bytes),
        props: words(&mesh.props, result.host_bytes),
        row_words: mesh.row_words,
        chunks: &mesh.chunks,
    };
    let rows = resolve_rows(gpu, plan, result, mesh, atlas, namespace, states, warnings);
    let added = renderer.add_mesh(gpu, resident.gpu, (node, mesh.extent), &data, &rows.rows());
    let id = match added {
        Ok(id) => id,
        Err(e) => {
            rows.release(gpu, atlas, states);
            return Err(e);
        }
    };
    resident.meshes.push(Resident {
        hits: mesh.hits.clone(),
        node: mesh.node,
        mesh: id,
        layout_rev: result.revision,
        rows_rev: result.revision,
        ramps: rows.ramps,
        states: rows.states,
    });
    Ok(())
}

/// Uploads a tile's meshes and symbol buckets, drawing through `patches`:
/// all of them or, when the scene is out of space, none.
#[allow(clippy::too_many_arguments)]
fn upload(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    (key, result, patches): (TileKey, &TileResult, Option<(u32, &[f32])>),
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    labels: &mut crate::labels::Labels,
    tile_zoom: f64,
    warnings: &mut Vec<String>,
) -> Result<ResidentTile, OutOfSpace> {
    let handle = renderer.add_tile(gpu, key, patches)?;
    let mut resident = ResidentTile::new(handle, tile_zoom);
    for mesh in &result.meshes {
        // Symbol nodes' rows go up with their layer's bucket.
        if plan
            .node(mesh.node)
            .is_some_and(|i| plan.nodes[i].family.is_symbol())
        {
            continue;
        }
        if let Err(e) = upload_mesh(
            gpu,
            renderer,
            plan,
            result,
            mesh,
            &mut resident,
            atlas,
            namespace,
            states,
            warnings,
        ) {
            atlas.release_ramps(gpu, remove_resident(renderer, resident, states));
            return Err(e);
        }
    }
    if let Err(e) = labels.upload(
        gpu,
        renderer,
        plan,
        result,
        &mut resident,
        atlas,
        namespace,
        states,
        tile_zoom,
    ) {
        atlas.release_ramps(gpu, remove_resident(renderer, resident, states));
        return Err(e);
    }
    Ok(resident)
}

/// Evicts the least recently used cached tile the view doesn't use, of
/// source `first` before others'.
fn evict_unused(sources: &mut [SourceState], first: usize) -> Option<ResidentTile> {
    let others = (0..sources.len()).filter(|&s| s != first);
    std::iter::once(first)
        .chain(others)
        .find_map(|s| match &mut sources[s] {
            SourceState::Ready { pyramid, .. } => pyramid.evict_lru(),
            _ => None,
        })
}

/// What a tile result's meshes take in the scene, as [`upload`] inserts
/// them.
fn mesh_sizes(plan: &StylePlan, result: &TileResult) -> Vec<MeshSize> {
    let words = |stream: &Stream| match stream {
        Stream::Core(words) => words.len() as u32,
        Stream::Host { len, .. } => *len,
    };
    result
        .meshes
        .iter()
        .filter(|mesh| {
            !plan
                .node(mesh.node)
                .is_some_and(|i| plan.nodes[i].family.is_symbol())
        })
        .map(|mesh| MeshSize {
            vertices: words(&mesh.vertices),
            indices: words(&mesh.indices),
            props: words(&mesh.props),
            chunks: mesh.chunks.len() as u32,
        })
        .collect()
}

/// What a resident tile costs to keep: the GPU bytes of its meshes and its
/// texture layer, and its picking data.
pub(crate) fn tile_cost(renderer: &Renderer, resident: &ResidentTile) -> u64 {
    let meshes: u64 = resident
        .meshes
        .iter()
        .map(|m| renderer.mesh_bytes(m.mesh) + m.hits.bytes())
        .sum();
    meshes + renderer.tile_bytes(resident.gpu)
}

/// Removes a tile from the scene. Returns the ramps its rows retained.
pub(crate) fn remove_resident(
    renderer: &mut Renderer,
    tile: ResidentTile,
    states: &mut FeatureStates,
) -> Vec<u32> {
    renderer.release_tile(tile.gpu);
    release_resident(tile, states)
}

/// Releases what a tile holds on the CPU, the feature states its features
/// refer to; returns the ramps its rows retained.
fn release_resident(tile: ResidentTile, states: &mut FeatureStates) -> Vec<u32> {
    let mut ramps = Vec::new();
    for mut mesh in tile.meshes {
        for index in mesh.states {
            states.release(index);
        }
        ramps.append(&mut mesh.ramps);
    }
    ramps
}

/// GL JS's aligned projection: the world offset that moves the camera
/// target to a whole pixel of the world at this zoom (half a pixel off for
/// odd viewport sizes, turned with the bearing), so image tiles at whole
/// pixels draw crisp.
fn pixel_alignment(view: &View, projection: &orb_crs::Projection, bearing: f64) -> [f32; 2] {
    let world_per_px = view.world_per_px();
    let bounds = projection.world_bounds();
    let camera = view.camera_world();
    let x = (camera.x - bounds.min.x) / world_per_px;
    let y = (bounds.max.y - camera.y) / world_per_px;
    let viewport = view.viewport();
    let shift = |size: f64| (size.round() as i64 % 2) as f64 / 2.0;
    let (x_shift, y_shift) = (shift(viewport.width), shift(viewport.height));
    let (sin, cos) = (-bearing.to_radians()).sin_cos();
    let dx = x - x.round() + cos * x_shift + sin * y_shift;
    let dy = y - y.round() + cos * y_shift + sin * x_shift;
    let wrap = |d: f64| if d > 0.5 { d - 1.0 } else { d };
    // GL JS's world pixels grow southward; camera-relative world northward.
    [
        (wrap(dx) * world_per_px) as f32,
        (-wrap(dy) * world_per_px) as f32,
    ]
}

/// Placement rows for drawing `draw.tile` in its world copy, clipped to
/// the regions it stands in for.
/// Where a tile's placement rows put it.
#[derive(Clone, Copy, Debug)]
struct Placement<'a> {
    /// Data units per tile side.
    extent: u32,
    /// The surface's wrap period, 0 if it doesn't wrap.
    period: f64,
    /// The camera's style zoom.
    zoom: f64,
    /// A warped tile's frame on the surface, which its mesh spans `extent`
    /// of; tiles on the surface's plane span their own bounds.
    frame: Option<orb_crs::WorldRect>,
    /// A vector tile drawn through a warp's patches, relative to its
    /// anchor (the renderer has the patches).
    series: Option<&'a orb_crs::Series>,
    /// On a globe: the tile's patches on the sphere, and their anchor.
    globe: Option<(TileWarp, glam::DVec3)>,
    /// Whether the tile has a place on the plane (on a globe, a warped
    /// tile without planar patches draws on the sphere alone).
    on_plane: bool,
    /// The style zoom the tile laid out at.
    tile_zoom: f64,
}

fn placements<'a>(
    tms: &TileMatrixSet,
    draw: &'a Draw,
    at: Placement<'_>,
) -> impl Iterator<Item = TileRow> + use<'a> {
    let Placement {
        extent,
        period,
        zoom,
        frame,
        series,
        globe,
        on_plane,
        tile_zoom,
    } = at;
    let bounds = frame.unwrap_or_else(|| tms.tile_bounds(draw.tile));
    let wrap = DVec2::new(f64::from(draw.wrap) * period, 0.0);
    let origin = match series {
        Some(series) => series.anchor + wrap,
        None => DVec2::new(bounds.min.x, bounds.max.y) + wrap,
    };
    let tile = draw.tile;
    // On a globe, regions on the world's edge reach on to the poles the
    // set stops short of (caps, `project.wgsl`). Sets are quadtrees.
    let (first, rows0) = (tms.min_zoom(), tms.matrix_size(tms.min_zoom()).1);
    let caps = match globe {
        Some(_) => {
            let bottom = TileIndex::new(first, 0, rows0 - 1);
            [
                orb_crs::sphere::poles(tms, TileIndex::new(first, 0, 0))[0],
                orb_crs::sphere::poles(tms, bottom)[1],
            ]
        }
        None => [false; 2],
    };
    let rect = move |region: TileIndex| {
        let depth = region.z - tile.z;
        let size = extent >> depth;
        let x = (region.col - (tile.col << depth)) * size;
        let y = (region.row - (tile.row << depth)) * size;
        let (mut min, mut max) = ([x as i16, y as i16], [(x + size) as i16, (y + size) as i16]);
        if caps[0] && region.row == 0 {
            min[1] = i16::MIN;
        }
        if caps[1] && region.row + 1 == rows0 << (region.z - first) {
            max[1] = i16::MAX;
        }
        ClipRect::new(min, max)
    };
    // One placement has four clip rectangles. Split larger partitions into
    // disjoint rows instead of expanding to quadrants that overlap loaded
    // descendants (and render translucent geometry twice).
    let pixel_size = TILE_PX * 2.0f64.powf(zoom.floor() - tile_zoom);
    let x = f64::from(tile.col) + f64::from(draw.wrap) * f64::from(tms.matrix_size(tile.z).0);
    let pattern_origin = [
        (x * pixel_size) as i64 as u32,
        (f64::from(tile.row) * pixel_size) as i64 as u32,
    ];
    let span = match (series, frame) {
        (Some(series), _) => series.scale,
        (None, Some(frame)) => frame.size().x,
        (None, None) => tms.tile_span(tile.z),
    };
    let scale = (span / f64::from(extent)) as f32;
    draw.regions.chunks(4).map(move |regions| {
        let mut clips = [ClipRect::EMPTY; 4];
        for (clip, &region) in clips.iter_mut().zip(regions) {
            *clip = rect(region);
        }
        TileRow {
            extent: extent as f32,
            wrap: draw.wrap,
            pattern_origin,
            origin: origin.to_array(),
            scale,
            zoom: tile_zoom as f32,
            clips,
            px_scale: 1.0,
            drape_zoom: 0.0,
            dem: DemRef::NONE,
            terrain_range: [0.0; 2],
            target: PlacementTarget::Map,
            warp: TileWarp::NONE,
            globe: globe.map(|(warp, origin)| orb_render::RowGlobe {
                warp,
                origin: origin.to_array(),
                on_plane,
            }),
        }
    })
}

/// Web Mercator, which a globe's sphere wraps.
fn mercator_projection() -> orb_crs::Projection {
    *TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// A globe view's frame terms: its matrix, the target, and the horizon
/// plane that depth and the far side's clipping come from.
fn frame_globe(globe: &orb_camera::GlobeView, view: &View) -> orb_render::FrameGlobe {
    let plane = globe.plane.is_some();
    let horizon = if plane {
        // A plane has no horizon: everything faces the eye.
        [0.0, 0.0, 0.0, 1.0]
    } else {
        let radius = orb_crs::sphere::RADIUS;
        let eye = globe.target + globe.eye;
        let distance = eye.length();
        let toward = eye / distance;
        // `facing` is 1 under the eye and 0 at the horizon circle.
        let horizon = radius * radius / distance;
        let scale = 1.0 / (radius - horizon).max(1e-9);
        let n = toward * scale;
        [
            n.x as f32,
            n.y as f32,
            n.z as f32,
            ((globe.target.dot(toward) - horizon) * scale) as f32,
        ]
    };
    orb_render::FrameGlobe {
        view_proj: globe.view_proj.as_mat4().to_cols_array_2d(),
        target: globe.target.to_array(),
        amount: globe.amount as f32,
        horizon,
        metres_per_unit: (globe.metres_per_px / view.world_per_px()) as f32,
        eye: globe.eye.as_vec3().to_array(),
        plane,
    }
}

/// Removes a terrain's DEM page; the renderer's terrain is off.
fn remove_terrain(gpu: &mut dyn Gpu, renderer: &mut Renderer, terrain: crate::terrain::Terrain) {
    if let Some(page) = terrain.page {
        renderer.remove_page(gpu, page);
    }
}

/// GL JS's `calculateFogBlendOpacity`: terrain fog fades in from 60° to
/// 70° of pitch.
fn fog_blend_opacity(pitch: f64) -> f32 {
    ((pitch - 60.0) / 10.0).clamp(0.0, 1.0) as f32
}

/// What a node's paint is evaluated at: a zoom, the bearing (for
/// viewport-anchored hillshade light) and the pattern cross-fade.
#[derive(Clone, Copy)]
struct PaintAt {
    zoom: f64,
    bearing: f64,
    crossfade: Crossfade,
}

/// Writes every node's paint and visibility for `zoom`, if it changed.
fn write_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    (zoom, bearing): (f64, f64),
    crossfade: Crossfade,
    atlas: &mut ImageAtlas,
) -> Vec<String> {
    let mut missing = Vec::new();
    if style.painted.is_none() {
        // What draped nodes draw with at their drape zooms may have changed.
        style.drape.painted.clear();
    }
    if style.painted == Some((zoom, bearing)) && style.painted_fade == Some(crossfade) {
        return missing;
    }
    style.painted = Some((zoom, bearing));
    style.painted_fade = Some(crossfade);
    // Draped nodes on terrain draw at their slots' zooms, which hide them
    // where their layers don't draw ([`write_drape_paint`]).
    let draping = renderer.terrain_on();
    for (index, node) in plan.nodes.iter().enumerate() {
        let state = style.nodes.entry(node.key).or_default();
        let index = index as u32;
        let layer = &plan.style.layers[node.layer];
        if node.family.is_symbol() {
            let lookup = style.states.lookup(index as usize);
            let paint = crate::labels::paint(plan, node, zoom, &state.maxima, lookup);
            renderer.set_visible(index, layer.is_visible_at(zoom) && paint.is_some());
            if let Some(paint) = paint {
                renderer.set_paint(gpu, index, &NodePaint::Symbol(paint));
            }
            continue;
        }
        let visible = layer.is_visible_at(zoom)
            && (layer.ty != LayerType::FillExtrusion
                || layer.get(orb_style::EXTRUSION_OPACITY).evaluate(zoom) > 0.0);
        let draped = draping && node.family.is_draped();
        renderer.set_visible(
            index,
            if draped {
                layer.visibility == orb_style::Visibility::Visible
            } else {
                visible
            },
        );
        if matches!(node.family, Family::Circle | Family::Heatmap) {
            let properties = plan.paint_properties(node).len();
            if state.maxima.len() < properties {
                state.maxima.resize(properties, 0.0);
            }
            let lookup = style.states.lookup(index as usize);
            crate::point::paint(
                gpu,
                renderer,
                plan,
                node,
                index,
                zoom,
                &state.maxima,
                lookup,
            );
            continue;
        }
        let at = PaintAt {
            zoom,
            bearing,
            crossfade,
        };
        let Some(paint) = node_paint(gpu, renderer, (style, plan), index, at, atlas) else {
            continue;
        };
        renderer.set_paint(gpu, index, &paint);
        if node.family.is_image() {
            continue;
        }
        // A pattern the same for every feature is the node's paint: the
        // style lacking it is reported here, as tiles report theirs.
        if let Some(pattern) = plan
            .pattern(node)
            .filter(|p| !matches!(p.kind(), orb_expr::Kind::Source | orb_expr::Kind::Composite))
            && visible
            && style.sprites_loading == 0
            && let Some(name) = pattern.evaluate(zoom.floor())
            && !name.is_empty()
            && !style.icons.contains(&name)
            && style.reported_missing.insert(name.clone())
        {
            missing.push(name);
        }
        // A fill or line layer below layer opacity 1, or blending other
        // than normally, composites once.
        let composite = match layer.ty {
            LayerType::Fill => Some((orb_style::FILL_LAYER_OPACITY, orb_style::FILL_LAYER_BLEND)),
            LayerType::Line => Some((orb_style::LINE_LAYER_OPACITY, orb_style::LINE_LAYER_BLEND)),
            _ => None,
        };
        if let Some((opacity, blend)) = composite {
            let (opacity, blend) = (layer.get(opacity), layer.get(blend));
            renderer.set_layer_composite(
                gpu,
                index,
                node.layer as u32,
                opacity.evaluate(zoom),
                layer_blend(blend.evaluate(zoom)),
            );
        }
    }
    missing
}

/// Writes the draped nodes' paint for the slots drawn at drape zoom
/// `zoom` (decision 0019): evaluated there, with the cross-fade of a zoom
/// at rest, and left out where their layers don't draw. Whether anything
/// changed, and whether anything drawn there turns with the bearing.
fn write_drape_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    zoom: u8,
    bearing: f64,
    atlas: &mut ImageAtlas,
) -> (bool, bool) {
    let at = PaintAt {
        zoom: f64::from(zoom),
        bearing,
        // At an integer zoom at rest: that zoom's images alone.
        crossfade: Crossfade {
            from_scale: 0.5,
            mix: 1.0,
        },
    };
    let (mut changed, mut turns) = (false, false);
    for (index, node) in plan.nodes.iter().enumerate() {
        if !node.family.is_draped() {
            continue;
        }
        let layer = &plan.style.layers[node.layer];
        let paint = layer
            .is_visible_at(at.zoom)
            .then(|| node_paint(gpu, renderer, (style, plan), index as u32, at, atlas))
            .flatten();
        turns |= paint.is_some() && turns_with_bearing(layer, at.zoom);
        changed |= renderer.set_drape_paint(gpu, index as u32, zoom, paint.as_ref());
    }
    (changed, turns)
}

/// Whether a layer drawn at `zoom` turns with the map's bearing: a
/// viewport-anchored hillshade's light, or a viewport-anchored translate.
fn turns_with_bearing(l: &orb_style::Layer, zoom: f64) -> bool {
    let translates = |px: orb_style::Key<[f32; 2]>, anchor: orb_style::Key<TranslateAnchor>| {
        l.get(anchor).evaluate(zoom) == TranslateAnchor::Viewport
            && l.get(px).evaluate(zoom) != [0.0; 2]
    };
    match l.ty {
        LayerType::Hillshade => {
            l.get(orb_style::HILLSHADE_ILLUMINATION_ANCHOR)
                .evaluate(zoom)
                == orb_style::IlluminationAnchor::Viewport
        }
        LayerType::Fill => translates(orb_style::FILL_TRANSLATE, orb_style::FILL_TRANSLATE_ANCHOR),
        LayerType::Line => translates(orb_style::LINE_TRANSLATE, orb_style::LINE_TRANSLATE_ANCHOR),
        _ => false,
    }
}

/// Node `index`'s paint at `at`, for a fill, line, background, extrusion
/// or image node; `None` for an image node that can't draw yet.
fn node_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    index: u32,
    at: PaintAt,
    atlas: &mut ImageAtlas,
) -> Option<NodePaint> {
    let PaintAt {
        zoom,
        bearing,
        crossfade,
    } = at;
    let node = &plan.nodes[index as usize];
    let state = style.nodes.entry(node.key).or_default();
    if node.family.is_image() {
        let unpack = match &style.sources[node.source] {
            SourceState::Ready { endpoint, .. } => endpoint.dem_unpack,
            _ => None,
        };
        let context = (plan, node, index);
        return image::paint(
            gpu,
            renderer,
            context,
            zoom,
            bearing,
            unpack,
            &mut state.ramp,
        );
    }
    let properties = plan.paint_properties(node);
    if state.maxima.len() < properties.len() {
        state.maxima.resize(properties.len(), 0.0);
    }
    let lookup = style.states.lookup(index as usize);
    let maxima = &state.maxima;
    let mut table_mask = 0;
    let mut curves = [Curve::Step; 6];
    let mut numbers = [0.0f32; 6];
    let mut color = [0.0f32; 4];
    for (slot, property) in properties.iter().enumerate() {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = property.curve();
            numbers[slot] = maxima[slot];
        } else if property.is_state_lookup() {
            curves[slot] = property.curve();
            match property {
                PaintProperty::Number(_) => numbers[slot] = property.default_number(zoom),
                _ => color = property.default_color(zoom),
            }
        } else {
            match property {
                PaintProperty::Number(p) => numbers[slot] = p.evaluate(zoom),
                color_property => color = color_property.color(zoom),
            }
        }
    }
    let layer = &plan.style.layers[node.layer];
    let translate = match layer.ty {
        LayerType::Fill => Some((orb_style::FILL_TRANSLATE, orb_style::FILL_TRANSLATE_ANCHOR)),
        LayerType::Line => Some((orb_style::LINE_TRANSLATE, orb_style::LINE_TRANSLATE_ANCHOR)),
        LayerType::FillExtrusion => Some((
            orb_style::EXTRUSION_TRANSLATE,
            orb_style::EXTRUSION_TRANSLATE_ANCHOR,
        )),
        _ => None,
    }
    .map_or(Translate::default(), |(px, anchor)| Translate {
        px: layer.get(px).evaluate(zoom),
        viewport: layer.get(anchor).evaluate(zoom) == TranslateAnchor::Viewport,
    });
    let paint = match node.family {
        Family::Fill | Family::FillOutline => NodePaint::Fill(FillPaint {
            color,
            opacity: numbers[1],
            translate,
            table_mask,
            curves: [curves[0], curves[1]],
            state: lookup,
        }),
        Family::FillExtrusion => {
            let e = layer;
            NodePaint::Extrusion(orb_render::ExtrusionPaint {
                color,
                height: numbers[1],
                base: numbers[2],
                opacity: e.get(orb_style::EXTRUSION_OPACITY).evaluate(zoom),
                translate,
                vertical_gradient: e.get(orb_style::EXTRUSION_VERTICAL_GRADIENT).evaluate(zoom),
                emissive: plan
                    .style
                    .lights
                    .emissive(&plan.style.layers[node.layer].id) as f32,
                table_mask,
                curves: [curves[0], curves[1], curves[2]],
                state: lookup,
            })
        }
        Family::Line => NodePaint::Line(LinePaint {
            color,
            opacity: numbers[1],
            translate,
            width: numbers[2],
            gap_width: numbers[3],
            offset: numbers[4],
            blur: numbers[5],
            table_mask,
            curves,
            dash: None,
            gradient: false,
            state: lookup,
        }),
        image => unreachable!("{image:?} nodes are painted above"),
    };
    let paint = crate::line::paint(
        gpu,
        paint,
        layer,
        atlas,
        style.image_namespace.unwrap(),
        zoom,
        crossfade,
    );
    let paint = pattern_paint(
        paint,
        plan.pattern(node).as_ref(),
        style.image_namespace.unwrap(),
        atlas,
        zoom,
        crossfade,
        match layer.ty {
            LayerType::Line => layer.get(orb_style::LINE_WIDTH).evaluate(zoom.floor()),
            _ => 0.0,
        },
    );
    let paint = if layer.ty == LayerType::Background && plan.pattern(node).is_none() {
        match paint {
            NodePaint::Fill(p) => NodePaint::Background(p),
            _ => unreachable!("solid background"),
        }
    } else {
        paint
    };
    Some(paint)
}

/// The renderer's blend for a v8 layer blend.
fn layer_blend(blend: orb_style::LayerBlend) -> orb_render::Blend {
    use orb_render::Blend;
    use orb_style::LayerBlend;
    match blend {
        LayerBlend::Normal => Blend::Normal,
        LayerBlend::Multiply => Blend::Multiply,
        LayerBlend::Screen => Blend::Screen,
        LayerBlend::Overlay => Blend::Overlay,
        LayerBlend::Color => Blend::Color,
        LayerBlend::Luminosity => Blend::Luminosity,
        LayerBlend::PlusLighter => Blend::PlusLighter,
    }
}

fn bend_of(cells: u32, [north, south]: [bool; 2]) -> crate::Bend {
    crate::Bend {
        cells,
        north,
        south,
    }
}

/// Cells per side of an image source's grid: GL JS's granularity for warped
/// image quads.
const IMAGE_SOURCE_CELLS: u32 = 16;

/// A series' planar patches as the scene takes them: cells per side and
/// their words.
fn series_patches(series: &orb_crs::Series) -> (u32, Vec<f32>) {
    let patches = series
        .patches
        .iter()
        .flat_map(|p| p.points.iter().flat_map(|v| [v.x as f32, v.y as f32]))
        .collect();
    (series.cells, patches)
}

/// An image tile's mesh: on a globe, a grid in the tile's own space that
/// the sphere and the plane's patches bend; on another plane, the warp's
/// mesh; else `None` for its quad. `Err` when nothing of it reaches the
/// surface.
fn image_mesh(
    globe_warps: &mut crate::globe::GlobeWarps,
    globe: bool,
    warp: Option<&mut WarpState>,
    endpoint: &TileEndpoint,
    tile: TileIndex,
) -> Result<Option<Arc<image::Warped>>, ()> {
    let tms = &endpoint.tms;
    // An image source: its quad's warp at the vertices of a grid over the
    // image, in its one tile; on a globe, as fine as the sphere bends.
    if let Some(quad) = &endpoint.quad {
        let cells = if globe {
            globe_warps.grid(tms, tile).max(IMAGE_SOURCE_CELLS)
        } else {
            IMAGE_SOURCE_CELLS
        };
        return Ok(Some(Arc::new(image::Warped::quad(
            tms.tile_bounds(tile),
            quad,
            cells,
        ))));
    }
    match warp {
        Some(warp) if !globe => image::Warped::new(&warp.warp, tile)
            .map(|mesh| Some(Arc::new(mesh)))
            .ok_or(()),
        _ => Ok(globe_grid(globe_warps, globe, tms, tile)),
    }
}

/// An image tile's mesh on a globe: a grid fine enough for the sphere to
/// bend, or `None` for its quad.
fn globe_grid(
    globe_warps: &mut crate::globe::GlobeWarps,
    globe: bool,
    tms: &TileMatrixSet,
    tile: TileIndex,
) -> Option<Arc<image::Warped>> {
    if !globe {
        return None;
    }
    let (cells, poles) = (
        globe_warps.grid(tms, tile),
        orb_crs::sphere::poles(tms, tile),
    );
    (cells > 1 || poles.contains(&true))
        .then(|| Arc::new(image::Warped::grid(tms.tile_bounds(tile), cells, poles)))
}

/// Cells per side of the background's grid on a globe: fine enough for a
/// whole-world tile.
const BACKGROUND_GLOBE_CELLS: u32 = 64;

/// One scene tile with a tile-sized quad for every background layer, or on
/// a globe a grid the sphere bends.
fn make_background(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &crate::style::StylePlan,
    globe: bool,
) -> Option<GpuTile> {
    let nodes: Vec<u32> = (plan.nodes.iter().enumerate())
        .filter(|(_, n)| n.source == usize::MAX)
        .map(|(i, _)| i as u32)
        .collect();
    if nodes.is_empty() {
        return None;
    }
    let cells = if globe { BACKGROUND_GLOBE_CELLS } else { 1 };
    let n = cells + 1;
    let step = 8192 / cells;
    let mut vertices: Vec<u32> = (0..n * n)
        .flat_map(|k| [(k % n * step) | (k / n * step) << 16, 0])
        .collect();
    let mut triangles: Vec<[u32; 3]> = (0..cells * cells)
        .flat_map(|c| {
            let k = c / cells * n + c % cells;
            [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
        })
        .collect();
    if globe {
        // Caps: the top and bottom edges fanned out to the poles, which
        // only the world's edge rows reach (`project.wgsl`).
        for (row, pole) in [
            (0, orb_layout::fill::NORTH_POLE_Y),
            (cells, orb_layout::fill::SOUTH_POLE_Y),
        ] {
            let tip = (vertices.len() / 2) as u32;
            vertices.extend([u32::from(pole as u16) << 16, 0]);
            triangles.extend((0..cells).map(|i| [row * n + i, row * n + i + 1, tip]));
        }
    }
    let chunks = triangles
        .chunks(256)
        .enumerate()
        .map(|(c, run)| {
            let rows = (c * 256 / 2) as u32 / cells;
            let last = ((c * 256 + run.len() - 1) / 2) as u32 / cells;
            orb_layout::Chunk {
                sort_key: None,
                first_index: (c * 256 * 3) as u32,
                index_count: (run.len() * 3) as u32,
                // The caps' triangles come last and span every row.
                bounds: if c * 256 >= (cells * cells * 2) as usize {
                    orb_layout::TileBox {
                        min: [0, i16::MIN],
                        max: [8192, i16::MAX],
                    }
                } else {
                    orb_layout::TileBox {
                        min: [0, (rows * step) as i16],
                        max: [8192, ((last + 1) * step).min(8192) as i16],
                    }
                },
            }
        })
        .collect();
    let mesh = orb_layout::Mesh {
        vertices,
        indices: triangles.into_iter().flatten().collect(),
        chunks,
    };
    // One quad (or grid) for every covered tile: no source's tile.
    let key = TileKey {
        source: u32::MAX,
        tile: TileIndex::new(0, 0, 0),
    };
    let tile = renderer
        .add_tile(gpu, key, None)
        .expect("a tile without patches fits");
    let data = MeshData::new(&mesh, &[], 0);
    for node in nodes {
        renderer
            .add_mesh(gpu, tile, (node, 8192), &data, &Default::default())
            .expect("background mesh fits");
    }
    Some(tile)
}

fn pattern_paint(
    paint: NodePaint,
    property: Option<&orb_style::PropertyRef<'_, Option<String>>>,
    namespace: u64,
    atlas: &mut ImageAtlas,
    zoom: f64,
    crossfade: Crossfade,
    floor_width: f32,
) -> NodePaint {
    let Some(property) = property else {
        return paint;
    };
    let per_feature = matches!(
        property.kind(),
        orb_expr::Kind::Source | orb_expr::Kind::Composite
    );
    let image = |atlas: &mut ImageAtlas, z| {
        property
            .evaluate(z)
            .and_then(|name| atlas.resolve(namespace, &name).ok())
            .unwrap_or(0)
    };
    let mut pattern = PatternPaint {
        from: image(
            atlas,
            zoom.floor()
                + if crossfade.from_scale > 1.0 {
                    -1.0
                } else {
                    1.0
                },
        ),
        to: image(atlas, zoom.floor()),
        from_scale: crossfade.from_scale,
        to_scale: 1.0,
        mix: crossfade.mix,
        floor_width: 0.0,
    };
    match paint {
        NodePaint::Fill(mut p) => {
            if per_feature {
                p.table_mask |= 64;
            }
            NodePaint::FillPattern(p, pattern)
        }
        NodePaint::Extrusion(mut p) => {
            if per_feature {
                p.table_mask |= 64;
            }
            NodePaint::ExtrusionPattern(p, pattern)
        }
        NodePaint::Line(mut p) => {
            pattern.floor_width = floor_width;
            if per_feature {
                p.table_mask |= 64;
            }
            NodePaint::LinePattern(p, pattern)
        }
        _ => unreachable!("base paint"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use orb_host::FetchDone;
    use orb_jobs::Inline;

    use super::*;
    use crate::PoolJobs;

    /// S5: a result laid out before an edit to a symbol layer brings
    /// neither its meshes nor its bucket.
    #[test]
    fn stale_results_drop_symbol_buckets_with_their_meshes() {
        let (worker, mut result) = crate::testing::symbol_tile();
        assert!(!result.symbols.is_empty());
        let text = result.symbols[0].layer * 2 + 1;
        drop_stale(&mut result, worker.plan(), &[text]);
        assert!(result.symbols.is_empty());
        assert!(result.meshes.iter().all(|m| m.node != text));
    }

    #[test]
    fn pattern_crossfade_tracks_integer_crossings_and_settles() {
        let mut h = ZoomHistory::default();
        assert_eq!(h.update(4.5, 0.0).mix, 1.0);
        let up = h.update(5.25, 10.0);
        assert_eq!(
            up,
            Crossfade {
                from_scale: 2.0,
                mix: 0.25
            }
        );
        assert_eq!(h.update(5.25, 160.0).mix, 0.625);
        assert_eq!(h.update(5.25, 310.0).mix, 1.0);
        let down = h.update(4.75, 320.0);
        assert_eq!(
            down,
            Crossfade {
                from_scale: 0.5,
                mix: 0.25
            }
        );
        assert_eq!(h.update(4.75, 620.0).mix, 1.0);
    }

    /// A fetch that holds requests until the test answers them.
    #[derive(Default)]
    struct HeldFetch {
        pending: Mutex<Vec<(String, FetchDone)>>,
    }

    impl Fetch for HeldFetch {
        fn get(&self, url: &str, done: FetchDone) {
            self.pending.lock().unwrap().push((url.to_string(), done));
        }

        fn get_range(&self, url: &str, _: u64, _: u64, done: FetchDone) {
            self.get(url, done);
        }
    }

    struct NoClock;
    impl orb_host::Clock for NoClock {
        fn now(&self) -> f64 {
            0.0
        }
    }

    struct NoDecode;
    impl ImageDecode for NoDecode {
        fn decode(&self, _: Vec<u8>, _: orb_host::ImageDone) {
            panic!("unexpected image");
        }
    }

    fn style(name: &str) -> String {
        format!(r#"{{"version": 8, "name": "{name}", "sources": {{}}, "layers": []}}"#)
    }

    #[test]
    fn centred_zoom_preserves_the_target_and_fresh_tile_cover() {
        let fetch = Arc::new(HeldFetch::default());
        let services = Services {
            clock: Arc::new(NoClock),
            image_decode: Arc::new(NoDecode),
            images: Default::default(),
            jobs: PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::jobs::tests::NoImages),
                Arc::new(Inline),
            ),
            fetch,
            wake: Arc::new(|| {}),
        };
        let target = orb_crs::Geodetic::new(139.74, 35.685);
        let mercator = *Surface::default().projection();
        let viewport = Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let center = DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        for (bearing, pitch) in [(0.0, 0.0), (30.0, 45.0)] {
            let camera = CameraState {
                bearing,
                pitch,
                ..CameraState::new(target, 15.0, &mercator)
            };
            let mut map = Map::new(services.clone(), Some(camera));
            for delta in [-0.5; 6].into_iter().chain([0.5; 6]) {
                let zoom = map.zoom() + delta;
                let surface = map.surface();
                let mut camera = *map.camera();
                camera.zoom_around(&surface, viewport, center, zoom);
                map.jump_to_state(camera);
                let fresh = CameraState {
                    bearing,
                    pitch,
                    ..CameraState::new(target, zoom, &mercator)
                };
                assert_eq!(
                    *map.camera(),
                    fresh,
                    "a centred zoom must not move the target"
                );
                assert_eq!(
                    map.view(viewport).cover(map.surface.tms(), 512.0, 14),
                    View::new(&fresh, viewport, &map.surface()).cover(map.surface.tms(), 512.0, 14)
                );
            }
        }
    }

    #[test]
    fn fallback_placements_keep_more_than_four_regions_exact() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let draw = Draw {
            tile: TileIndex::new(0, 0, 0),
            wrap: 1,
            regions: (0..8).map(|col| TileIndex::new(3, col, 0)).collect(),
        };
        let at = Placement {
            extent: 4096,
            period: 100.0,
            zoom: 3.0,
            frame: None,
            series: None,
            globe: None,
            on_plane: true,
            tile_zoom: 0.0,
        };
        let rows: Vec<_> = placements(&tms, &draw, at).collect();
        assert_eq!(rows.len(), 2);
        for (col, clip) in rows.iter().flat_map(|row| row.clips).enumerate() {
            assert_eq!(
                clip,
                ClipRect::new([(col * 512) as i16, 0], [((col + 1) * 512) as i16, 512])
            );
        }
        assert_eq!(rows[0].origin, rows[1].origin);
        assert_eq!(rows[0].scale, rows[1].scale);
    }

    #[test]
    fn a_style_set_directly_supersedes_one_still_loading() {
        let fetch = Arc::new(HeldFetch::default());
        let services = Services {
            clock: Arc::new(NoClock),
            image_decode: Arc::new(NoDecode),
            images: Default::default(),
            jobs: PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::jobs::tests::NoImages),
                Arc::new(Inline),
            ),
            fetch: fetch.clone(),
            wake: Arc::new(|| {}),
        };
        let mut map = Map::new(services, None);
        map.load_style_url("https://example.com/a.json");
        map.set_style_json(&style("b")).unwrap();
        let (_, done) = fetch.pending.lock().unwrap().pop().unwrap();
        done(Ok(style("a").into_bytes()));
        while let Ok(Message::Style {
            request,
            url,
            scene,
            result,
        }) = map.receiver.try_recv()
        {
            map.apply_style(request, &url, scene, result);
        }
        let name = map.style.as_ref().unwrap().worker.plan().style.name.clone();
        assert_eq!(name.as_deref(), Some("b"));
    }

    #[test]
    fn scene_documents_choose_the_surface_and_load_their_style() {
        let fetch = Arc::new(HeldFetch::default());
        let services = Services {
            clock: Arc::new(NoClock),
            image_decode: Arc::new(NoDecode),
            images: Default::default(),
            jobs: PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::jobs::tests::NoImages),
                Arc::new(Inline),
            ),
            fetch: fetch.clone(),
            wake: Arc::new(|| {}),
        };
        let mut map = Map::new(services, None);
        map.load_style_url("https://example.com/maps/polar.json");
        let deliver = |map: &mut Map, body: &str| {
            let (url, done) = fetch.pending.lock().unwrap().pop().unwrap();
            done(Ok(body.as_bytes().to_vec()));
            while let Ok(Message::Style {
                request,
                url,
                scene,
                result,
            }) = map.receiver.try_recv()
            {
                map.apply_style(request, &url, scene, result);
            }
            url
        };
        deliver(
            &mut map,
            r#"{"style": "styles/b.json", "projection": {"type": "EPSG:3413"},
                "sources": {"s": {"tileMatrixSet": "UPSArcticWGS84Quad"}}}"#,
        );
        // The surface comes with the scene's style, as one scene.
        assert_eq!(map.projection().descriptor().kind, "web-mercator");
        // The scene's relative style URL loads next, onto its surface.
        assert_eq!(
            deliver(&mut map, &style("b")),
            "https://example.com/maps/styles/b.json"
        );
        assert_eq!(map.projection().descriptor().kind, "EPSG:3413");
        assert_eq!(
            map.surface.source_tms("s", false).id(),
            "UPSArcticWGS84Quad"
        );
        // A plain v8 style is the default scene again.
        map.set_style_json(&style("c")).unwrap();
        assert_eq!(map.projection().descriptor().kind, "web-mercator");
    }
}

#[cfg(test)]
mod video_tests {
    use orb_hal_wgpu::WgpuGpu;
    use orb_jobs::Inline;

    use super::*;
    use crate::{ImagePixels, PoolJobs, TileImage, VideoRequest};

    struct NoFetch;
    impl Fetch for NoFetch {
        fn get(&self, url: &str, _: orb_host::FetchDone) {
            panic!("unexpected fetch of {url}");
        }
        fn get_range(&self, url: &str, _: u64, _: u64, _: orb_host::FetchDone) {
            panic!("unexpected fetch of {url}");
        }
    }

    struct NoDecode;
    impl ImageDecode for NoDecode {
        fn decode(&self, _: Vec<u8>, _: orb_host::ImageDone) {
            panic!("unexpected image");
        }
    }

    struct Clock(std::time::Instant);
    impl orb_host::Clock for Clock {
        fn now(&self) -> f64 {
            self.0.elapsed().as_secs_f64() * 1000.0
        }
    }

    fn frame(rgba: [u8; 4]) -> TileImage {
        TileImage {
            width: 4,
            height: 2,
            pixels: ImagePixels::Core(rgba.repeat(8)),
        }
    }

    fn requests(map: &mut Map) -> Vec<(String, VideoRequest)> {
        map.take_events()
            .into_iter()
            .filter_map(|e| match e {
                MapEvent::Video { source, request } => Some((source, request)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn video_frames_fill_their_tile_and_replace_it_in_place() {
        let Ok(mut gpu) = pollster::block_on(WgpuGpu::offscreen()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let fetch = Arc::new(NoFetch);
        let mut map = Map::new(
            Services {
                clock: Arc::new(Clock(std::time::Instant::now())),
                image_decode: Arc::new(NoDecode),
                images: Default::default(),
                jobs: PoolJobs::new(
                    fetch.clone(),
                    Arc::new(crate::jobs::tests::NoImages),
                    Arc::new(Inline),
                ),
                fetch,
                wake: Arc::new(|| {}),
            },
            None,
        );
        map.set_plays_video(true);
        // The video covers the whole view.
        map.set_style_json(
            r#"{"version": 8, "center": [0, 0], "zoom": 2,
                "sources": {"v": {"type": "video", "urls": ["a.mp4", "a.webm"],
                    "coordinates": [[-120, 70], [120, 70], [120, -70], [-120, -70]]}},
                "layers": [{"id": "v", "type": "raster", "source": "v",
                    "paint": {"raster-fade-duration": 0}}]}"#,
        )
        .unwrap();
        let format = TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let target = gpu.create_texture(&orb_hal::TextureDesc {
            label: "video test",
            width: 64,
            height: 64,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: format.color,
            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let viewport = Viewport {
            width: 64.0,
            height: 64.0,
        };
        let mut renderer = map.create_renderer(&mut gpu, format);
        let centre = |map: &mut Map, gpu: &mut WgpuGpu, renderer: &mut Renderer| {
            let mut idle = false;
            for _ in 0..200 {
                idle = map.render(gpu, renderer, target, viewport, 1.0, 0.0).idle;
                if idle {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let pixels = pollster::block_on(gpu.read_texture(target));
            let at = (32 * 64 + 32) * 4;
            (idle, [pixels[at], pixels[at + 1], pixels[at + 2]])
        };
        // Without a frame, the tile loads; the host is asked to play.
        for _ in 0..3 {
            assert!(
                !map.render(&mut gpu, &mut renderer, target, viewport, 1.0, 0.0)
                    .idle
            );
        }
        let urls = vec!["a.mp4".to_string(), "a.webm".to_string()];
        assert_eq!(
            requests(&mut map),
            [("v".into(), VideoRequest::Open { urls })]
        );
        map.video_frame("v", frame([255, 0, 0, 255])).unwrap();
        assert_eq!(
            centre(&mut map, &mut gpu, &mut renderer),
            (true, [255, 0, 0])
        );
        map.video_frame("v", frame([0, 0, 255, 255])).unwrap();
        assert_eq!(
            centre(&mut map, &mut gpu, &mut renderer),
            (true, [0, 0, 255])
        );
        map.control_video("v", VideoRequest::Pause).unwrap();
        assert!(map.control_video("w", VideoRequest::Play).is_err());
        assert!(map.video_frame("w", frame([0; 4])).is_err());
        assert_eq!(requests(&mut map), [("v".into(), VideoRequest::Pause)]);
        // Removing the source closes it.
        let remove = r#"[["removeLayer", "v"], ["removeSource", "v"]]"#;
        map.edit(&crate::Transaction::from_commands(remove).unwrap())
            .unwrap();
        centre(&mut map, &mut gpu, &mut renderer);
        assert_eq!(requests(&mut map), [("v".into(), VideoRequest::Close)]);
    }

    #[test]
    fn video_sources_fail_on_hosts_without_video() {
        let mut videos = crate::video::Videos::default();
        let mut events = Vec::new();
        let style = TileWorker::new(
            r#"{"version": 8, "sources": {"v": {"type": "video", "urls": ["a.mp4"],
                "coordinates": [[0, 1], [1, 1], [1, 0], [0, 0]]}},
                "layers": [{"id": "v", "type": "raster", "source": "v"}]}"#,
        )
        .unwrap();
        videos.sync(Some(style.plan()), &mut events);
        assert!(events.is_empty());
        let key = style.plan().sources[0].key;
        assert!(videos.want(key, TileIndex::new(0, 0, 0)));
        let work = videos.work();
        assert_eq!(work[0].failed, Some(crate::NO_VIDEO));
        assert_eq!(work[0].wanted.len(), 1);
    }
}
