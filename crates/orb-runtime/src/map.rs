use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use orb_camera::{CameraState, View, Viewport};
use orb_crs::{TileIndex, TileMatrixSet};

use crate::surface::Surface;
use orb_hal::{Gpu, TextureId};
use orb_host::{Fetch, FetchError, ImageDecode};
use orb_render::GpuTile;
use orb_render::ImageAtlas;
use orb_render::{
    Family, FrameBlock, FrameStats, PageId, RenderNode, Renderer, SceneCapacity, SceneStats,
    TargetFormat, micros,
};
use orb_source::Draw;
use orb_style::SourceKind;
use orb_style::v8::{StyleDoc, Transaction};

use crate::MapEvent;
use crate::frame::*;
use crate::jobs::{Jobs, TileResult};
use crate::meshes::*;
use crate::resident::*;
use crate::state::FeatureStates;
use crate::style::StylePlan;
use crate::worker::{StyleUpdate, TileRequest, TileWorker};
use orb_source::endpoint::TileEndpoint;

/// Logical pixel size of vector/background tiles and pattern origins.
pub(crate) const TILE_PX: f64 = 512.0;

/// Milliseconds a frame spends applying arrived tiles before it leaves the
/// rest for the next frame (upload budget at 60 Hz).
pub(crate) const TILE_APPLY_MS: f64 = 1.5;

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
    /// has all its jobs applied.
    pub generation: u64,
    /// Nothing is loading: the style, its sources and every tile the view
    /// needs have arrived, so the frame is complete.
    pub idle: bool,
    /// As `idle`, but fades may still run: labels are placed for this
    /// view and cross-fades need only time (GL JS `loaded()`).
    pub loaded: bool,
    /// GPU memory by pool, and the tile cache.
    pub memory: Memory,
    /// Where the frame's CPU time went.
    pub times: FrameTimes,
    /// What [`Map::prepare`] spent since the last frame drawn, summed.
    pub prepared: FrameTimes,
    /// Tiles whose data arrived (vector, raster and DEM; not loads given
    /// up, missing or failed) since the last frame drawn, in its
    /// preparations or in it.
    pub tiles_arrived: u32,
    /// Only compiling pipelines keep the frame from being complete: a
    /// host whose GPU wakes it when a pipeline compiles needs no frames
    /// until then.
    pub compiling: bool,
}

/// Where a frame's CPU time went on this thread, in microseconds of the
/// services' clock. What it doesn't name (paint, the camera, picking) is
/// the rest of `total`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTimes {
    /// The whole frame.
    pub total: u32,
    /// Applying what arrived (tile results, edits, styles, sprites), but
    /// for uploads and label commits.
    pub arrivals: u32,
    /// Arrived tiles' meshes, rows and images uploaded, and room made for
    /// them.
    pub upload: u32,
    /// The terrain's DEM tiles and cover.
    pub terrain: u32,
    /// The sources' tiles for the view: covered, asked for, retired.
    pub plan: u32,
    /// The terrain's drape slots chosen and their placements written.
    pub drape: u32,
    /// The view's tiles put in the scene, their slots written.
    pub place: u32,
    /// Placement on this thread: building the frame the placement lane
    /// places and applying its commits, not the placement itself, which
    /// is the lane's (inline on a host without a placement thread or
    /// worker).
    pub placement: u32,
    /// The renderer's share: the cull's layout, bundles, drape passes and
    /// encoding.
    pub render: orb_render::RenderTimes,
}

impl FrameTimes {
    /// Adds a prepared frame's times, which draw nothing.
    fn add_prepared(&mut self, t: &FrameTimes) {
        self.total += t.total;
        self.arrivals += t.arrivals;
        self.upload += t.upload;
        self.terrain += t.terrain;
        self.plan += t.plan;
        self.drape += t.drape;
        self.place += t.place;
        self.placement += t.placement;
    }
}

/// The map's memory against its budget, in bytes.
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
    /// The terrain's drape atlas, mips included.
    pub drape_atlas: u64,
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

pub(crate) enum Message {
    /// A placement committed, or failed, for the labels of `epoch`.
    Placed {
        epoch: u64,
        committed: Result<orb_symbol::Committed, String>,
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
        /// Where the document came from, for its relative URLs.
        url: String,
        result: Result<String, FetchError>,
    },
    /// A sprite's JSON: the images icon layout needs.
    SpriteIcons {
        generation: u64,
        /// The style's sprite load it answers ([`LoadedStyle::sprite_epoch`]).
        epoch: u64,
        /// The sprite's index in the style's.
        index: usize,
        result: Result<Vec<crate::sprite::SpriteIcon>, String>,
    },
    /// A sprite's pixels, after its icons.
    Sprite {
        generation: u64,
        epoch: u64,
        index: usize,
        result: Result<Vec<crate::sprite::SpriteImage>, String>,
    },
    TileJson {
        generation: u64,
        /// The source's key.
        source: u32,
        result: Result<String, FetchError>,
        /// A PMTiles archive's URL and first bytes, for the jobs.
        head: Option<(String, Vec<u8>)>,
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
        /// Boxed: the other messages are much smaller.
        result: Box<Result<TileResult, String>>,
    },
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
    /// The paint epoch the node's paint was last written in, if its paint
    /// is the same at every zoom and bearing ([`camera_free`]): written
    /// once per epoch, not per frame.
    pub(crate) painted: Option<u64>,
}

impl NodeState {
    /// A node new at revision `epoch`: tiles laid out before are stale.
    pub(crate) fn new(plan: &StylePlan, node: &crate::style::NodePlan, epoch: u64) -> Self {
        let maxima = vec![0.0; plan.paint_properties(node).len()];
        let (layout_epoch, rows_epoch) = (epoch, epoch);
        Self {
            maxima,
            layout_epoch,
            rows_epoch,
            ..Self::default()
        }
    }
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
    pub(crate) paint: crate::paint::PaintState,
    pub(crate) image_namespace: Option<u64>,
    pub(crate) sprites_started: bool,
    /// Sprites whose JSON hasn't arrived: icon layout waits for them.
    pub(crate) sprites_loading: u32,
    /// Sprites whose pixels haven't arrived: the map isn't loaded until
    /// they have.
    pub(crate) sprite_pixels_loading: u32,
    /// Each sprite's image names, by its index in the style's, to take
    /// back if its pixels fail.
    pub(crate) sprite_icon_names: Vec<Vec<String>>,
    /// Whether labels laid out again (an edit's) took their icons' atlas
    /// rectangles before the sprites' pixels were in: they lay out once
    /// more when the pixels are.
    pub(crate) relayout_after_pixels: bool,
    /// Counts the style's sprite loads: replies to an earlier one, which
    /// a new `sprite` superseded, are dropped.
    pub(crate) sprite_epoch: u64,
    /// The names the style's sprites gave images.
    pub(crate) sprite_names: HashSet<String>,
    /// While another sprite loads in place of the old one, the old one's
    /// names: those the new one lacks leave once it is in.
    pub(crate) stale_sprites: Option<HashSet<String>>,
    /// Whether the glyphs labels start with were asked for
    /// ([`TileWorker::first_glyphs`]).
    pub(crate) glyphs_fetched: bool,
    /// Sprite metadata loaded so far, and the images the host added; the
    /// jobs get it once every sprite has resolved, before tiles with symbol
    /// layers are requested.
    pub(crate) icons: crate::SpriteIcons,
    /// Image edits for the next frame's atlas.
    pub(crate) image_edits: Vec<crate::style_image::AtlasEdit>,
    /// The images the host added, as they are now: a style installed
    /// afresh (after a device loss) adds them again.
    pub(crate) added_images: std::collections::BTreeMap<String, Arc<crate::StyleImage>>,
    /// Image names reported missing since they were last added.
    pub(crate) reported_missing: HashSet<String>,
    /// The scene tile of the background layers' quads.
    pub(crate) background: Option<GpuTile>,
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
    /// Whether the background and image tiles' meshes are grids for a
    /// globe to bend.
    pub(crate) globe_meshes: bool,
}

impl LoadedStyle {
    /// Whether `build`'s labels would take image rectangles from the atlas
    /// before the sprites' pixels are in it. Labels of text alone take
    /// none, so they don't wait.
    pub(crate) fn awaits_sprite_pixels(&self, build: &crate::jobs::TileResult) -> bool {
        self.sprite_pixels_loading > 0 && build.symbols.iter().any(|s| !s.images.is_empty())
    }
}

/// What tiles and styles let go of between frames, released from the
/// renderer at the next.
#[derive(Debug, Default)]
pub(crate) struct Retired {
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

/// How a map renders: interactively, or the still output of
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
    /// The middle tile of a grid, the camera's view: `Still`, with labels
    /// placed in tile scope over the grid's tiles listed, so that tiles
    /// drawn apart agree on the labels that cross their edges. Sources
    /// whose content crosses tile edges load the whole grid.
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

/// A map: a style, a camera and the tiles they need, drawn by
/// [`render`](Self::render).
pub struct Map {
    /// Interactive, a still picture or a tile ([`Map::set_mode`]).
    pub(crate) mode: RenderMode,
    /// Placement and the scene side of labels.
    pub(crate) labels: crate::labels::Labels,
    /// Counts the [`Labels`](crate::labels::Labels) a GPU reset replaced.
    pub(crate) label_epoch: u64,
    /// The last frame's picking state; queries project from it.
    pub(crate) picking: std::cell::RefCell<crate::query::Picking>,
    /// This frame's placement and upload times so far, in milliseconds,
    /// and the tiles that arrived.
    pub(crate) placement_ms: f64,
    pub(crate) upload_ms: f64,
    pub(crate) tiles_arrived: u32,
    /// What preparing spent since the last frame drawn.
    prepared: (FrameTimes, u32),
    /// Arrived tiles wait for the next frame: this one spent its budget.
    pub(crate) tiles_deferred: bool,
    /// Tiles that arrived when the scene had no room for them, while other
    /// tiles' jobs were in flight: those tiles' arrivals free the stand-ins
    /// they replace, so these try again on the next frames.
    pub(crate) waiting: Vec<WaitingTile>,
    /// Chunks the last frame's cull left out for want of room.
    pub(crate) candidates_dropped: u32,
    pub(crate) services: Services,
    pub(crate) videos: crate::video::Videos,
    pub(crate) host_tiles: crate::host_tiles::HostTiles,
    /// The camera's only writer: the app's moves, the style's camera and
    /// gestures all go through it.
    pub(crate) camera: orb_camera::Camera,
    pub(crate) input: orb_camera::input::Interaction,
    /// The camera and terrain revisions the last frame saw.
    pub(crate) camera_seen: u64,
    pub(crate) terrain_seen: u64,
    /// Terrain tiles hang skirts ([`Map::set_terrain_skirts`]).
    pub(crate) terrain_skirts: bool,
    /// The previews styles parse with ([`Map::set_previews`]).
    previews: orb_style::Previews,
    pub(crate) antialias_3d: bool,
    /// The plane the map draws on and its sources' tile matrix sets: the
    /// style's, planned for its tiles.
    pub(crate) surface: Surface,
    /// The first frame's time: frames' time counts from it (fades,
    /// transitions and the shaders' time run on frame time).
    pub(crate) born: Option<f64>,
    /// The time of the frame being drawn, or the last one.
    pub(crate) frame_time: f64,
    /// When the last raster tile to arrive finishes fading in.
    pub(crate) image_fades_until: f64,
    /// Paint edits easing in.
    pub(crate) transitions: crate::transitions::Transitions,
    /// The latest `load_style_url` request; earlier ones are ignored.
    pub(crate) style_request: u64,
    /// Why the latest style request failed: nothing will load until
    /// another style replaces it.
    pub(crate) style_error: Option<String>,
    pub(crate) generation: u64,
    pub(crate) tile_generation: u64,
    pub(crate) style: Option<LoadedStyle>,
    /// The renderer whose tiles the map holds, by its id: a frame with
    /// another one forgets them.
    pub(crate) renderer: Option<u64>,
    /// What tiles and styles let go of between frames, for the next frame
    /// to release from the renderer.
    pub(crate) retired: Retired,
    /// The renderer's nodes are of a replaced style.
    pub(crate) nodes_stale: bool,
    pub(crate) capacity: SceneCapacity,
    /// The GPU memory budget.
    pub(crate) budget: u64,
    /// The last style id given to the jobs.
    pub(crate) style_id: u64,
    /// Counts style revisions: every new style and every edit.
    pub(crate) version: u64,
    /// Outstanding relayout and column jobs, per edit generation.
    pub(crate) pending: BTreeMap<u64, u32>,
    pub(crate) counts: JobCounts,
    pub(crate) clear_color: [f64; 4],
    pub(crate) sender: Sender<Message>,
    pub(crate) receiver: Receiver<Message>,
    pub(crate) warnings: Vec<String>,
    /// The warnings given since the style's last revision: each is given
    /// once.
    pub(crate) warned: HashSet<String>,
    pub(crate) events: crate::events::EventQueue,
    /// Terrain pages of a replaced style, to destroy with the next
    /// frame's GPU.
    pub(crate) retired_pages: Vec<PageId>,
    pub(crate) retired_images: Vec<u64>,
    pub(crate) retired_ramps: Vec<u32>,
    /// Tiles' patches on the globe.
    pub(crate) globe_warps: crate::globe::GlobeWarps,
    /// Terrains made so far: each new one's id.
    pub(crate) terrains: u64,
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
    surface: crate::surface::SurfaceTiles,
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
        source_diffs: Vec::new(),
        source_json: doc
            .source_keys()
            .enumerate()
            .map(|(i, k)| {
                let id = doc.style().sources[i].id.clone();
                (k.0, id, Arc::new(doc.source_json(i).clone()))
            })
            .collect(),
        state: doc.global_state(),
        previews: doc.style().previews,
        glyphs: doc.style().glyphs.clone(),
        font_faces: doc.style().font_faces.clone(),
        local_ideographs,
        surface,
    }
}

/// A revision of `doc`'s plan, keeping `previous`'s source data.
pub(crate) fn revise(
    previous: Option<&TileWorker>,
    doc: &StyleDoc,
    new_data: &[u32],
    revision: u64,
    local_ideographs: Option<String>,
    surface: crate::surface::SurfaceTiles,
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
        Self {
            mode: RenderMode::Continuous,
            born: None,
            frame_time: 0.0,
            image_fades_until: f64::NEG_INFINITY,
            picking: Default::default(),
            placement_ms: 0.0,
            upload_ms: 0.0,
            tiles_arrived: 0,
            prepared: Default::default(),
            tiles_deferred: false,
            waiting: Vec::new(),
            candidates_dropped: 0,
            services,
            videos: Default::default(),
            host_tiles: Default::default(),
            camera: store,
            input: orb_camera::input::Interaction::default(),
            camera_seen: 0,
            terrain_seen: 0,
            globe_warps: crate::globe::GlobeWarps::new(*surface.projection()),
            surface,
            transitions: Default::default(),
            style_request: 0,
            style_error: None,
            generation: 0,
            tile_generation: 0,
            style: None,
            renderer: None,
            retired: Retired::default(),
            terrain_skirts: true,
            previews: orb_style::Previews::NONE,
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
            warned: HashSet::new(),
            events: Default::default(),
            retired_pages: Vec::new(),
            retired_images: Vec::new(),
            retired_ramps: Vec::new(),
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
    /// `fadeDuration`, 300 by default; 0 for still output). As in GL JS,
    /// labels don't fade until the map is first idle.
    pub fn set_fade_duration(&mut self, ms: f64) {
        self.labels.fade_duration = ms;
    }

    /// Whether labels of different sources collide with each other (GL JS
    /// `crossSourceCollisions`, on by default).
    pub fn set_cross_source_collisions(&mut self, collide: bool) {
        self.labels.cross_source_collisions = collide;
    }

    /// Whether labels keep their place while the camera moves, as in
    /// MapLibre Native (on by default): a label tries its previous variable
    /// anchor first, and in tilted views labels shown before are placed
    /// first. What labels show then depends on how the camera got there;
    /// off, it depends on the camera alone. Still and tile output places
    /// afresh either way.
    pub fn set_stable_labels(&mut self, stable: bool) {
        self.labels.stable_labels = stable;
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

    /// The previews, style inputs beyond the approved v8 spec, that styles
    /// set after this may use; none by default. With a preview off, its
    /// inputs warn and fall back to their defaults.
    pub fn set_previews(&mut self, previews: orb_style::Previews) {
        self.previews = previews;
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

    /// Sets the GPU memory the map aims to stay within
    /// ([`DESKTOP_BUDGET`](orb_render::DESKTOP_BUDGET), the default, or
    /// [`MOBILE_BUDGET`](orb_render::MOBILE_BUDGET)). Tiles out of view stay
    /// cached within a quarter of it and give way first: the scene's pools
    /// and tile pages grow only as the tiles in view need, to the device's
    /// limits, so a view that needs more than the budget still draws.
    pub fn set_memory_budget(&mut self, bytes: u64) {
        self.budget = bytes;
    }

    /// What a job host that starts afresh needs to take this map's requests
    /// (a web job worker that replaces a lost one): the style's id, the
    /// whole style as its first update has it, and its sprite metadata.
    /// `None` before a style loads.
    pub fn jobs_style(&self) -> Option<(u64, crate::StyleUpdate, crate::SpriteIcons)> {
        let style = self.style.as_ref()?;
        let update = full_update(
            &style.doc,
            self.version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        );
        Some((style.id, update, style.icons.clone()))
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

    /// Per label text, the greatest opacity it drew with at the last
    /// frame's time: for benches that watch labels across gestures.
    pub fn shown_labels(&self) -> HashMap<Box<str>, f32> {
        self.labels.shown(self.frame_time)
    }

    /// The view's style zoom: GL JS's in Web Mercator,
    /// measured at the plane's reference latitude on other planes.
    pub fn zoom(&self) -> f64 {
        self.camera.state().zoom(self.surface.projection())
    }

    /// The surface's projection, for camera moves: the style's root
    /// `projection`'s plane. It changes when a style loads (with
    /// [`MapEvent::StyleLoad`]) or an edit sets another.
    pub fn projection(&self) -> orb_crs::Projection {
        *self.surface.projection()
    }

    /// What views draw on: the plane, or the globe the style's `projection`
    /// wraps it around. Camera stores follow it on
    /// [`MapEvent::StyleData`].
    pub fn surface(&self) -> orb_crs::Surface {
        self.surface.view_surface()
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

    /// Whether a paint transition is running.
    pub fn transitioning(&self) -> bool {
        self.transitions.running()
    }

    /// What the map did since the last call, in order: style and source
    /// data, load, idle, errors, the camera's moves and clicks, then the
    /// warnings since, each once per style revision.
    pub fn take_events(&mut self) -> Vec<MapEvent> {
        let warned = &mut self.warned;
        let warnings = self.warnings.drain(..).filter(|w| warned.insert(w.clone()));
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

    /// Loads a v8 style from a URL. Without a style it installs when it
    /// arrives; over one it applies as the diff from it (GL JS's `setStyle`
    /// with `diff: true`): sources defined alike keep their tiles, layers of
    /// the same id, type and source data keep their meshes, and their paint
    /// eases by the v8 `transition`. Root values `transition`, `sky`,
    /// `projection` and the rest follow the new document; the camera stays.
    pub fn load_style_url(&mut self, url: &str) {
        self.style_error = None;
        self.style_request += 1;
        let request = self.style_request;
        let owned = url.to_string();
        self.fetch_text(url, move |result| Message::Style {
            request,
            url: owned,
            result,
        });
    }

    /// Why the latest style request failed, if it did: a fetch or parse
    /// error. Frames draw nothing until another style replaces it.
    pub fn style_error(&self) -> Option<&str> {
        self.style_error.as_deref()
    }

    /// Replaces the style with a v8 document. Its tiles load as frames
    /// render.
    pub fn set_style_json(&mut self, json: &str) -> Result<(), String> {
        self.set_document(json, None)
    }

    /// Installs a v8 style; `base`, the URL it came from, resolves its
    /// relative URLs.
    fn set_document(&mut self, json: &str, base: Option<&str>) -> Result<(), String> {
        // A fetched style's relative URLs are relative to it.
        let resolved = base.map(|base| orb_source::endpoint::resolve_style_urls(json, base));
        let text = resolved.as_deref().unwrap_or(json);
        let doc = StyleDoc::parse_with(text, self.previews).map_err(|d| d.to_string())?;
        // A style URL still loading is superseded, and the old style's
        // paint transitions with it.
        self.style_request += 1;
        self.style_error = None;
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
    /// to draw on another device. This frame lays the style's
    /// tiles out again, from the jobs' cached tile frames where they have
    /// them (refetched where not); labels place afresh, and sprites and
    /// glyphs load again.
    fn forget_gpu(&mut self) {
        self.globe_warps = crate::globe::GlobeWarps::new(*self.surface.projection());
        self.retired_pages.clear();
        self.retired_images.clear();
        self.retired_ramps.clear();
        let old = &self.labels;
        let mut labels = crate::labels::Labels::new(Arc::clone(&old.lane));
        labels.fade_duration = old.fade_duration;
        labels.cross_source_collisions = old.cross_source_collisions;
        labels.ideographs = old.ideographs.clone();
        labels.stable_labels = old.stable_labels;
        self.labels = labels;
        self.label_epoch += 1;
        *self.picking.get_mut() = Default::default();
        if let Some(style) = self.style.take() {
            self.generation += 1;
            // What the host set at run time comes back with it.
            let states = style.states.carried();
            self.install_revision(style.doc, Some((style.id, &style.worker)));
            for (name, image) in style.added_images {
                let _ = self.add_image(&name, Arc::unwrap_or_clone(image));
            }
            let style = self.style.as_mut().expect("installed");
            for op in &states {
                style.states.apply(op);
            }
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
        self.warned.clear();
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
            nodes: (plan.nodes.iter())
                .map(|n| (n.key, NodeState::new(plan, n, 0)))
                .collect(),
            drawn: Vec::new(),
            placed: vec![Vec::new(); plan.sources.len()],
            held: vec![Vec::new(); plan.sources.len()],
            id: self.style_id,
            doc,
            worker: Arc::clone(&worker),
            sources,
            retired: Vec::new(),
            paint: Default::default(),
            image_namespace: None,
            sprites_started: false,
            sprites_loading: 0,
            sprite_pixels_loading: 0,
            sprite_icon_names: Vec::new(),
            relayout_after_pixels: false,
            sprite_epoch: 0,
            sprite_names: Default::default(),
            stale_sprites: None,
            glyphs_fetched: false,
            icons: crate::SpriteIcons::default(),
            image_edits: Vec::new(),
            added_images: Default::default(),
            reported_missing: Default::default(),
            background: None,
            states: FeatureStates::default(),
            symbol_tiles: Vec::new(),
            draws: Vec::new(),
            terrain: None,
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
                        // Its first bytes, the header and root directory,
                        // go to the jobs too.
                        let url = archive.to_string();
                        let done = self.reply(move |result: Result<Vec<u8>, FetchError>| {
                            let header = orb_source::pmtiles::HEADER_LEN as usize;
                            let read = result.and_then(|bytes| {
                                orb_source::endpoint::archive_tilejson(
                                    bytes.get(..header).unwrap_or(&bytes),
                                )
                                .map(|tilejson| (tilejson, bytes))
                                .map_err(FetchError::Range)
                            });
                            let (result, head) = match read {
                                Ok((tilejson, bytes)) => (Ok(tilejson), Some((url, bytes))),
                                Err(e) => (Err(e), None),
                            };
                            Message::TileJson {
                                generation,
                                source: key,
                                result,
                                head,
                            }
                        });
                        crate::loader::read_head(&self.services.fetch, archive, done);
                    } else {
                        self.fetch_text(url, move |result| Message::TileJson {
                            generation,
                            source: key,
                            result,
                            head: None,
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
                Ok((TileEndpoint::image(image, self.surface.projection()), false)),
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
    ) -> Result<(TileMatrixSet, bool), String> {
        let tms = self.surface.source_tms(id, geojson);
        self.surface.plan(&tms, image).map(|plan| (tms, plan))
    }

    /// Applies what has arrived (the style among it) with `renderer` and
    /// starts the style's sprites, without moving the camera, loading what
    /// a view needs or drawing: still output that can't place its camera
    /// before the style is in waits for it this way. [`render`](Self::render)
    /// begins with it.
    pub fn apply_arrivals(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer, pixel_ratio: f32) {
        if self.renderer != Some(renderer.id()) {
            if self.renderer.is_some() {
                self.forget_gpu();
                self.events.push(MapEvent::Restored);
            }
            self.renderer = Some(renderer.id());
            self.nodes_stale = true;
        }
        self.retired.release(renderer);
        self.start_sprites(gpu, pixel_ratio);
        self.apply_image_edits(gpu);
        // Bring the scene's nodes to the plan before results for it apply.
        self.sync_nodes(gpu, renderer);
        self.apply_messages(gpu, renderer);
        // A URL style may have arrived while applying messages.
        self.start_sprites(gpu, pixel_ratio);
        self.sync_nodes(gpu, renderer);
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
        self.frame_time = frame_time;
        self.born.get_or_insert(frame_time);
        self.frame(gpu, renderer, Some(target), viewport, pixel_ratio)
    }

    /// Applies what has arrived and asks for what the view needs next
    /// (tiles, glyphs, a placement), as a frame would, without moving the
    /// camera or drawing: for a host that hears of arrivals between
    /// frames, so that each request goes out without waiting for the next
    /// frame. The frames after it draw as they would without it. Before
    /// the first [`render`](Self::render), it takes in the style and asks
    /// for its first tiles, at the services' clock's time (which should be
    /// the host's frame clock). Nothing while the camera moves: frames come
    /// at the display's rate then, and what it would ask for is for a view
    /// about to change; nor with another renderer than the frames' (after
    /// a device loss, the next frame takes the new one).
    pub fn prepare(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        viewport: Viewport,
        pixel_ratio: f32,
    ) {
        if self.renderer.is_some_and(|id| id != renderer.id()) || self.is_moving() {
            return;
        }
        if self.born.is_none() {
            self.frame_time = self.services.clock.now();
        }
        self.frame(gpu, renderer, None, viewport, pixel_ratio);
    }

    /// A frame at `self.frame_time`, drawn into `target`, or only
    /// prepared without one.
    fn frame(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        target: Option<TextureId>,
        viewport: Viewport,
        pixel_ratio: f32,
    ) -> FrameReport {
        let frame_time = self.frame_time;
        let clock = Arc::clone(&self.services.clock);
        let started = clock.now();
        renderer.set_antialias_3d(gpu, self.antialias_3d);
        (self.placement_ms, self.upload_ms, self.tiles_arrived) = (0.0, 0.0, 0);
        self.apply_arrivals(gpu, renderer, pixel_ratio);
        let mut times = FrameTimes {
            // Uploads and label commits are stages of their own.
            arrivals: micros(started + self.upload_ms + self.placement_ms, clock.now()),
            ..FrameTimes::default()
        };
        // The camera moves for this frame on what has arrived.
        let moving = match target {
            Some(_) => {
                self.apply_videos(gpu, renderer);
                self.advance_camera(viewport, frame_time)
            }
            None => false,
        };

        // Tiles' patches are on the low-zoom surface the style draws on.
        let plane = self.surface.globe().and_then(|g| g.plane);
        let surface = *self.surface.projection();
        self.globe_warps.set_planes(renderer, surface, plane);
        let view = self.view(viewport);
        // What this frame draws, for picking.
        let mut drawn = Vec::new();
        let zoom = self.zoom();
        let mut report = FrameReport::default();
        let clear = self.clear_color;
        let model = self
            .style
            .as_ref()
            .map(|s| Arc::clone(&s.worker.plan().style));
        let (paint, at) =
            crate::light::atmosphere(model.as_deref(), &view, self.camera.state(), zoom);
        renderer.set_atmosphere(gpu, paint, at);

        let now = frame_time;
        let mut fading = false;
        // A prepared frame leaves paint to the frame that draws: it asks
        // for tiles sooner.
        if let Some(style) = &mut self.style
            && target.is_some()
        {
            let mut images = self.services.images.0.lock().expect("images poisoned");
            let missing;
            (fading, missing) = crate::paint::frame(
                gpu,
                renderer,
                style,
                &mut self.transitions,
                (zoom, self.camera.state().bearing, self.camera.state().pitch),
                now,
                images.atlas.as_mut().expect("initialized"),
            );
            drop(images);
            for name in missing {
                self.events.push(MapEvent::StyleImageMissing { name });
            }
        }
        if let Some(style) = &mut self.style {
            style.symbol_tiles.clear();
        }
        let default_light;
        let light = match model.as_deref() {
            Some(s) => &s.light,
            None => {
                default_light = orb_style::v8::default_light();
                &default_light
            }
        };
        let frame_light = crate::light::frame_light(light, zoom, self.camera.state().bearing);
        let mut frame_terrain = None;
        if let Some(style) = &mut self.style
            && !std::mem::replace(&mut style.glyphs_fetched, true)
        {
            // A style's first frame fetches the glyphs its labels start
            // with beside its sources' TileJSON, not after the tiles that
            // need them.
            let urls = style.worker.first_glyphs(zoom);
            if !urls.is_empty() {
                self.services.jobs.prefetch_glyphs(style.id, urls);
            }
        }
        // A prepared frame leaves the terrain's cover and drape, whose
        // redraws are the drawn frame's, to the next one.
        let drawing = target.is_some();
        // Slots redrawn: the next frame drops their drape placements.
        let mut redrew = false;
        if self.style.is_some() {
            // Terrain's DEM and cover first: the sources cover over its
            // elevations, and drape over its tiles.
            let at = clock.now();
            if drawing {
                frame_terrain = self.update_terrain(gpu, renderer, &view, &mut report);
            }
            let planning = clock.now();
            times.terrain = micros(at, planning);
            let elevation = self.terrain_elevation();
            // Failed tiles due for a retry go, so the plan asks for them.
            let now = self.services.clock.now();
            for source in self.style.iter_mut().flat_map(|s| &mut s.sources) {
                if let SourceState::Ready { pyramid, .. } = source {
                    pyramid.retry_failed(now);
                }
            }
            let plan = self.plan_frame(renderer, &view, elevation.as_deref());
            for (index, update) in plan.into_iter().enumerate() {
                self.apply_source(index, update, &mut report);
            }
            let draping = clock.now();
            times.plan = micros(planning, draping);
            let drape = match drawing {
                true => self.plan_drape(gpu, renderer, &view),
                false => None,
            };
            let placing = clock.now();
            times.drape = micros(draping, placing);
            redrew = drape.as_ref().is_some_and(|d| !d.redraw.is_empty());
            if drawing {
                self.place_background(
                    gpu,
                    renderer,
                    &view,
                    elevation.as_deref(),
                    drape.as_ref(),
                    &mut report,
                );
            }
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
            renderer.shrink(gpu);
            times.place = micros(placing, clock.now());
        }
        let labels_settled = self.place_labels(gpu, renderer, &view, now);
        times.placement = micros(0.0, self.placement_ms);
        times.upload = micros(0.0, self.upload_ms);
        let Some(target) = target else {
            self.globe_warps.end_frame();
            times.total = micros(started, clock.now());
            self.prepared.0.add_prepared(&times);
            self.prepared.1 += self.tiles_arrived;
            return report;
        };
        let (prepared, arrived) = std::mem::take(&mut self.prepared);
        report.prepared = prepared;
        report.tiles_arrived = self.tiles_arrived + arrived;

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
                && s.sprite_pixels_loading == 0
                && s.sources
                    .iter()
                    .all(|source| !matches!(source, SourceState::Resolving))
        }) && report.tiles_loading == 0
            && !self.tiles_deferred
            && self.waiting.is_empty()
            && self.pending.is_empty()
            && labels_settled;
        report.idle = report.loaded
            && !redrew
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
            time: (now - self.born.unwrap_or(now)) as f32,
            pitch: self.camera.state().pitch.to_radians() as f32,
            roll: self.camera.state().roll.to_radians() as f32,
            camera_to_center: view.camera_to_center() as f32,
            align: match globe {
                Some(_) => [0.0; 2],
                None => pixel_alignment(
                    &view,
                    self.surface.projection(),
                    self.camera.state().bearing,
                ),
            },
            terrain: frame_terrain,
            light: frame_light,
            globe,
        };
        // The shared atlas grows pages as images arrive.
        if let Some(atlas) = &self
            .services
            .images
            .0
            .lock()
            .expect("images poisoned")
            .atlas
        {
            renderer.set_images(atlas.bindings());
        }
        report.frame = renderer.render(gpu, &frame, target, clear, &|| clock.now());
        times.render = report.frame.times;
        // A frame that left something out for a compiling pipeline isn't
        // complete.
        report.compiling = report.idle && report.frame.waited;
        report.loaded &= !report.frame.waited;
        report.idle &= !report.frame.waited;
        // Idle once the camera rests too, and once per rest. DEM data that
        // arrived after the camera advanced may move it onto the ground it
        // lands on: then this frame isn't a rest.
        let moving = self.stand_on_terrain() || moving;
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
        times.total = micros(started, clock.now());
        report.times = times;
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
        let keys: Vec<u32> = plan.nodes.iter().map(|n| n.key).collect();
        let fresh = std::mem::take(&mut self.nodes_stale);
        // A projection edit put the map on or off a globe: the background
        // takes the mesh it bends (image tiles take theirs as they draw).
        let globe = self.surface.globe().is_some();
        if !fresh && keys == style.drawn && style.globe_meshes == globe {
            return;
        }
        style.globe_meshes = globe;
        if let Some(tile) = style.background.take() {
            renderer.release_tile(tile);
        }
        for source in &mut style.sources {
            let SourceState::Ready { pyramid, .. } = source else {
                continue;
            };
            for (_, resident) in pyramid.ready_mut() {
                resident
                    .symbols
                    .retain(|s| keys.contains(&crate::style::node_key(s.layer, 0)));
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
        style.background = make_background(gpu, renderer, plan, globe);
        style.drawn = keys;
        style.paint.invalidate();
        for placed in &mut style.placed {
            placed.clear();
        }
    }

    fn start_sprites(&mut self, gpu: &mut dyn Gpu, pixel_ratio: f32) {
        let mut guard = self.services.images.0.lock().expect("images poisoned");
        let retired = std::mem::take(&mut guard.retired);
        let retired_ramps = std::mem::take(&mut guard.retired_ramps);
        let images = guard.atlas.as_mut().expect("renderer initializes images");
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
        // Another sprite loads into the same namespace, over the old one.
        style
            .image_namespace
            .get_or_insert_with(|| images.namespace());
        style.sprites_started = true;
        style.sprite_epoch += 1;
        let epoch = style.sprite_epoch;
        let worker = Arc::clone(&style.worker);
        let sprites = &worker.plan().style.sprites;
        style.sprites_loading = sprites.len() as u32;
        style.sprite_pixels_loading = sprites.len() as u32;
        style.sprite_icon_names = vec![Vec::new(); sprites.len()];
        style.icons.pixel_ratio = f64::from(pixel_ratio);
        if sprites.is_empty() && style.stale_sprites.is_some() {
            drop(guard);
            self.sprites_done(gpu);
            return;
        }
        for (index, sprite) in sprites.iter().enumerate() {
            let generation = self.generation;
            let icons = self.reply(move |result| Message::SpriteIcons {
                generation,
                epoch,
                index,
                result,
            });
            let done = self.reply(move |result| Message::Sprite {
                generation,
                epoch,
                index,
                result,
            });
            let (fetch, decode) = (
                self.services.fetch.as_ref(),
                self.services.image_decode.clone(),
            );
            crate::sprite::load(fetch, decode, sprite, pixel_ratio, icons, done);
        }
    }

    /// Every sprite of the style has resolved: the jobs lay icons out with
    /// them. After another sprite replaced the old one, the old one's
    /// images it lacks go, and symbol layers lay out again.
    pub(crate) fn sprites_done(&mut self, gpu: &mut dyn Gpu) {
        let Some(style) = &mut self.style else { return };
        let reload = style.stale_sprites.take();
        if let Some(stale) = &reload {
            let namespace = style.image_namespace.expect("sprite namespace");
            let mut images = self.services.images.0.lock().expect("images poisoned");
            let atlas = images.atlas.as_mut().expect("initialized");
            for name in stale {
                if !style.sprite_names.contains(name) && !style.added_images.contains_key(name) {
                    style.icons.remove(name);
                    atlas.clear(gpu, namespace, name);
                }
            }
            style.paint.invalidate();
        }
        style.icons.epoch += 1;
        self.services
            .jobs
            .set_icons(style.id, &Arc::new(style.icons.clone()), &style.worker);
        if reload.is_some() {
            self.relayout_symbols();
        }
    }

    // Pinned GL JS v8 heights use the IUGG mean sphere (6371008.8 m),
    // whereas projected XY uses the WGS84 semi-major axis. Retain v8's
    // vertical scale without changing the camera's true ground resolution.
    pub(crate) fn extrusion_scale(&self) -> f64 {
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
        let SourceState::Ready {
            endpoint, warped, ..
        } = &mut style.sources[index]
        else {
            return;
        };
        let (warped, tms) = (*warped, &endpoint.tms);
        let mut bend = (warped && vector)
            .then(|| self.globe_warps.surface_series(tms, tile))
            .flatten()
            .map(|series| crate::Bend {
                cells: series.grid(),
                poles: [false; 2],
            });
        let footprint = warped
            .then(|| self.globe_warps.surface_footprint(tms, tile))
            .flatten();
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
                bend = Some(crate::Bend { cells, poles });
            }
        }
        let host = endpoint.host();
        let mut request = TileRequest {
            style: style.id,
            source: key,
            tile,
            zoom,
            bend,
            // A raster tile on another plane draws through its warp's mesh
            // there (on a globe, through a grid the sphere bends).
            warp: warped && !vector && endpoint.quad.is_none() && self.surface.globe().is_none(),
            data_zoom: tile.z.min(endpoint.maxzoom),
            location: endpoint.location(endpoint.data_tile(tile)),
            encoding: endpoint.encoding,
            nodes,
            rows_only,
            dem_unpack: None,
            icons: 0,
        };
        // Symbol layout needs the sprites' metadata: while the sprites load,
        // a source with symbol layers lays out once they're in (GL JS's
        // worker awaits `getImages` alike), and its tiles load meanwhile.
        if style.sprites_loading > 0
            && (style.worker.plan().nodes.iter())
                .any(|n| n.source == index && n.family == orb_layout::Family::SymbolIcon)
        {
            request.icons = style.icons.epoch + 1;
        }
        match (edit, rows_only) {
            (None, _) => self.counts.loads += 1,
            (Some(_), false) => self.counts.relayouts += 1,
            (Some(_), true) => self.counts.columns += 1,
        }
        if let Some(edit) = edit {
            *self.pending.entry(edit).or_default() += 1;
        }
        // A tile of a source the app feeds goes with the app's answer.
        if host {
            let data_tile = request.data_tile();
            match self.host_tiles.answer(key, data_tile) {
                Some(answer) => request.location = crate::TileLocation::Host(answer.clone()),
                None => {
                    if self.host_tiles.wait(key, data_tile, request, edit) {
                        let style = self.style.as_ref().expect("checked above");
                        let source = style.worker.plan().source_def(index).id.clone();
                        self.events.push(crate::MapEvent::TileRequest {
                            source,
                            tile: data_tile,
                        });
                    }
                    return;
                }
            }
        }
        self.load(request, edit, false);
    }

    /// The app's answer for a tile of source `id`, a source it feeds
    /// ([`HOST_TILES`](crate::HOST_TILES)), at the tile a
    /// [`MapEvent::TileRequest`](crate::MapEvent::TileRequest) named.
    pub fn host_tile(
        &mut self,
        id: &str,
        tile: TileIndex,
        answer: crate::HostTile,
    ) -> Result<(), String> {
        let style = self.style.as_ref().ok_or("no style is loaded")?;
        let plan = style.worker.plan();
        let index = plan
            .sources
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| format!("no source {id}"))?;
        let SourceState::Ready { endpoint, .. } = &style.sources[index] else {
            return Err(format!("{id} hasn't loaded"));
        };
        if !endpoint.host() {
            return Err(format!("{id} isn't a source the app feeds"));
        }
        let (key, current) = (plan.sources[index].key, style.id);
        for (mut request, edit) in self.host_tiles.answered(key, tile, answer.clone()) {
            // Jobs of a replaced style have nothing to lay out.
            if request.style == current {
                request.location = crate::TileLocation::Host(answer.clone());
                self.load(request, edit, false);
            }
        }
        self.events.wake();
        Ok(())
    }

    /// Asks the app for every tile of source `id` again: its data changed.
    /// The drawn tiles stay until the new answers replace them.
    pub fn reload_host_tiles(&mut self, id: &str) -> Result<(), String> {
        let style = self.style.as_ref().ok_or("no style is loaded")?;
        let plan = style.worker.plan();
        let index = plan
            .sources
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| format!("no source {id}"))?;
        let key = plan.sources[index].key;
        self.host_tiles.forget(key);
        // Relayouts of the resident tiles, as an edit's: each replaces its
        // tile's layers when it arrives.
        let version = self.version;
        for tile in self.resident_tiles(index, None) {
            self.send_job(index, tile, None, false, Some(version));
        }
        self.events.wake();
        Ok(())
    }

    /// Sends a request to the jobs; its result comes back as a message.
    pub(crate) fn load(&mut self, request: TileRequest, edit: Option<u64>, terrain: bool) {
        let (generation, source, tile) = (self.tile_generation, request.source, request.tile);
        let done = self.reply(move |result| Message::Tile {
            generation,
            source,
            tile,
            edit,
            terrain,
            result: Box::new(result),
        });
        self.services.jobs.load(request, done);
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
        let extrusion = self.extrusion_scale();
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
        // The labels' frame for a view of `camera` centred on `center`.
        let (plan, tiles, period) = (
            style.worker.plan(),
            &style.symbol_tiles,
            self.surface.period(),
        );
        let frame_at = |view: &View, center, camera: &orb_camera::CameraState| {
            let zoom = camera.zoom(self.surface.projection());
            let labels = &self.labels;
            labels.frame(
                plan, tiles, view, center, period, camera, zoom, extrusion, now,
            )
        };
        let mut frame = frame_at(view, view.camera_world(), self.camera.state());
        // Labels show at once until the map is first idle, as in GL JS: a
        // map's first view doesn't wait for fades.
        if self.mode != RenderMode::Continuous || !self.events.idled {
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
                        let f = frame_at(&view, t.center, &t.camera);
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
        self.labels.drawing(&frame);
        // Symbols stand on the terrain as drawn.
        frame.ground = style
            .terrain
            .as_ref()
            .map(|t| orb_symbol::GroundRef(t.snapshot() as Arc<dyn orb_symbol::Ground>));
        // Without symbol tiles there is nothing to place.
        if frame.layers.iter().all(|l| l.tiles.is_empty()) {
            self.labels.nothing_to_place();
            self.placement_ms += self.services.clock.now() - started;
            return !self.labels.settling();
        }
        self.labels.resend_if_restarted();
        let settled = self.labels.is_placed(&frame);
        if self.labels.wants_placement(&frame) {
            self.labels.placing(&frame);
            let placing = self.services.clock.now();
            let done = self.reply(move |committed| Message::Placed { epoch, committed });
            self.labels.lane.place(frame, done);
            self.placement_ms -= self.services.clock.now() - placing;
        }
        self.placement_ms += self.services.clock.now() - started;
        settled && !self.labels.settling()
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
            drape_atlas: renderer.drape_bytes(),
            cached_tiles,
            cached,
            cache_budget: self.budget / 4,
        }
    }

    /// Keeps the terrain to the style's `terrain`: made once its DEM
    /// source is ready and its DEM tiles requested for this view. Returns
    /// the frame's terrain.
    fn update_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        report: &mut FrameReport,
    ) -> Option<orb_render::FrameTerrain> {
        let style = self.style.as_mut()?;
        let plan = style.worker.plan();
        let mut wanted = plan.terrain();
        // Terrain tiles are the surface's; their DEM must be too.
        let unusable = match wanted.map(|(index, _)| &style.sources[index]) {
            Some(SourceState::Ready { endpoint, .. }) if endpoint.tms != *self.surface.tms() => {
                Some(format!(
                    "terrain: the DEM source's tiles ({}) are not the map's ({}); the map is flat",
                    endpoint.tms.id(),
                    self.surface.tms().id()
                ))
            }
            _ => None,
        };
        if let Some(warning) = unusable {
            self.warnings.push(warning);
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
                old.release(gpu, renderer);
            }
            if let Some((index, exaggeration)) = wanted
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
            renderer.set_terrain(gpu, None);
            return None;
        };
        let SourceState::Ready { endpoint, .. } = &style.sources[terrain.source] else {
            return None;
        };
        let endpoint = Arc::clone(endpoint);
        terrain.pyramid.retry_failed(self.services.clock.now());
        let unpack = endpoint.dem_unpack.expect("raster-dem").map(|v| v as f32);
        // GL JS's `getSkirtLength`: a fifth of a tile's width at the
        // camera's zoom, in metres at the equator.
        let zoom = self.camera.state().zoom(self.surface.projection());
        let skirt = match self.terrain_skirts {
            true => 2.0 * std::f64::consts::PI * 6_371_008.8 / zoom.max(0.0).exp2() / 5.0,
            false => 0.0,
        };
        // Viewport-anchored hillshades turn with the map: they drape their
        // slopes into a stack of their own, which the cover's room counts,
        // and the terrain pass lights them.
        for (index, node) in plan.nodes.iter().enumerate() {
            if node.family != Family::Hillshade {
                continue;
            }
            let layer = &plan.style.layers[node.layer];
            let viewport = layer
                .get(orb_style::HILLSHADE_ILLUMINATION_ANCHOR)
                .evaluate(zoom)
                == orb_style::IlluminationAnchor::Viewport;
            // Its source covers the terrain tiles' quarters, as the
            // surface's tiles.
            let on_surface = matches!(&style.sources[node.source],
                SourceState::Ready { endpoint, .. } if endpoint.tms == *self.surface.tms());
            renderer.set_shaded_on_terrain(gpu, index as u32, viewport && on_surface);
        }
        let at = (view, self.surface.tms());
        let (frame, requests) =
            terrain.upkeep(gpu, renderer, at, unpack, skirt * view.height_scale());
        let mut loads = Vec::new();
        for tile in requests {
            if !endpoint.contains(tile) {
                terrain.pyramid.insert_empty(tile);
                continue;
            }
            loads.push(TileRequest {
                style: style.id,
                source: terrain.key,
                tile,
                zoom: endpoint.tms.style_zoom(tile),
                bend: None,
                warp: false,
                data_zoom: tile.z,
                location: endpoint.location(tile),
                encoding: endpoint.encoding,
                nodes: None,
                rows_only: false,
                dem_unpack: endpoint.dem_unpack.map(crate::DemUnpack),
                icons: 0,
            });
        }
        terrain.loading += loads.len() as u32;
        report.tiles_loading += terrain.loading;
        for tile in terrain.pyramid.unwanted_loads() {
            self.services.jobs.cancel(style.id, terrain.key, tile);
        }
        self.counts.loads += loads.len() as u32;
        for request in loads {
            self.load(request, None, true);
        }
        Some(frame)
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

    /// Changes whenever [`elevation`](Self::elevation) has new data.
    pub fn elevation_revision(&self) -> u64 {
        self.style
            .as_ref()
            .and_then(|s| s.terrain.as_ref())
            .map_or(0, |t| t.id << 32 | t.revision)
    }

    /// A terrain DEM tile arrived.
    pub(crate) fn apply_terrain_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
    ) {
        let now = self.services.clock.now();
        let terrain = self.style.as_mut().and_then(|s| s.terrain.as_mut());
        if let Some(terrain) = terrain.filter(|t| t.key == source)
            && let Err(e) = terrain.arrive(gpu, renderer, tile, result, now)
        {
            self.warnings.push(format!("terrain tile {tile:?}: {e}"));
        }
        self.upload_ms += self.services.clock.now() - now;
    }

    /// Installs a style that `load_style_url` fetched, unless a later
    /// request or `set_style_json` superseded it.
    pub(crate) fn apply_style(
        &mut self,
        request: u64,
        url: &str,
        result: Result<String, FetchError>,
    ) {
        if request != self.style_request {
            return;
        }
        let installed = result.map_err(|e| e.to_string()).and_then(|json| {
            if self.style.is_some() {
                self.diff_document(&json, url)
            } else {
                self.set_document(&json, Some(url))
            }
        });
        if let Err(e) = installed {
            let message = format!("style: {e}");
            self.style_error = Some(message.clone());
            self.fail(message);
        }
    }

    /// Applies a fetched style as the diff from the loaded one; it loads
    /// (`styleload`) as a new one would.
    fn diff_document(&mut self, json: &str, base: &str) -> Result<(), String> {
        let resolved = orb_source::endpoint::resolve_style_urls(json, base);
        let tx = Transaction::from_commands(&format!(r#"[["setStyle",{resolved}]]"#))?;
        let id = self.style_id;
        self.edit(&tx)?;
        // A diff that needed a reload installed the document, which said so.
        if self.style_id == id {
            self.events.push(MapEvent::StyleLoad);
        }
        Ok(())
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
        let pixels = frame.pixels.upload(true);
        for (tile, resident) in pyramid.ready() {
            if resident.image.is_some() {
                let key = orb_render::TileKey { source: key, tile };
                renderer.rewrite_image(gpu, key, [frame.width, frame.height], pixels);
            }
        }
    }

    /// Whether the host plays `video` sources: it then gets
    /// [`MapEvent::Video`] requests and hands frames to
    /// [`source_frame`](Self::source_frame). Without it, video sources draw
    /// nothing, with a warning.
    pub fn set_plays_video(&mut self, plays: bool) {
        self.videos.enabled = plays;
    }

    /// The picture video or canvas source `id` shows now, straight-alpha
    /// RGBA rows, top first, or an image the host registered with the GPU;
    /// it replaces any frame not yet drawn.
    pub fn source_frame(&mut self, id: &str, frame: crate::TileImage) -> Result<(), String> {
        let plan = self.style.as_ref().map(|s| s.worker.plan());
        self.videos.sync(plan, &mut self.events.queue);
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

    fn fetch_text(
        &self,
        url: &str,
        message: impl FnOnce(Result<String, FetchError>) -> Message + Send + 'static,
    ) {
        let done = self.reply(move |result: Result<Vec<u8>, FetchError>| {
            message(result.map(|bytes| String::from_utf8_lossy(&bytes).into_owned()))
        });
        self.services.fetch.get(url, done);
    }

    /// A callback that sends `message(result)` to the map's next frame and
    /// wakes it, from any thread.
    fn reply<T>(
        &self,
        message: impl FnOnce(T) -> Message + Send + 'static,
    ) -> Box<dyn FnOnce(T) + Send> {
        let (sender, wake) = (self.sender.clone(), Arc::clone(&self.services.wake));
        Box::new(move |result| {
            let _ = sender.send(message(result));
            wake();
        })
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

/// Removes a terrain's DEM page; the renderer's terrain is off.
#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use orb_host::FetchDone;
    use orb_host::jobs::Inline;

    use super::*;
    use crate::PoolJobs;

    /// S5: a result laid out before an edit to a symbol layer brings
    /// neither its meshes nor its bucket.
    #[test]
    fn stale_results_drop_symbol_buckets_with_their_meshes() {
        let (worker, mut result) = crate::testing::symbol_tile();
        assert!(!result.symbols.is_empty());
        let text = crate::style::node_key(result.symbols[0].layer, 1);
        drop_stale(&mut result, worker.plan(), &[text]);
        assert!(result.symbols.is_empty());
        assert!(result.meshes.iter().all(|m| m.node != text));
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
    fn a_style_set_directly_supersedes_one_still_loading() {
        let fetch = Arc::new(HeldFetch::default());
        let services = Services {
            clock: Arc::new(NoClock),
            image_decode: Arc::new(NoDecode),
            images: Default::default(),
            jobs: PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::worker::tests::NoImages),
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
            result,
        }) = map.receiver.try_recv()
        {
            map.apply_style(request, &url, result);
        }
        let name = map.style.as_ref().unwrap().worker.plan().style.name.clone();
        assert_eq!(name.as_deref(), Some("b"));
    }

    #[test]
    fn the_root_projection_chooses_the_surface() {
        let fetch = Arc::new(HeldFetch::default());
        let services = Services {
            clock: Arc::new(NoClock),
            image_decode: Arc::new(NoDecode),
            images: Default::default(),
            jobs: PoolJobs::new(
                fetch.clone(),
                Arc::new(crate::worker::tests::NoImages),
                Arc::new(Inline),
            ),
            fetch: fetch.clone(),
            wake: Arc::new(|| {}),
        };
        let mut map = Map::new(services, None);
        map.set_previews(orb_style::Previews::ALL);
        map.load_style_url("https://example.com/maps/polar.json");
        let (_, done) = fetch.pending.lock().unwrap().pop().unwrap();
        done(Ok(br#"{"version": 8, "projection": {"type": "EPSG:3413"},
            "sources": {"s": {"type": "raster", "tiles": ["t/{z}/{x}/{y}.png"],
                              "tileMatrixSet": "UPSArcticWGS84Quad"}},
            "layers": []}"#
            .to_vec()));
        while let Ok(Message::Style {
            request,
            url,
            result,
        }) = map.receiver.try_recv()
        {
            map.apply_style(request, &url, result);
        }
        assert_eq!(map.projection().descriptor().kind, "EPSG:3413");
        assert_eq!(
            map.surface.source_tms("s", false).id(),
            "UPSArcticWGS84Quad"
        );
        // An edit to Web Mercator plans the surface again.
        let tx = orb_style::v8::Transaction::from_commands(
            r#"[["setProjection", {"type": "mercator"}]]"#,
        )
        .unwrap();
        map.edit(&tx).unwrap();
        assert_eq!(map.projection().descriptor().kind, "web-mercator");
        assert_eq!(
            map.surface.source_tms("s", false).id(),
            "UPSArcticWGS84Quad"
        );
        // A style without a projection is on Web Mercator.
        map.set_style_json(&style("c")).unwrap();
        assert_eq!(map.projection().descriptor().kind, "web-mercator");
    }
}

#[cfg(test)]
mod video_tests {
    use orb_hal_wgpu::WgpuGpu;
    use orb_host::jobs::Inline;

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
                    Arc::new(crate::worker::tests::NoImages),
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
        map.source_frame("v", frame([255, 0, 0, 255])).unwrap();
        assert_eq!(
            centre(&mut map, &mut gpu, &mut renderer),
            (true, [255, 0, 0])
        );
        map.source_frame("v", frame([0, 0, 255, 255])).unwrap();
        assert_eq!(
            centre(&mut map, &mut gpu, &mut renderer),
            (true, [0, 0, 255])
        );
        map.control_video("v", VideoRequest::Pause).unwrap();
        assert!(map.control_video("w", VideoRequest::Play).is_err());
        assert!(map.source_frame("w", frame([0; 4])).is_err());
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
