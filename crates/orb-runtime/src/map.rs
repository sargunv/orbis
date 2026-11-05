use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_camera::{CameraState, View, Viewport};
use orb_crs::{TileIndex, TileMatrixSet};

use crate::surface::{Plan, Surface};
use orb_hal::{Gpu, HostBytesId, TextureId};
use orb_host::{Fetch, FetchError, ImageDecode};
use orb_render::{
    ClipRect, Curve, DemRef, Family, FillPaint, FrameBlock, FrameStats, LinePaint, MeshData,
    MeshId, MeshSize, NodePaint, OutOfSpace, PageId, PatternPaint, PlacementTarget, Renderer,
    SceneCapacity, SceneStats, TargetFormat, TileId, TileRow, TileWarp, Translate, Words,
};
use orb_render::{ImageAtlas, ImageUpload};
use orb_source::{Draw, TilePyramid};
use orb_style::{CostModel, StyleDoc};
use orb_v8::{LayerKind, SourceKind, TranslateAnchor};

use crate::MapEvent;
use crate::image::{self, ResidentImage};
use crate::jobs::{Jobs, ResultMesh, Stream, StyleUpdate, TileRequest, TileResult, TileWorker};
use crate::source::TileEndpoint;
use crate::state::FeatureStates;
use crate::style::{PaintProperty, StylePlan};

/// Logical pixel size of vector/background tiles and pattern origins.
const TILE_PX: f64 = 512.0;

/// Ready tiles kept per source beyond those in view.
const TILE_CACHE: usize = 128;
/// Milliseconds a frame spends applying arrived tiles before it leaves the
/// rest for the next frame (plan §7.4's upload budget at 60 Hz).
const TILE_APPLY_MS: f64 = 1.5;
/// Ready image tiles kept per source beyond those in view: each holds a
/// texture layer.
const IMAGE_TILE_CACHE: usize = 32;

/// GPU images shared by the maps in one Runtime / host-service set. The
/// first render initializes them on that Runtime's device.
#[derive(Clone, Debug, Default)]
pub struct Images(pub(crate) Arc<Mutex<ImageState>>);

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
    /// Microseconds this thread spent on placement: building the frame the
    /// placement lane places and applying its commits (plan §6.1 step 10),
    /// not the placement itself, which is the lane's (inline on a host
    /// without a placement thread or worker).
    pub placement_us: u32,
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
    /// A placement committed.
    Placed(orb_symbol::Committed),
    /// A glyph raster job finished, for these code points and rows.
    Raster {
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
#[derive(Debug, Default)]
pub(crate) struct ResidentTile {
    pub(crate) parts: Vec<(u32, TileId)>,
    pub(crate) meshes: Vec<Resident>,
    pub(crate) image: Option<ResidentImage>,
    /// A vector tile drawn through a warp: its series, and its patches in
    /// the scene.
    pub(crate) series: Option<Arc<orb_crs::Series>>,
    pub(crate) warp: TileWarp,
    /// The style zoom it laid out at (decision 0022).
    pub(crate) zoom: f64,
    /// Its symbol layers' buckets; their meshes are among `meshes`.
    pub(crate) symbols: Vec<crate::labels::ResidentSymbols>,
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
        /// An image source's texture page, made for its first tile.
        page: Option<PageId>,
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
    painted_fade: Option<Crossfade>,
    zoom_history: ZoomHistory,
    pub(crate) image_namespace: Option<u64>,
    sprites_started: bool,
    sprites_loading: u32,
    /// Sprite metadata loaded so far; the jobs get it once every sprite has
    /// resolved, before tiles with symbol layers are requested.
    icons: crate::SpriteIcons,
    /// The scene tile of the background layers' quads.
    background: Option<TileId>,
    /// Tiles with placements in publication order, parallel to sources.
    pub(crate) placed: Vec<Vec<TileIndex>>,
    pub(crate) states: FeatureStates,
    /// This frame's drawn symbol buckets, for placement.
    pub(crate) symbol_tiles: Vec<crate::labels::SymbolTile>,
    pub(crate) programs: crate::gpu::ProgramTable,
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

/// A tile result waiting for room in the scene (`Map::apply_tile`).
#[derive(Debug)]
struct WaitingTile {
    generation: u64,
    /// The plan source's key.
    source: u32,
    tile: TileIndex,
    build: TileResult,
}

/// A map: a style, a camera and the tiles they need, drawn by
/// [`render`](Self::render).
pub struct Map {
    /// Placement and the scene side of labels.
    pub(crate) labels: crate::labels::Labels,
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
    camera: CameraState,
    antialias_3d: bool,
    cost: CostModel,
    /// The plane the map draws on and its sources' tile matrix sets, from
    /// the scene document.
    pub(crate) surface: Surface,
    /// The scene document's lights, which replace the style's.
    lights: Option<orb_v8::Lights>,
    /// Paint edits easing in (decision 0031).
    pub(crate) transitions: crate::transitions::Transitions,
    /// The latest `load_style_url` request; earlier ones are ignored.
    style_request: u64,
    generation: u64,
    tile_generation: u64,
    pub(crate) style: Option<LoadedStyle>,
    pub(crate) renderer: Option<(Renderer, TargetFormat)>,
    /// The renderer's nodes are of a replaced style.
    nodes_stale: bool,
    capacity: SceneCapacity,
    /// The last style id given to the jobs.
    style_id: u64,
    /// Counts style revisions: every new style and every edit.
    pub(crate) version: u64,
    /// Outstanding relayout and column jobs, per edit generation.
    pub(crate) pending: BTreeMap<u64, u32>,
    pub(crate) counts: JobCounts,
    /// Take the camera from the first style's `center`, `zoom`, `bearing`
    /// and `pitch`.
    camera_from_style: bool,
    clear_color: [f64; 4],
    sender: Sender<Message>,
    receiver: Receiver<Message>,
    pub(crate) warnings: Vec<String>,
    pub(crate) events: crate::events::EventQueue,
    /// Pages of a replaced style, to destroy with the next frame's GPU.
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
            .field("camera", &self.camera)
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
        cost: doc.cost_model(),
        surface,
    }
}

/// A revision of `doc`'s plan, keeping `previous`'s source data.
pub(crate) fn revise(
    previous: Option<&TileWorker>,
    doc: &StyleDoc,
    revision: u64,
    local_ideographs: Option<String>,
    surface: crate::SurfaceTiles,
) -> TileWorker {
    TileWorker::with_style(
        previous,
        Arc::clone(doc.style()),
        doc.layer_keys().map(|k| k.0).collect(),
        doc.source_keys().map(|k| k.0).collect(),
        revision,
        doc.cost_model(),
        local_ideographs,
        surface,
    )
}

impl Map {
    /// A map with no style. With `camera: None`, the first style's camera
    /// is used, as GL JS does when a map is created without one.
    pub fn new(services: Services, camera: Option<CameraState>) -> Self {
        let (sender, receiver) = channel();
        let camera_from_style = camera.is_none();
        let camera = camera.unwrap_or_else(|| {
            let mercator = *Surface::default().projection();
            CameraState::new(orb_geo::Geodetic::new(0.0, 0.0), 0.0, &mercator)
        });
        Self {
            picking: Default::default(),
            placement_ms: 0.0,
            tiles_deferred: false,
            waiting: Vec::new(),
            candidates_dropped: 0,
            services,
            camera,
            surface: Surface::default(),
            lights: None,
            transitions: Default::default(),
            style_request: 0,
            generation: 0,
            tile_generation: 0,
            style: None,
            renderer: None,
            antialias_3d: true,
            cost: CostModel::DEFAULT,
            nodes_stale: false,
            capacity: SceneCapacity::default(),
            style_id: 0,
            version: 0,
            pending: BTreeMap::new(),
            counts: JobCounts::default(),
            camera_from_style,
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
        }
    }

    /// The renderer's per-node draw arguments and cull counters, and its
    /// node count, once it exists: for diagnostics (see
    /// [`Renderer::draw_args`]).
    pub fn draw_args(&self) -> Option<(orb_hal::BufferId, u32)> {
        self.renderer
            .as_ref()
            .map(|(renderer, _)| renderer.draw_args())
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

    /// Enables 4× MSAA for 3D surfaces (the default). Disabling it uses
    /// single-sample rasterization, matching a GL JS map with antialias off.
    pub fn set_antialias_3d(&mut self, enabled: bool) {
        self.antialias_3d = enabled;
    }

    /// Chooses which per-feature paint the GPU interprets (decision 0017);
    /// benchmarks compare it with rows alone. A loaded style is installed
    /// again under it.
    pub fn set_cost_model(&mut self, cost: CostModel) {
        if cost == self.cost {
            return;
        }
        self.cost = cost;
        if let Some(style) = &self.style {
            let doc = style.doc.clone();
            self.install(doc);
        }
    }

    /// Sets how much the GPU scene holds. It's allocated on the first frame,
    /// so this has effect only before then.
    pub fn set_scene_capacity(&mut self, capacity: SceneCapacity) {
        assert!(self.renderer.is_none(), "the scene is already allocated");
        self.capacity = capacity;
    }

    /// The last placement commit's semantic snapshot (plan §13): the
    /// labels and icons it shows, a reading order and a region summary.
    /// `None` before labels are first placed.
    pub fn semantic_snapshot(&self) -> Option<orb_a11y::Snapshot> {
        let style = self.style.as_ref()?;
        let target = self.camera.target;
        self.labels.semantic_snapshot(
            style.worker.plan(),
            [target.lon, target.lat],
            self.zoom(),
            &self.bucket_planes(style),
        )
    }

    pub fn camera(&self) -> &CameraState {
        &self.camera
    }

    /// The camera, to move it. Moving it means the app owns the camera, so
    /// a style that arrives later doesn't override it.
    pub fn camera_mut(&mut self) -> &mut CameraState {
        self.camera_from_style = false;
        self.events.wake();
        &mut self.camera
    }

    /// Sets the camera, as a host's camera store does each frame it moved;
    /// a style that arrives later doesn't override it.
    pub fn set_camera(&mut self, camera: CameraState) {
        *self.camera_mut() = camera;
    }

    /// The view's style zoom (decision 0022): GL JS's in Web Mercator,
    /// measured at the plane's reference latitude on other planes.
    pub fn zoom(&self) -> f64 {
        self.camera.zoom(self.surface.projection())
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
        self.surface = self.surface.with_projection(projection);
        if let Some(style) = &self.style {
            let doc = style.doc.clone();
            self.install(doc);
        }
    }

    /// The view the camera gives for a viewport.
    pub fn view(&self, viewport: Viewport) -> View {
        match self.elevation() {
            Some(terrain) => View::over(
                &self.camera,
                viewport,
                &self.surface.view_surface(),
                &*terrain,
            ),
            None => View::new(&self.camera, viewport, &self.surface.view_surface()),
        }
    }

    /// Keeps the camera to the world's edge, as GL JS's transform does
    /// (`getConstrained`), and stands its target on the terrain, as a
    /// camera store at rest does. Camera stores keep to these themselves;
    /// hosts that write the camera directly call this, since rendering
    /// never writes the camera.
    pub fn constrain_camera(&mut self, viewport: Viewport) {
        let surface = self.surface.view_surface();
        self.camera
            .constrain(&orb_camera::Constraints::WORLD, &surface, viewport);
        match self.elevation() {
            Some(terrain) => {
                self.camera.settle_on(&*terrain, &surface.projection);
                self.camera.keep_above(&*terrain, &surface, viewport);
            }
            None => self.camera.elevation = 0.0,
        }
    }

    /// Whether a paint transition is running (decision 0031).
    pub fn transitioning(&self) -> bool {
        self.transitions.running()
    }

    /// What the map did since the last call: style and source data, load,
    /// idle and errors.
    pub fn take_events(&mut self) -> Vec<MapEvent> {
        std::mem::take(&mut self.events.queue)
    }

    /// Reports a failed load as a warning and an error event.
    pub(crate) fn fail(&mut self, message: String) {
        self.events.push(MapEvent::Error {
            message: message.clone(),
        });
        self.warnings.push(message);
    }

    /// Sets the host's backing colour (premultiplied RGBA). Transparent by
    /// default, like GL JS's canvas. Opaque hosts can supply their window's
    /// background so transparent sky does not become black at presentation.
    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.clear_color = color;
    }

    /// Diagnostics collected since the last call: style warnings, failed
    /// loads.
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
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

    /// Installs a v8 style or a scene document; `base` resolves a scene's
    /// relative style URL, and `scene` keeps the surface for a scene's own
    /// style.
    fn set_document(&mut self, json: &str, base: Option<&str>, scene: bool) -> Result<(), String> {
        if !scene && orb_v8::is_scene(json) {
            let parsed = orb_v8::parse_scene(json).map_err(|d| d.to_string())?;
            self.warnings
                .extend(parsed.warnings.iter().map(ToString::to_string));
            self.surface = Surface::new(parsed.projection, parsed.sources);
            self.lights = parsed.lights;
            return match parsed.style {
                orb_v8::StyleRef::Inline(style) => self.set_document(&style, base, true),
                orb_v8::StyleRef::Url(url) => {
                    let url = match base {
                        Some(base) => crate::source::join_url(base, &url),
                        None => url,
                    };
                    self.fetch_style(&url, true);
                    Ok(())
                }
            };
        }
        let doc = StyleDoc::parse(json).map_err(|d| d.to_string())?;
        if !scene {
            self.surface = Surface::default();
            self.lights = None;
        }
        // A style URL still loading is superseded.
        self.style_request += 1;
        self.install(doc);
        Ok(())
    }

    /// Forgets the GPU: after a device loss, or to draw on another device
    /// (plan §8.8). The next frame makes a renderer on its GPU and lays the
    /// style's tiles out again, from the jobs' cached tile frames where
    /// they have them (refetched where not); labels place afresh, and
    /// sprites and glyphs load again.
    pub fn reset_gpu(&mut self) {
        self.renderer = None;
        self.globe_warps = Default::default();
        *self.services.images.0.lock().expect("images poisoned") = ImageState::default();
        self.retired_pages.clear();
        self.retired_images.clear();
        self.retired_ramps.clear();
        let old = &self.labels;
        let mut labels = crate::labels::Labels::new(Arc::clone(&old.lane));
        labels.fade_duration = old.fade_duration;
        labels.cross_source_collisions = old.cross_source_collisions;
        labels.ideographs = old.ideographs.clone();
        self.labels = labels;
        *self.picking.get_mut() = Default::default();
        if let Some(style) = self.style.take() {
            self.generation += 1;
            self.install_revision(style.doc, Some((style.id, &style.worker)));
        }
    }

    /// Replaces the style with a document, every tile loaded fresh.
    pub(crate) fn install(&mut self, doc: StyleDoc) {
        self.install_revision(doc, None);
    }

    /// Installs a document: a new style, or with `previous` the same
    /// style again as a new revision of its jobs' style, which keeps their
    /// cached tile frames.
    fn install_revision(&mut self, mut doc: StyleDoc, previous: Option<(u64, &Arc<TileWorker>)>) {
        doc.set_cost_model(self.cost);
        self.version += 1;
        self.labels.invalidate();
        let worker = Arc::new(revise(
            previous.map(|(_, worker)| &**worker),
            &doc,
            self.version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        ));
        let plan = worker.plan();
        self.surface.set_globe(plan.style.globe);
        self.warnings
            .extend(plan.style.warnings.iter().map(ToString::to_string));
        self.warnings.extend(plan.warnings.iter().cloned());
        let mut style_camera = None;
        if previous.is_none() && std::mem::take(&mut self.camera_from_style) {
            if let Some([lon, lat]) = plan.style.center {
                self.camera.target = orb_geo::Geodetic::new(lon, lat);
            }
            if let Some(zoom) = plan.style.zoom {
                self.camera.set_zoom(self.surface.projection(), zoom);
            }
            self.camera.bearing = plan.style.bearing;
            self.camera.pitch = plan.style.pitch;
            self.camera.roll = plan.style.roll;
            style_camera = Some(self.camera);
        }
        self.events.wake();
        if previous.is_none() {
            self.events.push(MapEvent::StyleLoad {
                camera: style_camera,
            });
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
            id: self.style_id,
            doc,
            worker: Arc::clone(&worker),
            sources,
            retired: Vec::new(),
            painted: None,
            painted_fade: None,
            zoom_history: ZoomHistory::default(),
            image_namespace: None,
            sprites_started: false,
            sprites_loading: 0,
            icons: crate::SpriteIcons::default(),
            background: None,
            states: FeatureStates::default(),
            symbol_tiles: Vec::new(),
            programs: Default::default(),
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
                                    crate::source::archive_tilejson(&bytes)
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
                    &mut self.warnings,
                ),
            },
            (SourceKind::GeoJson(geojson), _) => ready_source(
                self.planned(&source.id, true, None)
                    .map(|(tms, warp)| (TileEndpoint::geojson(geojson, tms), warp)),
                &source.id,
                false,
                &mut self.warnings,
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

    /// Draws a frame into `target`: applies arrived data, loads what the
    /// view needs and renders. Every frame of a map uses the same `format`.
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TextureId,
        format: TargetFormat,
        viewport: Viewport,
        pixel_ratio: f32,
    ) -> FrameReport {
        match &self.renderer {
            // The renderer's pipelines are built for one target format.
            Some((_, current)) => {
                assert_eq!(*current, format, "a map renders to one target format")
            }
            None => {
                let mut images = self.services.images.0.lock().expect("images poisoned");
                let images = images
                    .atlas
                    .get_or_insert_with(|| ImageAtlas::new(gpu, 1024, 8, 8192));
                let renderer = Renderer::with_images(gpu, format, self.capacity, images.bindings());
                self.renderer = Some((renderer, format));
                self.nodes_stale = true;
                self.events.push(MapEvent::Capability {
                    tier: gpu.caps().tier,
                });
            }
        }
        self.renderer
            .as_mut()
            .expect("created above")
            .0
            .set_antialias_3d(gpu, self.antialias_3d);
        self.start_sprites(gpu, pixel_ratio);
        // Bring the scene's nodes to the plan before results for it apply.
        self.sync_nodes(gpu);
        self.placement_ms = 0.0;
        self.apply_messages(gpu);
        // A URL style may have arrived while applying messages.
        self.start_sprites(gpu, pixel_ratio);
        self.sync_nodes(gpu);

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
        let lights = self
            .lights
            .as_ref()
            .or(self.style.as_ref().map(|s| &s.worker.plan().style.lights));
        let globe = view
            .globe()
            .map(|g| crate::lights::globe_sky(g, &view, &self.camera, lights, zoom));
        self.renderer
            .as_mut()
            .expect("created above")
            .0
            .set_atmosphere(
                gpu,
                self.style
                    .as_ref()
                    .and_then(|style| style.worker.plan().style.atmosphere.as_ref())
                    .map(|atmosphere| orb_render::AtmospherePaint {
                        sky_color: atmosphere.sky_color.evaluate(zoom).premultiplied(),
                        horizon_color: atmosphere.horizon_color.evaluate(zoom).premultiplied(),
                        fog_color: atmosphere.fog_color.evaluate(zoom).premultiplied(),
                        sky_horizon_blend: atmosphere.sky_horizon_blend.evaluate(zoom),
                        horizon_fog_blend: atmosphere.horizon_fog_blend.evaluate(zoom),
                        fog_ground_blend: atmosphere.fog_ground_blend.evaluate(zoom),
                        atmosphere_blend: atmosphere.atmosphere_blend.evaluate(zoom),
                    }),
                orb_render::AtmosphereView {
                    horizon: horizon.as_vec2().to_array(),
                    horizon_normal: horizon_normal.as_vec2().to_array(),
                    height: viewport.height as f32,
                    fog_range: [fog_near as f32, fog_far as f32],
                    fog_opacity: fog_blend_opacity(self.camera.pitch),
                    globe,
                },
            );

        let now = self.services.clock.now();
        let mut fading = false;
        let (renderer, _) = self.renderer.as_mut().expect("created above");
        if let Some(style) = &mut self.style {
            let plan = style.worker.plan();
            if style.states.prepare(gpu, renderer, plan, zoom) {
                style.painted = None;
            }
            let crossfade = style.zoom_history.update(zoom, now);
            fading = crossfade.mix < 1.0
                && plan.nodes.iter().any(|n| {
                    plan.pattern(n).is_some()
                        || matches!(&plan.style.layers[n.layer].kind, LayerKind::Line(l) if l.paint.dasharray.as_constant() != Some(&None))
                });
            // Paint transitions write from the style overlaid with their
            // eased values, and once more when they end.
            let worker = Arc::clone(&style.worker);
            let overlay = self.transitions.overlay(worker.plan(), zoom, now);
            if overlay.is_some() || std::mem::take(&mut self.transitions.ran) {
                style.painted = None;
            }
            let mut images = self.services.images.0.lock().expect("images poisoned");
            write_paint(
                gpu,
                renderer,
                (style, overlay.as_ref().unwrap_or(worker.plan())),
                (zoom, self.camera.bearing),
                crossfade,
                images.atlas.as_mut().expect("initialized"),
                self.lights.as_ref(),
            );
            drop(images);
            style.symbol_tiles.clear();
            self.crossfade = crossfade;
        }
        if self.style.is_some() {
            // Terrain's DEM and cover first: the sources cover over its
            // elevations, and drape over its tiles.
            self.update_terrain(gpu, &view, &mut report);
            let elevation = self.terrain_elevation();
            for index in 0..self.style.as_ref().map_or(0, |s| s.sources.len()) {
                self.update_source(&view, index, elevation.as_deref(), &mut report);
            }
            let drape = self.plan_drape(gpu, &view);
            self.place_background(
                gpu,
                &view,
                elevation.as_deref(),
                drape.as_ref(),
                &mut report,
            );
            for index in 0..self.style.as_ref().map_or(0, |s| s.sources.len()) {
                self.place_source(gpu, &view, index, drape.as_ref(), &mut report, &mut drawn);
            }
        }
        let labels_settled = self.place_labels(gpu, &view, now);
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
        if picking.layers_version != Some(self.version)
            && let Some(style) = &self.style
        {
            picking.layers = Arc::new(crate::query::HitLayers::of(style.worker.plan()));
            picking.layers_version = Some(self.version);
        }
        picking.drew(view.clone(), self.camera, drawn, report.generation);
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
        report.idle =
            report.loaded && !fading && !self.labels.fading(now) && !self.transitions.running();
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
        self.events.frame(report.loaded, report.idle);

        let globe = view.globe().map(|globe| frame_globe(globe, &view));
        let frame = FrameBlock {
            view_proj: view.view_proj().as_mat4().to_cols_array_2d(),
            viewport: [viewport.width as f32, viewport.height as f32],
            world_per_px: view.world_per_px() as f32,
            pixel_ratio,
            zoom,
            bearing: self.camera.bearing.to_radians() as f32,
            extrusion_scale: self.extrusion_scale() as f32,
            ground_horizon: view.ground_horizon() as f32,
            symbol_fade_change: self.labels.fade_change(now),
            pitch: self.camera.pitch.to_radians() as f32,
            roll: self.camera.roll.to_radians() as f32,
            align: match globe {
                Some(_) => [0.0; 2],
                None => pixel_alignment(&view, self.surface.projection(), self.camera.bearing),
            },
            terrain: self.frame_terrain,
            lights: self
                .lights
                .as_ref()
                .or(self.style.as_ref().map(|s| &s.worker.plan().style.lights))
                .map(|lights| {
                    crate::lights::frame_lights(
                        lights,
                        zoom,
                        self.camera.bearing,
                        self.camera.target,
                    )
                }),
            globe,
        };
        let (renderer, _) = self.renderer.as_mut().expect("created above");
        report.frame = renderer.render(gpu, &frame, target, clear);
        self.globe_warps.end_frame(renderer.scene());
        report.scene = renderer.scene().stats();
        // Chunks the cull has no room for don't draw: say so when it starts.
        let dropped = report.scene.candidates_dropped;
        if dropped > 0 && self.candidates_dropped == 0 {
            let message = format!(
                "{dropped} chunks are not drawn: the scene's candidate or compacted-index \
                 memory is full"
            );
            self.events.push(MapEvent::Error {
                message: message.clone(),
            });
            self.warnings.push(message);
        }
        self.candidates_dropped = dropped;
        report
    }

    /// Makes the renderer's nodes the plan's: removes the meshes of nodes
    /// edits removed and the tiles of removed sources, keeps the others'
    /// meshes in their new places, and gives new image nodes their quads.
    fn sync_nodes(&mut self, gpu: &mut dyn Gpu) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
            return;
        };
        for source in style.retired.drain(..) {
            if let SourceState::Ready {
                mut pyramid, page, ..
            } = source
            {
                for tile in pyramid.drain() {
                    self.retired_ramps
                        .extend(remove_resident(renderer, tile, &mut style.states));
                }
                self.retired_pages.extend(page);
            }
        }
        let plan = style.worker.plan();
        // A projection edit put the map on or off a globe: the background
        // and image tiles take the meshes it bends.
        let globe = self.surface.globe().is_some();
        if style.globe_meshes != globe {
            style.globe_meshes = globe;
            if let Some(tile) = style.background.take() {
                renderer.scene().remove_tile(tile);
                style.background = make_background(gpu, renderer, plan, globe);
            }
            for (index, source) in style.sources.iter_mut().enumerate() {
                let SourceState::Ready {
                    pyramid,
                    endpoint,
                    page: Some(_),
                    warp,
                    ..
                } = source
                else {
                    continue;
                };
                let nodes: Vec<(u32, u32)> = (plan.nodes.iter().enumerate())
                    .filter(|(_, n)| n.source == index)
                    .map(|(i, n)| (i as u32, n.key))
                    .collect();
                for (tile, resident) in pyramid.ready_mut() {
                    let Ok(grid) = image_mesh(
                        &mut self.globe_warps,
                        globe,
                        warp.as_deref_mut(),
                        &endpoint.tms,
                        tile,
                    ) else {
                        continue;
                    };
                    let Some(image) = resident.image.as_mut() else {
                        continue;
                    };
                    if image.warped == grid {
                        continue;
                    }
                    image.warped = grid;
                    for mesh in resident.meshes.drain(..) {
                        renderer.scene().remove_mesh(mesh.mesh);
                    }
                    // A warped image draws through planar patches only on a
                    // globe surface; elsewhere through its warp's mesh.
                    renderer
                        .scene()
                        .remove_warp(std::mem::take(&mut resident.warp));
                    resident.series = None;
                    if let Some(warp) = warp.as_mut().filter(|_| globe) {
                        planar_patches(gpu, renderer, warp, tile, resident, &mut self.warnings);
                    }
                    for &(i, key) in &nodes {
                        if image::add_node(gpu, renderer, resident, tile, i, key).is_err() {
                            self.warnings
                                .push("out of scene memory; an image layer misses tiles".into());
                        }
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
            renderer.scene().remove_tile(tile);
        }
        for source in &mut style.sources {
            let SourceState::Ready { pyramid, .. } = source else {
                continue;
            };
            for (_, resident) in pyramid.ready_mut() {
                resident.symbols.retain(|s| keys.contains(&(s.layer * 2)));
                resident.meshes.retain_mut(|mesh| {
                    if keys.contains(&mesh.node) {
                        return true;
                    }
                    renderer.scene().remove_mesh(mesh.mesh);
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
        let families: Vec<Family> = plan.nodes.iter().map(|n| n.family).collect();
        if !gpu.caps().rgba16float_blendable && families.contains(&Family::Heatmap) {
            self.warnings.push(crate::COMPAT_NO_HEATMAP.into());
        }
        renderer.remap_nodes(gpu, &families, &retained);
        if !self.retired_pages.is_empty() {
            // The terrain may bind a retired page; it turns on again next.
            renderer.set_terrain(gpu, None);
        }
        for page in self.retired_pages.drain(..) {
            renderer.remove_page(gpu, page);
        }
        // New nodes of image sources draw the tiles already resident.
        for (index, node) in plan.nodes.iter().enumerate() {
            if retained[index].is_some() || node.source == usize::MAX {
                continue;
            }
            let SourceState::Ready {
                pyramid,
                page: Some(page),
                ..
            } = &mut style.sources[node.source]
            else {
                continue;
            };
            renderer.bind_page(index as u32, *page);
            for (tile, resident) in pyramid.ready_mut() {
                if image::add_node(gpu, renderer, resident, tile, index as u32, node.key).is_err() {
                    self.warnings
                        .push("out of scene memory; an image layer misses tiles".into());
                }
            }
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
            .style_height_scale(self.camera.target)
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
        let vector = style.worker.plan().image_kind(index).is_none();
        let SourceState::Ready { endpoint, warp, .. } = &mut style.sources[index] else {
            return;
        };
        let mut bend = warp
            .as_mut()
            .filter(|_| vector)
            .and_then(|warp| warp.series(tile))
            .map(|series| bend_of(series.grid(), [false; 2]));
        // On a globe, geometry bends with the sphere, and the world's edge
        // rows reach the poles.
        if vector && self.surface.globe().is_some() {
            let grid = self.globe_warps.grid(&endpoint.tms, tile);
            let poles = orb_crs::sphere::poles(&endpoint.tms, tile);
            if grid > 1 || poles.contains(&true) {
                let cells = bend.map_or(grid, |b| b.cells.max(grid));
                bend = Some(bend_of(cells, poles));
            }
        }
        let footprint = warp.as_mut().and_then(|warp| warp.footprint(tile));
        let zoom = self.surface.tile_zoom(&endpoint.tms, tile, footprint);
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
    fn place_labels(&mut self, gpu: &mut dyn Gpu, view: &View, now: f64) -> bool {
        let extrusion_scale = self.extrusion_scale();
        let (Some(style), Some((renderer, _))) = (&self.style, &mut self.renderer) else {
            return true;
        };
        let (sender, wake) = (self.sender.clone(), Arc::clone(&self.services.wake));
        let raster_done = move |rows: Vec<crate::labels::RasterRow>| -> crate::RasterDone {
            let (sender, wake) = (sender.clone(), Arc::clone(&wake));
            Box::new(move |result| {
                let _ = sender.send(Message::Raster { rows, result });
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
            self.surface.period(),
            &self.camera,
            self.camera.zoom(self.surface.projection()),
            extrusion_scale,
            now,
        );
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
                    let _ = sender.send(Message::Placed(committed));
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
        let zoom = self.camera.zoom(self.surface.projection());
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
                let LayerKind::FillExtrusion(e) = &plan.style.layers[node.layer].kind else {
                    unreachable!()
                };
                let maxima = style.nodes.get(&node.key).map_or(&[][..], |n| &n.maxima);
                let max = |i: usize| maxima.get(i).copied().unwrap_or(0.0);
                e.paint
                    .height
                    .evaluate(zoom)
                    .max(e.paint.base.evaluate(zoom))
                    .max(max(1))
                    .max(max(2))
            })
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
        let cover = match warp {
            None => view.cover_volume(
                tms,
                endpoint.cover_px,
                pyramid.max_zoom(),
                f64::from(height) * elevation_scale,
                elevation.map(|e| e as &dyn orb_camera::Elevation),
            ),
            Some(warp) => view.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                warp.footprint(tile)
            }),
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
    fn place_source(
        &mut self,
        gpu: &mut dyn Gpu,
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
        let (renderer, _) = self.renderer.as_mut().expect("created before placing");
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
        let camera_world = view.camera_world();
        let period = self.surface.period();
        let placed = &mut style.placed[index];
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
                Some(globe) => self
                    .globe_warps
                    .warp(gpu, renderer.scene(), tms, tile)
                    .map(|(warp, anchor)| Some((warp, anchor - globe.target))),
                None => on_plane.then_some(None),
            };
            // Nowhere to draw it this frame: no placements.
            let Some(globe) = globe else {
                for &(_, scene_tile) in &resident.parts {
                    renderer
                        .scene()
                        .place_tile(gpu, scene_tile, &[])
                        .expect("removing placements always fits");
                }
                continue;
            };
            for &(extent, scene_tile) in &resident.parts {
                let at = Placement {
                    extent,
                    camera_world,
                    period,
                    zoom: self.camera.zoom(self.surface.projection()),
                    frame,
                    world_per_px: view.world_per_px(),
                    series: series.as_deref(),
                    warp: resident.warp,
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
                                crate::terrain::drape_rows(
                                    &drape.surface,
                                    drawn,
                                    extent,
                                    (terrain, slot),
                                    &drape.layout,
                                    map_rows(draw).into_iter(),
                                    &mut rows,
                                );
                            }
                        }
                    }
                }
                report.tiles_drawn += rows.len() as u32;
                if renderer.scene().place_tile(gpu, scene_tile, &rows).is_err() {
                    self.warnings
                        .push("out of tile placements; some tiles are not drawn".into());
                }
            }
        }
        for tile in placed.iter().filter(|t| !rows.contains_key(t)) {
            if let Some(resident) = pyramid.get(*tile) {
                for &(_, scene_tile) in &resident.parts {
                    renderer
                        .scene()
                        .place_tile(gpu, scene_tile, &[])
                        .expect("removing placements always fits");
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
                .flat_map(|tile| &pyramid.get(*tile).expect("drawn tiles are ready").parts)
                .map(|&(_, tile)| tile)
                .collect();
            renderer.scene().set_tile_order(&tiles);
            *placed = ordered;
        }
        for resident in pyramid.evict() {
            self.retired_ramps
                .extend(remove_resident(renderer, resident, &mut style.states));
        }
    }

    /// Keeps the terrain to the style's `terrain`: made once its DEM
    /// source is ready, its DEM tiles requested for this view, and the
    /// renderer told, once a DEM tile has arrived.
    fn update_terrain(&mut self, gpu: &mut dyn Gpu, view: &View, report: &mut FrameReport) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
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
        let requests = terrain.update(view, self.surface.tms());
        for retired in terrain.evict() {
            if let Some(page) = terrain.page {
                renderer.free_layer(page, retired.layer);
            }
        }
        let paint = terrain.page.map(|page| {
            let zoom = self.camera.zoom(self.surface.projection());
            let height_scale = view.height_scale();
            // GL JS's `getSkirtLength`: a fifth of a tile's width at the
            // camera's zoom, in metres at the equator.
            let skirt = 2.0 * std::f64::consts::PI * 6_371_008.8 / zoom.max(0.0).exp2() / 5.0;
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
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
    ) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
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
                terrain.pyramid.insert_empty(tile);
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
        for job in image::dem_prepares(tile, layer, unpack, &layer_of) {
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

    /// Plans this frame's drape: makes room for the terrain tiles' slots
    /// and picks those to redraw, by what is draped under each tile and
    /// the zoom it was drawn at.
    fn plan_drape(&mut self, gpu: &mut dyn Gpu, view: &View) -> Option<DrapePlan> {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
            return None;
        };
        let terrain = style.terrain.as_mut()?;
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
        let plan = style.worker.plan();
        let draped: Vec<usize> = (0..style.sources.len())
            .filter(|&i| {
                plan.nodes
                    .iter()
                    .any(|n| n.source == i && n.family.is_draped())
            })
            .collect();
        let surface = self.surface.tms().clone();
        let period = self.surface.period();
        let sources = &style.sources;
        let draws = &style.draws;
        let revision = (self.version, style.states.writes());
        let content = |t: orb_crs::CoveredTile| {
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
                    let meshes: Vec<_> = resident
                        .map(|r| r.meshes.iter().map(|m| m.mesh).collect())
                        .unwrap_or_default();
                    parts.push((i, draw.tile, draw.wrap, draw.regions.clone(), meshes));
                }
            }
            crate::terrain::hash((revision, parts))
        };
        // Viewport-anchored hillshades turn their light with the map: the
        // bearing joins the look, so they redraw once the camera stops.
        let zoom = self.camera.zoom(self.surface.projection());
        let turns = plan.nodes.iter().any(|n| {
            matches!(&plan.style.layers[n.layer].kind, LayerKind::Hillshade(h)
                if h.paint.illumination_anchor.evaluate(zoom) == orb_v8::IlluminationAnchor::Viewport)
        });
        let look = [
            zoom.to_bits(),
            u64::from(self.crossfade.mix.to_bits())
                | u64::from(self.crossfade.from_scale.to_bits()) << 32,
            if turns {
                self.camera.bearing.to_bits()
            } else {
                0
            },
        ];
        let (tiles, redraw) = terrain.plan(&surface, view, &self.camera, layout, &content, look);
        renderer.set_terrain_tiles(gpu, &tiles);
        let slots: Vec<u32> = redraw.iter().map(|&(_, slot)| slot).collect();
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
        view: &View,
        elevation: Option<&crate::terrain::TerrainElevation>,
        drape: Option<&DrapePlan>,
        report: &mut FrameReport,
    ) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
            return;
        };
        // Backgrounds draw on every covered tile, as GL JS's do: never
        // beyond the world's edge.
        let Some(tile) = style.background else {
            return;
        };
        let zoom = self.camera.zoom(self.surface.projection());
        let tms = self.surface.tms();
        let cover = view.cover_volume(
            tms,
            TILE_PX,
            24,
            0.0,
            elevation.map(|e| e as &dyn orb_camera::Elevation),
        );
        let period = self.surface.period();
        let at = Placement {
            extent: 8192,
            camera_world: view.camera_world(),
            period,
            zoom,
            frame: None,
            world_per_px: view.world_per_px(),
            series: None,
            warp: TileWarp::NONE,
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
                    Some(globe) => {
                        let (warp, anchor) =
                            globe_warps.warp(gpu, renderer.scene(), tms, covered.index)?;
                        Some((warp, anchor - globe.target))
                    }
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
        if renderer.scene().place_tile(gpu, tile, &rows).is_err() {
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
    fn apply_messages(&mut self, gpu: &mut dyn Gpu) {
        let started = self.services.clock.now();
        self.tiles_deferred = false;
        for waiting in std::mem::take(&mut self.waiting) {
            let (bytes, image) = host_data(&waiting.build);
            let release = |gpu: &mut dyn Gpu| {
                if let Some(bytes) = bytes {
                    gpu.release_host_bytes(bytes);
                }
                if let Some(image) = image {
                    gpu.release_host_image(image);
                }
            };
            if self.tiles_deferred || waiting.generation != self.tile_generation {
                if waiting.generation == self.tile_generation {
                    self.waiting.push(waiting);
                } else {
                    release(gpu);
                }
                continue;
            }
            match self.apply_tile(gpu, waiting.source, waiting.tile, Ok(waiting.build), false) {
                Some(build) => self.waiting.push(WaitingTile { build, ..waiting }),
                None => release(gpu),
            }
            self.tiles_deferred = self.services.clock.now() - started > TILE_APPLY_MS;
        }
        if self.tiles_deferred {
            return;
        }
        while let Ok(message) = self.receiver.try_recv() {
            let new_tile = matches!(message, Message::Tile { edit: None, .. });
            match message {
                Message::Placed(committed) => {
                    if let Some((renderer, _)) = &mut self.renderer {
                        let started = self.services.clock.now();
                        self.labels.committed(gpu, renderer, committed);
                        self.placement_ms += self.services.clock.now() - started;
                    }
                }
                Message::Raster { rows, result } => {
                    if let Some((renderer, _)) = &mut self.renderer {
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
                                self.events.push(MapEvent::Error {
                                    message: message.clone(),
                                });
                                self.warnings.push(message);
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
                    style.sources[index] =
                        ready_source(endpoint, &def.id, image, &mut self.warnings);
                    if matches!(style.sources[index], SourceState::Failed) {
                        let message = self.warnings.last().cloned().unwrap_or_default();
                        self.events.push(MapEvent::Error { message });
                    }
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
                            None if terrain => self.apply_terrain_tile(gpu, source, tile, result),
                            None => {
                                if let Some(build) =
                                    self.apply_tile(gpu, source, tile, result, true)
                                {
                                    kept = true;
                                    self.waiting.push(WaitingTile {
                                        generation,
                                        source,
                                        tile,
                                        build,
                                    });
                                }
                            }
                            Some(_) => self.apply_edit(gpu, source, tile, result),
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
    }

    fn finish_job(&mut self, edit: u64) {
        if let Some(count) = self.pending.get_mut(&edit) {
            *count -= 1;
            if *count == 0 {
                self.pending.remove(&edit);
            }
        }
    }

    fn apply_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
        arrived: bool,
    ) -> Option<TileResult> {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
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
                self.events.push(MapEvent::Error {
                    message: message.clone(),
                });
                self.warnings.push(message);
                pyramid.insert_empty(tile);
                return None;
            }
        };
        self.warnings.append(&mut build.warnings);
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
                &endpoint.tms,
                tile,
            ) else {
                // Nothing of the tile reaches the surface.
                pyramid.insert_empty(tile);
                return None;
            };
            let upload = image::Upload {
                plan,
                source,
                kind,
                tile,
                image: tile_image,
                unpack: endpoint.dem_unpack,
                warped,
                zoom: tile_zoom,
            };
            // Room as for vector tiles: unused tiles of any source go first;
            // while other tiles load, this one waits for theirs to go; then
            // it gives up for what stands in.
            let sizes = image::mesh_sizes(plan, source, upload.warped.as_deref());
            let others_loading = style
                .sources
                .iter()
                .any(|s| matches!(s, SourceState::Ready { loading, .. } if *loading > 0));
            let uploaded = loop {
                let tried = renderer.scene().fits(sizes.iter().copied()).then(|| {
                    let SourceState::Ready { pyramid, page, .. } = &mut style.sources[source]
                    else {
                        unreachable!("checked above");
                    };
                    image::upload(gpu, renderer, page, pyramid, &upload)
                });
                match tried {
                    Some(Ok(resident)) => break Ok(resident),
                    Some(Err(image::UploadError::Invalid(e))) => break Err(Some(e)),
                    Some(Err(image::UploadError::NoRoom)) | None => {
                        match evict_unused(&mut style.sources, source) {
                            Some(victim) => self.retired_ramps.extend(remove_resident(
                                renderer,
                                victim,
                                &mut style.states,
                            )),
                            None => break Err(None),
                        }
                    }
                }
            };
            let SourceState::Ready { pyramid, warp, .. } = &mut style.sources[source] else {
                unreachable!("checked above");
            };
            match uploaded {
                Ok(mut resident) => {
                    // On a globe, a warped image draws in its own space:
                    // through planar patches on the plane.
                    if let Some(warp) = warp.as_mut().filter(|_| globe) {
                        planar_patches(
                            gpu,
                            renderer,
                            warp,
                            tile,
                            &mut resident,
                            &mut self.warnings,
                        );
                    }
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
                    self.events.push(MapEvent::Error {
                        message: message.clone(),
                    });
                    self.warnings.push(message);
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
        let mut atlas = self.services.images.0.lock().expect("images poisoned");
        // Make room for the whole tile before writing any of it: an upload
        // that runs out part way starts over, so making room one eviction
        // at a time would write the tile again for every tile evicted.
        let sizes = mesh_sizes(plan, &build);
        let mut fits = renderer.scene().fits(sizes.iter().copied());
        while !fits {
            let Some(victim) = evict_unused(&mut style.sources, source) else {
                break;
            };
            self.retired_ramps
                .extend(remove_resident(renderer, victim, &mut style.states));
            fits = renderer.scene().fits(sizes.iter().copied());
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
            match upload(
                gpu,
                renderer,
                plan,
                &build,
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
        let SourceState::Ready { pyramid, warp, .. } = &mut style.sources[source] else {
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
            self.events.push(MapEvent::Error {
                message: message.clone(),
            });
            self.warnings.push(message);
            pyramid.insert_missing(tile);
            return None;
        };
        // A tile on another plane draws through its series' patches.
        if let Some(warp) = warp.as_mut() {
            planar_patches(gpu, renderer, warp, tile, &mut resident, &mut self.warnings);
        }
        self.counts.meshes_uploaded += build.meshes.len() as u32;
        record_maxima(&mut style.nodes, &build.meshes, &mut style.painted);
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
        if let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) {
            if let Some(tile) = style.background.take() {
                renderer.scene().remove_tile(tile);
            }
            style.states.release_table(renderer);
            style.programs.release(renderer);
        }
        self.style = None;
    }

    /// Removes the style's tiles from the scene; they reload as needed.
    fn clear_style_tiles(&mut self) {
        let Some(style) = &mut self.style else { return };
        for source in style.sources.iter_mut().chain(&mut style.retired) {
            if let SourceState::Ready {
                pyramid,
                loading,
                page,
                ..
            } = source
            {
                let tiles = pyramid.drain();
                *loading = 0;
                if let Some((renderer, _)) = &mut self.renderer {
                    for tile in tiles {
                        self.retired_ramps.extend(remove_resident(
                            renderer,
                            tile,
                            &mut style.states,
                        ));
                    }
                }
                // Destroyed once the nodes that bind it are replaced.
                self.retired_pages.extend(page.take());
            }
        }
        for placed in &mut style.placed {
            placed.clear();
        }
        if let Some(terrain) = style.terrain.take() {
            // Destroyed with the source pages, the renderer's terrain off.
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
    warnings: &mut Vec<String>,
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
                if image { IMAGE_TILE_CACHE } else { TILE_CACHE },
            ),
            endpoint: Arc::new(endpoint),
            loading: 0,
            page: None,
        },
        Err(e) => {
            warnings.push(format!("source {id}: {e}"));
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

/// What a mesh's rows refer to besides their words: images patched into
/// them, retained ramps, and the feature-state remap.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attach_rows(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    result: &TileResult,
    mesh: &ResultMesh,
    id: MeshId,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    warnings: &mut Vec<String>,
) -> Result<(Vec<u32>, Vec<u32>), OutOfSpace> {
    let mut ramps = Vec::new();
    for image in &mesh.images {
        match atlas.resolve_key(gpu, namespace, &image.key) {
            Ok(image_id) => {
                if matches!(image.key, orb_render::ImageKey::Ramp { .. }) {
                    atlas.retain_ramp(image_id);
                    ramps.push(image_id);
                }
                renderer
                    .scene()
                    .set_mesh_image(gpu, id, image.word, image_id);
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
            .map(|&i| match result.ids.get(i as usize) {
                Some(id) => states.intern(source, layer, id),
                None => 0,
            })
            .collect();
    }
    if let Err(e) = renderer.scene().set_mesh_state(gpu, id, &indices) {
        for &index in &indices {
            states.release(index);
        }
        atlas.release_ramps(gpu, ramps);
        return Err(e);
    }
    Ok((ramps, indices))
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
    let part = match resident
        .parts
        .iter()
        .find(|(extent, _)| *extent == mesh.extent)
    {
        Some(&(_, id)) => id,
        None => {
            let id = renderer.scene().insert_tile();
            resident.parts.push((mesh.extent, id));
            id
        }
    };
    let data = MeshData {
        vertices: words(&mesh.vertices, result.host_bytes),
        indices: words(&mesh.indices, result.host_bytes),
        props: words(&mesh.props, result.host_bytes),
        row_words: mesh.row_words,
        chunks: &mesh.chunks,
    };
    let id = renderer.scene().insert_mesh(gpu, part, node, &data)?;
    let attached = attach_rows(
        gpu, renderer, plan, result, mesh, id, atlas, namespace, states, warnings,
    );
    let (ramps, indices) = match attached {
        Ok(attached) => attached,
        Err(e) => {
            renderer.scene().remove_mesh(id);
            return Err(e);
        }
    };
    resident.meshes.push(Resident {
        hits: mesh.hits.clone(),
        node: mesh.node,
        mesh: id,
        layout_rev: result.revision,
        rows_rev: result.revision,
        ramps,
        states: indices,
    });
    Ok(())
}

/// Uploads a tile's meshes, one scene tile per layer extent: all of them or,
/// when the scene is out of space, none.
#[allow(clippy::too_many_arguments)]
fn upload(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    result: &TileResult,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    labels: &mut crate::labels::Labels,
    tile_zoom: f64,
    warnings: &mut Vec<String>,
) -> Result<ResidentTile, OutOfSpace> {
    let mut resident = ResidentTile {
        zoom: tile_zoom,
        ..ResidentTile::default()
    };
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

/// Removes a tile from the scene. Returns the ramps its rows retained.
pub(crate) fn remove_resident(
    renderer: &mut Renderer,
    tile: ResidentTile,
    states: &mut FeatureStates,
) -> Vec<u32> {
    for (_, id) in tile.parts {
        renderer.scene().remove_tile(id);
    }
    renderer.scene().remove_warp(tile.warp);
    if let Some(image) = tile.image {
        renderer.free_layer(image.page, image.layer);
    }
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
    camera_world: DVec2,
    /// The surface's wrap period, 0 if it doesn't wrap.
    period: f64,
    /// The camera's style zoom.
    zoom: f64,
    /// A warped tile's frame on the surface, which its mesh spans `extent`
    /// of; tiles on the surface's plane span their own bounds.
    frame: Option<orb_crs::WorldRect>,
    /// World units per logical pixel at the camera target.
    world_per_px: f64,
    /// A vector tile drawn through a warp's patches, relative to its
    /// anchor, and the patches in the scene.
    series: Option<&'a orb_crs::Series>,
    warp: TileWarp,
    /// On a globe: the tile's patches on the sphere, and their anchor's
    /// camera-relative position.
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
        camera_world,
        period,
        zoom,
        frame,
        world_per_px,
        series,
        warp,
        globe,
        on_plane,
        tile_zoom,
    } = at;
    let world_per_px = world_per_px as f32;
    let bounds = frame.unwrap_or_else(|| tms.tile_bounds(draw.tile));
    let wrap = DVec2::new(f64::from(draw.wrap) * period, 0.0);
    let origin = match series {
        Some(series) => series.anchor + wrap - camera_world,
        None => DVec2::new(bounds.min.x, bounds.max.y) + wrap - camera_world,
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
            origin: [origin.x as f32, origin.y as f32],
            scale,
            zoom: tile_zoom as f32,
            clips,
            world_per_px,
            dem: DemRef::NONE,
            terrain_range: [0.0; 2],
            target: PlacementTarget::Map,
            warp,
            globe: globe.map(|(warp, origin)| orb_render::RowGlobe {
                warp,
                origin: origin.as_vec3().to_array(),
                on_plane,
            }),
        }
    })
}

/// A globe view's frame terms: its matrix, the target, and the horizon
/// plane that depth and the far side's clipping come from.
fn frame_globe(globe: &orb_camera::GlobeView, view: &View) -> orb_render::FrameGlobe {
    let radius = orb_crs::sphere::RADIUS;
    let eye = globe.target + globe.eye;
    let distance = eye.length();
    let toward = eye / distance;
    // `facing` is 1 under the eye and 0 at the horizon circle.
    let horizon = radius * radius / distance;
    let scale = 1.0 / (radius - horizon).max(1e-9);
    let n = toward * scale;
    orb_render::FrameGlobe {
        view_proj: globe.view_proj.as_mat4().to_cols_array_2d(),
        target: globe.target.as_vec3().to_array(),
        amount: globe.amount as f32,
        horizon: [
            n.x as f32,
            n.y as f32,
            n.z as f32,
            ((globe.target.dot(toward) - horizon) * scale) as f32,
        ],
        metres_per_unit: (globe.metres_per_px / view.world_per_px()) as f32,
        eye: globe.eye.as_vec3().to_array(),
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

/// Writes every node's paint and visibility for `zoom`, if it changed.
fn write_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (style, plan): (&mut LoadedStyle, &StylePlan),
    (zoom, bearing): (f64, f64),
    crossfade: Crossfade,
    atlas: &mut ImageAtlas,
    lights: Option<&orb_v8::Lights>,
) {
    if style.painted == Some((zoom, bearing)) && style.painted_fade == Some(crossfade) {
        return;
    }
    style.painted = Some((zoom, bearing));
    style.painted_fade = Some(crossfade);
    style.programs.prepare(gpu, renderer, plan);
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
            && match &layer.kind {
                LayerKind::FillExtrusion(e) => e.paint.opacity.evaluate(zoom) > 0.0,
                _ => true,
            };
        renderer.set_visible(index, visible);
        if node.family.is_image() {
            let unpack = match &style.sources[node.source] {
                SourceState::Ready { endpoint, .. } => endpoint.dem_unpack,
                _ => None,
            };
            let context = (plan, node, index);
            if let Some(paint) = image::paint(
                gpu,
                renderer,
                context,
                zoom,
                bearing,
                unpack,
                &mut state.ramp,
            ) {
                renderer.set_paint(gpu, index, &paint);
            }
            continue;
        }
        let properties = plan.paint_properties(node);
        if state.maxima.len() < properties.len() {
            state.maxima.resize(properties.len(), 0.0);
        }
        let lookup = style.states.lookup(index as usize);
        let interpreted = style.programs.paint(plan, index as usize, &style.states);
        if crate::point::paint(
            gpu,
            renderer,
            plan,
            node,
            index,
            zoom,
            &state.maxima,
            lookup,
            interpreted,
        ) {
            continue;
        }
        let maxima = &state.maxima;
        let mut table_mask = 0;
        let mut curves = [Curve::Step; 6];
        let mut numbers = [0.0f32; 6];
        let mut color = [0.0f32; 4];
        for (slot, property) in properties.iter().enumerate() {
            if interpreted.mask & 1 << slot != 0 {
                numbers[slot] = crate::gpu::bound(plan, node, slot, zoom, maxima);
            } else if property.per_feature() {
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
        let translate = match &plan.style.layers[node.layer].kind {
            LayerKind::Fill(fill) => Translate {
                px: fill.paint.translate.evaluate(zoom),
                viewport: fill.paint.translate_anchor.evaluate(zoom) == TranslateAnchor::Viewport,
            },
            LayerKind::Line(line) => Translate {
                px: line.paint.translate.evaluate(zoom),
                viewport: line.paint.translate_anchor.evaluate(zoom) == TranslateAnchor::Viewport,
            },
            LayerKind::FillExtrusion(e) => Translate {
                px: e.paint.translate.evaluate(zoom),
                viewport: e.paint.translate_anchor.evaluate(zoom) == TranslateAnchor::Viewport,
            },
            LayerKind::Background(_) => Translate::default(),
            _ => unreachable!("drawable layer"),
        };
        let paint = match node.family {
            Family::Fill | Family::FillOutline => NodePaint::Fill(FillPaint {
                color,
                opacity: numbers[1],
                translate,
                table_mask,
                curves: [curves[0], curves[1]],
                state: lookup,
                gpu: interpreted,
            }),
            Family::FillExtrusion => {
                let LayerKind::FillExtrusion(e) = &plan.style.layers[node.layer].kind else {
                    unreachable!()
                };
                NodePaint::Extrusion(orb_render::ExtrusionPaint {
                    color,
                    height: numbers[1],
                    base: numbers[2],
                    opacity: e.paint.opacity.evaluate(zoom),
                    translate,
                    vertical_gradient: e.paint.vertical_gradient.evaluate(zoom),
                    emissive: lights.map_or(0.0, |l| l.emissive(&plan.style.layers[node.layer].id))
                        as f32,
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
                gpu: interpreted,
            }),
            image => unreachable!("{image:?} nodes are painted above"),
        };
        let paint = crate::line::paint(
            gpu,
            paint,
            &plan.style.layers[node.layer].kind,
            atlas,
            style.image_namespace.unwrap(),
            zoom,
            crossfade,
        );
        let paint = pattern_paint(
            paint,
            plan.pattern(node),
            style.image_namespace.unwrap(),
            atlas,
            zoom,
            crossfade,
            match &plan.style.layers[node.layer].kind {
                LayerKind::Line(line) => line.paint.width.evaluate(zoom.floor()),
                _ => 0.0,
            },
        );
        let paint = if matches!(plan.style.layers[node.layer].kind, LayerKind::Background(_))
            && plan.pattern(node).is_none()
        {
            match paint {
                NodePaint::Fill(p) => NodePaint::Background(p),
                _ => unreachable!("solid background"),
            }
        } else {
            paint
        };
        renderer.set_paint(gpu, index, &paint);
        // A fill or line layer below layer opacity 1 composites once.
        let layer_opacity = match &layer.kind {
            LayerKind::Fill(f) => Some(f.paint.layer_opacity.evaluate(zoom)),
            LayerKind::Line(l) => Some(l.paint.layer_opacity.evaluate(zoom)),
            _ => None,
        };
        if let Some(opacity) = layer_opacity {
            renderer.set_layer_opacity(gpu, index, node.layer as u32, opacity);
        }
    }
}

fn bend_of(cells: u32, [north, south]: [bool; 2]) -> crate::Bend {
    crate::Bend {
        cells,
        north,
        south,
    }
}

/// Gives a tile on another plane its planar patches, if the warp has them
/// (it draws through them on the plane, and as a globe hands over).
fn planar_patches(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    warp: &mut WarpState,
    tile: TileIndex,
    resident: &mut ResidentTile,
    warnings: &mut Vec<String>,
) {
    let Some(series) = warp.series(tile) else {
        return;
    };
    let patches: Vec<f32> = series
        .patches
        .iter()
        .flat_map(|p| p.points.iter().flat_map(|v| [v.x as f32, v.y as f32]))
        .collect();
    match renderer.scene().add_warp(gpu, series.cells, &patches) {
        Ok(warp) => resident.warp = warp,
        Err(OutOfSpace) => warnings.push(format!(
            "tile {tile:?}: out of property memory for its warp"
        )),
    }
    resident.series = Some(series);
}

/// An image tile's mesh: on a globe, a grid in the tile's own space that
/// the sphere and the plane's patches bend; on another plane, the warp's
/// mesh; else `None` for its quad. `Err` when nothing of it reaches the
/// surface.
fn image_mesh(
    globe_warps: &mut crate::globe::GlobeWarps,
    globe: bool,
    warp: Option<&mut WarpState>,
    tms: &TileMatrixSet,
    tile: TileIndex,
) -> Result<Option<Arc<image::Warped>>, ()> {
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
) -> Option<TileId> {
    let mut tile = None;
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
    for (i, _) in plan
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.source == usize::MAX)
    {
        let id = *tile.get_or_insert_with(|| renderer.scene().insert_tile());
        renderer
            .scene()
            .insert_mesh(gpu, id, i as u32, &MeshData::new(&mesh, &[], 0))
            .expect("background mesh fits");
    }
    tile
}

fn pattern_paint(
    paint: NodePaint,
    property: Option<&orb_v8::Property<Option<String>>>,
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
        let target = orb_geo::Geodetic::new(139.74, 35.685);
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
                map.camera_mut()
                    .zoom_around(&surface, viewport, center, zoom);
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
            camera_world: DVec2::ZERO,
            period: 100.0,
            zoom: 3.0,
            frame: None,
            world_per_px: 1.0,
            series: None,
            warp: TileWarp::NONE,
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
        assert_eq!(map.projection().descriptor().kind, "EPSG:3413");
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
