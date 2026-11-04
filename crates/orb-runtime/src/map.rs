use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_camera::{CameraState, View, Viewport};
use orb_crs::{TileIndex, TileMatrixSet};
use orb_hal::{Gpu, HostBytesId, TextureId};
use orb_host::{Fetch, FetchError, ImageDecode};
use orb_render::{
    ClipRect, Curve, Family, FillPaint, FrameBlock, FrameStats, LinePaint, MeshData, MeshId,
    NodePaint, OutOfSpace, PageId, PatternPaint, Renderer, SceneCapacity, SceneStats, TargetFormat,
    TileId, TileRow, Translate, Words,
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
        result: Result<TileResult, String>,
    },
}

/// A tile in the scene: one scene tile per layer extent holding a mesh per
/// render node with features, and an image tile's texture layer.
#[derive(Debug, Default)]
pub(crate) struct ResidentTile {
    pub(crate) parts: Vec<(u32, TileId)>,
    pub(crate) meshes: Vec<Resident>,
    pub(crate) image: Option<ResidentImage>,
    /// Its symbol layers' buckets; their meshes are among `meshes`.
    pub(crate) symbols: Vec<crate::labels::ResidentSymbols>,
}

/// One node's mesh of a resident tile.
#[derive(Debug)]
pub(crate) struct Resident {
    /// The node's key.
    pub(crate) node: u32,
    pub(crate) mesh: MeshId,
    pub(crate) hits: Vec<crate::HitFeature>,
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
    },
    Failed,
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
}

/// A map: a style, a camera and the tiles they need, drawn by
/// [`render`](Self::render).
pub struct Map {
    /// Placement and the scene side of labels.
    pub(crate) labels: crate::labels::Labels,
    pub(crate) hits: Vec<crate::query::ScreenHit>,
    /// What the picking index was built for: camera, viewport, style
    /// revision, uploads and label commits.
    hit_stamp: Option<(CameraState, Viewport, u64, JobCounts, u64)>,
    pub(crate) rebuild_hits: bool,
    /// This frame's placement time so far, in milliseconds.
    placement_ms: f64,
    pub(crate) services: Services,
    camera: CameraState,
    antialias_3d: bool,
    cost: CostModel,
    tms: TileMatrixSet,
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
    }
}

/// A revision of `doc`'s plan, keeping `previous`'s source data.
pub(crate) fn revise(
    previous: Option<&TileWorker>,
    doc: &StyleDoc,
    revision: u64,
    local_ideographs: Option<String>,
) -> TileWorker {
    TileWorker::with_style(
        previous,
        Arc::clone(doc.style()),
        doc.layer_keys().map(|k| k.0).collect(),
        doc.source_keys().map(|k| k.0).collect(),
        revision,
        doc.cost_model(),
        local_ideographs,
    )
}

impl Map {
    /// A map with no style. With `camera: None`, the first style's camera
    /// is used, as GL JS does when a map is created without one.
    pub fn new(services: Services, camera: Option<CameraState>) -> Self {
        let (sender, receiver) = channel();
        let camera_from_style = camera.is_none();
        let camera =
            camera.unwrap_or_else(|| CameraState::new(orb_geo::Geodetic::new(0.0, 0.0), 0.0));
        Self {
            hits: Vec::new(),
            hit_stamp: None,
            rebuild_hits: true,
            placement_ms: 0.0,
            services,
            camera,
            tms: TileMatrixSet::by_id("WebMercatorQuad").expect("built-in tile matrix set"),
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
            self.camera.zoom(),
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

    /// The camera and viewport the picking index was built for.
    pub(crate) fn hit_stamp_view(&self) -> Option<(CameraState, Viewport)> {
        self.hit_stamp
            .map(|(camera, viewport, ..)| (camera, viewport))
    }

    /// The surface's projection, for camera moves.
    pub fn projection(&self) -> orb_crs::Projection {
        *self.tms.projection()
    }

    /// The view the camera gives for a viewport.
    pub fn view(&self, viewport: Viewport) -> View {
        View::new(&self.camera, viewport, self.tms.projection())
    }

    /// Keeps the camera to the world's edge, as GL JS's transform does
    /// (`getConstrained`). Camera stores keep to it themselves; hosts that
    /// write the camera directly call this, since rendering never writes
    /// the camera.
    pub fn constrain_camera(&mut self, viewport: Viewport) {
        let projection = *self.tms.projection();
        self.camera
            .constrain(&orb_camera::Constraints::WORLD, &projection, viewport);
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

    /// Loads a style from a URL; it replaces the current one when it
    /// arrives.
    pub fn load_style_url(&mut self, url: &str) {
        self.style_request += 1;
        let request = self.style_request;
        self.fetch_text(url, move |result| Message::Style { request, result });
    }

    /// Replaces the style with a v8 document. Its tiles load as frames
    /// render.
    pub fn set_style_json(&mut self, json: &str) -> Result<(), String> {
        let doc = StyleDoc::parse(json).map_err(|d| d.to_string())?;
        // A style URL still loading is superseded.
        self.style_request += 1;
        self.install(doc);
        Ok(())
    }

    /// Replaces the style with a document, every tile loaded fresh.
    pub(crate) fn install(&mut self, mut doc: StyleDoc) {
        doc.set_cost_model(self.cost);
        self.version += 1;
        self.labels.invalidate();
        let worker = Arc::new(revise(
            None,
            &doc,
            self.version,
            self.labels.ideographs.clone(),
        ));
        let plan = worker.plan();
        self.warnings
            .extend(plan.style.warnings.iter().map(ToString::to_string));
        self.warnings.extend(plan.warnings.iter().cloned());
        let mut style_camera = None;
        if std::mem::take(&mut self.camera_from_style) {
            if let Some([lon, lat]) = plan.style.center {
                self.camera.target = orb_geo::Geodetic::new(lon, lat);
            }
            if let Some(zoom) = plan.style.zoom {
                self.camera.set_zoom(zoom);
            }
            self.camera.bearing = plan.style.bearing;
            self.camera.pitch = plan.style.pitch;
            style_camera = Some(self.camera);
        }
        self.events.wake();
        self.events.push(MapEvent::StyleLoad {
            camera: style_camera,
        });
        self.events.push(MapEvent::StyleData {
            generation: self.version,
        });
        self.clear_style();
        self.pending.clear();
        self.style_id += 1;
        self.services.jobs.set_style(
            self.style_id,
            &full_update(&doc, self.version, self.labels.ideographs.clone()),
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
                    TileEndpoint::resolve(&source.kind, None),
                    &source.id,
                    plan.image_kind(index).is_some(),
                    &mut self.warnings,
                ),
            },
            (SourceKind::GeoJson(geojson), _) => ready_source(
                Ok(TileEndpoint::geojson(geojson)),
                &source.id,
                false,
                &mut self.warnings,
            ),
            _ => unreachable!("plan sources are drawable"),
        }
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
        let stamp = (
            self.camera,
            viewport,
            self.version,
            self.counts,
            self.labels.commits(),
        );
        self.rebuild_hits = self.hit_stamp != Some(stamp);
        if self.rebuild_hits {
            self.hits.clear();
            self.hit_stamp = Some(stamp);
        }
        let zoom = self.camera.zoom();
        let mut report = FrameReport::default();
        let clear = self.clear_color;
        self.renderer.as_mut().expect("created above").0.set_sky(
            gpu,
            self.style
                .as_ref()
                .and_then(|style| style.worker.plan().style.sky.as_ref())
                .map(|sky| orb_render::SkyPaint {
                    color: sky.color.evaluate(zoom).premultiplied(),
                    horizon_color: sky.horizon_color.evaluate(zoom).premultiplied(),
                    blend: sky.sky_horizon_blend.evaluate(zoom),
                }),
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
            let mut images = self.services.images.0.lock().expect("images poisoned");
            write_paint(
                gpu,
                renderer,
                style,
                zoom,
                self.camera.bearing,
                crossfade,
                images.atlas.as_mut().expect("initialized"),
            );
            // Backgrounds draw on every covered tile, as GL JS's do: never
            // beyond the world's edge.
            if let Some(tile) = style.background {
                let cover = view.cover(&self.tms, TILE_PX, 24);
                let period = self.tms.projection().wrap_period().unwrap_or(0.0);
                let rows: Vec<_> = cover
                    .iter()
                    .map(|covered| {
                        let draw = Draw {
                            tile: covered.index,
                            wrap: covered.wrap,
                            regions: vec![covered.index],
                        };
                        placements(&self.tms, &draw, 8192, view.camera_world(), period, zoom)
                            .next()
                            .expect("one background region")
                    })
                    .collect();
                report.tiles_drawn += rows.len() as u32;
                renderer
                    .scene()
                    .place_tile(gpu, tile, &rows)
                    .expect("background placements fit");
            }
            drop(images);
            style.symbol_tiles.clear();
            for index in 0..style.sources.len() {
                self.request_and_place(gpu, &view, index, &mut report);
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
        if self.rebuild_hits
            && let Some(style) = &self.style
        {
            let labels = self
                .labels
                .screen_hits(style.worker.plan(), &view, &self.tms);
            self.hits.extend(labels);
        }
        self.hits.sort_by_key(|h| h.order);
        report.generation = self.generation();
        for h in &mut self.hits {
            h.hit.generation = report.generation;
        }
        report.loaded = self.style.as_ref().is_some_and(|s| {
            s.sprites_loading == 0
                && s.sources
                    .iter()
                    .all(|source| !matches!(source, SourceState::Resolving))
        }) && report.tiles_loading == 0
            && self.pending.is_empty()
            && labels_settled;
        report.idle = report.loaded && !fading && !self.labels.fading(now);
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

        let frame = FrameBlock {
            view_proj: view.view_proj().as_mat4().to_cols_array_2d(),
            viewport: [viewport.width as f32, viewport.height as f32],
            world_per_px: view.world_per_px() as f32,
            pixel_ratio,
            zoom,
            bearing: self.camera.bearing.to_radians() as f32,
            extrusion_scale: self.extrusion_scale() as f32,
            ground_horizon: view.ground_horizon() as f32,
            sky_horizon: view.sky_horizon() as f32,
            symbol_fade_change: self.labels.fade_change(now),
            pitch: self.camera.pitch.to_radians() as f32,
            align: pixel_alignment(&view, &self.tms, self.camera.bearing),
        };
        let (renderer, _) = self.renderer.as_mut().expect("created above");
        report.frame = renderer.render(gpu, &frame, target, clear);
        report.scene = renderer.scene().stats();
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
            self.warnings.push("heatmap requires filterable, blendable rgba16float render attachments (Compat: EXT_color_buffer_float); heatmap layers are skipped".into());
        }
        renderer.remap_nodes(gpu, &families, &retained);
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
        style.background = make_background(gpu, renderer, plan);
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
        self.tms.projection().style_height_scale(self.camera.target)
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
        let style = self.style.as_ref().expect("jobs need a style");
        let SourceState::Ready { endpoint, .. } = &style.sources[index] else {
            return;
        };
        let key = style.worker.plan().sources[index].key;
        let request = TileRequest {
            style: style.id,
            source: key,
            tile,
            data_zoom: tile.z.min(endpoint.maxzoom),
            location: endpoint.location(endpoint.data_tile(tile)),
            encoding: endpoint.encoding,
            nodes,
            rows_only,
        };
        match (edit, rows_only) {
            (None, _) => self.counts.loads += 1,
            (Some(_), false) => self.counts.relayouts += 1,
            (Some(_), true) => self.counts.columns += 1,
        }
        if let Some(edit) = edit {
            *self.pending.entry(edit).or_default() += 1;
        }
        let (sender, wake, generation) = (
            self.sender.clone(),
            Arc::clone(&self.services.wake),
            self.tile_generation,
        );
        self.services.jobs.load(
            request,
            Box::new(move |result| {
                let _ = sender.send(Message::Tile {
                    generation,
                    source: key,
                    tile,
                    edit,
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
        let frame = self.labels.frame(
            style.worker.plan(),
            &style.symbol_tiles,
            view,
            &self.tms,
            &self.camera,
            extrusion_scale,
            now,
        );
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
    fn request_and_place(
        &mut self,
        gpu: &mut dyn Gpu,
        view: &View,
        index: usize,
        report: &mut FrameReport,
    ) {
        let elevation_scale = self.extrusion_scale();
        let style = self.style.as_mut().expect("called with a style");
        let zoom = self.camera.zoom();
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
            ..
        } = &mut style.sources[index]
        else {
            return;
        };
        let cover = view.cover_volume(
            &self.tms,
            endpoint.cover_px,
            pyramid.max_zoom(),
            f64::from(height) * elevation_scale,
        );
        let update = pyramid.update(&cover);
        let mut requests = Vec::new();
        for &tile in &update.request {
            if !endpoint.contains(tile) {
                pyramid.insert_empty(tile);
                continue;
            }
            *loading += 1;
            requests.push(tile);
        }
        report.tiles_loading += *loading;
        for tile in requests {
            self.send_job(index, tile, None, false, None);
        }

        let style = self.style.as_mut().expect("called with a style");
        let SourceState::Ready { pyramid, .. } = &mut style.sources[index] else {
            unreachable!("checked above");
        };
        let (renderer, _) = self.renderer.as_mut().expect("created before placing");
        let mut rows: HashMap<TileIndex, Vec<&Draw>> = HashMap::new();
        for draw in &update.draw {
            rows.entry(draw.tile).or_default().push(draw);
        }
        let camera_world = view.camera_world();
        let period = self.tms.projection().wrap_period().unwrap_or(0.0);
        let placed = &mut style.placed[index];
        for (&tile, draws) in &rows {
            let resident = pyramid.get(tile).expect("drawn tiles are ready");
            // Each world copy places its labels, as GL JS's wrapped tiles do.
            for symbols in &resident.symbols {
                let mut wraps: Vec<i32> = draws.iter().map(|d| d.wrap).collect();
                wraps.sort_unstable();
                wraps.dedup();
                for wrap in wraps {
                    style.symbol_tiles.push(crate::labels::SymbolTile {
                        layer: symbols.layer,
                        bucket: symbols.bucket,
                        tile,
                        wrap,
                    });
                }
            }
            if self.rebuild_hits {
                crate::query::Publisher {
                    hits: &mut self.hits,
                    plan: style.worker.plan(),
                    view,
                    tms: &self.tms,
                    camera: &self.camera,
                }
                .tile(resident, index, tile, draws);
            }
            for &(extent, scene_tile) in &resident.parts {
                let rows: Vec<TileRow> = draws
                    .iter()
                    .flat_map(|draw| {
                        placements(
                            &self.tms,
                            draw,
                            extent,
                            camera_world,
                            period,
                            self.camera.zoom(),
                        )
                    })
                    .collect();
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
        for draw in &update.draw {
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

    /// Installs a style that `load_style_url` fetched, unless a later
    /// request or `set_style_json` superseded it.
    fn apply_style(&mut self, request: u64, result: Result<String, FetchError>) {
        if request != self.style_request {
            return;
        }
        let installed = result
            .map_err(|e| e.to_string())
            .and_then(|json| self.set_style_json(&json));
        if let Err(e) = installed {
            self.fail(format!("style: {e}"));
        }
    }

    fn apply_messages(&mut self, gpu: &mut dyn Gpu) {
        while let Ok(message) = self.receiver.try_recv() {
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
                Message::Style { request, result } => self.apply_style(request, result),
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
                    let endpoint = match result {
                        Ok(text) => TileEndpoint::resolve(&def.kind, Some(&text)),
                        Err(e) => Err(format!("TileJSON: {e}")),
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
                    result,
                } => {
                    let host_bytes = result.as_ref().ok().and_then(|r| r.host_bytes);
                    let host_image = result.as_ref().ok().and_then(|r| match &r.image {
                        Some(crate::TileImage {
                            pixels: crate::ImagePixels::Host(image),
                            ..
                        }) => Some(*image),
                        _ => None,
                    });
                    if let Some(edit) = edit {
                        self.finish_job(edit);
                    }
                    if generation == self.tile_generation {
                        match edit {
                            None => self.apply_tile(gpu, source, tile, result),
                            Some(_) => self.apply_edit(gpu, source, tile, result),
                        }
                    }
                    // Uploaded or dropped, the result's host data is done.
                    if let Some(bytes) = host_bytes {
                        gpu.release_host_bytes(bytes);
                    }
                    if let Some(image) = host_image {
                        gpu.release_host_image(image);
                    }
                }
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
    ) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
            return;
        };
        let plan = style.worker.plan();
        let Some(source) = plan.source(source) else {
            return;
        };
        let SourceState::Ready {
            pyramid,
            loading,
            endpoint,
            page,
        } = &mut style.sources[source]
        else {
            return;
        };
        *loading = loading.saturating_sub(1);
        self.events.sources.insert(source);
        let mut build = match result {
            Ok(build) => build,
            Err(e) => {
                let message = format!("tile {tile:?}: {e}");
                self.events.push(MapEvent::Error {
                    message: message.clone(),
                });
                self.warnings.push(message);
                pyramid.insert_empty(tile);
                return;
            }
        };
        self.warnings.append(&mut build.warnings);
        if build.missing {
            pyramid.insert_missing(tile);
            return;
        }
        if let Some(tile_image) = &build.image {
            // Neighbours' prepares refer to the tile's layer, so only a
            // tile the pyramid still wants is uploaded.
            if !pyramid.is_loading(tile) {
                return;
            }
            let kind = plan
                .image_kind(source)
                .expect("image results are for image sources");
            let upload = image::Upload {
                plan,
                source,
                kind,
                tile,
                image: tile_image,
                unpack: endpoint.dem_unpack,
            };
            match image::upload(gpu, renderer, page, pyramid, &upload) {
                Ok(resident) => {
                    if let Some(unwanted) = pyramid.insert(tile, resident) {
                        self.retired_ramps.extend(remove_resident(
                            renderer,
                            unwanted,
                            &mut style.states,
                        ));
                    }
                }
                Err(e) => {
                    self.warnings.push(format!("tile {tile:?}: {e}"));
                    pyramid.insert_empty(tile);
                }
            }
            return;
        }
        if !pyramid.is_loading(tile) {
            return;
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
        let mut atlas = self.services.images.0.lock().expect("images poisoned");
        // Make room by evicting cached tiles the view doesn't use, least
        // recently used first, this source's before others'; then the tiles
        // this one replaces as soon as it's ready.
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
                f64::from(tile.z),
                &mut self.warnings,
            ) {
                Ok(resident) => break Some(resident),
                Err(OutOfSpace) => {
                    let unused = std::iter::once(source)
                        .chain((0..style.sources.len()).filter(|&s| s != source))
                        .find_map(|s| match &mut style.sources[s] {
                            SourceState::Ready { pyramid, .. } => pyramid.evict_lru(),
                            _ => None,
                        });
                    let victim = unused.or_else(|| match &mut style.sources[source] {
                        SourceState::Ready { pyramid, .. } => pyramid.evict_within(tile),
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
        let Some(resident) = resident else {
            // Even the tiles in view don't leave room: draw what stands in.
            self.warnings.push(format!(
                "tile {tile:?} doesn't fit in scene memory with the tiles in view"
            ));
            pyramid.insert_missing(tile);
            return;
        };
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
    endpoint: Result<TileEndpoint, String>,
    id: &str,
    image: bool,
    warnings: &mut Vec<String>,
) -> SourceState {
    match endpoint {
        Ok(endpoint) => SourceState::Ready {
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
    let mut resident = ResidentTile::default();
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

/// Removes a tile from the scene. Returns the ramps its rows retained.
pub(crate) fn remove_resident(
    renderer: &mut Renderer,
    tile: ResidentTile,
    states: &mut FeatureStates,
) -> Vec<u32> {
    for (_, id) in tile.parts {
        renderer.scene().remove_tile(id);
    }
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
fn pixel_alignment(view: &View, tms: &TileMatrixSet, bearing: f64) -> [f32; 2] {
    let world_per_px = view.world_per_px();
    let bounds = tms.projection().world_bounds();
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
fn placements<'a>(
    tms: &TileMatrixSet,
    draw: &'a Draw,
    extent: u32,
    camera_world: DVec2,
    period: f64,
    zoom: f64,
) -> impl Iterator<Item = TileRow> + use<'a> {
    let bounds = tms.tile_bounds(draw.tile);
    let origin =
        DVec2::new(bounds.min.x + f64::from(draw.wrap) * period, bounds.max.y) - camera_world;
    let tile = draw.tile;
    let rect = move |region: TileIndex| {
        let depth = region.z - tile.z;
        let size = extent >> depth;
        let x = (region.col - (tile.col << depth)) * size;
        let y = (region.row - (tile.row << depth)) * size;
        ClipRect::new([x as i16, y as i16], [(x + size) as i16, (y + size) as i16])
    };
    // One placement has four clip rectangles. Split larger partitions into
    // disjoint rows instead of expanding to quadrants that overlap loaded
    // descendants (and render translucent geometry twice).
    let pixel_size = TILE_PX * 2.0f64.powf(zoom.floor() - f64::from(tile.z));
    let x = f64::from(tile.col) + f64::from(draw.wrap) * 2.0f64.powi(i32::from(tile.z));
    let pattern_origin = [
        (x * pixel_size) as i64 as u32,
        (f64::from(tile.row) * pixel_size) as i64 as u32,
    ];
    let scale = (tms.tile_span(tile.z) / f64::from(extent)) as f32;
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
            zoom: f32::from(tile.z),
            clips,
        }
    })
}

/// Writes every node's paint and visibility for `zoom`, if it changed.
fn write_paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    style: &mut LoadedStyle,
    zoom: f64,
    bearing: f64,
    crossfade: Crossfade,
    atlas: &mut ImageAtlas,
) {
    if style.painted == Some((zoom, bearing)) && style.painted_fade == Some(crossfade) {
        return;
    }
    style.painted = Some((zoom, bearing));
    style.painted_fade = Some(crossfade);
    let plan = style.worker.plan();
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
                let light = &plan.style.light;
                let [radius, azimuth, polar] = light.position.evaluate(zoom);
                let azimuth = (azimuth + 90.0).to_radians();
                let polar = polar.to_radians();
                let lc = light.color.evaluate(zoom).premultiplied();
                NodePaint::Extrusion(orb_render::ExtrusionPaint {
                    color,
                    height: numbers[1],
                    base: numbers[2],
                    opacity: e.paint.opacity.evaluate(zoom),
                    translate,
                    vertical_gradient: e.paint.vertical_gradient.evaluate(zoom),
                    light_position: [
                        radius * azimuth.cos() * polar.sin(),
                        radius * azimuth.sin() * polar.sin(),
                        radius * polar.cos(),
                    ],
                    light_viewport: light.anchor.evaluate(zoom) == TranslateAnchor::Viewport,
                    light_color: [lc[0], lc[1], lc[2]],
                    light_intensity: light.intensity.evaluate(zoom),
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
    }
}

/// One scene tile with a tile-sized quad for every background layer.
fn make_background(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &crate::style::StylePlan,
) -> Option<TileId> {
    let mut tile = None;
    let mesh = orb_layout::Mesh {
        vertices: vec![0, 0, 8192, 0, 8192 | 8192 << 16, 0, 8192 << 16, 0],
        indices: vec![0, 1, 2, 0, 2, 3],
        chunks: vec![orb_layout::Chunk {
            sort_key: None,
            first_index: 0,
            index_count: 6,
            bounds: orb_layout::TileBox {
                min: [0, 0],
                max: [8192, 8192],
            },
        }],
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
        let viewport = Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let center = DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        for (bearing, pitch) in [(0.0, 0.0), (30.0, 45.0)] {
            let camera = CameraState {
                bearing,
                pitch,
                ..CameraState::new(target, 15.0)
            };
            let mut map = Map::new(services.clone(), Some(camera));
            for delta in [-0.5; 6].into_iter().chain([0.5; 6]) {
                let zoom = map.camera().zoom() + delta;
                let projection = map.projection();
                map.camera_mut()
                    .zoom_around(&projection, viewport, center, zoom);
                let fresh = CameraState {
                    bearing,
                    pitch,
                    ..CameraState::new(target, zoom)
                };
                assert_eq!(
                    *map.camera(),
                    fresh,
                    "a centred zoom must not move the target"
                );
                assert_eq!(
                    map.view(viewport).cover(&map.tms, 512.0, 14),
                    View::new(&fresh, viewport, map.tms.projection()).cover(&map.tms, 512.0, 14)
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
        let rows: Vec<_> = placements(&tms, &draw, 4096, DVec2::ZERO, 100.0, 3.0).collect();
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
        while let Ok(Message::Style { request, result }) = map.receiver.try_recv() {
            map.apply_style(request, result);
        }
        let name = map.style.as_ref().unwrap().worker.plan().style.name.clone();
        assert_eq!(name.as_deref(), Some("b"));
    }
}
