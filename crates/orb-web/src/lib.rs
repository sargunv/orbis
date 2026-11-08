//! The browser build of Orbis: `core.wasm` and its wasm-bindgen surface.
//!
//! One module, two places, chosen by the page's JS:
//! - **the page** ([`create_map`]): the map draws into a canvas on the
//!   page's thread, with its camera, gestures and synchronous `project`,
//!   `unproject` and `hitTest`.
//! - **workers**: job workers ([`jobs::JobWorker`]) fetch and lay out
//!   tiles; a placement worker ([`PlacementWorker`]) places labels.
//!
//! The map draws through WebGPU.

#![cfg(target_arch = "wasm32")]

mod camera;
mod canvas;
mod jobs;
mod placement;
pub mod regions;
mod snapshot;

pub use jobs::JobWorker;
pub use placement::PlacementWorker;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec2;
use orb_runtime::{Jobs, Map, MapEvent, Services, Viewport};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::canvas::Canvas;
use crate::jobs::WorkerJobs;
use crate::placement::WorkerPlacement;

thread_local! {
    /// Each map's callback for "draw a frame soon", by map. Wasm in a page
    /// is single-threaded, so the runtime's `Send` wake hook reaches it
    /// here.
    static WAKES: RefCell<HashMap<u32, js_sys::Function>> = RefCell::new(HashMap::new());
    static NEXT_MAP: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Calls map `id`'s wake callback, if the map is alive.
pub(crate) fn wake(id: u32) {
    let wake = WAKES.with(|wakes| wakes.borrow().get(&id).cloned());
    if let Some(f) = wake {
        let _ = f.call0(&JsValue::NULL);
    }
}

/// A map drawing into a canvas.
#[wasm_bindgen]
#[derive(Debug)]
pub struct OrbisMap {
    /// This map's key in [`WAKES`].
    id: u32,
    map: Map,
    canvas: Canvas,
    /// What draws the map into the canvas; a new GPU gets a new one.
    renderer: Option<orb_runtime::Renderer>,
    /// The map's images, on the canvas's GPU.
    images: orb_runtime::Images,
    pixel_ratio: f64,
    worker_jobs: Arc<WorkerJobs>,
    /// A snapshot being drawn ([`snapshot`]).
    snapshot: Option<snapshot::Snapshot>,
    /// Markers and the location layer.
    batteries: orb_runtime::batteries::Batteries,
    /// Where the next frame's pixels go ([`read_next_frame`]).
    ///
    /// [`read_next_frame`]: OrbisMap::read_next_frame
    read_next: Option<futures_channel::oneshot::Sender<Vec<u8>>>,
}

impl Drop for OrbisMap {
    fn drop(&mut self) {
        WAKES.with(|wakes| wakes.borrow_mut().remove(&self.id));
        jobs::forget(self.id);
        placement::forget(self.id);
    }
}

/// A frame's counters. Byte counts are since the map was created: buffer
/// bytes written from wasm memory and straight from JS `ArrayBuffer`s,
/// texel bytes from wasm memory and from `ImageBitmap`s, packed tile
/// headers copied into wasm memory.
#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    /// CPU milliseconds the frame took.
    #[wasm_bindgen(readonly)]
    pub ms: f64,
    #[wasm_bindgen(readonly)]
    pub zoom: f64,
    /// The style generation the frame shows in full.
    #[wasm_bindgen(readonly)]
    pub generation: f64,
    /// Nothing left to do: no loads, fades, placement or camera motion.
    #[wasm_bindgen(readonly)]
    pub idle: bool,
    /// The style and every tile in view are in (GL JS `loaded()`).
    #[wasm_bindgen(readonly)]
    pub loaded: bool,
    /// A paint transition runs.
    #[wasm_bindgen(readonly)]
    pub transitioning: bool,
    /// Whether the map needs another frame: it isn't idle, and isn't only
    /// waiting for pipelines that wake it when they compile.
    #[wasm_bindgen(readonly, js_name = needsFrame)]
    pub needs_frame: bool,
    /// The GPU was lost and nothing was drawn; the map draws again once
    /// [`OrbisMap::recover`] gives it another.
    #[wasm_bindgen(readonly)]
    pub lost: bool,
    #[wasm_bindgen(readonly, js_name = meshesUploaded)]
    pub meshes_uploaded: u32,
    #[wasm_bindgen(readonly, js_name = rowsUploaded)]
    pub rows_uploaded: u32,
    #[wasm_bindgen(readonly)]
    pub draws: u32,
    #[wasm_bindgen(readonly)]
    pub dispatches: u32,
    #[wasm_bindgen(readonly)]
    pub candidates: u32,
    #[wasm_bindgen(readonly, js_name = tilesDrawn)]
    pub tiles_drawn: u32,
    #[wasm_bindgen(readonly, js_name = tilesLoading)]
    pub tiles_loading: u32,
    #[wasm_bindgen(readonly, js_name = bundleRecorded)]
    pub bundle_recorded: bool,
    #[wasm_bindgen(readonly, js_name = placementsWritten)]
    pub placements_written: u32,
    /// Milliseconds of the frame spent on placement on this thread:
    /// building the frame the placement worker places, applying commits.
    #[wasm_bindgen(readonly, js_name = placementMs)]
    pub placement_ms: f64,
    #[wasm_bindgen(readonly, js_name = paintsWritten)]
    pub paints_written: u32,
    #[wasm_bindgen(readonly, js_name = uploadedFromWasm)]
    pub uploaded_from_wasm: f64,
    #[wasm_bindgen(readonly, js_name = uploadedFromJs)]
    pub uploaded_from_js: f64,
    #[wasm_bindgen(readonly, js_name = texelsFromWasm)]
    pub texels_from_wasm: f64,
    #[wasm_bindgen(readonly, js_name = texelsFromJs)]
    pub texels_from_js: f64,
    #[wasm_bindgen(readonly, js_name = headerBytes)]
    pub header_bytes: f64,
    #[wasm_bindgen(readonly, js_name = jobsDone)]
    pub jobs_done: u32,
    /// Pipelines still compiling: layers that need them aren't drawn yet.
    #[wasm_bindgen(readonly, js_name = pipelinesPending)]
    pub pipelines_pending: u32,
    /// Layers not drawn while their pipelines compile.
    #[wasm_bindgen(readonly, js_name = layersWaiting)]
    pub layers_waiting: u32,
    /// GPU memory against the budget.
    #[wasm_bindgen(readonly)]
    pub memory: FrameMemory,
}

/// The map's GPU memory by pool, in bytes but `cached_tiles`: the scene's
/// pools grow within ceilings the budget sets; tiles out of view stay
/// cached within `cache_budget`.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMemory {
    #[wasm_bindgen(readonly)]
    pub budget: f64,
    #[wasm_bindgen(readonly)]
    pub allocated: f64,
    #[wasm_bindgen(readonly, js_name = verticesUsed)]
    pub vertices_used: f64,
    #[wasm_bindgen(readonly, js_name = verticesAllocated)]
    pub vertices_allocated: f64,
    #[wasm_bindgen(readonly, js_name = verticesCeiling)]
    pub vertices_ceiling: f64,
    #[wasm_bindgen(readonly, js_name = verticesLargestFree)]
    pub vertices_largest_free: f64,
    #[wasm_bindgen(readonly, js_name = indicesUsed)]
    pub indices_used: f64,
    #[wasm_bindgen(readonly, js_name = indicesAllocated)]
    pub indices_allocated: f64,
    #[wasm_bindgen(readonly, js_name = indicesCeiling)]
    pub indices_ceiling: f64,
    #[wasm_bindgen(readonly, js_name = compactedUsed)]
    pub compacted_used: f64,
    #[wasm_bindgen(readonly, js_name = compactedAllocated)]
    pub compacted_allocated: f64,
    #[wasm_bindgen(readonly, js_name = propsUsed)]
    pub props_used: f64,
    #[wasm_bindgen(readonly, js_name = propsAllocated)]
    pub props_allocated: f64,
    /// Tile pages, glyph pages and the icon and pattern atlas.
    #[wasm_bindgen(readonly)]
    pub textures: f64,
    #[wasm_bindgen(readonly, js_name = cachedTiles)]
    pub cached_tiles: u32,
    #[wasm_bindgen(readonly)]
    pub cached: f64,
    #[wasm_bindgen(readonly, js_name = cacheBudget)]
    pub cache_budget: f64,
}

impl From<&orb_runtime::Memory> for FrameMemory {
    fn from(m: &orb_runtime::Memory) -> Self {
        let s = &m.scene;
        Self {
            budget: m.budget as f64,
            allocated: m.allocated() as f64,
            vertices_used: s.vertices.used as f64,
            vertices_allocated: s.vertices.allocated as f64,
            vertices_ceiling: s.vertices.ceiling as f64,
            vertices_largest_free: s.vertices.largest_free as f64,
            indices_used: s.indices.used as f64,
            indices_allocated: s.indices.allocated as f64,
            indices_ceiling: s.indices.ceiling as f64,
            compacted_used: s.compacted.used as f64,
            compacted_allocated: s.compacted.allocated as f64,
            props_used: s.props.used as f64,
            props_allocated: s.props.allocated as f64,
            textures: (m.tile_textures + m.glyph_textures + m.image_atlas) as f64,
            cached_tiles: m.cached_tiles,
            cached: m.cached as f64,
            cache_budget: m.cache_budget as f64,
        }
    }
}

/// Creates a map in `canvas` showing the style at `style_url`, drawing on
/// this thread. Its tile jobs go to `job_workers` job workers through
/// `post(worker, message)`, whose replies come back through
/// [`receive_job`](OrbisMap::receive_job); labels are placed by the
/// placement worker `place(message, transfer)` reaches, whose replies come
/// back through [`receive_placement`](OrbisMap::receive_placement).
/// `on_wake` is called when the map has new data to draw; the page should
/// call `render` on its next animation frame. The camera is the style's
/// until it moves. Video sources play on the page (`web/src/videos.ts`):
/// see [`take_events`](OrbisMap::take_events) and
/// [`video_frame`](OrbisMap::video_frame); canvas sources show what
/// [`source_frame`](OrbisMap::source_frame) hands in. `options` holds GL
/// JS's map options by name, each optional: `fadeDuration`,
/// `crossSourceCollisions`, `localIdeographFontFamily` (`false` draws the
/// style's glyphs), `antialias` and `terrainSkirts`.
#[wasm_bindgen(js_name = createMap)]
#[allow(clippy::too_many_arguments)]
pub async fn create_map(
    canvas: HtmlCanvasElement,
    style_url: String,
    options: JsValue,
    on_wake: js_sys::Function,
    job_workers: u32,
    post: js_sys::Function,
    place: js_sys::Function,
) -> Result<OrbisMap, JsError> {
    let size = (canvas.width(), canvas.height());

    let id = NEXT_MAP.with(|next| next.replace(next.get() + 1));
    let worker_jobs = WorkerJobs::new(id, job_workers, post);
    let jobs: Arc<dyn Jobs> = worker_jobs.clone();
    let lane: Arc<dyn orb_runtime::PlacementLane> = WorkerPlacement::new(id, place);
    WAKES.with(|wakes| wakes.borrow_mut().insert(id, on_wake));
    let canvas = Canvas::new(canvas.into(), size, id)
        .await
        .map_err(|e| JsError::new(&e))?;
    let images = orb_runtime::Images::default();
    let services = Services {
        clock: Arc::new(orb_host_web::WebClock),
        image_decode: Arc::new(orb_host_web::WebImageDecode),
        images: images.clone(),
        jobs,
        fetch: regions::service(),
        wake: Arc::new(move || wake(id)),
    };
    let mut map = Map::new(services, None);
    if mobile() {
        map.set_memory_budget(orb_runtime::MOBILE_BUDGET);
    }
    map.set_placement_lane(lane);
    // The page plays video sources (`web/src/videos.ts`).
    map.set_plays_video(true);
    let option = |name: &str| js_sys::Reflect::get(&options, &name.into()).ok();
    if let Some(ms) = option("fadeDuration").and_then(|v| v.as_f64()) {
        map.set_fade_duration(ms);
    }
    if let Some(collide) = option("crossSourceCollisions").and_then(|v| v.as_bool()) {
        map.set_cross_source_collisions(collide);
    }
    match option("localIdeographFontFamily") {
        Some(family) if family.as_bool() == Some(false) || family.is_null() => {
            map.set_local_ideograph_font_family(None);
        }
        Some(family) => {
            if let Some(family) = family.as_string() {
                map.set_local_ideograph_font_family(Some(&family));
            }
        }
        None => {}
    }
    if let Some(antialias) = option("antialias").and_then(|v| v.as_bool()) {
        map.set_antialias_3d(antialias);
    }
    if let Some(skirts) = option("terrainSkirts").and_then(|v| v.as_bool()) {
        map.set_terrain_skirts(skirts);
    }
    map.load_style_url(&style_url);
    Ok(OrbisMap {
        id,
        map,
        canvas,
        renderer: None,
        images,
        pixel_ratio: 1.0,
        worker_jobs,
        snapshot: None,
        batteries: Default::default(),
        read_next: None,
    })
}

#[wasm_bindgen]
impl OrbisMap {
    /// The current style JSON, including successful edits.
    #[wasm_bindgen(js_name = getStyle)]
    pub fn get_style(&self) -> Result<String, JsError> {
        self.map
            .style_doc()
            .map(|doc| doc.to_json())
            .ok_or_else(|| JsError::new("no style is loaded"))
    }

    /// Layer `id`'s JSON, its paint and layout properties included, or
    /// none.
    #[wasm_bindgen(js_name = getLayer)]
    pub fn get_layer(&self, id: &str) -> Option<String> {
        Some(self.map.style_doc()?.layer(id)?.to_string())
    }

    /// Source `id`'s JSON, or none.
    #[wasm_bindgen(js_name = getSource)]
    pub fn get_source(&self, id: &str) -> Option<String> {
        Some(self.map.style_doc()?.source(id)?.to_string())
    }

    /// The layers' ids in drawing order (none before a style loads).
    #[wasm_bindgen(js_name = layerIds)]
    pub fn layer_ids(&self) -> Vec<String> {
        let doc = self.map.style_doc();
        doc.into_iter()
            .flat_map(|d| d.layer_ids())
            .map(str::to_string)
            .collect()
    }

    #[wasm_bindgen(js_name = sourceIds)]
    pub fn source_ids(&self) -> Vec<String> {
        let doc = self.map.style_doc();
        doc.into_iter()
            .flat_map(|d| d.source_ids())
            .map(str::to_string)
            .collect()
    }

    /// The last placement commit's semantic snapshot as JSON,
    /// or null before labels are first placed.
    #[wasm_bindgen(js_name = semanticSnapshot)]
    pub fn semantic_snapshot(&self) -> Option<String> {
        self.map.semantic_snapshot().map(|s| s.to_json())
    }

    /// Topmost hit in the published frame, as a plain JS object or null.
    #[wasm_bindgen(js_name = hitTest)]
    pub fn hit_test(&self, x: f64, y: f64) -> JsValue {
        hit_object(self.map.hit_test(DVec2::new(x, y)))
    }

    /// GL JS `queryRenderedFeatures` in the box `[x1, y1, x2, y2]` (CSS
    /// pixels of the last frame; a point when the corners meet), of the
    /// layers in `layers` (a JSON array) or all. Resolves to a JSON array of
    /// GeoJSON features, topmost first, read in the tiles' job workers.
    #[wasm_bindgen(js_name = queryRendered)]
    pub fn query_rendered(
        &self,
        rect: &[f64],
        layers: Option<String>,
    ) -> Result<js_sys::Promise, JsError> {
        let [x1, y1, x2, y2] = *rect else {
            return Err(JsError::new("a query box has 4 numbers"));
        };
        let layers = layers
            .map(|json| {
                let json =
                    orb_expr::json::parse(&json).map_err(|e| JsError::new(&e.to_string()))?;
                Ok::<_, JsError>(
                    json.as_array()
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|l| l.as_str().map(str::to_string))
                        .collect::<Vec<_>>(),
                )
            })
            .transpose()?;
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.map.query_rendered(
            DVec2::new(x1, y1),
            DVec2::new(x2, y2),
            layers.as_deref(),
            Box::new(move |result| {
                let _ = sender.send(result);
            }),
        );
        Ok(features_promise(receiver))
    }

    /// GL JS `querySourceFeatures`: the features of `source` in its drawn
    /// tiles, of `source_layer` or every layer, that `filter` (a v8 filter
    /// as JSON) selects. Resolves to a JSON array of GeoJSON features.
    #[wasm_bindgen(js_name = querySource)]
    pub fn query_source(
        &self,
        source: &str,
        source_layer: Option<String>,
        filter: Option<String>,
    ) -> js_sys::Promise {
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.map.query_source(
            source,
            source_layer.as_deref(),
            filter.as_deref(),
            Box::new(move |result| {
                let _ = sender.send(result);
            }),
        );
        features_promise(receiver)
    }

    /// GL JS's cluster queries on clustered GeoJSON source `source`:
    /// `"children"`, `"leaves"` (`limit` past `offset`) or
    /// `"expansionZoom"` of cluster `id`. Resolves to JSON: an array of
    /// GeoJSON features, or a number.
    #[wasm_bindgen(js_name = queryCluster)]
    pub fn query_cluster(
        &self,
        source: &str,
        kind: &str,
        id: f64,
        limit: u32,
        offset: u32,
    ) -> js_sys::Promise {
        let id = id as u64;
        let query = match kind {
            "children" => orb_runtime::ClusterQuery::Children(id),
            "leaves" => orb_runtime::ClusterQuery::Leaves { id, limit, offset },
            _ => orb_runtime::ClusterQuery::ExpansionZoom(id),
        };
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.map.query_cluster(
            source,
            query,
            Box::new(move |result| {
                let _ = sender.send(result);
            }),
        );
        features_promise(receiver)
    }

    /// Matches the drawing buffer to the canvas's size in physical pixels.
    pub fn resize(&mut self, width: u32, height: u32, pixel_ratio: f64) {
        self.pixel_ratio = pixel_ratio;
        self.canvas.resize(width, height);
        // Fits and projections before the next frame use the new size.
        self.map.set_viewport(orb_runtime::Viewport {
            width: f64::from(width) / pixel_ratio,
            height: f64::from(height) / pixel_ratio,
        });
    }

    /// What the map did since the last call, as plain objects by `type`:
    /// `styleload`, `styledata` (`generation`), `sourcedata` (`source`,
    /// `loaded`), `load`, `idle`, `error` and `warning` (`message`),
    /// `styleimagemissing` (`id`), `restored`, `camera`
    /// (`change`: move, zoom, rotate or pitch; `phase`: start, progress or
    /// end; `user`), `animationend` (`id`, `interrupted`), `click` (`x`,
    /// `y`, `button`), `dblclick` (`x`, `y`), and for the page's video
    /// player `video` (`source`, `request`: `open` with `urls`, `play`,
    /// `pause`, `seek` with `seconds`, `close`).
    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        // Keys and names are the same few strings every frame: interned,
        // the page decodes each once.
        let name = |text: &'static str| JsValue::from(wasm_bindgen::intern(text));
        let set = |object: &js_sys::Object, key: &'static str, value: JsValue| {
            let _ = js_sys::Reflect::set(object, &name(key), &value);
        };
        for event in self.map.take_events() {
            let object = js_sys::Object::new();
            let kind = match event {
                MapEvent::StyleLoad => "styleload",
                MapEvent::StyleData { generation } => {
                    set(&object, "generation", (generation as f64).into());
                    "styledata"
                }
                MapEvent::SourceData { source, loaded } => {
                    set(&object, "source", source.into());
                    set(&object, "loaded", loaded.into());
                    "sourcedata"
                }
                MapEvent::Load => "load",
                MapEvent::Idle => "idle",
                MapEvent::Error { message } => {
                    set(&object, "message", message.into());
                    "error"
                }
                MapEvent::Warning { message } => {
                    set(&object, "message", message.into());
                    "warning"
                }
                MapEvent::StyleImageMissing { name } => {
                    set(&object, "id", name.into());
                    "styleimagemissing"
                }
                MapEvent::TileRequest { source, tile } => {
                    set(&object, "source", source.into());
                    set(&object, "z", tile.z.into());
                    set(&object, "x", tile.col.into());
                    set(&object, "y", tile.row.into());
                    "tilerequest"
                }
                MapEvent::Video { source, request } => {
                    set(&object, "source", source.into());
                    let request = match request {
                        orb_runtime::VideoRequest::Open { urls } => {
                            let urls: js_sys::Array = urls.iter().map(JsValue::from).collect();
                            set(&object, "urls", urls.into());
                            "open"
                        }
                        orb_runtime::VideoRequest::Play => "play",
                        orb_runtime::VideoRequest::Pause => "pause",
                        orb_runtime::VideoRequest::Seek(seconds) => {
                            set(&object, "seconds", seconds.into());
                            "seek"
                        }
                        orb_runtime::VideoRequest::Close => "close",
                    };
                    set(&object, "request", name(request));
                    "video"
                }
                MapEvent::Restored => "restored",
                MapEvent::Camera(orb_runtime::CameraEvent::Changed {
                    change,
                    phase,
                    user,
                }) => {
                    let change = match change {
                        orb_camera::Change::Move => "move",
                        orb_camera::Change::Zoom => "zoom",
                        orb_camera::Change::Rotate => "rotate",
                        orb_camera::Change::Pitch => "pitch",
                    };
                    let phase = match phase {
                        orb_camera::Phase::Start => "start",
                        orb_camera::Phase::Progress => "progress",
                        orb_camera::Phase::End => "end",
                    };
                    set(&object, "change", name(change));
                    set(&object, "phase", name(phase));
                    set(&object, "user", user.into());
                    "camera"
                }
                MapEvent::Camera(orb_runtime::CameraEvent::AnimationEnd { id, interrupted }) => {
                    set(&object, "id", (id.0 as f64).into());
                    set(&object, "interrupted", interrupted.into());
                    "animationend"
                }
                MapEvent::Click { position, button } => {
                    set(&object, "x", position.x.into());
                    set(&object, "y", position.y.into());
                    let button = match button {
                        orb_runtime::Button::Primary => 0,
                        orb_runtime::Button::Middle => 1,
                        orb_runtime::Button::Secondary => 2,
                    };
                    set(&object, "button", button.into());
                    "click"
                }
                MapEvent::DoubleClick { position } => {
                    set(&object, "x", position.x.into());
                    set(&object, "y", position.y.into());
                    "dblclick"
                }
            };
            set(&object, "type", name(kind));
            out.push(&object);
        }
        out
    }

    /// The video of source `id` has a new frame to show: the element's
    /// current frame is copied in the next frame drawn.
    #[wasm_bindgen(js_name = videoFrame)]
    pub fn video_frame(
        &mut self,
        id: &str,
        video: web_sys::HtmlVideoElement,
    ) -> Result<(), JsError> {
        let (width, height) = (video.video_width(), video.video_height());
        let image = self.canvas.gpu().register_video(video);
        let frame = orb_runtime::TileImage {
            width,
            height,
            pixels: orb_runtime::ImagePixels::Host(image),
        };
        self.map
            .source_frame(id, frame)
            .map_err(|e| JsError::new(&e))
    }

    /// The picture canvas (or video) source `id` shows now: straight RGBA8
    /// rows, top first.
    #[wasm_bindgen(js_name = sourceFrame)]
    pub fn source_frame(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), JsError> {
        if pixels.len() != width as usize * height as usize * 4 {
            return Err(JsError::new("pixels aren't width × height RGBA8"));
        }
        let frame = orb_runtime::TileImage {
            width,
            height,
            pixels: orb_runtime::ImagePixels::Core(pixels),
        };
        self.map
            .source_frame(id, frame)
            .map_err(|e| JsError::new(&e))
    }

    /// The app's answer for tile `z/x/y` of source `source`, which it
    /// feeds: `bytes` in the source's encoding, or `geojson` (an object of
    /// source layers and GeoJSON documents), or neither for an empty tile.
    #[wasm_bindgen(js_name = hostTile)]
    pub fn host_tile(
        &mut self,
        source: &str,
        z: u8,
        x: u32,
        y: u32,
        bytes: Option<Vec<u8>>,
        geojson: Option<String>,
    ) -> Result<(), JsError> {
        let answer = match (bytes, geojson) {
            (Some(bytes), _) => orb_runtime::HostTile::Encoded(bytes),
            (None, Some(text)) => orb_runtime::HostTile::GeoJson(text),
            (None, None) => orb_runtime::HostTile::Empty,
        };
        self.map
            .host_tile(source, orb_crs::TileIndex::new(z, x, y), answer)
            .map_err(|e| JsError::new(&e))
    }

    /// Asks for every tile of the app's source `source` again.
    #[wasm_bindgen(js_name = reloadHostTiles)]
    pub fn reload_host_tiles(&mut self, source: &str) -> Result<(), JsError> {
        self.map
            .reload_host_tiles(source)
            .map_err(|e| JsError::new(&e))
    }

    /// The video of source `id` can't play here.
    #[wasm_bindgen(js_name = videoFailed)]
    pub fn video_failed(&mut self, id: &str, message: &str) -> Result<(), JsError> {
        self.map
            .video_failed(id, message)
            .map_err(|e| JsError::new(&e))
    }

    /// Plays (`"play"`), pauses (`"pause"`) or seeks (`"seek"`, to
    /// `seconds`) the video of source `id`.
    #[wasm_bindgen(js_name = controlVideo)]
    pub fn control_video(&mut self, id: &str, request: &str, seconds: f64) -> Result<(), JsError> {
        let request = match request {
            "play" => orb_runtime::VideoRequest::Play,
            "pause" => orb_runtime::VideoRequest::Pause,
            "seek" => orb_runtime::VideoRequest::Seek(seconds),
            other => return Err(JsError::new(&format!("unknown video request {other}"))),
        };
        self.map
            .control_video(id, request)
            .map_err(|e| JsError::new(&e))
    }

    /// Applies GL JS map calls as one transaction: a JSON array
    /// such as `[["setPaintProperty", "water", "fill-color", "blue"],
    /// ["setFeatureState", {"source": "s", "id": 1}, {"hover": true}]]`.
    /// Every call applies or, on an error, none. Resolves to the style
    /// generation whose frames show the edit in full
    /// ([`FrameInfo::stats`]).
    pub fn edit(&mut self, commands: &str) -> Result<f64, JsError> {
        let tx = orb_runtime::Transaction::from_commands(commands).map_err(|e| JsError::new(&e))?;
        let generation = self.map.edit(&tx).map_err(|e| JsError::new(&e))?;
        Ok(generation as f64)
    }

    /// Adds an image to the style (GL JS `addImage`): straight RGBA8 rows,
    /// and `options` as JSON (`pixelRatio`, `sdf`, `stretchX`, `stretchY`,
    /// `content`, `textFitWidth`, `textFitHeight`), or empty.
    #[wasm_bindgen(js_name = addImage)]
    pub fn add_image(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        options: &str,
    ) -> Result<(), JsError> {
        let image = orb_runtime::StyleImage::with_options(width, height, pixels, options)
            .map_err(|e| JsError::new(&e))?;
        self.map
            .add_image(name, image)
            .map_err(|e| JsError::new(&e))
    }

    /// Replaces an image's pixels, of the same size (GL JS `updateImage`).
    #[wasm_bindgen(js_name = updateImage)]
    pub fn update_image(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), JsError> {
        self.map
            .update_image(name, width, height, pixels)
            .map_err(|e| JsError::new(&e))
    }

    /// Removes an image from the style (GL JS `removeImage`).
    #[wasm_bindgen(js_name = removeImage)]
    pub fn remove_image(&mut self, name: &str) -> Result<(), JsError> {
        self.map.remove_image(name).map_err(|e| JsError::new(&e))
    }

    /// Whether the style has an image by this name (GL JS `hasImage`).
    #[wasm_bindgen(js_name = hasImage)]
    pub fn has_image(&self, name: &str) -> bool {
        self.map.has_image(name)
    }

    /// Completes a tile job with a job worker's reply.
    #[wasm_bindgen(js_name = receiveJob)]
    pub fn receive_job(&mut self, reply: &JsValue) -> Result<(), JsError> {
        self.worker_jobs.receive(self.canvas.gpu(), reply)
    }

    /// Job worker `worker` is lost (its wasm trapped, or it died): the
    /// requests it had fail and are retried on the others.
    #[wasm_bindgen(js_name = jobWorkerLost)]
    pub fn job_worker_lost(&mut self, worker: u32, why: &str) {
        self.worker_jobs.lost(worker, why);
    }

    /// The placement worker is lost (its wasm trapped, or it died) and the
    /// page replaced it: the placements it owed fail, and the map sends the
    /// new one its buckets before it places again.
    #[wasm_bindgen(js_name = placementWorkerLost)]
    pub fn placement_worker_lost(&mut self, why: &str) {
        placement::lost(self.id, why);
        wake(self.id);
    }

    /// Completes a placement with the placement worker's reply.
    #[wasm_bindgen(js_name = receivePlacement)]
    pub fn receive_placement(&mut self, reply: &JsValue) -> Result<(), JsError> {
        WorkerPlacement::receive(self.id, reply)
    }

    /// Adds a marker drawn over the style: the style's image `image` (the
    /// built-in pin when absent) at `lon`, `lat`, `anchor` (`icon-anchor`)
    /// on the point. Rendered queries name it by the returned id, in the
    /// `orbis-markers` layer.
    #[wasm_bindgen(js_name = addMarker)]
    pub fn add_marker(
        &mut self,
        lon: f64,
        lat: f64,
        image: Option<String>,
        anchor: Option<String>,
    ) -> f64 {
        let marker = orb_runtime::batteries::Marker {
            position: [lon, lat],
            image,
            anchor,
        };
        self.batteries.add_marker(&mut self.map, marker) as f64
    }

    /// Moves or changes a marker.
    #[wasm_bindgen(js_name = updateMarker)]
    pub fn update_marker(
        &mut self,
        id: f64,
        lon: f64,
        lat: f64,
        image: Option<String>,
        anchor: Option<String>,
    ) -> Result<(), JsError> {
        let marker = orb_runtime::batteries::Marker {
            position: [lon, lat],
            image,
            anchor,
        };
        self.batteries
            .update_marker(&mut self.map, id as u64, marker)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = removeMarker)]
    pub fn remove_marker(&mut self, id: f64) -> Result<(), JsError> {
        self.batteries
            .remove_marker(&mut self.map, id as u64)
            .map_err(|e| JsError::new(&e))
    }

    /// Shows the device's location: a dot, a circle of `accuracy` metres
    /// and a cone toward `heading` (degrees from north) when given; hides
    /// it with `lon` NaN.
    #[wasm_bindgen(js_name = setLocation)]
    pub fn set_location(&mut self, lon: f64, lat: f64, accuracy: f64, heading: Option<f64>) {
        let location = (!lon.is_nan()).then_some(orb_runtime::batteries::Location {
            position: [lon, lat],
            accuracy,
            heading,
        });
        self.batteries.set_location(&mut self.map, location);
    }

    /// Requests the tiles that failed again now, as when the browser comes
    /// back online.
    #[wasm_bindgen(js_name = retryFailedTiles)]
    pub fn retry_failed_tiles(&mut self) {
        self.map.retry_failed_tiles();
    }

    /// The next frame's pixels, straight (not premultiplied) RGBA8 rows top
    /// to bottom, as `ImageData` holds them: what the canvas shows, for
    /// screenshots and tests. Resolves once that frame is drawn.
    #[wasm_bindgen(js_name = readNextFrame)]
    pub fn read_next_frame(&mut self) -> js_sys::Promise {
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.read_next = Some(sender);
        let bgra = snapshot::bgra(self.canvas.format().color);
        wasm_bindgen_futures::future_to_promise(async move {
            let pixels = receiver
                .await
                .map_err(|_| JsValue::from_str("the map went before its next frame"))?;
            Ok(snapshot::straight_rgba(pixels, bgra).into())
        })
    }

    /// Times the GPU's submissions from now on, for measurements
    /// (`gpuTimings` collects them).
    #[wasm_bindgen(js_name = setGpuTiming)]
    pub fn set_gpu_timing(&mut self, on: bool) {
        self.canvas.gpu().set_timing(on);
    }

    /// Each submission's times since the last call, once the GPU has
    /// finished them: `{ passes: [label, ms][], cpu: ms }`, passes timed
    /// where the device has timestamp queries.
    #[wasm_bindgen(js_name = gpuTimings)]
    pub fn gpu_timings(&mut self) -> js_sys::Promise {
        let timings = self.canvas.take_timings();
        wasm_bindgen_futures::future_to_promise(async move {
            let out = js_sys::Array::new();
            for (passes, cpu) in timings.await {
                let pairs = js_sys::Array::new();
                for (label, ms) in passes {
                    pairs.push(&js_sys::Array::of2(&label.into(), &ms.into()));
                }
                let submission = js_sys::Object::new();
                js_sys::Reflect::set(&submission, &"passes".into(), &pairs)?;
                js_sys::Reflect::set(&submission, &"cpu".into(), &cpu.into())?;
                out.push(&submission);
            }
            Ok(out.into())
        })
    }

    /// Loses the GPU on purpose (`GPUDevice.destroy()`), to exercise
    /// recovery.
    #[wasm_bindgen(js_name = loseDevice)]
    pub fn lose_device(&mut self) {
        self.canvas.lose();
    }

    /// Draws on a new GPU in `canvas` after a loss. The map keeps its
    /// style, camera and cached tiles; the next frames lay the visible
    /// tiles out again.
    pub async fn recover(&mut self, canvas: JsValue) -> Result<(), JsError> {
        let size = self.canvas.size();
        self.canvas = Canvas::new(canvas, size, self.id)
            .await
            .map_err(|e| JsError::new(&e))?;
        self.renderer = None;
        self.images.reset();
        wake(self.id);
        Ok(())
    }

    /// Draws the frame shown at `frame_time`, the animation frame's
    /// timestamp (`requestAnimationFrame`'s, on the clock input events'
    /// `timeStamp`s use).
    pub fn render(&mut self, frame_time: f64) -> Result<FrameStats, JsError> {
        if self.canvas.lost() {
            return Ok(FrameStats {
                idle: true,
                lost: true,
                ..FrameStats::default()
            });
        }
        let started = orb_host::Clock::now(&orb_host_web::WebClock);
        // Markers and the location layer come back with a new style.
        self.batteries.sync(&mut self.map);
        let viewport = self.viewport();
        let (map, ratio, renderer) = (&mut self.map, self.pixel_ratio as f32, &mut self.renderer);
        let (report, pixels) = self
            .canvas
            .frame(self.read_next.is_some(), |gpu, target, format| {
                let renderer = renderer.get_or_insert_with(|| map.create_renderer(gpu, format));
                map.render(gpu, renderer, target, viewport, ratio, frame_time)
            })
            .map_err(|e| JsError::new(&e))?;
        if let (Some(pixels), Some(sender)) = (pixels, self.read_next.take()) {
            wasm_bindgen_futures::spawn_local(async move {
                let _ = sender.send(pixels.await);
            });
        }
        let uploaded = orb_hal_wgpu::WgpuGpu::uploaded(self.canvas.gpu_ref());
        let jobs = self.worker_jobs.stats();
        let counts = self.map.job_counts();
        Ok(FrameStats {
            ms: orb_host::Clock::now(&orb_host_web::WebClock) - started,
            zoom: self.map.zoom(),
            generation: report.generation as f64,
            idle: report.idle,
            loaded: report.loaded,
            transitioning: self.map.transitioning(),
            needs_frame: !report.idle && !report.compiling,
            lost: false,
            meshes_uploaded: counts.meshes_uploaded,
            rows_uploaded: counts.rows_uploaded,
            draws: report.frame.draws,
            dispatches: report.frame.dispatches,
            candidates: report.frame.candidates,
            tiles_drawn: report.tiles_drawn,
            tiles_loading: report.tiles_loading,
            bundle_recorded: report.frame.bundle_recorded,
            placements_written: report.frame.placements_written,
            placement_ms: f64::from(report.placement_us) / 1000.0,
            paints_written: report.frame.paints_written,
            uploaded_from_wasm: uploaded.from_core as f64,
            uploaded_from_js: uploaded.from_js as f64,
            texels_from_wasm: uploaded.texels_from_core as f64,
            texels_from_js: uploaded.texels_from_js as f64,
            header_bytes: jobs.header_bytes as f64,
            jobs_done: jobs.done,
            pipelines_pending: report.frame.pipelines_pending,
            layers_waiting: report.frame.nodes_waiting,
            memory: FrameMemory::from(&report.memory),
        })
    }

    fn viewport(&self) -> Viewport {
        let (width, height) = self.canvas.size();
        Viewport {
            width: f64::from(width) / self.pixel_ratio,
            height: f64::from(height) / self.pixel_ratio,
        }
    }
}

/// Whether this browser is a phone's or a tablet's, for the mobile memory
/// budget: `navigator.userAgentData.mobile` where there is one
/// (Chromium, pages and workers), else the user agent string.
fn mobile() -> bool {
    let get = |object: &JsValue, key: &str| js_sys::Reflect::get(object, &key.into()).ok();
    let Some(navigator) = get(&js_sys::global(), "navigator") else {
        return false;
    };
    if let Some(mobile) = get(&navigator, "userAgentData")
        .and_then(|d| get(&d, "mobile"))
        .and_then(|m| m.as_bool())
    {
        return mobile;
    }
    get(&navigator, "userAgent")
        .and_then(|ua| ua.as_string())
        .is_some_and(|ua| {
            ["Android", "iPhone", "iPad", "Mobile"]
                .iter()
                .any(|s| ua.contains(s))
        })
}

/// A promise of a query's features as JSON text.
fn features_promise(
    receiver: futures_channel::oneshot::Receiver<Result<String, String>>,
) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        match receiver.await {
            Ok(Ok(json)) => Ok(json.into()),
            Ok(Err(e)) => Err(JsError::new(&e).into()),
            Err(_) => Err(JsError::new("the map was dropped during a query").into()),
        }
    })
}

pub(crate) fn hit_object(hit: Option<orb_runtime::Hit>) -> JsValue {
    let Some(hit) = hit else {
        return JsValue::NULL;
    };
    let object = js_sys::Object::new();
    let set = |key: &str, value: JsValue| {
        let _ = js_sys::Reflect::set(&object, &key.into(), &value);
    };
    set("layer", hit.layer.into());
    set("source", hit.source.into());
    set(
        "sourceLayer",
        hit.source_layer.map_or(JsValue::NULL, JsValue::from),
    );
    set("id", hit.id.map_or(JsValue::NULL, JsValue::from));
    set("generation", (hit.generation as f64).into());
    object.into()
}
