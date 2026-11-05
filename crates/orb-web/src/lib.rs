//! The browser build of Orbis: `core.wasm` and its wasm-bindgen surface.
//!
//! One module, three roles (plan §7.1), chosen by the page's JS:
//! - **ui** ([`UiRole`]), always on the page's thread: the camera, gestures
//!   and synchronous `project`, `unproject` and `hitTest`.
//! - **main thread** ([`create_map`], plan §7.2's `thread: "main"`): the
//!   map draws into a canvas on the page's thread.
//! - **render worker** ([`create_worker_map`]): the map draws into a
//!   transferred `OffscreenCanvas`.
//! - **job worker** ([`jobs::JobWorker`]): fetches and lays out tiles for
//!   either map; a placement worker ([`PlacementWorker`]) places its labels.
//!
//! Both maps draw through WebGPU. With the `harness` feature the module also
//! exports the benches (`runBench`) and the render fixture
//! driver (`FixtureRenderer`); `cargo xtask web` builds that into
//! `harness.wasm`, apart from `core.wasm`.

#![cfg(target_arch = "wasm32")]

#[cfg(feature = "harness")]
pub mod bench;
mod canvas;
#[cfg(feature = "harness")]
mod fixture;
mod jobs;
mod placement;
mod ui;

#[cfg(feature = "harness")]
pub use fixture::{FixturePixels, FixtureRenderer};
pub use jobs::JobWorker;
pub use placement::PlacementWorker;
pub use ui::UiRole;

use std::cell::RefCell;
use std::sync::Arc;

use glam::DVec2;
use orb_host_web::WebFetch;
use orb_runtime::{CameraState, Jobs, Map, MapEvent, Services, Viewport};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, OffscreenCanvas};

use crate::canvas::Canvas;
use crate::jobs::WorkerJobs;
use crate::placement::WorkerPlacement;

thread_local! {
    /// The page's callback for "draw a frame soon". Wasm in a page or
    /// worker is single-threaded, so the runtime's `Send` wake hook reaches
    /// it here.
    static WAKE: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

pub(crate) fn wake() {
    WAKE.with(|wake| {
        if let Some(f) = wake.borrow().as_ref() {
            let _ = f.call0(&JsValue::NULL);
        }
    });
}

/// A map drawing into a canvas.
#[wasm_bindgen]
#[derive(Debug)]
pub struct OrbisMap {
    map: Map,
    canvas: Canvas,
    pixel_ratio: f64,
    worker_jobs: Arc<WorkerJobs>,
    /// What picking the page's ui role has: in the render worker; on the
    /// page the map answers hit tests itself.
    hits: Option<orb_runtime::HitPublication>,
    /// The last frame's camera and how many frames in a row showed it, to
    /// publish the elevation pyramid once it settles.
    last_camera: Option<(CameraState, u32)>,
    /// The terrain revision last published to the ui role.
    elevation_revision: u64,
    /// Frames since the pyramid was last published.
    elevation_age: u32,
}

/// What a frame tells the page: its counters and camera packed in one
/// array, and what changed for the page's [`UiRole`].
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug)]
pub struct FrameInfo {
    /// `zoom, generation, idle, meshes_uploaded, rows_uploaded, draws,
    /// dispatches, candidates, tiles_drawn, tiles_loading, bundle_recorded,
    /// placements_written, paints_written, uploaded_from_wasm,
    /// uploaded_from_js, texels_from_wasm, texels_from_js, header_bytes,
    /// jobs_done, pipelines_pending, layers_waiting, lost`, then [`MEMORY`]
    /// words of memory (plan §8's budget: see [`memory_stats`]), then the
    /// frame's camera as [`UiRole::camera`]
    /// gives it, so DOM overlays can follow the frame on screen
    /// (`web/src/frames.ts` reads them by name). Booleans are 0 or 1; byte
    /// counts are since the map was created: buffer bytes written from wasm
    /// memory and straight from JS `ArrayBuffer`s, texel bytes from wasm
    /// memory and from `ImageBitmap`s, packed tile headers copied into wasm
    /// memory; the generation is the style generation the frame shows in
    /// full; `lost` means the GPU was lost, nothing was drawn and the map
    /// draws again once [`OrbisMap::recover`] gives it another.
    #[wasm_bindgen(readonly)]
    pub stats: Vec<f64>,
    /// Picking's changes for the page's [`UiRole`], in the render worker
    /// (`orb_runtime::Map::publish_hits`).
    #[wasm_bindgen(readonly)]
    pub hits: Option<Vec<u8>>,
    /// The terrain's elevation pyramid for the page's [`UiRole`] when it
    /// changed (empty: the terrain is gone), so its camera stands on the
    /// terrain (plan §7.1).
    #[wasm_bindgen(readonly)]
    pub elevation: Option<Vec<u8>>,
    #[wasm_bindgen(readonly)]
    pub warnings: Vec<String>,
}

/// Creates a map in `canvas` showing the style at `style_url`, drawing on
/// this thread. Its tile jobs and labels go to job workers and a placement
/// worker as [`create_worker_map`]'s do (plan §7.1: the page draws in
/// `thread: "main"`, but jobs stay off it). `on_wake` is called when the
/// map has new data to draw; the page should call `render` on its next
/// animation frame.
#[wasm_bindgen(js_name = createMap)]
pub async fn create_map(
    canvas: HtmlCanvasElement,
    style_url: String,
    on_wake: js_sys::Function,
    job_workers: u32,
    post: js_sys::Function,
    place: js_sys::Function,
) -> Result<OrbisMap, JsError> {
    let size = (canvas.width(), canvas.height());
    let mut map = create(
        canvas.into(),
        size,
        &style_url,
        on_wake,
        job_workers,
        post,
        place,
    )
    .await?;
    map.hits = None;
    Ok(map)
}

/// Creates a map in a render worker, in a canvas transferred from the page.
/// Tile jobs go to `job_workers` job workers through `post(worker,
/// message)`; their replies come back through
/// [`receive_job`](OrbisMap::receive_job). Labels are placed by the
/// placement worker `place(message, transfer)` reaches, whose replies come
/// back through [`receive_placement`](OrbisMap::receive_placement).
/// `on_wake` is as for [`create_map`].
#[wasm_bindgen(js_name = createWorkerMap)]
pub async fn create_worker_map(
    canvas: OffscreenCanvas,
    style_url: String,
    on_wake: js_sys::Function,
    job_workers: u32,
    post: js_sys::Function,
    place: js_sys::Function,
) -> Result<OrbisMap, JsError> {
    let size = (canvas.width(), canvas.height());
    create(
        canvas.into(),
        size,
        &style_url,
        on_wake,
        job_workers,
        post,
        place,
    )
    .await
}

async fn create(
    canvas: JsValue,
    size: (u32, u32),
    style_url: &str,
    on_wake: js_sys::Function,
    job_workers: u32,
    post: js_sys::Function,
    place: js_sys::Function,
) -> Result<OrbisMap, JsError> {
    let worker_jobs = WorkerJobs::new(job_workers, post);
    let jobs: Arc<dyn Jobs> = worker_jobs.clone();
    let lane: Arc<dyn orb_runtime::PlacementLane> = WorkerPlacement::new(place);
    WAKE.with(|w| *w.borrow_mut() = Some(on_wake));
    let canvas = Canvas::new(canvas, size)
        .await
        .map_err(|e| JsError::new(&e))?;

    let services = Services {
        clock: Arc::new(orb_host_web::WebClock),
        image_decode: Arc::new(orb_host_web::WebImageDecode),
        images: Default::default(),
        jobs,
        fetch: Arc::new(WebFetch),
        wake: Arc::new(wake),
    };
    let mut map = Map::new(services, None);
    if mobile() {
        map.set_memory_budget(orb_runtime::MOBILE_BUDGET);
    }
    map.set_placement_lane(lane);
    map.load_style_url(style_url);
    Ok(OrbisMap {
        map,
        canvas,
        pixel_ratio: 1.0,
        worker_jobs,
        hits: Some(Default::default()),
        last_camera: None,
        elevation_revision: 0,
        elevation_age: 0,
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

    /// The last placement commit's semantic snapshot (plan §13) as JSON,
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

    /// Matches the drawing buffer to the canvas's size in physical pixels.
    pub fn resize(&mut self, width: u32, height: u32, pixel_ratio: f64) {
        self.pixel_ratio = pixel_ratio;
        self.canvas.resize(width, height);
    }

    /// Sets the camera from the ui role: `[lon, lat, resolution, bearing,
    /// pitch, fov, …]`. Until the first call the map takes its style's.
    #[wasm_bindgen(js_name = setCamera)]
    pub fn set_camera(&mut self, camera: &[f64]) -> Result<(), JsError> {
        self.map.set_camera(ui::camera_from(camera)?);
        Ok(())
    }

    /// What the map did since the last call, as plain objects: `{type:
    /// "styleload", camera, projection}` (the style's camera if the map
    /// took it, else null; the surface's projection as JSON), `{type: "styledata", generation}`, `{type: "sourcedata",
    /// source, loaded}`, `{type: "load"}`, `{type: "idle"}`, `{type:
    /// "error", message}`.
    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        let set = |object: &js_sys::Object, key: &str, value: JsValue| {
            let _ = js_sys::Reflect::set(object, &key.into(), &value);
        };
        for event in self.map.take_events() {
            let object = js_sys::Object::new();
            let kind = match event {
                MapEvent::StyleLoad { camera } => {
                    let camera = camera.map_or(JsValue::NULL, |c| {
                        js_sys::Float64Array::from(
                            &ui::camera_array(&c, &self.map.projection())[..],
                        )
                        .into()
                    });
                    set(&object, "camera", camera);
                    "styleload"
                }
                MapEvent::StyleData { generation } => {
                    set(&object, "generation", (generation as f64).into());
                    // Loads and projection edits move the camera's surface.
                    let surface = orb_runtime::surface_to_json(&self.map.surface());
                    set(&object, "surface", surface.into());
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
                MapEvent::StyleImageMissing { name } => {
                    set(&object, "id", name.into());
                    "styleimagemissing"
                }
                MapEvent::Capability { tier } => {
                    let tier = match tier {
                        orb_hal::Tier::Modern => "modern",
                        orb_hal::Tier::Compat => "compat",
                    };
                    set(&object, "tier", tier.into());
                    "capability"
                }
            };
            set(&object, "type", kind.into());
            out.push(&object);
        }
        out
    }

    /// Applies GL JS map calls as one transaction (plan §11.2): a JSON array
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

    /// Completes a placement with the placement worker's reply.
    #[wasm_bindgen(js_name = receivePlacement)]
    pub fn receive_placement(&mut self, reply: &JsValue) -> Result<(), JsError> {
        WorkerPlacement::receive(reply)
    }

    /// The tier the map draws at: `"modern"` (WebGPU) or `"compat"`
    /// (WebGL2).
    #[wasm_bindgen(getter)]
    pub fn tier(&self) -> String {
        match self.canvas.gpu_ref().caps().tier {
            orb_hal::Tier::Modern => "modern".into(),
            orb_hal::Tier::Compat => "compat".into(),
        }
    }

    /// Loses the GPU on purpose (`GPUDevice.destroy()`, or WebGL's
    /// `WEBGL_lose_context`), to exercise recovery.
    #[wasm_bindgen(js_name = loseDevice)]
    pub fn lose_device(&mut self) {
        self.canvas.lose();
    }

    /// Draws on a new GPU in `canvas` after a loss (plan §8.8): the same
    /// canvas again, or a fresh one. The map keeps its style, camera and
    /// cached tiles; the next frames lay the visible tiles out again.
    /// Resolves to the new tier.
    pub async fn recover(&mut self, canvas: JsValue) -> Result<String, JsError> {
        let size = self.canvas.size();
        self.canvas = Canvas::new(canvas, size)
            .await
            .map_err(|e| JsError::new(&e))?;
        self.map.reset_gpu();
        // The ui role's picking is published afresh.
        if let Some(hits) = &mut self.hits {
            *hits = orb_runtime::HitPublication::default();
        }
        wake();
        Ok(self.tier())
    }

    /// Draws a frame.
    pub fn render(&mut self) -> Result<FrameInfo, JsError> {
        if self.canvas.lost() {
            return Ok(FrameInfo::lost());
        }
        let viewport = self.viewport();
        let (map, ratio) = (&mut self.map, self.pixel_ratio as f32);
        let report = self
            .canvas
            .frame(|gpu, target, format| map.render(gpu, target, format, viewport, ratio))
            .map_err(|e| JsError::new(&e))?;
        let uploaded = self.canvas.gpu_ref().uploaded();
        let jobs = self.worker_jobs.stats();
        // Picking publishes what changed: tiles, the drawn list, labels and
        // the layer table; camera moves alone publish nothing.
        let hits = self.hits.as_mut().and_then(|to| self.map.publish_hits(to));
        let camera = *self.map.camera();
        let still = match self.last_camera {
            Some((last, frames)) if last == camera => frames + 1,
            _ => 0,
        };
        self.last_camera = Some((camera, still));
        let settled = still == SETTLE_FRAMES || (still > 0 && report.idle);
        // The ui role's camera keeps above the terrain the pyramid shows,
        // and an animation ends on it, so it publishes while the camera
        // moves too: at once when the ui role has none, then at most every
        // `PUBLISH_FRAMES` while DEM tiles arrive in bursts, and once the
        // camera settles.
        let revision = self.map.elevation_revision();
        self.elevation_age = self.elevation_age.saturating_add(1);
        let due = settled || self.elevation_revision == 0 || self.elevation_age >= PUBLISH_FRAMES;
        let elevation = (due && revision != self.elevation_revision).then(|| {
            self.elevation_revision = revision;
            self.elevation_age = 0;
            self.map.elevation_pyramid().to_vec()
        });
        let counts = self.map.job_counts();
        let mut stats = vec![
            self.map.zoom(),
            report.generation as f64,
            f64::from(u8::from(report.idle)),
            f64::from(counts.meshes_uploaded),
            f64::from(counts.rows_uploaded),
            f64::from(report.frame.draws),
            f64::from(report.frame.dispatches),
            f64::from(report.frame.candidates),
            f64::from(report.tiles_drawn),
            f64::from(report.tiles_loading),
            f64::from(u8::from(report.frame.bundle_recorded)),
            f64::from(report.frame.placements_written),
            f64::from(report.frame.paints_written),
            uploaded.from_core as f64,
            uploaded.from_js as f64,
            uploaded.texels_from_core as f64,
            uploaded.texels_from_js as f64,
            jobs.header_bytes as f64,
            f64::from(jobs.done),
            f64::from(report.frame.pipelines_pending),
            f64::from(report.frame.nodes_waiting),
            0.0,
        ];
        stats.extend(memory_stats(&report.memory));
        stats.extend(ui::camera_array(&camera, &self.map.projection()));
        Ok(FrameInfo {
            stats,
            hits,
            elevation,
            warnings: self.map.take_warnings(),
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

/// [`FrameInfo::stats`]' counters before the camera, and where idleness is.
const STATS: usize = 22;
const IDLE: usize = 2;
/// Words of memory after the counters.
const MEMORY: usize = 17;

/// A frame's memory for the page: `budget, allocated, vertices used,
/// allocated, ceiling and largest free block, indices used, allocated and
/// ceiling, compacted indices used and allocated, properties used and
/// allocated, textures (tile pages, glyph pages and the image atlas),
/// cached tiles, their bytes and the cache budget`, in bytes.
fn memory_stats(m: &orb_runtime::Memory) -> [f64; MEMORY] {
    let s = &m.scene;
    [
        m.budget,
        m.allocated(),
        s.vertices.used,
        s.vertices.allocated,
        s.vertices.ceiling,
        s.vertices.largest_free,
        s.indices.used,
        s.indices.allocated,
        s.indices.ceiling,
        s.compacted.used,
        s.compacted.allocated,
        s.props.used,
        s.props.allocated,
        m.tile_textures + m.glyph_textures + m.image_atlas,
        u64::from(m.cached_tiles),
        m.cached,
        m.cache_budget,
    ]
    .map(|v| v as f64)
}

impl FrameInfo {
    /// A frame not drawn: the GPU is gone. Idle, lost, no camera.
    fn lost() -> Self {
        let mut stats = vec![0.0; STATS + MEMORY];
        stats[IDLE] = 1.0;
        stats[STATS - 1] = 1.0;
        Self {
            stats,
            hits: None,
            elevation: None,
            warnings: Vec::new(),
        }
    }
}

/// Frames a camera rests before the render worker publishes the
/// elevation pyramid.
const SETTLE_FRAMES: u32 = 3;

/// How often, in frames, the elevation pyramid may publish while the camera
/// moves: about twice a second.
const PUBLISH_FRAMES: u32 = 30;

/// Whether this browser is a phone's or a tablet's, for plan §8's mobile
/// memory budget: `navigator.userAgentData.mobile` where there is one
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
