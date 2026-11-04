//! The browser build of Orbis: `core.wasm` and its wasm-bindgen surface.
//!
//! One module, three roles (plan §7.1), chosen by the page's JS:
//! - **ui** ([`UiRole`]), always on the page's thread: the camera, gestures
//!   and synchronous `project`, `unproject` and `hitTest`.
//! - **main thread** ([`create_map`], plan §7.2's `thread: "main"`): the
//!   map, its jobs and rendering share the page's thread, drawing into a
//!   canvas.
//! - **render worker** ([`create_worker_map`]): the map draws into a
//!   transferred `OffscreenCanvas` and sends tile jobs to job workers.
//! - **job worker** ([`jobs::JobWorker`]): fetches and lays out tiles.
//!
//! Both maps draw through WebGPU. With the `harness` feature the module also
//! exports the frame and paint benches (`runBench`) and the render fixture
//! driver (`FixtureRenderer`); `cargo xtask web` builds that into
//! `harness.wasm`, apart from `core.wasm`.

#![cfg(target_arch = "wasm32")]

#[cfg(feature = "harness")]
pub mod bench;
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
use orb_hal::Gpu;
use orb_hal_wgpu::{WgpuGpu, from_texture_format};
use orb_host_web::{WebFetch, WebImageDecode};
use orb_jobs::Inline;
use orb_runtime::{CameraState, Jobs, Map, MapEvent, PoolJobs, Services, TargetFormat, Viewport};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, OffscreenCanvas};

use crate::jobs::WorkerJobs;
use crate::placement::WorkerPlacement;

thread_local! {
    /// The page's callback for "draw a frame soon". Wasm in a page or
    /// worker is single-threaded, so the runtime's `Send` wake hook reaches
    /// it here.
    static WAKE: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

fn wake() {
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
    gpu: WgpuGpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    format: TargetFormat,
    pixel_ratio: f64,
    /// Set in the render worker.
    worker_jobs: Option<Arc<WorkerJobs>>,
    /// The picking index changed since it was last published.
    hits_dirty: bool,
    /// The last frame's camera and how many frames in a row showed it, to
    /// publish picking once it settles.
    last_camera: Option<(CameraState, u32)>,
}

/// Frame counters, for the page's HUD.
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug)]
pub struct FrameInfo {
    #[wasm_bindgen(readonly)]
    pub zoom: f64,
    /// The frame's camera, as [`UiRole::camera`] gives it, so DOM overlays
    /// can follow the frame on screen.
    #[wasm_bindgen(readonly)]
    pub camera: Vec<f64>,
    /// A picking snapshot for the page's [`UiRole`]: in the render worker,
    /// once the camera settles on a changed index.
    #[wasm_bindgen(readonly)]
    pub hit_snapshot: Option<Vec<u8>>,
    #[wasm_bindgen(readonly)]
    pub meshes_uploaded: u32,
    #[wasm_bindgen(readonly)]
    pub rows_uploaded: u32,
    #[wasm_bindgen(readonly)]
    pub draws: u32,
    #[wasm_bindgen(readonly)]
    pub dispatches: u32,
    #[wasm_bindgen(readonly)]
    pub candidates: u32,
    #[wasm_bindgen(readonly)]
    pub tiles_drawn: u32,
    #[wasm_bindgen(readonly)]
    pub tiles_loading: u32,
    #[wasm_bindgen(readonly)]
    pub bundle_recorded: bool,
    #[wasm_bindgen(readonly)]
    pub placements_written: u32,
    #[wasm_bindgen(readonly)]
    pub paints_written: u32,
    #[wasm_bindgen(readonly)]
    pub idle: bool,
    /// The style generation the frame shows in full.
    #[wasm_bindgen(readonly)]
    pub generation: f64,
    #[wasm_bindgen(readonly)]
    pub warnings: Vec<String>,
    /// Bytes written to GPU buffers from wasm memory, since the map was
    /// created.
    #[wasm_bindgen(readonly)]
    pub uploaded_from_wasm: f64,
    /// Bytes written to GPU buffers straight from JS `ArrayBuffer`s.
    #[wasm_bindgen(readonly)]
    pub uploaded_from_js: f64,
    /// Texel bytes written to GPU textures from wasm memory.
    #[wasm_bindgen(readonly)]
    pub texels_from_wasm: f64,
    /// Texel bytes copied to GPU textures straight from `ImageBitmap`s.
    #[wasm_bindgen(readonly)]
    pub texels_from_js: f64,
    /// Bytes of packed tile headers copied into wasm memory (render
    /// worker).
    #[wasm_bindgen(readonly)]
    pub header_bytes: f64,
    /// Tile jobs answered by job workers (render worker).
    #[wasm_bindgen(readonly)]
    pub jobs_done: u32,
}

/// Creates a map in `canvas` showing the style at `style_url`, running its
/// jobs on this thread. `on_wake` is called when the map has new data to
/// draw; the page should call `render` on its next animation frame.
#[wasm_bindgen(js_name = createMap)]
pub async fn create_map(
    canvas: HtmlCanvasElement,
    style_url: String,
    on_wake: js_sys::Function,
    shaper_loader: js_sys::Function,
) -> Result<OrbisMap, JsError> {
    let size = (canvas.width(), canvas.height());
    let fetch = Arc::new(WebFetch);
    let jobs = PoolJobs::new(fetch, Arc::new(WebImageDecode), Arc::new(Inline));
    jobs.set_local_glyphs(Arc::new(orb_host_web::CanvasGlyphs));
    // `font-faces` files draw with shaper.wasm, loaded on first use.
    orb_host_web::ShaperFiles::set_loader(shaper_loader);
    jobs.set_font_files(Arc::new(orb_host_web::ShaperFiles));
    let target = wgpu::SurfaceTarget::Canvas(canvas);
    create(target, size, &style_url, on_wake, jobs, None, None).await
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
    let jobs = WorkerJobs::new(job_workers, post);
    let target = wgpu::SurfaceTarget::OffscreenCanvas(canvas);
    let lane: Arc<dyn orb_runtime::PlacementLane> = WorkerPlacement::new(place);
    create(
        target,
        size,
        &style_url,
        on_wake,
        jobs.clone(),
        Some(jobs),
        Some(lane),
    )
    .await
}

async fn create(
    target: wgpu::SurfaceTarget<'static>,
    (width, height): (u32, u32),
    style_url: &str,
    on_wake: js_sys::Function,
    jobs: Arc<dyn Jobs>,
    worker_jobs: Option<Arc<WorkerJobs>>,
    lane: Option<Arc<dyn orb_runtime::PlacementLane>>,
) -> Result<OrbisMap, JsError> {
    WAKE.with(|w| *w.borrow_mut() = Some(on_wake));
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let surface = instance
        .create_surface(target)
        .map_err(|e| JsError::new(&format!("surface: {e}")))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|e| JsError::new(&format!("no WebGPU adapter: {e}")))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("orbis"),
            ..Default::default()
        })
        .await
        .map_err(|e| JsError::new(&format!("no WebGPU device: {e}")))?;
    let caps = surface.get_capabilities(&adapter);
    let surface_format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb() && from_texture_format(*f).is_some())
        .ok_or_else(|| JsError::new("no non-sRGB 8-bit canvas format"))?;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
        view_formats: vec![],
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    surface.configure(&device, &config);

    let services = Services {
        clock: Arc::new(orb_host_web::WebClock),
        image_decode: Arc::new(orb_host_web::WebImageDecode),
        images: Default::default(),
        jobs,
        fetch: Arc::new(WebFetch),
        wake: Arc::new(wake),
    };
    let mut map = Map::new(services, None);
    if let Some(lane) = lane {
        map.set_placement_lane(lane);
    }
    map.load_style_url(style_url);
    Ok(OrbisMap {
        map,
        gpu: WgpuGpu::new(device, queue),
        surface,
        config,
        format: TargetFormat {
            color: from_texture_format(surface_format).expect("checked above"),
            sample_count: 1,
        },
        pixel_ratio: 1.0,
        worker_jobs,
        hits_dirty: false,
        last_camera: None,
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
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.pixel_ratio = pixel_ratio;
        self.surface.configure(self.gpu.device(), &self.config);
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
                        js_sys::Float64Array::from(&ui::camera_array(&c)[..]).into()
                    });
                    set(&object, "camera", camera);
                    let projection = orb_runtime::projection_to_json(&self.map.projection());
                    set(&object, "projection", projection.into());
                    "styleload"
                }
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
    /// ([`FrameInfo::generation`]).
    pub fn edit(&mut self, commands: &str) -> Result<f64, JsError> {
        let tx = orb_runtime::Transaction::from_commands(commands).map_err(|e| JsError::new(&e))?;
        let generation = self.map.edit(&tx).map_err(|e| JsError::new(&e))?;
        Ok(generation as f64)
    }

    /// Completes a tile job with a job worker's reply (render worker only).
    #[wasm_bindgen(js_name = receiveJob)]
    pub fn receive_job(&mut self, reply: &JsValue) -> Result<(), JsError> {
        let jobs = self
            .worker_jobs
            .as_ref()
            .ok_or_else(|| JsError::new("this map runs its jobs on its own thread"))?;
        jobs.receive(&mut self.gpu, reply)
    }

    /// Completes a placement with the placement worker's reply (render
    /// worker only).
    #[wasm_bindgen(js_name = receivePlacement)]
    pub fn receive_placement(&mut self, reply: &JsValue) -> Result<(), JsError> {
        WorkerPlacement::receive(reply)
    }

    /// Draws a frame.
    pub fn render(&mut self) -> Result<FrameInfo, JsError> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            other => return Err(JsError::new(&format!("canvas texture: {other:?}"))),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let target = self.gpu.register_view(view);
        let viewport = self.viewport();
        let report = self.map.render(
            &mut self.gpu,
            target,
            self.format,
            viewport,
            self.pixel_ratio as f32,
        );
        self.gpu.destroy_texture(target);
        self.gpu.queue().present(frame);
        let uploaded = self.gpu.uploaded();
        let jobs = self.worker_jobs.as_ref().map(|j| j.stats());
        // Picking publishes once the camera settles: a moving camera would
        // send a snapshot every frame, and the ui role maps points through
        // the ground meanwhile.
        let camera = *self.map.camera();
        self.hits_dirty |= self.map.hits_rebuilt();
        let still = match self.last_camera {
            Some((last, frames)) if last == camera => frames + 1,
            _ => 0,
        };
        self.last_camera = Some((camera, still));
        // A gesture's camera can rest for a frame between inputs; a
        // snapshot is worth sending once it has rested a few, or the map is
        // idle.
        let settled = still >= SETTLE_FRAMES || (still > 0 && report.idle);
        let hit_snapshot = (self.worker_jobs.is_some() && settled && self.hits_dirty).then(|| {
            self.hits_dirty = false;
            self.map.hit_snapshot()
        });
        Ok(FrameInfo {
            zoom: camera.zoom(),
            camera: ui::camera_array(&camera),
            hit_snapshot,
            meshes_uploaded: self.map.job_counts().meshes_uploaded,
            rows_uploaded: self.map.job_counts().rows_uploaded,
            draws: report.frame.draws,
            dispatches: report.frame.dispatches,
            candidates: report.frame.candidates,
            tiles_drawn: report.tiles_drawn,
            tiles_loading: report.tiles_loading,
            bundle_recorded: report.frame.bundle_recorded,
            placements_written: report.frame.placements_written,
            paints_written: report.frame.paints_written,
            idle: report.idle,
            generation: report.generation as f64,
            warnings: self.map.take_warnings(),
            uploaded_from_wasm: uploaded.from_core as f64,
            uploaded_from_js: uploaded.from_js as f64,
            texels_from_wasm: uploaded.texels_from_core as f64,
            texels_from_js: uploaded.texels_from_js as f64,
            header_bytes: jobs.map_or(0.0, |j| j.header_bytes as f64),
            jobs_done: jobs.map_or(0, |j| j.done),
        })
    }

    fn viewport(&self) -> Viewport {
        Viewport {
            width: f64::from(self.config.width) / self.pixel_ratio,
            height: f64::from(self.config.height) / self.pixel_ratio,
        }
    }
}

/// Frames a camera rests before the render worker publishes picking.
const SETTLE_FRAMES: u32 = 3;

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
