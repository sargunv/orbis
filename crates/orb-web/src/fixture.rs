//! Offscreen main-thread rendering for the native fixture driver, built into
//! `harness.wasm` with the `harness` feature, never into `core.wasm`. No
//! fixture selection, metadata interpretation, alpha conversion or comparison
//! lives here.

use std::sync::Arc;

use orb_hal::Gpu;
use orb_hal_wgpu::WgpuGpu;
use orb_host::{Fetch, FetchDone};
use orb_host_web::{WebFetch, WebImageDecode};
use orb_jobs::Inline;
use orb_runtime::{Map, PoolJobs, Services, TargetFormat, Viewport};
use wasm_bindgen::prelude::*;

/// One device shared across requests; every request owns a fresh map and scene.
#[wasm_bindgen]
#[derive(Debug)]
pub struct FixtureRenderer {
    device: Device,
    assets_url: String,
}

/// The device requests draw on: WebGPU (Modern), or a WebGL2 context
/// (Compat) each request makes a fresh executor on.
// One per driver page: the variants' sizes don't matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum Device {
    Modern(WgpuGpu),
    Compat(web_sys::WebGl2RenderingContext, u32),
}

/// A request's own GPU resource owner.
#[allow(clippy::large_enum_variant)]
enum Owner {
    Modern(WgpuGpu),
    Compat(orb_hal_gl::GlGpu),
}

impl Owner {
    fn gpu(&mut self) -> &mut dyn Gpu {
        match self {
            Self::Modern(gpu) => gpu,
            Self::Compat(gpu) => gpu,
        }
    }

    async fn read(&mut self, target: orb_hal::TextureId) -> Vec<u8> {
        match self {
            Self::Modern(gpu) => gpu.read_texture(target).await,
            Self::Compat(gpu) => gpu.read_rgba8(target),
        }
    }
}

/// Premultiplied, top-to-bottom RGBA8 pixels and runtime diagnostics.
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug)]
pub struct FixturePixels {
    pub pixels: Vec<u8>,
    pub warnings: Vec<String>,
    pub frames: u32,
    pub tiles_drawn: u32,
    pub timed_out: bool,
}

#[derive(Debug)]
struct AssetFetch(String);

impl AssetFetch {
    /// `local://` and GL JS's test server origin, which TileJSON in the
    /// assets names, are the assets directory.
    fn resolve(&self, url: &str) -> String {
        url.strip_prefix("local://")
            .or_else(|| url.strip_prefix("http://localhost:2900/"))
            .map_or_else(|| url.to_owned(), |path| format!("{}{path}", self.0))
    }
}

impl Fetch for AssetFetch {
    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone) {
        WebFetch.get_range(&self.resolve(url), offset, length, done);
    }
    fn get(&self, url: &str, done: FetchDone) {
        WebFetch.get(&self.resolve(url), done);
    }
}

#[wasm_bindgen]
impl FixtureRenderer {
    /// `assets_url` is the HTTP directory corresponding to `local://`;
    /// `renderer` is `"webgpu"` (Modern) or `"webgl2"` (Compat).
    #[wasm_bindgen(js_name = create)]
    pub async fn create(assets_url: String, renderer: String) -> Result<FixtureRenderer, JsError> {
        let device = match renderer.as_str() {
            "webgl2" => {
                let canvas = web_sys::OffscreenCanvas::new(1, 1)
                    .map_err(|e| JsError::new(&format!("{e:?}")))?;
                let gl = orb_hal_gl::GlGpu::from_canvas(&canvas.clone().into())
                    .map_err(|e| JsError::new(&e))?;
                let max = gl.caps().limits.max_texture_dimension_2d;
                let context = gl.webgl2().expect("a WebGL2 executor").clone();
                Device::Compat(context, max)
            }
            _ => Device::Modern(WgpuGpu::offscreen().await.map_err(|e| JsError::new(&e))?),
        };
        Ok(Self { device, assets_url })
    }

    fn max_size(&self) -> u32 {
        match &self.device {
            Device::Modern(gpu) => gpu.caps().limits.max_texture_dimension_2d,
            Device::Compat(_, max) => *max,
        }
    }

    /// The digest of `orb-text`'s right-to-left and complex-script matrix,
    /// as a decimal string; equal to the native one by decision 0009.
    #[wasm_bindgen(js_name = textMatrixDigest)]
    pub fn text_matrix_digest(&self) -> String {
        orb_text::matrix::digest().to_string()
    }

    /// Maximum physical canvas dimension supported by this device.
    #[wasm_bindgen(getter, js_name = maxTextureDimension2D)]
    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.max_size()
    }

    /// Render at a logical viewport and explicit physical dimensions. The
    /// driver applies GL JS's rounding rules before calling this method.
    /// `operations` is the test's JSON array of GL JS calls, applied as the
    /// native runner applies them: tests with a `fade_duration` follow GL
    /// JS's harness (start once loaded, wait in wall time, read after the
    /// last operation), the rest settle fully between steps. The driver has
    /// skipped tests with calls this can't apply.
    #[allow(clippy::too_many_arguments)]
    pub async fn render(
        &self,
        style: String,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        pixel_ratio: f32,
        timeout_ms: f64,
        fade_duration: f64,
        cross_source_collisions: bool,
        local_ideographs: Option<String>,
        max_pitch: f64,
        operations: String,
    ) -> Result<FixturePixels, JsError> {
        let (_, mut gpu, target, mut output) = self
            .settle(
                style,
                width,
                height,
                logical_width,
                logical_height,
                pixel_ratio,
                timeout_ms,
                fade_duration,
                cross_source_collisions,
                local_ideographs,
                max_pitch,
                operations,
            )
            .await?;
        output.pixels = gpu.read(target).await;
        Ok(output)
    }

    /// Settles a query fixture's style as [`render`](Self::render) does,
    /// then queries `rect` (`[x1, y1, x2, y2]`, logical pixels): the
    /// features' JSON, read through the main-thread jobs.
    #[allow(clippy::too_many_arguments)]
    pub async fn query(
        &self,
        style: String,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        pixel_ratio: f32,
        timeout_ms: f64,
        fade_duration: f64,
        cross_source_collisions: bool,
        local_ideographs: Option<String>,
        max_pitch: f64,
        operations: String,
        rect: Vec<f64>,
    ) -> Result<String, JsError> {
        let [x1, y1, x2, y2] = rect[..] else {
            return Err(JsError::new("a query box has 4 numbers"));
        };
        let (map, ..) = self
            .settle(
                style,
                width,
                height,
                logical_width,
                logical_height,
                pixel_ratio,
                timeout_ms,
                fade_duration,
                cross_source_collisions,
                local_ideographs,
                max_pitch,
                operations,
            )
            .await?;
        let (sender, receiver) = futures_channel::oneshot::channel();
        map.query_rendered(
            glam::DVec2::new(x1, y1),
            glam::DVec2::new(x2, y2),
            None,
            Box::new(move |result| {
                let _ = sender.send(result);
            }),
        );
        receiver
            .await
            .map_err(|_| JsError::new("query dropped"))?
            .map_err(|e| JsError::new(&e))
    }

    /// `["addImage", name, path, options?]`, from the assets, as the native
    /// runner applies it.
    async fn add_image(&self, map: &mut Map, op: &orb_expr::json::Json) -> Result<(), String> {
        let items = op.as_array().unwrap_or_default();
        let text = |i: usize| {
            items
                .get(i)
                .and_then(|v| v.as_str())
                .ok_or("expected a string")
        };
        let url = format!("{}{}", self.assets_url, text(2)?.trim_start_matches("./"));
        let bytes = orb_host_web::fetch(&url).await.map_err(|e| e.to_string())?;
        let image = orb_host_web::decode_image(&bytes)
            .await
            .map_err(|e| format!("{e:?}"))?;
        let options = items.get(3).map_or_else(String::new, ToString::to_string);
        let image = orb_runtime::StyleImage::with_options(
            image.width,
            image.height,
            image.pixels,
            &options,
        )?;
        map.add_image(text(1)?, image)
    }

    /// Renders a fixture's map until it settles, as both runners do.
    #[allow(clippy::too_many_arguments)]
    async fn settle(
        &self,
        style: String,
        width: u32,
        height: u32,
        logical_width: u32,
        logical_height: u32,
        pixel_ratio: f32,
        timeout_ms: f64,
        fade_duration: f64,
        cross_source_collisions: bool,
        local_ideographs: Option<String>,
        max_pitch: f64,
        operations: String,
    ) -> Result<(Map, Owner, orb_hal::TextureId, FixturePixels), JsError> {
        let operations =
            orb_expr::json::parse(&operations).map_err(|e| JsError::new(&e.to_string()))?;
        let operations = operations.as_array().unwrap_or_default();
        if width == 0 || height == 0 || width > self.max_size() || height > self.max_size() {
            return Err(JsError::new("device:canvas-size"));
        }
        let mut gpu = match &self.device {
            Device::Modern(gpu) => {
                Owner::Modern(WgpuGpu::new(gpu.device().clone(), gpu.queue().clone()))
            }
            Device::Compat(context, _) => Owner::Compat(
                orb_hal_gl::GlGpu::from_webgl2(context.clone()).map_err(|e| JsError::new(&e))?,
            ),
        };
        let fetch = Arc::new(AssetFetch(self.assets_url.clone()));
        let jobs = PoolJobs::new(fetch.clone(), Arc::new(WebImageDecode), Arc::new(Inline));
        jobs.set_local_glyphs(Arc::new(orb_host_web::CanvasGlyphs));
        jobs.set_font_files(Arc::new(orb_host::FontFileFaces::new()));
        let mut map = Map::new(
            Services {
                clock: Arc::new(orb_host_web::WebClock),
                image_decode: Arc::new(orb_host_web::WebImageDecode),
                images: Default::default(),
                jobs,
                fetch,
                wake: Arc::new(|| {}),
            },
            None,
        );
        map.set_antialias_3d(false);
        map.set_fade_duration(fade_duration);
        map.set_cross_source_collisions(cross_source_collisions);
        map.set_local_ideograph_font_family(local_ideographs.as_deref());
        map.set_style_json(&style)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let format = TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let target = gpu.gpu().create_texture(&orb_hal::TextureDesc {
            label: "render fixture",
            width,
            height,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: format.color,
            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let viewport = Viewport {
            width: logical_width as f64,
            height: logical_height as f64,
        };
        let mut output = FixturePixels {
            pixels: vec![],
            warnings: vec![],
            frames: 0,
            tiles_drawn: 0,
            timed_out: false,
        };
        let started = js_sys::Date::now();
        let fades = fade_duration > 0.0;
        let loaded = if fades { Next::Loaded } else { Next::Idle };
        // Renders until `until`, or for a wait's wall time; false on the
        // timeout.
        let settle =
            async |map: &mut Map, until: Next, gpu: &mut Owner, output: &mut FixturePixels| {
                let step = js_sys::Date::now();
                loop {
                    // GL JS's transform keeps to the world and its pitch
                    // limit as the harness moves it.
                    let camera = map.camera_mut();
                    camera.pitch = camera.pitch.min(max_pitch);
                    map.constrain_camera(viewport);
                    let report = map.render(gpu.gpu(), target, format, viewport, pixel_ratio);
                    output.frames += 1;
                    output.warnings.extend(map.take_warnings());
                    let done = match until {
                        Next::Continue => true,
                        Next::Loaded => report.loaded,
                        Next::Idle => report.idle,
                        Next::Wait(ms) => js_sys::Date::now() - step >= ms,
                    };
                    if done {
                        return Ok::<bool, JsError>(true);
                    }
                    if js_sys::Date::now() - started >= timeout_ms {
                        output.timed_out = true;
                        return Ok(false);
                    }
                    // Yield to fetch promises and WebGPU callbacks; a Rust busy loop
                    // would starve both on the browser's single thread.
                    let promise = js_sys::Promise::new(&mut |resolve, _| {
                        web_sys::window()
                            .expect("main thread")
                            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 1)
                            .expect("setTimeout");
                    });
                    wasm_bindgen_futures::JsFuture::from(promise)
                        .await
                        .map_err(|e| JsError::new(&format!("setTimeout: {e:?}")))?;
                }
            };
        let mut settled = settle(&mut map, loaded, &mut gpu, &mut output).await?;
        for op in operations {
            if !settled {
                break;
            }
            if op
                .as_array()
                .unwrap_or_default()
                .first()
                .and_then(|n| n.as_str())
                == Some("addImage")
            {
                if let Err(e) = self.add_image(&mut map, op).await {
                    output.warnings.push(format!("operation {op}: {e}"));
                }
                continue;
            }
            match apply(&mut map, op) {
                Ok(Next::Continue) => {}
                Ok(Next::Loaded) => {
                    settled = settle(&mut map, loaded, &mut gpu, &mut output).await?
                }
                Ok(next) => settled = settle(&mut map, next, &mut gpu, &mut output).await?,
                Err(e) => output.warnings.push(format!("operation {op}: {e}")),
            }
        }
        // A transition reads where the waits left it, as GL JS's harness
        // time moves only with them.
        let last = if fades || map.transitioning() {
            Next::Continue
        } else {
            Next::Idle
        };
        if settled && settle(&mut map, last, &mut gpu, &mut output).await? {
            // Draw the state installed by the last frame's messages.
            let report = map.render(gpu.gpu(), target, format, viewport, pixel_ratio);
            output.frames += 1;
            output.tiles_drawn = report.tiles_drawn;
            output.warnings.extend(map.take_warnings());
        }
        Ok((map, gpu, target, output))
    }
}

/// What the runner does after an operation (as `orb-fixtures`' `Next`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Next {
    Continue,
    /// `["wait"]`: until loaded (GL JS `loaded()`).
    Loaded,
    /// `["idle"]`: until idle, fades done.
    Idle,
    /// `["wait", ms]`: render for `ms` of wall time.
    Wait(f64),
}

/// Applies one test operation as `orb-fixtures`' native runner does.
fn apply(map: &mut Map, op: &orb_expr::json::Json) -> Result<Next, String> {
    let items = op.as_array().unwrap_or_default();
    let name = items
        .first()
        .and_then(|n| n.as_str())
        .ok_or("unnamed operation")?;
    let number = |i: usize| {
        items
            .get(i)
            .and_then(|n| n.as_f64())
            .ok_or("expected a number")
    };
    match name {
        "wait" => {
            return Ok(match items.get(1).and_then(|ms| ms.as_f64()) {
                Some(ms) => Next::Wait(ms),
                None => Next::Loaded,
            });
        }
        "idle" => return Ok(Next::Idle),
        "setZoom" => {
            let (zoom, projection) = (number(1)?, map.projection());
            map.camera_mut().set_zoom(&projection, zoom)
        }
        "setBearing" => map.camera_mut().bearing = number(1)?,
        "setPitch" => map.camera_mut().pitch = number(1)?,
        "setRoll" => map.camera_mut().roll = number(1)?,
        "setCenter" => {
            let center = items.get(1).and_then(|c| c.as_array()).unwrap_or_default();
            let at = |i: usize| {
                center
                    .get(i)
                    .and_then(|n| n.as_f64())
                    .ok_or("expected a number")
            };
            map.camera_mut().target = orb_geo::Geodetic::new(at(0)?, at(1)?);
        }
        "removeImage" => {
            let name = items
                .get(1)
                .and_then(|n| n.as_str())
                .ok_or("expected a name")?;
            map.remove_image(name)?;
        }
        _ => {
            let tx = orb_runtime::Transaction::from_commands(&format!("[{op}]"))?;
            map.edit(&tx)?;
        }
    }
    Ok(Next::Continue)
}
