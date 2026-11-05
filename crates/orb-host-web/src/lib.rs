//! Host services in the browser, for the main thread and workers alike.

#![forbid(unsafe_code)]
#![cfg(target_arch = "wasm32")]

mod storage;

pub use storage::OpfsStorage;

use js_sys::{Reflect, Uint8Array};
use orb_host::{DecodedImage, Fetch, FetchDone, FetchError, ImageDecode, ImageDone};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

#[derive(Clone, Copy, Debug, Default)]
pub struct WebFetch;

/// Decodes images with the browser's decoder: `createImageBitmap` without
/// premultiplication or colour conversion, read back through an
/// `OffscreenCanvas`, on the page or in a worker.
#[derive(Clone, Copy, Debug, Default)]
pub struct WebImageDecode;

impl ImageDecode for WebImageDecode {
    fn decode(&self, bytes: Vec<u8>, done: ImageDone) {
        wasm_bindgen_futures::spawn_local(async move {
            done(
                decode_image(&bytes)
                    .await
                    .map_err(|e| format!("image decode: {e:?}")),
            )
        });
    }
}

/// Decodes an image to straight RGBA8 texels.
pub async fn decode_image(bytes: &[u8]) -> Result<DecodedImage, JsValue> {
    let bitmap = decode_bitmap(bytes).await?;
    let image = read_bitmap(&bitmap);
    bitmap.close();
    image
}

/// Reads a bitmap's straight RGBA8 texels through an `OffscreenCanvas`.
fn read_bitmap(bitmap: &web_sys::ImageBitmap) -> Result<DecodedImage, JsValue> {
    let (width, height) = (bitmap.width(), bitmap.height());
    let canvas = web_sys::OffscreenCanvas::new(width, height)?;
    let context: web_sys::OffscreenCanvasRenderingContext2d = canvas
        .get_context("2d")?
        .ok_or_else(|| JsValue::from_str("no 2d context"))?
        .dyn_into()?;
    context.draw_image_with_image_bitmap(bitmap, 0.0, 0.0)?;
    let data = context.get_image_data(0.0, 0.0, f64::from(width), f64::from(height))?;
    Ok(DecodedImage {
        width,
        height,
        pixels: data.data().to_vec(),
    })
}

/// Decodes an image into an `ImageBitmap` with straight alpha and no colour
/// conversion, on the page or in a worker. Its pixels stay with the
/// browser: a job worker transfers the bitmap to the render worker, which
/// copies it to a texture.
pub async fn decode_bitmap(bytes: &[u8]) -> Result<web_sys::ImageBitmap, JsValue> {
    let parts = js_sys::Array::of1(&Uint8Array::from(bytes));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
    let options = js_sys::Object::new();
    Reflect::set(&options, &"premultiplyAlpha".into(), &"none".into())?;
    Reflect::set(&options, &"colorSpaceConversion".into(), &"none".into())?;
    let global = js_sys::global();
    let create =
        Reflect::get(&global, &"createImageBitmap".into())?.dyn_into::<js_sys::Function>()?;
    let promise = create
        .call2(&global, &blob, &options)?
        .dyn_into::<js_sys::Promise>()?;
    JsFuture::from(promise).await?.dyn_into()
}

impl Fetch for WebFetch {
    fn get(&self, url: &str, done: FetchDone) {
        let url = url.to_string();
        wasm_bindgen_futures::spawn_local(async move { done(fetch(&url).await) });
    }

    fn get_range(&self, url: &str, offset: u64, length: u64, done: FetchDone) {
        let url = url.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            done(fetch_body(&url, Some((offset, length))).await)
        });
    }
}

pub async fn fetch(url: &str) -> Result<Vec<u8>, FetchError> {
    fetch_body(url, None).await
}

thread_local! {
    static REQUEST_HOOK: std::cell::RefCell<Option<js_sys::Function>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Transforms every request this JS context's fetches make (GL JS
/// `transformRequest`): `hook(url)` answers `{url, headers}` (or a promise
/// of it), `null` to refuse, or `undefined` to leave the request.
pub fn set_request_hook(hook: Option<js_sys::Function>) {
    REQUEST_HOOK.with(|h| *h.borrow_mut() = hook);
}

async fn fetch_body(url: &str, range: Option<(u64, u64)>) -> Result<Vec<u8>, FetchError> {
    let network = |e: JsValue| FetchError::Network(format!("{e:?}"));
    let global = js_sys::global();
    let fetch = Reflect::get(&global, &"fetch".into())
        .map_err(network)?
        .dyn_into::<js_sys::Function>()
        .map_err(network)?;
    let init = web_sys::RequestInit::new();
    let headers = web_sys::Headers::new().map_err(network)?;
    let mut url = JsValue::from(url);
    if let Some(hook) = REQUEST_HOOK.with(|h| h.borrow().clone()) {
        let mut request = hook.call1(&JsValue::NULL, &url).map_err(network)?;
        if let Ok(promise) = request.clone().dyn_into::<js_sys::Promise>() {
            request = JsFuture::from(promise).await.map_err(network)?;
        }
        if request.is_null() {
            let url = url.as_string().unwrap_or_default();
            return Err(FetchError::UnsupportedUrl(format!("{url}: refused")));
        }
        if !request.is_undefined() {
            url = Reflect::get(&request, &"url".into()).map_err(network)?;
            let extra = Reflect::get(&request, &"headers".into()).map_err(network)?;
            if extra.is_object() {
                for entry in js_sys::Object::entries(extra.unchecked_ref()).iter() {
                    let entry: js_sys::Array = entry.unchecked_into();
                    let (name, value) = (entry.get(0), entry.get(1));
                    let (Some(name), Some(value)) = (name.as_string(), value.as_string()) else {
                        continue;
                    };
                    headers.set(&name, &value).map_err(network)?;
                }
            }
        }
    }
    if let Some((offset, length)) = range {
        let end = orb_host::range_end(offset, length)?;
        headers
            .set("Range", &format!("bytes={offset}-{end}"))
            .map_err(network)?;
    }
    init.set_headers(&headers);
    let promise = fetch
        .call2(&global, &url, &init)
        .map_err(network)?
        .dyn_into::<js_sys::Promise>()
        .map_err(network)?;
    let response: web_sys::Response = JsFuture::from(promise)
        .await
        .map_err(network)?
        .dyn_into()
        .map_err(network)?;
    let stream = response.body();
    let valid = match response.status() {
        404 => Err(FetchError::NotFound),
        // No content: a valid, empty tile.
        204 => {
            if let Some(stream) = stream {
                let _ = stream.cancel();
            }
            return Ok(Vec::new());
        }
        200..=299 => {
            if let Some((offset, length)) = range {
                let header = response.headers().get("content-range").map_err(network)?;
                orb_host::check_range(response.status(), header.as_deref(), offset, length)
            } else {
                Ok(())
            }
        }
        status => Err(FetchError::Status(status)),
    };
    if let Err(error) = valid {
        if let Some(stream) = stream {
            let _ = stream.cancel();
        }
        return Err(error);
    }
    let reader: web_sys::ReadableStreamDefaultReader = stream
        .ok_or_else(|| FetchError::Network("missing response body".into()))?
        .get_reader()
        .dyn_into()
        .map_err(|e: js_sys::Object| network(e.into()))?;
    let limit = range.map_or(64 << 20, |(_, length)| length) as usize;
    let mut bytes = Vec::new();
    loop {
        let chunk = JsFuture::from(reader.read()).await.map_err(network)?;
        if Reflect::get(&chunk, &"done".into())
            .map_err(network)?
            .as_bool()
            == Some(true)
        {
            break;
        }
        let chunk = Uint8Array::new(&Reflect::get(&chunk, &"value".into()).map_err(network)?);
        if bytes.len() + chunk.length() as usize > limit {
            let _ = reader.cancel();
            return Err(FetchError::Range("oversized response body".into()));
        }
        bytes.extend_from_slice(&chunk.to_vec());
    }
    if range.is_some() && bytes.len() != limit {
        return Err(FetchError::Range("short range body".into()));
    }
    Ok(bytes)
}

/// The browser's monotonic clock, available on the page and workers.
#[derive(Clone, Copy, Debug, Default)]
pub struct WebClock;

impl orb_host::Clock for WebClock {
    fn now(&self) -> f64 {
        let performance: web_sys::Performance =
            Reflect::get(&js_sys::global(), &"performance".into())
                .expect("performance clock")
                .unchecked_into();
        performance.now()
    }
}

/// Local glyphs drawn on the browser's canvas exactly as GL JS's TinySDF
/// 2.3.0 draws them: a 48 px font, measured with `measureText` and filled
/// at the ink box's pen offset inside a six-texel border, then read back.
#[derive(Clone, Copy, Debug, Default)]
pub struct CanvasGlyphs;

/// TinySDF's `buffer` once GL JS sets it to the atlas padding.
const CANVAS_BUFFER: i32 = 6;

thread_local! {
    /// One canvas, large enough for a cluster's three ems, and the font it
    /// is set to.
    static CANVAS: std::cell::RefCell<Option<(web_sys::OffscreenCanvasRenderingContext2d, String)>> =
        const { std::cell::RefCell::new(None) };
}

fn with_canvas<R>(
    font: &orb_host::LocalFace<'_>,
    f: impl FnOnce(&web_sys::OffscreenCanvasRenderingContext2d) -> R,
) -> Option<R> {
    let css = format!(
        "{} {} 48px {}",
        font.style,
        font.weight.unwrap_or("normal"),
        font.family
    );
    CANVAS.with(|canvas| {
        let mut canvas = canvas.borrow_mut();
        if canvas.is_none() {
            let element = web_sys::OffscreenCanvas::new(144, 144).ok()?;
            let options = js_sys::Object::new();
            Reflect::set(&options, &"willReadFrequently".into(), &true.into()).ok()?;
            let context: web_sys::OffscreenCanvasRenderingContext2d = element
                .get_context_with_context_options("2d", &options)
                .ok()??
                .dyn_into()
                .ok()?;
            context.set_text_baseline("alphabetic");
            context.set_text_align("left");
            context.set_fill_style_str("black");
            *canvas = Some((context, String::new()));
        }
        let (context, set) = canvas.as_mut()?;
        if *set != css {
            context.set_font(&css);
            *set = css;
        }
        Some(f(context))
    })
}

impl orb_host::LocalGlyphs for CanvasGlyphs {
    fn measure(&self, font: &orb_host::LocalFace<'_>, text: &str) -> orb_host::TextMetrics {
        with_canvas(font, |context| {
            context
                .measure_text(text)
                .map(|m| orb_host::TextMetrics {
                    advance: m.width(),
                    ascent: m.actual_bounding_box_ascent(),
                    descent: m.actual_bounding_box_descent(),
                    left: m.actual_bounding_box_left(),
                    right: m.actual_bounding_box_right(),
                })
                .unwrap_or_default()
        })
        .unwrap_or_default()
    }

    fn coverage(
        &self,
        font: &orb_host::LocalFace<'_>,
        text: &str,
        ink: orb_host::InkRect,
    ) -> Vec<u8> {
        let (w, h) = (ink.width, ink.height);
        if w == 0 || h == 0 {
            return Vec::new();
        }
        with_canvas(font, |context| {
            let b = f64::from(CANVAS_BUFFER);
            context.clear_rect(b, b, f64::from(w), f64::from(h));
            context
                .fill_text(text, b - f64::from(ink.left), b + f64::from(ink.top))
                .ok()?;
            let image = context
                .get_image_data(b, b, f64::from(w), f64::from(h))
                .ok()?;
            Some(
                image
                    .data()
                    .0
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| p[3])
                    .collect(),
            )
        })
        .flatten()
        .unwrap_or_else(|| vec![0; (w * h) as usize])
    }
}

/// `font-faces` files drawn by `shaper.wasm`, which the job
/// worker loads the first time a style declares a file: HarfRust and
/// skrifa, as natively, outside `core.wasm`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaperFiles;

thread_local! {
    /// The job worker's loader (a JS function returning a promise of a
    /// `Shaper`), the shaper once loaded, and the files it has open.
    static SHAPER_LOADER: std::cell::RefCell<Option<js_sys::Function>> =
        const { std::cell::RefCell::new(None) };
    static SHAPER: std::cell::RefCell<Option<JsValue>> = const { std::cell::RefCell::new(None) };
    /// The load in flight, which concurrent jobs share: a second shaper
    /// would have none of the first's files open.
    static SHAPER_LOAD: std::cell::RefCell<Option<js_sys::Promise>> =
        const { std::cell::RefCell::new(None) };
    static SHAPER_OPEN: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

impl ShaperFiles {
    /// Sets how this worker loads `shaper.wasm`: `loader()` returns a
    /// promise of a `Shaper`.
    pub fn set_loader(loader: js_sys::Function) {
        SHAPER_LOADER.with(|l| *l.borrow_mut() = Some(loader));
    }

    /// Loads the shaper unless this worker has it.
    pub async fn load() -> Result<(), String> {
        if SHAPER.with(|s| s.borrow().is_some()) {
            return Ok(());
        }
        let promise = match SHAPER_LOAD.with(|l| l.borrow().clone()) {
            Some(promise) => promise,
            None => {
                let loader = SHAPER_LOADER
                    .with(|l| l.borrow().clone())
                    .ok_or("this worker has no shaper.wasm")?;
                let promise = loader
                    .call0(&JsValue::NULL)
                    .ok()
                    .and_then(|p| p.dyn_into::<js_sys::Promise>().ok())
                    .ok_or("the shaper loader returned no promise")?;
                SHAPER_LOAD.with(|l| *l.borrow_mut() = Some(promise.clone()));
                promise
            }
        };
        let shaper = JsFuture::from(promise).await;
        SHAPER_LOAD.with(|l| *l.borrow_mut() = None);
        let shaper = shaper.map_err(|e| format!("shaper.wasm: {e:?}"))?;
        SHAPER.with(|s| {
            s.borrow_mut().get_or_insert(shaper);
        });
        Ok(())
    }

    /// Whether this worker has `url` open.
    pub fn is_open(url: &str) -> bool {
        SHAPER_OPEN.with(|o| o.borrow().contains(url))
    }

    fn call(method: &str, args: &[JsValue]) -> Result<JsValue, String> {
        SHAPER.with(|s| {
            let shaper = s.borrow();
            let shaper = shaper.as_ref().ok_or("shaper.wasm is not loaded")?;
            let f = Reflect::get(shaper, &method.into())
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| format!("Shaper.{method} is missing"))?;
            let args: js_sys::Array = args.iter().collect();
            f.apply(shaper, &args).map_err(|e| {
                e.dyn_ref::<js_sys::Error>()
                    .map_or_else(|| format!("{e:?}"), |e| String::from(e.message()))
            })
        })
    }
}

impl orb_host::FontFiles for ShaperFiles {
    fn prepare(&self, ready: Box<dyn FnOnce(Result<(), String>)>) {
        wasm_bindgen_futures::spawn_local(async move { ready(Self::load().await) });
    }

    fn is_open(&self, url: &str) -> bool {
        Self::is_open(url)
    }

    fn open(&self, url: &str, bytes: &[u8]) -> Result<(), String> {
        Self::call("open", &[url.into(), Uint8Array::from(bytes).into()])?;
        SHAPER_OPEN.with(|o| o.borrow_mut().insert(url.to_string()));
        Ok(())
    }

    fn measure(&self, url: &str, text: &str) -> orb_host::TextMetrics {
        let Ok(m) = Self::call("measure", &[url.into(), text.into()]) else {
            return orb_host::TextMetrics::default();
        };
        let m = js_sys::Float64Array::new(&m).to_vec();
        match m[..] {
            [advance, ascent, descent, left, right] => orb_host::TextMetrics {
                advance,
                ascent,
                descent,
                left,
                right,
            },
            _ => orb_host::TextMetrics::default(),
        }
    }

    fn coverage(&self, url: &str, text: &str, ink: orb_host::InkRect) -> Vec<u8> {
        let args = [
            url.into(),
            text.into(),
            ink.left.into(),
            ink.top.into(),
            ink.width.into(),
            ink.height.into(),
        ];
        match Self::call("coverage", &args) {
            Ok(alpha) => Uint8Array::new(&alpha).to_vec(),
            Err(_) => vec![0; (ink.width * ink.height) as usize],
        }
    }
}
