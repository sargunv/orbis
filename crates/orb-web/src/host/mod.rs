//! Host services in the browser, for the main thread and workers alike.

#![forbid(unsafe_code)]

mod locale;
mod storage;

pub(crate) use locale::BrowserLocale;

pub(crate) use storage::OpfsStorage;

use js_sys::{Reflect, Uint8Array};
use orb_host::{Abort, DecodedImage, Fetch, FetchDone, FetchError, ImageDecode, ImageDone};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WebFetch;

/// Decodes images with the browser's decoder: `createImageBitmap` without
/// premultiplication or colour conversion, read back through a
/// `VideoFrame` or a canvas ([`read_bitmap`]), on the page or in a worker.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WebImageDecode;

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

thread_local! {
    /// The browser's word and grapheme segmenters, each made once per
    /// context when first asked for: making one takes 4–6 ms, which a
    /// worker's first tiles would wait for.
    static WORDS: Option<JsValue> = segmenter("word");
    static GRAPHEMES: Option<JsValue> = segmenter("grapheme");
}

/// `new Intl.Segmenter(undefined, {granularity})`, as GL JS makes its
/// segmenters, where the browser has one.
fn segmenter(granularity: &str) -> Option<JsValue> {
    let class = segmenter_class()?;
    let options = js_sys::Object::new();
    Reflect::set(&options, &"granularity".into(), &granularity.into()).ok()?;
    let args = js_sys::Array::of2(&JsValue::UNDEFINED, &options);
    Reflect::construct(&class, &args).ok()
}

/// `Intl.Segmenter`, where the browser has one.
fn segmenter_class() -> Option<js_sys::Function> {
    let intl = Reflect::get(&js_sys::global(), &"Intl".into()).ok()?;
    Reflect::get(&intl, &"Segmenter".into())
        .ok()?
        .dyn_into::<js_sys::Function>()
        .ok()
}

/// Words in scripts written without spaces, from the browser's
/// `Intl.Segmenter` (as GL JS takes them), where it has one.
pub(crate) fn browser_words() -> Option<orb_text::WordBreaker> {
    segmenter_class().is_some().then(|| {
        orb_text::WordBreaker::from_host(Box::new(|text: &str| {
            WORDS.with(|segmenter| segment_starts(segmenter.as_ref(), text))
        }))
    })
}

/// Grapheme clusters from the browser's `Intl.Segmenter`, where it has
/// one, as GL JS takes them: core.wasm leaves ICU4X's rules out.
pub(crate) fn browser_graphemes() -> Option<Box<orb_text::HostGraphemes>> {
    segmenter_class()
        .is_some()
        .then(|| -> Box<orb_text::HostGraphemes> {
            Box::new(|text: &str| {
                let mut ends = GRAPHEMES.with(|segmenter| segment_starts(segmenter.as_ref(), text));
                ends.retain(|&end| end > 0);
                ends
            })
        })
}

/// The byte offsets at which `segmenter`'s segments of `text` start, and
/// the text's end.
fn segment_starts(segmenter: Option<&JsValue>, text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    // The segments' UTF-16 indices, in order.
    let mut indices = Vec::new();
    if let Some(segmenter) = segmenter
        && let Ok(segment) = Reflect::get(segmenter, &"segment".into())
        && let Ok(segment) = segment.dyn_into::<js_sys::Function>()
        && let Ok(segments) = segment.call1(segmenter, &text.into())
        && let Ok(Some(iter)) = js_sys::try_iter(&segments)
    {
        for item in iter.flatten() {
            if let Some(index) = Reflect::get(&item, &"index".into())
                .ok()
                .and_then(|i| i.as_f64())
            {
                indices.push(index as usize);
            }
        }
    }
    let mut indices = indices.into_iter().peekable();
    let mut utf16 = 0;
    for (byte, c) in text.char_indices() {
        while indices.next_if(|&i| i <= utf16).is_some_and(|i| i == utf16) {
            starts.push(byte);
        }
        utf16 += c.len_utf16();
    }
    starts.push(text.len());
    starts.dedup();
    starts
}

/// Decodes an image to straight RGBA8 texels.
pub(crate) async fn decode_image(bytes: &[u8]) -> Result<DecodedImage, JsValue> {
    let bitmap = decode_bitmap(bytes, false).await?;
    let image = read_bitmap(&bitmap).await;
    bitmap.close();
    image
}

/// Reads a bitmap's straight RGBA8 texels: through a `VideoFrame` where
/// this context reads one exactly, and otherwise through a canvas.
///
/// A canvas's readback isn't exact. It keeps texels premultiplied, so
/// translucent ones lose precision, and browsers that resist
/// fingerprinting (Brave) perturb what `getImageData` returns, which turns
/// a DEM tile's texels into spikes. GL JS reads through a `VideoFrame` once
/// a probe finds the canvas distorted (its #3185); a frame is exact and,
/// in Chrome, faster than the canvas (0.33 against 0.47 ms for a 512 px
/// tile), so here it's the first choice wherever it reads exactly.
async fn read_bitmap(bitmap: &web_sys::ImageBitmap) -> Result<DecodedImage, JsValue> {
    if let Ok(pixels) = JsFuture::from(frames::read_frame(bitmap)).await
        && let Some(pixels) = pixels.dyn_ref::<Uint8Array>()
    {
        return Ok(DecodedImage {
            width: bitmap.width(),
            height: bitmap.height(),
            pixels: pixels.to_vec(),
        });
    }
    read_canvas(bitmap)
}

mod frames {
    use wasm_bindgen::prelude::wasm_bindgen;

    // `readFrame(bitmap)` resolves to the bitmap's RGBA8 texels, read
    // through a `VideoFrame`, or to nothing where this context's frames
    // don't read exactly: the first call finds out, decoding a 4 × 4 PNG
    // whose texel bytes count up by 37 from 11 (alphas among them a
    // canvas loses precision at) and reading it back. A frame is copied
    // in its own format (RGBA or BGRA, X for opaque), as GL JS copies it:
    // copying with `format: "RGBA"` converts, and isn't exact.
    #[wasm_bindgen(inline_js = r#"
const PROBE = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAT0lEQVR42gFEALv/AAswVXqfxOkOM1h9osfsETYAW4Clyu8UOV6DqM3yFzxhhgCr0PUaP2SJrtP4HUJnjLHWAPsgRWqPtNn+I0htkrfcASYmMCAhKZAw6QAAAABJRU5ErkJggg==";
let exact;
async function read(bitmap) {
  const frame = new VideoFrame(bitmap, { timestamp: 0 });
  try {
    const { width, height } = bitmap;
    const format = frame.format ?? "";
    const bgr = format.startsWith("BGR");
    if (!bgr && !format.startsWith("RGB")) throw new Error(`frame format ${format}`);
    const data = new Uint8Array(width * height * 4);
    await frame.copyTo(data, { layout: [{ offset: 0, stride: width * 4 }] });
    const opaque = format.endsWith("X");
    for (let i = 0; i < data.length; i += 4) {
      if (bgr) [data[i], data[i + 2]] = [data[i + 2], data[i]];
      if (opaque) data[i + 3] = 255;
    }
    return data;
  } finally {
    frame.close();
  }
}
export async function readFrame(bitmap) {
  exact ??= createImageBitmap(new Blob([Uint8Array.from(atob(PROBE), (c) => c.charCodeAt(0))]),
    { premultiplyAlpha: "none", colorSpaceConversion: "none" })
    .then((probe) => read(probe).finally(() => probe.close()))
    .then((data) => data.every((v, i) => v === (11 + 37 * i) % 256), () => false);
  return (await exact) ? read(bitmap).catch(() => undefined) : undefined;
}
"#)]
    extern "C" {
        #[wasm_bindgen(js_name = readFrame)]
        pub(super) fn read_frame(bitmap: &web_sys::ImageBitmap) -> js_sys::Promise;
    }
}

/// Reads a bitmap's straight RGBA8 texels through an `OffscreenCanvas`
/// that reads back often: the CPU draws it, without the GPU context (about
/// 10 ms to create on a page) and readback a GPU canvas costs.
fn read_canvas(bitmap: &web_sys::ImageBitmap) -> Result<DecodedImage, JsValue> {
    let (width, height) = (bitmap.width(), bitmap.height());
    let canvas = web_sys::OffscreenCanvas::new(width, height)?;
    let options = js_sys::Object::new();
    Reflect::set(&options, &"willReadFrequently".into(), &true.into())?;
    let context: web_sys::OffscreenCanvasRenderingContext2d = canvas
        .get_context_with_context_options("2d", &options)?
        .ok_or_else(|| JsValue::from_str("no 2d context"))?
        .dyn_into()?;
    context.draw_image_with_image_bitmap(bitmap, 0.0, 0.0)?;
    let data = context.get_image_data(0.0, 0.0, f64::from(width), f64::from(height))?;
    Ok(DecodedImage {
        width,
        height,
        pixels: data.data().0,
    })
}

/// Decodes an image into an `ImageBitmap` with no colour conversion, on the
/// page or in a worker, its alpha straight. With `texture`, its pixels stay
/// with the browser: a job worker transfers the bitmap to the map's thread,
/// which copies it to a texture, premultiplying it; WebGL 2 copies an
/// `ImageBitmap` as it is, so there such a bitmap comes premultiplied (only
/// raster tiles arrive as bitmaps).
pub(crate) async fn decode_bitmap(
    bytes: &[u8],
    texture: bool,
) -> Result<web_sys::ImageBitmap, JsValue> {
    let parts = js_sys::Array::of1(&Uint8Array::from(bytes));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)?;
    let options = js_sys::Object::new();
    let alpha = if texture && crate::canvas::GL {
        "premultiply"
    } else {
        "none"
    };
    Reflect::set(&options, &"premultiplyAlpha".into(), &alpha.into())?;
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
    fn get(&self, _: orb_host::ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        start(url, None, abort, done);
    }

    fn get_range(
        &self,
        _: orb_host::ResourceKind,
        url: &str,
        offset: u64,
        length: u64,
        abort: &Abort,
        done: FetchDone,
    ) {
        start(url, Some((offset, length)), abort, done);
    }
}

/// Fetches `url` (its `range`) until it ends or `abort` fires, which
/// aborts the browser's request through an `AbortController`.
fn start(url: &str, range: Option<(u64, u64)>, abort: &Abort, done: FetchDone) {
    let done = abort.answer(done);
    if abort.is_aborted() {
        return;
    }
    let (url, abort) = (url.to_string(), abort.clone());
    wasm_bindgen_futures::spawn_local(async move {
        use std::task::Poll;
        let controller = web_sys::AbortController::new().ok();
        let signal = controller.as_ref().map(web_sys::AbortController::signal);
        let mut fetched = std::pin::pin!(fetch_body(&url, range, signal.as_ref()));
        let mut aborted = std::pin::pin!(abort.aborted());
        let first = std::future::poll_fn(|cx| match fetched.as_mut().poll(cx) {
            Poll::Ready(result) => Poll::Ready(Some(result)),
            Poll::Pending => aborted.as_mut().poll(cx).map(|()| None),
        });
        match first.await {
            Some(result) => done(result),
            // `done` has its answer.
            None => {
                if let Some(controller) = controller {
                    controller.abort();
                }
            }
        }
    });
}

thread_local! {
    static REQUEST_HOOK: std::cell::RefCell<Option<js_sys::Function>> = const {
        std::cell::RefCell::new(None)
    };
    static PRELOADS: std::cell::RefCell<Option<js_sys::Function>> = const {
        std::cell::RefCell::new(None)
    };
}

/// What this JS context fetched ahead of its maps (`web/src/preload.ts`):
/// `take(url)` answers a promise of the response to a whole-body request
/// for `url`, once, or `undefined`.
pub(crate) fn set_preloads(take: Option<js_sys::Function>) {
    PRELOADS.with(|p| *p.borrow_mut() = take);
}

/// The response the page fetched ahead for `url`, if it did and it came.
async fn preloaded(url: &str) -> Option<web_sys::Response> {
    let take = PRELOADS.with(|p| p.borrow().clone())?;
    let promise = take
        .call1(&JsValue::NULL, &url.into())
        .ok()?
        .dyn_into::<js_sys::Promise>()
        .ok()?;
    JsFuture::from(promise).await.ok()?.dyn_into().ok()
}

/// Transforms every request this JS context's fetches make (GL JS
/// `transformRequest`): `hook(url)` answers `{url, headers}` (or a promise
/// of it), `null` to refuse, or `undefined` to leave the request.
pub(crate) fn set_request_hook(hook: Option<js_sys::Function>) {
    REQUEST_HOOK.with(|h| *h.borrow_mut() = hook);
}

async fn fetch_body(
    url: &str,
    range: Option<(u64, u64)>,
    signal: Option<&web_sys::AbortSignal>,
) -> Result<Vec<u8>, FetchError> {
    let early = match range {
        None => preloaded(url).await,
        Some(_) => None,
    };
    let response = match early {
        Some(response) => response,
        None => request(url, range, signal).await?,
    };
    read(response, range).await
}

/// Fetches `url` (its `range`), through the request hook, until `signal`
/// aborts it.
async fn request(
    url: &str,
    range: Option<(u64, u64)>,
    signal: Option<&web_sys::AbortSignal>,
) -> Result<web_sys::Response, FetchError> {
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
    init.set_signal(signal);
    let promise = fetch
        .call2(&global, &url, &init)
        .map_err(network)?
        .dyn_into::<js_sys::Promise>()
        .map_err(network)?;
    JsFuture::from(promise)
        .await
        .map_err(network)?
        .dyn_into()
        .map_err(network)
}

/// A response's body, checked against its status and `range`.
async fn read(
    response: web_sys::Response,
    range: Option<(u64, u64)>,
) -> Result<Vec<u8>, FetchError> {
    let network = |e: JsValue| FetchError::Network(format!("{e:?}"));
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
    // A range's length is known; a whole body's is as its header says,
    // which is the encoded length, a lower bound for compressed bodies.
    let expected = match range {
        Some(_) => limit,
        None => (response.headers().get("content-length").ok().flatten())
            .and_then(|length| length.parse().ok())
            .unwrap_or(0),
    };
    let mut bytes = Vec::with_capacity(expected.min(limit));
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
        // Straight from the JS chunk into the body: one copy.
        let start = bytes.len();
        bytes.resize(start + chunk.length() as usize, 0);
        chunk.copy_to(&mut bytes[start..]);
    }
    if range.is_some() && bytes.len() != limit {
        return Err(FetchError::Range("short range body".into()));
    }
    Ok(bytes)
}

/// The browser's monotonic clock, available on the page and workers.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WebClock;

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
/// The readback is the canvas's, not a `VideoFrame`'s ([`read_bitmap`]):
/// coverage is synchronous, and the small changes a fingerprinting
/// browser makes to an alpha barely move a glyph's distance field.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CanvasGlyphs;

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
        ink: &orb_host::InkBox,
    ) -> Vec<u8> {
        let (w, h) = (ink.glyph_width, ink.glyph_height);
        if w == 0 || h == 0 {
            return Vec::new();
        }
        with_canvas(font, |context| {
            let b = f64::from(CANVAS_BUFFER);
            context.clear_rect(b, b, f64::from(w), f64::from(h));
            context
                .fill_text(
                    text,
                    b - f64::from(ink.glyph_left),
                    b + f64::from(ink.glyph_top),
                )
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
pub(crate) struct ShaperFiles;

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
    static SHAPER_OPEN: std::cell::RefCell<rustc_hash::FxHashSet<String>> =
        std::cell::RefCell::new(rustc_hash::FxHashSet::default());
}

impl ShaperFiles {
    /// Sets how this worker loads `shaper.wasm`: `loader()` returns a
    /// promise of a `Shaper`.
    pub(crate) fn set_loader(loader: js_sys::Function) {
        SHAPER_LOADER.with(|l| *l.borrow_mut() = Some(loader));
    }

    /// Loads the shaper unless this worker has it.
    pub(crate) async fn load() -> Result<(), String> {
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
    pub(crate) fn is_open(url: &str) -> bool {
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

    fn coverage(&self, url: &str, text: &str, ink: &orb_host::InkBox) -> Vec<u8> {
        let args = [
            url.into(),
            text.into(),
            ink.glyph_left.into(),
            ink.glyph_top.into(),
            ink.glyph_width.into(),
            ink.glyph_height.into(),
        ];
        match Self::call("coverage", &args) {
            Ok(alpha) => Uint8Array::new(&alpha).to_vec(),
            Err(_) => vec![0; (ink.glyph_width * ink.glyph_height) as usize],
        }
    }
}
