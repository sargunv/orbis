//! Tile jobs across workers (plan §7.1, §7.3, §10.4): both ends of the
//! protocol between the page and its job workers.
//!
//! The page's JS owns the transport, one `MessagePort` per job worker, and
//! passes messages through unread. The page's [`WorkerJobs`] posts each
//! request as a [`JobRequest`] in [`orb_runtime::wire`]'s bytes; a job
//! worker's [`JobWorker`] answers each one with an id as
//! `{message, buffer?, image?}`: a [`JobReply`], with a packed tile ([`pack`])
//! or a DEM tile's pixels as the transferred `buffer` and an image tile as
//! the transferred `image` bitmap. The page copies only a packed tile's
//! header into wasm memory; the payload goes from the buffer to the GPU.
//! A tile, its relayouts and its queries go to job worker
//! `hash(source, tile) mod N`, which has its data; a glyph unit to its
//! owner worker.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use js_sys::{ArrayBuffer, Object, Reflect};
use orb_crs::TileIndex;
use orb_host::FetchError;
use orb_runtime::wire;
use orb_runtime::{
    Built, GlyphRaster, ImagePixels, JobReply, JobRequest, Jobs, RasterDone, SpriteIcons,
    StyleUpdate, TileBuild, TileDone, TileImage, TileLoader, TileRequest, TileResult, TileWorker,
    pack, packed_header, unpack,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Each map's `self.post(worker, message)`, which sends a message to its job
    /// worker `worker`, by map. Wasm in a page is single-threaded, so the
    /// runtime's `Send` jobs reach it here.
    static POSTS: std::cell::RefCell<HashMap<u32, js_sys::Function>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Forgets map `map`'s job workers' `post`.
pub(crate) fn forget(map: u32) {
    POSTS.with(|posts| posts.borrow_mut().remove(&map));
}

/// [`Jobs`] for the render worker: each tile goes to job worker
/// `hash(source, tile) mod N`, so a tile's relayouts and queries find its
/// data where it was decoded.
pub(crate) struct WorkerJobs {
    /// The map whose job workers these are.
    map: u32,
    workers: u32,
    state: Mutex<State>,
}

impl std::fmt::Debug for WorkerJobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerJobs")
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct State {
    next_id: u32,
    pending: HashMap<u32, TileDone>,
    rasters: HashMap<u32, RasterDone>,
    features: HashMap<u32, orb_runtime::FeaturesDone>,
    /// Bytes of packed headers copied into this worker's wasm memory.
    header_bytes: u64,
    /// Replies received.
    done: u32,
}

/// What the render worker's jobs have done so far.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JobStats {
    pub header_bytes: u64,
    pub done: u32,
}

impl WorkerJobs {
    /// Map `map`'s jobs on `workers` job workers, reached through `post`.
    pub(crate) fn new(map: u32, workers: u32, post: js_sys::Function) -> Arc<Self> {
        POSTS.with(|posts| posts.borrow_mut().insert(map, post));
        Arc::new(Self {
            map,
            workers: workers.max(1),
            state: Mutex::new(State::default()),
        })
    }

    fn post(&self, worker: u32, message: &JsValue) {
        POSTS.with(|posts| {
            if let Some(f) = posts.borrow().get(&self.map) {
                let _ = f.call2(&JsValue::NULL, &worker.into(), message);
            }
        });
    }

    pub(crate) fn stats(&self) -> JobStats {
        let state = self.state.lock().expect("jobs poisoned");
        JobStats {
            header_bytes: state.header_bytes,
            done: state.done,
        }
    }

    /// Completes a request with a job worker's reply. A packed tile's
    /// buffer is registered with `gpu`; only its header enters wasm memory.
    pub(crate) fn receive(
        &self,
        gpu: &mut dyn crate::canvas::WebGpu,
        reply: &JsValue,
    ) -> Result<(), JsError> {
        let message = field(reply, "message")
            .dyn_into::<js_sys::Uint8Array>()
            .map_err(|_| JsError::new("job reply without a message"))?;
        let message: JobReply = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let buffer = field(reply, "buffer").dyn_into::<ArrayBuffer>().ok();
        let id = message.id();
        let mut state = self.state.lock().expect("jobs poisoned");
        if let Some(done) = state.features.remove(&id) {
            drop(state);
            done(match message {
                JobReply::Features { features, .. } => Ok(features),
                JobReply::Error { message, .. } => Err(message),
                _ => Err("malformed features reply".into()),
            });
            return Ok(());
        }
        if let Some(done) = state.rasters.remove(&id) {
            drop(state);
            done(match message {
                JobReply::Glyphs { bitmaps, .. } => Ok(bitmaps),
                JobReply::Error { message, .. } => Err(message),
                _ => Err("malformed raster reply".into()),
            });
            return Ok(());
        }
        state.done += 1;
        let done = state
            .pending
            .remove(&id)
            .ok_or_else(|| JsError::new(&format!("reply to unknown job {id}")))?;
        let result = match message {
            JobReply::Missing { .. } => Ok(TileResult {
                missing: true,
                ..TileResult::default()
            }),
            // A DEM tile, decoded by the core in the job worker.
            JobReply::Dem {
                width, elevations, ..
            } => {
                let pixels = buffer.map(|b| js_sys::Uint8Array::new(&b).to_vec());
                orb_tile::DemGrid::from_elevations(width, elevations).map(|dem| TileResult {
                    image: Some(TileImage {
                        width,
                        height: width,
                        pixels: ImagePixels::Core(pixels.unwrap_or_default()),
                    }),
                    dem: Some(Arc::new(dem)),
                    ..TileResult::default()
                })
            }
            JobReply::Image { .. } => {
                match field(reply, "image").dyn_into::<web_sys::ImageBitmap>() {
                    Ok(bitmap) => Ok(TileResult {
                        image: Some(TileImage {
                            width: bitmap.width(),
                            height: bitmap.height(),
                            pixels: ImagePixels::Host(gpu.register_image_bitmap(bitmap)),
                        }),
                        ..TileResult::default()
                    }),
                    Err(_) => Err("image reply without a bitmap".into()),
                }
            }
            JobReply::Tile { .. } => match buffer {
                Some(buffer) => {
                    let header = header_bytes(&buffer);
                    if let Ok(header) = &header {
                        state.header_bytes += header.len() as u64;
                    }
                    header.and_then(|header| {
                        let bytes = gpu.register_array_buffer(buffer);
                        let result = unpack(&header, bytes);
                        if result.is_err() {
                            gpu.release_host_bytes(bytes);
                        }
                        result
                    })
                }
                None => Err("tile reply without a buffer".into()),
            },
            JobReply::Error { message, .. } => Err(message),
            _ => Err("malformed job reply".into()),
        };
        drop(state);
        done(result);
        Ok(())
    }
}

/// The job worker that loaded `tile` of `source`: `hash(source, tile) mod N`.
fn owner(source: u32, tile: TileIndex, workers: u32) -> u32 {
    let mut hasher = DefaultHasher::new();
    (source, tile).hash(&mut hasher);
    (hasher.finish() % u64::from(workers)) as u32
}

/// Copies a packed tile's header out of `buffer`: its length, then its
/// bytes ([`packed_header`]).
fn header_bytes(buffer: &ArrayBuffer) -> Result<Vec<u8>, String> {
    let bad = || "malformed packed tile".to_string();
    if buffer.byte_length() < 4 {
        return Err(bad());
    }
    let len = js_sys::Uint8Array::new_with_byte_offset_and_length(buffer, 0, 4).to_vec();
    let len = u32::from_le_bytes(len.try_into().expect("four bytes"));
    if 4 + len > buffer.byte_length() {
        return Err(bad());
    }
    let prefix = js_sys::Uint8Array::new_with_byte_offset_and_length(buffer, 0, 4 + len).to_vec();
    Ok(packed_header(&prefix)?.to_vec())
}

impl WorkerJobs {
    /// A request's id, its done kept by `keep`.
    fn next_id(&self, keep: impl FnOnce(&mut State, u32)) -> u32 {
        let mut state = self.state.lock().expect("jobs poisoned");
        let id = state.next_id;
        state.next_id = id.wrapping_add(1);
        keep(&mut state, id);
        id
    }
}

impl Jobs for WorkerJobs {
    fn set_style(&self, style: u64, update: &StyleUpdate, _: &Arc<TileWorker>) {
        let message = bytes(&wire::encode(&JobRequest::Style {
            style,
            update: Box::new(update.clone()),
        }));
        for worker in 0..self.workers {
            self.post(worker, &message);
        }
    }

    fn set_icons(&self, style: u64, icons: &Arc<SpriteIcons>, _: &Arc<TileWorker>) {
        let message = bytes(&wire::encode(&JobRequest::Icons {
            style,
            icons: (**icons).clone(),
        }));
        for worker in 0..self.workers {
            self.post(worker, &message);
        }
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        let id = self.next_id(|state, id| {
            state.rasters.insert(id, done);
        });
        // A unit's owner worker rasterizes it (plan §7.3 affinity).
        let worker = request.owner(self.workers);
        self.post(
            worker,
            &bytes(&wire::encode(&JobRequest::Raster { id, request })),
        );
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let id = self.next_id(|state, id| {
            state.pending.insert(id, done);
        });
        let worker = owner(request.source, request.tile, self.workers);
        self.post(
            worker,
            &bytes(&wire::encode(&JobRequest::Load { id, request })),
        );
    }

    fn features(&self, request: orb_runtime::FeatureRequest, done: orb_runtime::FeaturesDone) {
        let id = self.next_id(|state, id| {
            state.features.insert(id, done);
        });
        let worker = owner(request.tile.source, request.tile.tile, self.workers);
        self.post(
            worker,
            &bytes(&wire::encode(&JobRequest::Features { id, request })),
        );
    }
}

/// A job worker: fetches and lays out tiles for the render worker.
#[wasm_bindgen]
#[derive(Debug)]
pub struct JobWorker {
    /// The latest revision of the current style and of the one before it,
    /// whose requests may still be in flight.
    styles: Vec<(u64, Rc<TileWorker>)>,
    /// Concurrent tiles share the source document fetch. Once indexed,
    /// `has_data` bypasses fetch entirely; each worker owns its own index.
    source_fetches: Rc<RefCell<HashMap<(u64, u32), js_sys::Promise>>>,
    /// Glyph range and word data fetches in flight, by style and URL.
    glyph_fetches: Rc<RefCell<HashMap<(u64, String), js_sys::Promise>>>,
    /// Where the Southeast Asian word data is, and the data once a style's
    /// worker has loaded it, for the others.
    word_data_url: Option<String>,
    words: Rc<RefCell<Option<Arc<orb_text::WordBreaker>>>>,

    loader: Arc<TileLoader>,
}

#[wasm_bindgen]
impl JobWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            styles: Vec::new(),
            source_fetches: Rc::default(),
            glyph_fetches: Rc::default(),
            word_data_url: None,
            words: Rc::default(),
            loader: TileLoader::new(crate::regions::service()),
        }
    }

    /// Fetches the Southeast Asian word data from `url` when labels first
    /// need it, rather than compiling it into `core.wasm` (decision 0009).
    #[wasm_bindgen(js_name = setWordDataUrl)]
    pub fn set_word_data_url(&mut self, url: String) {
        self.word_data_url = Some(url);
    }

    /// How this worker loads `shaper.wasm` for `font-faces` files:
    /// `loader()` returns a promise of a `Shaper` (decision 0009).
    #[wasm_bindgen(js_name = setShaperLoader)]
    pub fn set_shaper_loader(&mut self, loader: js_sys::Function) {
        orb_host_web::ShaperFiles::set_loader(loader);
    }

    /// Handles a request from the page: a [`JobRequest`] in
    /// [`orb_runtime::wire`]'s bytes. Returns a promise of the reply to
    /// post back, transferring its `buffer` and `image`, or nothing.
    pub fn receive(
        &mut self,
        message: &js_sys::Uint8Array,
    ) -> Result<Option<js_sys::Promise>, JsError> {
        let request: JobRequest = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let worker = |styles: &[(u64, Rc<TileWorker>)], style: u64| {
            styles
                .iter()
                .find(|(id, _)| *id == style)
                .map(|(_, w)| Rc::clone(w))
        };
        match request {
            JobRequest::Style { style, update } => {
                let previous = worker(&self.styles, style);
                let worker = TileWorker::update(previous.as_deref(), &update)
                    .map_err(|e| JsError::new(&e))?;
                worker.set_local_glyphs(Arc::new(orb_host_web::CanvasGlyphs));
                worker.set_font_files(Arc::new(orb_host_web::ShaperFiles));
                if let Some(words) = self.words.borrow().as_ref() {
                    worker.set_word_breaker(Arc::clone(words));
                } else if let Some(url) = &self.word_data_url {
                    worker.set_word_data_url(url);
                }
                self.styles
                    .retain(|(id, _)| *id + 1 >= style && *id != style);
                self.styles.push((style, Rc::new(worker)));
                Ok(None)
            }
            JobRequest::Icons { style, icons } => {
                // Revisions share their worker data, so this reaches every one.
                if let Some(worker) = worker(&self.styles, style) {
                    worker.set_icons(Arc::new(icons));
                }
                Ok(None)
            }
            JobRequest::Raster { id, request } => Ok(Some(raster(id, request))),
            JobRequest::Features { id, request } => {
                let worker = worker(&self.styles, request.tile.style);
                Ok(Some(query(worker, request, &self.loader, id)))
            }
            JobRequest::Load { id, request } => {
                let worker = worker(&self.styles, request.style);
                Ok(Some(self.load(worker, id, request)))
            }
        }
    }
}

impl JobWorker {
    fn load(
        &self,
        worker: Option<Rc<TileWorker>>,
        id: u32,
        request: TileRequest,
    ) -> js_sys::Promise {
        let (style, source, encoding) = (request.style, request.source, request.encoding);
        let loaded = if worker
            .as_ref()
            .is_some_and(|worker| !worker.has_data(&request) && worker.data_url(source).is_none())
        {
            let (sender, loaded) = futures_channel::oneshot::channel();
            self.loader.load(
                request.location.clone(),
                request.data_tile(),
                encoding,
                Box::new(move |r| {
                    let _ = sender.send(r);
                }),
            );
            Some(loaded)
        } else {
            None
        };
        let source_fetch = worker.as_ref().and_then(|worker| {
            if worker.has_data(&request) {
                return None;
            }
            let url = worker.data_url(source)?.to_string();
            let mut fetches = self.source_fetches.borrow_mut();
            let promise = fetches.entry((style, source)).or_insert_with(|| {
                wasm_bindgen_futures::future_to_promise(async move {
                    match crate::regions::fetch(&url).await {
                        Ok(bytes) => Ok(js_sys::Uint8Array::from(bytes.as_slice()).into()),
                        Err(e) => Err(JsValue::from_str(&e.to_string())),
                    }
                })
            });
            Some(promise.clone())
        });
        let fetches = Rc::clone(&self.source_fetches);
        let glyph_fetches = Rc::clone(&self.glyph_fetches);
        let words = Rc::clone(&self.words);
        let image = worker.as_ref().and_then(|w| w.image_kind(source));
        wasm_bindgen_futures::future_to_promise(async move {
            let error = |message: String| Ok(reply(&JobReply::Error { id, message }, None, None));
            let Some(worker) = worker else {
                return error(format!("style {style} was replaced"));
            };
            // The tile's bytes, unless it needs none.
            let bytes: Vec<u8> = if worker.has_data(&request) {
                Vec::new()
            } else if let Some(source_fetch) = source_fetch {
                let fetched = wasm_bindgen_futures::JsFuture::from(source_fetch).await;
                fetches.borrow_mut().remove(&(style, source));
                match fetched {
                    Ok(_) if worker.has_data(&request) => Vec::new(),
                    Ok(value) => js_sys::Uint8Array::new(&value).to_vec(),
                    Err(e) => {
                        return error(
                            e.as_string()
                                .unwrap_or_else(|| "GeoJSON fetch failed".into()),
                        );
                    }
                }
            } else {
                match loaded
                    .expect("tile load")
                    .await
                    .map_err(|_| JsError::new("tile loader dropped"))?
                {
                    // An archive's left-out tile draws nothing.
                    Ok((bytes, _)) if bytes.is_empty() => {
                        return Ok(tile_reply(id, TileBuild::default()));
                    }
                    // A terrain DEM tile is data: the core decodes it exactly
                    // (`orb_host::decode_image`), for the GPU and the CPU.
                    Ok((bytes, _)) if image.is_some() && request.dem_unpack.is_some() => {
                        let unpack = request.dem_unpack.expect("checked").0;
                        let decoded = orb_host::decode_image(&bytes).and_then(|image| {
                            let grid =
                                orb_tile::DemGrid::from_rgba(image.width, &image.pixels, unpack)?;
                            Ok((image, grid))
                        });
                        return match decoded {
                            Ok((image, grid)) => Ok(reply(
                                &JobReply::Dem {
                                    id,
                                    width: image.width,
                                    elevations: grid.elevations().to_vec(),
                                },
                                Some(js_sys::Uint8Array::from(image.pixels.as_slice()).buffer()),
                                None,
                            )),
                            Err(e) => error(format!("image decode: {e}")),
                        };
                    }
                    Ok((bytes, _)) if image.is_some() => {
                        return match orb_host_web::decode_bitmap(&bytes).await {
                            Ok(bitmap) => {
                                Ok(reply(&JobReply::Image { id }, None, Some(bitmap.into())))
                            }
                            Err(e) => error(format!("image decode: {e:?}")),
                        };
                    }
                    Ok((bytes, _)) => bytes,
                    Err(FetchError::NotFound) => {
                        return Ok(reply(&JobReply::Missing { id }, None, None));
                    }
                    Err(e) => return error(e.to_string()),
                }
            };
            // Labels may need glyph ranges or word data first: fetch them
            // (once per worker) and build again.
            let mut built = worker.build(&bytes, encoding, &request);
            while let Ok(Built::Fetch(urls)) = &built {
                for url in urls {
                    let key = (style, url.clone());
                    let promise = glyph_fetches
                        .borrow_mut()
                        .entry(key.clone())
                        .or_insert_with(|| {
                            let url = url.clone();
                            wasm_bindgen_futures::future_to_promise(async move {
                                match crate::regions::fetch(&url).await {
                                    Ok(bytes) => {
                                        Ok(js_sys::Uint8Array::from(bytes.as_slice()).into())
                                    }
                                    Err(e) => Err(JsValue::from_str(&e.to_string())),
                                }
                            })
                        })
                        .clone();
                    let fetched = wasm_bindgen_futures::JsFuture::from(promise).await;
                    glyph_fetches.borrow_mut().remove(&key);
                    let mut bytes = fetched
                        .map(|v| js_sys::Uint8Array::new(&v).to_vec())
                        .map_err(|e| e.as_string().unwrap_or_else(|| "fetch failed".into()));
                    // A font file opens in shaper.wasm, loaded first.
                    if worker.declares_font(url)
                        && let Err(e) = orb_host_web::ShaperFiles::load().await
                    {
                        bytes = Err(e);
                    }
                    worker.add_resource(url, bytes.as_deref().map_err(Clone::clone));
                }
                if let Some(breaker) = worker.word_breaker() {
                    words.borrow_mut().get_or_insert(breaker);
                }
                built = worker.build(&bytes, encoding, &request);
            }
            match built {
                Ok(Built::Tile(build)) => Ok(tile_reply(id, build)),
                Ok(Built::Fetch(_)) => unreachable!("fetched above"),
                Err(e) => error(e),
            }
        })
    }
}

/// A reply for the page: the message, and the buffer and bitmap it
/// transfers.
fn reply(message: &JobReply, buffer: Option<ArrayBuffer>, image: Option<JsValue>) -> JsValue {
    object(&[
        ("message", bytes(&wire::encode(message))),
        ("buffer", buffer.map_or(JsValue::UNDEFINED, Into::into)),
        ("image", image.unwrap_or(JsValue::UNDEFINED)),
    ])
}

/// A packed tile's reply.
fn tile_reply(id: u32, build: TileBuild) -> JsValue {
    let packed = pack(build);
    let buffer = js_sys::Uint8Array::from(packed.as_slice()).buffer();
    reply(&JobReply::Tile { id }, Some(buffer), None)
}

/// Answers a features query from the tile's cached frame, or from the tile
/// fetched again when the frame left the cache.
fn query(
    worker: Option<Rc<TileWorker>>,
    request: orb_runtime::FeatureRequest,
    loader: &Arc<TileLoader>,
    id: u32,
) -> js_sys::Promise {
    let tile = &request.tile;
    let loaded = worker.as_ref().filter(|w| !w.has_data(tile)).map(|_| {
        let (sender, loaded) = futures_channel::oneshot::channel();
        loader.load(
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
            Box::new(move |r| {
                let _ = sender.send(r);
            }),
        );
        loaded
    });
    wasm_bindgen_futures::future_to_promise(async move {
        let error = |message: String| Ok(reply(&JobReply::Error { id, message }, None, None));
        let Some(worker) = worker else {
            return error(format!("style {} was replaced", request.tile.style));
        };
        let (bytes, encoding) = match loaded {
            None => (Vec::new(), request.tile.encoding),
            Some(loaded) => match loaded.await {
                Ok(Ok(fetched)) => fetched,
                Ok(Err(FetchError::NotFound)) => (Vec::new(), request.tile.encoding),
                Ok(Err(e)) => return error(e.to_string()),
                Err(_) => return error("tile loader dropped".into()),
            },
        };
        match worker.features(&bytes, encoding, &request) {
            Ok(features) => Ok(reply(&JobReply::Features { id, features }, None, None)),
            Err(e) => error(e),
        }
    })
}

/// Answers a raster job: fetches the range (from the HTTP cache, usually)
/// and replies with the requested bitmaps.
fn raster(id: u32, request: GlyphRaster) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move {
        let bitmaps = match &request {
            GlyphRaster::Pbf { url, .. } => match crate::regions::fetch(url).await {
                Ok(bytes) => request.run(&bytes),
                Err(e) => Err(format!("glyphs {url}: {e}")),
            },
            GlyphRaster::Local { .. } => request.draw(&orb_host_web::CanvasGlyphs),
            // This worker may not have opened the file a tile resolved
            // elsewhere: load the shaper and fetch the file (the HTTP
            // cache, usually) first.
            GlyphRaster::File { url, .. } => {
                let files = orb_host_web::ShaperFiles;
                match orb_host_web::ShaperFiles::load().await {
                    Err(e) => Err(e),
                    Ok(()) if orb_host_web::ShaperFiles::is_open(url) => {
                        request.draw_file(&files, None)
                    }
                    Ok(()) => match crate::regions::fetch(url).await {
                        Ok(bytes) => request.draw_file(&files, Some(&bytes)),
                        Err(e) => Err(format!("font face {url}: {e}")),
                    },
                }
            }
        };
        Ok(reply(
            &match bitmaps {
                Ok(bitmaps) => JobReply::Glyphs { id, bitmaps },
                Err(message) => JobReply::Error { id, message },
            },
            None,
            None,
        ))
    })
}

/// A message's bytes as a JS array.
fn bytes(bytes: &[u8]) -> JsValue {
    js_sys::Uint8Array::from(bytes).into()
}

fn field(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let object = Object::new();
    for (key, value) in fields {
        let _ = Reflect::set(&object, &JsValue::from_str(key), value);
    }
    object.into()
}

impl Default for JobWorker {
    fn default() -> Self {
        Self::new()
    }
}
