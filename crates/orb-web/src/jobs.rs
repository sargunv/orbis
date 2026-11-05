//! Tile jobs across workers (plan §7.1, §7.3, §10.4): both ends of the
//! protocol between the render worker and its job workers.
//!
//! The page's JS owns the transport, one `MessagePort` per job worker, and
//! passes messages through unread. The render worker's [`WorkerJobs`] posts
//! requests; a job worker's [`JobWorker`] answers each with a packed tile
//! (see [`orb_runtime::pack`]) in a transferred `ArrayBuffer`. The render
//! worker copies only that buffer's header into wasm memory; the payload
//! goes from the buffer to the GPU.
//!
//! Messages, render worker to job worker:
//! - `{style, update}`: lay out later requests for style id `style` with a
//!   revision of it, a [`StyleUpdate`] as JSON.
//! - `{id, style, source, z, zoom, subdivision?, poles?, dataZoom, col, row, url, archive,
//!   encoding, nodes?, rowsOnly}`: load a tile, or lay out some of its nodes (by key)
//!   again from the worker's cached tile, or only their property rows.
//!
//! - `{features: true, id, …tile, layer?, indices?, filter?}`: features of
//!   a tile for a query, sent to the tile's owner worker with its load's
//!   fields; answered with `{id, layers, indices, ids, stateIds, bodies}`
//!   (see [`orb_runtime::TileFeature`]) or `{id, error}`.
//! - `{raster: true, id, url, codePoints}`: glyph bitmaps of one PBF range,
//!   sent to the range's owner worker `hash(url) mod N`; answered with
//!   `{id, glyphs}`, the packed bitmaps, or `{id, error}`.
//!
//! Job worker to render worker, one per load: `{id, buffer}`, the packed
//! tile; `{id, image}`, an image source's tile as a transferred
//! `ImageBitmap`, decoded in the job worker and copied to a texture by the
//! render worker, so its pixels enter neither worker's wasm memory, with
//! `dem`, its elevations as an f32 `ArrayBuffer`, for a terrain DEM tile;
//! `{id, missing: true}` for a tile the source doesn't have; or
//! `{id, error}`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use js_sys::{ArrayBuffer, Object, Reflect, Uint32Array};
use orb_crs::TileIndex;
use orb_host::FetchError;
use orb_runtime::{
    Built, GlyphRaster, ImagePixels, Jobs, RasterDone, SpriteIcons, StyleUpdate, TileBuild,
    TileDone, TileImage, TileLoader, TileLocation, TileRequest, TileResult, TileWorker, pack,
    unpack,
};
use orb_style::Encoding;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    /// The render worker's `post(worker, message)`, which sends a message to
    /// job worker `worker`. Wasm in a worker is single-threaded, so the
    /// runtime's `Send` jobs reach it here.
    static POST: std::cell::RefCell<Option<js_sys::Function>> =
        const { std::cell::RefCell::new(None) };
}

fn post(worker: u32, message: &JsValue) {
    POST.with(|post| {
        if let Some(f) = post.borrow().as_ref() {
            let _ = f.call2(&JsValue::NULL, &worker.into(), message);
        }
    });
}

/// [`Jobs`] for the render worker: each tile goes to job worker
/// `hash(source, tile) mod N`, so a tile's relayouts and queries find its
/// data where it was decoded.
pub(crate) struct WorkerJobs {
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
    /// Jobs on `workers` job workers, reached through `post`.
    pub(crate) fn new(workers: u32, post: js_sys::Function) -> Arc<Self> {
        POST.with(|p| *p.borrow_mut() = Some(post));
        Arc::new(Self {
            workers: workers.max(1),
            state: Mutex::new(State::default()),
        })
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
        let id = number(reply, "id")? as u32;
        let raster = self
            .state
            .lock()
            .expect("jobs poisoned")
            .rasters
            .remove(&id);
        let features = self
            .state
            .lock()
            .expect("jobs poisoned")
            .features
            .remove(&id);
        if let Some(done) = features {
            done(read_features(reply));
            return Ok(());
        }
        if let Some(done) = raster {
            let result = match field(reply, "glyphs").dyn_into::<ArrayBuffer>() {
                Ok(buffer) => GlyphRaster::unpack(&js_sys::Uint8Array::new(&buffer).to_vec()),
                Err(_) => Err(field(reply, "error")
                    .as_string()
                    .unwrap_or_else(|| "malformed raster reply".into())),
            };
            done(result);
            return Ok(());
        }
        let buffer = field(reply, "buffer").dyn_into::<ArrayBuffer>().ok();
        let header = buffer.as_ref().map(header_words).transpose();
        let done = {
            let mut state = self.state.lock().expect("jobs poisoned");
            state.done += 1;
            if let Ok(Some(header)) = &header {
                state.header_bytes += header.len() as u64 * 4;
            }
            state
                .pending
                .remove(&id)
                .ok_or_else(|| JsError::new(&format!("reply to unknown job {id}")))?
        };
        let image = field(reply, "image")
            .dyn_into::<web_sys::ImageBitmap>()
            .ok();
        let result = match (buffer, header) {
            _ if field(reply, "missing").as_bool() == Some(true) => Ok(TileResult {
                missing: true,
                ..TileResult::default()
            }),
            // A DEM tile, decoded by the core in the job worker.
            _ if field(reply, "pixels").is_instance_of::<ArrayBuffer>() => {
                let pixels = js_sys::Uint8Array::new(&field(reply, "pixels")).to_vec();
                let width = field(reply, "width").as_f64().unwrap_or(0.0) as u32;
                let elevations = js_sys::Float32Array::new(&field(reply, "dem")).to_vec();
                orb_tile::DemGrid::from_elevations(width, elevations).map(|dem| TileResult {
                    image: Some(TileImage {
                        width,
                        height: width,
                        pixels: ImagePixels::Core(pixels),
                    }),
                    dem: Some(Arc::new(dem)),
                    ..TileResult::default()
                })
            }
            _ if image.is_some() => {
                let bitmap = image.expect("checked");
                Ok(TileResult {
                    image: Some(TileImage {
                        width: bitmap.width(),
                        height: bitmap.height(),
                        pixels: ImagePixels::Host(gpu.register_image_bitmap(bitmap)),
                    }),
                    ..TileResult::default()
                })
            }
            (Some(buffer), Ok(Some(header))) => {
                let bytes = gpu.register_array_buffer(buffer);
                let result = unpack(&header, bytes);
                if result.is_err() {
                    gpu.release_host_bytes(bytes);
                }
                result
            }
            (_, Err(e)) => Err(e),
            _ => Err(field(reply, "error")
                .as_string()
                .unwrap_or_else(|| "malformed job reply".into())),
        };
        done(result);
        Ok(())
    }
}

/// A features reply's features, or its error.
fn read_features(reply: &JsValue) -> Result<Vec<orb_runtime::TileFeature>, String> {
    if let Some(e) = field(reply, "error").as_string() {
        return Err(e);
    }
    let strings = |key: &str| -> Vec<Option<String>> {
        js_sys::Array::from(&field(reply, key))
            .iter()
            .map(|v| v.as_string())
            .collect()
    };
    let layers = strings("layers");
    let ids = strings("ids");
    let state_ids = strings("stateIds");
    let bodies = strings("bodies");
    let indices = field(reply, "indices")
        .dyn_into::<js_sys::Uint32Array>()
        .map_err(|_| "malformed features reply".to_string())?
        .to_vec();
    if [layers.len(), ids.len(), state_ids.len(), bodies.len()] != [indices.len(); 4] {
        return Err("malformed features reply".into());
    }
    Ok((0..indices.len())
        .map(|i| orb_runtime::TileFeature {
            layer: layers[i].clone().unwrap_or_default(),
            index: indices[i],
            id: ids[i].clone(),
            state_id: state_ids[i].clone(),
            body: bodies[i].clone().unwrap_or_default(),
        })
        .collect())
}

/// The job worker that loaded `tile` of `source`: `hash(source, tile) mod N`.
fn owner(source: u32, tile: TileIndex, workers: u32) -> u32 {
    let mut hasher = DefaultHasher::new();
    (source, tile).hash(&mut hasher);
    (hasher.finish() % u64::from(workers)) as u32
}

/// A tile request's fields, as a load and a features query send them.
fn tile_fields(request: &TileRequest) -> Vec<(&'static str, JsValue)> {
    let (url, archive) = match &request.location {
        TileLocation::Url(url) => (url.clone(), false),
        TileLocation::Pmtiles(url) => (url.clone(), true),
    };
    vec![
        ("archive", archive.into()),
        ("style", (request.style as f64).into()),
        ("source", request.source.into()),
        ("z", request.tile.z.into()),
        ("zoom", request.zoom.into()),
        (
            "subdivision",
            request.bend.map_or(JsValue::UNDEFINED, |b| b.cells.into()),
        ),
        (
            "poles",
            request.bend.map_or(JsValue::UNDEFINED, |b| {
                (u32::from(b.north) | u32::from(b.south) << 1).into()
            }),
        ),
        ("dataZoom", request.data_zoom.into()),
        ("col", request.tile.col.into()),
        ("row", request.tile.row.into()),
        ("url", url.into()),
        (
            "encoding",
            match request.encoding {
                Encoding::Mvt => 0u32,
                Encoding::Mlt => 1,
            }
            .into(),
        ),
    ]
}

/// Copies a packed tile's header, whose first word is its length in words,
/// out of `buffer`. Typed arrays use the platform's byte order, which is
/// little-endian everywhere wasm runs, as [`pack`] writes.
fn header_words(buffer: &ArrayBuffer) -> Result<Vec<u32>, String> {
    let bad = || "malformed packed tile".to_string();
    let words = buffer.byte_length() / 4;
    if words == 0 {
        return Err(bad());
    }
    let len = Uint32Array::new_with_byte_offset_and_length(buffer, 0, 1).get_index(0);
    if len == 0 || len > words {
        return Err(bad());
    }
    Ok(Uint32Array::new_with_byte_offset_and_length(buffer, 0, len).to_vec())
}

impl Jobs for WorkerJobs {
    fn set_style(&self, id: u64, update: &StyleUpdate, _: &Arc<TileWorker>) {
        let message = object(&[
            ("style", (id as f64).into()),
            ("update", JsValue::from_str(&update.to_json())),
        ]);
        for worker in 0..self.workers {
            post(worker, &message);
        }
    }

    fn set_icons(&self, id: u64, icons: &Arc<SpriteIcons>, _: &Arc<TileWorker>) {
        let message = object(&[
            ("style", (id as f64).into()),
            ("icons", JsValue::from_str(&icons.to_json())),
        ]);
        for worker in 0..self.workers {
            post(worker, &message);
        }
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        let id = {
            let mut state = self.state.lock().expect("jobs poisoned");
            let id = state.next_id;
            state.next_id = id.wrapping_add(1);
            state.rasters.insert(id, done);
            id
        };
        // A unit's owner worker rasterizes it (plan §7.3 affinity).
        let worker = request.owner(self.workers);
        let message = match request {
            GlyphRaster::Pbf { url, code_points } => object(&[
                ("raster", true.into()),
                ("id", id.into()),
                ("url", url.into()),
                (
                    "codePoints",
                    js_sys::Uint32Array::from(code_points.as_slice()).into(),
                ),
            ]),
            GlyphRaster::Local {
                stack,
                texts,
                ideographs,
            } => object(&[
                ("raster", true.into()),
                ("id", id.into()),
                ("stack", stack.into()),
                (
                    "texts",
                    texts
                        .into_iter()
                        .map(JsValue::from)
                        .collect::<js_sys::Array>()
                        .into(),
                ),
                (
                    "ideographs",
                    ideographs.map_or(JsValue::NULL, JsValue::from),
                ),
            ]),
            GlyphRaster::File { url, texts } => object(&[
                ("raster", true.into()),
                ("id", id.into()),
                ("file", url.into()),
                (
                    "texts",
                    texts
                        .into_iter()
                        .map(JsValue::from)
                        .collect::<js_sys::Array>()
                        .into(),
                ),
            ]),
        };
        post(worker, &message);
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let id = {
            let mut state = self.state.lock().expect("jobs poisoned");
            let id = state.next_id;
            state.next_id = id.wrapping_add(1);
            state.pending.insert(id, done);
            id
        };
        let worker = owner(request.source, request.tile, self.workers);
        let nodes = match &request.nodes {
            Some(nodes) => js_sys::Uint32Array::from(nodes.as_slice()).into(),
            None => JsValue::UNDEFINED,
        };
        let mut fields = tile_fields(&request);
        fields.extend([
            ("id", id.into()),
            ("nodes", nodes),
            ("rowsOnly", request.rows_only.into()),
            (
                "demUnpack",
                match request.dem_unpack {
                    Some(unpack) => js_sys::Float64Array::from(unpack.0.as_slice()).into(),
                    None => JsValue::UNDEFINED,
                },
            ),
        ]);
        post(worker, &object(&fields));
    }

    fn features(&self, request: orb_runtime::FeatureRequest, done: orb_runtime::FeaturesDone) {
        let id = {
            let mut state = self.state.lock().expect("jobs poisoned");
            let id = state.next_id;
            state.next_id = id.wrapping_add(1);
            state.features.insert(id, done);
            id
        };
        let tile = &request.tile;
        let worker = owner(tile.source, tile.tile, self.workers);
        let mut fields = tile_fields(tile);
        fields.extend([("id", id.into()), ("features", true.into())]);
        match request.select {
            orb_runtime::FeatureSelect::Indices { layer, features } => fields.extend([
                ("layer", layer.into()),
                (
                    "indices",
                    js_sys::Uint32Array::from(features.as_slice()).into(),
                ),
            ]),
            orb_runtime::FeatureSelect::Filter { layer, filter } => fields.extend([
                ("layer", layer.map_or(JsValue::UNDEFINED, JsValue::from)),
                ("filter", filter.map_or(JsValue::UNDEFINED, JsValue::from)),
            ]),
        }
        post(worker, &object(&fields));
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

    /// Handles a message from the render worker. Returns a promise of the
    /// reply to post back, transferring its `buffer`, or nothing.
    pub fn receive(&mut self, message: &JsValue) -> Result<Option<js_sys::Promise>, JsError> {
        if field(message, "raster").as_bool() == Some(true) {
            return Ok(Some(raster(message)?));
        }
        let style = number(message, "style")? as u64;
        if let Some(update) = field(message, "update").as_string() {
            let update = StyleUpdate::parse(&update).map_err(|e| JsError::new(&e))?;
            let previous = self
                .styles
                .iter()
                .find(|(id, _)| *id == style)
                .map(|(_, w)| Rc::clone(w));
            let worker =
                TileWorker::update(previous.as_deref(), &update).map_err(|e| JsError::new(&e))?;
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
            return Ok(None);
        }
        if let Some(icons) = field(message, "icons").as_string() {
            let icons = SpriteIcons::parse(&icons).map_err(|e| JsError::new(&e))?;
            // Revisions share their worker data, so this reaches every one.
            if let Some((_, worker)) = self.styles.iter().find(|(id, _)| *id == style) {
                worker.set_icons(Arc::new(icons));
            }
            return Ok(None);
        }
        let id = number(message, "id")?;
        let source = number(message, "source")? as u32;
        let data_zoom = number(message, "dataZoom")? as u8;
        let tile = TileIndex {
            z: number(message, "z")? as u8,
            col: number(message, "col")? as u32,
            row: number(message, "row")? as u32,
        };
        let encoding = match number(message, "encoding")? as u32 {
            0 => Encoding::Mvt,
            _ => Encoding::Mlt,
        };
        let url = field(message, "url")
            .as_string()
            .ok_or_else(|| JsError::new("tile request without a url"))?;
        let location = if field(message, "archive").as_bool() == Some(true) {
            TileLocation::Pmtiles(url.clone())
        } else {
            TileLocation::Url(url.clone())
        };
        let request = TileRequest {
            style,
            source,
            tile,
            zoom: number(message, "zoom")?,
            bend: field(message, "subdivision").as_f64().map(|cells| {
                let poles = field(message, "poles").as_f64().unwrap_or(0.0) as u32;
                orb_runtime::Bend {
                    cells: cells as u32,
                    north: poles & 1 != 0,
                    south: poles & 2 != 0,
                }
            }),
            data_zoom,
            location: location.clone(),
            encoding,
            nodes: field(message, "nodes")
                .dyn_into::<js_sys::Uint32Array>()
                .ok()
                .map(|nodes| nodes.to_vec()),
            rows_only: field(message, "rowsOnly").as_bool() == Some(true),
            dem_unpack: field(message, "demUnpack")
                .dyn_into::<js_sys::Float64Array>()
                .ok()
                .and_then(|unpack| unpack.to_vec().try_into().ok())
                .map(orb_runtime::DemUnpack),
        };
        let worker = self
            .styles
            .iter()
            .find(|(i, _)| *i == style)
            .map(|(_, w)| Rc::clone(w));
        if field(message, "features").as_bool() == Some(true) {
            return Ok(Some(query(worker, request, message, &self.loader, id)));
        }
        let loaded = if worker
            .as_ref()
            .is_some_and(|worker| !worker.has_data(&request) && worker.data_url(source).is_none())
        {
            let (sender, loaded) = futures_channel::oneshot::channel();
            self.loader.load(
                location,
                TileIndex {
                    z: data_zoom,
                    col: tile.col >> (tile.z - data_zoom),
                    row: tile.row >> (tile.z - data_zoom),
                },
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
            if worker.has_data(&request) || worker.data_url(source).is_none() {
                return None;
            }
            let mut fetches = self.source_fetches.borrow_mut();
            let promise = fetches.entry((style, source)).or_insert_with(|| {
                let url = url.clone();
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
        Ok(Some(wasm_bindgen_futures::future_to_promise(async move {
            let reply = |fields: &[(&str, JsValue)]| {
                let mut all = vec![("id", JsValue::from(id))];
                all.extend(fields.iter().cloned());
                Ok(object(&all))
            };
            let Some(worker) = worker else {
                return reply(&[("error", format!("style {style} was replaced").into())]);
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
                        let e = e
                            .as_string()
                            .unwrap_or_else(|| "GeoJSON fetch failed".into());
                        return reply(&[("error", e.into())]);
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
                        let packed = pack(&TileBuild::default());
                        let buffer = js_sys::Uint8Array::from(packed.as_slice()).buffer();
                        return reply(&[("buffer", buffer.into())]);
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
                            Ok((image, grid)) => reply(&[
                                (
                                    "pixels",
                                    js_sys::Uint8Array::from(image.pixels.as_slice())
                                        .buffer()
                                        .into(),
                                ),
                                ("width", image.width.into()),
                                (
                                    "dem",
                                    js_sys::Float32Array::from(grid.elevations())
                                        .buffer()
                                        .into(),
                                ),
                            ]),
                            Err(e) => reply(&[("error", format!("image decode: {e}").into())]),
                        };
                    }
                    Ok((bytes, _)) if image.is_some() => {
                        return match orb_host_web::decode_bitmap(&bytes).await {
                            Ok(bitmap) => reply(&[("image", bitmap.into())]),
                            Err(e) => reply(&[("error", format!("image decode: {e:?}").into())]),
                        };
                    }
                    Ok((bytes, _)) => bytes,
                    Err(FetchError::NotFound) => return reply(&[("missing", true.into())]),
                    Err(e) => return reply(&[("error", e.to_string().into())]),
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
                Ok(Built::Tile(build)) => {
                    let packed = pack(&build);
                    let buffer = js_sys::Uint8Array::from(packed.as_slice()).buffer();
                    reply(&[("buffer", buffer.into())])
                }
                Ok(Built::Fetch(_)) => unreachable!("fetched above"),
                Err(e) => reply(&[("error", e.into())]),
            }
        })))
    }
}

/// Answers a features query from the tile's cached frame, or from the tile
/// fetched again when the frame left the cache.
fn query(
    worker: Option<Rc<TileWorker>>,
    tile: TileRequest,
    message: &JsValue,
    loader: &Arc<TileLoader>,
    id: f64,
) -> js_sys::Promise {
    let layer = field(message, "layer").as_string();
    let select = match field(message, "indices").dyn_into::<js_sys::Uint32Array>() {
        Ok(indices) => orb_runtime::FeatureSelect::Indices {
            layer: layer.unwrap_or_default(),
            features: indices.to_vec(),
        },
        Err(_) => orb_runtime::FeatureSelect::Filter {
            layer,
            filter: field(message, "filter").as_string(),
        },
    };
    let loaded = worker.as_ref().filter(|w| !w.has_data(&tile)).map(|_| {
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
        let reply = |fields: Vec<(&str, JsValue)>| {
            let mut all = vec![("id", JsValue::from(id))];
            all.extend(fields);
            Ok(object(&all))
        };
        let error = |e: String| reply(vec![("error", e.into())]);
        let Some(worker) = worker else {
            return error(format!("style {} was replaced", tile.style));
        };
        let (bytes, encoding) = match loaded {
            None => (Vec::new(), tile.encoding),
            Some(loaded) => match loaded.await {
                Ok(Ok(fetched)) => fetched,
                Ok(Err(FetchError::NotFound)) => (Vec::new(), tile.encoding),
                Ok(Err(e)) => return error(e.to_string()),
                Err(_) => return error("tile loader dropped".into()),
            },
        };
        let request = orb_runtime::FeatureRequest { tile, select };
        match worker.features(&bytes, encoding, &request) {
            Ok(features) => {
                let strings = |f: &dyn Fn(&orb_runtime::TileFeature) -> Option<&str>| {
                    let array = js_sys::Array::new();
                    for feature in &features {
                        array.push(&f(feature).map_or(JsValue::NULL, JsValue::from));
                    }
                    JsValue::from(array)
                };
                let indices: Vec<u32> = features.iter().map(|f| f.index).collect();
                reply(vec![
                    ("layers", strings(&|f| Some(f.layer.as_str()))),
                    (
                        "indices",
                        js_sys::Uint32Array::from(indices.as_slice()).into(),
                    ),
                    ("ids", strings(&|f| f.id.as_deref())),
                    ("stateIds", strings(&|f| f.state_id.as_deref())),
                    ("bodies", strings(&|f| Some(f.body.as_str()))),
                ])
            }
            Err(e) => error(e),
        }
    })
}

/// Answers a raster job: fetches the range (from the HTTP cache, usually)
/// and replies with the requested bitmaps.
fn raster(message: &JsValue) -> Result<js_sys::Promise, JsError> {
    let id = number(message, "id")?;
    let texts = || {
        js_sys::Array::from(&field(message, "texts"))
            .iter()
            .filter_map(|t| t.as_string())
            .collect()
    };
    let request = if let Some(url) = field(message, "file").as_string() {
        GlyphRaster::File {
            url,
            texts: texts(),
        }
    } else if let Some(stack) = field(message, "stack").as_string() {
        GlyphRaster::Local {
            stack,
            texts: texts(),
            ideographs: field(message, "ideographs").as_string(),
        }
    } else {
        GlyphRaster::Pbf {
            url: field(message, "url")
                .as_string()
                .ok_or_else(|| JsError::new("raster job without a url"))?,
            code_points: field(message, "codePoints")
                .dyn_into::<js_sys::Uint32Array>()
                .map_err(|_| JsError::new("raster job without code points"))?
                .to_vec(),
        }
    };
    Ok(wasm_bindgen_futures::future_to_promise(async move {
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
        Ok(match bitmaps {
            Ok(bitmaps) => {
                let packed = GlyphRaster::pack(&bitmaps);
                let buffer = js_sys::Uint8Array::from(packed.as_slice()).buffer();
                object(&[("id", id.into()), ("glyphs", buffer.into())])
            }
            Err(e) => object(&[("id", id.into()), ("error", e.into())]),
        })
    }))
}

fn field(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn number(object: &JsValue, key: &str) -> Result<f64, JsError> {
    field(object, key)
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("job message without `{key}`")))
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
