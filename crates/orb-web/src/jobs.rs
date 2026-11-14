//! Tile jobs across workers: both ends of the
//! protocol between the page and its job workers.
//!
//! The page's JS owns the transport, one `MessagePort` per job worker, and
//! passes messages through unread. The page's [`WorkerJobs`] posts each
//! request as a [`JobRequest`] in [`orb_wire`]'s bytes; a job
//! worker's [`JobWorker`] answers each one with an id as
//! `{message, buffer?, image?}`: a [`JobReply`], with a packed tile ([`pack`])
//! or a DEM tile's pixels as the transferred `buffer` and an image tile as
//! the transferred `image` bitmap. The page copies only a packed tile's
//! header into wasm memory; the payload goes from the buffer to the GPU.
//! Which worker a request goes to, and what happens when one is lost, is
//! `dispatch`'s.

use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

use js_sys::ArrayBuffer;
use orb_hal::Gpu as _;
use orb_runtime::{
    DemUnpack, ImageKind, ImagePixels, JobHost, JobReply, JobRequest, Loaded, TileBuild, TileImage,
    TileResult, pack, packed_header, unpack,
};
use orb_wire as wire;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::dispatch::WorkerJobs;

thread_local! {
    /// Each map's `post(worker, message)`, which sends a message to its job
    /// worker `worker`, by map. Wasm in a page is single-threaded, so the
    /// runtime's `Send` jobs reach it here.
    static POSTS: std::cell::RefCell<HashMap<u32, js_sys::Function>> =
        std::cell::RefCell::new(HashMap::default());
    /// A job worker's `notify(reply)`, which posts a reply to the page
    /// ahead of a request's answer: a load's fetch ended.
    static NOTIFY: std::cell::RefCell<Option<js_sys::Function>> =
        const { std::cell::RefCell::new(None) };
}

/// Forgets map `map`'s job workers' `post`.
pub(crate) fn forget(map: u32) {
    POSTS.with(|posts| posts.borrow_mut().remove(&map));
}

impl WorkerJobs {
    /// Map `map`'s jobs on `workers` job workers, reached through `post`,
    /// with `fetches` tile fetches in flight at once.
    pub(crate) fn posting(
        map: u32,
        workers: u32,
        fetches: usize,
        post: js_sys::Function,
    ) -> Arc<Self> {
        POSTS.with(|posts| posts.borrow_mut().insert(map, post));
        let post = move |to: &[u32], message: &[u8]| {
            let message = bytes(message);
            POSTS.with(|posts| {
                if let Some(f) = posts.borrow().get(&map) {
                    for &worker in to {
                        let _ = f.call2(&JsValue::NULL, &worker.into(), &message);
                    }
                }
            });
        };
        Self::new(workers, fetches, Box::new(post), crate::regions::service())
    }

    /// Whether a job worker's reply carries a tile's data beside its
    /// message: a packed tile or a DEM tile's texels, or an image.
    pub(crate) fn carries_data(reply: &JsValue) -> bool {
        let has = |key| !crate::js::get(reply, key).is_undefined();
        has("buffer") || has("image")
    }

    /// Completes a request with a job worker's reply. A packed tile's
    /// buffer is registered with `gpu`; only its header enters wasm memory,
    /// counted in `headers`.
    pub(crate) fn receive(
        &self,
        gpu: &mut orb_hal_wgpu::WgpuGpu,
        reply: &JsValue,
        headers: &mut u64,
    ) -> Result<(), JsError> {
        let message = crate::js::get(reply, "message")
            .dyn_into::<js_sys::Uint8Array>()
            .map_err(|_| JsError::new("job reply without a message"))?;
        let message: JobReply = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let buffer = crate::js::get(reply, "buffer")
            .dyn_into::<ArrayBuffer>()
            .ok();
        self.complete(message, |message| {
            payload(message, gpu, buffer, reply, headers)
        })
        .map_err(|e| JsError::new(&e))
    }
}

/// A load's result from a reply whose data travels beside it: a packed
/// tile or a DEM tile's pixels in `buffer`, the packed tile registered
/// with `gpu` (its header counted in `headers`), or an image tile's bitmap
/// in `reply`.
fn payload(
    message: JobReply,
    gpu: &mut orb_hal_wgpu::WgpuGpu,
    buffer: Option<ArrayBuffer>,
    reply: &JsValue,
    headers: &mut u64,
) -> Result<TileResult, String> {
    match message {
        // A DEM tile, decoded by the core in the job worker: its texels go
        // from the buffer to the GPU, and its elevations alone into wasm
        // memory, for the CPU.
        JobReply::Dem { width, .. } => {
            let buffer = buffer.ok_or("DEM reply without a buffer")?;
            let texels = u64::from(width) * u64::from(width) * 4;
            if u64::from(buffer.byte_length()) != 2 * texels {
                return Err("malformed DEM reply".into());
            }
            // Its elevations straight from the buffer into wasm memory.
            let count = width * width;
            let elevations = js_sys::Float32Array::new_with_byte_offset_and_length(
                &buffer,
                texels as u32,
                count,
            )
            .to_vec();
            let dem = orb_tile::DemGrid::from_elevations(width, elevations)?;
            let bytes = gpu.register_array_buffer(buffer);
            Ok(TileResult {
                image: Some(TileImage {
                    width,
                    height: width,
                    pixels: ImagePixels::Bytes { bytes, offset: 0 },
                }),
                dem: Some(Arc::new(dem)),
                ..TileResult::default()
            })
        }
        JobReply::Image { warp_grid, .. } => {
            let bitmap = crate::js::get(reply, "image")
                .dyn_into::<web_sys::ImageBitmap>()
                .map_err(|_| "image reply without a bitmap")?;
            Ok(TileResult {
                image: Some(TileImage {
                    width: bitmap.width(),
                    height: bitmap.height(),
                    pixels: ImagePixels::Host(gpu.register_image_bitmap(bitmap)),
                }),
                warp_grid: warp_grid.map(Arc::new),
                ..TileResult::default()
            })
        }
        JobReply::Tile { .. } => {
            let buffer = buffer.ok_or("tile reply without a buffer")?;
            let header = header_bytes(&buffer)?;
            *headers += header.len() as u64;
            let len = u64::from(buffer.byte_length());
            let bytes = gpu.register_array_buffer(buffer);
            let result = unpack(&header, bytes, len);
            if result.is_err() {
                gpu.release_host_bytes(bytes);
            }
            result
        }
        _ => Err("malformed job reply".into()),
    }
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

/// A job worker: a [`JobHost`] behind the page's messages.
#[wasm_bindgen]
#[derive(Debug)]
pub struct JobWorker {
    host: Arc<JobHost>,
}

#[wasm_bindgen]
impl JobWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // The page's turns order the fetches across its workers.
        let host = JobHost::unlimited(crate::regions::service(), Arc::new(orb_host::jobs::Inline));
        host.set_local_glyphs(Arc::new(crate::host::CanvasGlyphs));
        host.set_font_files(Arc::new(crate::host::ShaperFiles));
        if let Some(words) = crate::host::browser_words() {
            host.set_word_breaker(words);
        }
        Self { host }
    }

    /// How this worker loads `shaper.wasm` for `font-faces` files:
    /// `loader()` returns a promise of a `Shaper`.
    #[wasm_bindgen(js_name = setShaperLoader)]
    pub fn set_shaper_loader(&mut self, loader: js_sys::Function) {
        crate::host::ShaperFiles::set_loader(loader);
    }

    /// How this worker tells the page a load's fetch ended:
    /// `notify(reply)` posts a reply, as `receive`'s are posted.
    #[wasm_bindgen(js_name = setNotify)]
    pub fn set_notify(&mut self, notify: js_sys::Function) {
        NOTIFY.with(|n| *n.borrow_mut() = Some(notify));
    }

    /// Traps, as a bug would: the page replaces the worker. To exercise
    /// that recovery.
    pub fn trap(&self) {
        panic!("a trap asked for");
    }

    /// Handles a request from the page: a [`JobRequest`] in
    /// [`orb_wire`]'s bytes. Returns a promise of the reply to
    /// post back, transferring its `buffer` and `image`, or nothing.
    pub fn receive(
        &mut self,
        message: &js_sys::Uint8Array,
    ) -> Result<Option<js_sys::Promise>, JsError> {
        let request: JobRequest = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let (sender, answer) = futures_channel::oneshot::channel::<JobAnswer>();
        let (id, dem_unpack) = match request {
            JobRequest::Style { style, update } => {
                self.host
                    .update_style(style, &update)
                    .map_err(|e| JsError::new(&e))?;
                return Ok(None);
            }
            JobRequest::Icons { style, icons } => {
                self.host.set_icons(style, Arc::new(icons));
                return Ok(None);
            }
            JobRequest::ArchiveHead { url, head } => {
                self.host.add_archive_head(&url, &head);
                return Ok(None);
            }
            JobRequest::Cancel {
                style,
                source,
                tile,
            } => {
                self.host.cancel(style, source, tile);
                return Ok(None);
            }
            JobRequest::Glyphs { style, urls } => {
                self.host.expect_glyphs(style, urls);
                return Ok(None);
            }
            JobRequest::GlyphRange { style, url, bytes } => {
                self.host.glyph_range(style, url, bytes);
                return Ok(None);
            }
            JobRequest::RetryGlyphs => {
                self.host.retry_glyphs();
                return Ok(None);
            }
            JobRequest::Cluster { id, request } => {
                self.host.cluster(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Cluster(r)))),
                );
                (id, None)
            }
            JobRequest::Load { id, request } => {
                let dem_unpack = request.dem_unpack;
                let fetched = Box::new(move || {
                    let message = reply(&JobReply::Fetched { id }, None, None);
                    NOTIFY.with(|n| {
                        if let Some(notify) = &*n.borrow() {
                            let _ = notify.call1(&JsValue::NULL, &message);
                        }
                    });
                });
                self.host.load(
                    request,
                    Some(fetched),
                    Box::new(move |r| drop(sender.send(JobAnswer::Load(r)))),
                );
                (id, dem_unpack)
            }
            JobRequest::Features { id, request } => {
                self.host.features(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Features(r)))),
                );
                (id, None)
            }
            JobRequest::Raster { id, request } => {
                self.host.raster(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Raster(r)))),
                );
                (id, None)
            }
        };
        Ok(Some(wasm_bindgen_futures::future_to_promise(async move {
            let error = |message: String| reply(&JobReply::Error { id, message }, None, None);
            let Ok(answer) = answer.await else {
                return Ok(error("the job host dropped the request".into()));
            };
            Ok(match answer {
                JobAnswer::Load(Ok(Loaded::Tile(build))) => tile_reply(id, build),
                JobAnswer::Load(Ok(Loaded::Missing)) => {
                    reply(&JobReply::Missing { id }, None, None)
                }
                JobAnswer::Load(Ok(Loaded::Cancelled)) => {
                    reply(&JobReply::Cancelled { id }, None, None)
                }
                // A terrain DEM tile's texels, for the GPU and its
                // elevations for the CPU.
                JobAnswer::Load(Ok(Loaded::Image {
                    bytes,
                    kind: ImageKind::Dem,
                    ..
                })) if let Some(DemUnpack(unpack)) = dem_unpack => {
                    let decoded = crate::host::decode_image(&bytes)
                        .await
                        .map_err(|e| format!("{e:?}"))
                        .and_then(|image| {
                            let grid =
                                orb_tile::DemGrid::from_rgba(image.width, &image.pixels, unpack)?;
                            Ok((image, grid))
                        });
                    match decoded {
                        Ok((image, grid)) => reply(
                            &JobReply::Dem {
                                id,
                                width: image.width,
                            },
                            Some(dem_buffer(&image.pixels, grid.elevations())),
                            None,
                        ),
                        Err(e) => error(format!("image decode: {e}")),
                    }
                }
                JobAnswer::Load(Ok(Loaded::Image {
                    bytes, warp_grid, ..
                })) => match crate::host::decode_bitmap(&bytes, true).await {
                    Ok(bitmap) => {
                        let warp_grid = warp_grid.map(Arc::unwrap_or_clone);
                        let message = JobReply::Image { id, warp_grid };
                        reply(&message, None, Some(bitmap.into()))
                    }
                    Err(e) => error(format!("image decode: {e:?}")),
                },
                JobAnswer::Features(Ok(features)) => {
                    reply(&JobReply::Features { id, features }, None, None)
                }
                JobAnswer::Raster(Ok(bitmaps)) => {
                    reply(&JobReply::Glyphs { id, bitmaps }, None, None)
                }
                JobAnswer::Cluster(Ok(json)) => reply(&JobReply::Answer { id, json }, None, None),
                JobAnswer::Load(Err(e))
                | JobAnswer::Features(Err(e))
                | JobAnswer::Raster(Err(e))
                | JobAnswer::Cluster(Err(e)) => error(e),
            })
        })))
    }
}

/// What the host answered a request with.
enum JobAnswer {
    Load(Result<Loaded, String>),
    Features(Result<Vec<orb_runtime::TileFeature>, String>),
    Raster(Result<Vec<orb_runtime::GlyphBitmap>, String>),
    Cluster(Result<String, String>),
}

/// A reply for the page: the message, and the buffer and bitmap it
/// transfers.
fn reply(message: &JobReply, buffer: Option<ArrayBuffer>, image: Option<JsValue>) -> JsValue {
    crate::js::object(&[
        ("message", bytes(&wire::encode(message))),
        ("buffer", buffer.map_or(JsValue::UNDEFINED, Into::into)),
        ("image", image.unwrap_or(JsValue::UNDEFINED)),
    ])
    .into()
}

/// A DEM tile's texels, then its elevations (little-endian `f32`s), in
/// one buffer to transfer.
fn dem_buffer(texels: &[u8], elevations: &[f32]) -> ArrayBuffer {
    let len = texels.len() + elevations.len() * 4;
    let array = js_sys::Uint8Array::new_with_length(len as u32);
    array.subarray(0, texels.len() as u32).copy_from(texels);
    // A whole number of texels: the elevations start on a word.
    let buffer = array.buffer();
    js_sys::Float32Array::new_with_byte_offset_and_length(
        &buffer,
        texels.len() as u32,
        elevations.len() as u32,
    )
    .copy_from(elevations);
    buffer
}

/// A packed tile's reply: its head and payload copied once, into the
/// buffer it transfers.
fn tile_reply(id: u32, build: TileBuild) -> JsValue {
    let packed = pack(build);
    let array = js_sys::Uint8Array::new_with_length(packed.len() as u32);
    let head = packed.head.len() as u32;
    array.subarray(0, head).copy_from(&packed.head);
    array
        .subarray(head, packed.len() as u32)
        .copy_from(packed.payload_bytes());
    reply(&JobReply::Tile { id }, Some(array.buffer()), None)
}

/// A message's bytes as a JS array.
fn bytes(bytes: &[u8]) -> JsValue {
    js_sys::Uint8Array::from(bytes).into()
}

impl Default for JobWorker {
    fn default() -> Self {
        Self::new()
    }
}
