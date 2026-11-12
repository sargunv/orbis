//! The Rust embedding API for native hosts: a [`Runtime`]
//! owns a GPU device and the host services, a [`Map`] owns a style, its
//! camera store and its gestures, and a [`Target`] is what a map draws
//! into: a window surface or a texture the runtime owns.
//!
//! Everything here is `Send`, so a host calls from whichever
//! thread it draws on, which becomes the scene thread: jobs, fetches and
//! placement run on the runtime's pools, and a map's [`Map::set_wake`]
//! callback tells the host, from any thread, that a frame is needed.
//! `orb-demo`, `orb-android` and `orb-ffi` draw through it.

#![forbid(unsafe_code)]

mod device;
mod map;
mod regions;
mod target;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

pub use map::{CullCount, Event, Image, Map, STILL_TIMEOUT, TileRequest};
pub use orb_camera::input::{
    Button, Fling, GestureOptions, GesturePhase, Input, Key, Modifiers, Movement, PointerKind,
    Thresholds, WheelDelta,
};
pub use orb_camera::{
    AnchoredMomentum, AnimationId, Bounds, CameraEvent, CameraTarget, Change, Constraints,
    DecayModel, EaseOptions, Easing, FitOptions, FlyOptions, FreeCamera, GestureMomentum,
    MomentumDecay, MomentumSnap, MomentumVelocity, Padding, PanMomentum, Phase, Skew, TileLod,
    TileLodMode,
};
pub use orb_crs::{Geodetic, Projection, TileIndex, TileMatrixSet};
pub use orb_hal::GpuCounters;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use orb_hal_wgpu::{MetalTexture, metal_texture};
pub use orb_hal_wgpu::{RawSurface, SubmitTimes, Surface};
pub use orb_host::abort::Listening;
pub use orb_host::{
    Abort, FetchError, NetworkAnswer, NetworkDone, NetworkHook, NetworkRequest, Prior, Request,
    RequestTransform, ResourceKind, redacted_url,
};
pub use orb_runtime::{
    CameraState, ClusterQuery, FeatureTarget, FrameReport, FrameTimes, HOST_TILES, Hit, HostTile,
    MapEvent, MapOptions, Memory, MissingTiles, Op, PoolMemory, Preview, Previews, Progress,
    Resized, StyleDoc, StyleImage, Transaction, VideoRequest, json,
};
pub use regions::{RegionDefinition, RegionEvent, RegionInfo};
pub use target::Target;

use orb_hal_wgpu::WgpuGpu;
use orb_host::jobs::ThreadPool;
pub use orb_host_native::CacheStats;
/// Gives HTTPS the app's JVM and `Context`: from then on it verifies
/// servers with Android's trust store (`orb_host_native::android::init`).
#[cfg(target_os = "android")]
pub use orb_host_native::android::init as init_android;
use orb_host_native::{HttpCache, NativeFetch, NativeImageDecode};

/// Why a call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// An argument is invalid, such as malformed JSON.
    InvalidArgument(String),
    /// The call doesn't fit the current state, such as editing before a
    /// style is loaded.
    InvalidState(String),
    /// The platform or the build lacks the feature.
    Unsupported(String),
    /// The GPU or the platform failed.
    Native(String),
    /// The target can't be drawn to again: make another, as after the
    /// GPU was replaced.
    TargetLost(String),
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(m)
            | Self::InvalidState(m)
            | Self::Unsupported(m)
            | Self::Native(m)
            | Self::TargetLost(m) => m,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

/// How a runtime is set up.
#[derive(Clone, Debug, Default)]
pub struct RuntimeOptions {
    /// Tile job threads; the platform default when `None`.
    pub job_threads: Option<usize>,
    /// Where the runtime keeps state between runs: compiled
    /// pipelines (`pipelines/`), the HTTP cache (`http/`,
    /// [`HTTP_CACHE_BYTES`]) and offline regions. Without one, pipelines
    /// compile on every start and nothing fetched is kept.
    pub state_dir: Option<PathBuf>,
    /// Where offline regions are kept without a state directory, which
    /// otherwise keeps them (`regions/` in it).
    pub regions_dir: Option<PathBuf>,
}

/// A GPU device and host services, shared by the maps made from it. A
/// runtime, its maps and its targets are `Send`, and a runtime is `Sync`:
/// an embedder can make them on one thread and draw on another, such as
/// a render thread of its own, on macOS too (`orb-demo` presents from
/// one). The device is locked for each call that uses it, so maps of one
/// runtime on different threads take turns on it.
#[derive(Clone, Debug)]
pub struct Runtime {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    gpu: Mutex<WgpuGpu>,
    options: RuntimeOptions,
    /// The device's adapter.
    adapter: Mutex<String>,
    /// Counts devices made: maps and targets of an older one rebuild.
    generation: AtomicU64,
    /// Device losses in a row, and how the next is answered.
    recovery: Mutex<Recovery>,
    /// Fetches, through the HTTP cache with a state directory.
    fetch: Arc<NativeFetch>,
    /// Offline regions, with a state directory.
    regions: Option<regions::Regions>,
    /// Tile jobs and pipeline compiles.
    executor: Arc<ThreadPool>,
    images: orb_runtime::Images,
    /// Where the pipeline cache is kept, and the pipeline count it was
    /// last saved at.
    cache: Mutex<Option<(PathBuf, u32)>>,
    /// Every map's wake: the device calls them all when a pipeline
    /// compiles.
    wakes: Wakes,
    /// Whether submissions are timed ([`Runtime::set_gpu_timing`]).
    timing: AtomicBool,
}

/// What a runtime's device is and has done ([`Runtime::stats`]).
#[derive(Clone, Debug)]
pub struct RuntimeStats {
    /// The device's adapter: its name and graphics API.
    pub adapter: String,
    /// What the device has done so far: pipelines asked for and compiled,
    /// memory made, bytes and calls uploaded.
    pub counters: GpuCounters,
    /// Time spent making shader modules and pipelines so far.
    pub pipeline_time: std::time::Duration,
    /// Each pipeline compiled so far and how long it took, in the order
    /// they finished (pipelines created cold and warm).
    pub pipeline_timings: Vec<(String, std::time::Duration)>,
    /// The compile the watchdog found hung, or the pipeline the driver
    /// couldn't make, if one was: the GPU is then lost, and frames fail
    /// with the reason.
    pub hung_compile: Option<String>,
    /// What the HTTP cache holds, if there is one.
    pub http_cache: Option<CacheStats>,
    /// Bytes the offline regions keep on disk.
    pub region_bytes: u64,
    /// The widest and tallest texture target the device makes, in pixels.
    pub max_texture_size: u32,
    /// Whether [`Runtime::gpu_timings`] times passes on the GPU: the
    /// device has timestamp queries.
    pub gpu_timestamps: bool,
    /// The GPU timestamps' resolution, once timing has seen a step.
    pub timestamp_resolution: Option<std::time::Duration>,
    /// GPU resources alive: buffers, textures, pipelines and the rest.
    pub live_resources: usize,
}

/// A device that ran this long before it was lost ends a run of losses:
/// the next loss is the first of a new run.
const STABLE: std::time::Duration = std::time::Duration::from_secs(30);

/// How long after each loss of a run a new device is made: the first at
/// once (an app switch, a driver reset), later ones after a pause. A loss
/// past the last isn't answered: a driver that loses every device it gives
/// (Adreno X1's Vulkan lost one a second, #141) would otherwise reset
/// the map forever.
const BACKOFF: [std::time::Duration; 3] = [
    std::time::Duration::ZERO,
    std::time::Duration::from_millis(250),
    std::time::Duration::from_secs(1),
];

/// How the runtime answers device losses.
#[derive(Debug)]
struct Recovery {
    /// When the device was made.
    made: std::time::Instant,
    /// Losses in the current run.
    losses: u32,
    /// When the loss being answered gets its new device.
    due: Option<std::time::Instant>,
    /// Why the runtime stopped making devices.
    failed: Option<String>,
}

impl Recovery {
    fn new() -> Self {
        Self {
            made: std::time::Instant::now(),
            losses: 0,
            due: None,
            failed: None,
        }
    }

    /// Stops making devices, for `why`.
    fn fail(&mut self, why: String) -> Error {
        self.failed = Some(why.clone());
        Error::Native(why)
    }
}

/// Every map's wake, held weakly: a dropped map's goes.
type Wakes = Arc<std::sync::Mutex<Vec<std::sync::Weak<dyn Fn() + Send + Sync>>>>;

/// A wake for the device that wakes every map still alive.
fn wake_all(wakes: &Wakes) -> Arc<dyn Fn() + Send + Sync> {
    let wakes = wakes.clone();
    Arc::new(move || {
        let live: Vec<_> = {
            let mut wakes = wakes.lock().expect("wakes poisoned");
            wakes.retain(|w| w.strong_count() > 0);
            wakes.iter().filter_map(std::sync::Weak::upgrade).collect()
        };
        for wake in live {
            wake();
        }
    })
}

/// Asks for a capture of a new runtime's frame `n`, from 1, or (0) only
/// for every frame's debug groups ([`Runtime::capture_gpu_frames`]).
const GPU_CAPTURE: &str = "ORB_GPU_CAPTURE";

impl Runtime {
    /// Makes a device on the default adapter and starts the job and fetch
    /// pools. Blocks while the device is made.
    pub fn new(options: &RuntimeOptions) -> Result<Self, Error> {
        let (mut gpu, adapter) = device::make()?;
        let wakes = Arc::default();
        let threads = options
            .job_threads
            .unwrap_or_else(ThreadPool::default_size)
            .max(1);
        let executor = Arc::new(ThreadPool::new("jobs", threads));
        device::set_compiler(&mut gpu, executor.clone(), wake_all(&wakes));
        let fetch = NativeFetch::with_cache(http_cache(options)?);
        let runtime = Self {
            shared: Arc::new(Shared {
                gpu: Mutex::new(gpu),
                options: options.clone(),
                adapter: Mutex::new(adapter),
                generation: AtomicU64::new(0),
                recovery: Mutex::new(Recovery::new()),
                fetch: fetch.clone(),
                regions: options
                    .state_dir
                    .as_ref()
                    .or(options.regions_dir.as_ref())
                    .map(|dir| regions::Regions::new(dir, fetch)),
                executor,
                images: Default::default(),
                cache: Mutex::new(None),
                wakes,
                timing: AtomicBool::new(false),
            }),
        };
        runtime.start_pipeline_cache()?;
        if let Some(frame) = std::env::var(GPU_CAPTURE)
            .ok()
            .and_then(|n| n.parse::<u64>().ok())
        {
            let mut gpu = runtime.shared.gpu();
            gpu.set_debug_groups(true);
            if let Some(skip) = frame.checked_sub(1) {
                gpu.capture_frames(skip, 1);
            }
        }
        Ok(runtime)
    }

    /// Asks the graphics debugger the process runs under (Xcode's Metal
    /// debugger, RenderDoc) to capture `count` of the frames this
    /// runtime's maps draw, after the next `skip`: each a frame's passes
    /// grouped by stage, every pass, pipeline and buffer labelled by its
    /// role. Without a debugger attached it does nothing. The
    /// `ORB_GPU_CAPTURE` environment variable asks for frame `n` (from 1)
    /// of a new runtime, and with 0 for none: either way every frame then
    /// carries its debug groups, for captures the debugger's own controls
    /// start.
    pub fn capture_gpu_frames(&self, skip: u64, count: u64) {
        self.shared.gpu().capture_frames(skip, count);
    }

    /// Rewrites every request the runtime's maps and region downloads make
    /// from here on (GL JS `transformRequest`): its URL, before the HTTP
    /// cache, and headers to send, such as a tile server's token; `None`
    /// from the transform refuses the request.
    pub fn set_request_transform(&self, transform: Option<Arc<orb_host::RequestTransform>>) {
        self.shared.fetch.set_transform(transform);
    }

    /// Serves the requests of the runtime's maps and region downloads that
    /// the HTTP cache and offline regions can't answer through `hook`, in
    /// the network's place, from here on: any URL scheme, with the cached
    /// copy it would replace. `None` hands them back to the network.
    pub fn set_network_hook(&self, hook: Option<Arc<dyn orb_host::NetworkHook>>) {
        self.shared.fetch.set_network_hook(hook);
    }

    /// Off, nothing goes to the network: requests answer from the HTTP
    /// cache and offline regions, stale or not, and fail without them.
    pub fn set_online(&self, online: bool) {
        self.shared.fetch.set_online(online);
    }

    pub fn online(&self) -> bool {
        self.shared.fetch.online()
    }

    /// What the runtime's device is and has done: for diagnostics and
    /// measurements.
    pub fn stats(&self) -> RuntimeStats {
        let gpu = self.shared.gpu();
        let caps = orb_hal::Gpu::caps(&*gpu);
        RuntimeStats {
            adapter: lock(&self.shared.adapter).clone(),
            counters: orb_hal::Gpu::counters(&*gpu),
            pipeline_time: gpu.pipeline_time(),
            pipeline_timings: gpu.pipeline_timings().to_vec(),
            hung_compile: gpu.hung().map(str::to_owned),
            live_resources: orb_hal::Gpu::live_resources(&*gpu),
            http_cache: self.shared.fetch.cache().map(|c| c.stats()),
            region_bytes: self
                .regions()
                .map_or(0, |r| r.iter().map(|r| r.bytes).sum()),
            max_texture_size: caps.limits.max_texture_dimension_2d,
            gpu_timestamps: caps.timestamp_query,
            timestamp_resolution: gpu.timestamp_resolution(),
        }
    }

    /// Makes the shader modules the device makes from now on unique to
    /// `nonce`, past every shader cache, for cold-start measurements
    /// (`WgpuGpu::set_shader_nonce`). A device made after a loss keeps
    /// none.
    pub fn set_shader_nonce(&self, nonce: Option<u32>) {
        self.shared.gpu().set_shader_nonce(nonce);
    }

    /// Times every submission from now on, for measurements:
    /// [`gpu_timings`](Self::gpu_timings) collects them.
    pub fn set_gpu_timing(&self, on: bool) {
        self.shared.timing.store(on, Ordering::Relaxed);
        self.shared.gpu().set_timing(on);
    }

    /// Each submission's times since the last call, in order, once the GPU
    /// has finished them: its CPU time, and its passes' GPU times where
    /// the device has timestamp queries. Blocks until the GPU is done. None
    /// from a lost device, whose wait wgpu would panic on.
    pub fn gpu_timings(&self) -> Vec<SubmitTimes> {
        if self.lost() {
            return Vec::new();
        }
        pollster::block_on(self.shared.gpu().take_timings())
    }

    /// Whether the GPU was lost; the next frame drawn makes another.
    pub fn lost(&self) -> bool {
        self.shared.gpu().lost()
    }

    /// Loses the GPU on purpose, to exercise recovery.
    pub fn lose_device(&self) {
        self.shared.gpu().lose();
    }

    /// Whether a lost device's replacement is due ([`BACKOFF`]): `false`
    /// while it waits, with every map woken when it's due. An error once the
    /// runtime has stopped making devices: after a run of losses past the
    /// backoff's, or for a driver that hung or failed compiling, which
    /// another device would do the same in.
    pub(crate) fn recovery_due(&self) -> Result<bool, Error> {
        let mut recovery = lock(&self.shared.recovery);
        if let Some(failed) = &recovery.failed {
            return Err(Error::Native(failed.clone()));
        }
        // A compile that lost the device names its pipeline. Running
        // compiles get a second to finish when the loss is first seen, not
        // on each check while the backoff runs.
        let wait = if recovery.due.is_none() {
            std::time::Duration::from_secs(1)
        } else {
            std::time::Duration::ZERO
        };
        let hung = {
            let mut gpu = self.shared.gpu();
            gpu.settle_loss(wait);
            gpu.hung().map(str::to_owned)
        };
        if let Some(hung) = hung {
            return Err(recovery.fail(format!("the GPU driver hung or failed making {hung}")));
        }
        let now = std::time::Instant::now();
        let due = match recovery.due {
            Some(due) => due,
            None => {
                let lived = now - recovery.made;
                recovery.losses = if lived < STABLE {
                    recovery.losses + 1
                } else {
                    1
                };
                let Some(&wait) = BACKOFF.get(recovery.losses as usize - 1) else {
                    let reason = (self.shared.gpu().lost_reason())
                        .map(|r| format!("; the last: {r}"))
                        .unwrap_or_default();
                    let why = format!(
                        "the GPU was lost {} times in a row, each device within {} s of \
                         the last, so no other is made{reason}",
                        recovery.losses,
                        STABLE.as_secs()
                    );
                    return Err(recovery.fail(why));
                };
                let due = now + wait;
                recovery.due = Some(due);
                if !wait.is_zero() {
                    let wake = wake_all(&self.shared.wakes);
                    std::thread::Builder::new()
                        .name("orbis device backoff".into())
                        .spawn(move || {
                            std::thread::sleep(wait);
                            wake();
                        })
                        .map_err(|e| Error::Native(format!("a backoff timer: {e}")))?;
                }
                due
            }
        };
        Ok(now >= due)
    }

    /// Makes a new device after a loss, once [`recovery_due`](Self::recovery_due),
    /// with the pipeline cache started again. Maps lay their tiles out
    /// again on their next frame from cached data; texture targets are
    /// made again, window targets fail with [`Error::TargetLost`].
    pub(crate) fn recover(&self) -> Result<(), Error> {
        let made = device::make();
        let mut recovery = lock(&self.shared.recovery);
        recovery.due = None;
        let (mut gpu, adapter) = made.map_err(|e| {
            recovery.fail(format!(
                "the GPU was lost and no other could be made: {}",
                e.message()
            ))
        })?;
        recovery.made = std::time::Instant::now();
        drop(recovery);
        device::set_compiler(
            &mut gpu,
            self.shared.executor.clone(),
            wake_all(&self.shared.wakes),
        );
        drop(std::mem::replace(&mut *self.shared.gpu(), gpu));
        *lock(&self.shared.adapter) = adapter;
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        // The maps' images were on the old device.
        self.shared.images.reset();
        *lock(&self.shared.cache) = None;
        self.start_pipeline_cache()?;
        self.set_gpu_timing(self.shared.timing.load(Ordering::Relaxed));
        Ok(())
    }

    /// Keeps compiled pipelines in the state directory's `pipelines/`
    /// between runs, so warm starts skip compiling them (Vulkan's pipeline
    /// cache, and D3D12's of the shader compiler's output; elsewhere the
    /// platform caches and this does nothing). Maps save the cache when
    /// they go idle with new pipelines.
    fn start_pipeline_cache(&self) -> Result<(), Error> {
        let Some(dir) = self
            .shared
            .options
            .state_dir
            .as_ref()
            .map(|d| d.join("pipelines"))
        else {
            return Ok(());
        };
        let mut device = self.shared.gpu();
        let Some(key) = device.pipeline_cache_key() else {
            return Ok(());
        };
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))?;
        let file = dir.join(key);
        let data = std::fs::read(&file).ok();
        device.start_pipeline_cache(data.as_deref());
        let made = orb_hal::Gpu::counters(&*device).pipelines_compiled;
        // The cache's lock goes before the device's, as in
        // `save_pipeline_cache`.
        drop(device);
        *lock(&self.shared.cache) = Some((file, made));
        Ok(())
    }

    /// Writes the pipeline cache if pipelines were made since it was last
    /// written and they changed its data.
    fn save_pipeline_cache(&self) -> Result<(), Error> {
        let mut cache = lock(&self.shared.cache);
        let Some((file, saved)) = cache.as_mut() else {
            return Ok(());
        };
        let mut device = self.shared.gpu();
        let made = orb_hal::Gpu::counters(&*device).pipelines_compiled;
        if made == *saved {
            return Ok(());
        }
        // `None`: the file holds this data already.
        if let Some(data) = device.pipeline_cache_data() {
            orb_host_native::write_atomically(file, &data)
                .map_err(|e| Error::Native(format!("{}: {e}", file.display())))?;
            device.pipeline_cache_saved(&data);
        }
        *saved = made;
        Ok(())
    }

    /// Downloads a style's map of an area between two zooms into the state
    /// directory, for maps to open with no network: its style, sources'
    /// tiles, glyphs and sprites. Progress, completion and failure arrive
    /// from [`poll_region_event`](Self::poll_region_event), with every
    /// map's wake. Returns the region's id.
    pub fn download_region(&self, definition: RegionDefinition) -> Result<u32, Error> {
        self.region_store()?
            .download(definition, wake_all(&self.shared.wakes))
    }

    /// Adds a region to the state directory without downloading it: listed
    /// incomplete, and served by no map.
    pub fn create_region(&self, definition: RegionDefinition) -> Result<RegionInfo, Error> {
        self.region_store()?.create(definition)
    }

    /// Copies the regions of another store into the state directory: an
    /// Orbis state directory's, or, with the `sqlite` feature, those of
    /// MapLibre Native's offline database file. A region whose definition
    /// (metadata included) matches one already here isn't copied again.
    /// Returns this directory's copy of each, and reads `source` only.
    pub fn merge_regions(&self, source: &std::path::Path) -> Result<Vec<RegionInfo>, Error> {
        self.region_store()?.merge(source)
    }

    /// Deletes a region, stopping its download.
    pub fn delete_region(&self, id: u32) -> Result<(), Error> {
        self.region_store()?.delete(id)
    }

    /// The regions in the state directory.
    pub fn regions(&self) -> Result<Vec<RegionInfo>, Error> {
        self.region_store()?.list()
    }

    /// The oldest region download event not yet taken.
    pub fn poll_region_event(&self) -> Option<RegionEvent> {
        self.shared.regions.as_ref()?.poll_event()
    }

    fn region_store(&self) -> Result<&regions::Regions, Error> {
        self.shared
            .regions
            .as_ref()
            .ok_or_else(|| Error::InvalidState("offline regions need a state directory".into()))
    }

    /// Empties the HTTP cache.
    pub fn clear_http_cache(&self) {
        if let Some(cache) = self.shared.fetch.cache() {
            cache.clear();
        }
    }

    /// Keeps the HTTP cache's entries but has each revalidated with its
    /// server before it is used again.
    pub fn invalidate_http_cache(&self) {
        if let Some(cache) = self.shared.fetch.cache() {
            cache.invalidate();
        }
    }

    /// Keeps at most `bytes` of response bodies in the HTTP cache (100 MiB,
    /// [`HTTP_CACHE_BYTES`], by default), evicting the least recently used;
    /// 0 keeps none. A runtime without a state directory keeps none anyway.
    pub fn set_http_cache_size(&self, bytes: u64) {
        if let Some(cache) = self.shared.fetch.cache() {
            cache.set_max_bytes(bytes);
        }
    }

    /// A surface on a window or view that wgpu accepts safely (a winit
    /// window), in physical pixels.
    pub fn create_surface(
        &self,
        window: impl Into<orb_hal_wgpu::wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Surface, Error> {
        self.shared
            .gpu()
            .create_surface(window, width, height)
            .map_err(Error::Unsupported)
    }

    /// Lets `f` make a surface from a raw platform handle with the device,
    /// such as a `CAMetalLayer` or an `ANativeWindow`
    /// ([`WgpuGpu::create_surface_raw`]).
    pub fn with_device<R>(&self, f: impl FnOnce(&WgpuGpu) -> R) -> R {
        f(&self.shared.gpu())
    }

    /// Wakes `wake` when a pipeline compiles, while it lives.
    fn add_wake(&self, wake: &Arc<dyn Fn() + Send + Sync>) {
        self.shared
            .wakes
            .lock()
            .expect("wakes poisoned")
            .push(Arc::downgrade(wake));
    }

    fn services(&self, wake: Arc<dyn Fn() + Send + Sync>) -> orb_runtime::Services {
        let fetch: Arc<dyn orb_host::Fetch> = match &self.shared.regions {
            Some(regions) => regions.fetch.clone(),
            None => self.shared.fetch.clone(),
        };
        let jobs = orb_runtime::PoolJobs::new(
            fetch.clone(),
            Arc::new(NativeImageDecode),
            self.shared.executor.clone(),
        );
        jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
        jobs.set_font_files(Arc::new(orb_host_native::FontFileFaces::new()));
        orb_runtime::Services {
            fetch,
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(NativeImageDecode),
            images: self.shared.images.clone(),
            jobs,
            wake,
        }
    }
}

// Hosts move these across threads (`orb-ffi`'s owner thread, a render
// thread of the host's), so losing `Send` is a breaking change. A runtime
// is shared by its maps, so it's `Sync` too; a map isn't (the scene keeps
// `RefCell`s), and needn't be, since its calls take turns behind `&mut`.
const _: () = {
    const fn send<T: Send>() {}
    const fn sync<T: Sync>() {}
    send::<Runtime>();
    sync::<Runtime>();
    send::<Map>();
    send::<Target>();
};

impl Shared {
    /// The device, for the call.
    fn gpu(&self) -> MutexGuard<'_, WgpuGpu> {
        lock(&self.gpu)
    }

    /// Counts devices made: maps and targets of an older one rebuild.
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// Locks `mutex`, past a panic that poisoned it: a caller that catches
/// a panic (a C boundary) goes on with the state as the panic left it,
/// as `RefCell` did.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The most response bodies the HTTP cache keeps: fresh ones load without
/// the network, stale ones are revalidated, and either loads when the
/// network fails, so a map seen before opens offline.
pub const HTTP_CACHE_BYTES: u64 = 100 << 20;

/// The HTTP cache in the state directory, if there is one.
fn http_cache(options: &RuntimeOptions) -> Result<Option<Arc<HttpCache>>, Error> {
    let Some(dir) = &options.state_dir else {
        return Ok(None);
    };
    let storage = Arc::new(orb_host_native::FileStorage::new(&dir.join("http")));
    HttpCache::open(storage, HTTP_CACHE_BYTES)
        .map(Some)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))
}
