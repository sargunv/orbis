//! The Rust embedding API for native hosts (plan §5.4, §14): a [`Runtime`]
//! owns a GPU device and the host services, a [`Map`] owns a style, its
//! camera store and its gestures, and a [`Target`] is what a map draws
//! into: a window surface or a texture the runtime owns.
//!
//! Everything here belongs to the thread that created the runtime (plan
//! §7.1's Owned mode with the host's thread as the scene thread): jobs,
//! fetches and placement run on the runtime's pools, and a map's
//! [`Map::set_wake`] callback tells the host, from any thread, that a frame
//! is needed. `orb-capi` exports this API to C, Kotlin and Swift.

#![forbid(unsafe_code)]

mod device;
mod map;
mod regions;
mod target;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

pub use map::{Event, Image, Map, MapOptions, STILL_TIMEOUT, TileRequest};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraTarget, Change, Constraints, EaseOptions, Easing,
    FitOptions, FlyOptions, FreeCamera, Padding, Phase,
};
pub use orb_crs::{Geodetic, Projection, TileIndex, TileMatrixSet};
pub use orb_hal::Tier;
pub use orb_hal_gl::egl::EglWindow;
pub use orb_hal_wgpu::{RawSurface, SubmitTimes};
pub use orb_host::{Request, RequestTransform};
pub use orb_interact::{Button, Feel, Input, Key, Modifiers, PointerKind, WheelUnit};
pub use orb_runtime::batteries::{Location, Marker};
pub use orb_runtime::{
    CameraState, ClusterQuery, FeatureTarget, FrameReport, HOST_TILES, Hit, HostTile, MapEvent, Op,
    StyleImage, Transaction, VideoRequest, json,
};
pub use regions::{RegionDefinition, RegionEvent, RegionInfo};
pub use target::Surface;
pub use target::Target;

use device::Device;
use orb_hal_wgpu::WgpuGpu;
pub use orb_host_native::CacheStats;
use orb_host_native::{HttpCache, NativeFetch, NativeImageDecode};
use orb_jobs::ThreadPool;

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
    /// The GPU tier to draw at.
    pub backend: Backend,
    /// Where the runtime keeps state between runs: the crash sentinel
    /// (plan §8.8), the driver's compiled pipelines (`pipelines/`), the
    /// HTTP cache (`http/`, [`HTTP_CACHE_BYTES`]) and offline regions.
    /// Without one, a device that crashed its driver last time is tried on
    /// Modern again, pipelines compile on every start, and nothing fetched
    /// is kept.
    pub state_dir: Option<PathBuf>,
    /// Android's API level (`Build.VERSION.SDK_INT`), for the allowlist.
    pub android_api_level: u32,
    /// The libEGL Compat loads (ANGLE's, say); the platform's by default.
    pub egl_library: Option<String>,
}

/// Which GPU tier a runtime draws at (plan §4.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Modern where the device policy allows it (the Android allowlist,
    /// the crash sentinel) and a Modern adapter exists, else Compat.
    #[default]
    Auto,
    /// WebGPU-class (wgpu: Vulkan, Metal, D3D12).
    Modern,
    /// GLES 3.0 through EGL.
    Compat,
}

/// A GPU device and host services, shared by the maps made from it. A
/// runtime and its maps and targets belong to the thread that made it,
/// which needn't be the main thread: an embedder draws on a render thread
/// of its own by making them there. On macOS a `CAMetalLayer` inside a
/// window's view must still present from the main thread: wgpu-hal asks
/// the view for its window when it acquires a frame (#35).
#[derive(Clone, Debug)]
pub struct Runtime {
    shared: Rc<Shared>,
}

#[derive(Debug)]
struct Shared {
    gpu: RefCell<Device>,
    options: RuntimeOptions,
    /// Why the device is the tier it is.
    selection: RefCell<String>,
    /// Counts devices made: maps and targets of an older one rebuild.
    generation: std::cell::Cell<u64>,
    /// The crash sentinel's marker while armed.
    sentinel: RefCell<Option<PathBuf>>,
    /// Fetches, through the HTTP cache with a state directory.
    fetch: Arc<NativeFetch>,
    /// Offline regions, with a state directory.
    regions: Option<regions::Regions>,
    /// Tile jobs and pipeline compiles.
    executor: Arc<ThreadPool>,
    images: orb_runtime::Images,
    /// Where the pipeline cache is kept, and the pipeline count it was
    /// last saved at.
    cache: RefCell<Option<(PathBuf, u32)>>,
    /// Every map's wake: the device calls them all when a pipeline
    /// compiles.
    wakes: Wakes,
    /// Whether the device wakes the maps when a pipeline compiles.
    compile_wakes: std::cell::Cell<bool>,
    /// Whether submissions are timed ([`Runtime::set_gpu_timing`]).
    timing: std::cell::Cell<bool>,
}

/// What a runtime's device is and has done ([`Runtime::stats`]).
#[derive(Clone, Debug)]
pub struct RuntimeStats {
    /// The tier the runtime draws at.
    pub tier: Tier,
    /// Why the device is the tier it is: "Modern: …" or "Compat: …".
    pub selection: String,
    /// Render and compute pipelines the device has made so far.
    pub pipelines_created: u32,
    /// Time spent making shader modules and pipelines so far.
    pub pipeline_time: std::time::Duration,
    /// Each pipeline compiled so far and how long it took, in the order
    /// they finished (plan §8.9: pipelines created cold and warm).
    pub pipeline_timings: Vec<(String, std::time::Duration)>,
    /// The compile the watchdog found hung, or the pipeline the driver
    /// couldn't make, if one was (plan §8.8 3): a Modern GPU is then lost,
    /// and [`Runtime::recover`] demotes it.
    pub hung_compile: Option<String>,
    /// What the HTTP cache holds, if there is one.
    pub http_cache: Option<CacheStats>,
    /// The widest and tallest texture target the device makes, in pixels.
    pub max_texture_size: u32,
    /// Bytes written to the device's buffers and textures so far.
    pub uploaded_bytes: u64,
    /// Whether [`Runtime::gpu_timings`] times passes on the GPU: the
    /// device has timestamp queries.
    pub gpu_timestamps: bool,
    /// The GPU timestamps' resolution, once timing has seen a step.
    pub timestamp_resolution: Option<std::time::Duration>,
    /// GPU resources alive: buffers, textures, pipelines and the rest.
    pub live_resources: usize,
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

impl Runtime {
    /// Makes a device on the default adapter and starts the job and fetch
    /// pools. Blocks while the device is made.
    pub fn new(options: &RuntimeOptions) -> Result<Self, Error> {
        let (mut gpu, selection, sentinel) = Device::choose(options)?;
        let wakes = Arc::default();
        let threads = options
            .job_threads
            .unwrap_or_else(ThreadPool::default_size)
            .max(1);
        let executor = Arc::new(ThreadPool::new("jobs", threads));
        let compile_wakes = gpu.set_compiler(executor.clone(), wake_all(&wakes));
        let fetch = NativeFetch::with_cache(http_cache(options)?);
        let runtime = Self {
            shared: Rc::new(Shared {
                gpu: RefCell::new(gpu),
                options: options.clone(),
                selection: RefCell::new(selection),
                generation: std::cell::Cell::new(0),
                sentinel: RefCell::new(sentinel),
                fetch: fetch.clone(),
                regions: options
                    .state_dir
                    .as_ref()
                    .map(|dir| regions::Regions::new(dir, fetch)),
                executor,
                images: Default::default(),
                cache: RefCell::new(None),
                wakes,
                compile_wakes: std::cell::Cell::new(compile_wakes),
                timing: std::cell::Cell::new(false),
            }),
        };
        runtime.start_pipeline_cache()?;
        Ok(runtime)
    }

    /// Rewrites every request the runtime's maps and region downloads make
    /// from here on (GL JS `transformRequest`): its URL, before the HTTP
    /// cache, and headers to send, such as a tile server's token; `None`
    /// from the transform refuses the request.
    pub fn set_request_transform(&self, transform: Option<Arc<orb_host::RequestTransform>>) {
        self.shared.fetch.set_transform(transform);
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
        let gpu = self.shared.gpu.borrow();
        let (pipeline_time, pipeline_timings) = match &*gpu {
            Device::Modern(gpu) => (gpu.pipeline_time(), gpu.pipeline_timings().to_vec()),
            Device::Compat(c) => (c.gpu.pipeline_time(), c.gpu.pipeline_timings().to_vec()),
        };
        RuntimeStats {
            tier: gpu.tier(),
            selection: self.shared.selection.borrow().clone(),
            pipelines_created: gpu.pipelines_created(),
            pipeline_time,
            pipeline_timings,
            hung_compile: gpu.hung(),
            live_resources: match &*gpu {
                Device::Modern(gpu) => orb_hal::Gpu::live_resources(gpu),
                Device::Compat(c) => orb_hal::Gpu::live_resources(&c.gpu),
            },
            http_cache: self.shared.fetch.cache().map(|c| c.stats()),
            max_texture_size: match &*gpu {
                Device::Modern(gpu) => orb_hal::Gpu::caps(gpu),
                Device::Compat(c) => orb_hal::Gpu::caps(&c.gpu),
            }
            .limits
            .max_texture_dimension_2d,
            uploaded_bytes: match &*gpu {
                Device::Modern(gpu) => {
                    let u = gpu.uploaded();
                    u.from_core + u.from_js + u.texels_from_core + u.texels_from_js
                }
                Device::Compat(c) => c.gpu.uploaded(),
            },
            gpu_timestamps: match &*gpu {
                Device::Modern(gpu) => orb_hal::Gpu::caps(gpu).timestamp_query,
                Device::Compat(_) => false,
            },
            timestamp_resolution: match &*gpu {
                Device::Modern(gpu) => gpu.timestamp_resolution(),
                Device::Compat(_) => None,
            },
        }
    }

    /// Times every submission from now on, for measurements:
    /// [`gpu_timings`](Self::gpu_timings) collects them. Modern only.
    pub fn set_gpu_timing(&self, on: bool) {
        self.shared.timing.set(on);
        if let Device::Modern(gpu) = &mut *self.shared.gpu.borrow_mut() {
            gpu.set_timing(on);
        }
    }

    /// Each submission's times since the last call, in order, once the GPU
    /// has finished them: its CPU time, and its passes' GPU times where
    /// the device has timestamp queries. Blocks until the GPU is done.
    pub fn gpu_timings(&self) -> Vec<SubmitTimes> {
        match &mut *self.shared.gpu.borrow_mut() {
            Device::Modern(gpu) => pollster::block_on(gpu.take_timings()),
            Device::Compat(_) => Vec::new(),
        }
    }

    /// Whether the GPU was lost; the next frame drawn makes another.
    pub fn lost(&self) -> bool {
        self.shared.gpu.borrow().lost()
    }

    /// Loses the GPU on purpose, to exercise recovery.
    pub fn lose_device(&self) {
        self.shared.gpu.borrow_mut().lose();
    }

    /// Makes a new device after a loss (plan §8.8), on the tier the
    /// policy chooses again, with the pipeline cache started again. Maps
    /// lay their tiles out again on their next frame from cached data;
    /// texture targets are made again, window targets fail with
    /// [`Error::TargetLost`].
    pub(crate) fn recover(&self) -> Result<Tier, Error> {
        // A loss mid-compile isn't a crash in it: only a compile the
        // watchdog found hung keeps the sentinel armed, so the device
        // chosen now demotes as at the next start (plan §8.8 3).
        let hung = self.shared.gpu.borrow().hung();
        if let Some(marker) = self.shared.sentinel.borrow_mut().take()
            && hung.is_none()
        {
            device::disarm(&marker);
        }
        let (mut gpu, selection, sentinel) = Device::choose(&self.shared.options)?;
        self.shared
            .compile_wakes
            .set(gpu.set_compiler(self.shared.executor.clone(), wake_all(&self.shared.wakes)));
        drop(std::mem::replace(&mut *self.shared.gpu.borrow_mut(), gpu));
        self.shared.gpu.borrow().make_current();
        *self.shared.selection.borrow_mut() = selection;
        *self.shared.sentinel.borrow_mut() = sentinel;
        self.shared.generation.set(self.shared.generation.get() + 1);
        // The maps' images were on the old device.
        self.shared.images.reset();
        *self.shared.cache.borrow_mut() = None;
        self.start_pipeline_cache()?;
        self.set_gpu_timing(self.shared.timing.get());
        Ok(self.shared.gpu.borrow().tier())
    }

    /// Keeps the driver's compiled pipelines in the state directory's
    /// `pipelines/` between runs, so warm starts skip compiling them
    /// (Vulkan's pipeline cache, GLES's program binaries; elsewhere the
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
        let mut device = self.shared.gpu.borrow_mut();
        let key = match &*device {
            Device::Modern(gpu) => gpu.pipeline_cache_key(),
            // Compat keeps program binaries (GLES 3.0) the same way.
            Device::Compat(c) => Some(c.gpu.program_cache_key()),
        };
        let Some(key) = key else {
            return Ok(());
        };
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))?;
        let file = dir.join(key);
        let data = std::fs::read(&file).ok();
        match &mut *device {
            Device::Modern(gpu) => gpu.start_pipeline_cache(data.as_deref()),
            Device::Compat(c) => c.gpu.start_program_cache(data.as_deref()),
        }
        *self.shared.cache.borrow_mut() = Some((file, device.pipelines_created()));
        Ok(())
    }

    /// Writes the pipeline cache if pipelines were made since it was last
    /// written.
    fn save_pipeline_cache(&self) -> Result<(), Error> {
        let mut cache = self.shared.cache.borrow_mut();
        let Some((file, saved)) = cache.as_mut() else {
            return Ok(());
        };
        let mut device = self.shared.gpu.borrow_mut();
        let made = device.pipelines_created();
        if made == *saved {
            return Ok(());
        }
        let data = match &mut *device {
            Device::Modern(gpu) => gpu.pipeline_cache_data(),
            Device::Compat(c) => c.gpu.program_cache_data(),
        };
        let Some(data) = data else {
            return Ok(());
        };
        orb_host_native::write_atomically(file, &data)
            .map_err(|e| Error::Native(format!("{}: {e}", file.display())))?;
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

    /// A surface on a window or view that wgpu accepts safely (a winit
    /// window), in physical pixels.
    pub fn create_surface(
        &self,
        window: impl Into<orb_hal_wgpu::wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Surface, Error> {
        match &*self.shared.gpu.borrow() {
            Device::Modern(gpu) => gpu
                .create_surface(window, width, height)
                .map(Surface::Modern)
                .map_err(Error::Unsupported),
            Device::Compat(_) => Err(Error::Unsupported(
                "Compat draws into raw native windows only".into(),
            )),
        }
    }

    /// Lets `f` make a surface from a raw platform handle with the device;
    /// `orb-capi` does so for `CAMetalLayer`s and `ANativeWindow`s: with
    /// wgpu on Modern, through EGL on Compat.
    pub fn with_device<R>(&self, f: impl FnOnce(DeviceRef<'_>) -> R) -> R {
        match &*self.shared.gpu.borrow() {
            Device::Modern(gpu) => f(DeviceRef::Modern(gpu)),
            Device::Compat(c) => f(DeviceRef::Compat(&c.egl)),
        }
    }

    /// Wakes `wake` when a pipeline compiles, while it lives.
    fn add_wake(&self, wake: &Arc<dyn Fn() + Send + Sync>) {
        self.shared
            .wakes
            .lock()
            .expect("wakes poisoned")
            .push(Arc::downgrade(wake));
    }

    /// Whether the device wakes the maps when a pipeline compiles, so a
    /// map waiting only for pipelines needs no frames.
    fn compile_wakes(&self) -> bool {
        self.shared.compile_wakes.get()
    }

    /// Clears the crash sentinel once the pipelines are made.
    fn pipelines_made(&self) {
        if let Some(marker) = self.shared.sentinel.borrow_mut().take() {
            device::disarm(&marker);
        }
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

/// The device a raw surface is made with.
#[derive(Debug)]
pub enum DeviceRef<'a> {
    Modern(&'a WgpuGpu),
    Compat(&'a orb_hal_gl::egl::EglContext),
}
