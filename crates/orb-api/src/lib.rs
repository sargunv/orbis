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

mod map;
mod target;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

pub use map::{Event, Map, MapOptions};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraTarget, Change, EaseOptions, Easing, FitOptions,
    FlyOptions, Padding, Phase,
};
pub use orb_geo::Geodetic;
pub use orb_hal_wgpu::{RawSurface, Surface};
pub use orb_interact::{Button, Feel, Input, Key, Modifiers, PointerKind, WheelUnit};
pub use orb_runtime::{CameraState, FrameReport, Hit, MapEvent};
pub use target::Target;

use orb_hal_wgpu::WgpuGpu;
use orb_host_native::{NativeFetch, NativeImageDecode};
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
    /// The target can't be drawn to again.
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
}

/// A GPU device and host services, shared by the maps made from it.
#[derive(Clone, Debug)]
pub struct Runtime {
    shared: Rc<Shared>,
}

#[derive(Debug)]
struct Shared {
    gpu: RefCell<WgpuGpu>,
    fetch: Arc<NativeFetch>,
    executor: Arc<ThreadPool>,
    images: orb_runtime::Images,
    /// The gesture and animation clock's origin.
    started: Instant,
    /// Where the pipeline cache is kept, and the pipeline count it was
    /// last saved at.
    cache: RefCell<Option<(PathBuf, u32)>>,
}

impl Runtime {
    /// Makes a device on the default adapter and starts the job and fetch
    /// pools. Blocks while the device is made.
    pub fn new(options: &RuntimeOptions) -> Result<Self, Error> {
        let gpu = pollster::block_on(WgpuGpu::offscreen()).map_err(Error::Native)?;
        let threads = options
            .job_threads
            .unwrap_or_else(ThreadPool::default_size)
            .max(1);
        Ok(Self {
            shared: Rc::new(Shared {
                gpu: RefCell::new(gpu),
                fetch: NativeFetch::new(),
                executor: Arc::new(ThreadPool::new("jobs", threads)),
                images: Default::default(),
                started: Instant::now(),
                cache: RefCell::new(None),
            }),
        })
    }

    /// Render and compute pipelines the device has made so far.
    pub fn pipelines_created(&self) -> u32 {
        self.shared.gpu.borrow().pipelines_created()
    }

    /// Time spent making shader modules and pipelines so far.
    pub fn pipeline_time(&self) -> std::time::Duration {
        self.shared.gpu.borrow().pipeline_time()
    }

    /// Keeps the driver's compiled pipelines in `dir` between runs, so
    /// warm starts skip compiling them (Vulkan; elsewhere the platform
    /// caches and this does nothing). Call before the first frame; maps
    /// save the cache when they go idle with new pipelines.
    pub fn set_pipeline_cache_dir(&self, dir: &Path) -> Result<(), Error> {
        let mut gpu = self.shared.gpu.borrow_mut();
        let Some(key) = gpu.pipeline_cache_key() else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))?;
        let file = dir.join(key);
        let data = std::fs::read(&file).ok();
        gpu.start_pipeline_cache(data.as_deref());
        *self.shared.cache.borrow_mut() = Some((file, gpu.pipelines_created()));
        Ok(())
    }

    /// Writes the pipeline cache if pipelines were made since it was last
    /// written.
    pub fn save_pipeline_cache(&self) -> Result<(), Error> {
        let mut cache = self.shared.cache.borrow_mut();
        let Some((file, saved)) = cache.as_mut() else {
            return Ok(());
        };
        let gpu = self.shared.gpu.borrow();
        let made = gpu.pipelines_created();
        if made == *saved {
            return Ok(());
        }
        let Some(data) = gpu.pipeline_cache_data() else {
            return Ok(());
        };
        // Written aside and renamed, so a reader never sees half a file.
        let partial = file.with_extension("partial");
        std::fs::write(&partial, data)
            .and_then(|()| std::fs::rename(&partial, &*file))
            .map_err(|e| Error::Native(format!("{}: {e}", file.display())))?;
        *saved = made;
        Ok(())
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
            .gpu
            .borrow()
            .create_surface(window, width, height)
            .map_err(Error::Unsupported)
    }

    /// Lets `f` make a surface from a raw platform handle with the device;
    /// `orb-capi` does so for `CAMetalLayer`s and `ANativeWindow`s.
    pub fn with_gpu<R>(&self, f: impl FnOnce(&WgpuGpu) -> R) -> R {
        f(&self.shared.gpu.borrow())
    }

    fn services(&self, wake: Arc<dyn Fn() + Send + Sync>) -> orb_runtime::Services {
        let jobs = orb_runtime::PoolJobs::new(
            self.shared.fetch.clone(),
            Arc::new(NativeImageDecode),
            self.shared.executor.clone(),
        );
        jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
        jobs.set_font_files(Arc::new(orb_host_native::FontFileFaces::new()));
        orb_runtime::Services {
            fetch: self.shared.fetch.clone(),
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(NativeImageDecode),
            images: self.shared.images.clone(),
            jobs,
            wake,
        }
    }

    /// Milliseconds on the runtime's gesture and animation clock.
    fn now(&self) -> f64 {
        self.shared.started.elapsed().as_secs_f64() * 1000.0
    }
}
