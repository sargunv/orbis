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
mod target;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

pub use map::{Event, Map, MapOptions, TileRequest};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraTarget, Change, EaseOptions, Easing, FitOptions,
    FlyOptions, Padding, Phase,
};
pub use orb_crs::TileMatrixSet;
pub use orb_geo::Geodetic;
pub use orb_hal::Tier;
pub use orb_hal_gl::egl::EglWindow;
pub use orb_hal_wgpu::RawSurface;
pub use orb_interact::{Button, Feel, Input, Key, Modifiers, PointerKind, WheelUnit};
pub use orb_runtime::{CameraState, FrameReport, Hit, MapEvent, StyleImage};
pub use target::Surface;
pub use target::Target;

use device::Device;
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
    /// The GPU was lost: [`Runtime::recover`] makes another.
    DeviceLost,
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(m)
            | Self::InvalidState(m)
            | Self::Unsupported(m)
            | Self::Native(m)
            | Self::TargetLost(m) => m,
            Self::DeviceLost => "the GPU was lost",
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
    /// Where the runtime keeps device state between runs: the crash
    /// sentinel (plan §8.8). Without one, a device that crashed its driver
    /// last time is tried on Modern again.
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

/// A GPU device and host services, shared by the maps made from it.
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
    fetch: Arc<NativeFetch>,
    executor: Arc<ThreadPool>,
    images: orb_runtime::Images,
    /// The gesture and animation clock's origin.
    started: Instant,
    /// Where the pipeline cache is kept, and the pipeline count it was
    /// last saved at.
    cache: RefCell<Option<(PathBuf, u32)>>,
    /// The directory the host asked to keep pipeline caches in.
    cache_dir: RefCell<Option<PathBuf>>,
    /// Every map's wake: the device calls them all when a pipeline
    /// compiles.
    wakes: Wakes,
    /// Whether the device wakes the maps when a pipeline compiles.
    compile_wakes: std::cell::Cell<bool>,
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
        let (gpu, selection, sentinel) = Device::choose(options)?;
        let wakes = Arc::default();
        let compile_wakes = gpu.set_compile_wake(wake_all(&wakes));
        let threads = options
            .job_threads
            .unwrap_or_else(ThreadPool::default_size)
            .max(1);
        Ok(Self {
            shared: Rc::new(Shared {
                gpu: RefCell::new(gpu),
                options: options.clone(),
                selection: RefCell::new(selection),
                generation: std::cell::Cell::new(0),
                sentinel: RefCell::new(sentinel),
                fetch: NativeFetch::new(),
                executor: Arc::new(ThreadPool::new("jobs", threads)),
                images: Default::default(),
                started: Instant::now(),
                cache: RefCell::new(None),
                cache_dir: RefCell::new(None),
                wakes,
                compile_wakes: std::cell::Cell::new(compile_wakes),
            }),
        })
    }

    /// Render and compute pipelines the device has made so far.
    pub fn pipelines_created(&self) -> u32 {
        self.shared.gpu.borrow().pipelines_created()
    }

    /// Time spent making shader modules and pipelines so far.
    pub fn pipeline_time(&self) -> std::time::Duration {
        match &*self.shared.gpu.borrow() {
            Device::Modern(gpu) => gpu.pipeline_time(),
            Device::Compat(c) => c.gpu.pipeline_time(),
        }
    }

    /// Each pipeline compiled so far and how long it took, in the order
    /// they finished (plan §8.9: pipelines created cold and warm).
    pub fn pipeline_timings(&self) -> Vec<(String, std::time::Duration)> {
        match &*self.shared.gpu.borrow() {
            Device::Modern(gpu) => gpu.pipeline_timings().to_vec(),
            Device::Compat(c) => c.gpu.pipeline_timings().to_vec(),
        }
    }

    /// The tier the runtime draws at.
    pub fn tier(&self) -> Tier {
        self.shared.gpu.borrow().tier()
    }

    /// Why the device is the tier it is, for diagnostics: "Modern: …" or
    /// "Compat: …".
    pub fn selection(&self) -> String {
        self.shared.selection.borrow().clone()
    }

    /// Whether the GPU was lost; [`recover`](Self::recover) makes another.
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
    /// texture targets are made again. Drop window targets first (their
    /// surfaces hold their windows) and make them again after.
    pub fn recover(&self) -> Result<Tier, Error> {
        // A loss mid-compile isn't a crash in it: only a compile the
        // watchdog found hung keeps the sentinel armed, so the device
        // chosen now demotes as at the next start (plan §8.8 3).
        let hung = self.shared.gpu.borrow().hung();
        if let Some(marker) = self.shared.sentinel.borrow_mut().take()
            && hung.is_none()
        {
            device::disarm(&marker);
        }
        let (gpu, selection, sentinel) = Device::choose(&self.shared.options)?;
        self.shared
            .compile_wakes
            .set(gpu.set_compile_wake(wake_all(&self.shared.wakes)));
        drop(std::mem::replace(&mut *self.shared.gpu.borrow_mut(), gpu));
        self.shared.gpu.borrow().make_current();
        *self.shared.selection.borrow_mut() = selection;
        *self.shared.sentinel.borrow_mut() = sentinel;
        self.shared.generation.set(self.shared.generation.get() + 1);
        *self.shared.cache.borrow_mut() = None;
        let dir = self.shared.cache_dir.borrow().clone();
        if let Some(dir) = dir {
            self.set_pipeline_cache_dir(&dir)?;
        }
        Ok(self.tier())
    }

    /// Keeps the driver's compiled pipelines in `dir` between runs, so
    /// warm starts skip compiling them (Vulkan's pipeline cache, GLES's
    /// program binaries; elsewhere the platform caches and this does
    /// nothing). Call before the first frame; maps
    /// save the cache when they go idle with new pipelines.
    pub fn set_pipeline_cache_dir(&self, dir: &Path) -> Result<(), Error> {
        *self.shared.cache_dir.borrow_mut() = Some(dir.to_owned());
        let mut device = self.shared.gpu.borrow_mut();
        let key = match &*device {
            Device::Modern(gpu) => gpu.pipeline_cache_key(),
            // Compat keeps program binaries (GLES 3.0) the same way.
            Device::Compat(c) => Some(c.gpu.program_cache_key()),
        };
        let Some(key) = key else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)
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
    pub fn save_pipeline_cache(&self) -> Result<(), Error> {
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
        write_atomically(file, &data)
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

    /// The compile the watchdog found hung, if one was (plan §8.8 3): the
    /// GPU is then lost, and [`recover`](Self::recover) demotes it.
    pub fn hung_compile(&self) -> Option<String> {
        self.shared.gpu.borrow().hung()
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

/// The device a raw surface is made with.
#[derive(Debug)]
pub enum DeviceRef<'a> {
    Modern(&'a WgpuGpu),
    Compat(&'a orb_hal_gl::egl::EglContext),
}

/// Writes `data` to a file of its own beside `path`, then renames it over
/// `path`: readers see the old file or the new one whole, and writers in
/// other runtimes or processes never share the temporary file.
fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(format!(".{}.{n}.partial", std::process::id()));
    let partial = path.with_file_name(name);
    let written = std::fs::write(&partial, data).and_then(|()| std::fs::rename(&partial, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
}

#[cfg(test)]
mod tests {
    #[test]
    fn concurrent_writers_publish_whole_files() {
        let dir = std::env::temp_dir().join(format!("orb-api-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache");
        let writers: Vec<_> = (0..8u8)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        super::write_atomically(&path, &vec![i; 64 << 10]).unwrap();
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 64 << 10);
        assert!(data.iter().all(|&b| b == data[0]), "bytes of two writers");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "partial files left"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
