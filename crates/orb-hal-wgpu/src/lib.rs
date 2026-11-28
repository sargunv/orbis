//! [`orb_hal::Gpu`] on wgpu: Vulkan, Metal, D3D12 and GLES natively, chosen
//! at runtime; in the browser, WebGPU or WebGL 2, chosen by the build
//! (the `webgpu` and `gl` features).
//!
//! [`WgpuGpu`] owns every wgpu handle behind the ids it hands out. Hosts
//! register their own targets: a surface's frame with
//! [`WgpuGpu::register_surface`], a view of a texture of theirs with
//! [`WgpuGpu::register_view`].

// `unsafe` only where a surface or a host's texture is made from a raw
// platform handle, and where D3D12's compiler or the Vulkan loader is
// loaded.
#![deny(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod compile;
mod convert;
#[cfg(windows)]
mod dx12;
mod ledger;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
mod offscreen;
mod slots;
#[cfg(not(target_arch = "wasm32"))]
mod staging;
mod surface;
mod texture_writes;
mod timing;
mod wait;
mod writes;

#[cfg(not(target_arch = "wasm32"))]
pub use compile::CompileHold;
pub use convert::from_texture_format;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use metal::{MetalTexture, metal_texture};
use orb_hal::*;
use slots::Slots;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::{RawSurface, Surface};
pub use surface::{SharedSurface, SurfaceError};
pub use timing::SubmitTimes;
pub use wait::Lost;
/// The wgpu this HAL is built on, for hosts that hand it windows.
pub use wgpu;

/// The largest buffer, and storage binding, Orbis asks a device for.
const LARGEST_BUFFER: u64 = 1 << 30;

/// The bytes of a texture's first level GL takes
/// ([`Limits::max_texture_bytes`]), over the layers wgpu makes it with
/// ([`gl_array_layers`]): below what Chrome's ANGLE allocates on Metal
/// (1.25 GiB on an M6) and on GL (2 GiB).
const GL_TEXTURE_BYTES: u64 = 1 << 30;

/// Textures from this many bytes up are ones GL is asked about after it
/// makes them ([`Gpu::try_create_texture`]): large enough that a driver
/// may refuse one, and few, so the synchronous query (a round trip to
/// Chrome's GPU process) is rare.
const CHECKED_TEXTURE_BYTES: u64 = 64 << 20;

/// The passes one command encoder takes: a submission of more goes to the
/// queue in parts. wgpu-core records each pass in a command buffer of its
/// own, with another for the barriers before it, and its Metal back end
/// loses the device rather than hold more than 4096 command buffers made
/// and not yet submitted, which a frame that redraws every drape slot of
/// a steep view passes where it draws their mip levels (WebGL 2's frame
/// driver: 11 render passes and 10 copies a slot, 3000 passes in all).
/// The copies and debug markers between two passes share one, so 1024
/// passes make at most 3072, and the uploads and readback a few more.
const PASSES_PER_ENCODER: usize = 1024;

/// The frame path a native device takes at most ([`FramePath`]): `gpu`
/// (the default), `cpu`, `cpu-tex` or `webgl2`. `cpu-tex` and `webgl2`
/// also have the device offer only the storage bindings their path fits
/// in, so validation holds the pipelines to them.
#[cfg(not(target_arch = "wasm32"))]
const DRIVER: &str = "ORB_DRIVER";

/// The storage bindings a stage takes at most with the scene's tables in
/// storage buffers (the GPU frame driver's cull and fetches, the compute
/// placement): WebGPU's defaults, 8 storage buffers a stage, the most
/// [`device_limits`] asks for, so every test holds the pipelines to them.
const STORAGE_TABLES: wgpu::Limits = wgpu::Limits::defaults();

/// The storage bindings a stage takes at most with the tables in data
/// textures, with compute: downlevel's, Vulkan's least (4 storage buffers
/// and 4 storage textures a stage).
const TEXTURE_TABLES: wgpu::Limits = wgpu::Limits::downlevel_defaults();

/// WebGL 2's storage bindings: none.
#[cfg(not(target_arch = "wasm32"))]
const NO_COMPUTE: wgpu::Limits = wgpu::Limits::downlevel_webgl2_defaults();

/// The frame path a device with `flags` and `limits` takes: compute needs
/// compute shaders and the storage bindings of [`TEXTURE_TABLES`]; tables
/// in storage buffers need the vertex stage's storage and
/// [`STORAGE_TABLES`]; the GPU-driven frame needs indirect draws too. GL
/// takes WebGL 2's, even where it offers compute (#140).
fn frame_path(flags: wgpu::DownlevelFlags, limits: &wgpu::Limits, gl: bool) -> FramePath {
    use wgpu::DownlevelFlags as F;
    let fits = |floor: &wgpu::Limits| {
        limits.max_storage_buffers_per_shader_stage >= floor.max_storage_buffers_per_shader_stage
            && limits.max_storage_textures_per_shader_stage
                >= floor.max_storage_textures_per_shader_stage
    };
    if gl || !flags.contains(F::COMPUTE_SHADERS) || !fits(&TEXTURE_TABLES) {
        FramePath::Webgl2
    } else if !flags.contains(F::VERTEX_STORAGE) || !fits(&STORAGE_TABLES) {
        FramePath::CpuTex
    } else if !flags.contains(F::INDIRECT_EXECUTION) {
        FramePath::Cpu
    } else {
        FramePath::Gpu
    }
}

/// The frame path `ORB_DRIVER` asks of native devices at most.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn driver_from_env() -> FramePath {
    match std::env::var(DRIVER).as_deref() {
        Ok("" | "gpu") | Err(_) => FramePath::Gpu,
        Ok("cpu") => FramePath::Cpu,
        Ok("cpu-tex") => FramePath::CpuTex,
        Ok("webgl2") => FramePath::Webgl2,
        Ok(other) => panic!("{DRIVER}={other}: expected gpu, cpu, cpu-tex or webgl2"),
    }
}

/// `limits` lowered to the storage bindings a device that takes `path` at
/// most binds, where it keeps the tables in textures.
#[cfg(not(target_arch = "wasm32"))]
fn lower_storage(path: FramePath, limits: &mut wgpu::Limits) {
    let floor = match path {
        FramePath::Gpu | FramePath::Cpu => return,
        FramePath::CpuTex => &TEXTURE_TABLES,
        FramePath::Webgl2 => &NO_COMPUTE,
    };
    let buffers = &mut limits.max_storage_buffers_per_shader_stage;
    *buffers = (*buffers).min(floor.max_storage_buffers_per_shader_stage);
    let textures = &mut limits.max_storage_textures_per_shader_stage;
    *textures = (*textures).min(floor.max_storage_textures_per_shader_stage);
}

/// A native device's 2D texture side, lowered to this to stand in for a
/// device that offers no more ([`Limits::max_texture_dimension_2d`]): at
/// `4096`, the terrain drape's pages hold a quarter of the slots a layer.
#[cfg(not(target_arch = "wasm32"))]
const TEXTURE_SIDE: &str = "ORB_TEXTURE_SIDE";

/// The features Orbis asks `adapter`'s device for, those it has of them:
/// natively, a pipeline cache, which warm starts reuse pipelines through;
/// timestamps, which time passes for measurements; clip distances, which
/// clip cut buildings without discarding; depth as floats beside a
/// stencil (`Caps::depth32_stencil8`); and on GL, passthrough shaders,
/// which take the build's GLSL (`ShaderModuleDesc::glsl`).
fn device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let mut wanted = wgpu::Features::PIPELINE_CACHE
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::CLIP_DISTANCES
        | wgpu::Features::DEPTH32FLOAT_STENCIL8;
    if !cfg!(webgpu) && adapter.get_info().backend == wgpu::Backend::Gl {
        wanted |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    adapter.features() & wanted
}

/// The limits Orbis asks `adapter`'s device for: WebGPU's defaults, lowered
/// where the adapter falls short of them (the iOS simulator has 15
/// inter-stage variables), with 2D textures as wide as it allows, and
/// buffers and storage bindings as large as it allows up to 1 GiB. A
/// frame is as wide as the device's textures (16384 on most GPUs, past
/// WebGPU's default 8192: a 1792 × 256 still at ratio 5), while what the
/// renderer keeps sizes itself. The scene's pools grow within one storage
/// binding to what the tiles in view need: at WebGPU's default 128 MiB, a
/// pitched view over a dense city ran out.
pub fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let offered = adapter.limits();
    let mut limits = wgpu::Limits::default().or_worse_values_from(&offered);
    limits.max_texture_dimension_2d = offered.max_texture_dimension_2d;
    limits.max_buffer_size =
        (offered.max_buffer_size.min(LARGEST_BUFFER)).max(limits.max_buffer_size);
    limits.max_storage_buffer_binding_size = (offered.max_storage_buffer_binding_size)
        .min(LARGEST_BUFFER as _)
        .max(limits.max_storage_buffer_binding_size);
    limits
}

/// The wgpu device and queue, and every resource created through them.
#[derive(Debug)]
pub struct WgpuGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The instance and adapter behind the device, when it was made here,
    /// for surfaces.
    #[cfg(not(target_arch = "wasm32"))]
    native: Option<(wgpu::Instance, wgpu::Adapter)>,
    /// Pipelines are made through it once started (natively, Vulkan and
    /// D3D12).
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_cache: Option<wgpu::PipelineCache>,
    /// The SHA-256 of the cache data the host keeps, as far as this device
    /// knows: what it started from, or last saw saved.
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_cache_kept: Option<[u8; 32]>,
    caps: Caps,
    /// The device is wgpu's GL back end (WebGL 2, GLES).
    gl: bool,
    /// The device is a GL one that takes GLSL ES (WebGL 2, OpenGL ES):
    /// pipelines compile the build's GLSL there (`ShaderModuleDesc::glsl`).
    glsl_es: bool,
    /// Texture usages the platform defines. Browsers before
    /// `GPUTextureUsage.TRANSIENT_ATTACHMENT` (Samsung Internet 30,
    /// Chromium 143) reject it, so that hint is dropped there.
    texture_usages: wgpu::TextureUsages,
    buffers: Slots<wgpu::Buffer>,
    /// Buffers made with [`BufferUsages::IDLE_WRITES`], which writes go to
    /// in place.
    idle_writes: rustc_hash::FxHashSet<u32>,
    textures: Slots<TextureEntry>,
    shader_modules: Slots<compile::Shared<compile::Module>>,
    bind_group_layouts: Slots<BindGroupLayout>,
    bind_groups: Slots<wgpu::BindGroup>,
    /// Render and compute pipelines, their ids from one count.
    pipelines: Slots<Compiled>,
    /// Zeros that WebGL 2's shaders read at `@location(0)`, a `u32` per
    /// instance ([`Caps::gl_shaders`], #179): set with every pipeline,
    /// and grown to the most instances a pass or bundle draws.
    attribute_zero: Option<wgpu::Buffer>,
    compiler: compile::Compiler,
    /// Pipelines [`Compiled::Pending`] outside the background, kept in
    /// step by [`settle_pipeline`](Self::settle_pipeline) and the other
    /// transitions.
    pending: u32,
    /// The surface a frame draws into, acquired once a pass draws into it
    /// ([`register_surface`](Self::register_surface)).
    surface: Option<surface::Pending>,
    /// Later requests are background ones
    /// ([`Gpu::set_background_compiles`]).
    background: bool,
    /// Pipelines compiled that a frame asked for ([`PipelineCounts::made`]).
    made: u32,
    /// What kept a compile from finishing, if anything did.
    fault: Option<Fault>,
    /// A pipeline the driver fails to make loses the device: natively,
    /// unless faults are injected ([`inject_failed_compile`](Self::inject_failed_compile)).
    /// On the web the device lives on, and the map with it, drawing
    /// without the pipeline.
    #[cfg(not(target_arch = "wasm32"))]
    failures_lose: bool,
    /// Each pipeline compiled, how long it took, in the order they
    /// finished.
    compiled: Vec<(String, std::time::Duration)>,
    /// `None` for a bundle recorded on a lost device, which wgpu can't
    /// finish.
    render_bundles: Slots<Option<wgpu::RenderBundle>>,
    /// Submissions' times, while timing ([`WgpuGpu::set_timing`]).
    timing: Option<timing::Timing>,
    /// The last submission, and each unfinished frame's
    /// ([`WgpuGpu::end_frame`]).
    last: Option<wgpu::SubmissionIndex>,
    frames: std::collections::VecDeque<wgpu::SubmissionIndex>,
    samplers: Slots<wgpu::Sampler>,
    host_bytes: Slots<HostBytes>,
    /// `ImageBitmap`s and video elements for
    /// [`Gpu::write_texture_from_host`], copied with
    /// `GPUQueue.copyExternalImageToTexture`. Native hosts decode into core
    /// memory and register none.
    #[cfg(target_arch = "wasm32")]
    host_images: Slots<wgpu::ExternalImageSource>,
    /// Host images released while a held texture write still reads them,
    /// closed at the next submission.
    #[cfg(target_arch = "wasm32")]
    released_images: Vec<HostImageId>,
    /// Host bytes released while a held texture write still reads them.
    #[cfg(target_arch = "wasm32")]
    released_bytes: Vec<HostBytesId>,
    /// What the device did ([`Gpu::counters`]).
    counters: GpuCounters,
    /// The resources it holds by label, when `ORB_GPU_LEDGER` asks.
    ledger: Option<Box<ledger::Ledger>>,
    /// Makes shader modules unique to it ([`WgpuGpu::set_shader_nonce`]).
    shader_nonce: Option<u32>,
    /// Small buffer writes, copied in at the next submission.
    writes: writes::Writes,
    /// Texture writes since the last submission, those sharing a block
    /// with an earlier one issued in waves at the next.
    texture_writes: texture_writes::TextureWrites,
    /// Readback buffers back from their last readback, for the next of
    /// their size: a new one maps fresh pages each time.
    readbacks: offscreen::Readbacks,
    /// The texture the next submission copies out after its passes, and
    /// the copy the last one made, for the readback after it.
    copy_next: Option<TextureId>,
    copied: Option<(TextureId, offscreen::Readback)>,
    /// Frames started ([`WgpuGpu::start_frame`]), and those to capture by
    /// number, while one is asked for; whether a capture is running.
    frames_started: u64,
    capture: Option<std::ops::Range<u64>>,
    capturing: bool,
    /// Whether submissions carry their debug groups outside captures too
    /// ([`WgpuGpu::set_debug_groups`]).
    debug_groups: bool,
    /// Set once the device is lost: by its lost callback, by a call that
    /// shows it, or by the host ([`marks_lost`](Self::marks_lost)).
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Which thread wgpu told of the loss, and what the driver said.
    #[cfg(not(target_arch = "wasm32"))]
    loss: compile::Loss,
    /// D3D12's shader compiler, on a D3D12 device `offscreen` made.
    #[cfg(not(target_arch = "wasm32"))]
    shader_compiler: Option<String>,
}

/// Where a pipeline's compile has come to.
#[derive(Debug)]
enum Compiled {
    /// Compiling, in the background while its flag is set
    /// ([`Gpu::set_background_compiles`]).
    Pending(compile::Background),
    Ready(compile::Built),
    /// The driver failed to make it: what draws or dispatches with it does
    /// nothing.
    Failed,
}

/// A pipeline's slot in [`WgpuGpu`]'s table.
fn slot(id: PipelineId) -> u32 {
    match id {
        PipelineId::Render(id) => id.0,
        PipelineId::Compute(id) => id.0,
    }
}

/// What kept a compile from finishing ([`WgpuGpu::fault`]). Another device
/// on the same driver would do the same, so a map doesn't make one for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// A compile ran past the watchdog's limit, as in a driver hung in its
    /// compiler: the device is lost.
    Hung {
        label: String,
        after: std::time::Duration,
    },
    /// The driver failed to make a pipeline: natively the device is lost,
    /// unless failures are injected; on the web, or under injection, it
    /// lives on, drawing without what draws with it.
    Failed { label: String, error: String },
}

/// As in "the GPU driver hung making …".
impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hung { label, after } => {
                write!(f, "hung making {label} ({:.1} s)", after.as_secs_f64())
            }
            Self::Failed { label, error } => write!(f, "failed making pipeline {label}: {error}"),
        }
    }
}

/// What hosts registered with a device that lives in host memory rather
/// than on the device: bytes for [`Gpu::write_buffer_from_host`] and, in
/// the browser, images for [`Gpu::write_texture_from_host`], by their ids.
/// A device made to replace a lost one takes the lost one's
/// ([`WgpuGpu::take_host_data`]), so that what was registered before the
/// loss is still found, and released, under its id.
#[derive(Debug, Default)]
pub struct HostData {
    bytes: Slots<HostBytes>,
    #[cfg(target_arch = "wasm32")]
    images: Slots<wgpu::ExternalImageSource>,
}

impl HostData {
    /// Whether it holds nothing: all that was registered was released.
    pub fn is_empty(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if self.images.len() != 0 {
            return false;
        }
        self.bytes.len() == 0
    }
}

/// Bytes held for [`Gpu::write_buffer_from_host`].
#[derive(Debug)]
enum HostBytes {
    /// A JS `ArrayBuffer`, written with `GPUQueue.writeBuffer` without
    /// entering wasm memory (on WebGL 2, copied through it).
    #[cfg(target_arch = "wasm32")]
    Js(js_sys::ArrayBuffer),
    /// Bytes in Rust memory, for native hosts and tests.
    Owned(Vec<u8>),
}

#[derive(Debug)]
struct BindGroupLayout {
    layout: wgpu::BindGroupLayout,
    /// The browser's own, for async compiles.
    #[cfg(webgpu)]
    raw: wasm_bindgen::JsValue,
}

/// What a [`TextureId`] names: a texture, or a view that passes draw into
/// and bindings sample as it is.
#[derive(Debug)]
enum TextureEntry {
    /// A texture made here, which destroying the id destroys, viewed at
    /// its desc's dimension.
    Owned {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        dimension: wgpu::TextureViewDimension,
    },
    /// A view of some of another entry's texture: some mip levels at its
    /// dimension, or one layer at one level as a 2D texture.
    View {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    /// A view the host registered and owns.
    Host(wgpu::TextureView),
    /// A surface's texture, once acquired ([`WgpuGpu::register_surface`]).
    Surface(Option<wgpu::TextureView>),
}

impl TextureEntry {
    fn texture(&self) -> Option<&wgpu::Texture> {
        match self {
            Self::Owned { texture, .. } | Self::View { texture, .. } => Some(texture),
            Self::Host(_) | Self::Surface(_) => None,
        }
    }

    fn view(&self) -> Option<&wgpu::TextureView> {
        match self {
            Self::Owned { view, .. } | Self::View { view, .. } | Self::Host(view) => Some(view),
            Self::Surface(view) => view.as_ref(),
        }
    }
}

/// The layers a texture of `desc` is made with, on wgpu's GL back end
/// where `gl`. GL picks an array's target from its layer count alone: one
/// layer is a 2D texture, and a square one of 6n layers a cube map (an
/// array of them past 6, an enum WebGL 2 lacks), both incomplete to an
/// array's sampler, which reads black (pages, atlases). Such an array gets
/// a layer more that it never uses.
fn gl_array_layers(desc: &TextureDesc<'_>, gl: bool) -> u32 {
    let layers = desc.array_layers;
    let cube = desc.width == desc.height && desc.sample_count == 1 && layers.is_multiple_of(6);
    let array = gl && desc.dimension == TextureViewDimension::D2Array;
    if array && (layers == 1 || cube) {
        layers + 1
    } else {
        layers
    }
}

/// `source` unique to `nonce` in every entry point
/// ([`WgpuGpu::set_shader_nonce`]): each opens with a local holding it,
/// which compaction to the entry point keeps and each back end writes, so
/// what the driver compiles differs too. (A change to a function the
/// entry point doesn't call wouldn't reach the driver: compaction drops
/// it before the back end writes the stage.)
pub fn nonced(source: &str, nonce: u32) -> String {
    let mut out = String::with_capacity(source.len() + 512);
    let mut rest = source;
    while let Some(stage) = ["@vertex", "@fragment", "@compute"]
        .iter()
        .filter_map(|s| rest.find(s))
        .min()
        && let Some(body) = rest[stage..].find('{')
    {
        let (head, tail) = rest.split_at(stage + body + 1);
        out.push_str(head);
        out.push_str(&format!(" var orb_nonce: u32 = {nonce}u;"));
        rest = tail;
    }
    out.push_str(rest);
    out
}

/// Bytes a texture of `desc` with `layers` layers takes, every mip level.
fn texture_bytes(desc: &TextureDesc<'_>, layers: u32) -> u64 {
    (0..desc.mip_levels)
        .map(|level| {
            let side = |px: u32| u64::from((px >> level).max(1));
            side(desc.width)
                * side(desc.height)
                * u64::from(layers)
                * u64::from(desc.format.texel_bytes())
                * u64::from(desc.sample_count.max(1))
        })
        .sum()
}

#[cfg(not(target_arch = "wasm32"))]
fn platform_texture_usages() -> wgpu::TextureUsages {
    wgpu::TextureUsages::all()
}

#[cfg(target_arch = "wasm32")]
fn platform_texture_usages() -> wgpu::TextureUsages {
    let defined = js_sys::Reflect::get(&js_sys::global(), &"GPUTextureUsage".into())
        .and_then(|usage| js_sys::Reflect::has(&usage, &"TRANSIENT_ATTACHMENT".into()))
        .unwrap_or(false);
    if defined {
        wgpu::TextureUsages::all()
    } else {
        wgpu::TextureUsages::all() - wgpu::TextureUsages::TRANSIENT_ATTACHMENT
    }
}

/// Whether two copies into one texture level and layer in one submission
/// can race on `device` (`texture_writes`, #156): on Apple GPUs, natively
/// through Metal or MoltenVK, and in a browser that says it runs on one
/// (or withholds its vendor). GL orders its uploads as they come.
fn copies_race(device: &wgpu::Device) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let info = device.adapter_info();
        info.backend == wgpu::Backend::Metal || info.driver.contains("MoltenVK")
    }
    #[cfg(webgpu)]
    {
        let raw = device.as_webgpu().expect("the web uses the WebGPU backend");
        let vendor = js_sys::Reflect::get(raw.as_ref(), &"adapterInfo".into())
            .and_then(|info| js_sys::Reflect::get(&info, &"vendor".into()))
            .ok()
            .and_then(|vendor| vendor.as_string());
        vendor.is_none_or(|vendor| vendor.is_empty() || vendor == "apple")
    }
    #[cfg(webgl)]
    {
        let _ = device;
        false
    }
}

/// What `device`'s GL context is, on wgpu's GL back end: whether it
/// takes GLSL ES (WebGL 2's or OpenGL ES's, not desktop OpenGL's), and
/// whether it clips depth from 0 to 1 as the shaders write it
/// (`EXT_clip_control`; Chrome, Safari and ANGLE have it, Firefox doesn't)
/// rather than wgpu remapping it into GL's -1 to 1.
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(feature = "gl", target_os = "android")),
    allow(unused_variables)
)]
fn gl_context(device: &wgpu::Device) -> Option<(bool, bool)> {
    // SAFETY: only reads flags of the device's.
    #[cfg(any(feature = "gl", target_os = "android"))]
    return unsafe { device.as_hal::<wgpu::hal::api::Gles>() }
        .map(|d| (d.is_embedded(), d.zero_to_one_depth()));
    #[cfg(not(any(feature = "gl", target_os = "android")))]
    None
}

/// Whether GL raised an error since it was last asked, clearing it: after
/// a texture is made, whether the driver refused it. Never on other back
/// ends.
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(feature = "gl", target_os = "android")),
    allow(unused_variables)
)]
fn gl_error(device: &wgpu::Device) -> bool {
    #[cfg(any(feature = "gl", target_os = "android"))]
    {
        use glow::HasContext;
        // SAFETY: only reads and clears GL's error flags.
        let Some(device) = (unsafe { device.as_hal::<wgpu::hal::api::Gles>() }) else {
            return false;
        };
        let gl = device.context().lock();
        // A context holds a flag per kind of error at most.
        (0..8)
            .take_while(|_| unsafe { gl.get_error() } != glow::NO_ERROR)
            .count()
            > 0
    }
    #[cfg(not(any(feature = "gl", target_os = "android")))]
    false
}

impl WgpuGpu {
    /// A device on `adapter`, with the features and limits Orbis asks of
    /// it ([`device_limits`]) and the render formats it reads from it
    /// where WebGPU doesn't guarantee them. It marks itself lost when the
    /// driver loses it and then calls `on_lost` with what the driver said
    /// ([`lost`](Self::lost)).
    pub async fn open(
        adapter: &wgpu::Adapter,
        most: FramePath,
        on_lost: impl Fn(&str) + Send + 'static,
    ) -> Result<Self, wgpu::RequestDeviceError> {
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut required_limits = device_limits(adapter);
        #[cfg(not(target_arch = "wasm32"))]
        lower_storage(most, &mut required_limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                required_features: device_features(adapter),
                required_limits,
                ..Default::default()
            })
            .await?;
        let flags = adapter.get_downlevel_capabilities().flags;
        #[cfg_attr(webgpu, allow(unused_mut))]
        let mut gpu = Self::new(device, queue, flags, most);
        #[cfg(not(webgpu))]
        gpu.read_formats(adapter);
        gpu.watch_loss(on_lost);
        Ok(gpu)
    }

    /// A device on `device`, whose adapter has `flags`, taking the frame
    /// path they and its limits allow, or `most` where that is less.
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        flags: wgpu::DownlevelFlags,
        most: FramePath,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let gl = device.adapter_info().backend == wgpu::Backend::Gl;
        #[cfg(target_arch = "wasm32")]
        let gl = cfg!(webgl);
        let limits = device.limits();
        let gl_context = gl_context(&device);
        let frame_path = frame_path(flags, &limits, gl).max(most);
        let webgl2 = frame_path == FramePath::Webgl2;
        // Another back end standing in for WebGL 2 takes the limits wgpu
        // asks of it by default.
        let stand_in = webgl2 && !gl;
        let caps = Caps {
            limits: Limits {
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_texture_dimension_2d: limits.max_texture_dimension_2d,
                max_texture_array_layers: match stand_in {
                    true => limits.max_texture_array_layers.min(256),
                    false => limits.max_texture_array_layers,
                },
                max_texture_bytes: if webgl2 { GL_TEXTURE_BYTES } else { u64::MAX },
                min_uniform_buffer_offset_alignment: match stand_in {
                    true => limits.min_uniform_buffer_offset_alignment.max(256),
                    false => limits.min_uniform_buffer_offset_alignment,
                },
            },
            timestamp_query: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            // WebGL 2's shaders have none, though native GLES (Mesa's)
            // offers them.
            clip_distances: !webgl2 && device.features().contains(wgpu::Features::CLIP_DISTANCES),
            frame_path,
            // WebGPU guarantees both; elsewhere `read_formats` asks the
            // adapter (WebGL 2 has them with `EXT_color_buffer_float`).
            float_targets: cfg!(webgpu),
            zero_to_one_depth: gl_context.is_none_or(|(_, zero_to_one)| zero_to_one),
            depth32_stencil8: (device.features()).contains(wgpu::Features::DEPTH32FLOAT_STENCIL8),
            // `ORB_COMPILES_HOLD_PRESENTS` has a native device stand in for
            // Firefox's: what a map does there, in tests and benches.
            #[cfg(webgpu)]
            compiles_hold_presents: compile::compiles_hold_presents(),
            #[cfg(not(target_arch = "wasm32"))]
            compiles_hold_presents: std::env::var_os("ORB_COMPILES_HOLD_PRESENTS").is_some(),
            #[cfg(webgl)]
            compiles_hold_presents: false,
            // GL ES's only convention, unless the host asked for the first
            // (`flat_from_first`).
            flat_from_last: gl,
        };
        let glsl_es = gl_context.is_some_and(|(glsl_es, _)| glsl_es);
        let texture_writes = texture_writes::TextureWrites::new(copies_race(&device));
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut gpu = Self {
            device,
            queue,
            #[cfg(not(target_arch = "wasm32"))]
            native: None,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_cache: None,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_cache_kept: None,
            caps,
            gl,
            glsl_es,
            texture_usages: platform_texture_usages(),
            buffers: Slots::default(),
            idle_writes: Default::default(),
            textures: Slots::default(),
            shader_modules: Slots::default(),
            bind_group_layouts: Slots::default(),
            bind_groups: Slots::default(),
            pipelines: Slots::default(),
            attribute_zero: None,
            compiler: Default::default(),
            pending: 0,
            surface: None,
            background: false,
            made: 0,
            fault: None,
            #[cfg(not(target_arch = "wasm32"))]
            failures_lose: true,
            compiled: Vec::new(),
            render_bundles: Slots::default(),
            timing: None,
            last: None,
            frames: Default::default(),
            samplers: Slots::default(),
            host_bytes: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            host_images: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            released_images: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            released_bytes: Vec::new(),
            counters: Default::default(),
            ledger: ledger::Ledger::from_env(),
            shader_nonce: None,
            writes: writes::Writes::default(),
            texture_writes,
            readbacks: Default::default(),
            copy_next: None,
            copied: Default::default(),
            frames_started: 0,
            capture: None,
            capturing: false,
            debug_groups: false,
            lost: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            loss: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            shader_compiler: None,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(side) = std::env::var(TEXTURE_SIDE) {
            let side: u32 = (side.parse()).unwrap_or_else(|e| panic!("{TEXTURE_SIDE}={side}: {e}"));
            let limits = &mut gpu.caps.limits;
            limits.max_texture_dimension_2d = limits.max_texture_dimension_2d.min(side);
        }
        gpu
    }

    /// A buffer write from core memory: gathered when small, or when it
    /// overlaps a gathered one it must land after; otherwise staged
    /// natively, and on the web straight to the queue.
    fn write_core(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        // On WebGL 2 every write is gathered, whatever its size: a
        // `bufferSubData` into a buffer the GPU still reads makes ANGLE's
        // Metal back end copy the whole buffer into a new one, and a
        // frame's writes into the scene's arenas then cost hundreds of
        // milliseconds (#140). Gathered, they land in a staging buffer the
        // GPU is done with and are copied into place on the GPU. A buffer
        // the GPU no longer reads is written in place: a copy on the GPU
        // into an index buffer makes ANGLE wait for it before it next
        // reads the buffer for restart indices.
        if !self.idle_writes.contains(&id.0)
            && (cfg!(webgl)
                || data.len() <= writes::GATHERED
                || self.writes.overlaps(id, offset, data.len() as u64))
        {
            self.writes.gather(id, offset, data);
            return;
        }
        self.counters.upload_calls += 1;
        #[cfg(not(target_arch = "wasm32"))]
        if self.writes.stage(&self.device, id, offset, data) {
            return;
        }
        self.queue.write_buffer(&self.buffers[id.0], offset, data);
    }

    /// Copies the gathered writes into place ahead of what `encoder`
    /// records next, after issuing the held texture writes.
    /// On a device known to be lost the writes are dropped; one whose
    /// staging buffer comes back invalid is taken as lost.
    fn encode_writes(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.lost() {
            self.writes.discard();
            return;
        }
        self.issue_texture_writes();
        let issued = self.writes.encode(
            (&self.device, &self.queue),
            (&self.buffers, &self.textures),
            encoder,
        );
        match issued {
            Some(issued) => {
                self.counters.upload_calls += issued.calls;
                self.counters.texture_write_submissions += issued.submissions;
            }
            None => self.mark_lost(),
        }
    }

    /// Issues the held texture writes, wave by wave, each wave before the
    /// last its own submission (see `texture_writes`): natively staged,
    /// each in its wave, and on the web to the queue.
    fn issue_texture_writes(&mut self) {
        let mut held = std::mem::take(&mut self.texture_writes);
        // The waves the web has ended with a submission.
        let mut ended = 0;
        held.issue(|write, texels, wave| {
            while cfg!(target_arch = "wasm32") && ended < wave {
                self.queue.submit([]);
                self.counters.texture_write_submissions += 1;
                ended += 1;
            }
            self.issue_texture_write(write, texels, wave);
        });
        self.texture_writes = held;
    }

    /// Issues a texture write in `wave`: natively staged, else (or where
    /// staging failed, on a lost device) to the queue.
    fn issue_texture_write(
        &mut self,
        write: &texture_writes::TextureWrite,
        texels: &[u8],
        wave: u32,
    ) {
        self.counters.upload_calls += 1;
        #[cfg(not(target_arch = "wasm32"))]
        if let texture_writes::Source::Core { bytes_per_row, .. } = write.source
            && (self.writes).stage_texture(
                &self.device,
                (&write.region, wave),
                texels,
                bytes_per_row,
            )
        {
            return;
        }
        let _ = wave;
        let region = &write.region;
        let origin = wgpu::Origin3d {
            x: region.x,
            y: region.y,
            z: region.layer,
        };
        let size = wgpu::Extent3d {
            width: region.width,
            height: region.height,
            depth_or_array_layers: 1,
        };
        match write.source {
            texture_writes::Source::Issued => unreachable!("issued when made"),
            texture_writes::Source::Core { bytes_per_row, .. } => self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.texture(region.texture),
                    mip_level: region.level,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
                size,
            ),
            #[cfg(target_arch = "wasm32")]
            texture_writes::Source::Bytes {
                bytes,
                offset,
                bytes_per_row,
            } => self.write_texture_from_js(region, (bytes, offset), bytes_per_row),
            #[cfg(target_arch = "wasm32")]
            texture_writes::Source::Host { image, premultiply } => {
                let source = self.host_images[image.0].clone();
                // WebGL 2 copies an `ImageBitmap` as it is: the host made
                // it premultiplied (orb-web's `decode_bitmap`).
                #[cfg(webgl)]
                let premultiply =
                    premultiply && !matches!(source, wgpu::ExternalImageSource::ImageBitmap(_));
                self.queue.copy_external_image_to_texture(
                    &wgpu::CopyExternalImageSourceInfo {
                        source,
                        origin: wgpu::Origin2d::ZERO,
                        flip_y: false,
                    },
                    wgpu::CopyExternalImageDestInfo {
                        texture: self.texture(region.texture),
                        mip_level: region.level,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: premultiply,
                    },
                    size,
                )
            }
        }
    }

    /// Writes `region` from a JS buffer's texels at `offset`, rows
    /// `bytes_per_row` apart: WebGPU's `writeTexture` straight from the
    /// buffer, WebGL 2's from a copy.
    #[cfg(target_arch = "wasm32")]
    fn write_texture_from_js(
        &mut self,
        region: &TextureRegion,
        (bytes, offset): (HostBytesId, u64),
        bytes_per_row: u32,
    ) {
        let HostBytes::Js(data) = &self.host_bytes[bytes.0] else {
            unreachable!("host bytes on the web are JS buffers")
        };
        #[cfg(webgl)]
        {
            let len = bytes_per_row * region.height;
            let texels =
                js_sys::Uint8Array::new_with_byte_offset_and_length(data, offset as u32, len)
                    .to_vec();
            let write = texture_writes::TextureWrite {
                region: *region,
                source: texture_writes::Source::Core {
                    at: 0,
                    len: texels.len(),
                    bytes_per_row,
                },
            };
            self.issue_texture_write(&write, &texels, 0);
        }
        #[cfg(webgpu)]
        {
            // wgpu exposes the queue and texture, not the dictionaries
            // `writeTexture` takes: they're plain objects.
            use wasm_bindgen::{JsCast, JsValue};
            let object = |fields: &[(&str, JsValue)]| -> JsValue {
                let object = js_sys::Object::new();
                for (key, value) in fields {
                    let _ = js_sys::Reflect::set(&object, &JsValue::from_str(key), value);
                }
                object.into()
            };
            let texture = self
                .texture(region.texture)
                .as_webgpu()
                .expect("the web uses the WebGPU backend");
            let origin = object(&[
                ("x", region.x.into()),
                ("y", region.y.into()),
                ("z", region.layer.into()),
            ]);
            let destination = object(&[
                ("texture", JsValue::from(texture)),
                ("mipLevel", region.level.into()),
                ("origin", origin),
            ]);
            let layout = object(&[
                ("offset", (offset as f64).into()),
                ("bytesPerRow", bytes_per_row.into()),
            ]);
            let size = object(&[
                ("width", region.width.into()),
                ("height", region.height.into()),
                ("depthOrArrayLayers", 1.into()),
            ]);
            let queue = JsValue::from(
                self.queue
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend"),
            );
            let write = js_sys::Reflect::get(&queue, &JsValue::from_str("writeTexture"))
                .expect("GPUQueue.writeTexture")
                .unchecked_into::<js_sys::Function>();
            let args = js_sys::Array::of4(&destination, data, &layout, &size);
            write
                .apply(&queue, &args)
                .expect("writeTexture from a host ArrayBuffer");
        }
    }

    /// Asks the graphics debugger the process runs under (Xcode's Metal
    /// debugger, RenderDoc) to capture `count` frames after the next
    /// `skip`: each frame from [`start_frame`](Self::start_frame) to
    /// [`finish_frame`](Self::finish_frame), its passes grouped by stage
    /// and labelled by role. Without one attached, and in the browser, it
    /// does nothing.
    pub fn capture_frames(&mut self, skip: u64, count: u64) {
        let first = self.frames_started + skip;
        self.capture = Some(first..first + count);
    }

    /// Encodes submissions' debug groups ([`Pass::PushDebugGroup`]) outside
    /// captures too, for captures a debugger's own controls start. Off by
    /// default: each is a driver call with a string made for it, which
    /// only a capture reads.
    pub fn set_debug_groups(&mut self, on: bool) {
        self.debug_groups = on;
    }

    /// Makes the shader modules made from now on unique to `nonce`, so
    /// that no shader cache (the driver's, the system's) holds their
    /// pipelines: cold compiles for measurements, as the web pages'
    /// `?nonce=` makes them. Every entry point opens with a local holding
    /// `nonce`, and each module's WGSL is parsed rather than the build's
    /// IR taken, which adds the parse to the compiles. GL's GLSL is left
    /// alone.
    pub fn set_shader_nonce(&mut self, nonce: Option<u32>) {
        self.shader_nonce = nonce;
    }

    /// Marks the start of a frame, its submissions and a host's present
    /// after them. Starts a capture asked for.
    #[allow(unsafe_code)]
    pub fn start_frame(&mut self) {
        let frame = self.frames_started;
        self.frames_started += 1;
        if !self.capturing && self.capture.as_ref().is_some_and(|c| c.start == frame) {
            // SAFETY: no other capture runs: this device starts one only
            // when none of its own is, and nothing else in Orbis starts
            // one. The debuggers' own rules hold as for any wgpu capture.
            unsafe { self.device.start_graphics_debugger_capture() };
            self.capturing = true;
        }
    }

    /// Marks the end of a frame. Ends a capture after its last frame,
    /// once the GPU finished the work, as debuggers want.
    #[allow(unsafe_code)]
    pub fn finish_frame(&mut self) {
        if self.capturing
            && self
                .capture
                .as_ref()
                .is_some_and(|c| self.frames_started >= c.end)
        {
            let _ = self.wait(None);
            // SAFETY: the capture `start_frame` started runs.
            unsafe { self.device.stop_graphics_debugger_capture() };
            self.capturing = false;
            self.capture = None;
        }
    }

    /// Each pipeline compiled so far and how long it took, in the order
    /// they finished: natively the worker's time in wgpu and the driver;
    /// in the browser from the request to its promise resolving.
    pub fn pipeline_timings(&self) -> &[(String, std::time::Duration)] {
        &self.compiled
    }

    /// Compiles every later pipeline on `executor`, the host's job threads,
    /// or (`None`, the default) on the calling thread, so it is ready when
    /// [`Gpu::create_render_pipeline`] returns.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_executor(
        &mut self,
        executor: Option<std::sync::Arc<dyn orb_host::jobs::Executor>>,
    ) {
        self.compiler.set_executor(executor);
    }

    /// Blocks until every pipeline asked for so far compiled: for tools
    /// and tests that want a complete frame now.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish_pipelines(&mut self) {
        self.poll_pipelines();
        while self.pending > 0 && !matches!(self.fault, Some(Fault::Hung { .. })) {
            self.compiler.wait();
            self.poll_pipelines();
        }
    }

    /// Calls `wake` from a compile thread whenever a pipeline compiles, so
    /// a host that stopped drawing draws the layers that waited for it.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_wake(&self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
        self.compiler.set_wake(wake);
    }

    /// Reads what `adapter`, this device's, renders to, where WebGPU
    /// doesn't guarantee it: [`Caps::float_targets`].
    #[cfg(not(webgpu))]
    fn read_formats(&mut self, adapter: &wgpu::Adapter) {
        let renders = |format, blended: bool| {
            let features = adapter.get_texture_format_features(format);
            features
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
                && (!blended
                    || features
                        .flags
                        .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE))
        };
        self.caps.float_targets = renders(wgpu::TextureFormat::Rgba16Float, true)
            && renders(wgpu::TextureFormat::R32Float, false);
    }

    /// Records that the host set its GL context's flat varyings to take
    /// a primitive's first vertex, as WebGPU's do (`WEBGL_provoking_vertex`
    /// on the web): [`Caps::flat_from_last`].
    pub fn flat_from_first(&mut self) {
        self.caps.flat_from_last = false;
    }

    /// Calls `wake` whenever a pipeline's `create*PipelineAsync` promise
    /// settles (on WebGL 2, while pipelines wait to compile), so a host
    /// that stopped drawing draws the layers that waited for it.
    #[cfg(target_arch = "wasm32")]
    pub fn set_compile_wake(&self, wake: std::rc::Rc<dyn Fn()>) {
        self.compiler.set_wake(wake);
    }

    /// Makes later compiles stall forever, as in a driver hung in its
    /// compiler: fault injection for the watchdog's tests. Needs compile
    /// threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inject_compile_hang(&self) {
        self.compiler.hang();
    }

    /// Holds compiles of pipelines labelled one of `labels` from finishing
    /// until the hold is dropped, as a slow compiler would: fault
    /// injection for tests of what waits for them. Needs compile threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn hold_compiles(&self, labels: &[&str]) -> CompileHold {
        self.compiler.hold(labels)
    }

    /// Makes later compiles of pipelines labelled one of `labels` fail, and
    /// the device live on past a failed compile, as on the web: fault
    /// injection for tests of maps drawing without them. Needs compile
    /// threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inject_failed_compile(&mut self, labels: &[&str]) {
        self.failures_lose = false;
        self.compiler.inject_failure(labels);
    }

    /// Makes later compiles of pipelines labelled one of `labels` take
    /// `by` longer, as on a device compiling cold: fault injection for
    /// tests of what waits for them. Needs compile threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inject_slow_compile(&self, labels: &[&str], by: std::time::Duration) {
        self.compiler.inject_slow(labels, by);
    }

    /// How long a compile may run before the driver is taken to have hung
    /// (60 s by default).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_watchdog(&self, limit: std::time::Duration) {
        self.compiler.set_watchdog(limit);
    }

    /// What kept a compile from finishing, if anything did: a hang loses
    /// the device, as does a failure natively.
    pub fn fault(&self) -> Option<&Fault> {
        self.fault.as_ref()
    }

    /// On a lost device, lets the compiles running finish (up to `wait`)
    /// and takes them, so a pipeline whose compile lost the device is its
    /// [`fault`](Self::fault): a loss is checked for before frames take
    /// their compiles. The caller waits once per loss: a compile still
    /// running then (a slow compiler, a hung driver) mustn't stall every
    /// later check.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn settle_loss(&mut self, wait: std::time::Duration) {
        if self.fault.is_none() && self.lost() {
            self.compiler.settle(wait);
            self.take_compiled(false);
        }
    }

    /// Takes the compiles that finished; on WebGL 2, `compile` compiles
    /// the next for a frame's share of time first.
    #[cfg_attr(not(webgl), allow(unused_variables))]
    fn take_compiled(&mut self, compile: bool) {
        #[cfg(not(target_arch = "wasm32"))]
        let done: Vec<_> = self.compiler.take().collect();
        #[cfg(webgpu)]
        let done = self.compiler.take();
        #[cfg(webgl)]
        let done = self.compiler.take(&self.device, compile);
        for done in done {
            let built = match done.built {
                Ok(built) => {
                    self.counters.pipelines_compiled += 1;
                    self.compiled.push((done.label, done.took));
                    Compiled::Ready(built)
                }
                // A pipeline the driver can't make goes the way of a hung
                // compile natively: the device is lost, and recovery
                // demotes it. Where the device lives on, so does the map,
                // drawing without the pipeline: it is settled, so nothing
                // waits for it.
                Err(e) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    if self.failures_lose {
                        self.mark_lost();
                    }
                    self.fault.get_or_insert(Fault::Failed {
                        label: done.label,
                        error: e,
                    });
                    Compiled::Failed
                }
            };
            self.settle_pipeline(done.id, built);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if !matches!(self.fault, Some(Fault::Hung { .. }))
            && self.pending > 0
            && let Some(hung) = self.compiler.hung()
        {
            self.fault = Some(hung);
            self.mark_lost();
        }
        self.gate_background();
    }

    /// Lets background compiles start only while no compile a frame waits
    /// for is pending: they are what is left once the view has what it
    /// needs.
    fn gate_background(&mut self) {
        let open = self.pending == 0;
        self.compiler.open_background(&self.device, open);
    }

    /// Settles a pipeline's compile (`state` is ready or failed): one
    /// destroyed while it compiled is dropped.
    fn settle_pipeline(&mut self, id: PipelineId, state: Compiled) {
        let Some(entry) = self.pipelines.get_mut(slot(id)) else {
            return;
        };
        if let Compiled::Pending(background) = entry
            && !background.load(std::sync::atomic::Ordering::Relaxed)
        {
            self.pending -= 1;
            self.made += 1;
        }
        *entry = state;
    }

    /// Makes a pipeline pending and asks for its compile, a background
    /// one while [`background`](Self::background) is set: its slot.
    fn submit_job(&mut self, pipeline: compile::Pipeline) -> u32 {
        self.counters.pipelines_requested += 1;
        let background = compile::Background::new(self.background.into());
        let slot = self.pipelines.insert(Compiled::Pending(background.clone()));
        let id = match &pipeline {
            compile::Pipeline::Render(_) => RenderPipelineId(slot).into(),
            compile::Pipeline::Compute(_) => ComputePipelineId(slot).into(),
        };
        if !self.background {
            self.pending += 1;
            self.gate_background();
        }
        let job = compile::Job {
            id,
            background,
            pipeline,
        };
        self.compiler.submit(&self.device, job);
        slot
    }

    /// The browser's bind group layouts of `layouts`, for async compiles.
    #[cfg(webgpu)]
    fn raw_layouts(&self, layouts: &[BindGroupLayoutId]) -> Vec<wasm_bindgen::JsValue> {
        (layouts.iter())
            .map(|id| self.bind_group_layouts[id.0].raw.clone())
            .collect()
    }

    /// The pipeline cache pipelines are made through, if any.
    #[cfg(not(webgpu))]
    fn cache(&self) -> Option<wgpu::PipelineCache> {
        #[cfg(not(target_arch = "wasm32"))]
        return self.pipeline_cache.clone();
        #[cfg(target_arch = "wasm32")]
        None
    }

    /// Time spent compiling pipelines so far, summed over the threads
    /// that compiled them (natively wgpu translates and the driver
    /// compiles there).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pipeline_time(&self) -> std::time::Duration {
        self.compiled.iter().map(|(_, took)| *took).sum()
    }

    /// Whether the device is lost: the host needs another. Calls on a lost
    /// device do nothing, and none panics; what waits on the GPU, such as
    /// a readback, resolves to [`Lost`].
    pub fn lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn mark_lost(&self) {
        self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Marks the device lost when called: for a host that hears of a loss
    /// the driver doesn't report, as a WebGL 2 canvas does of its context.
    pub fn marks_lost(&self) -> impl Fn() + Send + Sync + 'static {
        let lost = self.lost.clone();
        move || lost.store(true, std::sync::atomic::Ordering::Relaxed)
    }

    /// What the driver said when the device was lost, if it was.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lost_reason(&self) -> Option<String> {
        self.loss.reason.get().cloned()
    }

    /// Loses the device on purpose (`destroy`), to exercise recovery. The
    /// loss is marked at once rather than by the lost callback, which
    /// hears of it later or (Firefox's, after `destroy()`) not at all.
    pub fn lose(&self) {
        self.mark_lost();
        self.device.destroy();
    }

    /// Marks the device lost when the driver loses it, calling `on_lost`
    /// with what the driver said if the loss wasn't already known (as one
    /// [`lose`](Self::lose) made is).
    ///
    /// Natively it also takes the errors calls on a lost device raise as
    /// part of the loss: a loss can come in the middle of a frame (in a
    /// render bundle's `finish`, at a submit), and the frame's remaining
    /// calls each fail with "Parent device is lost", which wgpu's default
    /// handler would panic on. Any other uncaptured error still panics, as
    /// it would without this.
    fn watch_loss(&self, on_lost: impl Fn(&str) + Send + 'static) {
        #[cfg(not(target_arch = "wasm32"))]
        let loss = self.loss.clone();
        #[cfg(not(target_arch = "wasm32"))]
        self.compiler.watch_loss(loss.clone());
        let lost = self.lost.clone();
        self.device
            .set_device_lost_callback(move |reason, message| {
                let why = format!("{reason:?}: {message}");
                #[cfg(not(target_arch = "wasm32"))]
                {
                    loss.reason.get_or_init(|| why.clone());
                    loss.thread.get_or_init(|| std::thread::current().id());
                }
                if !lost.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    on_lost(&why);
                }
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let lost = self.lost.clone();
            self.device
                .on_uncaptured_error(std::sync::Arc::new(move |error: wgpu::Error| {
                    let loss = lost.load(std::sync::atomic::Ordering::Relaxed)
                        || error.to_string().contains("device is lost");
                    if !loss {
                        panic!("wgpu error: {error}");
                    }
                    lost.store(true, std::sync::atomic::Ordering::Relaxed);
                }));
        }
    }

    /// D3D12's shader compiler ([`offscreen`](Self::offscreen) chooses DXC
    /// over FXC): where it was found, or `FXC`. `None` on other back ends.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn shader_compiler(&self) -> Option<&str> {
        self.shader_compiler.as_deref()
    }

    /// The adapter behind a device `offscreen` made.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn adapter_info(&self) -> Option<wgpu::AdapterInfo> {
        self.native.as_ref().map(|(_, adapter)| adapter.get_info())
    }

    /// Registers bytes in Rust memory for [`Gpu::write_buffer_from_host`].
    pub fn register_bytes(&mut self, bytes: Vec<u8>) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Owned(bytes)))
    }

    /// Hands over the host data registered here, for the device that
    /// replaces this lost one: results that arrived before the loss hold
    /// its ids. Images released while a write still read them are closed
    /// now: that write went with the device.
    pub fn take_host_data(&mut self) -> HostData {
        #[cfg(target_arch = "wasm32")]
        for bytes in std::mem::take(&mut self.released_bytes) {
            self.host_bytes.remove(bytes.0);
        }
        #[cfg(target_arch = "wasm32")]
        for image in std::mem::take(&mut self.released_images) {
            if let wgpu::ExternalImageSource::ImageBitmap(bitmap) = self.host_images.remove(image.0)
            {
                bitmap.close();
            }
        }
        HostData {
            bytes: std::mem::take(&mut self.host_bytes),
            #[cfg(target_arch = "wasm32")]
            images: std::mem::take(&mut self.host_images),
        }
    }

    /// Takes over a lost device's host data ([`take_host_data`](Self::take_host_data)),
    /// in place of this new device's, which holds none yet: its ids find
    /// the same data here, and ids registered from now on follow them.
    pub fn adopt_host_data(&mut self, data: HostData) {
        assert_eq!(self.host_bytes.len(), 0, "a device that holds host bytes");
        self.host_bytes = data.bytes;
        #[cfg(target_arch = "wasm32")]
        {
            assert_eq!(self.host_images.len(), 0, "a device that holds host images");
            self.host_images = data.images;
        }
    }

    /// Registers a JS `ArrayBuffer` for [`Gpu::write_buffer_from_host`].
    #[cfg(target_arch = "wasm32")]
    pub fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Js(buffer)))
    }

    /// Registers a JS `ImageBitmap` for [`Gpu::write_texture_from_host`].
    #[cfg(target_arch = "wasm32")]
    pub fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId {
        HostImageId(
            self.host_images
                .insert(wgpu::ExternalImageSource::ImageBitmap(bitmap)),
        )
    }

    /// Registers a video element, whose frame at the time of the write
    /// [`Gpu::write_texture_from_host`] copies.
    #[cfg(target_arch = "wasm32")]
    pub fn register_video(&mut self, video: web_sys::HtmlVideoElement) -> HostImageId {
        HostImageId(
            self.host_images
                .insert(wgpu::ExternalImageSource::HTMLVideoElement(video)),
        )
    }

    /// The instance and adapter behind a device made here, for a host
    /// that draws with the same device (a UI toolkit on wgpu).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn instance_and_adapter(&self) -> Option<(&wgpu::Instance, &wgpu::Adapter)> {
        self.native
            .as_ref()
            .map(|(instance, adapter)| (instance, adapter))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Registers a texture view the host owns, such as the current surface
    /// frame, as a render target. Release it with
    /// [`Gpu::destroy_texture`] before the view is dropped or presented.
    pub fn register_view(&mut self, view: wgpu::TextureView) -> TextureId {
        TextureId(self.textures.insert(TextureEntry::Host(view)))
    }

    /// The view of a texture a pass draws into or a binding samples.
    fn view(&self, id: TextureId) -> &wgpu::TextureView {
        (self.textures[id.0].view()).expect("a surface's texture acquired for its pass")
    }

    /// The wgpu texture behind an id made with [`Gpu::create_texture`], or
    /// of a view of one.
    pub fn texture(&self, id: TextureId) -> &wgpu::Texture {
        (self.textures[id.0].texture()).expect("a texture, not a host's view")
    }

    /// The wgpu buffer behind an id.
    pub fn buffer(&self, id: BufferId) -> &wgpu::Buffer {
        &self.buffers[id.0]
    }

    /// Ends a frame drawn into a texture: waits until no more than
    /// `in_flight` frames are unfinished, as a swapchain does.
    pub fn end_frame(&mut self, in_flight: usize) {
        self.frames.extend(self.last.take());
        while self.frames.len() > in_flight {
            let index = self.frames.pop_front();
            // A lost device has no frames in flight.
            if self.wait(index).is_err() {
                self.frames.clear();
            }
        }
    }

    /// Nanoseconds per timestamp tick in resolved query sets.
    pub(crate) fn timestamp_period_ns(&self) -> f32 {
        self.queue.get_timestamp_period()
    }

    #[cfg(not(webgpu))]
    fn pipeline_layout(&self, label: &str, layouts: &[BindGroupLayoutId]) -> wgpu::PipelineLayout {
        let layouts: Vec<_> = layouts
            .iter()
            .map(|id| Some(&self.bind_group_layouts[id.0].layout))
            .collect();
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            })
    }

    /// Grows [`attribute_zero`](Self::attribute_zero) to the most
    /// instances `commands` draw, where shaders read it. Bundles recorded
    /// before keep the smaller buffer they set.
    fn reserve_attribute_zero<'c>(
        &mut self,
        commands: impl IntoIterator<Item = &'c RenderCommand>,
    ) {
        if !self.caps.gl_shaders() {
            return;
        }
        let instances = (commands.into_iter())
            .filter_map(|command| match command {
                RenderCommand::Draw { instances, .. }
                | RenderCommand::DrawIndexed { instances, .. } => Some(instances.end),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let needed = u64::from(instances.max(1)) * 4;
        if (self.attribute_zero.as_ref()).is_some_and(|b| b.size() >= needed) {
            return;
        }
        // wgpu zeroes a buffer before its first use.
        self.attribute_zero = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attribute zero"),
            size: needed.next_power_of_two().max(4 << 10),
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        }));
    }

    /// Encodes a command bundles may hold. `skip`, the encoder's: the
    /// pipeline set last failed, so its draws are left out until another
    /// is set.
    fn encode_render<'p>(
        &'p self,
        pass: &mut impl wgpu::util::RenderEncoder<'p>,
        command: &'p RenderCommand,
        skip: &mut bool,
    ) {
        match command {
            RenderCommand::SetPipeline(id) => {
                *skip = match &self.pipelines[id.0] {
                    Compiled::Ready(compile::Built::Render(pipeline)) => {
                        pass.set_pipeline(pipeline);
                        false
                    }
                    Compiled::Failed => true,
                    _ => panic!("render pipeline {} used before it compiled", id.0),
                };
                if let Some(zeros) = &self.attribute_zero {
                    pass.set_vertex_buffer(0, Some(zeros.slice(..)));
                }
            }
            RenderCommand::Draw { .. }
            | RenderCommand::DrawIndexed { .. }
            | RenderCommand::DrawIndirect { .. }
            | RenderCommand::DrawIndexedIndirect { .. }
                if *skip => {}
            RenderCommand::SetBindGroup { index, group } => {
                pass.set_bind_group(*index, Some(&self.bind_groups[group.0]), &[])
            }
            RenderCommand::Draw {
                vertices,
                instances,
            } => pass.draw(vertices.clone(), instances.clone()),
            RenderCommand::SetIndexBuffer { buffer, offset } => pass.set_index_buffer(
                self.buffers[buffer.0].slice(*offset..),
                wgpu::IndexFormat::Uint32,
            ),
            RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => pass.draw_indexed(indices.clone(), *base_vertex, instances.clone()),
            RenderCommand::DrawIndirect { buffer, offset } => {
                pass.draw_indirect(&self.buffers[buffer.0], *offset)
            }
            RenderCommand::DrawIndexedIndirect { buffer, offset } => {
                pass.draw_indexed_indirect(&self.buffers[buffer.0], *offset)
            }
            RenderCommand::ExecuteBundles(_) => panic!("render bundles cannot execute bundles"),
            RenderCommand::SetScissor(_) => panic!("render bundles cannot set a scissor"),
            RenderCommand::SetViewport(_) => panic!("render bundles cannot set a viewport"),
            RenderCommand::SetStencilReference(_) => {
                panic!("render bundles cannot set a stencil reference")
            }
        }
    }
}

impl Gpu for WgpuGpu {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(desc.label),
            size: desc.size,
            usage: convert::buffer_usages(desc.usage),
            mapped_at_creation: false,
        });
        self.counters.buffers_created += 1;
        self.counters.bytes_allocated += desc.size;
        let id = self.buffers.insert(buffer);
        if let Some(l) = &mut self.ledger {
            l.made(false, id, desc.label, desc.size);
        }
        if desc.usage.contains(BufferUsages::IDLE_WRITES) {
            self.idle_writes.insert(id);
        }
        BufferId(id)
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        self.writes.forget(id);
        self.idle_writes.remove(&id.0);
        if let Some(l) = &mut self.ledger {
            l.freed(false, id.0);
        }
        self.buffers.remove(id.0).destroy();
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.counters.buffer_bytes_from_core += data.len() as u64;
        self.write_core(id, offset, data);
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        #[cfg(target_arch = "wasm32")]
        let len = range.end - range.start;
        match &mut self.host_bytes[bytes.0] {
            // Taken out while the device writes from them.
            HostBytes::Owned(data) => {
                let data = std::mem::take(data);
                self.write_buffer(id, offset, &data[range.start as usize..range.end as usize]);
                self.host_bytes[bytes.0] = HostBytes::Owned(data);
            }
            #[cfg(webgl)]
            HostBytes::Js(data) => {
                let data = js_sys::Uint8Array::new_with_byte_offset_and_length(
                    data,
                    range.start as u32,
                    len as u32,
                )
                .to_vec();
                self.counters.buffer_bytes_from_host += len;
                self.write_core(id, offset, &data);
            }
            #[cfg(webgpu)]
            HostBytes::Js(data) => {
                // A handle, cloned so the device is free to flush.
                let data = data.clone();
                self.counters.buffer_bytes_from_host += len;
                self.counters.upload_calls += 1;
                // It lands before gathered writes: those it overlaps go first.
                if self.writes.overlaps(id, offset, len) {
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("uploads"),
                            });
                    self.encode_writes(&mut encoder);
                    self.queue.submit([encoder.finish()]);
                }
                let queue = self
                    .queue
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend");
                let buffer = self.buffers[id.0]
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend");
                queue
                    .write_buffer_with_f64_and_buffer_source_and_f64_and_f64(
                        buffer,
                        offset as f64,
                        &data,
                        range.start as f64,
                        len as f64,
                    )
                    .expect("writeBuffer from a host ArrayBuffer");
            }
        }
    }

    fn read_host_bytes(&self, bytes: HostBytesId, range: std::ops::Range<u64>) -> Vec<u8> {
        match &self.host_bytes[bytes.0] {
            HostBytes::Owned(data) => data[range.start as usize..range.end as usize].to_vec(),
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(data) => js_sys::Uint8Array::new_with_byte_offset_and_length(
                data,
                range.start as u32,
                (range.end - range.start) as u32,
            )
            .to_vec(),
        }
    }

    fn release_host_bytes(&mut self, bytes: HostBytesId) {
        // A held write still reads them: released once that is issued.
        #[cfg(target_arch = "wasm32")]
        if self.texture_writes.reads_bytes(bytes) {
            self.released_bytes.push(bytes);
            return;
        }
        self.host_bytes.remove(bytes.0);
    }

    fn write_texture_from_host_bytes(
        &mut self,
        region: &TextureRegion,
        bytes: HostBytesId,
        offset: u64,
    ) {
        let texture = self.texture(region.texture);
        let bytes_per_row = region.width * convert::texel_bytes(texture.format());
        let len = u64::from(bytes_per_row * region.height);
        match &mut self.host_bytes[bytes.0] {
            // Taken out while the device writes from them.
            HostBytes::Owned(data) => {
                let data = std::mem::take(data);
                self.write_texture(region, &data[offset as usize..(offset + len) as usize]);
                self.host_bytes[bytes.0] = HostBytes::Owned(data);
            }
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(_) => {
                self.counters.texel_bytes_from_host += len;
                let held = self
                    .texture_writes
                    .bytes(region, (bytes, offset), bytes_per_row);
                if !held {
                    self.counters.upload_calls += 1;
                    self.write_texture_from_js(region, (bytes, offset), bytes_per_row);
                }
            }
        }
    }

    /// As [`gl_array_layers`] on GL, and on another back end standing in
    /// for WebGL 2 as GL would make it.
    fn allocated_layers(&self, desc: &TextureDesc<'_>) -> u32 {
        gl_array_layers(desc, self.caps.gl_shaders())
    }

    fn try_create_texture(&mut self, desc: &TextureDesc<'_>) -> Option<TextureId> {
        let bytes = orb_hal::level_bytes(desc, gl_array_layers(desc, self.gl));
        let checked = self.gl && bytes >= CHECKED_TEXTURE_BYTES;
        // An error raised before it isn't the texture's.
        if checked {
            gl_error(&self.device);
        }
        let texture = self.create_texture(desc);
        if !(checked && gl_error(&self.device)) {
            return Some(texture);
        }
        self.destroy_texture(texture);
        let limits = &mut self.caps.limits;
        limits.max_texture_bytes = limits.max_texture_bytes.min(bytes / 2);
        None
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> TextureId {
        let layers = gl_array_layers(desc, self.gl);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(desc.label),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: layers,
            },
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: convert::texture_format(desc.format),
            usage: convert::texture_usages(desc.usage) & self.texture_usages,
            view_formats: &[],
        });
        self.counters.textures_created += 1;
        self.counters.bytes_allocated += texture_bytes(desc, layers);
        let dimension = convert::view_dimension(desc.dimension);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(dimension),
            ..Default::default()
        });
        let id = self.textures.insert(TextureEntry::Owned {
            texture,
            view,
            dimension,
        });
        if let Some(l) = &mut self.ledger {
            l.made(true, id, desc.label, texture_bytes(desc, layers));
        }
        TextureId(id)
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        let TextureEntry::Owned {
            texture, dimension, ..
        } = &self.textures[texture.0]
        else {
            panic!("a view of a view");
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(*dimension),
            base_mip_level: levels.start,
            mip_level_count: Some(levels.len() as u32),
            ..Default::default()
        });
        let texture = texture.clone();
        TextureId(self.textures.insert(TextureEntry::View { texture, view }))
    }

    fn create_layer_view(&mut self, texture: TextureId, layer: u32, level: u32) -> TextureId {
        let texture = self.texture(texture).clone();
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        TextureId(self.textures.insert(TextureEntry::View { texture, view }))
    }

    fn destroy_texture(&mut self, id: TextureId) {
        self.texture_writes.forget(id);
        #[cfg(not(target_arch = "wasm32"))]
        self.writes.forget_texture(id);
        if let TextureEntry::Owned { texture, .. } = self.textures.remove(id.0) {
            if let Some(l) = &mut self.ledger {
                l.freed(true, id.0);
            }
            texture.destroy();
        }
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.counters.texel_bytes_from_core += data.len() as u64;
        let texture = self.texture(region.texture);
        let bytes_per_row = region.width * convert::texel_bytes(texture.format());
        assert_eq!(
            data.len(),
            (bytes_per_row * region.height) as usize,
            "texture write size mismatch"
        );
        let held = self.texture_writes.core(region, data, bytes_per_row);
        if !held {
            let source = texture_writes::Source::Core {
                at: 0,
                len: data.len(),
                bytes_per_row,
            };
            let write = texture_writes::TextureWrite {
                region: *region,
                source,
            };
            self.issue_texture_write(&write, data, 0);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        self.counters.texel_bytes_from_host +=
            u64::from(region.width) * u64::from(region.height) * 4;
        // Issued at the next submission; the image is closed after it,
        // which WebGL 2, copying only at a submission, needs too.
        self.texture_writes.host(region, image, premultiply);
    }

    #[cfg(target_arch = "wasm32")]
    fn release_host_image(&mut self, image: HostImageId) {
        // A held write still reads it: closed once that is issued.
        if self.texture_writes.reads(image) {
            self.released_images.push(image);
            return;
        }
        if let wgpu::ExternalImageSource::ImageBitmap(bitmap) = self.host_images.remove(image.0) {
            bitmap.close();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_texture_from_host(&mut self, _: &TextureRegion, image: HostImageId, _: bool) {
        unreachable!("{image:?}: native hosts register no images");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn release_host_image(&mut self, image: HostImageId) {
        unreachable!("{image:?}: native hosts register no images");
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId {
        let mut descriptor = convert::sampler(desc);
        descriptor.label = Some(desc.label);
        SamplerId(
            self.samplers
                .insert(self.device.create_sampler(&descriptor)),
        )
    }

    fn destroy_sampler(&mut self, id: SamplerId) {
        self.samplers.remove(id.0);
    }

    fn destroy_shader_module(&mut self, id: ShaderModuleId) {
        self.shader_modules.remove(id.0);
    }

    fn destroy_bind_group_layout(&mut self, id: BindGroupLayoutId) {
        self.bind_group_layouts.remove(id.0);
    }

    fn destroy_pipeline(&mut self, id: PipelineId) {
        // A compile running when it lands finds no slot and is dropped.
        // Until then it is outside `pending`, so the watchdog and the
        // present and background gates no longer cover it.
        if let Compiled::Pending(background) = self.pipelines.remove(slot(id)) {
            self.compiler.cancel(id);
            if !background.load(std::sync::atomic::Ordering::Relaxed) {
                self.pending -= 1;
                self.gate_background();
            }
        }
    }

    fn live_resources(&self) -> usize {
        let counts = [
            self.buffers.len(),
            self.textures.len(),
            self.shader_modules.len(),
            self.bind_group_layouts.len(),
            self.bind_groups.len(),
            self.pipelines.len(),
            self.render_bundles.len(),
            self.samplers.len(),
        ];
        counts.iter().sum()
    }

    fn counters(&self) -> GpuCounters {
        self.counters
    }

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        let (source, ir) = match self.shader_nonce {
            Some(nonce) => (nonced(&desc.source, nonce), None),
            None => (desc.source.to_string(), desc.ir),
        };
        ShaderModuleId(self.shader_modules.insert(compile::Module::new(
            desc.label,
            source,
            ir,
            (desc.glsl.filter(|_| self.glsl_es)).map(|g| (g, self.caps.zero_to_one_depth)),
            desc.fold,
        )))
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        let entries: Vec<_> = desc
            .entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect();
        let layout = BindGroupLayout {
            layout: self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(desc.label),
                    entries: &entries,
                }),
            #[cfg(webgpu)]
            raw: compile::raw_bind_group_layout(
                self.device
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend")
                    .as_ref(),
                desc,
            ),
        };
        BindGroupLayoutId(self.bind_group_layouts.insert(layout))
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId {
        let entries: Vec<_> = desc
            .entries
            .iter()
            .map(|entry| wgpu::BindGroupEntry {
                binding: entry.binding,
                resource: match entry.resource {
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers[buffer.0],
                        offset,
                        size: size.map(|s| s.try_into().expect("binding size must be non-zero")),
                    }),
                    BindingResource::Texture(id) => {
                        wgpu::BindingResource::TextureView(self.view(id))
                    }
                    BindingResource::Sampler(id) => {
                        wgpu::BindingResource::Sampler(&self.samplers[id.0])
                    }
                },
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.label),
            layout: &self.bind_group_layouts[desc.layout.0].layout,
            entries: &entries,
        });
        BindGroupId(self.bind_groups.insert(group))
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.bind_groups.remove(id.0);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        let module = |stage: &ShaderStage<'_>| self.shader_modules[stage.module.0].clone();
        let job = compile::RenderJob {
            label: desc.label.to_owned(),
            #[cfg(not(webgpu))]
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            #[cfg(webgpu)]
            layouts: self.raw_layouts(desc.bind_group_layouts),
            vertex: compile::Stage::new(&desc.vertex, module(&desc.vertex)),
            fragment: compile::Stage::new(&desc.fragment, module(&desc.fragment)),
            targets: desc.targets.to_vec(),
            depth: desc.depth,
            topology: desc.topology,
            cull_back: desc.cull_back,
            sample_count: desc.sample_count,
            #[cfg(not(webgpu))]
            attribute_zero: self.caps.gl_shaders(),
            #[cfg(not(webgpu))]
            cache: self.cache(),
        };
        RenderPipelineId(self.submit_job(compile::Pipeline::Render(job)))
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        let job = compile::ComputeJob {
            label: desc.label.to_owned(),
            #[cfg(not(webgpu))]
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            #[cfg(webgpu)]
            layouts: self.raw_layouts(desc.bind_group_layouts),
            compute: compile::Stage::new(
                &desc.compute,
                self.shader_modules[desc.compute.module.0].clone(),
            ),
            #[cfg(not(webgpu))]
            cache: self.cache(),
        };
        ComputePipelineId(self.submit_job(compile::Pipeline::Compute(job)))
    }

    fn pipeline_failed(&self, id: PipelineId) -> bool {
        matches!(self.pipelines.get(slot(id)), Some(Compiled::Failed))
    }

    fn pipeline_ready(&mut self, id: PipelineId) -> bool {
        // A destroyed one (ids aren't reused) has nothing waiting for it.
        let settled = |gpu: &Self| {
            (gpu.pipelines.get(slot(id))).is_none_or(|p| !matches!(p, Compiled::Pending(_)))
        };
        if settled(self) {
            return true;
        }
        self.take_compiled(false);
        // A frame asks for a background pipeline: it waits for it now. A
        // compile finishing as it does woke nothing, so it is taken again,
        // past a fence the compile mirrors after its send.
        if let Some(Compiled::Pending(background)) = self.pipelines.get(slot(id))
            && background.swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            self.pending += 1;
            self.gate_background();
            self.compiler.promote(&self.device, id);
            self.take_compiled(false);
        }
        settled(self)
    }

    fn poll_pipelines(&mut self) -> PipelineCounts {
        self.take_compiled(true);
        PipelineCounts {
            made: self.made,
            pending: self.pending,
        }
    }

    fn prioritize_pipeline(&mut self, id: PipelineId) {
        self.compiler.prioritize(id);
    }

    fn set_background_compiles(&mut self, on: bool) {
        self.background = on && !self.caps.compiles_hold_presents;
    }

    fn presents_wait(&mut self) -> bool {
        if !self.caps.compiles_hold_presents {
            return false;
        }
        self.take_compiled(false);
        self.pending > 0
    }

    fn create_render_bundle(
        &mut self,
        desc: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId {
        self.reserve_attribute_zero(commands);
        let color_formats: Vec<_> = desc
            .color_formats
            .iter()
            .map(|f| Some(convert::texture_format(*f)))
            .collect();
        let mut encoder =
            self.device
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some(desc.label),
                    color_formats: &color_formats,
                    depth_stencil: desc.depth_format.map(|f| wgpu::RenderBundleDepthStencil {
                        format: convert::texture_format(f),
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: desc.sample_count,
                    multiview: None,
                });
        self.counters.render_commands += commands.len() as u32;
        let mut skip = false;
        for command in commands {
            self.encode_render(&mut encoder, command, &mut skip);
        }
        // wgpu panics when `finish` fails, whatever the error handler, and
        // on a lost device it fails ("Parent device is lost"): the frame
        // goes on without the bundle, and the next one makes a new device.
        let bundle = (!self.lost()).then(|| {
            encoder.finish(&wgpu::RenderBundleDescriptor {
                label: Some(desc.label),
            })
        });
        RenderBundleId(self.render_bundles.insert(bundle))
    }

    fn destroy_render_bundle(&mut self, id: RenderBundleId) {
        self.render_bundles.remove(id.0);
    }

    fn submit(&mut self, passes: &[Pass]) {
        if let Some(l) = &mut self.ledger {
            l.submitted();
        }
        let started = timing::now();
        // The frame's surface texture, once a pass draws into it.
        for pass in passes {
            if let Pass::Render(render) = pass {
                for color in &render.colors {
                    self.acquire_surface(color.target);
                    if let Some(resolve) = color.resolve {
                        self.acquire_surface(resolve);
                    }
                }
            }
        }
        self.reserve_attribute_zero(passes.iter().flat_map(|pass| match pass {
            Pass::Render(render) => render.commands.as_slice(),
            _ => &[],
        }));
        // Debug groups and the encoder's label, for captures only.
        let groups = self.capturing || self.debug_groups;
        let joined = joined_compute(passes, groups);
        // While timing, each compute and render pass's first query; a
        // compute pass joined to the one before it times with it.
        let queries = match &mut self.timing {
            Some(timing) => timing.begin(
                &self.device,
                passes
                    .iter()
                    .zip(&joined)
                    .map(|(pass, &joined)| match pass {
                        Pass::Compute(p) if !joined => Some(p.label),
                        Pass::Render(p) => Some(p.label),
                        _ => None,
                    }),
            ),
            None => Vec::new(),
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: groups.then_some("passes"),
            });
        // wgpu-core takes a texture-to-texture copy's barriers as at most
        // two, which a texture whose layers a staged write left in another
        // state than the rest outnumbers (a glyph page growing): such a
        // submission's writes go in a command buffer of their own, as the
        // queue's pending writes would.
        if (passes.iter())
            .any(|p| matches!(p, Pass::CopyTexture { .. } | Pass::CopyTextureLevel { .. }))
        {
            let mut uploads = (self.device).create_command_encoder(&Default::default());
            self.encode_writes(&mut uploads);
            self.queue.submit([uploads.finish()]);
        }
        if groups {
            encoder.push_debug_group("uploads");
        }
        self.encode_writes(&mut encoder);
        if groups {
            encoder.pop_debug_group();
        }
        let timing = self.timing.as_ref();
        let writes = |i: usize| timing?.query((*queries.get(i)?)?);
        // The passes in `encoder`, and the debug groups open in it, which
        // the next encoder reopens.
        let mut encoded = 0;
        let mut open: Vec<&str> = Vec::new();
        for (i, pass) in passes.iter().enumerate() {
            let records = match pass {
                Pass::PushDebugGroup(_) | Pass::PopDebugGroup => false,
                Pass::Compute(_) => !joined[i],
                _ => true,
            };
            if records && encoded == PASSES_PER_ENCODER {
                for _ in &open {
                    encoder.pop_debug_group();
                }
                self.queue.submit([encoder.finish()]);
                encoder = (self.device).create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: groups.then_some("passes"),
                });
                for label in &open {
                    encoder.push_debug_group(label);
                }
                encoded = 0;
            }
            encoded += usize::from(records);
            match pass {
                Pass::PushDebugGroup(label) if groups => {
                    open.push(label);
                    encoder.push_debug_group(label);
                }
                Pass::PopDebugGroup if groups => {
                    open.pop();
                    encoder.pop_debug_group();
                }
                Pass::PushDebugGroup(_) | Pass::PopDebugGroup => {}
                Pass::CopyTexture {
                    source,
                    destination,
                    copies,
                } => {
                    let (source, destination) = (self.texture(*source), self.texture(*destination));
                    let levels = source.mip_level_count().min(destination.mip_level_count());
                    let mut copied = 0;
                    for (copy, level) in
                        copies.iter().flat_map(|c| (0..levels).map(move |l| (c, l)))
                    {
                        let at = |[x, y, z]: [u32; 3]| [x >> level, y >> level, z];
                        let [w, h, layers] = copy.size;
                        let copy = TextureCopy {
                            from: at(copy.from),
                            to: at(copy.to),
                            size: [(w >> level).max(1), (h >> level).max(1), layers],
                        };
                        copied +=
                            copy_box(&mut encoder, (source, level), (destination, level), &copy);
                    }
                    self.counters.bytes_copied += copied;
                }
                Pass::CopyTextureLevel {
                    source: (source, from),
                    destination: (destination, to),
                    copies,
                } => {
                    let (source, destination) = (self.texture(*source), self.texture(*destination));
                    let copied: u64 = (copies.iter())
                        .map(|c| copy_box(&mut encoder, (source, *from), (destination, *to), c))
                        .sum();
                    self.counters.bytes_copied += copied;
                }
                Pass::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    encoder.clear_buffer(&self.buffers[buffer.0], *offset, *size);
                }
                Pass::CopyBuffer {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } => {
                    self.counters.bytes_copied += size;
                    encoder.copy_buffer_to_buffer(
                        &self.buffers[source.0],
                        *source_offset,
                        &self.buffers[destination.0],
                        *destination_offset,
                        *size,
                    );
                }
                Pass::Compute(_) if joined[i] => {}
                Pass::Compute(compute) => {
                    // It and the compute passes joined to it.
                    let run = (passes[i + 1..].iter().zip(&joined[i + 1..]))
                        .take_while(|&(p, &joined)| match p {
                            Pass::Compute(_) => joined,
                            Pass::PushDebugGroup(_) | Pass::PopDebugGroup => !groups,
                            _ => false,
                        })
                        .filter_map(|(p, _)| match p {
                            Pass::Compute(c) => Some(c),
                            _ => None,
                        });
                    let commands = (std::iter::once(compute).chain(run))
                        .flat_map(|c| &c.commands)
                        .collect::<Vec<_>>();
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(compute.label),
                        timestamp_writes: writes(i).map(|(query_set, first)| {
                            wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: Some(first),
                                end_of_pass_write_index: Some(first + 1),
                            }
                        }),
                    });
                    // Past a pipeline that failed, its dispatches are left
                    // out until another is set.
                    let mut skip = false;
                    for command in commands {
                        match command {
                            ComputeCommand::SetPipeline(id) => {
                                skip = match &self.pipelines[id.0] {
                                    Compiled::Ready(compile::Built::Compute(pipeline)) => {
                                        pass.set_pipeline(pipeline);
                                        false
                                    }
                                    Compiled::Failed => true,
                                    _ => {
                                        panic!("compute pipeline {} used before it compiled", id.0)
                                    }
                                };
                            }
                            ComputeCommand::SetBindGroup { index, group } => {
                                pass.set_bind_group(*index, &self.bind_groups[group.0], &[]);
                            }
                            ComputeCommand::Dispatch { .. } if skip => {}
                            ComputeCommand::Dispatch { x, y, z } => {
                                pass.dispatch_workgroups(*x, *y, *z)
                            }
                        }
                    }
                }
                // Into a surface whose texture couldn't be acquired
                // (`take_surface` says why), drawn or resolved into.
                Pass::Render(render)
                    if render.colors.iter().any(|c| {
                        (Some(c.target).into_iter().chain(c.resolve))
                            .any(|id| self.textures[id.0].view().is_none())
                    }) => {}
                Pass::Render(render) => {
                    let colors: Vec<Option<wgpu::RenderPassColorAttachment>> = render
                        .colors
                        .iter()
                        .map(|color| {
                            Some(wgpu::RenderPassColorAttachment {
                                view: self.view(color.target),
                                depth_slice: None,
                                resolve_target: color.resolve.map(|id| self.view(id)),
                                ops: wgpu::Operations {
                                    load: convert::load_op(color.load, |[r, g, b, a]| {
                                        wgpu::Color { r, g, b, a }
                                    }),
                                    store: convert::store_op(color.store),
                                },
                            })
                        })
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(render.label),
                        color_attachments: &colors,
                        depth_stencil_attachment: render.depth.as_ref().map(|depth| {
                            let store = convert::store_op(depth.store);
                            // A stencil clears and stores as the depth does.
                            let stencil = (self.textures[depth.target.0].texture())
                                .is_some_and(|t| t.format().has_stencil_aspect());
                            wgpu::RenderPassDepthStencilAttachment {
                                view: self.view(depth.target),
                                depth_ops: Some(wgpu::Operations {
                                    load: convert::load_op(depth.load, |v| v),
                                    store,
                                }),
                                stencil_ops: stencil.then_some(wgpu::Operations {
                                    load: convert::load_op(depth.load, |_| 0),
                                    store,
                                }),
                            }
                        }),
                        timestamp_writes: writes(i).map(|(query_set, first)| {
                            wgpu::RenderPassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: Some(first),
                                end_of_pass_write_index: Some(first + 1),
                            }
                        }),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.counters.render_commands += render.commands.len() as u32;
                    let mut skip = false;
                    for command in &render.commands {
                        match command {
                            RenderCommand::ExecuteBundles(ids) => pass.execute_bundles(
                                ids.iter()
                                    .filter_map(|id| self.render_bundles[id.0].as_ref()),
                            ),
                            &RenderCommand::SetScissor([x, y, width, height]) => {
                                pass.set_scissor_rect(x, y, width, height)
                            }
                            &RenderCommand::SetViewport([x, y, width, height]) => {
                                pass.set_viewport(x, y, width, height, 0.0, 1.0)
                            }
                            &RenderCommand::SetStencilReference(reference) => {
                                pass.set_stencil_reference(reference)
                            }
                            command => self.encode_render(&mut pass, command, &mut skip),
                        }
                    }
                }
            }
        }
        if groups {
            encoder.push_debug_group("readback");
        }
        self.encode_armed_copy(&mut encoder);
        if groups {
            encoder.pop_debug_group();
        }
        self.last = Some(self.queue.submit([encoder.finish()]));
        // The writes reading them are issued.
        #[cfg(target_arch = "wasm32")]
        for image in std::mem::take(&mut self.released_images) {
            self.release_host_image(image);
        }
        #[cfg(target_arch = "wasm32")]
        for bytes in std::mem::take(&mut self.released_bytes) {
            self.host_bytes.remove(bytes.0);
        }
        if let Some(timing) = &mut self.timing {
            timing.end(timing::now() - started);
        }
    }
}

/// Encodes a copy of `copy`'s box, in texels of the levels given, from
/// `source`'s level into `destination`'s: the bytes it copies.
fn copy_box(
    encoder: &mut wgpu::CommandEncoder,
    (source, from): (&wgpu::Texture, u32),
    (destination, to): (&wgpu::Texture, u32),
    copy: &TextureCopy,
) -> u64 {
    let at = |[x, y, z]: [u32; 3]| wgpu::Origin3d { x, y, z };
    let [width, height, layers] = copy.size;
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            mip_level: from,
            origin: at(copy.from),
            ..source.as_image_copy()
        },
        wgpu::TexelCopyTextureInfo {
            mip_level: to,
            origin: at(copy.to),
            ..destination.as_image_copy()
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
    );
    let texel = u64::from(convert::texel_bytes(source.format()));
    u64::from(width) * u64::from(height) * u64::from(layers) * texel
}

/// Per pass, whether it is a compute pass joined to the one before it:
/// compute passes with only debug group markers between them (none while
/// `groups` keeps them) encode as one. wgpu-core gives each pass a command
/// buffer of its own, which on Metal costs the driver a command buffer and
/// an encoder, and the GPU a gap between them; WebGPU orders a pass's
/// dispatches as it orders passes, so a frame without terrain makes its
/// cull, image mips and symbol placements in one.
fn joined_compute(passes: &[Pass], groups: bool) -> Vec<bool> {
    let mut after_compute = false;
    passes
        .iter()
        .map(|pass| match pass {
            Pass::Compute(_) => std::mem::replace(&mut after_compute, true),
            Pass::PushDebugGroup(_) | Pass::PopDebugGroup if !groups => false,
            _ => {
                after_compute = false;
                false
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_takes_the_most_its_flags_and_storage_bindings_allow() {
        use wgpu::DownlevelFlags as F;
        let all = F::all();
        let storage = |buffers| wgpu::Limits {
            max_storage_buffers_per_shader_stage: buffers,
            ..wgpu::Limits::defaults()
        };
        let cases = [
            (all, wgpu::Limits::defaults(), false, FramePath::Gpu),
            (
                all - F::INDIRECT_EXECUTION,
                storage(8),
                false,
                FramePath::Cpu,
            ),
            // Vulkan's least: 4 storage buffers a stage, short of the cull.
            (
                all,
                wgpu::Limits::downlevel_defaults(),
                false,
                FramePath::CpuTex,
            ),
            (all, storage(7), false, FramePath::CpuTex),
            (
                all - F::VERTEX_STORAGE,
                storage(8),
                false,
                FramePath::CpuTex,
            ),
            (all, storage(3), false, FramePath::Webgl2),
            (
                all - F::COMPUTE_SHADERS,
                storage(8),
                false,
                FramePath::Webgl2,
            ),
            (all, NO_COMPUTE, false, FramePath::Webgl2),
            (all, wgpu::Limits::defaults(), true, FramePath::Webgl2),
        ];
        for (flags, limits, gl, path) in cases {
            assert_eq!(frame_path(flags, &limits, gl), path, "{flags:?} gl {gl}");
        }
    }

    #[test]
    fn a_shader_nonce_reaches_every_entry_point() {
        let source = "fn unused() -> vec4<f32> { return vec4<f32>(2.0); }
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i));
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
@compute @workgroup_size(64)
fn cs() {}
";
        let module = naga::front::wgsl::parse_str(&nonced(source, 1_234_567)).expect("parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("valid");
        assert_eq!(module.entry_points.len(), 3);
        for entry in &module.entry_points {
            let f = &entry.function;
            let nonce = (f.local_variables.iter())
                .find(|(_, v)| v.name.as_deref() == Some("orb_nonce"))
                .and_then(|(_, v)| v.init)
                .map(|init| &f.expressions[init]);
            assert!(
                matches!(
                    nonce,
                    Some(naga::Expression::Literal(naga::Literal::U32(1_234_567)))
                ),
                "{}: {nonce:?}",
                entry.name
            );
        }
    }

    #[test]
    fn gl_arrays_take_neither_a_2d_nor_a_cube_target() {
        let layers = |width, height, array_layers| {
            gl_array_layers(
                &TextureDesc {
                    label: "array",
                    width,
                    height,
                    array_layers,
                    dimension: TextureViewDimension::D2Array,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    sample_count: 1,
                    mip_levels: 1,
                },
                true,
            )
        };
        assert_eq!(layers(258, 258, 1), 2);
        assert_eq!(layers(258, 258, 5), 5);
        assert_eq!(layers(258, 258, 6), 7);
        assert_eq!(layers(258, 258, 12), 13);
        assert_eq!(layers(256, 512, 6), 6);
        // Within WebGL 2's 256 layers wherever the asked count is.
        for n in 1..=256 {
            let made = layers(258, 258, n);
            assert!(
                made > 1 && !made.is_multiple_of(6) && made <= 256,
                "{n} layers made {made}"
            );
        }
    }
}
