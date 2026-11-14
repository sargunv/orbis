//! [`orb_hal::Gpu`] on wgpu: Vulkan, Metal, D3D12 and GLES natively, chosen
//! at runtime; in the browser, WebGPU or WebGL 2, chosen by the build
//! (the `webgpu` and `gl` features).
//!
//! [`WgpuGpu`] owns every wgpu handle behind the ids it hands out. Hosts
//! register their own targets (a surface frame, an offscreen texture) with
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
#[cfg(not(target_arch = "wasm32"))]
mod surface;
mod texture_writes;
mod timing;
mod writes;

pub use convert::from_texture_format;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use metal::{MetalTexture, metal_texture};
use orb_hal::*;
use slots::Slots;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::{RawSurface, Surface, SurfaceError};
pub use timing::SubmitTimes;
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

/// The frame driver a native device's renderers take: `gpu` (the default
/// where the device has it), or `cpu` and `cpu-tex`
/// ([`WgpuGpu::force_cpu_driver`]).
#[cfg(not(target_arch = "wasm32"))]
const DRIVER: &str = "ORB_DRIVER";

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
pub fn device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
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
/// inter-stage variables), with buffers and storage bindings as large as it
/// allows up to 1 GiB. The scene's pools grow within one storage binding
/// to what the tiles in view need: at WebGPU's default 128 MiB, a pitched
/// view over a dense city ran out.
pub fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let offered = adapter.limits();
    let mut limits = wgpu::Limits::default().or_worse_values_from(&offered);
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
    /// Array textures count the layers GL makes them with
    /// ([`gl_array_layers`]) against [`Limits::max_texture_bytes`]: on GL,
    /// and under [`force_webgl2`](Self::force_webgl2), which stands in for it.
    gl_layers: bool,
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
    textures: Slots<Texture>,
    shader_modules: Slots<compile::Shared<compile::Module>>,
    bind_group_layouts: Slots<BindGroupLayout>,
    bind_groups: Slots<wgpu::BindGroup>,
    /// `None` while compiling.
    render_pipelines: Slots<Option<wgpu::RenderPipeline>>,
    compute_pipelines: Slots<Option<wgpu::ComputePipeline>>,
    /// Zeros that WebGL 2's shaders read at `@location(0)`, a `u32` per
    /// instance ([`Caps::gl_shaders`], #179): set with every pipeline,
    /// and grown to the most instances a pass or bundle draws.
    attribute_zero: Option<wgpu::Buffer>,
    compiler: compile::Compiler,
    /// Pipelines asked for and not yet compiled, background ones aside.
    pending: u32,
    /// The surface texture a frame draws into, once a pass draws into it
    /// ([`register_surface`](Self::register_surface)).
    #[cfg(target_arch = "wasm32")]
    surface: Option<PendingSurface>,
    /// Later requests are background ones
    /// ([`Gpu::set_background_compiles`]).
    background: bool,
    /// Background pipelines compiling that no frame asked for.
    background_pending: rustc_hash::FxHashSet<PipelineId>,
    /// Pipelines compiled that a frame asked for ([`PipelineCounts::made`]).
    made: u32,
    /// A compile that ran past the watchdog's limit, or failed: the device
    /// is lost.
    hung: Option<String>,
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
    /// What the device did ([`Gpu::counters`]), counted from `&self` too:
    /// where held writes are issued.
    counters: std::cell::Cell<GpuCounters>,
    /// The resources it holds by label, when `ORB_GPU_LEDGER` asks.
    ledger: Option<Box<ledger::Ledger>>,
    /// Makes shader modules unique to it ([`WgpuGpu::set_shader_nonce`]).
    shader_nonce: Option<u32>,
    /// Small buffer writes, copied in at the next submission.
    writes: writes::Writes,
    /// Texture writes since the last submission, those sharing a block
    /// with an earlier one issued in waves at the next.
    texture_writes: std::sync::Mutex<texture_writes::TextureWrites>,
    /// Readback buffers back from their last readback, for the next of
    /// their size: a new one maps fresh pages each time.
    readbacks: offscreen::Readbacks,
    /// The texture the next submission copies out after its passes, and
    /// the copy the last one made, for the readback after it.
    copy_next: Option<TextureId>,
    copied: std::sync::Mutex<Option<(TextureId, offscreen::Readback)>>,
    /// Frames started ([`WgpuGpu::start_frame`]), and those to capture by
    /// number, while one is asked for; whether a capture is running.
    frames_started: u64,
    capture: Option<std::ops::Range<u64>>,
    capturing: bool,
    /// Whether submissions carry their debug groups outside captures too
    /// ([`WgpuGpu::set_debug_groups`]).
    debug_groups: bool,
    /// Set when the device made by `offscreen` is lost.
    #[cfg(not(target_arch = "wasm32"))]
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Which thread wgpu told of the loss, and what the driver said.
    #[cfg(not(target_arch = "wasm32"))]
    loss: compile::Loss,
    /// D3D12's shader compiler, on a D3D12 device `offscreen` made.
    #[cfg(not(target_arch = "wasm32"))]
    shader_compiler: Option<String>,
}

/// A write from core memory: gathered when small, or when it overlaps a
/// gathered one it must land after; otherwise staged natively, and on the
/// web straight to the queue. Whether it was issued now, not gathered.
fn write_core(
    writes: &mut writes::Writes,
    (device, queue): (&wgpu::Device, &wgpu::Queue),
    buffers: &Slots<wgpu::Buffer>,
    (id, idle): (BufferId, bool),
    offset: u64,
    data: &[u8],
) -> bool {
    // On WebGL 2 every write is gathered, whatever its size: a
    // `bufferSubData` into a buffer the GPU still reads makes ANGLE's Metal
    // back end copy the whole buffer into a new one, and a frame's writes
    // into the scene's arenas then cost hundreds of milliseconds (#140).
    // Gathered, they land in a staging buffer the GPU is done with and are
    // copied into place on the GPU. A buffer the GPU no longer reads is
    // written in place: a copy on the GPU into an index buffer makes ANGLE
    // wait for it before it next reads the buffer for restart indices.
    if !idle
        && (cfg!(webgl)
            || data.len() <= writes::GATHERED
            || writes.overlaps(id, offset, data.len() as u64))
    {
        writes.gather(id, offset, data);
        return false;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if writes.stage(device, id, offset, data) {
        return true;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    queue.write_buffer(&buffers[id.0], offset, data);
    true
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

#[derive(Debug)]
struct Texture {
    /// `None` for views the host registered and owns.
    texture: Option<wgpu::Texture>,
    /// `None` for a surface's texture not yet acquired
    /// ([`WgpuGpu::register_surface`]).
    view: Option<wgpu::TextureView>,
    /// A view of these mip levels of another entry's texture, which it
    /// doesn't own.
    levels: Option<std::ops::Range<u32>>,
}

/// A surface's texture for the frame ([`WgpuGpu::register_surface`]).
#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
struct PendingSurface {
    id: TextureId,
    surface: std::rc::Rc<wgpu::Surface<'static>>,
    /// Configured so before its texture is acquired.
    configure: Option<wgpu::SurfaceConfiguration>,
    /// Once a pass drew into it.
    frame: Option<Result<wgpu::SurfaceTexture, String>>,
}

/// The layers wgpu's GL back end makes an array texture of `desc` with,
/// which picks the GL target from the layer count alone: one layer is a
/// 2D texture, and a square one of 6n layers a cube map (an array of them
/// past 6, an enum WebGL 2 lacks), both incomplete to an array's sampler,
/// which reads black (pages, atlases). Such an array gets a layer more
/// that it never uses.
fn gl_array_layers(desc: &TextureDesc<'_>) -> u32 {
    let layers = desc.array_layers;
    let cube = desc.width == desc.height && desc.sample_count == 1 && layers.is_multiple_of(6);
    if layers == 1 || cube {
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

/// Whether `device`, a GL one, takes GLSL ES (WebGL 2's or OpenGL ES's
/// context, not desktop OpenGL's).
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(feature = "gl", target_os = "android")),
    allow(unused_variables)
)]
fn gl_takes_glsl_es(device: &wgpu::Device) -> bool {
    // SAFETY: only reads a flag of the device's.
    #[cfg(any(feature = "gl", target_os = "android"))]
    return unsafe { device.as_hal::<wgpu::hal::api::Gles>() }.is_some_and(|d| d.is_embedded());
    #[cfg(not(any(feature = "gl", target_os = "android")))]
    false
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

/// Whether `device`, a GL one, clips depth from 0 to 1 as the shaders
/// write it (`EXT_clip_control`; Chrome, Safari and ANGLE have it, Firefox
/// doesn't), rather than wgpu remapping it into GL's -1 to 1.
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(feature = "gl", target_os = "android")),
    allow(unused_variables)
)]
fn gl_clips_zero_to_one(device: &wgpu::Device) -> bool {
    // SAFETY: only reads a flag of the device's.
    #[cfg(any(feature = "gl", target_os = "android"))]
    return unsafe { device.as_hal::<wgpu::hal::api::Gles>() }
        .is_some_and(|d| d.zero_to_one_depth());
    #[cfg(not(any(feature = "gl", target_os = "android")))]
    false
}

/// Whether `device`, a GL one, gives a flat varying its triangle's last
/// vertex's value, as GL's convention has it, found by drawing a triangle
/// into a pixel of its own and reading it back: not every GL keeps the
/// convention (Chrome's WebGL 2 on a Mac, ANGLE on Metal, gives the first
/// vertex's; the ANGLE a native map loads, the last's). It runs before the
/// device draws anything, in the vertex array wgpu binds once, whose
/// attribute it enables and disables again, and leaves bound what wgpu
/// binds again for each submission.
#[allow(unsafe_code)]
#[cfg_attr(
    not(any(feature = "gl", target_os = "android")),
    allow(unused_variables)
)]
fn gl_flat_from_last(device: &wgpu::Device) -> bool {
    #[cfg(any(feature = "gl", target_os = "android"))]
    {
        use glow::HasContext;
        let es = gl_takes_glsl_es(device);
        // SAFETY: wgpu's GL device, whose context this locks, before it
        // records anything.
        let Some(device) = (unsafe { device.as_hal::<wgpu::hal::api::Gles>() }) else {
            return true;
        };
        let gl = device.context().lock();
        let version = if es {
            "#version 300 es\nprecision highp float;"
        } else {
            "#version 330 core"
        };
        let vertex = format!(
            "{version}\nlayout(location = 0) in vec2 p;\nflat out uint v;\nvoid main() {{\n    \
             v = uint(gl_VertexID);\n    gl_Position = vec4(p, 0.0, 1.0);\n}}\n"
        );
        let fragment = format!(
            "{version}\nflat in uint v;\nout vec4 colour;\nvoid main() {{\n    \
             colour = vec4(float(v) / 2.0, 0.0, 0.0, 1.0);\n}}\n"
        );
        let mut last = true;
        unsafe {
            if let (Ok(program), Ok(renderbuffer), Ok(framebuffer), Ok(buffer)) = (
                gl.create_program(),
                gl.create_renderbuffer(),
                gl.create_framebuffer(),
                gl.create_buffer(),
            ) {
                let shaders: Vec<_> = [
                    (glow::VERTEX_SHADER, vertex),
                    (glow::FRAGMENT_SHADER, fragment),
                ]
                .iter()
                .filter_map(|(kind, source)| {
                    let shader = gl.create_shader(*kind).ok()?;
                    gl.shader_source(shader, source);
                    gl.compile_shader(shader);
                    gl.attach_shader(program, shader);
                    Some(shader)
                })
                .collect();
                gl.link_program(program);
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));
                gl.renderbuffer_storage(glow::RENDERBUFFER, glow::RGBA8, 1, 1);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::RENDERBUFFER,
                    Some(renderbuffer),
                );
                if shaders.len() == 2 && gl.get_program_link_status(program) {
                    for state in [
                        glow::SCISSOR_TEST,
                        glow::BLEND,
                        glow::DEPTH_TEST,
                        glow::STENCIL_TEST,
                        glow::CULL_FACE,
                        glow::RASTERIZER_DISCARD,
                    ] {
                        gl.disable(state);
                    }
                    gl.color_mask(true, true, true, true);
                    gl.viewport(0, 0, 1, 1);
                    gl.clear_color(0.0, 0.0, 1.0, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT);
                    gl.use_program(Some(program));
                    // From an array, not none: Firefox warns of emulating
                    // attribute 0 otherwise.
                    let corners: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
                    let bytes: Vec<u8> = corners.iter().flat_map(|c| c.to_ne_bytes()).collect();
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &bytes, glow::STATIC_DRAW);
                    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 0, 0);
                    gl.enable_vertex_attrib_array(0);
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                    gl.disable_vertex_attrib_array(0);
                    gl.bind_buffer(glow::ARRAY_BUFFER, None);
                    let mut pixel = [0u8; 4];
                    gl.read_pixels(
                        0,
                        0,
                        1,
                        1,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut pixel)),
                    );
                    // 0 from the first vertex, 255 from the last; blue
                    // where it drew nothing, and GL's convention stands.
                    last = pixel[2] > 127 || pixel[0] > 191;
                }
                gl.use_program(None);
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.bind_renderbuffer(glow::RENDERBUFFER, None);
                for shader in shaders {
                    gl.delete_shader(shader);
                }
                gl.delete_program(program);
                gl.delete_framebuffer(framebuffer);
                gl.delete_renderbuffer(renderbuffer);
                gl.delete_buffer(buffer);
            }
        }
        last
    }
    #[cfg(not(any(feature = "gl", target_os = "android")))]
    true
}

impl WgpuGpu {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        // GLES runs the CPU frame driver, as WebGL 2 does, even where it
        // offers compute (#140); WebGL 2's limits already say so.
        #[cfg(not(target_arch = "wasm32"))]
        let gl = device.adapter_info().backend == wgpu::Backend::Gl;
        #[cfg(target_arch = "wasm32")]
        let gl = cfg!(webgl);
        let limits = device.limits();
        let caps = Caps {
            limits: Limits {
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_texture_dimension_2d: (limits.max_texture_dimension_2d)
                    .min(wgpu::Limits::default().max_texture_dimension_2d),
                max_texture_array_layers: limits.max_texture_array_layers,
                max_texture_bytes: if gl { GL_TEXTURE_BYTES } else { u64::MAX },
                min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            },
            timestamp_query: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            // Not on GL: its shaders are WebGL 2's, which has none, though
            // native GLES (Mesa's) offers them.
            clip_distances: !gl && device.features().contains(wgpu::Features::CLIP_DISTANCES),
            // WebGPU and the native back ends have all of these; a
            // downlevel device (wgpu's GL back end) reports no compute
            // and no vertex-stage storage in its limits.
            compute: !gl && limits.max_compute_invocations_per_workgroup > 0,
            indirect: !gl && limits.max_compute_invocations_per_workgroup > 0,
            vertex_storage: !gl && limits.max_storage_buffers_per_shader_stage > 0,
            // WebGPU guarantees both; elsewhere `read_formats` asks the
            // adapter (WebGL 2 has them with `EXT_color_buffer_float`).
            float_targets: cfg!(webgpu),
            zero_to_one_depth: !gl || gl_clips_zero_to_one(&device),
            depth32_stencil8: (device.features()).contains(wgpu::Features::DEPTH32FLOAT_STENCIL8),
            // `ORB_COMPILES_HOLD_PRESENTS` has a native device stand in for
            // Firefox's: what a map does there, in tests and benches.
            #[cfg(webgpu)]
            compiles_hold_presents: compile::compiles_hold_presents(),
            #[cfg(not(target_arch = "wasm32"))]
            compiles_hold_presents: std::env::var_os("ORB_COMPILES_HOLD_PRESENTS").is_some(),
            #[cfg(webgl)]
            compiles_hold_presents: false,
            flat_from_last: gl && gl_flat_from_last(&device),
        };
        let glsl_es = gl && gl_takes_glsl_es(&device);
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
            gl_layers: gl,
            glsl_es,
            texture_usages: platform_texture_usages(),
            buffers: Slots::default(),
            idle_writes: Default::default(),
            textures: Slots::default(),
            shader_modules: Slots::default(),
            bind_group_layouts: Slots::default(),
            bind_groups: Slots::default(),
            render_pipelines: Slots::default(),
            compute_pipelines: Slots::default(),
            attribute_zero: None,
            compiler: compile::Compiler::new(),
            pending: 0,
            #[cfg(target_arch = "wasm32")]
            surface: None,
            background: false,
            background_pending: Default::default(),
            made: 0,
            hung: None,
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
            texture_writes: Default::default(),
            readbacks: Default::default(),
            copy_next: None,
            copied: Default::default(),
            frames_started: 0,
            capture: None,
            capturing: false,
            debug_groups: false,
            #[cfg(not(target_arch = "wasm32"))]
            lost: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            loss: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            shader_compiler: None,
        };
        #[cfg(not(target_arch = "wasm32"))]
        match std::env::var(DRIVER).as_deref() {
            Ok("cpu") => gpu.force_cpu_driver(false),
            Ok("cpu-tex") => gpu.force_cpu_driver(true),
            Ok("webgl2") => gpu.force_webgl2(),
            Ok("" | "gpu") | Err(_) => {}
            Ok(other) => panic!("{DRIVER}={other}: expected gpu, cpu, cpu-tex or webgl2"),
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(side) = std::env::var(TEXTURE_SIDE) {
            let side: u32 = (side.parse()).unwrap_or_else(|e| panic!("{TEXTURE_SIDE}={side}: {e}"));
            let limits = &mut gpu.caps.limits;
            limits.max_texture_dimension_2d = limits.max_texture_dimension_2d.min(side);
        }
        gpu
    }

    /// Has a device report everything WebGL 2 lacks, compute and clip
    /// distances too ([`force_cpu_driver`](Self::force_cpu_driver) with
    /// textures keeps them): WebGL 2's whole path on a native device, its
    /// drawn mips and its gaps included, short of GL itself.
    /// `ORB_DRIVER=webgl2` sets it for every native device.
    pub fn force_webgl2(&mut self) {
        self.force_cpu_driver(true);
        self.caps.compute = false;
        self.caps.clip_distances = false;
        // What wgpu asks of WebGL 2 by default (`device_limits`).
        let limits = &mut self.caps.limits;
        limits.max_texture_dimension_2d = limits.max_texture_dimension_2d.min(8192);
        limits.max_texture_array_layers = limits.max_texture_array_layers.min(256);
        limits.max_texture_bytes = limits.max_texture_bytes.min(GL_TEXTURE_BYTES);
        self.gl_layers = true;
        limits.min_uniform_buffer_offset_alignment =
            limits.min_uniform_buffer_offset_alignment.max(256);
    }

    /// Has a device that draws GPU-driven report what WebGL 2 lacks, so
    /// renderers made on it after take the CPU frame driver (no indirect
    /// draws) and, with `textures`, keep their tables in data textures (no
    /// vertex-stage storage): the WebGL 2 path's logic on a native device,
    /// for tests that hold it to the GPU-driven one (#140).
    /// `ORB_DRIVER=cpu` or `cpu-tex` sets it for every native device.
    pub fn force_cpu_driver(&mut self, textures: bool) {
        self.caps.indirect = false;
        self.caps.vertex_storage &= !textures;
    }

    /// Copies the gathered writes into place ahead of what `encoder`
    /// records next, after issuing the held texture writes.
    /// On a device known to be lost the writes are dropped; one whose
    /// staging buffer comes back invalid is taken as lost.
    fn encode_writes(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.known_lost() {
            self.writes.discard();
            return;
        }
        self.issue_texture_writes();
        #[cfg(not(target_arch = "wasm32"))]
        let issued = self.writes.encode(
            (&self.device, &self.queue),
            |id| self.buffers[id.0].clone(),
            |id| self.texture(id).clone(),
            encoder,
        );
        #[cfg(target_arch = "wasm32")]
        let issued = Some(self.writes.encode(
            &self.device,
            &self.queue,
            |id| self.buffers[id.0].clone(),
            encoder,
        ));
        match issued {
            Some(issued) => self.count(|c| {
                c.upload_calls += issued.calls;
                c.texture_write_submissions += issued.submissions;
            }),
            #[cfg(not(target_arch = "wasm32"))]
            None => self.lost.store(true, std::sync::atomic::Ordering::Relaxed),
            #[cfg(target_arch = "wasm32")]
            None => {}
        }
    }

    /// Adds to the device's counters.
    fn count(&self, add: impl FnOnce(&mut GpuCounters)) {
        let mut counters = self.counters.get();
        add(&mut counters);
        self.counters.set(counters);
    }

    /// Issues the held texture writes, wave by wave, each wave before the
    /// last its own submission (see `texture_writes`): natively staged,
    /// each in its wave, and on the web to the queue. GL orders its
    /// texture uploads as they come, so there the waves share one, in
    /// order, natively as on the web: each submission is a flush, which
    /// ANGLE's Metal back end commits as a command buffer of its own
    /// (#140), and the CPU frame driver's data tables write a texture's
    /// layer many times a frame.
    fn issue_texture_writes(&self) {
        let mut held = self.texture_writes.lock().expect("texture writes poisoned");
        let wave = std::cell::Cell::new(0);
        held.issue(
            |write, texels| self.issue_texture_write(write, texels, wave.get()),
            || {
                if self.gl {
                    return;
                }
                wave.set(wave.get() + 1);
                if cfg!(target_arch = "wasm32") {
                    self.queue.submit([]);
                    self.count(|c| c.texture_write_submissions += 1);
                }
            },
        );
    }

    /// Issues a texture write in `wave`: natively staged, else (or where
    /// staging failed, on a lost device) to the queue.
    fn issue_texture_write(&self, write: &texture_writes::TextureWrite, texels: &[u8], wave: u32) {
        self.count(|c| c.upload_calls += 1);
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
        &self,
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

    /// Marks the start of a frame: its submissions, and a host target's
    /// acquire and present around them. Starts a capture asked for.
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
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
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
        while self.pending > 0 && self.hung.is_none() {
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
    pub fn read_formats(&mut self, adapter: &wgpu::Adapter) {
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
        self.compiler.inject_hang();
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

    /// The compile that ran past the watchdog's limit, or failed, if one
    /// did: the driver is taken to have hung, and the device is lost.
    pub fn hung(&self) -> Option<&str> {
        self.hung.as_deref()
    }

    /// On a lost device, lets the compiles running finish (up to `wait`)
    /// and takes them, so a pipeline whose compile lost the device is
    /// [`hung`](Self::hung): a loss is checked for before frames take
    /// their compiles. The caller waits once per loss: a compile still
    /// running then (a slow compiler, a hung driver) mustn't stall every
    /// later check.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn settle_loss(&mut self, wait: std::time::Duration) {
        if self.hung.is_none() && self.lost() {
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
            if !self.background_pending.remove(&done.id) {
                self.pending -= 1;
                self.made += 1;
            }
            let built = match done.built {
                Ok(built) => built,
                // A pipeline the driver can't make goes the way of a hung
                // compile: the device is lost, and recovery demotes it.
                Err(e) => {
                    self.hung
                        .get_or_insert_with(|| format!("pipeline {}: {e}", done.label));
                    #[cfg(not(target_arch = "wasm32"))]
                    self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
            };
            // A pipeline destroyed while it compiled is dropped.
            match built {
                compile::Built::Render(slot, p) if self.render_pipelines.contains(slot) => {
                    self.render_pipelines.set(slot, Some(p));
                }
                compile::Built::Compute(slot, p) if self.compute_pipelines.contains(slot) => {
                    self.compute_pipelines.set(slot, Some(p));
                }
                _ => {}
            }
            self.count(|c| c.pipelines_compiled += 1);
            self.compiled.push((done.label, done.took));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.hung.is_none()
            && self.pending > 0
            && let Some(label) = self.compiler.hung()
        {
            self.hung = Some(label);
            self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
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

    fn submit_job(&mut self, job: compile::Job, layouts: &[BindGroupLayoutId]) {
        let background = self.background;
        if background {
            self.background_pending.insert(job.id());
        } else {
            self.pending += 1;
            self.gate_background();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = layouts;
            if background {
                self.compiler.submit_background(&self.device, job);
            } else {
                self.compiler.submit(&self.device, job);
            }
        }
        #[cfg(webgpu)]
        {
            let raw = layouts
                .iter()
                .map(|id| self.bind_group_layouts[id.0].raw.clone())
                .collect();
            self.compiler.submit(&self.device, raw, job, background);
        }
        #[cfg(webgl)]
        {
            let _ = layouts;
            self.compiler.submit(&self.device, job, background);
        }
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

    /// Whether the device `offscreen` made was lost.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the device is known to be lost: natively, as `lost`; in
    /// the browser, whose failures don't stop the page, never.
    fn known_lost(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.lost();
        #[cfg(target_arch = "wasm32")]
        false
    }

    /// What the driver said when the device was lost, if it was.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lost_reason(&self) -> Option<String> {
        self.loss.reason.get().cloned()
    }

    /// Loses the device on purpose (`destroy`), to exercise recovery.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lose(&self) {
        self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
        self.device.destroy();
    }

    /// Marks the device lost when the driver loses it, and takes the
    /// errors calls on a lost device raise as part of the loss: a loss
    /// can come in the middle of a frame (in a render bundle's `finish`, at
    /// a submit), and the frame's remaining calls each fail with "Parent
    /// device is lost", which wgpu's default handler would panic on. Any
    /// other uncaptured error still panics, as it would without this.
    #[cfg(not(target_arch = "wasm32"))]
    fn watch_loss(&self) {
        let loss = self.loss.clone();
        self.compiler.watch_loss(loss.clone());
        let lost = self.lost.clone();
        self.device.set_device_lost_callback(move |why, message| {
            loss.reason.get_or_init(|| format!("{why:?}: {message}"));
            loss.thread.get_or_init(|| std::thread::current().id());
            lost.store(true, std::sync::atomic::Ordering::Relaxed);
        });
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
        TextureId(self.textures.insert(Texture {
            texture: None,
            view: Some(view),
            levels: None,
        }))
    }

    /// A target for a frame into `surface`, whose texture is acquired
    /// when a pass first draws into it, so a frame that draws nothing
    /// (one held while compiles run, [`Gpu::presents_wait`]) presents
    /// nothing. [`take_surface`](Self::take_surface) takes the texture.
    /// The surface takes `configure`, if given, before its texture.
    #[cfg(target_arch = "wasm32")]
    pub fn register_surface(
        &mut self,
        surface: std::rc::Rc<wgpu::Surface<'static>>,
        configure: Option<wgpu::SurfaceConfiguration>,
    ) -> TextureId {
        let id = TextureId(self.textures.insert(Texture {
            texture: None,
            view: None,
            levels: None,
        }));
        self.surface = Some(PendingSurface {
            id,
            surface,
            configure,
            frame: None,
        });
        id
    }

    /// The texture the frame acquired for `id` (`None` if no pass drew
    /// into it), or why acquiring it failed; `id` is released.
    #[cfg(target_arch = "wasm32")]
    pub fn take_surface(&mut self, id: TextureId) -> Option<Result<wgpu::SurfaceTexture, String>> {
        let pending = self.surface.take().filter(|s| s.id == id);
        self.textures.remove(id.0);
        pending.and_then(|s| s.frame)
    }

    /// Acquires the pending surface's texture if a pass draws into it
    /// (`id`) and it wasn't yet.
    #[cfg(target_arch = "wasm32")]
    fn acquire_surface(&mut self, id: TextureId) {
        let Some(pending) = (self.surface.as_mut()).filter(|s| s.id == id && s.frame.is_none())
        else {
            return;
        };
        if let Some(config) = pending.configure.take() {
            pending.surface.configure(&self.device, &config);
        }
        let frame = match pending.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Ok(frame),
            other => Err(format!("canvas texture: {other:?}")),
        };
        if let Ok(frame) = &frame {
            let view = (frame.texture).create_view(&wgpu::TextureViewDescriptor::default());
            self.textures[id.0].view = Some(view);
        }
        pending.frame = Some(frame);
    }

    /// The view of a texture a pass draws into.
    fn view(&self, id: TextureId) -> &wgpu::TextureView {
        (self.textures[id.0].view.as_ref()).expect("a surface's texture acquired for its pass")
    }

    /// The wgpu texture behind an id created with [`Gpu::create_texture`].
    pub fn texture(&self, id: TextureId) -> &wgpu::Texture {
        self.textures[id.0]
            .texture
            .as_ref()
            .expect("texture is a registered view")
    }

    /// The wgpu buffer behind an id.
    pub fn buffer(&self, id: BufferId) -> &wgpu::Buffer {
        &self.buffers[id.0]
    }

    /// Ends a frame drawn into a texture: waits until no more than
    /// `in_flight` frames are unfinished, as a swapchain does.
    pub fn end_frame(&mut self, in_flight: usize) {
        self.frames.extend(self.last.take());
        // wgpu panics when a wait fails, as it does on a lost device.
        if self.known_lost() {
            self.frames.clear();
        }
        while self.frames.len() > in_flight {
            let index = self.frames.pop_front();
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: index,
                timeout: None,
            });
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

    /// Encodes a command bundles may hold.
    fn encode_render<'p>(
        &'p self,
        pass: &mut impl wgpu::util::RenderEncoder<'p>,
        command: &'p RenderCommand,
    ) {
        match command {
            RenderCommand::SetPipeline(id) => {
                pass.set_pipeline(
                    self.render_pipelines[id.0]
                        .as_ref()
                        .expect("a pipeline used before it compiled"),
                );
                if let Some(zeros) = &self.attribute_zero {
                    pass.set_vertex_buffer(0, Some(zeros.slice(..)));
                }
            }
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
        self.count(|c| {
            c.buffers_created += 1;
            c.bytes_allocated += desc.size;
        });
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
        let issued = write_core(
            &mut self.writes,
            (&self.device, &self.queue),
            &self.buffers,
            (id, self.idle_writes.contains(&id.0)),
            offset,
            data,
        );
        self.count(|c| {
            c.buffer_bytes_from_core += data.len() as u64;
            c.upload_calls += u32::from(issued);
        });
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        let len = range.end - range.start;
        match &self.host_bytes[bytes.0] {
            HostBytes::Owned(data) => {
                let range = range.start as usize..range.end as usize;
                let issued = write_core(
                    &mut self.writes,
                    (&self.device, &self.queue),
                    &self.buffers,
                    (id, self.idle_writes.contains(&id.0)),
                    offset,
                    &data[range],
                );
                self.count(|c| {
                    c.buffer_bytes_from_core += len;
                    c.upload_calls += u32::from(issued);
                });
            }
            #[cfg(webgl)]
            HostBytes::Js(data) => {
                let data = js_sys::Uint8Array::new_with_byte_offset_and_length(
                    data,
                    range.start as u32,
                    len as u32,
                )
                .to_vec();
                let issued = write_core(
                    &mut self.writes,
                    (&self.device, &self.queue),
                    &self.buffers,
                    (id, self.idle_writes.contains(&id.0)),
                    offset,
                    &data,
                );
                self.count(|c| {
                    c.buffer_bytes_from_host += len;
                    c.upload_calls += u32::from(issued);
                });
            }
            #[cfg(webgpu)]
            HostBytes::Js(data) => {
                self.count(|c| {
                    c.buffer_bytes_from_host += len;
                    c.upload_calls += 1;
                });
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
                        data,
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
        if self
            .texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .reads_bytes(bytes)
        {
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
        match &self.host_bytes[bytes.0] {
            HostBytes::Owned(data) => {
                let texels = data[offset as usize..(offset + len) as usize].to_vec();
                self.write_texture(region, &texels);
            }
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(_) => {
                self.count(|c| c.texel_bytes_from_host += len);
                let held = self
                    .texture_writes
                    .get_mut()
                    .expect("texture writes poisoned")
                    .bytes(region, (bytes, offset), bytes_per_row);
                if !held {
                    self.count(|c| c.upload_calls += 1);
                    self.write_texture_from_js(region, (bytes, offset), bytes_per_row);
                }
            }
        }
    }

    fn texture_fits(&self, desc: &TextureDesc<'_>) -> bool {
        orb_hal::level_bytes(desc, self.allocated_layers(desc))
            <= self.caps.limits.max_texture_bytes
    }

    fn allocated_layers(&self, desc: &TextureDesc<'_>) -> u32 {
        if self.gl_layers && desc.dimension == TextureViewDimension::D2Array {
            gl_array_layers(desc)
        } else {
            desc.array_layers
        }
    }

    fn try_create_texture(&mut self, desc: &TextureDesc<'_>) -> Option<TextureId> {
        let layers = if self.gl && desc.dimension == TextureViewDimension::D2Array {
            gl_array_layers(desc)
        } else {
            desc.array_layers
        };
        let bytes = orb_hal::level_bytes(desc, layers);
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
        let layers = if self.gl && desc.dimension == TextureViewDimension::D2Array {
            gl_array_layers(desc)
        } else {
            desc.array_layers
        };
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
        self.count(|c| {
            c.textures_created += 1;
            c.bytes_allocated += texture_bytes(desc, layers);
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.textures.insert(Texture {
            texture: Some(texture),
            view: Some(view),
            levels: None,
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
        let texture = self.texture(texture).clone();
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: levels.start,
            mip_level_count: Some(levels.len() as u32),
            ..Default::default()
        });
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view: Some(view),
            levels: Some(levels),
        }))
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
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view: Some(view),
            levels: Some(level..level + 1),
        }))
    }

    fn destroy_texture(&mut self, id: TextureId) {
        self.texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .forget(id);
        #[cfg(not(target_arch = "wasm32"))]
        self.writes.forget_texture(id);
        let entry = self.textures.remove(id.0);
        if let (Some(texture), None) = (entry.texture, entry.levels) {
            if let Some(l) = &mut self.ledger {
                l.freed(true, id.0);
            }
            texture.destroy();
        }
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.count(|c| c.texel_bytes_from_core += data.len() as u64);
        let texture = self.texture(region.texture);
        let bytes_per_row = region.width * convert::texel_bytes(texture.format());
        assert_eq!(
            data.len(),
            (bytes_per_row * region.height) as usize,
            "texture write size mismatch"
        );
        let held = self
            .texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .core(region, data, bytes_per_row);
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
        self.count(|c| {
            c.texel_bytes_from_host += u64::from(region.width) * u64::from(region.height) * 4;
        });
        // Issued at the next submission; the image is closed after it,
        // which WebGL 2, copying only at a submission, needs too.
        self.texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .host(region, image, premultiply);
    }

    #[cfg(target_arch = "wasm32")]
    fn release_host_image(&mut self, image: HostImageId) {
        // A held write still reads it: closed once that is issued.
        if self
            .texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .reads(image)
        {
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
        match id {
            PipelineId::Render(id) => drop(self.render_pipelines.remove(id.0)),
            PipelineId::Compute(id) => drop(self.compute_pipelines.remove(id.0)),
        }
    }

    fn live_resources(&self) -> usize {
        let counts = [
            self.buffers.len(),
            self.textures.len(),
            self.shader_modules.len(),
            self.bind_group_layouts.len(),
            self.bind_groups.len(),
            self.render_pipelines.len(),
            self.compute_pipelines.len(),
            self.render_bundles.len(),
            self.samplers.len(),
        ];
        counts.iter().sum()
    }

    fn counters(&self) -> GpuCounters {
        self.counters.get()
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
        // Texture views live until the bind group is created; wgpu keeps
        // what it needs.
        let views: Vec<Option<wgpu::TextureView>> = desc
            .entries
            .iter()
            .map(|entry| match entry.resource {
                BindingResource::Texture { texture, dimension } => {
                    let entry = &self.textures[texture.0];
                    let levels = entry.levels.clone();
                    let texture = self.texture(texture);
                    Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(convert::view_dimension(dimension)),
                        base_mip_level: levels.as_ref().map_or(0, |l| l.start),
                        mip_level_count: levels.map(|l| l.len() as u32),
                        ..Default::default()
                    }))
                }
                _ => None,
            })
            .collect();
        let entries: Vec<_> = desc
            .entries
            .iter()
            .zip(&views)
            .map(|(entry, view)| wgpu::BindGroupEntry {
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
                    BindingResource::Texture { .. } => {
                        wgpu::BindingResource::TextureView(view.as_ref().expect("created above"))
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
        self.count(|c| c.pipelines_requested += 1);
        let slot = self.render_pipelines.insert(None);
        self.submit_job(compile::Job::Render(slot, job), desc.bind_group_layouts);
        RenderPipelineId(slot)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        let job = compile::ComputeJob {
            label: desc.label.to_owned(),
            #[cfg(not(webgpu))]
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            compute: compile::Stage::new(
                &desc.compute,
                self.shader_modules[desc.compute.module.0].clone(),
            ),
            #[cfg(not(webgpu))]
            cache: self.cache(),
        };
        self.count(|c| c.pipelines_requested += 1);
        let slot = self.compute_pipelines.insert(None);
        self.submit_job(compile::Job::Compute(slot, job), desc.bind_group_layouts);
        ComputePipelineId(slot)
    }

    fn pipeline_ready(&mut self, id: PipelineId) -> bool {
        let ready = |gpu: &Self| match id {
            PipelineId::Render(id) => gpu.render_pipelines[id.0].is_some(),
            PipelineId::Compute(id) => gpu.compute_pipelines[id.0].is_some(),
        };
        if ready(self) {
            return true;
        }
        self.take_compiled(false);
        if ready(self) {
            return true;
        }
        // A frame asks for a background pipeline: it waits for it now.
        if self.background_pending.remove(&id) {
            self.pending += 1;
            self.gate_background();
            #[cfg(not(target_arch = "wasm32"))]
            self.compiler.promote(&self.device, id);
            #[cfg(target_arch = "wasm32")]
            self.compiler.promote(id);
        }
        false
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
        for command in commands {
            self.encode_render(&mut encoder, command);
        }
        // wgpu panics when `finish` fails, whatever the error handler, and
        // on a lost device it fails ("Parent device is lost"): the frame
        // goes on without the bundle, and the next one makes a new device.
        let bundle = (!self.known_lost()).then(|| {
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
        #[cfg(target_arch = "wasm32")]
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
        let timing = self.timing.as_ref();
        let writes = |i: usize| timing?.query((*queries.get(i)?)?);
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
                    let source = self.texture(*source);
                    let destination = self.texture(*destination);
                    let levels = source.mip_level_count().min(destination.mip_level_count());
                    for (copy, level) in
                        copies.iter().flat_map(|c| (0..levels).map(move |l| (c, l)))
                    {
                        let at = |[x, y, z]: [u32; 3]| wgpu::Origin3d {
                            x: x >> level,
                            y: y >> level,
                            z,
                        };
                        let [w, h, layers] = copy.size;
                        let size = wgpu::Extent3d {
                            width: (w >> level).max(1),
                            height: (h >> level).max(1),
                            depth_or_array_layers: layers,
                        };
                        let texel = u64::from(convert::texel_bytes(source.format()));
                        self.count(|c| {
                            c.bytes_copied += u64::from(size.width)
                                * u64::from(size.height)
                                * u64::from(layers)
                                * texel;
                        });
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                mip_level: level,
                                origin: at(copy.from),
                                ..source.as_image_copy()
                            },
                            wgpu::TexelCopyTextureInfo {
                                mip_level: level,
                                origin: at(copy.to),
                                ..destination.as_image_copy()
                            },
                            size,
                        );
                    }
                }
                Pass::CopyTextureLevel {
                    source: (source, source_level),
                    destination: (destination, level),
                    copies,
                } => {
                    let source = self.texture(*source);
                    let destination = self.texture(*destination);
                    let texel = u64::from(convert::texel_bytes(source.format()));
                    for copy in copies {
                        let at = |[x, y, z]: [u32; 3]| wgpu::Origin3d { x, y, z };
                        let [width, height, layers] = copy.size;
                        self.count(|c| {
                            c.bytes_copied +=
                                u64::from(width) * u64::from(height) * u64::from(layers) * texel;
                        });
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                mip_level: *source_level,
                                origin: at(copy.from),
                                ..source.as_image_copy()
                            },
                            wgpu::TexelCopyTextureInfo {
                                mip_level: *level,
                                origin: at(copy.to),
                                ..destination.as_image_copy()
                            },
                            wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: layers,
                            },
                        );
                    }
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
                    self.count(|c| c.bytes_copied += size);
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
                    for command in commands {
                        match command {
                            ComputeCommand::SetPipeline(id) => pass.set_pipeline(
                                self.compute_pipelines[id.0]
                                    .as_ref()
                                    .expect("a pipeline used before it compiled"),
                            ),
                            ComputeCommand::SetBindGroup { index, group } => {
                                pass.set_bind_group(*index, &self.bind_groups[group.0], &[]);
                            }
                            ComputeCommand::Dispatch { x, y, z } => {
                                pass.dispatch_workgroups(*x, *y, *z)
                            }
                        }
                    }
                }
                // Into a surface whose texture couldn't be acquired
                // (`take_surface` says why).
                Pass::Render(render)
                    if (render.colors.iter()).any(|c| self.textures[c.target.0].view.is_none()) => {
                }
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
                            let stencil = (self.textures[depth.target.0].texture.as_ref())
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
                            command => self.encode_render(&mut pass, command),
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
    fn compute_passes_join_across_debug_markers_only() {
        let compute = || {
            Pass::Compute(ComputePass {
                label: "c",
                commands: Vec::new(),
            })
        };
        let render = || {
            Pass::Render(RenderPass {
                label: "r",
                colors: Vec::new(),
                depth: None,
                commands: Vec::new(),
            })
        };
        let passes = [
            compute(),
            Pass::PushDebugGroup("g"),
            compute(),
            Pass::PopDebugGroup,
            compute(),
            render(),
            compute(),
        ];
        assert_eq!(
            joined_compute(&passes, false),
            [false, false, true, false, true, false, false]
        );
        assert_eq!(
            joined_compute(&passes, true),
            [false, false, false, false, false, false, false]
        );
    }

    #[test]
    fn gl_arrays_take_neither_a_2d_nor_a_cube_target() {
        let layers = |width, height, array_layers| {
            gl_array_layers(&TextureDesc {
                label: "array",
                width,
                height,
                array_layers,
                dimension: TextureViewDimension::D2Array,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_levels: 1,
            })
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
