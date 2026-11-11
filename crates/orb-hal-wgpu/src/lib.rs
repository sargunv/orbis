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
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
mod offscreen;
mod slots;
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

/// The frame driver a native device's renderers take: `gpu` (the default
/// where the device has it), or `cpu` and `cpu-tex`
/// ([`WgpuGpu::force_cpu_driver`]).
#[cfg(not(target_arch = "wasm32"))]
const DRIVER: &str = "ORB_DRIVER";

/// A native device's 2D texture side, lowered to this to stand in for a
/// device that offers no more ([`Limits::largest_texture_dimension_2d`]):
/// at `8192`, as in Firefox's WebGL 2, a terrain view of more than 32
/// tiles gives each drape stack a layer of its own.
#[cfg(not(target_arch = "wasm32"))]
const TEXTURE_SIDE: &str = "ORB_TEXTURE_SIDE";

/// The features Orbis asks `adapter`'s device for, those it has of them:
/// natively, a pipeline cache, which warm starts reuse pipelines through;
/// timestamps, which time passes for measurements; clip distances, which
/// clip cut buildings without discarding; and on GL, passthrough shaders,
/// which take the build's GLSL (`ShaderModuleDesc::glsl`).
pub fn device_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let mut wanted = wgpu::Features::PIPELINE_CACHE
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::CLIP_DISTANCES;
    if !cfg!(webgpu) && adapter.get_info().backend == wgpu::Backend::Gl {
        wanted |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    adapter.features() & wanted
}

/// The limits Orbis asks `adapter`'s device for: WebGPU's defaults, lowered
/// where the adapter falls short of them (the iOS simulator has 15
/// inter-stage variables), with buffers and storage bindings as large as it
/// allows up to 1 GiB, and 2D textures as large as it allows. The scene's
/// pools grow within one storage binding, up to the map's memory budget
/// and past it for the tiles in view: at WebGPU's default 128 MiB, a
/// pitched view over a dense city ran out. Textures past WebGPU's default
/// side are only for those whose size changes how they pack what they
/// hold ([`Limits::largest_texture_dimension_2d`]).
pub fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let offered = adapter.limits();
    let mut limits = wgpu::Limits::default().or_worse_values_from(&offered);
    limits.max_buffer_size =
        (offered.max_buffer_size.min(LARGEST_BUFFER)).max(limits.max_buffer_size);
    limits.max_storage_buffer_binding_size = (offered.max_storage_buffer_binding_size)
        .min(LARGEST_BUFFER as _)
        .max(limits.max_storage_buffer_binding_size);
    limits.max_texture_dimension_2d =
        (offered.max_texture_dimension_2d).max(limits.max_texture_dimension_2d);
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
    idle_writes: std::collections::HashSet<u32>,
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
    /// Later requests are background ones
    /// ([`Gpu::set_background_compiles`]).
    background: bool,
    /// Background pipelines compiling that no frame asked for.
    background_pending: std::collections::HashSet<PipelineId>,
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
    uploaded: Uploaded,
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
    pipelines_created: u32,
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
    /// What the driver said when it lost the device.
    #[cfg(not(target_arch = "wasm32"))]
    lost_reason: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// D3D12's shader compiler, on a D3D12 device `offscreen` made.
    #[cfg(not(target_arch = "wasm32"))]
    shader_compiler: Option<String>,
}

/// Bytes written to buffers and textures since the device was created, by
/// where they came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Uploaded {
    /// From Rust memory: [`Gpu::write_buffer`] and host bytes registered
    /// with [`WgpuGpu::register_bytes`].
    pub from_core: u64,
    /// From JS `ArrayBuffer`s registered with
    /// `WgpuGpu::register_array_buffer`: on WebGPU never copied into wasm
    /// memory, on WebGL 2 copied through it.
    pub from_js: u64,
    /// Texel bytes written to textures from Rust memory.
    pub texels_from_core: u64,
    /// Texel bytes copied to textures from JS images registered with
    /// `WgpuGpu::register_image_bitmap` or `register_video`.
    pub texels_from_js: u64,
}

/// A write from core memory: gathered when small, or when it overlaps a
/// gathered one it must land after; otherwise straight to the queue.
fn write_core(
    writes: &mut writes::Writes,
    queue: &wgpu::Queue,
    buffers: &Slots<wgpu::Buffer>,
    (id, idle): (BufferId, bool),
    offset: u64,
    data: &[u8],
) {
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
    } else {
        queue.write_buffer(&buffers[id.0], offset, data);
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

#[derive(Debug)]
struct Texture {
    /// `None` for views the host registered and owns.
    texture: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    /// A view of these mip levels of another entry's texture, which it
    /// doesn't own.
    levels: Option<std::ops::Range<u32>>,
    /// The format a view of another entry's texture takes, if not the
    /// texture's.
    format: Option<wgpu::TextureFormat>,
}

/// The usages of a view of `texture` in `format`: the texture's, except
/// that a view in another format (its sRGB twin) can't be storage.
fn view_usage(
    texture: &wgpu::Texture,
    format: Option<wgpu::TextureFormat>,
) -> Option<wgpu::TextureUsages> {
    format.filter(|&f| f != texture.format()).map(|_| {
        texture.usage()
            & (wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
    })
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

impl WgpuGpu {
    /// A view of `levels` of `texture`'s, in `format` or the texture's.
    fn view_of(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
        format: Option<wgpu::TextureFormat>,
    ) -> TextureId {
        let texture = self.texture(texture).clone();
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format,
            usage: view_usage(&texture, format),
            base_mip_level: levels.start,
            mip_level_count: Some(levels.len() as u32),
            ..Default::default()
        });
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: Some(levels),
            format,
        }))
    }

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
                max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
                max_texture_dimension_2d: (limits.max_texture_dimension_2d)
                    .min(wgpu::Limits::default().max_texture_dimension_2d),
                largest_texture_dimension_2d: limits.max_texture_dimension_2d,
                max_texture_array_layers: limits.max_texture_array_layers,
                min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
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
            multi_draw: false,
            zero_to_one_depth: !gl || gl_clips_zero_to_one(&device),
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
            uploaded: Uploaded::default(),
            writes: writes::Writes::default(),
            texture_writes: Default::default(),
            readbacks: Default::default(),
            copy_next: None,
            copied: Default::default(),
            pipelines_created: 0,
            frames_started: 0,
            capture: None,
            capturing: false,
            debug_groups: false,
            #[cfg(not(target_arch = "wasm32"))]
            lost: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            lost_reason: Default::default(),
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
            limits.largest_texture_dimension_2d = limits.largest_texture_dimension_2d.min(side);
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
        let uploaded = self.writes.encode(
            &self.device,
            &self.queue,
            |id| self.buffers[id.0].clone(),
            encoder,
        );
        #[cfg(not(target_arch = "wasm32"))]
        if !uploaded {
            self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = uploaded;
    }

    /// Issues the held texture writes to the queue, wave by wave, each wave
    /// before the last its own submission (see `texture_writes`). GL
    /// orders its texture uploads as they come, so there the waves share
    /// one: each submission is a flush, which ANGLE's Metal back end
    /// commits as a command buffer of its own (#140).
    fn issue_texture_writes(&self) {
        let mut held = self.texture_writes.lock().expect("texture writes poisoned");
        held.issue(
            |write, texels| self.issue_texture_write(write, texels),
            || {
                if !self.gl {
                    self.queue.submit([]);
                }
            },
        );
    }

    fn issue_texture_write(&self, write: &texture_writes::TextureWrite, texels: &[u8]) {
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

    /// Bytes written to buffers so far.
    pub fn uploaded(&self) -> Uploaded {
        self.uploaded
    }

    /// Render and compute pipelines compiled so far.
    pub fn pipelines_created(&self) -> u32 {
        self.pipelines_created
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

    /// Makes later compiles of pipelines labelled `label` take `by` longer,
    /// as on a device compiling cold: fault injection for tests of what
    /// waits for them. Needs compile threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inject_slow_compile(&self, label: &str, by: std::time::Duration) {
        self.compiler.inject_slow(label, by);
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
            self.pipelines_created += 1;
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
    }

    fn submit_job(&mut self, job: compile::Job, layouts: &[BindGroupLayoutId]) {
        let background = self.background;
        if background {
            self.background_pending.insert(job.id());
        } else {
            self.pending += 1;
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
        self.lost_reason.lock().expect("loss poisoned").clone()
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
        let loss = compile::Loss {
            thread: Default::default(),
            reason: self.lost_reason.clone(),
        };
        self.compiler.watch_loss(loss.clone());
        let lost = self.lost.clone();
        self.device.set_device_lost_callback(move |why, message| {
            loss.reason
                .lock()
                .expect("loss poisoned")
                .get_or_insert_with(|| format!("{why:?}: {message}"));
            loss.thread
                .lock()
                .expect("loss poisoned")
                .get_or_insert_with(|| std::thread::current().id());
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
            view,
            levels: None,
            format: None,
        }))
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
        let id = self.buffers.insert(buffer);
        if desc.usage.contains(BufferUsages::IDLE_WRITES) {
            self.idle_writes.insert(id);
        }
        BufferId(id)
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        self.writes.forget(id);
        self.idle_writes.remove(&id.0);
        self.buffers.remove(id.0).destroy();
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.uploaded.from_core += data.len() as u64;
        write_core(
            &mut self.writes,
            &self.queue,
            &self.buffers,
            (id, self.idle_writes.contains(&id.0)),
            offset,
            data,
        );
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
                self.uploaded.from_core += len;
                let range = range.start as usize..range.end as usize;
                write_core(
                    &mut self.writes,
                    &self.queue,
                    &self.buffers,
                    (id, self.idle_writes.contains(&id.0)),
                    offset,
                    &data[range],
                );
            }
            #[cfg(webgl)]
            HostBytes::Js(data) => {
                self.uploaded.from_js += len;
                let data = js_sys::Uint8Array::new_with_byte_offset_and_length(
                    data,
                    range.start as u32,
                    len as u32,
                )
                .to_vec();
                write_core(
                    &mut self.writes,
                    &self.queue,
                    &self.buffers,
                    (id, self.idle_writes.contains(&id.0)),
                    offset,
                    &data,
                );
            }
            #[cfg(webgpu)]
            HostBytes::Js(data) => {
                self.uploaded.from_js += len;
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
        self.host_bytes.remove(bytes.0);
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
            view_formats: &desc
                .view_formats
                .iter()
                .map(|&f| convert::texture_format(f))
                .collect::<Vec<_>>(),
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: None,
            format: None,
        }))
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        self.view_of(texture, levels, None)
    }

    fn create_texture_view_as(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
        format: TextureFormat,
    ) -> TextureId {
        self.view_of(texture, levels, Some(convert::texture_format(format)))
    }

    fn create_layer_view(
        &mut self,
        texture: TextureId,
        layer: u32,
        level: u32,
        format: Option<TextureFormat>,
    ) -> TextureId {
        let texture = self.texture(texture).clone();
        let format = format.map(convert::texture_format);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format,
            usage: view_usage(&texture, format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: Some(level..level + 1),
            format,
        }))
    }

    fn destroy_texture(&mut self, id: TextureId) {
        self.texture_writes
            .get_mut()
            .expect("texture writes poisoned")
            .forget(id);
        let entry = self.textures.remove(id.0);
        if let (Some(texture), None) = (entry.texture, entry.levels) {
            texture.destroy();
        }
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.uploaded.texels_from_core += data.len() as u64;
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
            self.issue_texture_write(&write, data);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        self.uploaded.texels_from_js += u64::from(region.width * region.height * 4);
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

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        let source = desc.source.to_string();
        ShaderModuleId(self.shader_modules.insert(compile::Module::new(
            desc.label,
            source,
            desc.ir,
            (desc.glsl.filter(|_| self.glsl_es)).map(|g| (g, self.caps.zero_to_one_depth)),
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
                    let (levels, format) = (entry.levels.clone(), entry.format);
                    let texture = self.texture(texture);
                    Some(texture.create_view(&wgpu::TextureViewDescriptor {
                        format,
                        usage: view_usage(texture, format),
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
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            vertex: compile::Stage::new(&desc.vertex, module(&desc.vertex)),
            fragment: compile::Stage::new(&desc.fragment, module(&desc.fragment)),
            targets: desc.targets.to_vec(),
            depth: desc.depth,
            topology: desc.topology,
            cull_back: desc.cull_back,
            sample_count: desc.sample_count,
            attribute_zero: self.caps.gl_shaders(),
            cache: self.cache(),
        };
        let slot = self.render_pipelines.insert(None);
        self.submit_job(compile::Job::Render(slot, job), desc.bind_group_layouts);
        RenderPipelineId(slot)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        let job = compile::ComputeJob {
            label: desc.label.to_owned(),
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            compute: compile::Stage::new(
                &desc.compute,
                self.shader_modules[desc.compute.module.0].clone(),
            ),
            cache: self.cache(),
        };
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
        self.background = on;
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
        let started = timing::now();
        self.reserve_attribute_zero(passes.iter().flat_map(|pass| match pass {
            Pass::Render(render) => render.commands.as_slice(),
            _ => &[],
        }));
        // While timing, each compute and render pass's first query.
        let queries = match &mut self.timing {
            Some(timing) => timing.begin(
                &self.device,
                passes.iter().map(|pass| match pass {
                    Pass::Compute(p) => Some(p.label),
                    Pass::Render(p) => Some(p.label),
                    _ => None,
                }),
            ),
            None => Vec::new(),
        };
        let timing = self.timing.as_ref();
        let writes = |i: usize| timing?.query((*queries.get(i)?)?);
        // Debug groups and the encoder's label, for captures only.
        let groups = self.capturing || self.debug_groups;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: groups.then_some("passes"),
            });
        if groups {
            encoder.push_debug_group("uploads");
        }
        self.encode_writes(&mut encoder);
        if groups {
            encoder.pop_debug_group();
        }
        for (i, pass) in passes.iter().enumerate() {
            match pass {
                Pass::PushDebugGroup(label) if groups => encoder.push_debug_group(label),
                Pass::PopDebugGroup if groups => encoder.pop_debug_group(),
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
                } => encoder.copy_buffer_to_buffer(
                    &self.buffers[source.0],
                    *source_offset,
                    &self.buffers[destination.0],
                    *destination_offset,
                    *size,
                ),
                Pass::Compute(compute) => {
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
                    for command in &compute.commands {
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
                Pass::Render(render) => {
                    let colors: Vec<Option<wgpu::RenderPassColorAttachment>> = render
                        .colors
                        .iter()
                        .map(|color| {
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.textures[color.target.0].view,
                                depth_slice: None,
                                resolve_target: color.resolve.map(|id| &self.textures[id.0].view),
                                ops: wgpu::Operations {
                                    load: match color.load {
                                        LoadOp::Clear([r, g, b, a]) => {
                                            wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
                                        }
                                        LoadOp::Load => wgpu::LoadOp::Load,
                                    },
                                    store: match color.store {
                                        StoreOp::Store => wgpu::StoreOp::Store,
                                        StoreOp::Discard => wgpu::StoreOp::Discard,
                                    },
                                },
                            })
                        })
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(render.label),
                        color_attachments: &colors,
                        depth_stencil_attachment: render.depth.as_ref().map(|depth| {
                            wgpu::RenderPassDepthStencilAttachment {
                                view: &self.textures[depth.target.0].view,
                                depth_ops: Some(wgpu::Operations {
                                    load: match depth.load {
                                        LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
                                        LoadOp::Load => wgpu::LoadOp::Load,
                                    },
                                    store: match depth.store {
                                        StoreOp::Store => wgpu::StoreOp::Store,
                                        StoreOp::Discard => wgpu::StoreOp::Discard,
                                    },
                                }),
                                stencil_ops: None,
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
        if let Some(timing) = &mut self.timing {
            timing.end(timing::now() - started);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                view_formats: &[],
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
