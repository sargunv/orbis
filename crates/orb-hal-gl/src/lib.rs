//! [`orb_hal::Gpu`] on one hand-written `glow` executor: WebGL2 and GLES
//! 3.0, the Compat tier (plan §8.1, §8.7).
//!
//! Shaders arrive as offline GLSL ES 3.00 (`orb-shaders`' `glsl`
//! feature). What WebGL2-class GL lacks is emulated here, beneath the
//! WebGPU-shaped interface, so the renderer's Compat path is its Modern one
//! less compute:
//! - **Storage buffers** are mirrored into R32UI data textures
//!   [`DATA_WIDTH`] words wide (the `fetch` module's `fetch_compat.wgsl`
//!   reads them with `texelFetch`).
//! - **Indirect draws** read their arguments from a CPU copy of the
//!   indirect buffer, which the renderer's CPU cull wrote.
//! - **Render bundles** are recorded command lists, replayed.
//! - **Clip space and orientation:** shaders flip Y into GL's clip space,
//!   so textures hold rows in WebGPU's order; [`GlGpu::present`] flips the
//!   frame into the default framebuffer.
//!
//! GL state goes through a cache after GL JS's (`state.rs`), so redundant
//! binds and enables are skipped.

// glow's calls are all `unsafe`: they are raw GL, valid for the context
// the executor owns and the objects it created on it.
#![allow(unsafe_code)]

mod cache;
mod data;
#[cfg(all(feature = "egl", not(target_arch = "wasm32")))]
pub mod egl;
mod exec;
mod slots;
mod state;
#[cfg(target_arch = "wasm32")]
mod web;

use glow::HasContext;
use orb_hal::*;
use slots::Slots;

pub use data::{DATA_WIDTH, DATA_WIDTH_LOG2};

/// The GL context and every resource created on it.
pub struct GlGpu {
    gl: glow::Context,
    caps: Caps,
    info: Info,
    buffers: Slots<Buffer>,
    textures: Slots<Texture>,
    samplers: Slots<glow::Sampler>,
    modules: Slots<Module>,
    layouts: Slots<Vec<BindGroupLayoutEntry>>,
    groups: Slots<Vec<BindGroupEntry>>,
    pipelines: Slots<Pipeline>,
    bundles: Slots<Vec<RenderCommand>>,
    host_bytes: Slots<HostBytes>,
    #[cfg(target_arch = "wasm32")]
    host_images: Slots<web_sys::ImageBitmap>,
    /// The WebGL2 context behind `gl`, for its extensions.
    #[cfg(target_arch = "wasm32")]
    webgl: Option<web_sys::WebGl2RenderingContext>,
    state: state::State,
    /// The one vertex array: draws pull their vertices, so it holds only
    /// the index buffer.
    vao: glow::VertexArray,
    /// Integer data textures sample with nearest filtering.
    data_sampler: glow::Sampler,
    framebuffers: exec::Framebuffers,
    uploaded: u64,
    pipelines_created: u32,
    /// Lost on purpose ([`GlGpu::mark_lost`]).
    marked_lost: bool,
    /// Linked programs from earlier runs, once started.
    programs: Option<cache::ProgramCache>,
    /// Compiled stages by kind and source: programs that share a stage
    /// (the symbol passes' vertex shader) compile it once.
    stages: std::collections::HashMap<(u32, u64), glow::Shader>,
    /// Time spent compiling and linking (natively).
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_time: std::time::Duration,
}

impl std::fmt::Debug for GlGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlGpu")
            .field("info", &self.info)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

/// What the driver reports about itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Info {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
    /// The unmasked renderer, where WebGL exposes it.
    pub unmasked_renderer: Option<String>,
}

/// Bytes held for [`Gpu::write_buffer_from_host`].
enum HostBytes {
    #[cfg(target_arch = "wasm32")]
    Js(js_sys::ArrayBuffer),
    Owned(Vec<u8>),
}

struct Buffer {
    size: u64,
    /// For vertex, index and uniform use, bound at `target`.
    gl: Option<(glow::Buffer, u32)>,
    /// For indirect arguments and readback.
    shadow: Option<Vec<u8>>,
    /// For storage use.
    data: Option<data::DataTexture>,
}

#[derive(Clone, Copy)]
enum TextureObject {
    Texture(glow::Texture),
    /// Multisampled attachments: GLES 3.0 has no multisampled textures.
    Renderbuffer(glow::Renderbuffer),
}

#[derive(Clone, Copy)]
struct Texture {
    object: TextureObject,
    format: TextureFormat,
    width: u32,
    height: u32,
    dimension: TextureViewDimension,
    /// The first mip level it covers: a view's, or 0.
    base_level: u32,
    /// A layer view's layer of an array texture.
    layer: u32,
    /// Views don't own their object.
    owned: bool,
}

impl Texture {
    fn target(&self) -> u32 {
        match self.dimension {
            TextureViewDimension::D2 => glow::TEXTURE_2D,
            TextureViewDimension::D2Array => glow::TEXTURE_2D_ARRAY,
        }
    }
}

/// A texture's (group, binding), and its sampler's if it samples.
type TextureBinding = ((u32, u32), Option<(u32, u32)>);

struct Module {
    vertex_source: String,
    fragment_source: String,
    /// Compiled on first use: a cached program needs no shaders.
    shaders: Option<(glow::Shader, glow::Shader)>,
    label: String,
    blocks: Vec<(String, u32, u32)>,
    textures: Vec<(String, TextureBinding)>,
}

struct Pipeline {
    program: glow::Program,
    /// Uniform block binding points in order, with the (group, binding)
    /// each reads.
    blocks: Vec<(u32, u32)>,
    /// Texture units in order: the texture's (group, binding) and its
    /// sampler's.
    textures: Vec<TextureBinding>,
    /// `naga_vs_first_instance`, for draws that start past instance 0.
    first_instance: Option<glow::UniformLocation>,
    topology: u32,
    blend: Option<BlendState>,
    depth: Option<DepthState>,
    cull: Option<Face>,
}

/// GL formats of a texture format: internal format, format, type.
fn gl_format(format: TextureFormat) -> (u32, u32, u32) {
    match format {
        TextureFormat::R8Unorm => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
        TextureFormat::R32Float => (glow::R32F, glow::RED, glow::FLOAT),
        TextureFormat::Rgba16Float => (glow::RGBA16F, glow::RGBA, glow::HALF_FLOAT),
        // GL has no BGRA attachments; hosts read frames back as RGBA.
        TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm => {
            (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)
        }
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Bgra8UnormSrgb => {
            (glow::SRGB8_ALPHA8, glow::RGBA, glow::UNSIGNED_BYTE)
        }
        TextureFormat::Depth32Float => {
            (glow::DEPTH_COMPONENT32F, glow::DEPTH_COMPONENT, glow::FLOAT)
        }
    }
}

fn texel_bytes(format: TextureFormat) -> u32 {
    match format {
        TextureFormat::R8Unorm => 1,
        TextureFormat::R32Float => 4,
        TextureFormat::Rgba16Float => 8,
        _ => 4,
    }
}

impl GlGpu {
    /// The executor over a WebGL2 or GLES 3.0+ context, current on this
    /// thread for as long as the executor lives.
    pub fn new(gl: glow::Context) -> Result<Self, String> {
        let version = gl.version();
        if !version.is_embedded || version.major < 3 {
            return Err(format!(
                "Compat needs WebGL2 or GLES 3.0, not {}.{} ({})",
                version.major, version.minor, version.vendor_info
            ));
        }
        // SAFETY: queries and object creation on the current context.
        let (info, limits, float_targets, vao, data_sampler) = unsafe {
            let extensions = gl.supported_extensions();
            let float_targets = extensions.contains("EXT_color_buffer_float")
                || extensions.contains("GL_EXT_color_buffer_float")
                || extensions.contains("GL_EXT_color_buffer_half_float");
            let unmasked = extensions
                .contains("WEBGL_debug_renderer_info")
                .then(|| gl.get_parameter_string(0x9246));
            let info = Info {
                vendor: gl.get_parameter_string(glow::VENDOR),
                renderer: gl.get_parameter_string(glow::RENDERER),
                version: gl.get_parameter_string(glow::VERSION),
                unmasked_renderer: unmasked,
            };
            let max_texture = gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) as u32;
            let limits = Limits {
                // A data texture holds DATA_WIDTH × max_texture words.
                max_buffer_size: u64::from(DATA_WIDTH) * u64::from(max_texture) * 4,
                max_storage_buffer_binding_size: u64::from(DATA_WIDTH) * u64::from(max_texture) * 4,
                max_storage_buffers_per_shader_stage: 8,
                max_texture_dimension_2d: max_texture,
                max_texture_array_layers: gl.get_parameter_i32(glow::MAX_ARRAY_TEXTURE_LAYERS)
                    as u32,
                min_storage_buffer_offset_alignment: 4,
                min_uniform_buffer_offset_alignment: gl
                    .get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT)
                    .max(1) as u32,
            };
            let vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(vao));
            let data_sampler = gl.create_sampler()?;
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
                (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.sampler_parameter_i32(data_sampler, name, value as i32);
            }
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            // Shaders flip Y, which turns counterclockwise faces clockwise.
            gl.front_face(glow::CW);
            (info, limits, float_targets, vao, data_sampler)
        };
        if limits.max_texture_dimension_2d < DATA_WIDTH {
            return Err(format!(
                "Compat needs {DATA_WIDTH}-texel textures; this device has {}",
                limits.max_texture_dimension_2d
            ));
        }
        Ok(Self {
            gl,
            caps: Caps {
                tier: Tier::Compat,
                limits,
                timestamp_query: false,
                rgba16float_blendable: float_targets,
            },
            info,
            buffers: Slots::default(),
            textures: Slots::default(),
            samplers: Slots::default(),
            modules: Slots::default(),
            layouts: Slots::default(),
            groups: Slots::default(),
            pipelines: Slots::default(),
            bundles: Slots::default(),
            host_bytes: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            host_images: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            webgl: None,
            state: state::State::default(),
            vao,
            data_sampler,
            framebuffers: exec::Framebuffers::default(),
            uploaded: 0,
            pipelines_created: 0,
            marked_lost: false,
            programs: None,
            stages: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_time: std::time::Duration::ZERO,
        })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    /// The context, for hosts that make surfaces or check for loss.
    pub fn context(&self) -> &glow::Context {
        &self.gl
    }

    /// Bytes written to buffers since the context was made.
    pub fn uploaded(&self) -> u64 {
        self.uploaded
    }

    pub fn pipelines_created(&self) -> u32 {
        self.pipelines_created
    }

    /// Bytes in Rust memory, for [`Gpu::write_buffer_from_host`].
    pub fn register_bytes(&mut self, bytes: Vec<u8>) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Owned(bytes)))
    }

    /// Whether the context was lost: everything made on it is gone. Takes
    /// the context's error flag.
    pub fn lost(&self) -> bool {
        if self.marked_lost {
            return true;
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(webgl) = &self.webgl {
            return webgl.is_context_lost();
        }
        /// WebGL's `CONTEXT_LOST_WEBGL`.
        const CONTEXT_LOST_WEBGL: u32 = 0x9242;
        // SAFETY: a query on the current context.
        let error = unsafe { self.gl.get_error() };
        matches!(error, glow::CONTEXT_LOST | CONTEXT_LOST_WEBGL)
    }

    /// Reports the context lost from now on, to exercise recovery where GL
    /// can't lose a context on demand.
    pub fn mark_lost(&mut self) {
        self.marked_lost = true;
    }

    /// Reads a colour target's level 0 back as tightly packed RGBA8 rows,
    /// top row first.
    pub fn read_rgba8(&mut self, texture: TextureId) -> Vec<u8> {
        let t = self.textures[texture.0];
        assert!(
            matches!(
                t.format,
                TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm
            ),
            "reads RGBA8 targets"
        );
        let fbo = self
            .framebuffers
            .color(&self.gl, &mut self.state, &t, t.base_level, t.layer);
        let mut pixels = vec![0u8; (t.width * t.height * 4) as usize];
        // SAFETY: reads the framebuffer made for this texture into a
        // buffer of its size.
        unsafe {
            self.state.read_framebuffer(&self.gl, Some(fbo));
            self.gl.read_pixels(
                0,
                0,
                t.width as i32,
                t.height as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
        }
        pixels
    }

    /// Copies a frame into the default framebuffer, flipped into GL's
    /// orientation; the host then swaps buffers (natively) or the browser
    /// presents (WebGL).
    pub fn present(&mut self, frame: TextureId, width: u32, height: u32) {
        let t = self.textures[frame.0];
        let fbo = self
            .framebuffers
            .color(&self.gl, &mut self.state, &t, t.base_level, t.layer);
        // SAFETY: a blit between framebuffers of this context.
        unsafe {
            self.state.read_framebuffer(&self.gl, Some(fbo));
            self.state.draw_framebuffer(&self.gl, None);
            self.gl.blit_framebuffer(
                0,
                0,
                t.width as i32,
                t.height as i32,
                0,
                height as i32,
                width as i32,
                0,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
        }
    }

    /// The buffer's CPU copy (indirect and readback buffers).
    pub fn read_buffer(&self, buffer: BufferId) -> &[u8] {
        self.buffers[buffer.0]
            .shadow
            .as_deref()
            .expect("buffers with COPY_SRC or INDIRECT use keep a CPU copy")
    }

    /// A module's program: from the program cache, or compiled and linked
    /// (and then cached).
    fn link(&mut self, module: ShaderModuleId) -> glow::Program {
        let key = {
            let m = &self.modules[module.0];
            cache::key(&m.vertex_source, &m.fragment_source)
        };
        // SAFETY: program objects of this context.
        unsafe {
            let program = self.gl.create_program().expect("program object");
            if let Some(binary) = self.programs.as_ref().and_then(|c| c.get(key)) {
                self.gl.program_binary(program, &binary);
                if self.gl.get_program_link_status(program) {
                    return program;
                }
                // The driver turned it down (it changed): compile.
            }
            if self.modules[module.0].shaders.is_none() {
                let m = &self.modules[module.0];
                let (vertex, fragment, label) = (
                    m.vertex_source.clone(),
                    m.fragment_source.clone(),
                    m.label.clone(),
                );
                let shaders = (
                    self.stage(glow::VERTEX_SHADER, &vertex, &label),
                    self.stage(glow::FRAGMENT_SHADER, &fragment, &label),
                );
                self.modules[module.0].shaders = Some(shaders);
            }
            let m = &self.modules[module.0];
            let (vertex, fragment) = m.shaders.expect("compiled above");
            self.gl.attach_shader(program, vertex);
            self.gl.attach_shader(program, fragment);
            if self.programs.is_some() {
                self.gl.program_binary_retrievable_hint(program, true);
            }
            self.gl.link_program(program);
            if !self.gl.get_program_link_status(program) {
                panic!(
                    "{}: GLSL link failed: {}",
                    m.label,
                    self.gl.get_program_info_log(program)
                );
            }
            if let Some(cache) = &mut self.programs
                && let Some(binary) = self.gl.get_program_binary(program)
            {
                cache.insert(key, binary);
            }
            program
        }
    }

    /// The program cache's key for this driver: data from another driver
    /// (or version of it) is never offered to this one.
    pub fn program_cache_key(&self) -> String {
        cache::driver_key(&self.info)
    }

    /// Starts the program cache (GLES 3.0 program binaries, the GL analog
    /// of a pipeline cache), from data an earlier run's
    /// [`program_cache_data`](Self::program_cache_data) gave on this
    /// driver. Damaged data starts it empty.
    pub fn start_program_cache(&mut self, data: Option<&[u8]>) {
        if self.programs.is_none() {
            self.programs = Some(cache::ProgramCache::open(data));
        }
    }

    /// The program cache's data, when programs were added since it was
    /// started or last taken.
    pub fn program_cache_data(&mut self) -> Option<Vec<u8>> {
        self.programs.as_mut()?.take_data()
    }

    /// Time spent compiling and linking programs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pipeline_time(&self) -> std::time::Duration {
        self.pipeline_time
    }

    /// A compiled stage: one compile per distinct source.
    fn stage(&mut self, kind: u32, source: &str, label: &str) -> glow::Shader {
        let key = (kind, cache::key(source, ""));
        if let Some(&shader) = self.stages.get(&key) {
            return shader;
        }
        let shader = self.compile(kind, source, label);
        self.stages.insert(key, shader);
        shader
    }

    fn compile(&self, kind: u32, source: &str, label: &str) -> glow::Shader {
        // SAFETY: shader creation and compilation on the current context.
        unsafe {
            let shader = self.gl.create_shader(kind).expect("shader object");
            self.gl.shader_source(shader, source);
            self.gl.compile_shader(shader);
            if !self.gl.get_shader_compile_status(shader) {
                panic!(
                    "{label}: GLSL compile failed: {}",
                    self.gl.get_shader_info_log(shader)
                );
            }
            shader
        }
    }
}

impl Drop for GlGpu {
    /// Deletes everything made on the context, which outlives the executor.
    fn drop(&mut self) {
        let gl = &self.gl;
        // SAFETY: deletes objects this executor made on its context.
        unsafe {
            for buffer in self.buffers.drain() {
                if let Some((b, _)) = buffer.gl {
                    gl.delete_buffer(b);
                }
                if let Some(data) = buffer.data {
                    gl.delete_texture(data.texture);
                }
            }
            for texture in self.textures.drain().filter(|t| t.owned) {
                match texture.object {
                    TextureObject::Texture(t) => gl.delete_texture(t),
                    TextureObject::Renderbuffer(rb) => gl.delete_renderbuffer(rb),
                }
            }
            for sampler in self.samplers.drain() {
                gl.delete_sampler(sampler);
            }
            for (_, shader) in self.stages.drain() {
                gl.delete_shader(shader);
            }
            for pipeline in self.pipelines.drain() {
                gl.delete_program(pipeline.program);
            }
            self.framebuffers.clear(gl);
            gl.delete_sampler(self.data_sampler);
            gl.delete_vertex_array(self.vao);
        }
    }
}

impl Gpu for GlGpu {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId {
        let gl_use = desc
            .usage
            .intersects(BufferUsages::VERTEX | BufferUsages::INDEX | BufferUsages::UNIFORM);
        let gl = gl_use.then(|| {
            // WebGL2 binds an index buffer to no other target.
            let target = if desc.usage.contains(BufferUsages::INDEX) {
                glow::ELEMENT_ARRAY_BUFFER
            } else {
                glow::UNIFORM_BUFFER
            };
            // SAFETY: a new buffer on the current context, sized once.
            unsafe {
                let buffer = self.gl.create_buffer().expect("buffer object");
                self.state.bind_buffer(&self.gl, target, Some(buffer));
                self.gl
                    .buffer_data_size(target, desc.size as i32, glow::DYNAMIC_DRAW);
                (buffer, target)
            }
        });
        // Shadows serve indirect draws and readback; storage buffers copy
        // texture to texture.
        let shadow = (desc.usage.contains(BufferUsages::INDIRECT)
            || desc.usage.contains(BufferUsages::COPY_SRC)
                && !desc.usage.contains(BufferUsages::STORAGE))
        .then(|| vec![0; desc.size as usize]);
        let data = desc
            .usage
            .contains(BufferUsages::STORAGE)
            .then(|| data::DataTexture::new(&self.gl, &mut self.state, desc.size, desc.label));
        BufferId(self.buffers.insert(Buffer {
            size: desc.size,
            gl,
            shadow,
            data,
        }))
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        let buffer = self.buffers.remove(id.0);
        // SAFETY: deletes objects this buffer owned.
        unsafe {
            if let Some((b, target)) = buffer.gl {
                self.state.forget_buffer(target, b);
                self.gl.delete_buffer(b);
            }
            if let Some(data) = buffer.data {
                self.state.forget_texture(data.texture);
                self.gl.delete_texture(data.texture);
            }
        }
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        self.uploaded += data.len() as u64;
        let buffer = &mut self.buffers[id.0];
        assert!(
            offset + data.len() as u64 <= buffer.size,
            "write past the end of a buffer"
        );
        if let Some(shadow) = &mut buffer.shadow {
            shadow[offset as usize..offset as usize + data.len()].copy_from_slice(data);
        }
        if let Some((b, target)) = buffer.gl {
            // SAFETY: a write inside the buffer's size.
            unsafe {
                self.state.bind_buffer(&self.gl, target, Some(b));
                self.gl
                    .buffer_sub_data_u8_slice(target, offset as i32, data);
            }
        }
        if let Some(texture) = &buffer.data {
            texture.write(&self.gl, &mut self.state, offset, data);
        }
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        let data = self.read_host_bytes(bytes, range);
        self.write_buffer(id, offset, &data);
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
        let (internal, ..) = gl_format(desc.format);
        assert!(
            desc.dimension == TextureViewDimension::D2Array || desc.array_layers == 1,
            "{}: 2D textures have one layer",
            desc.label
        );
        // SAFETY: new storage on the current context.
        let object = unsafe {
            if desc.sample_count > 1 {
                let rb = self.gl.create_renderbuffer().expect("renderbuffer");
                self.gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
                self.gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    desc.sample_count as i32,
                    internal,
                    desc.width as i32,
                    desc.height as i32,
                );
                TextureObject::Renderbuffer(rb)
            } else {
                let texture = self.gl.create_texture().expect("texture object");
                let target = match desc.dimension {
                    TextureViewDimension::D2 => glow::TEXTURE_2D,
                    TextureViewDimension::D2Array => glow::TEXTURE_2D_ARRAY,
                };
                self.state.bind_texture(&self.gl, 0, target, Some(texture));
                if target == glow::TEXTURE_2D {
                    self.gl.tex_storage_2d(
                        target,
                        desc.mip_levels as i32,
                        internal,
                        desc.width as i32,
                        desc.height as i32,
                    );
                } else {
                    self.gl.tex_storage_3d(
                        target,
                        desc.mip_levels as i32,
                        internal,
                        desc.width as i32,
                        desc.height as i32,
                        desc.array_layers as i32,
                    );
                }
                TextureObject::Texture(texture)
            }
        };
        TextureId(self.textures.insert(Texture {
            object,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            dimension: desc.dimension,
            base_level: 0,
            layer: 0,
            owned: true,
        }))
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        let mut view = self.textures[texture.0];
        view.width = (view.width >> levels.start).max(1);
        view.height = (view.height >> levels.start).max(1);
        assert_eq!(levels.len(), 1, "Compat views render into one level");
        view.base_level = levels.start;
        view.owned = false;
        TextureId(self.textures.insert(view))
    }

    fn create_layer_view(&mut self, texture: TextureId, layer: u32) -> TextureId {
        let mut view = self.textures[texture.0];
        assert_eq!(
            view.dimension,
            TextureViewDimension::D2Array,
            "layers of arrays"
        );
        view.layer = layer;
        view.owned = false;
        TextureId(self.textures.insert(view))
    }

    fn destroy_texture(&mut self, id: TextureId) {
        let texture = self.textures.remove(id.0);
        self.framebuffers
            .forget(&self.gl, &mut self.state, &texture);
        if !texture.owned {
            return;
        }
        // SAFETY: deletes the object this texture owned.
        unsafe {
            match texture.object {
                TextureObject::Texture(t) => {
                    self.state.forget_texture(t);
                    self.gl.delete_texture(t);
                }
                TextureObject::Renderbuffer(rb) => self.gl.delete_renderbuffer(rb),
            }
        }
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        let t = self.textures[region.texture.0];
        let TextureObject::Texture(texture) = t.object else {
            panic!("multisampled attachments take no writes");
        };
        assert_eq!(
            data.len() as u32,
            region.width * region.height * texel_bytes(t.format),
            "tightly packed rows"
        );
        let (_, format, ty) = gl_format(t.format);
        let target = t.target();
        // SAFETY: a write inside the texture's level 0.
        unsafe {
            self.state.bind_texture(&self.gl, 0, target, Some(texture));
            let pixels = glow::PixelUnpackData::Slice(Some(data));
            if target == glow::TEXTURE_2D {
                self.gl.tex_sub_image_2d(
                    target,
                    0,
                    region.x as i32,
                    region.y as i32,
                    region.width as i32,
                    region.height as i32,
                    format,
                    ty,
                    pixels,
                );
            } else {
                self.gl.tex_sub_image_3d(
                    target,
                    0,
                    region.x as i32,
                    region.y as i32,
                    region.layer as i32,
                    region.width as i32,
                    region.height as i32,
                    1,
                    format,
                    ty,
                    pixels,
                );
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        let pixels = web::read_bitmap(&self.host_images[image.0], premultiply);
        self.write_texture(region, &pixels);
    }

    #[cfg(target_arch = "wasm32")]
    fn release_host_image(&mut self, image: HostImageId) {
        self.host_images.remove(image.0).close();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_texture_from_host(&mut self, _: &TextureRegion, image: HostImageId, _: bool) {
        panic!("native hosts hold no images ({image:?})");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn release_host_image(&mut self, image: HostImageId) {
        panic!("native hosts hold no images ({image:?})");
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId {
        let filter = |f| match f {
            FilterMode::Nearest => glow::NEAREST,
            FilterMode::Linear => glow::LINEAR,
        };
        let min = match (desc.filter, desc.mipmap_filter) {
            (FilterMode::Nearest, FilterMode::Nearest) => glow::NEAREST_MIPMAP_NEAREST,
            (FilterMode::Nearest, FilterMode::Linear) => glow::NEAREST_MIPMAP_LINEAR,
            (FilterMode::Linear, FilterMode::Nearest) => glow::LINEAR_MIPMAP_NEAREST,
            (FilterMode::Linear, FilterMode::Linear) => glow::LINEAR_MIPMAP_LINEAR,
        };
        let wrap = match desc.address {
            AddressMode::ClampToEdge => glow::CLAMP_TO_EDGE,
            AddressMode::Repeat => glow::REPEAT,
        };
        // SAFETY: a new sampler object on the current context.
        let sampler = unsafe {
            let s = self.gl.create_sampler().expect("sampler object");
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, min),
                (glow::TEXTURE_MAG_FILTER, filter(desc.filter)),
                (glow::TEXTURE_WRAP_S, wrap),
                (glow::TEXTURE_WRAP_T, wrap),
            ] {
                self.gl.sampler_parameter_i32(s, name, value as i32);
            }
            s
        };
        SamplerId(self.samplers.insert(sampler))
    }

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        let ShaderSource::Glsl(program) = &desc.source else {
            panic!("{}: Compat draws offline GLSL, not WGSL", desc.label);
        };
        let module = Module {
            vertex_source: program.vertex.to_string(),
            fragment_source: program.fragment.to_string(),
            shaders: None,
            label: desc.label.to_string(),
            blocks: program
                .blocks
                .iter()
                .map(|&(n, g, b)| (n.to_string(), g, b))
                .collect(),
            textures: program
                .textures
                .iter()
                .map(|t| (t.name.to_string(), (t.texture, t.sampler)))
                .collect(),
        };
        ShaderModuleId(self.modules.insert(module))
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        BindGroupLayoutId(self.layouts.insert(desc.entries.to_vec()))
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId {
        BindGroupId(self.groups.insert(desc.entries.to_vec()))
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.groups.remove(id.0);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        assert_eq!(
            desc.vertex.module, desc.fragment.module,
            "a Compat program is one module"
        );
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let program = self.link(desc.vertex.module);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pipeline_time += started.elapsed();
        }
        let module = &self.modules[desc.vertex.module.0];
        let gl = &self.gl;
        // SAFETY: sets up a linked program of this context and queries it.
        let pipeline = unsafe {
            gl.use_program(Some(program));
            self.state.forget_program();
            let mut blocks = Vec::new();
            for (name, group, binding) in &module.blocks {
                if let Some(index) = gl.get_uniform_block_index(program, name) {
                    // Both stages' blocks of one binding share a point.
                    let point = match blocks.iter().position(|b| b == &(*group, *binding)) {
                        Some(p) => p,
                        None => {
                            blocks.push((*group, *binding));
                            blocks.len() - 1
                        }
                    };
                    gl.uniform_block_binding(program, index, point as u32);
                }
            }
            let mut textures = Vec::new();
            for (name, (texture, sampler)) in &module.textures {
                if let Some(location) = gl.get_uniform_location(program, name) {
                    let unit = match textures.iter().position(|t| t == &(*texture, *sampler)) {
                        Some(u) => u,
                        None => {
                            textures.push((*texture, *sampler));
                            textures.len() - 1
                        }
                    };
                    gl.uniform_1_i32(Some(&location), unit as i32);
                }
            }
            let first_instance = gl.get_uniform_location(program, "naga_vs_first_instance");
            if let Some(location) = &first_instance {
                gl.uniform_1_u32(Some(location), 0);
            }
            Pipeline {
                program,
                blocks,
                textures,
                first_instance,
                topology: match desc.topology {
                    PrimitiveTopology::LineList => glow::LINES,
                    PrimitiveTopology::TriangleList => glow::TRIANGLES,
                    PrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
                },
                blend: desc.targets.first().and_then(|t| t.blend),
                depth: desc.depth,
                cull: desc.cull_mode,
            }
        };
        self.pipelines_created += 1;
        RenderPipelineId(self.pipelines.insert(pipeline))
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        panic!("{}: Compat has no compute", desc.label);
    }

    fn create_render_bundle(
        &mut self,
        _: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId {
        RenderBundleId(self.bundles.insert(commands.to_vec()))
    }

    fn destroy_render_bundle(&mut self, id: RenderBundleId) {
        self.bundles.remove(id.0);
    }

    fn create_query_set(&mut self, desc: &QuerySetDesc<'_>) -> QuerySetId {
        panic!("{}: Compat has no timestamp queries", desc.label);
    }

    fn destroy_query_set(&mut self, _: QuerySetId) {}

    fn submit(&mut self, passes: &[Pass]) {
        for pass in passes {
            self.execute(pass);
        }
    }
}
