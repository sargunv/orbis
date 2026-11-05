//! The map's canvas and the GPU drawing into it: WebGPU (the Modern tier,
//! `core.wasm`) or WebGL2 (Compat, `compat.wasm`; plan §8.7). A build with
//! both, the harness, tries WebGPU first.

use orb_hal::{Gpu, HostBytesId, HostImageId, TextureId};
use orb_runtime::TargetFormat;
use wasm_bindgen::JsValue;

/// The page's GPU, with what the job replies hand it.
pub(crate) trait WebGpu: Gpu {
    /// A tile's transferred `ArrayBuffer`, never copied into wasm memory
    /// on Modern.
    fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId;
    fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId;
    fn register_video(&mut self, video: web_sys::HtmlVideoElement) -> HostImageId;
    /// Bytes uploaded since the device was made: from wasm memory, from JS.
    fn uploaded(&self) -> Uploaded;
    /// A colour target's pixels, rows top to bottom in the target's own
    /// byte order; the future holds no borrow of the device.
    fn read_texture(
        &mut self,
        id: TextureId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>>>>;
    /// Times submissions from now on; WebGL2 has no timing.
    fn set_timing(&mut self, _on: bool) {}
    /// Each submission's times since the last call once the GPU finished
    /// them: its passes' GPU milliseconds by label, and its CPU ones.
    fn take_timings(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Timed>>>> {
        Box::pin(async { Vec::new() })
    }
}

/// A submission's times, milliseconds: per pass on the GPU, and the CPU's.
pub(crate) type Timed = (Vec<(String, f64)>, f64);

/// [`WebGpu::uploaded`]'s counts.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Uploaded {
    pub(crate) from_core: u64,
    pub(crate) from_js: u64,
    pub(crate) texels_from_core: u64,
    pub(crate) texels_from_js: u64,
}

#[cfg(feature = "modern")]
impl WebGpu for orb_hal_wgpu::WgpuGpu {
    fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId {
        orb_hal_wgpu::WgpuGpu::register_array_buffer(self, buffer)
    }
    fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId {
        orb_hal_wgpu::WgpuGpu::register_image_bitmap(self, bitmap)
    }
    fn register_video(&mut self, video: web_sys::HtmlVideoElement) -> HostImageId {
        orb_hal_wgpu::WgpuGpu::register_video(self, video)
    }
    fn uploaded(&self) -> Uploaded {
        let u = orb_hal_wgpu::WgpuGpu::uploaded(self);
        Uploaded {
            from_core: u.from_core,
            from_js: u.from_js,
            texels_from_core: u.texels_from_core,
            texels_from_js: u.texels_from_js,
        }
    }
    fn read_texture(
        &mut self,
        id: TextureId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>>>> {
        Box::pin(orb_hal_wgpu::WgpuGpu::read_texture(self, id))
    }
    fn set_timing(&mut self, on: bool) {
        orb_hal_wgpu::WgpuGpu::set_timing(self, on);
    }
    fn take_timings(&mut self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Timed>>>> {
        let timings = orb_hal_wgpu::WgpuGpu::take_timings(self);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        Box::pin(async move {
            timings
                .await
                .into_iter()
                .map(|t| {
                    let passes = t.passes.into_iter().map(|(l, d)| (l, ms(d))).collect();
                    (passes, ms(t.cpu))
                })
                .collect()
        })
    }
}

#[cfg(feature = "compat")]
impl WebGpu for orb_hal_gl::GlGpu {
    fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId {
        orb_hal_gl::GlGpu::register_array_buffer(self, buffer)
    }
    fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId {
        orb_hal_gl::GlGpu::register_image_bitmap(self, bitmap)
    }
    fn register_video(&mut self, video: web_sys::HtmlVideoElement) -> HostImageId {
        orb_hal_gl::GlGpu::register_video(self, video)
    }
    fn uploaded(&self) -> Uploaded {
        Uploaded {
            from_core: orb_hal_gl::GlGpu::uploaded(self),
            ..Uploaded::default()
        }
    }
    fn read_texture(
        &mut self,
        id: TextureId,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>>>> {
        let pixels = self.read_rgba8(id);
        Box::pin(async move { pixels })
    }
}

/// WebGL2 frames draw into an RGBA8 texture, presented to the canvas.
#[cfg(feature = "compat")]
const COMPAT_FORMAT: TargetFormat = TargetFormat {
    color: orb_hal::TextureFormat::Rgba8Unorm,
    sample_count: 1,
};

/// A canvas and the device drawing into it.
#[derive(Debug)]
pub(crate) enum Canvas {
    #[cfg(feature = "modern")]
    Modern(Box<Modern>),
    #[cfg(feature = "compat")]
    Compat(Box<Compat>),
}

#[cfg(feature = "modern")]
#[derive(Debug)]
pub(crate) struct Modern {
    gpu: orb_hal_wgpu::WgpuGpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    format: TargetFormat,
    /// Set by the device's lost callback.
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "compat")]
#[derive(Debug)]
pub(crate) struct Compat {
    gpu: orb_hal_gl::GlGpu,
    size: (u32, u32),
    /// The frame the map draws, copied into the canvas on present.
    frame: Option<(TextureId, (u32, u32))>,
}

impl Canvas {
    /// A device drawing into `canvas` (an `HTMLCanvasElement` or an
    /// `OffscreenCanvas`) of `size` physical pixels, which wakes map `map`
    /// when it's lost (Compat: the page hears `webglcontextlost` itself).
    #[cfg_attr(not(feature = "modern"), allow(unused_variables))]
    pub(crate) async fn new(canvas: JsValue, size: (u32, u32), map: u32) -> Result<Self, String> {
        let mut errors = Vec::new();
        #[cfg(feature = "modern")]
        match Modern::new(&canvas, size, map).await {
            Ok(modern) => return Ok(Self::Modern(Box::new(modern))),
            Err(e) => errors.push(e),
        }
        #[cfg(feature = "compat")]
        match orb_hal_gl::GlGpu::from_canvas(&canvas) {
            Ok(gpu) => {
                return Ok(Self::Compat(Box::new(Compat {
                    gpu,
                    size,
                    frame: None,
                })));
            }
            Err(e) => errors.push(e),
        }
        Err(errors.join("; "))
    }

    pub(crate) fn gpu(&mut self) -> &mut dyn WebGpu {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => &mut m.gpu,
            #[cfg(feature = "compat")]
            Self::Compat(c) => &mut c.gpu,
        }
    }

    /// Whether the GPU wakes the map when a pipeline compiles: WebGPU's
    /// promises do; WebGL2 has no callback, so frames poll.
    pub(crate) fn compile_wakes(&self) -> bool {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(_) => true,
            #[cfg(feature = "compat")]
            Self::Compat(_) => false,
        }
    }

    pub(crate) fn gpu_ref(&self) -> &dyn WebGpu {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => &m.gpu,
            #[cfg(feature = "compat")]
            Self::Compat(c) => &c.gpu,
        }
    }

    /// The format frames draw in: offscreen targets share it, and the
    /// map's pipelines with it.
    pub(crate) fn format(&self) -> TargetFormat {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => m.format,
            #[cfg(feature = "compat")]
            Self::Compat(_) => COMPAT_FORMAT,
        }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => (m.config.width, m.config.height),
            #[cfg(feature = "compat")]
            Self::Compat(c) => c.size,
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => {
                m.config.width = width;
                m.config.height = height;
                m.surface.configure(m.gpu.device(), &m.config);
            }
            #[cfg(feature = "compat")]
            Self::Compat(c) => c.size = (width, height),
        }
    }

    /// Whether the device was lost: the map needs another.
    pub(crate) fn lost(&self) -> bool {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => m.lost.load(std::sync::atomic::Ordering::Relaxed),
            #[cfg(feature = "compat")]
            Self::Compat(c) => c.gpu.lost(),
        }
    }

    /// Loses the device on purpose (`device.destroy()`, or WebGL's
    /// `WEBGL_lose_context`), to exercise recovery.
    pub(crate) fn lose(&mut self) {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => m.gpu.device().destroy(),
            #[cfg(feature = "compat")]
            Self::Compat(c) => c.gpu.lose_context(),
        }
    }

    /// Draws a frame with `draw`, which renders into the target it's
    /// given, and presents it.
    pub(crate) fn frame<R>(
        &mut self,
        draw: impl FnOnce(&mut dyn Gpu, TextureId, TargetFormat) -> R,
    ) -> Result<R, String> {
        match self {
            #[cfg(feature = "modern")]
            Self::Modern(m) => {
                let frame = match m.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(frame)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                    other => return Err(format!("canvas texture: {other:?}")),
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let target = m.gpu.register_view(view);
                let result = draw(&mut m.gpu, target, m.format);
                m.gpu.destroy_texture(target);
                m.gpu.queue().present(frame);
                Ok(result)
            }
            #[cfg(feature = "compat")]
            Self::Compat(c) => {
                let format = COMPAT_FORMAT;
                let frame = match c.frame {
                    Some((frame, size)) if size == c.size => frame,
                    stale => {
                        if let Some((frame, _)) = stale {
                            c.gpu.destroy_texture(frame);
                        }
                        let frame = c.gpu.create_texture(&orb_hal::TextureDesc {
                            label: "frame",
                            width: c.size.0,
                            height: c.size.1,
                            array_layers: 1,
                            dimension: orb_hal::TextureViewDimension::D2,
                            format: format.color,
                            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT,
                            sample_count: 1,
                            mip_levels: 1,
                        });
                        c.frame = Some((frame, c.size));
                        frame
                    }
                };
                let result = draw(&mut c.gpu, frame, format);
                c.gpu.present(frame, c.size.0, c.size.1);
                Ok(result)
            }
        }
    }
}

#[cfg(feature = "modern")]
impl Modern {
    async fn new(canvas: &JsValue, (width, height): (u32, u32), map: u32) -> Result<Self, String> {
        use orb_hal_wgpu::from_texture_format;
        use wasm_bindgen::JsCast;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        // An adapter before a surface: a canvas that gave out a WebGPU
        // context can't give a WebGL2 one, should there be no adapter.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let target = if let Some(canvas) = canvas.dyn_ref::<web_sys::HtmlCanvasElement>() {
            wgpu::SurfaceTarget::Canvas(canvas.clone())
        } else {
            wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone().unchecked_into())
        };
        let surface = instance
            .create_surface(target)
            .map_err(|e| format!("surface: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                // Timestamps time passes for measurements (`setGpuTiming`).
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU device: {e}"))?;
        let lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = lost.clone();
        device.set_device_lost_callback(move |_, _| {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            // The next frame sees the loss and the host recovers.
            crate::wake(map);
        });
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb() && from_texture_format(*f).is_some())
            .ok_or("no non-sRGB 8-bit canvas format")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let gpu = orb_hal_wgpu::WgpuGpu::new(device, queue);
        // A map waiting only for pipelines draws when one arrives (#70).
        gpu.set_compile_wake(std::rc::Rc::new(move || crate::wake(map)));
        Ok(Self {
            gpu,
            surface,
            config,
            format: TargetFormat {
                color: from_texture_format(surface_format).expect("checked above"),
                sample_count: 1,
            },
            lost,
        })
    }
}
