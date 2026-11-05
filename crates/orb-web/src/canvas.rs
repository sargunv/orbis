//! The map's canvas and the WebGPU device drawing into it.

use orb_hal::{Gpu, TextureId};
use orb_hal_wgpu::WgpuGpu;
use orb_runtime::TargetFormat;
use wasm_bindgen::JsValue;

/// A frame's pixels, once read back.
pub(crate) type Pixels = std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>>>>;

/// A submission's times, milliseconds: per pass on the GPU, and the CPU's.
pub(crate) type Timed = (Vec<(String, f64)>, f64);

/// A canvas and the device drawing into it.
#[derive(Debug)]
pub(crate) struct Canvas {
    gpu: WgpuGpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    format: TargetFormat,
    /// Set by the device's lost callback.
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Canvas {
    pub(crate) fn gpu(&mut self) -> &mut WgpuGpu {
        &mut self.gpu
    }

    pub(crate) fn gpu_ref(&self) -> &WgpuGpu {
        &self.gpu
    }

    /// The format frames draw in: offscreen targets share it, and the
    /// map's pipelines with it.
    pub(crate) fn format(&self) -> TargetFormat {
        self.format
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(self.gpu.device(), &self.config);
    }

    /// Whether the device was lost: the map needs another.
    pub(crate) fn lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Loses the device on purpose (`device.destroy()`), to exercise
    /// recovery.
    pub(crate) fn lose(&mut self) {
        self.gpu.device().destroy();
    }

    /// Each submission's times since the last call once the GPU finished
    /// them: its passes' GPU milliseconds by label, and its CPU ones.
    pub(crate) fn take_timings(
        &mut self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<Timed>>>> {
        let timings = self.gpu.take_timings();
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

    /// Draws a frame with `draw`, which renders into the target it's
    /// given, and presents it; with `read`, also its pixels, in the
    /// canvas's byte order ([`format`](Self::format)).
    pub(crate) fn frame<R>(
        &mut self,
        read: bool,
        draw: impl FnOnce(&mut dyn Gpu, TextureId, TargetFormat) -> R,
    ) -> Result<(R, Option<Pixels>), String> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            other => return Err(format!("canvas texture: {other:?}")),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let target = self.gpu.register_view(view);
        let result = draw(&mut self.gpu, target, self.format);
        let pixels = read.then(|| Box::pin(self.gpu.read_wgpu_texture(&frame.texture)) as Pixels);
        self.gpu.destroy_texture(target);
        self.gpu.queue().present(frame);
        Ok((result, pixels))
    }

    /// A device drawing into `canvas` (an `HTMLCanvasElement` or an
    /// `OffscreenCanvas`) of `size` physical pixels, which wakes map `map`
    /// when it's lost.
    pub(crate) async fn new(
        canvas: JsValue,
        (width, height): (u32, u32),
        map: u32,
    ) -> Result<Self, String> {
        use orb_hal_wgpu::from_texture_format;
        use wasm_bindgen::JsCast;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let target = if let Some(canvas) = canvas.dyn_ref::<web_sys::HtmlCanvasElement>() {
            wgpu::SurfaceTarget::Canvas(canvas.clone())
        } else {
            wgpu::SurfaceTarget::OffscreenCanvas(canvas.unchecked_into())
        };
        let surface = instance
            .create_surface(target)
            .map_err(|e| format!("surface: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                // Timestamps time passes for measurements (`setGpuTiming`);
                // clip distances clip cut buildings without discarding.
                required_features: adapter.features()
                    & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLIP_DISTANCES),
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
            // Frames are read back for `readNextFrame`; WebGPU canvases take
            // any usage, though wgpu's capabilities list only this one.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
        let gpu = WgpuGpu::new(device, queue);
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
