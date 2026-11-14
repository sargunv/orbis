//! The map's canvas and the device drawing into it: WebGPU in core.wasm,
//! WebGL 2 in core-gl.wasm (the `gl` feature).

use orb_hal::{Gpu, TextureId};
use orb_hal_wgpu::WgpuGpu;
use orb_runtime::TargetFormat;

/// Whether this build draws with WebGL 2 (core-gl.wasm).
pub(crate) const GL: bool = cfg!(feature = "gl");

/// The GPU API's name, in messages: a literal, for `concat!`.
#[cfg(not(feature = "gl"))]
macro_rules! api {
    () => {
        "WebGPU"
    };
}
#[cfg(feature = "gl")]
macro_rules! api {
    () => {
        "WebGL 2"
    };
}

/// A frame's pixels, once read back.
pub(crate) type Pixels = std::pin::Pin<Box<dyn std::future::Future<Output = Vec<u8>>>>;

/// A submission's times, milliseconds: per pass and in all on the GPU,
/// and the CPU's.
pub(crate) type Timed = (Vec<(String, f64)>, f64, f64);

/// A canvas and the device drawing into it.
#[derive(Debug)]
pub(crate) struct Canvas {
    gpu: WgpuGpu,
    /// The instance and surface, which a new device after a loss draws
    /// into again ([`again`](Self::again)).
    instance: wgpu::Instance,
    surface: std::rc::Rc<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    /// WebGPU's context takes `config` with the next frame it presents,
    /// not at once: Firefox paints a context configured (or a canvas
    /// resized) since its last frame from a snapshot it waits for, on the
    /// page's thread, behind every compile asked for before it.
    configure_due: bool,
    format: TargetFormat,
    /// Set by the device's lost callback.
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// WebGL 2's frames read back: drawn here, then copied to the canvas.
    read: Option<ReadTarget>,
    /// The canvas, whose WebGL 2 context says when it's lost.
    #[cfg_attr(not(feature = "gl"), allow(dead_code))]
    element: web_sys::HtmlCanvasElement,
}

/// What [`Canvas::frame`] drew: the draw's result, the pixels read, and
/// whether the canvas presented the frame (not if the frame drew nothing
/// into it, as one held while compiles run).
pub(crate) struct Drawn<R> {
    pub(crate) result: R,
    pub(crate) pixels: Option<Pixels>,
    pub(crate) presented: bool,
}

/// Where a WebGL 2 frame that's read back draws: its canvas frame is a
/// renderbuffer wgpu blits to the canvas, which nothing else may read. The
/// frame draws into a texture, which is read and then drawn onto the
/// canvas.
#[derive(Debug)]
struct ReadTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl ReadTarget {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        (width, height): (u32, u32),
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("read frame"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let [vertex, fragment] = READ_BLIT.map(|glsl| {
            // SAFETY: GLSL ES 3.00 that binds what the layout below has.
            #[allow(unsafe_code)]
            unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("read frame"),
                    entry_points: std::borrow::Cow::Borrowed(&[
                        wgpu::PassthroughShaderEntryPoint {
                            name: std::borrow::Cow::Borrowed("main"),
                            workgroup_size: (0, 0, 0),
                        },
                    ]),
                    glsl: Some(glsl.into()),
                    ..Default::default()
                })
            }
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("read frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("read frame"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("read frame"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &vertex,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("read frame"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            view,
            group,
            pipeline,
        }
    }

    /// Draws the frame read onto the canvas's, `onto`.
    fn copy_to(&self, device: &wgpu::Device, queue: &wgpu::Queue, onto: &wgpu::TextureView) {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("read frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: onto,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

/// A triangle over the canvas, each pixel the read frame's, as GLSL ES
/// 3.00 wgpu takes as it is (its passthrough binding line names the frame's
/// binding).
const READ_BLIT: [&str; 2] = [
    "#version 300 es
void main() {
    uint i = uint(gl_VertexID);
    vec2 uv = vec2(float((i << 1u) & 2u), float(i & 2u));
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
",
    "#version 300 es
precision highp float;
//@texture frame 0 0
uniform highp sampler2D frame;
layout(location = 0) out vec4 color;
void main() {
    color = texelFetch(frame, ivec2(gl_FragCoord.xy), 0);
}
",
];

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

    /// Sizes the drawing buffer; a lost device's waits for the new
    /// device, made at this size.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        if GL {
            if !self.lost() {
                self.surface.configure(self.gpu.device(), &self.config);
            }
        } else {
            self.configure_due = true;
        }
    }

    /// Whether the device was lost: the map needs another.
    pub(crate) fn lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Loses the device on purpose (`device.destroy()`; on WebGL 2, the
    /// context, as `WEBGL_lose_context` does), to exercise recovery.
    /// WebGPU's loss is marked here at once rather than by the device's
    /// lost callback: Firefox can leave `device.lost` pending after
    /// `destroy()` (a map on the page after the e2e `edits` check's edits
    /// and queries never heard of it).
    pub(crate) fn lose(&mut self) {
        #[cfg(feature = "gl")]
        let _ = crate::gl_context::lose(&self.element);
        #[cfg(not(feature = "gl"))]
        {
            self.gpu.device().destroy();
            self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
        }
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
                    (passes, ms(t.span), ms(t.cpu))
                })
                .collect()
        })
    }

    /// Draws a frame with `draw`, which renders into the target it's
    /// given, and presents it; with `read`, also its pixels, in the
    /// canvas's byte order ([`format`](Self::format)). The canvas's
    /// texture is taken only once a pass draws into it: a frame that
    /// draws nothing presents nothing (`None` pixels then).
    pub(crate) fn frame<R>(
        &mut self,
        read: bool,
        draw: impl FnOnce(&mut dyn Gpu, TextureId, TargetFormat) -> R,
    ) -> Result<Drawn<R>, String> {
        // WebGL 2's frame read back draws into a texture first.
        if read && GL {
            return self.frame_read_gl(draw);
        }
        let config = self.configure_due.then(|| self.config.clone());
        let target = self.gpu.register_surface(self.surface.clone(), config);
        let result = draw(&mut self.gpu, target, self.format);
        let frame = self.gpu.take_surface(target).transpose()?;
        let presented = frame.is_some();
        self.configure_due &= !presented;
        let pixels = frame.and_then(|frame| {
            let pixels =
                read.then(|| Box::pin(self.gpu.read_wgpu_texture(&frame.texture)) as Pixels);
            self.gpu.queue().present(frame);
            pixels
        });
        Ok(Drawn {
            result,
            pixels,
            presented,
        })
    }

    /// [`frame`](Self::frame) read back on WebGL 2: drawn into a texture,
    /// read from there and copied into the canvas's.
    fn frame_read_gl<R>(
        &mut self,
        draw: impl FnOnce(&mut dyn Gpu, TextureId, TargetFormat) -> R,
    ) -> Result<Drawn<R>, String> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            other => return Err(format!("canvas texture: {other:?}")),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let size = self.size();
        let fits = |r: &ReadTarget| (r.texture.width(), r.texture.height()) == size;
        if !self.read.as_ref().is_some_and(fits) {
            let format = frame.texture.format();
            self.read = Some(ReadTarget::new(self.gpu.device(), format, size));
        }
        let drawn = self.read.as_ref().expect("made above");
        let target = self.gpu.register_view(drawn.view.clone());
        let result = draw(&mut self.gpu, target, self.format);
        let drawn = self.read.as_ref().expect("made above");
        let pixels = Box::pin(self.gpu.read_wgpu_texture(&drawn.texture)) as Pixels;
        drawn.copy_to(self.gpu.device(), self.gpu.queue(), &view);
        self.gpu.destroy_texture(target);
        self.gpu.queue().present(frame);
        Ok(Drawn {
            result,
            pixels: Some(pixels),
            presented: true,
        })
    }

    /// A device drawing into `canvas`, of `size` physical pixels, which
    /// wakes map `map` when it's lost.
    pub(crate) async fn new(
        canvas: web_sys::HtmlCanvasElement,
        size: (u32, u32),
        map: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("surface: {e}"))?;
        Self::on(instance, std::rc::Rc::new(surface), canvas, size, map).await
    }

    /// A new device drawing into this canvas, after a loss. WebGPU's draws
    /// into the same surface, its canvas's one context configured anew.
    /// WebGL 2's context comes back on the same canvas once restored, and
    /// a new surface takes it.
    pub(crate) fn again(
        &self,
        map: u32,
    ) -> impl std::future::Future<Output = Result<Self, String>> + use<> {
        let (element, size) = (self.element.clone(), self.size());
        let reuse = (!GL).then(|| (self.instance.clone(), self.surface.clone()));
        async move {
            match reuse {
                Some((instance, surface)) => Self::on(instance, surface, element, size, map).await,
                None => {
                    #[cfg(feature = "gl")]
                    crate::gl_context::restored(&element).await?;
                    Self::new(element, size, map).await
                }
            }
        }
    }

    /// A device of `instance` drawing into `surface`, `canvas`'s.
    async fn on(
        instance: wgpu::Instance,
        surface: std::rc::Rc<wgpu::Surface<'static>>,
        canvas: web_sys::HtmlCanvasElement,
        (width, height): (u32, u32),
        map: u32,
    ) -> Result<Self, String> {
        use orb_hal_wgpu::from_texture_format;

        let lost = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        #[cfg(feature = "gl")]
        crate::gl_context::watch(&canvas, lost.clone(), map);
        #[cfg(not(feature = "gl"))]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        // WebGL 2's adapter is the canvas's context: it comes after the
        // surface.
        #[cfg(feature = "gl")]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGL 2 adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                required_features: orb_hal_wgpu::device_features(&adapter),
                required_limits: orb_hal_wgpu::device_limits(&adapter),
                ..Default::default()
            })
            .await
            .map_err(|e| format!(concat!("no ", api!(), " device: {}"), e))?;
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            // Why, where the map didn't lose it on purpose (`lose`): the
            // map draws on through another device, which says nothing of
            // this one.
            if reason != wgpu::DeviceLostReason::Destroyed {
                let why = format!("orbis: the GPU device was lost: {message}");
                web_sys::console::warn_1(&why.into());
            }
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            // The next frame sees the loss and the host recovers.
            crate::map::wake(map);
        });
        // A context made while the GPU was gone can come lost.
        #[cfg(feature = "gl")]
        if crate::gl_context::is_lost(&canvas) {
            lost.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "gl")]
        crate::gl_context::first_vertex(&canvas);
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
            #[cfg(not(feature = "gl"))]
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            #[cfg(feature = "gl")]
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            // wgpu says WebGL 2 presents opaque, yet blends nothing: the
            // context, premultiplied as browsers make it, shows the page
            // through.
            #[cfg(not(feature = "gl"))]
            alpha_mode: wgpu::CompositeAlphaMode::PreMultiplied,
            #[cfg(feature = "gl")]
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        if GL {
            surface.configure(&device, &config);
        }
        // WebGPU's copies only: WebGL 2 copies images at submission.
        if !GL {
            warm_image_copies(device.clone(), queue.clone());
        }
        #[cfg_attr(not(feature = "gl"), allow(unused_mut))]
        let mut gpu = WgpuGpu::new(device, queue);
        #[cfg(feature = "gl")]
        gpu.read_formats(&adapter);
        // A map waiting only for pipelines draws when one arrives.
        gpu.set_compile_wake(std::rc::Rc::new(move || crate::map::wake(map)));
        Ok(Self {
            gpu,
            instance,
            surface,
            config,
            configure_due: !GL,
            format: TargetFormat {
                color: from_texture_format(surface_format).expect("checked above"),
                sample_count: 1,
            },
            lost,
            read: None,
            element: canvas,
        })
    }
}

/// Copies a 1×1 bitmap to a texture, in the background. A page's first
/// image copy makes the browser's copying context (about 10 ms on the
/// page's thread in Chrome): made now, while the style loads, rather than
/// when the first raster tile arrives.
fn warm_image_copies(device: wgpu::Device, queue: wgpu::Queue) {
    use wasm_bindgen::JsCast;
    wasm_bindgen_futures::spawn_local(async move {
        let bitmap = async {
            let pixels = web_sys::ImageData::new_with_sw(1, 1)?;
            let global = js_sys::global();
            let create = js_sys::Reflect::get(&global, &"createImageBitmap".into())?
                .dyn_into::<js_sys::Function>()?;
            let promise = create
                .call1(&global, &pixels)?
                .dyn_into::<js_sys::Promise>()?;
            wasm_bindgen_futures::JsFuture::from(promise)
                .await?
                .dyn_into::<web_sys::ImageBitmap>()
        };
        let Ok(bitmap) = bitmap.await else {
            return;
        };
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("warm image copies"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::ImageBitmap(bitmap.clone()),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            size,
        );
        bitmap.close();
        texture.destroy();
    });
}
