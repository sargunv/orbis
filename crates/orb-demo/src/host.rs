//! The window: winit's event loop on the main thread, and a render thread
//! that runs egui and the map and presents.
//!
//! The main thread only hands input over and carries out what a frame asks
//! of the window (cursor, IME, clipboard, accessibility, closing). Frames
//! wait for the display on the render thread: when they waited on the main
//! thread, as eframe has them, the window's input waited with them, and on
//! macOS a trackpad scroll's first moves came ~100 ms late
//! (rust-windowing/winit#1737).

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use egui::{RawInput, ViewportCommand, ViewportId, ViewportInfo};
use egui_winit::accesskit_winit;
use orb_api::Runtime;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::app::{Demo, Launch};

/// Makes the runtime, on the render thread that keeps it.
type MakeRuntime = Box<dyn FnOnce() -> Runtime + Send>;

/// Opens the window and runs the demo in it until it closes.
pub(crate) fn run(
    runtime: impl FnOnce() -> Runtime + Send + 'static,
    launch: Launch,
) -> Result<(), winit::error::EventLoopError> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut host = Host {
        start: Some((Box::new(runtime), launch)),
        proxy: event_loop.create_proxy(),
        open: None,
    };
    event_loop.run_app(&mut host)
}

/// What the render thread asks of the main thread.
enum UserEvent {
    /// A frame's requests of the window.
    Output {
        platform: egui::PlatformOutput,
        close: bool,
    },
    /// A surface for the window from this instance, which the window's
    /// thread makes: at the start, and when the last one is lost.
    MakeSurface(wgpu::Instance),
    /// The render thread has dropped the map and ended.
    Ended,
    AccessKit(accesskit_winit::Event),
}

impl From<accesskit_winit::Event> for UserEvent {
    fn from(event: accesskit_winit::Event) -> Self {
        Self::AccessKit(event)
    }
}

/// A trackpad gesture the platform recognized (macOS magnify and rotate),
/// with its phase, which egui drops, and when it came.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Trackpad {
    pub at: Instant,
    /// Above 1 spreads the fingers.
    pub scale: f64,
    /// Degrees the fingers turned clockwise.
    pub rotation: f64,
    pub phase: TouchPhase,
    /// A rotation's stream, not a pinch's: macOS phases each apart.
    pub rotating: bool,
}

/// What the main thread hands the render thread.
#[derive(Default)]
struct Inbox {
    /// Input since the last frame took it.
    input: Option<RawInput>,
    /// Trackpad gestures since the last frame took them.
    trackpad: Vec<Trackpad>,
    size: Option<PhysicalSize<u32>>,
    surface: Option<wgpu::Surface<'static>>,
    /// When egui or the map next wants a frame.
    repaint_at: Option<Instant>,
    quit: bool,
}

#[derive(Default)]
struct Shared {
    inbox: Mutex<Inbox>,
    changed: Condvar,
}

impl Shared {
    fn inbox(&self) -> MutexGuard<'_, Inbox> {
        self.inbox.lock().expect("inbox poisoned")
    }

    fn send(&self, f: impl FnOnce(&mut Inbox)) {
        f(&mut self.inbox());
        self.changed.notify_one();
    }

    /// Asks for a frame at `at`, or sooner if one is already due.
    fn repaint_at(&self, at: Instant) {
        self.send(|inbox| inbox.repaint_at = Some(inbox.repaint_at.map_or(at, |t| t.min(at))));
    }

    /// Blocks until a frame is due: input, a resize, a new surface or a
    /// repaint asked for. Returns false to quit.
    fn wait(&self) -> bool {
        let mut inbox = self.inbox();
        loop {
            if inbox.quit {
                return false;
            }
            if inbox.input.is_some()
                || !inbox.trackpad.is_empty()
                || inbox.size.is_some()
                || inbox.surface.is_some()
            {
                return true;
            }
            inbox = match inbox.repaint_at {
                Some(at) => match at.checked_duration_since(Instant::now()) {
                    Some(wait) if !wait.is_zero() => {
                        self.changed
                            .wait_timeout(inbox, wait)
                            .expect("inbox poisoned")
                            .0
                    }
                    _ => return true,
                },
                None => self.changed.wait(inbox).expect("inbox poisoned"),
            };
        }
    }

    /// Blocks until the main thread hands over a surface, or `None` if the
    /// window closes first.
    fn surface(&self) -> Option<wgpu::Surface<'static>> {
        let mut inbox = self.inbox();
        loop {
            if inbox.quit {
                return None;
            }
            if let Some(surface) = inbox.surface.take() {
                return Some(surface);
            }
            inbox = self.changed.wait(inbox).expect("inbox poisoned");
        }
    }
}

/// Folds `newer` input into what the render thread hasn't taken yet.
fn merge(older: Option<RawInput>, mut newer: RawInput) -> RawInput {
    let Some(mut older) = older else {
        return newer;
    };
    older.events.append(&mut newer.events);
    newer.events = older.events;
    older.dropped_files.append(&mut newer.dropped_files);
    newer.dropped_files = older.dropped_files;
    // `take_egui_input` hands these over once.
    newer.screen_rect = newer.screen_rect.or(older.screen_rect);
    newer.safe_area_insets = newer.safe_area_insets.or(older.safe_area_insets);
    newer.max_texture_side = newer.max_texture_side.or(older.max_texture_side);
    for (id, info) in &mut newer.viewports {
        if let Some(mut old) = older.viewports.remove(id) {
            old.events.append(&mut info.events);
            info.events = old.events;
        }
    }
    newer
}

struct Host {
    /// What the render thread starts with, until the window opens.
    start: Option<(MakeRuntime, Launch)>,
    proxy: EventLoopProxy<UserEvent>,
    open: Option<Open>,
}

/// The open window, on the main thread.
struct Open {
    window: Arc<Window>,
    state: egui_winit::State,
    info: ViewportInfo,
    shared: Arc<Shared>,
}

impl Open {
    /// Hands the input egui-winit has gathered to the render thread.
    fn send_input(&mut self) {
        let mut raw = self.state.take_egui_input(&self.window);
        let mut info = self.info.clone();
        info.events = raw
            .viewports
            .remove(&ViewportId::ROOT)
            .map(|v| v.events)
            .unwrap_or_default();
        raw.viewports.insert(ViewportId::ROOT, info);
        self.shared
            .send(|inbox| inbox.input = Some(merge(inbox.input.take(), raw)));
    }
}

impl ApplicationHandler<UserEvent> for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some((runtime, launch)) = self.start.take() else {
            return;
        };
        let attributes = Window::default_attributes()
            .with_title("Orbis")
            .with_inner_size(LogicalSize::new(1280.0, 800.0))
            // AccessKit's adapter must come before the window shows.
            .with_visible(false);
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        let ctx = egui::Context::default();
        let mut state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            window.theme(),
            // The render thread, which has the device, tells egui.
            None,
        );
        state.init_accesskit(event_loop, &window, self.proxy.clone());
        window.set_visible(true);
        let mut info = ViewportInfo::default();
        egui_winit::update_viewport_info(&mut info, &ctx, &window, true);

        let shared = Arc::new(Shared::default());
        let repaint = shared.clone();
        ctx.set_request_repaint_callback(move |request| {
            repaint.repaint_at(Instant::now() + request.delay);
        });
        let size = window.inner_size();
        let mut open = Open {
            window,
            state,
            info,
            shared: shared.clone(),
        };
        open.send_input();
        self.open = Some(open);

        let proxy = self.proxy.clone();
        std::thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                render(runtime(), launch, &ctx, &shared, &proxy, size);
                let _ = proxy.send_event(UserEvent::Ended);
            })
            .expect("render thread");
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        match &event {
            // The render thread drops the map, then says it has ended.
            WindowEvent::CloseRequested => {
                open.shared.send(|inbox| inbox.quit = true);
                return;
            }
            // The render thread draws when it has something new.
            WindowEvent::RedrawRequested => return,
            WindowEvent::Resized(size) => {
                let size = *size;
                open.shared.send(|inbox| inbox.size = Some(size));
            }
            // The map's, with their phases; egui has no use for them.
            &WindowEvent::PinchGesture { delta, phase, .. } => {
                let gesture = Trackpad {
                    at: Instant::now(),
                    scale: delta.exp(),
                    rotation: 0.0,
                    phase,
                    rotating: false,
                };
                open.shared.send(|inbox| inbox.trackpad.push(gesture));
                return;
            }
            &WindowEvent::RotationGesture { delta, phase, .. } => {
                // winit's turn counterclockwise.
                let gesture = Trackpad {
                    at: Instant::now(),
                    scale: 1.0,
                    rotation: -f64::from(delta),
                    phase,
                    rotating: true,
                };
                open.shared.send(|inbox| inbox.trackpad.push(gesture));
                return;
            }
            _ => {}
        }
        if matches!(
            event,
            WindowEvent::Resized(_)
                | WindowEvent::Moved(_)
                | WindowEvent::ScaleFactorChanged { .. }
                | WindowEvent::Focused(_)
                | WindowEvent::Occluded(_)
                | WindowEvent::ThemeChanged(_)
        ) {
            egui_winit::update_viewport_info(
                &mut open.info,
                open.state.egui_ctx(),
                &open.window,
                false,
            );
        }
        if open.state.on_window_event(&open.window, &event).repaint {
            open.send_input();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        match event {
            UserEvent::Output { platform, close } => {
                open.state.handle_platform_output(&open.window, platform);
                if close {
                    open.shared.send(|inbox| inbox.quit = true);
                }
            }
            UserEvent::MakeSurface(instance) => {
                let surface = instance
                    .create_surface(open.window.clone())
                    .expect("a surface for the window");
                open.shared.send(|inbox| inbox.surface = Some(surface));
            }
            UserEvent::Ended => event_loop.exit(),
            UserEvent::AccessKit(event) => match event.window_event {
                accesskit_winit::WindowEvent::InitialTreeRequested => {
                    open.state.egui_ctx().enable_accesskit();
                    open.state.egui_ctx().request_repaint();
                }
                accesskit_winit::WindowEvent::ActionRequested(request) => {
                    open.state.on_accesskit_action_request(request);
                    open.send_input();
                }
                accesskit_winit::WindowEvent::AccessibilityDeactivated => {
                    open.state.egui_ctx().disable_accesskit();
                }
            },
        }
    }
}

/// The render thread: a frame whenever one is due, until the window closes.
fn render(
    runtime: Runtime,
    launch: Launch,
    ctx: &egui::Context,
    shared: &Shared,
    proxy: &EventLoopProxy<UserEvent>,
    size: PhysicalSize<u32>,
) {
    eprintln!("orb-demo: {}", runtime.stats().adapter);
    // egui draws with the runtime's device, so the map's texture is one
    // it can sample.
    let made = runtime.with_device(|gpu| {
        let (instance, adapter) = gpu
            .instance_and_adapter()
            .expect("the runtime made its own device");
        (
            instance.clone(),
            adapter.clone(),
            gpu.device().clone(),
            gpu.queue().clone(),
        )
    });
    let (instance, adapter, device, queue) = match made {
        Ok(made) => made,
        Err(e) => {
            eprintln!("orb-demo: no GPU device: {e}");
            return;
        }
    };
    let _ = proxy.send_event(UserEvent::MakeSurface(instance.clone()));
    let Some(surface) = shared.surface() else {
        return;
    };
    let mut painter = Painter::new(surface, &adapter, device, queue, size);
    let mut demo = Demo::new(ctx, runtime, launch);

    let started = Instant::now();
    let max_texture_side = painter.device.limits().max_texture_dimension_2d as usize;
    // The window's state as the last input had it, for frames without
    // input: a default `RawInput` would reset the scale, focus and theme.
    let mut last = RawInput {
        max_texture_side: Some(max_texture_side),
        ..RawInput::default()
    };
    while shared.wait() {
        let (size, surface) = {
            let mut inbox = shared.inbox();
            (inbox.size.take(), inbox.surface.take())
        };
        if let Some(surface) = surface {
            painter.surface = surface;
            painter.configure();
        }
        if let Some(size) = size {
            painter.resize(size);
        }
        // Wait for the display first, so the frame takes the input that
        // came meanwhile.
        let frame = match painter.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                painter.stale = true;
                frame
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                painter.configure();
                shared.repaint_at(Instant::now());
                continue;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                let _ = proxy.send_event(UserEvent::MakeSurface(instance.clone()));
                continue;
            }
            // Hidden, or the display is slow: try again in a while.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                shared.repaint_at(Instant::now() + Duration::from_millis(100));
                continue;
            }
        };
        let input = {
            let mut inbox = shared.inbox();
            inbox.repaint_at = None;
            demo.trackpad(std::mem::take(&mut inbox.trackpad));
            inbox.input.take()
        };
        let mut raw = match input {
            Some(mut raw) => {
                raw.screen_rect = raw.screen_rect.or(last.screen_rect);
                raw.safe_area_insets = raw.safe_area_insets.or(last.safe_area_insets);
                raw.max_texture_side = raw.max_texture_side.or(last.max_texture_side);
                last = raw.clone();
                last.events.clear();
                last.dropped_files.clear();
                for info in last.viewports.values_mut() {
                    info.events.clear();
                }
                raw
            }
            None => last.clone(),
        };
        raw.time = Some(started.elapsed().as_secs_f64());

        let mut output = ctx.run_ui(raw, |ui| {
            demo.ui(ui, &painter.device, &mut painter.renderer);
        });
        let close = output
            .viewport_output
            .get(&ViewportId::ROOT)
            .is_some_and(|v| v.commands.contains(&ViewportCommand::Close));
        let _ = proxy.send_event(UserEvent::Output {
            platform: std::mem::take(&mut output.platform_output),
            close,
        });
        let shapes = std::mem::take(&mut output.shapes);
        let primitives = ctx.tessellate(shapes, output.pixels_per_point);
        painter.paint(
            frame,
            &primitives,
            &mut output.textures_delta,
            output.pixels_per_point,
        );
    }
}

/// egui's renderer and the window's surface, on the render thread.
struct Painter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: egui_wgpu::Renderer,
    /// The surface wants configuring again before the next frame.
    stale: bool,
}

impl Painter {
    fn new(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        size: PhysicalSize<u32>,
    ) -> Self {
        let caps = surface.get_capabilities(adapter);
        let format = egui_wgpu::preferred_framebuffer_format(&caps.formats)
            .expect("a surface format egui draws to");
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            // Low latency: a frame takes its input as late as it can.
            desired_maximum_frame_latency: 1,
            alpha_mode: caps.alpha_modes[0],
            view_formats: Vec::new(),
        };
        let renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        let painter = Self {
            device,
            queue,
            surface,
            config,
            renderer,
            stale: false,
        };
        painter.configure();
        painter
    }

    fn configure(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.configure();
    }

    fn paint(
        &mut self,
        frame: wgpu::SurfaceTexture,
        primitives: &[egui::ClippedPrimitive],
        textures: &mut egui::TexturesDelta,
        pixels_per_point: f32,
    ) {
        let size = frame.texture.size();
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point,
        };
        #[expect(clippy::iter_over_hash_type, reason = "textures update in any order")]
        for (id, deltas) in textures.set.drain() {
            for delta in deltas {
                self.renderer
                    .update_texture(&self.device, &self.queue, id, &delta);
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("egui"),
            });
        let commands = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            primitives,
            &screen,
        );
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(&mut pass.forget_lifetime(), primitives, &screen);
        self.queue
            .submit(commands.into_iter().chain([encoder.finish()]));
        self.queue.present(frame);
        #[expect(clippy::iter_over_hash_type, reason = "textures free in any order")]
        for id in textures.free.drain() {
            self.renderer.free_texture(&id);
        }
        if std::mem::take(&mut self.stale) {
            self.configure();
        }
    }
}
