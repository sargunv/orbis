//! The native demo: a v8 style in a window, moved with `orb-interact`'s
//! gestures (mouse, trackpad, touch and keyboard). A click prints the
//! features under it.
//!
//! `cargo run --release -p orb-demo -- [style URL] [--at lon,lat,zoom[,bearing,pitch]]
//! [--format mvt|mlt] [--flight flight.json] [--snapshot out.png [--a11y out.json]]`
//!
//! Frame counters are shown in the window title. `--flight` flies a loop
//! of `flyTo` legs until the user takes the camera (the terrain demo:
//! `file://$PWD/web/demo/data/alps.json --flight
//! web/demo/data/alps-flight.json`). `--snapshot` renders 1024×768
//! offscreen until everything has loaded, writes a PNG and exits; `--a11y`
//! also writes the frame's semantic snapshot (plan §13) as JSON.
//! `--count-frames <seconds>` prints how many frames the window drew in
//! that long after the map first went idle, then exits.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Instant;

use glam::DVec2;
use orb_camera::{Camera, CameraTarget};
use orb_geo::Geodetic;
use orb_hal::Gpu;
use orb_hal_wgpu::{WgpuGpu, from_texture_format};
use orb_host_native::{NativeFetch, NativeImageDecode};
use orb_interact::{Button, InteractEvent, Interaction, Key, Modifiers, PointerKind, WheelUnit};
use orb_jobs::ThreadPool;
use orb_runtime::{
    CameraState, Map, MapEvent, PoolJobs, SceneCapacity, Services, TargetFormat, Viewport,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::{Window, WindowId};

const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

/// Data arrived; draw a frame.
#[derive(Debug)]
enum Wake {
    Data,
    Edit(String),
}

struct Options {
    style: Option<String>,
    encoding: String,
    /// Target, zoom, bearing and pitch.
    at: Option<(Geodetic, f64, f64, f64)>,
    snapshot: Option<String>,
    /// With `snapshot`: where to write the semantic snapshot's JSON.
    a11y: Option<String>,
    watch_style: Option<String>,
    /// Count the frames drawn this many seconds after the map first goes
    /// idle, print them and exit: idle means zero frames (plan §7.5).
    count_frames: Option<f64>,
    /// A flight file to fly in a loop (`web/demo/data/alps-flight.json`).
    flight: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        style: None,
        encoding: "mvt".into(),
        at: None,
        snapshot: None,
        a11y: None,
        watch_style: None,
        count_frames: None,
        flight: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--at" {
            let value = args
                .next()
                .ok_or("--at needs lon,lat,zoom[,bearing,pitch]")?;
            let parts: Vec<f64> = value
                .split(',')
                .map(|p| p.trim().parse::<f64>().map_err(|e| format!("--at: {e}")))
                .collect::<Result<_, _>>()?;
            let (lon, lat, zoom, bearing, pitch) = match parts[..] {
                [lon, lat, zoom] => (lon, lat, zoom, 0.0, 0.0),
                [lon, lat, zoom, bearing, pitch] => (lon, lat, zoom, bearing, pitch),
                _ => return Err("--at needs lon,lat,zoom[,bearing,pitch]".into()),
            };
            options.at = Some((Geodetic::new(lon, lat), zoom, bearing, pitch));
        } else if arg == "--count-frames" {
            let seconds = args.next().ok_or("--count-frames needs seconds")?;
            options.count_frames = Some(
                seconds
                    .parse()
                    .map_err(|e| format!("--count-frames: {e}"))?,
            );
        } else if arg == "--flight" {
            options.flight = Some(args.next().ok_or("--flight needs a flight file")?);
        } else if arg == "--watch-style" {
            options.watch_style = Some(args.next().ok_or("--watch-style needs a path")?);
        } else if arg == "--snapshot" {
            options.snapshot = Some(args.next().ok_or("--snapshot needs a path")?);
        } else if arg == "--a11y" {
            options.a11y = Some(args.next().ok_or("--a11y needs a path")?);
        } else if arg == "--format" {
            options.encoding = args.next().ok_or("--format needs mvt or mlt")?;
            if !matches!(options.encoding.as_str(), "mvt" | "mlt") {
                return Err("--format needs mvt or mlt".into());
            }
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg}"));
        } else {
            options.style = Some(arg);
        }
    }
    Ok(options)
}

fn main() {
    let options = match options() {
        Ok(options) => options,
        Err(e) => {
            eprintln!("orb-demo: {e}");
            std::process::exit(2);
        }
    };
    if let Some(path) = &options.snapshot {
        if let Err(e) = snapshot(&options, path) {
            eprintln!("orb-demo: {e}");
            std::process::exit(1);
        }
        return;
    }
    let event_loop = EventLoop::<Wake>::with_user_event()
        .build()
        .expect("event loop");
    if let Some(path) = options.watch_style.clone() {
        let proxy = event_loop.create_proxy();
        std::thread::Builder::new()
            .name("playground-style".into())
            .spawn(move || {
                let mut previous = String::new();
                loop {
                    if let Ok(json) = std::fs::read_to_string(&path)
                        && json != previous
                    {
                        previous.clone_from(&json);
                        if proxy
                            .send_event(Wake::Edit(if json.trim_start().starts_with('[') {
                                json
                            } else {
                                format!("[[\"setStyle\",{json}]]")
                            }))
                            .is_err()
                        {
                            break;
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(16));
                }
            })
            .expect("style watcher");
    }
    let mut app = App {
        options,
        proxy: event_loop.create_proxy(),
        state: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// HTTP and file fetches, and tile jobs on a thread pool.
fn native_services(wake: Arc<dyn Fn() + Send + Sync>) -> Services {
    let fetch = NativeFetch::new();
    let executor = Arc::new(ThreadPool::new("jobs", ThreadPool::default_size()));
    let jobs = PoolJobs::new(fetch.clone(), Arc::new(NativeImageDecode), executor);
    jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
    jobs.set_font_files(Arc::new(orb_host_native::FontFileFaces::new()));
    Services {
        clock: Arc::new(orb_host_native::NativeClock::default()),
        image_decode: Arc::new(orb_host_native::NativeImageDecode),
        images: Default::default(),
        jobs,
        fetch,
        wake,
    }
}

/// The committed extract and reduced Liberty style, or an explicit style URL.
fn load_style(map: &mut Map, options: &Options) -> Result<(), String> {
    if let Some(url) = &options.style {
        map.load_style_url(url);
        return Ok(());
    }
    let path = std::path::Path::new(DEMO_DATA)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let json = std::fs::read_to_string(path.join(format!("tokyo-{}.json", options.encoding)))
        .map_err(|e| e.to_string())?;
    // Tiles, glyphs and sprites beside the style.
    let json = json
        .replace(
            "pmtiles://./data/",
            &format!("pmtiles://file://{}/", path.display()),
        )
        .replace("\"./data/", &format!("\"file://{}/", path.display()));
    map.set_style_json(&json)
}

/// The default surface's projection.
fn web_mercator() -> orb_crs::Projection {
    *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// The camera from `--at`, or `None` to use the style's. Its zoom is on
/// the default Web Mercator surface: a scene document's own plane is known
/// only once it loads.
fn camera(options: &Options) -> Option<CameraState> {
    options
        .at
        .map(|(target, zoom, bearing, pitch)| CameraState {
            bearing,
            pitch,
            ..CameraState::new(target, zoom, &web_mercator())
        })
}

/// Renders offscreen until the map is idle and writes the frame as a PNG.
/// A map rendering offscreen at 1024×768, for snapshots and tests.
struct Offscreen {
    gpu: WgpuGpu,
    target: orb_hal::TextureId,
    format: TargetFormat,
    viewport: Viewport,
    map: Map,
}

impl Offscreen {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 768;

    /// Loads the style and renders until everything in view has arrived.
    /// Returns the map, the last frame's report and the frame count.
    fn load(options: &Options) -> Result<(Self, orb_runtime::FrameReport, u32), String> {
        Self::load_with(options, SceneCapacity::default())
    }

    /// Like [`load`](Self::load), with a given GPU scene capacity.
    fn load_with(
        options: &Options,
        capacity: SceneCapacity,
    ) -> Result<(Self, orb_runtime::FrameReport, u32), String> {
        let mut this = Self::new(options, capacity)?;
        let started = Instant::now();
        let mut frames = 0;
        // Two idle frames in a row: the second draws what the first's
        // messages installed.
        let mut was_idle = false;
        loop {
            let report = this.frame();
            frames += 1;
            for warning in this.map.take_warnings() {
                eprintln!("orb-demo: {warning}");
            }
            if report.idle && was_idle {
                return Ok((this, report, frames));
            }
            was_idle = report.idle;
            if started.elapsed().as_secs() > 60 {
                return Err("timed out waiting for tiles".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// The map with its style set, before anything has loaded.
    fn new(options: &Options, capacity: SceneCapacity) -> Result<Self, String> {
        let mut gpu = pollster::block_on(WgpuGpu::offscreen())?;
        let format = TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let target = gpu.create_texture(&orb_hal::TextureDesc {
            label: "snapshot",
            width: Self::WIDTH,
            height: Self::HEIGHT,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: format.color,
            usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let services = native_services(Arc::new(|| {}));
        let mut map = Map::new(services, camera(options));
        // Snapshots use the same white backing as the native demo window.
        map.set_clear_color([1.0; 4]);
        map.set_scene_capacity(capacity);
        load_style(&mut map, options)?;
        Ok(Self {
            gpu,
            target,
            format,
            viewport: Viewport {
                width: f64::from(Self::WIDTH),
                height: f64::from(Self::HEIGHT),
            },
            map,
        })
    }

    fn frame(&mut self) -> orb_runtime::FrameReport {
        // The camera is written directly: keep it to the world and on the
        // terrain, as a camera store would.
        self.map.constrain_camera(self.viewport);
        self.map
            .render(&mut self.gpu, self.target, self.format, self.viewport, 1.0)
    }

    fn pixels(&self) -> Vec<u8> {
        pollster::block_on(self.gpu.read_texture(self.target))
    }
}

/// Renders offscreen until idle and writes the frame as a PNG.
fn snapshot(options: &Options, path: &str) -> Result<Vec<u8>, String> {
    let started = Instant::now();
    let (mut offscreen, mut report, mut frames) = Offscreen::load(options)?;
    if let Some(path) = &options.watch_style {
        let json = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let commands = if json.trim_start().starts_with('[') {
            json
        } else {
            format!("[[\"setStyle\",{json}]]")
        };
        let started = Instant::now();
        let generation = offscreen
            .map
            .edit(&orb_runtime::Transaction::from_commands(&commands)?)?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut edit_frames = 0;
        loop {
            report = offscreen.frame();
            frames += 1;
            edit_frames += 1;
            if report.generation >= generation {
                break;
            }
            if started.elapsed().as_secs() > 60 {
                return Err("snapshot edit did not settle".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        eprintln!(
            "playground: generation {generation} visible in {edit_frames} frames; edit {ms:.4} ms"
        );
    }
    let pixels = offscreen.pixels();
    write_png(path, &pixels)?;
    eprintln!(
        "orb-demo: wrote {path} after {frames} frames, {:.0} ms, {} pipelines; {:?}",
        started.elapsed().as_secs_f64() * 1000.0,
        offscreen.gpu.pipelines_created(),
        report
    );
    if let Some(path) = &options.a11y {
        let snapshot = offscreen
            .map
            .semantic_snapshot()
            .ok_or("no labels were placed")?;
        std::fs::write(path, snapshot.to_json()).map_err(|e| format!("{path}: {e}"))?;
        eprintln!(
            "orb-demo: wrote {path}: {} nodes, {:?}",
            snapshot.nodes.len(),
            snapshot.summary.highlights
        );
    }
    if report.tiles_drawn == 0 {
        return Err("snapshot drew no tiles".into());
    }
    Ok(pixels)
}

struct App {
    options: Options,
    proxy: EventLoopProxy<Wake>,
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: WgpuGpu,
    format: TargetFormat,
    map: Map,
    /// The camera's single writer; the map draws what it shows.
    camera: Camera,
    input: Interaction,
    /// The camera revision last given to the map.
    pushed: u64,
    /// The map's terrain revision the camera stands on.
    terrain: u64,
    started: Instant,
    /// Frames drawn, and when the map first went idle with the count then.
    frames: u32,
    idle_at: Option<(Instant, u32)>,
    cursor: DVec2,
    modifiers: Modifiers,
    last_title: Instant,
    edit_pending: Vec<(u64, u32, f64)>,
    queued_edit: Option<String>,
    /// Flying until the user takes the camera.
    flight: Option<Flight>,
}

/// A loop of camera flights: legs of `flyTo` options (center, zoom,
/// bearing, pitch and duration in milliseconds), as the web demo flies them.
struct Flight {
    legs: Vec<(orb_camera::CameraTarget, orb_camera::FlyOptions)>,
    next: usize,
}

impl Flight {
    fn read(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let legs = json["legs"]
            .as_array()
            .ok_or_else(|| format!("{path}: no legs"))?
            .iter()
            .map(|leg| {
                let number = |key: &str| leg[key].as_f64();
                let center = leg["center"]
                    .as_array()
                    .and_then(|c| Some(Geodetic::new(c.first()?.as_f64()?, c.get(1)?.as_f64()?)));
                (
                    orb_camera::CameraTarget {
                        center,
                        zoom: number("zoom"),
                        bearing: number("bearing"),
                        pitch: number("pitch"),
                        ..Default::default()
                    },
                    orb_camera::FlyOptions {
                        duration: number("duration"),
                        ..Default::default()
                    },
                )
            })
            .collect::<Vec<_>>();
        if legs.is_empty() {
            return Err(format!("{path}: no legs"));
        }
        Ok(Self { legs, next: 0 })
    }
}

impl State {
    fn new(event_loop: &ActiveEventLoop, options: &Options, proxy: EventLoopProxy<Wake>) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Orbis"))
                .expect("window"),
        );
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("orbis"),
            ..Default::default()
        }))
        .expect("device");
        let caps = surface.get_capabilities(&adapter);
        // GL JS blends in the style's colour space, so draw to a non-sRGB
        // target.
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb() && from_texture_format(*f).is_some())
            .expect("a non-sRGB 8-bit surface format");
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let services = native_services(Arc::new(move || {
            let _ = proxy.send_event(Wake::Data);
        }));
        let initial = camera(options);
        let mut map = Map::new(services, initial);
        // The opaque swapchain has no page behind transparent style pixels.
        map.set_clear_color([1.0; 4]);
        load_style(&mut map, options).expect("demo style");
        let scale = window.scale_factor();
        let viewport = Viewport {
            width: f64::from(config.width) / scale,
            height: f64::from(config.height) / scale,
        };
        let camera = Camera::new(initial.unwrap_or(*map.camera()), map.surface(), viewport);

        Self {
            window,
            surface,
            config,
            gpu: WgpuGpu::new(device, queue),
            format: TargetFormat {
                color: from_texture_format(surface_format).expect("checked above"),
                sample_count: 1,
            },
            map,
            camera,
            input: Interaction::default(),
            pushed: 0,
            terrain: 0,
            started: Instant::now(),
            frames: 0,
            idle_at: None,
            cursor: DVec2::ZERO,
            modifiers: Modifiers::default(),
            last_title: Instant::now(),
            edit_pending: Vec::new(),
            queued_edit: None,
            flight: options
                .flight
                .as_deref()
                .map(|path| Flight::read(path).expect("flight file")),
        }
    }

    fn viewport(&self) -> Viewport {
        let scale = self.window.scale_factor();
        Viewport {
            width: f64::from(self.config.width) / scale,
            height: f64::from(self.config.height) / scale,
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(self.gpu.device(), &self.config);
        self.camera.set_viewport(self.viewport());
        self.window.request_redraw();
    }

    /// Milliseconds on the gesture clock.
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn input(&mut self, input: orb_interact::Input) {
        // The user takes the camera from a flight.
        self.flight = None;
        let now = self.now();
        self.input.handle(&mut self.camera, input, now);
        self.window.request_redraw();
    }

    /// Advances gestures and animations and gives the map the camera.
    fn move_camera(&mut self) -> bool {
        let now = self.now();
        if let Some(flight) = &mut self.flight
            && !self.camera.is_animating()
            && !self.camera.is_held()
        {
            let (target, options) = &flight.legs[flight.next];
            self.camera.fly_to(target, options, now);
            flight.next = (flight.next + 1) % flight.legs.len();
        }
        let gesture = self.input.frame(&mut self.camera, now);
        let animating = self.camera.frame(now);
        self.camera.take_events();
        for event in self.input.take_events() {
            if let InteractEvent::Click {
                position,
                button: Button::Primary,
            } = event
            {
                match self.map.hit_test(position) {
                    Some(hit) => println!(
                        "clicked {} ({}{}) id {}",
                        hit.layer,
                        hit.source,
                        hit.source_layer
                            .map(|l| format!(" / {l}"))
                            .unwrap_or_default(),
                        hit.id.as_deref().unwrap_or("none")
                    ),
                    None => println!("clicked nothing"),
                }
            }
        }
        // Until the app or the user moves it, the map's camera follows the
        // style.
        if self.camera.modified() && self.camera.revision() != self.pushed {
            self.pushed = self.camera.revision();
            self.map.set_camera(*self.camera.state());
        }
        gesture || animating
    }

    fn map_events(&mut self) {
        for event in self.map.take_events() {
            // Loads and projection edits move the camera's surface.
            if let MapEvent::StyleData { .. } | MapEvent::StyleLoad { .. } = event {
                self.camera.set_surface(self.map.surface());
            }
            match event {
                MapEvent::StyleLoad {
                    camera: Some(style),
                } if !self.camera.modified() => {
                    self.camera.jump_to(&CameraTarget {
                        center: Some(style.target),
                        zoom: Some(style.zoom(self.camera.projection())),
                        bearing: Some(style.bearing),
                        pitch: Some(style.pitch),
                        ..CameraTarget::default()
                    });
                    self.window.request_redraw();
                }
                MapEvent::Error { message } => eprintln!("orb-demo: {message}"),
                _ => {}
            }
        }
    }

    fn edit(&mut self, commands: String) {
        if self.map.style_doc().is_none() {
            self.queued_edit = Some(commands);
            return;
        }
        let started = Instant::now();
        match orb_runtime::Transaction::from_commands(&commands).and_then(|tx| self.map.edit(&tx)) {
            Ok(generation) => {
                self.edit_pending
                    .push((generation, 0, started.elapsed().as_secs_f64() * 1000.0))
            }
            Err(e) => eprintln!("playground: {e}"),
        }
        self.window.request_redraw();
    }

    fn redraw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(self.gpu.device(), &self.config);
                self.window.request_redraw();
                return;
            }
            other => {
                eprintln!("orb-demo: surface: {other:?}");
                return;
            }
        };
        let started = Instant::now();
        let moving = self.move_camera();
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let target = self.gpu.register_view(view);
        let viewport = self.viewport();
        let report = self.map.render(
            &mut self.gpu,
            target,
            self.format,
            viewport,
            self.window.scale_factor() as f32,
        );
        // The camera stands on the terrain the map has loaded.
        if self.map.elevation_revision() != self.terrain {
            self.terrain = self.map.elevation_revision();
            self.camera.set_terrain(self.map.elevation());
        }
        self.edit_pending.retain_mut(|(generation, frames, ms)| {
            *frames += 1;
            if report.generation >= *generation {
                eprintln!("playground: generation {generation} visible in {frames} frames; edit {ms:.4} ms; meshes {} rows {}",
                    self.map.job_counts().meshes_uploaded, self.map.job_counts().rows_uploaded);
                false
            } else { true }
        });
        if self.map.style_doc().is_some()
            && let Some(commands) = self.queued_edit.take()
        {
            self.edit(commands);
        }
        self.gpu.destroy_texture(target);
        self.window.pre_present_notify();
        self.gpu.queue().present(frame);
        self.map_events();
        self.frames += 1;
        if report.idle && !moving && self.idle_at.is_none() {
            self.idle_at = Some((Instant::now(), self.frames));
        }
        if !report.idle || moving {
            self.window.request_redraw();
        }
        for warning in self.map.take_warnings() {
            eprintln!("orb-demo: {warning}");
        }

        if self.last_title.elapsed().as_millis() > 200 {
            self.last_title = Instant::now();
            self.window.set_title(&format!(
                "Orbis · z{:.2} · {:.1} ms CPU · {} draws · {} dispatch · {} candidates · {} tiles drawn · {} loading · bundle {} · wrote {} placements, {} paints",
                self.map.zoom(),
                started.elapsed().as_secs_f64() * 1000.0,
                report.frame.draws,
                report.frame.dispatches,
                report.frame.candidates,
                report.tiles_drawn,
                report.tiles_loading,
                if report.frame.bundle_recorded { "recorded" } else { "replayed" },
                report.frame.placements_written,
                report.frame.paints_written,
            ));
        }
    }
}

impl ApplicationHandler<Wake> for App {
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(seconds), Some(state)) = (self.options.count_frames, &self.state) else {
            return;
        };
        let Some((idle, at)) = state.idle_at else {
            return;
        };
        let deadline = idle + std::time::Duration::from_secs_f64(seconds);
        if Instant::now() >= deadline {
            println!(
                "orb-demo: idle after {at} frames; {} frames in the {seconds} s since",
                state.frames - at
            );
            event_loop.exit();
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(State::new(event_loop, &self.options, self.proxy.clone()));
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: Wake) {
        if let Some(state) = &mut self.state {
            if let Wake::Edit(commands) = event {
                state.edit(commands);
            }
            state.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::Occluded(false) => state.window.request_redraw(),
            WindowEvent::RedrawRequested => state.redraw(),
            WindowEvent::ModifiersChanged(modifiers) => {
                let m = modifiers.state();
                state.modifiers = Modifiers {
                    shift: m.shift_key(),
                    ctrl: m.control_key(),
                    alt: m.alt_key(),
                    meta: m.super_key(),
                };
            }
            WindowEvent::MouseInput {
                state: pressed,
                button,
                ..
            } => {
                let button = match button {
                    MouseButton::Left => Button::Primary,
                    MouseButton::Right => Button::Secondary,
                    MouseButton::Middle => Button::Middle,
                    _ => return,
                };
                let position = state.cursor;
                state.input(match pressed {
                    ElementState::Pressed => orb_interact::Input::PointerDown {
                        id: 0,
                        kind: PointerKind::Mouse,
                        button,
                        position,
                        modifiers: state.modifiers,
                    },
                    ElementState::Released => orb_interact::Input::PointerUp { id: 0, position },
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = DVec2::new(position.x, position.y) / state.window.scale_factor();
                state.cursor = position;
                state.input(orb_interact::Input::PointerMove { id: 0, position });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // winit's deltas point away from the user; the DOM's, which
                // the gestures follow, toward.
                let (delta, unit) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (-DVec2::new(f64::from(x), f64::from(y)), WheelUnit::Line)
                    }
                    MouseScrollDelta::PixelDelta(p) => (
                        -DVec2::new(p.x, p.y) / state.window.scale_factor(),
                        WheelUnit::Pixel,
                    ),
                };
                let position = state.cursor;
                let modifiers = state.modifiers;
                state.input(orb_interact::Input::Wheel {
                    position,
                    delta,
                    unit,
                    modifiers,
                });
            }
            WindowEvent::Touch(touch) => {
                // Touch ids above the mouse's.
                let id = (touch.id as u32).wrapping_add(1);
                let position =
                    DVec2::new(touch.location.x, touch.location.y) / state.window.scale_factor();
                state.input(match touch.phase {
                    TouchPhase::Started => orb_interact::Input::PointerDown {
                        id,
                        kind: PointerKind::Touch,
                        button: Button::Primary,
                        position,
                        modifiers: state.modifiers,
                    },
                    TouchPhase::Moved => orb_interact::Input::PointerMove { id, position },
                    TouchPhase::Ended => orb_interact::Input::PointerUp { id, position },
                    TouchPhase::Cancelled => orb_interact::Input::PointerCancel { id },
                });
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let key = match event.logical_key.as_ref() {
                    WinitKey::Named(NamedKey::ArrowLeft) => Key::Left,
                    WinitKey::Named(NamedKey::ArrowRight) => Key::Right,
                    WinitKey::Named(NamedKey::ArrowUp) => Key::Up,
                    WinitKey::Named(NamedKey::ArrowDown) => Key::Down,
                    WinitKey::Character("+" | "=") => Key::ZoomIn,
                    WinitKey::Character("-" | "_") => Key::ZoomOut,
                    _ => return,
                };
                let modifiers = state.modifiers;
                state.input(orb_interact::Input::Key { key, modifiers });
            }
            WindowEvent::Focused(false) => state.input(orb_interact::Input::Blur),
            _ => {}
        }
    }
}

#[cfg(test)]
mod edits;

/// Writes an offscreen frame's RGBA pixels as a PNG.
fn write_png(path: &str, pixels: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        Offscreen::WIDTH,
        Offscreen::HEIGHT,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(pixels))
        .map_err(|e| format!("{path}: {e}"))
}

/// Pixels that differ by more than one level in a channel. Glyphs sit where
/// the shared atlas packed them, which depends on the order they arrived,
/// and filtering at another texel origin can round one level apart.
#[cfg(test)]
fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.abs_diff(*y) > 1))
        .count()
}

#[cfg(test)]
mod tests {
    fn tokyo(zoom: f64) -> super::Options {
        super::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_geo::Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0)),
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        }
    }

    /// The web ui role's approximate terrain (decision 0019): how far the
    /// elevation pyramid's elevations and terrain unprojections are from
    /// the map's over the Alps demo. Needs the network (Mapterhorn and
    /// OpenFreeMap tiles): `cargo test --release -p orb-demo -- --ignored
    /// pyramid`.
    #[test]
    #[ignore = "fetches tiles over the network"]
    fn pyramid_error_over_the_alps() {
        use orb_camera::Elevation;
        let style = format!(
            "file://{}/../../web/demo/data/alps.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let options = super::Options {
            style: Some(style),
            at: None,
            ..tokyo(0.0)
        };
        let (offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let full = offscreen.map.elevation().unwrap();
        let pyramid =
            orb_runtime::TerrainElevation::from_pyramid(&offscreen.map.elevation_pyramid())
                .unwrap();
        let viewport = offscreen.viewport;
        let surface = offscreen.map.surface();
        let view = orb_camera::View::over(offscreen.map.camera(), viewport, &surface, &*full);
        let (mut metres, mut pixels) = (Vec::new(), Vec::new());
        for i in 0..40 {
            for j in 0..30 {
                let px = glam::DVec2::new(
                    (f64::from(i) + 0.5) / 40.0 * viewport.width,
                    (f64::from(j) + 0.5) / 30.0 * viewport.height,
                );
                let Some((approx, _)) = view.unproject_terrain(px, &pyramid) else {
                    continue;
                };
                let (Some(true_m), Some(approx_m)) = (full.at(approx), pyramid.at(approx)) else {
                    continue;
                };
                metres.push((true_m - approx_m).abs());
                // Where the approximate point really draws.
                if let Some(drawn) = view.project_3d(approx, true_m * view.height_scale()) {
                    pixels.push(drawn.distance(px));
                }
            }
        }
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[v.len() * 95 / 100], v[v.len() - 1])
        };
        let (m, p) = (stats(&mut metres), stats(&mut pixels));
        println!(
            "pyramid over the Alps, {} pixels: elevation error median {:.1} m, p95 {:.1} m, max {:.1} m; unproject error median {:.2} px, p95 {:.2} px, max {:.2} px; {} bytes",
            metres.len(),
            m.0,
            m.1,
            m.2,
            p.0,
            p.1,
            p.2,
            offscreen.map.elevation_pyramid().len()
        );
        assert!(p.1 < 8.0, "p95 {} px", p.1);
    }

    /// The Alps flight through a camera store, as the demos
    /// fly them from the moment the style loads, with tiles loading in wall
    /// time: how far the eye and its near plane stay above the map's
    /// full-resolution terrain. The store takes the terrain as the native
    /// demo does (each revision) and as the web ui role does (the pyramid,
    /// at once and then every 30 frames). Needs the network: `cargo test
    /// --release -p orb-demo -- --ignored flight`.
    #[test]
    #[ignore = "fetches tiles over the network"]
    fn the_alps_flight_stays_above_the_terrain() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
        let options = super::Options {
            style: Some(format!("file://{root}/web/demo/data/alps.json")),
            at: None,
            ..tokyo(0.0)
        };
        let flight =
            super::Flight::read(&format!("{root}/web/demo/data/alps-flight.json")).unwrap();
        for web in [false, true] {
            let mut offscreen =
                super::Offscreen::new(&options, orb_runtime::SceneCapacity::default()).unwrap();
            // The style's camera.
            while offscreen.map.elevation_revision() == 0 {
                offscreen.frame();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let (viewport, surface) = (offscreen.viewport, offscreen.map.surface());
            let mut camera = orb_camera::Camera::new(*offscreen.map.camera(), surface, viewport);
            let (mut published, mut age) = (0, 0);
            let started = std::time::Instant::now();
            let (mut lowest, mut under, mut frames) = (f64::INFINITY, 0, 0);
            let (mut last, mut steps) = (None, Vec::new());
            for (target, fly) in &flight.legs {
                let now = started.elapsed().as_secs_f64() * 1000.0;
                camera.fly_to(target, fly, now);
                while camera.is_animating() {
                    let now = started.elapsed().as_secs_f64() * 1000.0;
                    camera.frame(now);
                    *offscreen.map.camera_mut() = *camera.state();
                    offscreen.map.render(
                        &mut offscreen.gpu,
                        offscreen.target,
                        offscreen.format,
                        viewport,
                        1.0,
                    );
                    age += 1;
                    let revision = offscreen.map.elevation_revision();
                    if revision != published && (!web || published == 0 || age >= 30) {
                        (published, age) = (revision, 0);
                        camera.set_terrain(if web {
                            orb_runtime::TerrainElevation::from_pyramid(
                                &offscreen.map.elevation_pyramid(),
                            )
                            .ok()
                            .map(|t| Arc::new(t) as _)
                        } else {
                            offscreen.map.elevation().map(|t| t as _)
                        });
                    }
                    let Some(full) = offscreen.map.elevation() else {
                        continue;
                    };
                    let view = orb_camera::View::new(camera.state(), viewport, &surface);
                    let Some(clearance) = view.clearance(&*full) else {
                        continue;
                    };
                    frames += 1;
                    lowest = lowest.min(clearance);
                    under += usize::from(clearance < 0.0);
                    // The camera's elevation step per frame, for jumps: the
                    // first DEM data under it moves it once.
                    let elevation = camera.state().elevation;
                    if let Some(previous) = last.replace(elevation) {
                        steps.push((elevation - previous).abs());
                    }
                }
            }
            steps.sort_by(f64::total_cmp);
            println!(
                "{} store: {frames} frames, lowest clearance {lowest:.1} m, {under} frames cut into the terrain; elevation steps per frame: p99 {:.1} m, largest {:.1} m, second {:.1} m",
                if web { "web" } else { "native" },
                steps[steps.len() * 99 / 100],
                steps[steps.len() - 1],
                steps[steps.len() - 2],
            );
            assert_eq!(under, 0);
        }
    }

    /// Zooms by `delta` levels around a pixel, as the gestures do.
    fn zoom_by(
        map: &mut orb_runtime::Map,
        viewport: orb_runtime::Viewport,
        at: glam::DVec2,
        delta: f64,
    ) {
        let surface = map.surface();
        let zoom = map.zoom() + delta;
        map.camera_mut().zoom_around(&surface, viewport, at, zoom);
        map.constrain_camera(viewport);
    }

    /// Pans the ground under `from` to `to`, as the gestures do.
    fn pan(
        map: &mut orb_runtime::Map,
        viewport: orb_runtime::Viewport,
        from: glam::DVec2,
        to: glam::DVec2,
    ) {
        let surface = map.surface();
        map.camera_mut().pan(&surface, viewport, from, to);
        map.constrain_camera(viewport);
    }

    /// Renders until two frames in a row are idle: the second draws what
    /// the first's messages installed, as the fixture runner's last frame
    /// does.
    fn settle(offscreen: &mut super::Offscreen) -> orb_runtime::FrameReport {
        let started = std::time::Instant::now();
        let mut was_idle = false;
        loop {
            let report = offscreen.frame();
            if report.idle && was_idle {
                assert!(offscreen.map.take_warnings().is_empty());
                return report;
            }
            was_idle = report.idle;
            assert!(started.elapsed().as_secs() < 60, "tiles did not settle");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn assert_matches_fresh(offscreen: &mut super::Offscreen) {
        let camera = *offscreen.map.camera();
        let options = super::Options {
            at: Some((
                camera.target,
                offscreen.map.zoom(),
                camera.bearing,
                camera.pitch,
            )),
            ..tokyo(offscreen.map.zoom())
        };
        let (mut fresh, _, _) = super::Offscreen::load(&options).unwrap();
        // Use exactly the same camera, including its ground resolution.
        *fresh.map.camera_mut() = camera;
        let fresh_report = settle(&mut fresh);
        let report = settle(offscreen);
        let pixels = offscreen.pixels();
        let reference = fresh.pixels();
        let differing = super::differing_pixels(&pixels, &reference);
        assert_eq!(
            differing, 0,
            "pixels differ from a fresh view: moved {report:?}, fresh {fresh_report:?}"
        );
    }

    #[test]
    fn zooming_out_matches_a_fresh_view() {
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(15.0)).unwrap();
        let viewport = offscreen.viewport;
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        for _ in 0..6 {
            zoom_by(&mut offscreen.map, viewport, center, -0.5);
            settle(&mut offscreen);
        }
        let (fresh, report, _) = super::Offscreen::load(&tokyo(12.0)).unwrap();
        let pixels = offscreen.pixels();
        let reference = fresh.pixels();
        let differing = super::differing_pixels(&pixels, &reference);
        let moved = offscreen.frame();
        eprintln!("zoom-out: {differing} differing pixels; moved {moved:?}; fresh {report:?}");
        if differing > 0 {
            // Keep both frames for the hunt (#63).
            let dir = std::env::temp_dir().join(format!(
                "orb-demo-zoom-out-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            std::fs::create_dir(&dir).unwrap();
            for (name, image) in [("moved.png", &pixels), ("fresh.png", &reference)] {
                super::write_png(dir.join(name).to_str().unwrap(), image).unwrap();
            }
            eprintln!("zoom-out: frames in {}", dir.display());
        }
        assert_eq!(
            differing, 0,
            "nominal z12: moved {moved:?}, fresh {report:?}"
        );
    }

    #[test]
    fn zooming_in_matches_a_fresh_view() {
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(12.0)).unwrap();
        let viewport = offscreen.viewport;
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        for _ in 0..6 {
            zoom_by(&mut offscreen.map, viewport, center, 0.5);
            settle(&mut offscreen);
        }
        assert_matches_fresh(&mut offscreen);
    }

    #[test]
    fn panning_matches_a_fresh_view() {
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(12.0)).unwrap();
        let viewport = offscreen.viewport;
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        for offset in [
            glam::DVec2::new(512.0, 128.0),
            glam::DVec2::new(-256.0, -64.0),
        ] {
            pan(&mut offscreen.map, viewport, center, center + offset);
            settle(&mut offscreen);
        }
        assert_matches_fresh(&mut offscreen);
    }

    /// A fling past zoom 0 lands a hair below it; the map must still draw
    /// there, as GL JS's does.
    #[test]
    fn draws_just_below_zoom_zero() {
        let options = super::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_geo::Geodetic::new(139.74, 35.685), 3.0, 0.0, 0.0)),
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let projection = offscreen.map.projection();
        offscreen.map.camera_mut().set_zoom(&projection, -1e-9);
        let mut report = offscreen.frame();
        while !report.idle {
            std::thread::sleep(std::time::Duration::from_millis(5));
            report = offscreen.frame();
        }
        report = offscreen.frame();
        assert!(report.frame.candidates > 0, "{report:?}");
        assert_ne!(
            &offscreen.pixels()[..4],
            [0, 0, 0, 0],
            "the background draws"
        );
    }

    /// Tiles never draw with some of their layers missing: when the scene
    /// is full, uploads evict unused cached tiles, so a small scene that has
    /// zoomed through several levels draws what an unbounded one does.
    #[test]
    fn a_full_scene_evicts_cached_tiles_instead_of_dropping_layers() {
        let at = |zoom| super::Options {
            style: None,
            encoding: "mvt".into(),
            at: Some((orb_geo::Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0)),
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (_, report, _) = super::Offscreen::load(&at(12.0)).unwrap();
        let used = report.scene;
        // Labels take more property words at the start than at 12.
        let (_, start, _) = super::Offscreen::load(&at(15.0)).unwrap();
        // Room for the views on the way, with their fallbacks, but not for
        // every level's tiles at once: vertex words run out.
        let capacity = super::SceneCapacity {
            vertex_words: used.vertex_words * 7 / 4,
            index_words: used.index_words * 2,
            prop_words: (used.prop_words.max(start.scene.prop_words) * 2).max(1024),
            ..super::SceneCapacity::fixed()
        };
        let zoom_out = |capacity| {
            let (mut offscreen, _, _) = super::Offscreen::load_with(&at(15.0), capacity).unwrap();
            let viewport = offscreen.viewport;
            let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
            let mut report = offscreen.frame();
            for _ in 0..6 {
                zoom_by(&mut offscreen.map, viewport, center, -0.5);
                report = offscreen.frame();
                while !report.idle {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    report = offscreen.frame();
                }
                report = offscreen.frame();
                let warnings = offscreen.map.take_warnings();
                assert!(
                    !warnings.iter().any(|w| w.contains("scene memory")),
                    "{warnings:?}"
                );
            }
            (offscreen, report)
        };
        let (unbounded, report) = zoom_out(super::SceneCapacity::default());
        assert!(
            report.scene.vertex_words > capacity.vertex_words,
            "the path must need eviction to fit: {} cached, {} budget",
            report.scene.vertex_words,
            capacity.vertex_words
        );
        let reference = unbounded.pixels();
        let (zoomed, _) = zoom_out(capacity);
        let pixels = zoomed.pixels();
        let differing = super::differing_pixels(&pixels, &reference);
        assert_eq!(differing, 0, "pixels differ from an unbounded scene");
    }

    /// A tile the scene can't hold even with nothing else to wait for is
    /// an error event and a warning, never a silent hole: its coarser
    /// stand-in draws instead.
    #[test]
    fn tiles_that_cannot_fit_are_errors() {
        let capacity = super::SceneCapacity {
            vertex_words: 2048,
            ..super::SceneCapacity::fixed()
        };
        let (mut offscreen, _, _) = super::Offscreen::load_with(&tokyo(15.0), capacity).unwrap();
        let started = std::time::Instant::now();
        let mut errors = Vec::new();
        loop {
            let report = offscreen.frame();
            offscreen.map.take_warnings();
            for event in offscreen.map.take_events() {
                if let orb_runtime::MapEvent::Error { message } = event {
                    errors.push(message);
                }
            }
            if report.idle {
                break;
            }
            assert!(started.elapsed().as_secs() < 60, "did not settle");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            errors
                .iter()
                .any(|e| e.contains("doesn't fit in scene memory")),
            "{errors:?}"
        );
    }

    /// Raster tiles make room as vector tiles do (decision 0032): with the
    /// scene full of cached vector tiles the view no longer uses, arriving
    /// raster tiles evict them rather than being dropped, which on the
    /// globe left Liberty's ocean (its shaded-relief raster) with
    /// tile-shaped gaps.
    #[test]
    fn raster_tiles_evict_unused_tiles_for_room() {
        let dir = std::env::temp_dir().join(format!("orbis-raster-room-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tile = dir.join("tile.png");
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 256, 256);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[64u8, 128, 255, 255].repeat(256 * 256))
                .unwrap();
        }
        std::fs::write(&tile, png).unwrap();
        // A polygon that only low zooms draw; imagery only from zoom 5.
        let ring: Vec<String> = (0..=360)
            .map(|i| {
                let a = f64::from(i).to_radians();
                format!("[{:.4},{:.4}]", 60.0 * a.cos(), 45.0 * a.sin())
            })
            .collect();
        let style = format!(
            r#"{{"version":8,"sources":{{
                "shapes":{{"type":"geojson","data":{{"type":"Polygon","coordinates":[[{ring}]]}}}},
                "imagery":{{"type":"raster","tiles":["file://{tile}"],"tileSize":256}}}},
              "layers":[{{"id":"fill","type":"fill","source":"shapes","maxzoom":4.5}},
                {{"id":"raster","type":"raster","source":"imagery","minzoom":5}}]}}"#,
            ring = ring.join(","),
            tile = tile.display(),
        );
        let path = dir.join("style.json");
        std::fs::write(&path, style).unwrap();
        let options = |lon: f64, zoom: f64| super::Options {
            style: Some(format!("file://{}", path.display())),
            at: Some((orb_geo::Geodetic::new(lon, 0.0), zoom, 0.0, 0.0)),
            ..tokyo(zoom)
        };
        let settle = |offscreen: &mut super::Offscreen| {
            let started = std::time::Instant::now();
            let mut warnings = Vec::new();
            loop {
                let report = offscreen.frame();
                warnings.extend(offscreen.map.take_warnings());
                if report.idle {
                    return (report, warnings);
                }
                assert!(started.elapsed().as_secs() < 60, "did not settle");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        // The scene holds exactly the low-zoom view's vector tiles.
        let (_, low, _) = super::Offscreen::load(&options(0.0, 3.0)).unwrap();
        let capacity = super::SceneCapacity {
            vertex_words: low.scene.vertex_words,
            index_words: low.scene.index_words,
            ..super::SceneCapacity::fixed()
        };
        let (mut offscreen, _, _) =
            super::Offscreen::load_with(&options(0.0, 3.0), capacity).unwrap();
        let (_, warnings) = settle(&mut offscreen);
        assert!(warnings.is_empty(), "{warnings:?}");
        *offscreen.map.camera_mut() = orb_runtime::CameraState::new(
            orb_geo::Geodetic::new(150.0, 0.0),
            6.0,
            &offscreen.map.projection(),
        );
        let (report, warnings) = settle(&mut offscreen);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(report.tiles_drawn > 0);
    }

    /// Plan §8: tiles out of view stay cached by their memory, within a
    /// quarter of the budget, the least recently used giving way first.
    #[test]
    fn tile_cache_stays_within_its_memory_budget() {
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(14.0)).unwrap();
        offscreen.map.set_memory_budget(16 << 20);
        let viewport = offscreen.viewport;
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        let mut most = 0;
        for step in 0..8 {
            let angle = f64::from(step) * 0.8;
            pan(
                &mut offscreen.map,
                viewport,
                center,
                center + glam::DVec2::new(angle.cos(), angle.sin()) * 600.0,
            );
            let report = settle(&mut offscreen);
            let memory = report.memory;
            assert_eq!(memory.cache_budget, 4 << 20);
            assert!(memory.cached <= memory.cache_budget, "{memory:?}");
            most = most.max(memory.cached_tiles);
        }
        assert!(most > 0, "tiles out of view are kept");
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct HeldExecutor(Mutex<Vec<orb_jobs::Job>>);

    impl orb_jobs::Executor for HeldExecutor {
        fn spawn(&self, job: orb_jobs::Job) {
            self.0.lock().unwrap().push(job);
        }
    }

    /// Plan §7.5: once idle, a map, its camera and its gestures ask for no
    /// frames: no wake-ups, no animation, nothing that redraws.
    #[test]
    fn idle_maps_ask_for_no_frames() {
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = wakes.clone();
        let services = super::native_services(Arc::new(move || {
            wake_count.fetch_add(1, Ordering::SeqCst);
        }));
        let mut offscreen = super::Offscreen::load(&tokyo(15.0)).unwrap().0;
        offscreen.map = super::Map::new(services, super::camera(&tokyo(15.0)));
        super::load_style(&mut offscreen.map, &tokyo(15.0)).unwrap();
        let report = settle(&mut offscreen);
        assert!(report.idle);
        assert!(
            offscreen
                .map
                .take_events()
                .contains(&orb_runtime::MapEvent::Idle)
        );
        let mut camera = super::Camera::new(
            *offscreen.map.camera(),
            offscreen.map.surface(),
            offscreen.viewport,
        );
        let mut input = super::Interaction::default();
        let before = wakes.load(Ordering::SeqCst);
        let started = std::time::Instant::now();
        let mut now = 0.0;
        while started.elapsed() < std::time::Duration::from_secs(2) {
            now += 16.0;
            assert!(!input.frame(&mut camera, now) && !camera.frame(now));
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        assert_eq!(
            wakes.load(Ordering::SeqCst),
            before,
            "no wake-ups while idle"
        );
        assert!(camera.take_events().is_empty());
        // The next frame, were one drawn, is still idle and reports nothing.
        assert!(offscreen.frame().idle);
        assert!(offscreen.map.take_events().is_empty());
    }

    #[test]
    fn delayed_results_wake_and_publish_without_camera_input() {
        let options = super::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let executor = Arc::new(HeldExecutor::default());
        let wakes = Arc::new(AtomicUsize::new(0));
        let wake_count = wakes.clone();
        let fetch = super::NativeFetch::new();
        let services = super::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(super::NativeImageDecode),
            images: Default::default(),
            jobs: super::PoolJobs::new(
                fetch.clone(),
                Arc::new(super::NativeImageDecode),
                executor.clone(),
            ),
            fetch,
            wake: Arc::new(move || {
                wake_count.fetch_add(1, Ordering::SeqCst);
            }),
        };
        let camera = super::CameraState::new(
            orb_geo::Geodetic::new(0.0, 0.0),
            1.5,
            &super::web_mercator(),
        );
        offscreen.map = super::Map::new(services, Some(camera));
        offscreen
            .map
            .set_style_json(
                r##"{
            "version":8,
            "sources":{"test":{"type":"geojson","data":{
                "type":"Polygon","coordinates":[[[-40,-40],[40,-40],[40,40],[-40,40],[-40,-40]]]
            }}},
            "layers":[{"id":"fill","type":"fill","source":"test","paint":{"fill-color":"#ff0000"}}]
        }"##,
            )
            .unwrap();
        let first = offscreen.frame();
        assert!(!first.idle);
        assert_eq!(first.tiles_drawn, 0);
        assert!(first.tiles_loading > 0);
        let waiting = offscreen.frame();
        assert_eq!(waiting.tiles_loading, first.tiles_loading);
        let jobs = std::mem::take(&mut *executor.0.lock().unwrap());
        assert_eq!(jobs.len(), first.tiles_loading as usize);
        for job in jobs.into_iter().rev() {
            job();
        }
        assert_eq!(wakes.load(Ordering::SeqCst), first.tiles_loading as usize);
        // Each frame applies at least one arrived tile (plan §7.4's budget).
        let published = (0..first.tiles_loading)
            .map(|_| offscreen.frame())
            .find(|report| report.idle)
            .expect("one frame per arrived tile at most");
        assert_eq!(published.tiles_loading, 0);
        assert!(published.tiles_drawn > 0);
        assert_eq!(offscreen.map.zoom(), camera.zoom(&super::web_mercator()));
        assert_eq!(offscreen.pixels(), {
            offscreen.frame();
            offscreen.pixels()
        });
        assert!(offscreen.map.take_warnings().is_empty());
    }

    /// M0 exit criterion 4: a camera-only frame is one dispatch and one
    /// bundle replay, records no commands, and touches only the placement
    /// rows of the tiles in view.
    #[test]
    fn panning_frame_by_frame_matches_a_fresh_view() {
        // Compacted indices rewrite only the slots of chunks that enter or
        // leave the view, and every slot when tiles arrive or leave.
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(12.0)).unwrap();
        let viewport = offscreen.viewport;
        // Around a circle of 128 px, as the bench's moving camera, stopping
        // six times to compare with a map loaded there.
        let point = |f: usize| {
            let angle = std::f64::consts::TAU * f as f64 / 120.0;
            glam::DVec2::new(512.0, 384.0)
                + 128.0 * glam::DVec2::new(angle.cos() - 1.0, angle.sin())
        };
        for stop in 0..6 {
            for f in stop * 20..(stop + 1) * 20 {
                pan(&mut offscreen.map, viewport, point(f), point(f + 1));
                offscreen.frame();
            }
            assert_matches_fresh(&mut offscreen);
        }
    }

    #[test]
    fn camera_only_frames_touch_only_placements() {
        for patterned in [false, true] {
            let options = super::Options {
                style: None,
                encoding: "mvt".into(),
                at: None,
                snapshot: None,
                a11y: None,
                watch_style: None,
                count_frames: None,
                flight: None,
            };
            let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
            if patterned {
                let sprite = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../web/demo/data/sprites/ofm");
                offscreen.map.set_style_json(&format!(r#"{{"version":8,"sprite":"file://{}","sources":{{}},"layers":[{{"id":"pattern","type":"background","paint":{{"background-pattern":"wetland"}}}}]}}"#, sprite.display())).unwrap();
                let start = std::time::Instant::now();
                while !offscreen.frame().idle {
                    assert!(start.elapsed().as_secs() < 10);
                    std::thread::yield_now();
                }
                offscreen.frame();
            }
            let viewport = offscreen.viewport;
            pan(
                &mut offscreen.map,
                viewport,
                glam::DVec2::new(512.0, 384.0),
                glam::DVec2::new(520.0, 380.0),
            );
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert!(report.tiles_drawn > 0);
            assert_eq!(report.frame.placements_written, report.tiles_drawn);
            assert_eq!(report.tiles_loading, 0);
        }
    }

    #[test]
    fn sky_keeps_camera_only_uploads_to_placements() {
        let (mut offscreen, _, _) = super::Offscreen::load(&tokyo(14.0)).unwrap();
        offscreen.map.set_style_json(r##"{"version":8,"sources":{},"sky":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#ccaa83"}}]}"##).unwrap();
        offscreen.frame();
        offscreen.frame();
        for pitch in [60.0, 80.0, 85.0] {
            offscreen.map.camera_mut().pitch = pitch;
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert_eq!(report.frame.placements_written, report.tiles_drawn);
            assert_eq!(report.frame.draws, 2);
            assert!(report.idle);
        }
    }

    #[test]
    fn extrusions_keep_camera_only_uploads_to_placements() {
        let options = super::Options {
            style: Some(format!(
                "file://{}/tests/data/extrusion.json",
                env!("CARGO_MANIFEST_DIR")
            )),
            encoding: "mvt".into(),
            at: Some((orb_geo::Geodetic::new(0.0, 0.0), 19.0, -20.0, 60.0)),
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        for (bearing, pitch) in [(31.0, 61.0), (10.0, 40.0)] {
            let camera = offscreen.map.camera_mut();
            camera.bearing = bearing;
            camera.pitch = pitch;
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert_eq!(report.frame.placements_written, report.tiles_drawn);
            assert!(report.tiles_drawn > 0);
        }
        // Switching 1×/4× rebuilds dependent targets and bundles once; the
        // next camera-only frame still replays them without paint writes.
        for enabled in [false, true] {
            offscreen.map.set_antialias_3d(enabled);
            let changed = offscreen.frame();
            assert!(changed.frame.bundle_recorded);
            let replay = offscreen.frame();
            assert!(!replay.frame.bundle_recorded);
            assert_eq!(replay.frame.paints_written, 0);
            assert_eq!(replay.frame.dispatches, 1);
        }
        assert!(
            !offscreen
                .map
                .take_warnings()
                .iter()
                .any(|w| w.contains("fill-extrusion"))
        );
    }

    /// A scene's lights (decision 0026) shade an extrusion's roof in the
    /// response v8's light uses, and emissive strength shows its colour.
    #[test]
    fn scene_lights_shade_extrusions() {
        let options = super::Options {
            style: Some(format!(
                "file://{}/tests/data/extrusion.json",
                env!("CARGO_MANIFEST_DIR")
            )),
            encoding: "mvt".into(),
            at: Some((orb_geo::Geodetic::new(-0.00015, -0.00025), 19.0, 0.0, 0.0)),
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let style = r##"{"version": 8,
            "sources": {"b": {"type": "geojson", "data": {"type": "Polygon",
                "coordinates": [[[-0.0003, -0.0004], [-0.0003, -0.0001], [0, -0.0001], [0, -0.0004], [-0.0003, -0.0004]]]}}},
            "layers": [
                {"id": "ground", "type": "background", "paint": {"background-color": "#000000"}},
                {"id": "block", "type": "fill-extrusion", "source": "b",
                 "paint": {"fill-extrusion-height": 10, "fill-extrusion-color": "#808080"}}]}"##;
        let mut roof = |direction: &str, emissive: f64| {
            let scene = format!(
                r#"{{"style": {style}, "lights": [
                    {{"type": "ambient", "properties": {{"intensity": 0.5}}}},
                    {{"type": "directional", "properties": {{"direction": {direction}, "intensity": 0.5}}}}],
                    "layers": {{"block": {{"emissive-strength": {emissive}}}}}}}"#
            );
            offscreen.map.set_style_json(&scene).unwrap();
            let projection = offscreen.map.projection();
            *offscreen.map.camera_mut() = orb_runtime::CameraState::new(
                orb_geo::Geodetic::new(-0.00015, -0.00025),
                19.0,
                &projection,
            );
            settle(&mut offscreen);
            let pixels = offscreen.pixels();
            let at = (super::Offscreen::HEIGHT as usize / 2 * super::Offscreen::WIDTH as usize
                + super::Offscreen::WIDTH as usize / 2)
                * 4;
            pixels[at]
        };
        // GL JS's flat response (decision 0026). Overhead the roof gets
        // max(0.5 + 0.5 − its luminance, 1) of its colour lifted by 0.03.
        assert!(roof("[0, 0]", 0.0).abs_diff(136) <= 1);
        // At the horizon only the ambient level, 0.5.
        assert!(roof("[90, 90]", 0.0).abs_diff(68) <= 1);
        // Fully emissive: its own colour.
        assert!(roof("[90, 90]", 1.0).abs_diff(128) <= 1);
    }

    #[test]
    fn point_and_line_families_keep_camera_only_uploads_to_placements() {
        for style in [
            r#"{
            "version":8,
            "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
            "layers":[{"id":"heat","type":"heatmap","source":"s"},{"id":"circle","type":"circle","source":"s"}]
        }"#,
            r#"{
            "version":8,
            "sources":{"s":{"type":"geojson","lineMetrics":true,"data":{"type":"LineString","coordinates":[[-30,0],[30,0]]}}},
            "layers":[
                {"id":"dash","type":"line","source":"s","paint":{"line-width":8,"line-dasharray":[2,1]}},
                {"id":"gradient","type":"line","source":"s","paint":{"line-width":5,"line-gradient":["interpolate",["linear"],["line-progress"],0,"red",1,"blue"],"line-dasharray":[1,1]}}
            ]
        }"#,
        ] {
            let options = super::Options {
                style: None,
                encoding: "mvt".into(),
                at: None,
                snapshot: None,
                a11y: None,
                watch_style: None,
                count_frames: None,
                flight: None,
            };
            let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
            let fetch = super::NativeFetch::new();
            let services = super::Services {
                clock: Arc::new(orb_host_native::NativeClock::default()),
                image_decode: Arc::new(super::NativeImageDecode),
                images: Default::default(),
                jobs: super::PoolJobs::new(
                    fetch.clone(),
                    Arc::new(super::NativeImageDecode),
                    Arc::new(orb_jobs::Inline),
                ),
                fetch,
                wake: Arc::new(|| {}),
            };
            let camera = super::CameraState::new(
                orb_geo::Geodetic::new(0.0, 0.0),
                1.5,
                &super::web_mercator(),
            );
            offscreen.map = super::Map::new(services, Some(camera));
            offscreen.map.set_style_json(style).unwrap();
            // Inline jobs publish their completion on the following frame.
            let mut loaded = false;
            for _ in 0..8 {
                let report = offscreen.frame();
                if report.idle && report.tiles_drawn > 0 {
                    loaded = true;
                    break;
                }
            }
            assert!(loaded);
            pan(
                &mut offscreen.map,
                offscreen.viewport,
                glam::DVec2::new(512.0, 384.0),
                glam::DVec2::new(520.0, 380.0),
            );
            let report = offscreen.frame();
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
            assert_eq!(report.frame.placements_written, report.tiles_drawn);
            assert_eq!(report.tiles_loading, 0);
            assert!(offscreen.map.take_warnings().is_empty());
        }
    }

    /// A style with a symbol layer loads, places and goes idle. Without a
    /// glyphs URL its text needs local fonts, which warn.
    /// Plan §13: a placement commit's semantic snapshot reads the labels it
    /// shows.
    #[test]
    fn semantic_snapshots_read_shown_labels() {
        let options = super::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let fetch = super::NativeFetch::new();
        let jobs = super::PoolJobs::new(
            fetch.clone(),
            Arc::new(super::NativeImageDecode),
            Arc::new(orb_jobs::Inline),
        );
        jobs.set_local_glyphs(orb_host_native::SystemFonts::shared());
        let services = super::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(super::NativeImageDecode),
            images: Default::default(),
            jobs,
            fetch,
            wake: Arc::new(|| {}),
        };
        let camera = super::CameraState::new(
            orb_geo::Geodetic::new(0.0, 0.0),
            1.5,
            &super::web_mercator(),
        );
        offscreen.map = super::Map::new(services, Some(camera));
        offscreen
            .map
            .set_style_json(
                r#"{"version":8,
                "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
                "layers":[{"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}]}"#,
            )
            .unwrap();
        assert!(
            offscreen.map.semantic_snapshot().is_none(),
            "nothing placed yet"
        );
        // The commit comes from the placement thread.
        let started = std::time::Instant::now();
        let mut snapshot = None;
        while snapshot.is_none() && started.elapsed() < std::time::Duration::from_secs(10) {
            let report = offscreen.frame();
            if report.idle && report.tiles_drawn > 0 {
                snapshot = offscreen.map.semantic_snapshot();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let snapshot = snapshot.expect("labels were placed");
        let node = &snapshot.nodes[snapshot.reading_order[0] as usize];
        assert_eq!(
            (node.text.as_str(), node.role, node.layer.as_str()),
            ("Null Island", orb_a11y::Role::Label, "label")
        );
        assert!(node.distance < 1.0, "{node:?}");
        let [x0, y0, x1, y1] = node.bounds;
        let center = (
            f64::from(super::Offscreen::WIDTH) / 2.0,
            f64::from(super::Offscreen::HEIGHT) / 2.0,
        );
        assert!(
            x0 < center.0 && center.0 < x1 && y0 < center.1 && center.1 < y1,
            "{node:?}"
        );
        assert_eq!(snapshot.summary.highlights, vec!["Null Island"]);
    }

    #[test]
    fn styles_with_symbol_layers_render() {
        let options = super::Options {
            style: None,
            encoding: "mvt".into(),
            at: None,
            snapshot: None,
            a11y: None,
            watch_style: None,
            count_frames: None,
            flight: None,
        };
        let (mut offscreen, _, _) = super::Offscreen::load(&options).unwrap();
        let fetch = super::NativeFetch::new();
        let services = super::Services {
            clock: Arc::new(orb_host_native::NativeClock::default()),
            image_decode: Arc::new(super::NativeImageDecode),
            images: Default::default(),
            jobs: super::PoolJobs::new(
                fetch.clone(),
                Arc::new(super::NativeImageDecode),
                Arc::new(orb_jobs::Inline),
            ),
            fetch,
            wake: Arc::new(|| {}),
        };
        let camera = super::CameraState::new(
            orb_geo::Geodetic::new(0.0, 0.0),
            1.5,
            &super::web_mercator(),
        );
        offscreen.map = super::Map::new(services, Some(camera));
        offscreen
            .map
            .set_style_json(
                r#"{"version":8,
                "sources":{"s":{"type":"geojson","data":{"type":"Point","coordinates":[0,0]}}},
                "layers":[
                    {"id":"circle","type":"circle","source":"s"},
                    {"id":"label","type":"symbol","source":"s","layout":{"text-field":"Null Island"}}
                ]}"#,
            )
            .unwrap();
        let mut loaded = false;
        for _ in 0..8 {
            let report = offscreen.frame();
            if report.idle && report.tiles_drawn > 0 {
                loaded = true;
                break;
            }
        }
        assert!(loaded);
        let warnings = offscreen.map.take_warnings();
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("glyphs that need local fonts")),
            "{warnings:?}"
        );
    }

    #[test]
    fn mvt_and_mlt_extract_pixels_match() {
        let mut images = Vec::new();
        for encoding in ["mvt", "mlt"] {
            let options = super::Options {
                style: None,
                encoding: encoding.into(),
                at: None,
                snapshot: None,
                a11y: None,
                watch_style: None,
                count_frames: None,
                flight: None,
            };
            let path =
                std::env::temp_dir().join(format!("orbis-{encoding}-{}.png", std::process::id()));
            images.push(super::snapshot(&options, path.to_str().unwrap()).unwrap());
            std::fs::remove_file(path).unwrap();
        }
        let differing = super::differing_pixels(&images[0], &images[1]);
        assert_eq!(differing, 0, "MVT/MLT differing pixels");
    }
}
