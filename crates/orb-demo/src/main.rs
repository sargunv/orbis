//! The native demo: a v8 style in a window, through `orb-api` as any host
//! draws it, moved with mouse, trackpad, touch and keyboard. A click prints
//! the features under it.
//!
//! `cargo run --release -p orb-demo -- [style URL] [--at lon,lat,zoom[,bearing,pitch]]
//! [--format mvt|mlt] [--flight flight.json] [--cache dir] [--count-frames seconds]`
//!
//! Frame counters are shown in the window title. `--flight` flies a loop
//! of `flyTo` legs until the user takes the camera (the terrain demo:
//! `file://$PWD/web/demo/data/alps.json --flight
//! web/demo/data/alps-flight.json`). `--cache <dir>` keeps state there
//! between runs: the HTTP cache and the driver's pipelines.
//! `--count-frames <seconds>` prints how many frames the window drew in
//! that long after the map first went idle, then exits. Still pictures are
//! `orbis render` (`orb-cli`).

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Instant;

use glam::DVec2;
use orb_api::{
    Button, CameraState, CameraTarget, Error, Event, FlyOptions, Geodetic, Input, Key, Map,
    MapEvent, MapOptions, Modifiers, PointerKind, Previews, Runtime, RuntimeOptions, Target,
    TileMatrixSet, WheelUnit,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::{Window, WindowId};

const DEMO_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data");

/// Data arrived; draw a frame.
#[derive(Debug)]
enum Wake {
    Data,
}

struct Options {
    style: Option<String>,
    encoding: String,
    /// Target, zoom, bearing and pitch.
    at: Option<(Geodetic, f64, f64, f64)>,
    /// Count the frames drawn this many seconds after the map first goes
    /// idle, print them and exit: idle means zero frames.
    count_frames: Option<f64>,
    /// A flight file to fly in a loop (`web/demo/data/alps-flight.json`).
    flight: Option<String>,
    /// The runtime's state directory.
    cache: Option<String>,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        style: None,
        encoding: "mvt".into(),
        at: None,
        count_frames: None,
        flight: None,
        cache: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--at" => {
                let parts: Vec<f64> = value()?
                    .split(',')
                    .map(|p| p.trim().parse::<f64>().map_err(|e| format!("--at: {e}")))
                    .collect::<Result<_, _>>()?;
                let (lon, lat, zoom, bearing, pitch) = match parts[..] {
                    [lon, lat, zoom] => (lon, lat, zoom, 0.0, 0.0),
                    [lon, lat, zoom, bearing, pitch] => (lon, lat, zoom, bearing, pitch),
                    _ => return Err("--at needs lon,lat,zoom[,bearing,pitch]".into()),
                };
                options.at = Some((Geodetic::new(lon, lat), zoom, bearing, pitch));
            }
            "--count-frames" => {
                let seconds = value()?;
                options.count_frames = Some(
                    seconds
                        .parse()
                        .map_err(|e| format!("--count-frames: {e}"))?,
                );
            }
            "--flight" => options.flight = Some(value()?),
            "--cache" => options.cache = Some(value()?),
            "--format" => {
                options.encoding = value()?;
                if !matches!(options.encoding.as_str(), "mvt" | "mlt") {
                    return Err("--format needs mvt or mlt".into());
                }
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => options.style = Some(arg),
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
    let event_loop = EventLoop::<Wake>::with_user_event()
        .build()
        .expect("event loop");
    let mut app = App {
        options,
        proxy: event_loop.create_proxy(),
        state: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// The style's URL: the committed extract and reduced Liberty style, or
/// the one given.
fn style_url(options: &Options) -> Result<String, String> {
    if let Some(url) = &options.style {
        return Ok(url.clone());
    }
    let path = std::path::Path::new(DEMO_DATA)
        .join(format!("tokyo-{}.json", options.encoding))
        .canonicalize()
        .map_err(|e| e.to_string())?;
    Ok(format!("file://{}", path.display()))
}

/// The camera from `--at`, or `None` to use the style's. Its zoom is on
/// the default Web Mercator surface: a style's own plane is known only
/// once it loads.
fn camera(options: &Options) -> Option<CameraState> {
    let mercator = *TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built in")
        .projection();
    options
        .at
        .map(|(target, zoom, bearing, pitch)| CameraState {
            bearing,
            pitch,
            ..CameraState::new(target, zoom, &mercator)
        })
}

/// A loop of camera flights: legs of `flyTo` options (center, zoom,
/// bearing, pitch and duration in milliseconds), as the web demo flies them.
struct Flight {
    legs: Vec<(CameraTarget, FlyOptions)>,
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
                let target = CameraTarget {
                    center,
                    zoom: number("zoom"),
                    bearing: number("bearing"),
                    pitch: number("pitch"),
                    ..Default::default()
                };
                let fly = FlyOptions {
                    duration: number("duration"),
                    ..Default::default()
                };
                (target, fly)
            })
            .collect::<Vec<_>>();
        if legs.is_empty() {
            return Err(format!("{path}: no legs"));
        }
        Ok(Self { legs, next: 0 })
    }
}

struct App {
    options: Options,
    proxy: EventLoopProxy<Wake>,
    state: Option<State>,
}

struct State {
    window: Arc<Window>,
    runtime: Runtime,
    /// The window's surface; made again when it's lost.
    target: Option<Target>,
    map: Map,
    started: Instant,
    /// Frames drawn, and when the map first went idle with the count then.
    frames: u32,
    idle_at: Option<(Instant, u32)>,
    cursor: DVec2,
    modifiers: Modifiers,
    last_title: Instant,
    /// Flying until the user takes the camera.
    flight: Option<Flight>,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, options: &Options, proxy: EventLoopProxy<Wake>) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Orbis"))
                .expect("window"),
        );
        let runtime = Runtime::new(&RuntimeOptions {
            state_dir: options.cache.as_ref().map(Into::into),
            ..RuntimeOptions::default()
        })
        .expect("runtime");
        eprintln!("orb-demo: {}", runtime.stats().adapter);
        let mut map = Map::new(
            &runtime,
            MapOptions {
                camera: camera(options),
                // The opaque window has no page behind transparent pixels.
                clear_color: [1.0; 4],
                // The demo shows what Orbis can do beyond the spec too.
                previews: Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_wake(move || {
            let _ = proxy.send_event(Wake::Data);
        });
        map.load_style_url(&style_url(options).expect("demo style"));
        let mut state = Self {
            window,
            runtime,
            target: None,
            map,
            started: Instant::now(),
            frames: 0,
            idle_at: None,
            cursor: DVec2::ZERO,
            modifiers: Modifiers::default(),
            last_title: Instant::now(),
            flight: options
                .flight
                .as_deref()
                .map(|path| Flight::read(path).expect("flight file")),
        };
        state.make_target();
        state
    }

    /// The window's surface, made with the runtime's device.
    fn make_target(&mut self) {
        let size = self.window.inner_size();
        let surface = self
            .runtime
            .create_surface(
                Arc::clone(&self.window),
                size.width.max(1),
                size.height.max(1),
            )
            .expect("surface");
        self.target = Some(Target::surface(
            &self.runtime,
            surface,
            self.window.scale_factor(),
        ));
    }

    fn resize(&mut self) {
        let size = self.window.inner_size();
        let scale = self.window.scale_factor();
        if let Some(target) = &mut self.target {
            target.resize(size.width.max(1), size.height.max(1), scale);
        }
        self.window.request_redraw();
    }

    /// Milliseconds on the clock frames and input share.
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn input(&mut self, input: Input) {
        // The user takes the camera from a flight.
        self.flight = None;
        let now = self.now();
        if self.map.input(input, now) {
            self.window.request_redraw();
        }
    }

    /// Starts the flight's next leg when the last one ended.
    fn fly(&mut self) {
        if let Some(flight) = &mut self.flight
            && !self.map.is_moving()
        {
            let (target, options) = &flight.legs[flight.next];
            self.map.fly_to(target, options);
            flight.next = (flight.next + 1) % flight.legs.len();
        }
    }

    fn events(&mut self) {
        while let Some(event) = self.map.poll_event() {
            match event {
                Event::Map(MapEvent::Click {
                    position,
                    button: Button::Primary,
                }) => match self.map.hit_test(position) {
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
                },
                Event::Map(MapEvent::Error { message } | MapEvent::Warning { message }) => {
                    eprintln!("orb-demo: {message}");
                }
                _ => {}
            }
        }
    }

    fn redraw(&mut self) {
        let started = Instant::now();
        self.fly();
        let now = self.now();
        let Some(target) = &mut self.target else {
            return;
        };
        let report = match self.map.render(target, now) {
            Ok(report) => report,
            // A new GPU, or the surface went: the old surface goes first.
            Err(Error::TargetLost(_)) => {
                self.target = None;
                self.make_target();
                self.window.request_redraw();
                return;
            }
            Err(e) => panic!("orb-demo: {e}"),
        };
        self.events();
        let Some(report) = report else {
            self.window.request_redraw();
            return;
        };
        self.frames += 1;
        if report.idle && self.idle_at.is_none() {
            self.idle_at = Some((Instant::now(), self.frames));
        }
        if self.map.needs_frame() || self.flight.is_some() {
            self.window.request_redraw();
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
        let Wake::Data = event;
        if let Some(state) = &mut self.state {
            state.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        let scale = state.window.scale_factor();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => state.resize(),
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
                    ElementState::Pressed => Input::PointerDown {
                        id: 0,
                        kind: PointerKind::Mouse,
                        button,
                        position,
                        modifiers: state.modifiers,
                    },
                    ElementState::Released => Input::PointerUp { id: 0, position },
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = DVec2::new(position.x, position.y) / scale;
                state.cursor = position;
                state.input(Input::PointerMove { id: 0, position });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // winit's deltas point away from the user; the DOM's, which
                // the gestures follow, toward.
                let (delta, unit) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        (-DVec2::new(f64::from(x), f64::from(y)), WheelUnit::Line)
                    }
                    MouseScrollDelta::PixelDelta(p) => {
                        (-DVec2::new(p.x, p.y) / scale, WheelUnit::Pixel)
                    }
                };
                let (position, modifiers) = (state.cursor, state.modifiers);
                state.input(Input::Wheel {
                    position,
                    delta,
                    unit,
                    modifiers,
                });
            }
            WindowEvent::Touch(touch) => {
                // Touch ids above the mouse's.
                let id = (touch.id as u32).wrapping_add(1);
                let position = DVec2::new(touch.location.x, touch.location.y) / scale;
                state.input(match touch.phase {
                    TouchPhase::Started => Input::PointerDown {
                        id,
                        kind: PointerKind::Touch,
                        button: Button::Primary,
                        position,
                        modifiers: state.modifiers,
                    },
                    TouchPhase::Moved => Input::PointerMove { id, position },
                    TouchPhase::Ended => Input::PointerUp { id, position },
                    TouchPhase::Cancelled => Input::PointerCancel { id },
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
                    WinitKey::Named(NamedKey::Escape) => Key::Escape,
                    _ => return,
                };
                let modifiers = state.modifiers;
                state.input(Input::Key { key, modifiers });
            }
            WindowEvent::Focused(false) => state.input(Input::Blur),
            _ => {}
        }
    }
}
