//! The Android app: one map in a `NativeActivity`, drawn through `orb-api`
//! with winit's Android backend, as `orb-demo` draws its window.
//! `cargo xtask android` builds it as `liborbis.so`; `src/android.rs` is
//! the entry point and lists the intent extras it takes. The map logs its
//! first load's milestones under the logcat tag `Orbis`: the first frame
//! that draws a layer, the last pipeline and the first idle frame, with
//! each pipeline's compile time. A tap logs the feature under it.
//!
//! The app itself builds on every platform, so the workspace's lints and
//! checks cover it; only the entry point is Android's.

#![deny(unsafe_code)]

#[cfg(target_os = "android")]
mod android;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec2;
use orb_api::{
    Button, CameraState, Error, Event, Fling, Geodetic, GestureOptions, Input, Map, MapEvent,
    MapOptions, Modifiers, Movement, PointerKind, Previews, Runtime, RuntimeOptions, Target,
    Thresholds, TileMatrixSet, WheelSource, WheelUnit,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

/// Writes a line where the platform shows the app's output: logcat (tag
/// `Orbis`) on Android, stderr elsewhere.
fn log(line: &str) {
    #[cfg(target_os = "android")]
    android::log(line);
    #[cfg(not(target_os = "android"))]
    eprintln!("orb-android: {line}");
}

/// Data arrived; draw a frame.
#[derive(Debug)]
pub enum Wake {
    Data,
}

/// What the app shows and does.
#[derive(Debug, Default)]
pub struct Options {
    /// The style's URL.
    pub style: String,
    /// Target, zoom, bearing and pitch; the style's camera when `None`.
    pub at: Option<(Geodetic, f64, f64, f64)>,
    /// The runtime's state directory.
    pub cache: Option<PathBuf>,
    /// Lose the GPU this long after the map first goes idle, to show
    /// recovery.
    pub lose: Option<Duration>,
    /// First make, draw and drop 20 maps of this style, logging the live
    /// GPU resources after each: they should stay flat.
    pub leakcheck: Option<String>,
}

impl Options {
    /// The camera from `at`. Its zoom is on the default Web Mercator
    /// surface: a style's own plane is known only once it loads.
    fn camera(&self) -> Option<CameraState> {
        self.at.map(|(target, zoom, bearing, pitch)| CameraState {
            bearing,
            pitch,
            ..CameraState::new(target, zoom, &mercator())
        })
    }
}

/// Android's gesture thresholds (`ViewConfiguration`), in physical pixels
/// and milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct ViewConfiguration {
    pub touch_slop: f64,
    pub double_tap_slop: f64,
    /// Pixels per second.
    pub min_fling: f64,
    pub max_fling: f64,
    pub double_tap_timeout: f64,
    pub long_press_timeout: f64,
}

/// The gestures as Android's own feel: pans fling along its
/// `OverScroller` (Compose's Android default), and taps, slop and long
/// presses keep the platform's thresholds, in logical pixels at `scale`
/// device pixels each.
fn android_gestures(config: Option<ViewConfiguration>, scale: f64) -> GestureOptions {
    let defaults = GestureOptions::default();
    let Some(c) = config else {
        return GestureOptions {
            pan: Movement {
                momentum: Some(Fling::ANDROID),
                ..defaults.pan
            },
            ..defaults
        };
    };
    GestureOptions {
        pan: Movement {
            momentum: Some(Fling::Android {
                min_speed: c.min_fling / scale,
                max_speed: c.max_fling / scale,
                max_distance: f64::INFINITY,
            }),
            ..defaults.pan
        },
        thresholds: Thresholds {
            click_tolerance: c.touch_slop / scale,
            tap_tolerance: c.double_tap_slop / scale,
            tap_interval: c.double_tap_timeout,
            long_press: c.long_press_timeout,
        },
        ..defaults
    }
}

fn mercator() -> orb_api::Projection {
    *TileMatrixSet::web_mercator().projection()
}

/// Runs the app on `event_loop` until the activity finishes. `started` is
/// when the process started, for the load timings.
pub fn run(event_loop: EventLoop<Wake>, options: Options, started: Instant) {
    let mut app = App {
        options,
        proxy: event_loop.create_proxy(),
        state: None,
        started,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

struct App {
    options: Options,
    proxy: EventLoopProxy<Wake>,
    state: Option<State>,
    started: Instant,
}

/// What has been logged of the first load, and the GPU loss to recover from.
#[derive(Default)]
struct Load {
    first_draw: bool,
    compiled: bool,
    idle: bool,
    /// When the GPU is to be lost, then when it was until a complete frame.
    lose_at: Option<Instant>,
    lost_at: Option<Instant>,
}

struct State {
    window: Arc<Window>,
    runtime: Runtime,
    /// The window's surface; made again when it's lost, and dropped while
    /// the app is suspended (Android takes the window away).
    target: Option<Target>,
    map: Map,
    /// When the process started, and the clock frames and input share.
    started: Instant,
    /// How long the device took to make, from `started`.
    device: Duration,
    frames: u32,
    load: Load,
    cursor: DVec2,
}

impl State {
    fn new(
        event_loop: &ActiveEventLoop,
        options: &Options,
        proxy: EventLoopProxy<Wake>,
        started: Instant,
    ) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Orbis"))
                .expect("window"),
        );
        let runtime = Runtime::new(&RuntimeOptions {
            state_dir: options.cache.clone(),
            ..RuntimeOptions::default()
        })
        .expect("runtime");
        log(&runtime.stats().adapter);
        if let Some(style) = &options.leakcheck {
            leakcheck(&runtime, style);
        }
        let device = started.elapsed();
        let mut map = Map::new(
            &runtime,
            MapOptions {
                camera: options.camera(),
                // The opaque window has no page behind transparent pixels.
                clear_color: [1.0; 4],
                previews: Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_wake(move || {
            let _ = proxy.send_event(Wake::Data);
        });
        #[cfg(target_os = "android")]
        let config = android::view_configuration();
        #[cfg(not(target_os = "android"))]
        let config = None;
        let gestures = android_gestures(config, window.scale_factor());
        log(&format!("gestures: {:?}", gestures.thresholds));
        map.set_gestures(gestures);
        map.load_style_url(&options.style);
        let mut state = Self {
            window,
            runtime,
            target: None,
            map,
            started,
            device,
            frames: 0,
            load: Load::default(),
            cursor: DVec2::ZERO,
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
        self.window.request_redraw();
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

    /// Milliseconds since the process started.
    fn since_start(&self) -> u128 {
        self.started.elapsed().as_millis()
    }

    fn input(&mut self, input: Input) {
        let now = self.now();
        if self.map.input(input, now) {
            self.window.request_redraw();
        }
    }

    fn events(&mut self) {
        while let Some(event) = self.map.poll_event() {
            match event {
                Event::Map(
                    ref event @ (MapEvent::Click { position } | MapEvent::LongClick { position }),
                ) => match self.map.hit_test(position) {
                    Some(hit) => log(&format!(
                        "{} {} ({}{}) id {}",
                        if matches!(event, MapEvent::Click { .. }) {
                            "clicked"
                        } else {
                            "long-clicked"
                        },
                        hit.layer,
                        hit.source,
                        hit.source_layer
                            .map(|l| format!(" / {l}"))
                            .unwrap_or_default(),
                        hit.id.as_deref().unwrap_or("none")
                    )),
                    None => log(if matches!(event, MapEvent::Click { .. }) {
                        "clicked nothing"
                    } else {
                        "long-clicked nothing"
                    }),
                },
                Event::Map(MapEvent::Error { message } | MapEvent::Warning { message }) => {
                    log(&message);
                }
                _ => {}
            }
        }
    }

    /// Logs the first load's milestones: the first frame that draws a
    /// layer, the last pipeline compiled and the first idle frame; then
    /// schedules the GPU loss if asked, and logs the recovery.
    fn log_load(&mut self, report: &orb_api::FrameReport, lose: Option<Duration>) {
        let (tiles, draws) = (report.tiles_drawn, report.frame.draws);
        let pending = report.frame.pipelines_pending;
        if !self.load.first_draw && tiles > 0 && draws > 0 {
            self.load.first_draw = true;
            log(&format!(
                "first draw {} ms after start; {pending} pipelines compiling, {} layers waiting",
                self.since_start(),
                report.frame.nodes_waiting,
            ));
        }
        if !self.load.compiled && tiles > 0 && pending == 0 {
            self.load.compiled = true;
            log(&format!(
                "pipelines compiled {} ms after start",
                self.since_start()
            ));
        }
        if let Some(at) = self.load.lost_at
            && report.idle
        {
            self.load.lost_at = None;
            log(&format!(
                "recovered from device loss: complete frame {} ms after it",
                at.elapsed().as_millis()
            ));
        }
        if !report.idle || self.load.idle {
            return;
        }
        self.load.idle = true;
        self.load.lose_at = lose.map(|after| Instant::now() + after);
        let stats = self.runtime.stats();
        log(&format!(
            "loaded in {} ms (device {} ms) after {} frames; {} pipelines in {:.0} ms; {tiles} tiles, {draws} draws",
            self.since_start(),
            self.device.as_millis(),
            self.frames,
            stats.pipelines_created,
            stats.pipeline_time.as_secs_f64() * 1000.0,
        ));
        let mut timings = stats.pipeline_timings;
        timings.sort_by_key(|t| std::cmp::Reverse(t.1));
        for (label, took) in timings {
            log(&format!(
                "  pipeline {label} {:.1} ms",
                took.as_secs_f64() * 1000.0
            ));
        }
    }

    /// Loses the GPU once its time comes.
    fn lose(&mut self) {
        if let Some(at) = self.load.lose_at
            && Instant::now() >= at
        {
            self.load.lose_at = None;
            self.load.lost_at = Some(Instant::now());
            self.runtime.lose_device();
            self.window.request_redraw();
        }
    }

    fn redraw(&mut self, lose: Option<Duration>) {
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
                return;
            }
            Err(e) => panic!("{e}"),
        };
        self.events();
        let Some(report) = report else {
            self.window.request_redraw();
            return;
        };
        self.frames += 1;
        self.log_load(&report, lose);
        if self.map.needs_frame() {
            self.window.request_redraw();
        }
    }
}

/// Makes, draws and drops 20 maps of `style` one after another, and logs
/// the device's live GPU resources after each.
fn leakcheck(runtime: &Runtime, style: &str) {
    let live: Vec<usize> = (0..20)
        .map(|_| {
            let mut map = Map::new(
                runtime,
                MapOptions {
                    camera: Some(CameraState::new(
                        Geodetic::new(139.74, 35.685),
                        15.0,
                        &mercator(),
                    )),
                    ..MapOptions::default()
                },
            );
            map.load_style_url(style);
            if let Err(e) = map.snapshot(256, 256, 1.0) {
                log(&format!("leakcheck: {e}"));
            }
            drop(map);
            runtime.stats().live_resources
        })
        .collect();
    log(&format!(
        "leakcheck: live resources after each map {live:?}"
    ));
}

impl ApplicationHandler<Wake> for App {
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &mut self.state else {
            return;
        };
        state.lose();
        if let Some(at) = state.load.lose_at {
            event_loop.set_control_flow(ControlFlow::WaitUntil(at));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match &mut self.state {
            None => {
                self.state = Some(State::new(
                    event_loop,
                    &self.options,
                    self.proxy.clone(),
                    self.started,
                ))
            }
            Some(state) if state.target.is_none() => state.make_target(),
            Some(_) => {}
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.target = None;
        }
        // An activity going for good (back) ends the loop, and the app's
        // process with it: winit runs one event loop a process and doesn't
        // end it when the activity is destroyed.
        #[cfg(target_os = "android")]
        if android::finishing() {
            event_loop.exit();
        }
        #[cfg(not(target_os = "android"))]
        let _ = event_loop;
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
            WindowEvent::RedrawRequested => state.redraw(self.options.lose),
            // A mouse, on ChromeOS or a desktop-mode phone.
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
                        modifiers: Modifiers::default(),
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
                // Wheels scroll by lines, trackpads by pixels. (winit's
                // Android backend reports every pointer as touches.)
                let (delta, unit, source) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (
                        -DVec2::new(f64::from(x), f64::from(y)),
                        WheelUnit::Line,
                        WheelSource::Wheel,
                    ),
                    MouseScrollDelta::PixelDelta(p) => (
                        -DVec2::new(p.x, p.y) / scale,
                        WheelUnit::Pixel,
                        WheelSource::Trackpad,
                    ),
                };
                let position = state.cursor;
                state.input(Input::Wheel {
                    position,
                    delta,
                    unit,
                    modifiers: Modifiers::default(),
                    source,
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
                        modifiers: Modifiers::default(),
                    },
                    TouchPhase::Moved => Input::PointerMove { id, position },
                    TouchPhase::Ended => Input::PointerUp { id, position },
                    TouchPhase::Cancelled => Input::PointerCancel { id },
                });
            }
            _ => {}
        }
    }
}
