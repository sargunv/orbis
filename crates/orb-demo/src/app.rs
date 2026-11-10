//! The demo window: the map full-bleed in a texture target of the
//! runtime's device, which egui composites under the web demo's controls
//! (the map picker, the map options, zoom and the compass) on that same
//! device.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, Frame, Id, Margin, Order, Pos2, Rect, Response,
    RichText, Sense, Shadow, Shape, Stroke, StrokeKind, Ui, Vec2, pos2, vec2,
};
use glam::DVec2;
use orb_api::{
    Button, CameraTarget, EaseOptions, Error, Event, GesturePhase, Input, Key, Map, MapEvent,
    MapOptions, Modifiers, PointerKind, Previews, Runtime, Target, Transaction, WheelSource,
    WheelUnit,
};
use serde_json::{Value, json};

use crate::host::Trackpad;
use crate::options::{Context, Kind, Opt, Panel};
use crate::scenes::{Catalogue, Flight, Scene};

/// What the command line asks for.
pub(crate) struct Launch {
    pub(crate) scene: Scene,
    pub(crate) camera: Option<orb_api::CameraState>,
    /// Fly the scene's looping flight until the user takes the camera.
    pub(crate) flight: bool,
    /// Count the frames drawn this many seconds after the map first goes
    /// idle, print them and exit: idle means zero frames.
    pub(crate) count_frames: Option<f64>,
}

pub(crate) struct Demo {
    catalogue: Catalogue,
    runtime: Runtime,
    map: Map,
    target: Target,
    /// The target's texture as egui knows it, and the wgpu texture behind it.
    texture: Option<(egui::TextureId, wgpu::Texture)>,
    started: Instant,
    scene: Scene,
    /// A style asked for and not yet in.
    loading: bool,
    /// The style as it loaded, before the scene's changes.
    loaded: Option<Value>,
    /// The style as the demo last set it.
    style: Value,
    panel: Panel,
    attribution: String,
    toast: Option<(String, bool)>,
    scenes_open: bool,
    options_open: bool,
    input: InputState,
    flight: Option<Flight>,
    fly: bool,
    frames: u32,
    /// Whether the map stopped drawing (the runtime stopped replacing lost
    /// GPUs, say), once it has.
    stopped: bool,
    idle_at: Option<(Instant, u32)>,
    count_frames: Option<f64>,
}

#[derive(Default)]
struct InputState {
    /// Mouse buttons pressed on the map, whose drags it keeps.
    owned: BTreeSet<u8>,
    /// Touches down: egui also reports the first as a mouse, which the
    /// map mustn't see twice.
    touches: usize,
    modifiers: Modifiers,
    cursor: DVec2,
    /// Trackpad gestures not yet handed over.
    trackpad: Vec<Trackpad>,
    /// The trackpad's pinch and rotation streams under way: the map sees
    /// one pinch from the first's start to the last's end.
    pinching: bool,
    rotating: bool,
}

impl Demo {
    pub(crate) fn new(ctx: &egui::Context, runtime: Runtime, launch: Launch) -> Self {
        let catalogue = Catalogue::read();
        let camera = launch
            .camera
            .or_else(|| catalogue.opening_camera(&launch.scene).map(|c| c.state()));
        let mut map = Map::new(
            &runtime,
            MapOptions {
                camera,
                // The opaque window has no page behind transparent pixels.
                clear_color: [1.0; 4],
                // The demo shows what Orbis can do beyond the spec too.
                previews: Previews::ALL,
                ..MapOptions::default()
            },
        );
        let wake = ctx.clone();
        map.set_wake(move || wake.request_repaint());
        ctx.options_mut(|o| o.zoom_with_keyboard = false);
        let target = Target::texture(&runtime, 1, 1, 1.0);
        let mut demo = Self {
            catalogue,
            runtime,
            map,
            target,
            texture: None,
            started: Instant::now(),
            scene: launch.scene.clone(),
            loading: false,
            loaded: None,
            style: Value::Null,
            panel: Panel::default(),
            attribution: String::new(),
            toast: None,
            scenes_open: false,
            options_open: false,
            input: InputState::default(),
            flight: None,
            fly: launch.flight,
            frames: 0,
            stopped: false,
            idle_at: None,
            count_frames: launch.count_frames,
        };
        demo.load(launch.scene);
        demo
    }

    /// Milliseconds on the clock frames and input share.
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn say(&mut self, text: impl Into<String>, error: bool) {
        self.toast = Some((text.into(), error));
    }

    /// Shows `scene`, keeping the camera, though its constraints clamp it
    /// where they must. The same style with another projection or sky is
    /// edited in place; another style loads.
    fn go(&mut self, scene: Scene) {
        let same = scene.style_url() == self.scene.style_url()
            && scene.restyle.is_none()
            && self.scene.restyle.is_none()
            && !self.loading
            && self.loaded.is_some();
        self.map.stop();
        self.map.set_constraints(scene.constraints());
        if same {
            self.scene = scene;
            self.install();
            self.start_flight();
        } else {
            self.load(scene);
        }
    }

    fn load(&mut self, scene: Scene) {
        self.map.set_constraints(scene.constraints());
        self.say(format!("Loading {}…", scene.name), false);
        self.loading = true;
        self.loaded = None;
        self.panel.loading();
        self.flight = None;
        self.map.load_style_url(&scene.style_url());
        self.scene = scene;
    }

    /// The scene's style is in: puts in what it designs (restyle,
    /// projection, sky), then the options' defaults, as one edit.
    fn install(&mut self) {
        let Some(loaded) = &self.loaded else {
            return;
        };
        let designed = self.scene.designed(loaded, false);
        let terrain = match &self.scene.terrain {
            Some(Value::Bool(true)) => loaded.get("terrain").cloned(),
            Some(terrain @ Value::Object(_)) => Some(terrain.clone()),
            _ => None,
        };
        // A projection the style had, a scene without one sets back.
        let mut style = designed.clone();
        if self.scene.projection.is_none() && loaded.get("projection").is_none() {
            style["projection"] = self.scene.projection_or_mercator();
        }
        self.attribution = attribution(&designed);
        self.panel.attach(
            Context {
                scene: self.scene.clone(),
                designed,
                terrain,
            },
            &mut style,
        );
        self.style = style;
        self.set_style();
    }

    /// Hands the map the demo's style, as the diff from its own.
    fn set_style(&mut self) {
        let commands = json!([["setStyle", self.style]]).to_string();
        let result = Transaction::from_commands(&commands)
            .map_err(|e| e.to_string())
            .and_then(|tx| self.map.edit(&tx).map_err(|e| e.to_string()));
        if let Err(e) = result {
            self.say(format!("The style edit failed: {e}"), true);
        }
    }

    fn start_flight(&mut self) {
        self.flight = None;
        if !self.fly {
            return;
        }
        match self.scene.read_flight() {
            Some(Ok(flight)) => self.flight = Some(flight),
            Some(Err(e)) => eprintln!("orb-demo: {e}"),
            None => {}
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
            let Event::Map(event) = event else { continue };
            match event {
                MapEvent::StyleLoad if self.loading => {
                    self.loading = false;
                    self.loaded = self
                        .map
                        .style_json()
                        .and_then(|s| serde_json::from_str(&s).ok());
                    self.install();
                    self.start_flight();
                }
                MapEvent::Load | MapEvent::Idle if !self.loading => {
                    if self.toast.as_ref().is_some_and(|(_, error)| !error) {
                        self.toast = None;
                    }
                }
                MapEvent::Click { position } | MapEvent::LongClick { position } => {
                    let verb = if matches!(event, MapEvent::Click { .. }) {
                        "clicked"
                    } else {
                        "long-clicked"
                    };
                    match self.map.hit_test(position) {
                        Some(hit) => println!(
                            "{verb} {} ({}{}) id {}",
                            hit.layer,
                            hit.source,
                            hit.source_layer
                                .map(|l| format!(" / {l}"))
                                .unwrap_or_default(),
                            hit.id.as_deref().unwrap_or("none")
                        ),
                        None => println!("{verb} nothing"),
                    }
                }
                MapEvent::Error { message } => {
                    eprintln!("orb-demo: {message}");
                    if self.loading {
                        self.say(message, true);
                    }
                }
                MapEvent::Warning { message } => eprintln!("orb-demo: {message}"),
                _ => {}
            }
        }
    }

    fn map_input(&mut self, input: Input) {
        let now = self.now();
        self.map_input_at(input, now);
    }

    fn map_input_at(&mut self, input: Input, now: f64) {
        // The user takes the camera from a flight.
        self.flight = None;
        self.map.input(input, now);
    }

    /// Trackpad gestures for the next frame to hand over.
    pub(crate) fn trackpad(&mut self, gestures: Vec<Trackpad>) {
        self.input.trackpad.extend(gestures);
    }

    /// Hands the map the trackpad's pinches and rotations as one pinch,
    /// at their own times.
    fn forward_trackpad(&mut self, over_ui: impl Fn(Pos2) -> bool) {
        for gesture in std::mem::take(&mut self.input.trackpad) {
            let at = gesture
                .at
                .saturating_duration_since(self.started)
                .as_secs_f64()
                * 1000.0;
            let center = self.input.cursor;
            let active = self.input.pinching || self.input.rotating;
            let stream = if gesture.rotating {
                &mut self.input.rotating
            } else {
                &mut self.input.pinching
            };
            let phase = match gesture.phase {
                winit::event::TouchPhase::Started => {
                    // The second stream joins the first's pinch.
                    if active {
                        *stream = true;
                        continue;
                    }
                    if over_ui(pos2(center.x as f32, center.y as f32)) {
                        continue;
                    }
                    *stream = true;
                    GesturePhase::Begin
                }
                winit::event::TouchPhase::Moved if *stream => GesturePhase::Update,
                winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled
                    if *stream =>
                {
                    *stream = false;
                    if self.input.pinching || self.input.rotating {
                        continue;
                    }
                    if gesture.phase == winit::event::TouchPhase::Ended {
                        GesturePhase::End
                    } else {
                        GesturePhase::Cancel
                    }
                }
                _ => continue,
            };
            let input = Input::Pinch {
                center,
                scale: gesture.scale,
                rotation: gesture.rotation,
                phase,
            };
            self.map_input_at(input, at);
        }
    }

    /// Hands the map the window's input that egui's controls don't take.
    fn forward_input(&mut self, ctx: &egui::Context) {
        let over_ui = |pos: Pos2| {
            ctx.layer_id_at(pos)
                .is_some_and(|l| l.order != Order::Background)
        };
        let keyboard = !ctx.egui_wants_keyboard_input();
        self.forward_trackpad(over_ui);
        let events = ctx.input(|i| i.events.clone());
        for event in events {
            match event {
                egui::Event::ModifiersChanged(m) => {
                    self.input.modifiers = Modifiers {
                        shift: m.shift,
                        ctrl: m.ctrl,
                        alt: m.alt,
                        meta: m.mac_cmd,
                    };
                }
                egui::Event::PointerMoved(pos) => {
                    let position = point(pos);
                    self.input.cursor = position;
                    if self.input.touches == 0 && (!self.input.owned.is_empty() || !over_ui(pos)) {
                        self.map_input(Input::PointerMove { id: 0, position });
                    }
                }
                egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    ..
                } if self.input.touches == 0 => {
                    let (code, button) = match button {
                        egui::PointerButton::Primary => (0, Button::Primary),
                        egui::PointerButton::Secondary => (1, Button::Secondary),
                        egui::PointerButton::Middle => (2, Button::Middle),
                        _ => continue,
                    };
                    let position = point(pos);
                    if pressed && !over_ui(pos) {
                        // A press on the map closes the sheets, as a popover does.
                        self.scenes_open = false;
                        self.options_open = false;
                        self.input.owned.insert(code);
                        let modifiers = self.input.modifiers;
                        self.map_input(Input::PointerDown {
                            id: 0,
                            kind: PointerKind::Mouse,
                            button,
                            position,
                            modifiers,
                        });
                    } else if !pressed && self.input.owned.remove(&code) {
                        self.map_input(Input::PointerUp { id: 0, position });
                    }
                }
                egui::Event::MouseWheel {
                    unit,
                    delta,
                    source,
                    ..
                } => {
                    let pos = pos2(self.input.cursor.x as f32, self.input.cursor.y as f32);
                    if over_ui(pos) {
                        continue;
                    }
                    // egui's deltas, as winit's, point away from the user;
                    // the DOM's, which the gestures follow, toward.
                    let delta = -DVec2::new(f64::from(delta.x), f64::from(delta.y));
                    let unit = match unit {
                        egui::MouseWheelUnit::Point => WheelUnit::Pixel,
                        egui::MouseWheelUnit::Line => WheelUnit::Line,
                        egui::MouseWheelUnit::Page => WheelUnit::Page,
                    };
                    // egui tells trackpads, and their momentum, from
                    // wheels on macOS.
                    let source = match source {
                        egui::MouseWheelSource::Wheel => WheelSource::Wheel,
                        egui::MouseWheelSource::Trackpad | egui::MouseWheelSource::Momentum => {
                            WheelSource::Trackpad
                        }
                        _ => WheelSource::Unknown,
                    };
                    let (position, modifiers) = (self.input.cursor, self.input.modifiers);
                    self.map_input(Input::Wheel {
                        position,
                        delta,
                        unit,
                        modifiers,
                        source,
                    });
                }
                egui::Event::Touch { id, phase, pos, .. } => {
                    // Touch ids above the mouse's.
                    let id = (id.0 as u32).wrapping_add(1);
                    let position = point(pos);
                    let modifiers = self.input.modifiers;
                    match phase {
                        egui::TouchPhase::Start => {
                            if over_ui(pos) {
                                continue;
                            }
                            self.input.touches += 1;
                            self.map_input(Input::PointerDown {
                                id,
                                kind: PointerKind::Touch,
                                button: Button::Primary,
                                position,
                                modifiers,
                            });
                        }
                        egui::TouchPhase::Move if self.input.touches > 0 => {
                            self.map_input(Input::PointerMove { id, position });
                        }
                        egui::TouchPhase::End if self.input.touches > 0 => {
                            self.input.touches -= 1;
                            self.map_input(Input::PointerUp { id, position });
                        }
                        egui::TouchPhase::Cancel if self.input.touches > 0 => {
                            self.input.touches -= 1;
                            self.map_input(Input::PointerCancel { id });
                        }
                        _ => {}
                    }
                }
                egui::Event::Key {
                    key, pressed: true, ..
                } if keyboard => {
                    if key == egui::Key::Escape && (self.scenes_open || self.options_open) {
                        self.scenes_open = false;
                        self.options_open = false;
                        continue;
                    }
                    let key = match key {
                        egui::Key::ArrowLeft => Key::Left,
                        egui::Key::ArrowRight => Key::Right,
                        egui::Key::ArrowUp => Key::Up,
                        egui::Key::ArrowDown => Key::Down,
                        egui::Key::Plus | egui::Key::Equals => Key::ZoomIn,
                        egui::Key::Minus => Key::ZoomOut,
                        egui::Key::Escape => Key::Escape,
                        _ => continue,
                    };
                    let modifiers = self.input.modifiers;
                    self.map_input(Input::Key { key, modifiers });
                }
                egui::Event::WindowFocused(false) => {
                    self.input.owned.clear();
                    self.input.touches = 0;
                    self.map_input(Input::Blur);
                }
                _ => {}
            }
        }
    }

    /// Draws a frame of the map into the target, sized to `rect`, if it
    /// needs one, and keeps egui's view of its texture current.
    fn draw_map(
        &mut self,
        rect: Rect,
        pixels_per_point: f32,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
    ) {
        let width = (rect.width() * pixels_per_point).round().max(1.0) as u32;
        let height = (rect.height() * pixels_per_point).round().max(1.0) as u32;
        self.target
            .resize(width, height, f64::from(pixels_per_point));
        self.fly();
        if !self.stopped && (self.map.needs_frame() || self.texture.is_none()) {
            let now = self.now();
            match self.map.render(&mut self.target, now) {
                Ok(Some(report)) => {
                    self.frames += 1;
                    if report.idle && self.idle_at.is_none() {
                        self.idle_at = Some((Instant::now(), self.frames));
                    }
                }
                Ok(None) => {}
                // A new GPU: the texture goes with the old one.
                Err(Error::TargetLost(_)) => {
                    self.target =
                        Target::texture(&self.runtime, width, height, f64::from(pixels_per_point));
                }
                Err(e) => {
                    eprintln!("orb-demo: stopped drawing: {e}");
                    self.say(format!("Stopped drawing: {e}"), true);
                    self.stopped = true;
                }
            }
        }
        self.events();
        let Some(texture) = self.target.wgpu_texture() else {
            return;
        };
        if self.texture.as_ref().is_some_and(|(_, t)| *t == texture) {
            return;
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let id = match &self.texture {
            Some((id, _)) => {
                renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    &view,
                    wgpu::FilterMode::Linear,
                    *id,
                );
                *id
            }
            None => renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear),
        };
        self.texture = Some((id, texture));
    }

    fn count_frames(&self, ctx: &egui::Context) {
        let (Some(seconds), Some((idle, at))) = (self.count_frames, self.idle_at) else {
            return;
        };
        let deadline = idle + Duration::from_secs_f64(seconds);
        let now = Instant::now();
        if now >= deadline {
            println!(
                "orb-demo: idle after {at} frames; {} frames in the {seconds} s since",
                self.frames - at
            );
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else {
            ctx.request_repaint_after(deadline - now);
        }
    }
}

impl Demo {
    /// A frame of the window: the map under the controls.
    pub(crate) fn ui(
        &mut self,
        ui: &mut Ui,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
    ) {
        let ctx = ui.ctx().clone();
        let palette =
            Palette::new(self.scene.dark || ctx.system_theme() == Some(egui::Theme::Dark));
        palette.apply(&ctx);
        self.forward_input(&ctx);
        let rect = ui.max_rect();
        self.draw_map(rect, ctx.pixels_per_point(), device, renderer);
        if let Some((id, _)) = &self.texture {
            ui.painter().image(
                *id,
                rect,
                Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        // A Shift-drag's box, as the web map draws it.
        if let Some([a, b]) = self.map.box_zoom() {
            let corner = |p: DVec2| rect.min + vec2(p.x as f32, p.y as f32);
            let shape = Rect::from_two_pos(corner(a), corner(b));
            let painter = ui.painter();
            painter.rect_filled(shape, 0.0, Color32::from_white_alpha(128));
            painter.rect_stroke(
                shape,
                0.0,
                Stroke::new(2.0, Color32::from_gray(32)),
                StrokeKind::Inside,
            );
        }
        self.scene_button(&ctx, &palette);
        if self.scenes_open {
            self.scene_sheet(&ctx, &palette);
        }
        self.tools(&ctx, &palette);
        if self.options_open {
            self.options_sheet(&ctx, &palette);
        }
        self.camera_tools(&ctx, &palette);
        self.attribution(&ctx, &palette);
        self.toast(&ctx, &palette);
        if self.map.needs_frame() || self.flight.is_some() {
            ctx.request_repaint();
        }
        self.count_frames(&ctx);
    }
}

// The controls, after the web demo's (web/demo/app/app.css).

const EDGE: f32 = 12.0;
const TARGET: f32 = 44.0;
const SHEET_WIDTH: f32 = 340.0;

/// The web demo's colour tokens, light or dark.
struct Palette {
    dark: bool,
    surface: Color32,
    well: Color32,
    ink: Color32,
    muted: Color32,
    hairline: Color32,
    accent: Color32,
    accent_ink: Color32,
    accent_wash: Color32,
    danger: Color32,
}

impl Palette {
    fn new(dark: bool) -> Self {
        let hex = |s: &str| Color32::from_hex(s).expect("palette colour");
        if dark {
            Self {
                dark,
                surface: hex("#1d232af0"),
                well: hex("#2a323b"),
                ink: hex("#eef1f4"),
                muted: hex("#a7b1bc"),
                hairline: hex("#3a434d"),
                accent: hex("#ff6fae"),
                accent_ink: hex("#1d232a"),
                accent_wash: hex("#4a2337"),
                danger: hex("#ff8a80"),
            }
        } else {
            Self {
                dark,
                surface: hex("#fffffff0"),
                well: hex("#eef1f4"),
                ink: hex("#17202a"),
                muted: hex("#56616d"),
                hairline: hex("#d3d9df"),
                accent: hex("#b3195f"),
                accent_ink: hex("#ffffff"),
                accent_wash: hex("#fbe7f0"),
                danger: hex("#b42318"),
            }
        }
    }

    fn apply(&self, ctx: &egui::Context) {
        let mut visuals = if self.dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        };
        visuals.override_text_color = Some(self.ink);
        visuals.selection.bg_fill = self.accent;
        visuals.selection.stroke = Stroke::new(1.0, self.accent_ink);
        visuals.widgets.hovered.bg_fill = self.well;
        visuals.widgets.hovered.weak_bg_fill = self.well;
        visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
        visuals.widgets.inactive.bg_fill = self.well;
        ctx.set_visuals(visuals);
    }

    fn shadow(&self, high: bool) -> Shadow {
        Shadow {
            offset: [0, if high { 12 } else { 2 }],
            blur: if high { 40 } else { 10 },
            spread: 0,
            color: Color32::from_black_alpha(if self.dark { 140 } else { 40 }),
        }
    }

    /// A floating control's frame.
    fn control(&self) -> Frame {
        Frame::new()
            .fill(self.surface)
            .stroke(Stroke::new(1.0, self.hairline))
            .corner_radius(CornerRadius::same(12))
            .shadow(self.shadow(false))
    }

    /// A sheet's frame.
    fn sheet(&self) -> Frame {
        Frame::new()
            .fill(self.surface)
            .stroke(Stroke::new(1.0, self.hairline))
            .corner_radius(CornerRadius::same(18))
            .shadow(self.shadow(true))
            .inner_margin(Margin::same(14))
    }
}

/// A floating layer above the map.
fn layer(id: &str, anchor: Align2, offset: Vec2) -> egui::Area {
    egui::Area::new(Id::unique(id))
        .anchor(anchor, offset)
        .order(Order::Foreground)
        .constrain(true)
}

/// A square tool button with a painted icon.
fn tool(
    ui: &mut Ui,
    palette: &Palette,
    hint: &str,
    icon: impl FnOnce(&egui::Painter, Rect, Color32),
) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(TARGET), Sense::click());
    if response.hovered() {
        ui.painter()
            .rect_filled(rect.shrink(4.0), CornerRadius::same(8), palette.well);
    }
    icon(ui.painter(), rect, palette.ink);
    response
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(hint)
}

fn close_button(ui: &mut Ui, palette: &Palette) -> Response {
    tool(ui, palette, "Close", |p, rect, ink| {
        let c = rect.center();
        let s = Stroke::new(1.8, ink);
        p.line_segment([c + vec2(-6.0, -6.0), c + vec2(6.0, 6.0)], s);
        p.line_segment([c + vec2(-6.0, 6.0), c + vec2(6.0, -6.0)], s);
    })
}

fn sheet_head(ui: &mut Ui, palette: &Palette, title: &str) -> bool {
    let (_, close) = egui::Sides::new().show(
        ui,
        |ui| ui.label(RichText::new(title).size(18.0).strong()),
        |ui| close_button(ui, palette).clicked(),
    );
    close
}

fn group_heading(ui: &mut Ui, palette: &Palette, text: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(text).size(12.0).strong().color(palette.muted));
    ui.add_space(2.0);
}

/// A small pill of text.
fn tag(ui: &mut Ui, palette: &Palette, text: &str, hint: Option<&str>) {
    let response = Frame::new()
        .fill(palette.well)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.0).color(palette.muted));
        })
        .response;
    if let Some(hint) = hint {
        response.on_hover_text(hint);
    }
}

/// The picker's chip in a style's colours: land, a corner of water, a line.
fn chip(ui: &mut Ui, swatch: &[String; 3]) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(32.0), Sense::hover());
    let color = |s: &String| Color32::from_hex(s).unwrap_or(Color32::GRAY);
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(8), color(&swatch[0]));
    let water = rect.right_bottom();
    let points = (0..=8)
        .map(|i| {
            let a = std::f32::consts::FRAC_PI_2 * (i as f32 / 8.0);
            water + vec2(-a.cos(), -a.sin()) * 18.0
        })
        .chain([water])
        .collect();
    p.add(Shape::convex_polygon(
        points,
        color(&swatch[1]),
        Stroke::NONE,
    ));
    p.line_segment(
        [
            rect.left_top() + vec2(4.0, 22.0),
            rect.left_top() + vec2(24.0, 4.0),
        ],
        Stroke::new(3.0, color(&swatch[2])),
    );
    p.rect_stroke(
        rect,
        CornerRadius::same(8),
        Stroke::new(1.0, Color32::from_black_alpha(30)),
        StrokeKind::Inside,
    );
}

/// An on-off switch.
fn switch(ui: &mut Ui, palette: &Palette, on: bool) -> Response {
    let size = vec2(40.0, 24.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let t = ui.ctx().animate_bool_responsive(response.id, on);
    let fill = if ui.is_enabled() {
        palette.hairline.lerp_to_gamma(palette.accent, t)
    } else {
        palette.well
    };
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(12), fill);
    let x = egui::lerp((rect.left() + 12.0)..=(rect.right() - 12.0), t);
    p.circle_filled(pos2(x, rect.center().y), 9.0, Color32::WHITE);
    response.on_hover_cursor(CursorIcon::PointingHand)
}

/// A row of choices; the one picked, if it changed.
fn segmented(
    ui: &mut Ui,
    palette: &Palette,
    choices: &[(&'static str, &'static str)],
    current: &str,
) -> Option<&'static str> {
    let mut picked = None;
    Frame::new()
        .fill(palette.well)
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for &(value, label) in choices {
                    let on = value == current;
                    let text = RichText::new(label).size(13.0).color(if on {
                        palette.accent_ink
                    } else {
                        palette.ink
                    });
                    let button = egui::Button::new(text)
                        .fill(if on {
                            palette.accent
                        } else {
                            Color32::TRANSPARENT
                        })
                        .corner_radius(CornerRadius::same(8))
                        .min_size(vec2(64.0, 26.0));
                    if ui.add(button).clicked() && !on {
                        picked = Some(value);
                    }
                }
            });
        });
    picked
}

impl Demo {
    fn scene_button(&mut self, ctx: &egui::Context, palette: &Palette) {
        let response = layer("scene-button", Align2::LEFT_TOP, vec2(EDGE, EDGE)).show(ctx, |ui| {
            let inner = palette
                .control()
                .inner_margin(Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let (mark, _) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::hover());
                        orbis_mark(ui.painter(), mark, palette);
                        ui.vertical(|ui| {
                            ui.spacing_mut().item_spacing.y = 0.0;
                            ui.label(RichText::new(&self.scene.name).size(15.0).strong());
                            ui.label(
                                RichText::new(&self.scene.provider)
                                    .size(12.0)
                                    .color(palette.muted),
                            );
                        });
                        let (chevron, _) = ui.allocate_exact_size(vec2(14.0, 22.0), Sense::hover());
                        let c = chevron.center();
                        ui.painter().add(Shape::line(
                            vec![
                                c + vec2(-4.0, -2.0),
                                c + vec2(0.0, 2.0),
                                c + vec2(4.0, -2.0),
                            ],
                            Stroke::new(1.6, palette.muted),
                        ));
                    });
                });
            inner.response.interact(Sense::click())
        });
        let clicked = response
            .inner
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text("Choose a map")
            .clicked();
        if clicked {
            self.scenes_open = !self.scenes_open;
            self.options_open = false;
        }
    }

    fn scene_sheet(&mut self, ctx: &egui::Context, palette: &Palette) {
        let screen = ctx.content_rect();
        let mut picked = None;
        let mut close = false;
        layer("scenes", Align2::LEFT_TOP, vec2(EDGE, EDGE + 60.0)).show(ctx, |ui| {
            palette.sheet().show(ui, |ui| {
                ui.set_width(SHEET_WIDTH);
                close = sheet_head(ui, palette, "Choose a map");
                egui::ScrollArea::vertical()
                    .max_height((screen.height() - 160.0).max(200.0))
                    .show(ui, |ui| {
                        for group in &self.catalogue.groups {
                            group_heading(ui, palette, &group.name);
                            for scene in &group.scenes {
                                if scene_row(ui, palette, scene, scene.key == self.scene.key)
                                    .clicked()
                                {
                                    picked = Some(scene.clone());
                                }
                            }
                        }
                    });
            });
        });
        if close || picked.is_some() {
            self.scenes_open = false;
        }
        if let Some(scene) = picked
            && scene.key != self.scene.key
        {
            self.go(scene);
        }
    }

    fn tools(&mut self, ctx: &egui::Context, palette: &Palette) {
        layer("tools", Align2::RIGHT_TOP, vec2(-EDGE, EDGE)).show(ctx, |ui| {
            palette.control().show(ui, |ui| {
                let clicked = tool(ui, palette, "Map options", sliders_icon).clicked();
                if clicked {
                    self.options_open = !self.options_open;
                    self.scenes_open = false;
                }
            });
        });
    }

    fn options_sheet(&mut self, ctx: &egui::Context, palette: &Palette) {
        let screen = ctx.content_rect();
        let mut close = false;
        let mut picks = Vec::new();
        layer(
            "options",
            Align2::RIGHT_TOP,
            vec2(-(EDGE + TARGET + 8.0), EDGE),
        )
        .show(ctx, |ui| {
            palette.sheet().show(ui, |ui| {
                ui.set_width(SHEET_WIDTH);
                close = sheet_head(ui, palette, "Map options");
                if self.panel.available.is_empty() {
                    let text = if self.loading {
                        "Loading the map…"
                    } else {
                        "This map has no options."
                    };
                    ui.label(RichText::new(text).color(palette.muted));
                    return;
                }
                egui::ScrollArea::vertical()
                    .max_height((screen.height() - 120.0).max(200.0))
                    .show(ui, |ui| {
                        let mut group = "";
                        for &option in &self.panel.available {
                            if option.group() != group {
                                group = option.group();
                                group_heading(ui, palette, group);
                            }
                            let value = self.panel.value(option).cloned().unwrap_or(Value::Null);
                            let disabled = self.panel.disabled(option);
                            if let Some(value) = option_row(ui, palette, option, &value, disabled) {
                                picks.push((option, value));
                            }
                        }
                    });
            });
        });
        if close {
            self.options_open = false;
        }
        for (option, value) in picks {
            if self.panel.pick(option, value, &mut self.style) {
                self.set_style();
            }
        }
    }

    fn camera_tools(&mut self, ctx: &egui::Context, palette: &Palette) {
        let camera = self.map.camera();
        let zoom = self.map.zoom();
        let mut target = None;
        layer(
            "camera-tools",
            Align2::RIGHT_BOTTOM,
            vec2(-EDGE, -(EDGE + 28.0)),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            palette.control().show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                if tool(ui, palette, "Zoom in", |p, r, ink| {
                    plus_minus(p, r, ink, true)
                })
                .clicked()
                {
                    target = Some((
                        CameraTarget {
                            zoom: Some(zoom + 1.0),
                            ..Default::default()
                        },
                        300.0,
                    ));
                }
                let (line, _) = ui.allocate_exact_size(vec2(TARGET, 1.0), Sense::hover());
                ui.painter().hline(
                    line.x_range().shrink(8.0),
                    line.center().y,
                    Stroke::new(1.0, palette.hairline),
                );
                if tool(ui, palette, "Zoom out", |p, r, ink| {
                    plus_minus(p, r, ink, false)
                })
                .clicked()
                {
                    target = Some((
                        CameraTarget {
                            zoom: Some(zoom - 1.0),
                            ..Default::default()
                        },
                        300.0,
                    ));
                }
            });
            palette.control().show(ui, |ui| {
                let (bearing, pitch) = (camera.bearing, camera.pitch);
                let compass = tool(ui, palette, "Reset north and tilt", |p, r, _| {
                    needle(p, r, palette, bearing, pitch);
                });
                if compass.clicked() {
                    target = Some((
                        CameraTarget {
                            bearing: Some(0.0),
                            pitch: Some(0.0),
                            ..Default::default()
                        },
                        400.0,
                    ));
                }
            });
        });
        if let Some((target, duration)) = target {
            self.flight = None;
            self.map.ease_to(
                &target,
                &EaseOptions {
                    duration,
                    ..EaseOptions::default()
                },
            );
        }
    }

    fn attribution(&self, ctx: &egui::Context, palette: &Palette) {
        if self.attribution.is_empty() {
            return;
        }
        layer("attribution", Align2::RIGHT_BOTTOM, vec2(-EDGE, -6.0)).show(ctx, |ui| {
            Frame::new()
                .fill(palette.surface)
                .corner_radius(CornerRadius::same(6))
                .inner_margin(Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(&self.attribution)
                            .size(11.0)
                            .color(palette.muted),
                    );
                });
        });
    }

    fn toast(&self, ctx: &egui::Context, palette: &Palette) {
        let Some((text, error)) = &self.toast else {
            return;
        };
        layer("toast", Align2::CENTER_BOTTOM, vec2(0.0, -(EDGE + 36.0))).show(ctx, |ui| {
            palette
                .control()
                .inner_margin(Margin::symmetric(14, 8))
                .show(ui, |ui| {
                    let color = if *error { palette.danger } else { palette.ink };
                    ui.label(RichText::new(text).size(14.0).color(color));
                });
        });
    }
}

fn scene_row(ui: &mut Ui, palette: &Palette, scene: &Scene, current: bool) -> Response {
    let background = ui.painter().add(Shape::Noop);
    let inner = Frame::new()
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                chip(ui, &scene.swatch);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&scene.name).size(15.0));
                        if scene.dark {
                            tag(ui, palette, "Dark", None);
                        }
                    });
                    if let Some(detail) = &scene.detail {
                        ui.add(
                            egui::Label::new(RichText::new(detail).size(12.0).color(palette.muted))
                                .wrap(),
                        );
                    }
                });
            });
        });
    let response = inner
        .response
        .interact(Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);
    let fill = if current {
        palette.accent_wash
    } else if response.hovered() {
        palette.well
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().set(
        background,
        Shape::rect_filled(response.rect, CornerRadius::same(10), fill),
    );
    response
}

/// An option's row; the value picked, if any.
fn option_row(
    ui: &mut Ui,
    palette: &Palette,
    option: Opt,
    value: &Value,
    disabled: Option<&str>,
) -> Option<Value> {
    let mut picked = None;
    let row = egui::Sides::new().shrink_left().wrap().show(
        ui,
        |ui| {
            ui.set_min_height(TARGET);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.horizontal(|ui| {
                    let label = RichText::new(option.label()).size(15.0);
                    ui.label(if disabled.is_some() {
                        label.color(palette.muted)
                    } else {
                        label
                    });
                    if option.orbis_only() {
                        tag(
                            ui,
                            palette,
                            "Orbis only",
                            Some(
                                "A preview of a style-spec proposal: GL JS and MapLibre Native don't draw it",
                            ),
                        );
                    }
                });
                if let Some(note) = option.note() {
                    ui.label(RichText::new(note).size(12.0).color(palette.muted));
                }
            })
            .response
        },
        |ui| {
            ui.add_enabled_ui(disabled.is_none(), |ui| match option.kind() {
                Kind::Switch => {
                    let on = value.as_bool() == Some(true);
                    if switch(ui, palette, on).clicked() {
                        picked = Some(json!(!on));
                    }
                }
                Kind::Choices(choices) => {
                    let current = value.as_str().unwrap_or_default();
                    if let Some(choice) = segmented(ui, palette, choices, current) {
                        picked = Some(json!(choice));
                    }
                }
            });
        },
    );
    if let Some(why) = disabled {
        row.0.on_hover_text(why);
    }
    picked
}

/// The site's mark: a planet in its ring.
fn orbis_mark(p: &egui::Painter, rect: Rect, palette: &Palette) {
    let c = rect.center();
    p.circle_filled(c, 6.0, palette.accent);
    let (a, b) = (10.5_f32, 4.2_f32);
    let tilt = (-28.0_f32).to_radians();
    let points = (0..48)
        .map(|i| {
            let t = std::f32::consts::TAU * i as f32 / 48.0;
            let (x, y) = (a * t.cos(), b * t.sin());
            c + vec2(
                x * tilt.cos() - y * tilt.sin(),
                x * tilt.sin() + y * tilt.cos(),
            )
        })
        .collect();
    p.add(Shape::closed_line(points, Stroke::new(1.4, palette.ink)));
}

fn sliders_icon(p: &egui::Painter, rect: Rect, ink: Color32) {
    let c = rect.center();
    let s = Stroke::new(1.8, ink);
    for (dy, knob) in [(-5.0, 3.0), (0.0, -3.0), (5.0, 1.0)] {
        p.line_segment([c + vec2(-8.0, dy), c + vec2(8.0, dy)], s);
        p.circle_filled(c + vec2(knob, dy), 2.6, ink);
    }
}

fn plus_minus(p: &egui::Painter, rect: Rect, ink: Color32, plus: bool) {
    let c = rect.center();
    let s = Stroke::new(1.8, ink);
    p.line_segment([c + vec2(-7.0, 0.0), c + vec2(7.0, 0.0)], s);
    if plus {
        p.line_segment([c + vec2(0.0, -7.0), c + vec2(0.0, 7.0)], s);
    }
}

/// The compass needle: north turned by the bearing, flattened by the tilt.
fn needle(p: &egui::Painter, rect: Rect, palette: &Palette, bearing: f64, pitch: f64) {
    let c = rect.center();
    let angle = (-bearing as f32).to_radians();
    let flatten = ((pitch * 0.7) as f32).to_radians().cos();
    let at = |x: f32, y: f32| {
        let (x, y) = (
            x * angle.cos() - y * angle.sin(),
            x * angle.sin() + y * angle.cos(),
        );
        c + vec2(x, y * flatten)
    };
    p.add(Shape::convex_polygon(
        vec![at(0.0, -11.0), at(4.0, 0.0), at(-4.0, 0.0)],
        palette.accent,
        Stroke::NONE,
    ));
    p.add(Shape::convex_polygon(
        vec![at(0.0, 11.0), at(-4.0, 0.0), at(4.0, 0.0)],
        palette.muted,
        Stroke::NONE,
    ));
}

fn point(pos: Pos2) -> DVec2 {
    DVec2::new(f64::from(pos.x), f64::from(pos.y))
}

/// The style's sources' attributions as text, each once.
fn attribution(style: &Value) -> String {
    let mut seen = Vec::new();
    for source in style["sources"]
        .as_object()
        .into_iter()
        .flat_map(|s| s.values())
    {
        if let Some(html) = source["attribution"].as_str() {
            let text = strip_html(html);
            if !text.is_empty() && !seen.contains(&text) {
                seen.push(text);
            }
        }
    }
    seen.join(" · ")
}

fn strip_html(html: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    text.replace("&copy;", "©")
        .replace("&amp;", "&")
        .replace("&nbsp;", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
