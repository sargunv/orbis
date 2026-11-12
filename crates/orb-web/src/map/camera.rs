//! The map's camera and input: DOM input with its event times (forwarded
//! from the page where the map runs in a worker), camera moves, and
//! `project` and `unproject`.
//! Camera targets and options cross as the package's own objects
//! (`CameraOptions`, `FlyOptions`, `FitOptions` in `web/src/types.ts`),
//! read by name.

use glam::DVec2;
use orb_camera::input::{
    Button, Fling, GestureOptions, GesturePhase, Input, Key, Modifiers, Movement, PointerKind,
    Thresholds, WheelSource, WheelUnit,
};
use orb_camera::{
    AnchoredMomentum, Bounds, CameraTarget, Constraints, DecayModel, EaseOptions, Easing,
    FitOptions, FlyOptions, GestureMomentum, MomentumDecay, MomentumSnap, Padding, PanMomentum,
};
use orb_crs::Geodetic;
use wasm_bindgen::prelude::*;

use crate::OrbisMap;

/// The camera as the package's `Camera`: degrees, zoom, and metres the
/// center stands above sea level.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    #[wasm_bindgen(readonly)]
    pub lon: f64,
    #[wasm_bindgen(readonly)]
    pub lat: f64,
    #[wasm_bindgen(readonly)]
    pub zoom: f64,
    #[wasm_bindgen(readonly)]
    pub bearing: f64,
    #[wasm_bindgen(readonly)]
    pub pitch: f64,
    #[wasm_bindgen(readonly)]
    pub roll: f64,
    #[wasm_bindgen(readonly)]
    pub elevation: f64,
    #[wasm_bindgen(readonly)]
    pub fov: f64,
    padding: Padding,
}

#[wasm_bindgen]
impl Camera {
    /// `{top, right, bottom, left}`.
    #[wasm_bindgen(getter)]
    pub fn padding(&self) -> JsValue {
        padding_object(self.padding)
    }
}

fn padding_object(p: Padding) -> JsValue {
    let object = js_sys::Object::new();
    for (key, value) in [
        ("top", p.top),
        ("right", p.right),
        ("bottom", p.bottom),
        ("left", p.left),
    ] {
        crate::js::set(&object, key, &value.into());
    }
    object.into()
}

/// A `padding`: a number for every side, or `{top, right, bottom, left}`
/// with the sides left out as in `current`; `None` when absent.
fn padding(value: &JsValue, current: Padding) -> Option<Padding> {
    if let Some(all) = value.as_f64() {
        return Some(Padding::uniform(all));
    }
    if !value.is_object() {
        return None;
    }
    Some(Padding {
        top: number(value, "top").unwrap_or(current.top),
        right: number(value, "right").unwrap_or(current.right),
        bottom: number(value, "bottom").unwrap_or(current.bottom),
        left: number(value, "left").unwrap_or(current.left),
    })
}

/// `current` with the fields a `TileLod` object gives, MapLibre Native's
/// tile level of detail: `mode` (`"default"` or `"distance"`),
/// `minRadius`, `scale`, `pitchThreshold` (radians) and `zoomShift`. Fields
/// it leaves out stay, as maplibre-native-ffi's tile options' field mask
/// keeps them.
pub(crate) fn tile_lod(current: orb_camera::TileLod, value: &JsValue) -> orb_camera::TileLod {
    use orb_camera::TileLodMode;
    let mode = match crate::js::get(value, "mode").as_string().as_deref() {
        Some("default") => TileLodMode::Default,
        Some("distance") => TileLodMode::Distance,
        _ => current.mode,
    };
    orb_camera::TileLod {
        mode,
        min_radius: number(value, "minRadius").unwrap_or(current.min_radius),
        scale: number(value, "scale").unwrap_or(current.scale),
        pitch_threshold: number(value, "pitchThreshold").unwrap_or(current.pitch_threshold),
        zoom_shift: number(value, "zoomShift").unwrap_or(current.zoom_shift),
    }
}

fn number(object: &JsValue, key: &str) -> Option<f64> {
    crate::js::get(object, key).as_f64()
}

/// `[a, b]` as numbers.
fn pair(value: &JsValue) -> Option<[f64; 2]> {
    let array = value.dyn_ref::<js_sys::Array>()?;
    Some([array.get(0).as_f64()?, array.get(1).as_f64()?])
}

fn lng_lat(value: &JsValue) -> Option<Geodetic> {
    pair(value).map(|[lon, lat]| Geodetic::new(lon, lat))
}

/// A `CameraOptions` object; its padding's sides left out keep
/// `padding`'s.
fn target(options: &JsValue, current: Padding) -> CameraTarget {
    CameraTarget {
        fov: number(options, "fov"),
        padding: padding(&crate::js::get(options, "padding"), current),
        center: lng_lat(&crate::js::get(options, "center")),
        zoom: number(options, "zoom"),
        bearing: number(options, "bearing"),
        pitch: number(options, "pitch"),
        roll: number(options, "roll"),
        elevation: number(options, "elevation"),
        around: lng_lat(&crate::js::get(options, "around")),
        offset: pair(&crate::js::get(options, "offset")).map_or(DVec2::ZERO, DVec2::from),
    }
}

/// An `Easing`: a cubic Bézier's control points, `"linear"`, or GL JS's
/// default when absent.
fn easing(options: &JsValue) -> Easing {
    let value = crate::js::get(options, "easing");
    if value.as_string().as_deref() == Some("linear") {
        return Easing::Linear;
    }
    let Some(array) = value.dyn_ref::<js_sys::Array>() else {
        return Easing::DEFAULT;
    };
    let at = |i| array.get(i).as_f64();
    match (at(0), at(1), at(2), at(3)) {
        (Some(x1), Some(y1), Some(x2), Some(y2)) => Easing::Bezier([x1, y1, x2, y2]),
        _ => Easing::DEFAULT,
    }
}

/// A `MomentumDecay` object over `default`: `model` `"friction"` (with
/// `halfLife`, `stopSpeed`) or `"ease"` (with `distance`, `duration`,
/// `easing`), and `minSpeed`, `maxSpeed`, `maxDistance`.
fn decay(value: &JsValue, default: MomentumDecay) -> MomentumDecay {
    if !value.is_object() {
        return default;
    }
    let (half_life, stop_speed) = match default.model {
        DecayModel::Friction {
            half_life,
            stop_speed,
        } => (half_life, stop_speed),
        DecayModel::Ease { .. } => (140.0, 1.0),
    };
    let model = match crate::js::get(value, "model").as_string().as_deref() {
        Some("ease") => DecayModel::Ease {
            distance: number(value, "distance").unwrap_or(0.0),
            duration: number(value, "duration").unwrap_or(0.0),
            easing: if crate::js::get(value, "easing").is_undefined() {
                Easing::Bezier([0.0, 0.0, 0.25, 1.0])
            } else {
                easing(value)
            },
        },
        _ => DecayModel::Friction {
            half_life: number(value, "halfLife").unwrap_or(half_life),
            stop_speed: number(value, "stopSpeed").unwrap_or(stop_speed),
        },
    };
    MomentumDecay {
        model,
        min_speed: number(value, "minSpeed").unwrap_or(default.min_speed),
        max_speed: number(value, "maxSpeed").unwrap_or(default.max_speed),
        max_distance: number(value, "maxDistance").unwrap_or(default.max_distance),
    }
}

/// An `AnchoredMomentum` object: `velocity`, `anchor` `[x, y]`, `decay`,
/// and `snap`, `true` for the part's default or a `MomentumSnap` over it.
fn anchored(
    value: &JsValue,
    part: fn(Option<f64>) -> AnchoredMomentum,
    snap_default: MomentumSnap,
) -> Option<AnchoredMomentum> {
    if !value.is_object() {
        return None;
    }
    let base = part(number(value, "velocity"));
    let snap = crate::js::get(value, "snap");
    let snap = if snap.as_bool() == Some(true) {
        Some(snap_default)
    } else if snap.is_object() {
        Some(snap_options(&snap, snap_default))
    } else {
        None
    };
    Some(AnchoredMomentum {
        anchor: pair(&crate::js::get(value, "anchor")).map(DVec2::from),
        decay: decay(&crate::js::get(value, "decay"), base.decay),
        snap,
        ..base
    })
}

/// A `GestureOptions` object over the defaults.
fn gestures(value: &JsValue) -> GestureOptions {
    let defaults = GestureOptions::default();
    let movement = |key: &str, default: Movement, snap_default: MomentumSnap| {
        let value = crate::js::get(value, key);
        if let Some(enabled) = value.as_bool() {
            return Movement { enabled, ..default };
        }
        if !value.is_object() {
            return default;
        }
        let decay_default = match default.momentum {
            Some(Fling::Decay(decay)) => decay,
            _ => MomentumDecay::PAN,
        };
        let momentum = crate::js::get(&value, "momentum");
        let momentum = match momentum.as_bool() {
            Some(false) => None,
            Some(true) => default.momentum.or(Some(Fling::Decay(decay_default))),
            None if momentum.is_object() => Some(Fling::Decay(decay(&momentum, decay_default))),
            None => default.momentum,
        };
        let snap = crate::js::get(&value, "snap");
        let snap = match snap.as_bool() {
            Some(on) => on.then_some(default.snap.unwrap_or(snap_default)),
            None if snap.is_object() => Some(snap_options(&snap, snap_default)),
            None => default.snap,
        };
        Movement {
            enabled: crate::js::get(&value, "enabled")
                .as_bool()
                .unwrap_or(default.enabled),
            momentum,
            snap,
        }
    };
    let t = defaults.thresholds;
    GestureOptions {
        pan: movement("pan", defaults.pan, MomentumSnap::ZOOM),
        zoom: movement("zoom", defaults.zoom, MomentumSnap::ZOOM),
        rotate: movement("rotate", defaults.rotate, MomentumSnap::BEARING),
        pitch: movement("pitch", defaults.pitch, MomentumSnap::PITCH),
        thresholds: Thresholds {
            click_tolerance: number(value, "clickTolerance").unwrap_or(t.click_tolerance),
            tap_tolerance: number(value, "tapTolerance").unwrap_or(t.tap_tolerance),
            tap_interval: number(value, "tapInterval").unwrap_or(t.tap_interval),
            long_press: number(value, "longPress").unwrap_or(t.long_press),
        },
    }
}

/// A `MomentumSnap` object over `default`.
fn snap_options(value: &JsValue, default: MomentumSnap) -> MomentumSnap {
    MomentumSnap {
        spacing: number(value, "spacing").unwrap_or(default.spacing),
        offset: number(value, "offset").unwrap_or(default.offset),
        tolerance: number(value, "tolerance").unwrap_or(default.tolerance),
        duration: number(value, "duration").unwrap_or(default.duration),
    }
}

fn modifiers(bits: u32) -> Modifiers {
    Modifiers {
        shift: bits & 1 != 0,
        ctrl: bits & 2 != 0,
        alt: bits & 4 != 0,
        meta: bits & 8 != 0,
    }
}

impl OrbisMap {
    fn target(&self, options: &JsValue) -> CameraTarget {
        target(options, self.map.camera().padding)
    }

    fn camera_for(&self, options: &JsValue) -> orb_camera::CameraState {
        self.map.camera_for(&self.target(options))
    }
}

#[wasm_bindgen]
impl OrbisMap {
    /// A `TileLod` object ([`tile_lod`]): how tiles coarsen with distance
    /// in pitched views. The next frame covers with it.
    #[wasm_bindgen(js_name = setTileLod)]
    pub fn set_tile_lod(&mut self, lod: &JsValue) {
        let lod = tile_lod(self.map.tile_lod(), lod);
        self.map.set_tile_lod(lod);
    }

    /// A `Constraints` object: zoom and pitch ranges, and bounds.
    #[wasm_bindgen(js_name = setConstraints)]
    pub fn set_constraints(&mut self, constraints: &JsValue) {
        let defaults = Constraints::default();
        let bounds = crate::js::get(constraints, "bounds")
            .dyn_ref::<js_sys::Array>()
            .and_then(|b| {
                let at = |i| b.get(i).as_f64();
                Some(Bounds {
                    west: at(0)?,
                    south: at(1)?,
                    east: at(2)?,
                    north: at(3)?,
                })
            });
        self.map.set_constraints(Constraints {
            min_zoom: number(constraints, "minZoom").unwrap_or(defaults.min_zoom),
            max_zoom: number(constraints, "maxZoom").unwrap_or(defaults.max_zoom),
            min_pitch: number(constraints, "minPitch").unwrap_or(defaults.min_pitch),
            max_pitch: number(constraints, "maxPitch").unwrap_or(defaults.max_pitch),
            bounds,
        });
    }

    /// One input at `time`, applied as it arrives (what it does shows from
    /// the next frame on); whether the map used it, or the page keeps the
    /// platform's default.
    fn input(&mut self, input: Input, time: f64) -> bool {
        self.map.input(input, time)
    }

    /// A pointer pressed at `time` (the event's `timeStamp`): `kind` is the
    /// DOM `pointerType`, `button` the DOM button, `modifiers` bits shift
    /// 1, ctrl 2, alt 4, meta 8.
    #[wasm_bindgen(js_name = pointerDown)]
    #[allow(clippy::too_many_arguments)]
    pub fn pointer_down(
        &mut self,
        id: u32,
        kind: &str,
        button: u32,
        x: f64,
        y: f64,
        modifiers_bits: u32,
        time: f64,
    ) -> bool {
        let kind = match kind {
            "touch" => PointerKind::Touch,
            "pen" => PointerKind::Pen,
            _ => PointerKind::Mouse,
        };
        let button = match button {
            0 => Button::Primary,
            1 => Button::Middle,
            2 => Button::Secondary,
            _ => return false,
        };
        let input = Input::PointerDown {
            id,
            kind,
            button,
            position: DVec2::new(x, y),
            modifiers: modifiers(modifiers_bits),
        };
        self.input(input, time)
    }

    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(&mut self, id: u32, x: f64, y: f64, time: f64) -> bool {
        let position = DVec2::new(x, y);
        self.input(Input::PointerMove { id, position }, time)
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, id: u32, x: f64, y: f64, time: f64) -> bool {
        let position = DVec2::new(x, y);
        self.input(Input::PointerUp { id, position }, time)
    }

    #[wasm_bindgen(js_name = pointerCancel)]
    pub fn pointer_cancel(&mut self, id: u32, time: f64) -> bool {
        self.input(Input::PointerCancel { id }, time)
    }

    /// A DOM wheel event: `mode` is its `deltaMode`.
    #[allow(clippy::too_many_arguments)]
    pub fn wheel(
        &mut self,
        x: f64,
        y: f64,
        dx: f64,
        dy: f64,
        mode: u32,
        modifiers_bits: u32,
        time: f64,
    ) -> bool {
        let unit = match mode {
            1 => WheelUnit::Line,
            2 => WheelUnit::Page,
            _ => WheelUnit::Pixel,
        };
        let input = Input::Wheel {
            position: DVec2::new(x, y),
            delta: DVec2::new(dx, dy),
            unit,
            modifiers: modifiers(modifiers_bits),
            // The DOM doesn't say.
            source: WheelSource::Unknown,
        };
        self.input(input, time)
    }

    /// A pinch the browser recognized (Safari's `gesture*` events) at
    /// `x, y`: since the last, it scaled by `scale` and turned `rotation`
    /// degrees clockwise; `phase` is 0 begin, 1 update, 2 end, 3 cancel.
    #[allow(clippy::too_many_arguments)]
    pub fn pinch(
        &mut self,
        x: f64,
        y: f64,
        scale: f64,
        rotation: f64,
        phase: u32,
        time: f64,
    ) -> bool {
        let phase = match phase {
            0 => GesturePhase::Begin,
            1 => GesturePhase::Update,
            2 => GesturePhase::End,
            _ => GesturePhase::Cancel,
        };
        let input = Input::Pinch {
            center: DVec2::new(x, y),
            scale,
            rotation,
            phase,
        };
        self.input(input, time)
    }

    /// Sets what gestures may move from a `GestureOptions` object: `pan`,
    /// `zoom`, `rotate` and `pitch`, each a boolean or `{ enabled,
    /// momentum, snap }` (`momentum`: a boolean or a `MomentumDecay`;
    /// `snap`: a boolean or a `MomentumSnap`), and `clickTolerance`,
    /// `tapTolerance`, `tapInterval`, `longPress`. What it leaves out is
    /// the default. A change cancels a gesture under way.
    #[wasm_bindgen(js_name = setGestures)]
    pub fn set_gestures(&mut self, options: &JsValue) {
        self.map.set_gestures(gestures(options));
    }

    /// A DOM `keydown`'s `key`.
    pub fn key(&mut self, key: &str, modifiers_bits: u32, time: f64) -> bool {
        let key = match key {
            "ArrowLeft" => Key::Left,
            "ArrowRight" => Key::Right,
            "ArrowUp" => Key::Up,
            "ArrowDown" => Key::Down,
            "+" | "=" => Key::ZoomIn,
            "-" | "_" => Key::ZoomOut,
            "Escape" => Key::Escape,
            _ => return false,
        };
        let modifiers = modifiers(modifiers_bits);
        self.input(Input::Key { key, modifiers }, time)
    }

    pub fn blur(&mut self, time: f64) {
        self.map.input(Input::Blur, time);
    }

    /// The camera now.
    pub fn camera(&self) -> Camera {
        let c = self.map.camera();
        Camera {
            lon: c.target.lon,
            lat: c.target.lat,
            zoom: self.map.zoom(),
            bearing: c.bearing,
            pitch: c.pitch,
            roll: c.roll,
            elevation: c.elevation,
            fov: c.fov,
            padding: c.padding,
        }
    }

    /// The eye now, as a `FreeCamera` (GL JS `getFreeCameraOptions`).
    #[wasm_bindgen(js_name = freeCamera)]
    pub fn free_camera(&self) -> JsValue {
        let f = self.map.free_camera();
        let object = js_sys::Object::new();
        let set = |key: &str, value: JsValue| crate::js::set(&object, key, &value);
        set(
            "position",
            js_sys::Array::of2(&f.position.lon.into(), &f.position.lat.into()).into(),
        );
        for (key, value) in [
            ("altitude", f.altitude),
            ("bearing", f.bearing),
            ("pitch", f.pitch),
            ("roll", f.roll),
        ] {
            set(key, value.into());
        }
        object.into()
    }

    /// Puts the eye at a `FreeCamera` at once (GL JS
    /// `setFreeCameraOptions`); what it leaves out stays.
    #[wasm_bindgen(js_name = jumpToFree)]
    pub fn jump_to_free(&mut self, camera: &JsValue) {
        let now = self.map.free_camera();
        self.map.jump_to_free(&orb_camera::FreeCamera {
            position: lng_lat(&crate::js::get(camera, "position")).unwrap_or(now.position),
            altitude: number(camera, "altitude").unwrap_or(now.altitude),
            bearing: number(camera, "bearing").unwrap_or(now.bearing),
            pitch: number(camera, "pitch").unwrap_or(now.pitch),
            roll: number(camera, "roll").unwrap_or(now.roll),
        });
    }

    /// GL JS `jumpTo` with a `CameraOptions` object.
    #[wasm_bindgen(js_name = jumpTo)]
    pub fn jump_to(&mut self, options: &JsValue) {
        self.map.jump_to(&self.target(options));
    }

    /// GL JS `easeTo` with an `EaseOptions` object, from the next frame;
    /// returns the animation's id.
    #[wasm_bindgen(js_name = easeTo)]
    pub fn ease_to(&mut self, options: &JsValue) -> f64 {
        let ease = EaseOptions {
            duration: number(options, "duration").unwrap_or(500.0),
            easing: easing(options),
        };
        self.map.ease_to(&self.target(options), ease).0 as f64
    }

    /// GL JS `flyTo` with a `FlyOptions` object, from the next frame;
    /// returns the animation's id.
    #[wasm_bindgen(js_name = flyTo)]
    pub fn fly_to(&mut self, options: &JsValue) -> f64 {
        let defaults = FlyOptions::default();
        let fly = FlyOptions {
            curve: number(options, "curve").unwrap_or(defaults.curve),
            speed: number(options, "speed").unwrap_or(defaults.speed),
            screen_speed: number(options, "screenSpeed"),
            duration: number(options, "duration"),
            max_duration: number(options, "maxDuration"),
            min_zoom: number(options, "minZoom"),
            easing: easing(options),
        };
        self.map.fly_to(&self.target(options), &fly).0 as f64
    }

    /// GL JS `cameraForBounds`: `[west, south, east, north]` and a
    /// `FitOptions` object. Returns `CameraOptions`, or null when the
    /// padding leaves no room.
    #[wasm_bindgen(js_name = cameraForBounds)]
    pub fn camera_for_bounds(&self, bounds: &[f64], options: &JsValue) -> JsValue {
        let [west, south, east, north] = *bounds else {
            return JsValue::NULL;
        };
        let fit = FitOptions {
            padding: padding(&crate::js::get(options, "padding"), Padding::default())
                .unwrap_or_default(),
            offset: pair(&crate::js::get(options, "offset")).map_or(DVec2::ZERO, DVec2::from),
            max_zoom: number(options, "maxZoom"),
            bearing: number(options, "bearing"),
            pitch: number(options, "pitch"),
        };
        let bounds = Bounds {
            west,
            south,
            east,
            north,
        };
        let Some(target) = self.map.camera_for_bounds(bounds, &fit) else {
            return JsValue::NULL;
        };
        let object = js_sys::Object::new();
        let set = |key: &str, value: JsValue| crate::js::set(&object, key, &value);
        if let Some(c) = target.center {
            set(
                "center",
                js_sys::Array::of2(&c.lon.into(), &c.lat.into()).into(),
            );
        }
        for (key, value) in [
            ("zoom", target.zoom),
            ("bearing", target.bearing),
            ("pitch", target.pitch),
            ("roll", target.roll),
            ("elevation", target.elevation),
        ] {
            if let Some(value) = value {
                set(key, value.into());
            }
        }
        object.into()
    }

    /// Continues a gesture the page recognized itself with a
    /// `GestureMomentum` object (`pan`, `zoom`, `bearing`, `pitch`), from
    /// the next frame; returns the animation's id, or -1 when nothing
    /// moves.
    #[wasm_bindgen(js_name = startMomentum)]
    pub fn start_momentum(&mut self, options: &JsValue) -> f64 {
        let pan = crate::js::get(options, "pan");
        let momentum = GestureMomentum {
            pan: pair(&crate::js::get(&pan, "velocity")).map(|v| PanMomentum {
                decay: decay(&crate::js::get(&pan, "decay"), MomentumDecay::PAN),
                ..PanMomentum::new(DVec2::from(v))
            }),
            zoom: anchored(
                &crate::js::get(options, "zoom"),
                AnchoredMomentum::zoom,
                MomentumSnap::ZOOM,
            ),
            bearing: anchored(
                &crate::js::get(options, "bearing"),
                AnchoredMomentum::bearing,
                MomentumSnap::BEARING,
            ),
            pitch: anchored(
                &crate::js::get(options, "pitch"),
                AnchoredMomentum::pitch,
                MomentumSnap::PITCH,
            ),
        };
        self.map
            .start_momentum(&momentum)
            .map_or(-1.0, |id| id.0 as f64)
    }

    /// The box a Shift-drag zooms to while it is drawn, `[x0, y0, x1, y1]`
    /// in CSS pixels, or undefined.
    #[wasm_bindgen(js_name = boxZoom)]
    pub fn box_zoom(&self) -> Option<Vec<f64>> {
        self.map.box_zoom().map(|[a, b]| vec![a.x, a.y, b.x, b.y])
    }

    /// Stops every running animation, as interrupted.
    pub fn stop(&mut self) {
        self.map.stop();
    }

    /// Whether an animation or a gesture moves the camera.
    #[wasm_bindgen(js_name = isMoving)]
    pub fn is_moving(&self) -> bool {
        self.map.is_moving()
    }

    /// Whether the target stands on the terrain at rest (GL JS
    /// `setCenterClampedToGround`, on by default).
    #[wasm_bindgen(js_name = setCenterClampedToGround)]
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.map.set_center_clamped_to_ground(clamped);
    }

    #[wasm_bindgen(js_name = centerClampedToGround)]
    pub fn center_clamped_to_ground(&self) -> bool {
        self.map.center_clamped_to_ground()
    }

    /// Puts the target this many metres above sea level at once (GL JS
    /// `setCenterElevation`); while clamped, the terrain takes it back.
    #[wasm_bindgen(js_name = setCenterElevation)]
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.map.set_center_elevation(elevation);
    }

    /// The CSS pixel showing a point, on the terrain, or none behind the
    /// eye: for the camera now, or with `CameraOptions` the one a jump
    /// there would make.
    pub fn project(&self, lon: f64, lat: f64, camera: &JsValue) -> Option<Vec<f64>> {
        let point = Geodetic::new(lon, lat);
        match camera.is_object() {
            true => self.map.project_with(&self.camera_for(camera), point),
            false => self.map.project(point),
        }
        .map(|p| vec![p.x, p.y])
    }

    /// The ground point under a CSS pixel, on the terrain, or none above
    /// the horizon, for the camera now or the one `CameraOptions` make.
    pub fn unproject(&self, x: f64, y: f64, camera: &JsValue) -> Option<Vec<f64>> {
        let point = DVec2::new(x, y);
        match camera.is_object() {
            true => self.map.unproject_with(&self.camera_for(camera), point),
            false => self.map.unproject(point),
        }
        .map(|g| vec![g.lon, g.lat])
    }

    /// Projects `[lon, lat, …]` pairs into `[x, y, …]` CSS pixels, NaN
    /// behind the eye: DOM overlays at the points.
    #[wasm_bindgen(js_name = projectPoints)]
    pub fn project_points(&self, points: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(points.len());
        for &[lon, lat] in points.as_chunks::<2>().0 {
            let p = self
                .map
                .project(Geodetic::new(lon, lat))
                .unwrap_or(DVec2::NAN);
            out.extend([p.x, p.y]);
        }
        out
    }
}
