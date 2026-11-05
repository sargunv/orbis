//! The map's camera and input on the page: DOM input with its
//! event times, camera moves, and synchronous `project` and `unproject`.
//! Camera targets and options cross as the package's own objects
//! (`CameraOptions`, `FlyOptions`, `FitOptions` in `web/src/types.ts`),
//! read by name.

use glam::DVec2;
use orb_camera::{
    Bounds, CameraTarget, Constraints, EaseOptions, Easing, FitOptions, FlyOptions, Padding,
};
use orb_crs::Geodetic;
use orb_interact::{Button, Feel, Input, Key, Modifiers, PointerKind, WheelUnit};
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
        let _ = js_sys::Reflect::set(&object, &key.into(), &value.into());
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

fn get(object: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(object, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

fn number(object: &JsValue, key: &str) -> Option<f64> {
    get(object, key).as_f64()
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
        padding: padding(&get(options, "padding"), current),
        center: lng_lat(&get(options, "center")),
        zoom: number(options, "zoom"),
        bearing: number(options, "bearing"),
        pitch: number(options, "pitch"),
        roll: number(options, "roll"),
        elevation: number(options, "elevation"),
        around: lng_lat(&get(options, "around")),
        offset: pair(&get(options, "offset")).map_or(DVec2::ZERO, DVec2::from),
    }
}

/// An `Easing`: a cubic Bézier's control points, `"linear"`, or GL JS's
/// default when absent.
fn easing(options: &JsValue) -> Easing {
    let value = get(options, "easing");
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
    /// `"desktop"`, `"ios"` or `"android"`.
    #[wasm_bindgen(js_name = setFeel)]
    pub fn set_feel(&mut self, name: &str) -> Result<(), JsError> {
        self.map.set_feel(match name {
            "desktop" => Feel::DESKTOP,
            "ios" => Feel::IOS,
            "android" => Feel::ANDROID,
            _ => return Err(JsError::new("feel is desktop, ios or android")),
        });
        Ok(())
    }

    /// A `Constraints` object: zoom and pitch ranges, and bounds.
    #[wasm_bindgen(js_name = setConstraints)]
    pub fn set_constraints(&mut self, constraints: &JsValue) {
        let defaults = Constraints::default();
        let bounds = get(constraints, "bounds")
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

    /// A pointer pressed at `time` (the event's `timeStamp`): `kind` is the
    /// DOM `pointerType`, `button` the DOM button, `modifiers` bits shift
    /// 1, ctrl 2, alt 4, meta 8. Returns whether the map used it.
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
        self.map.input(input, time)
    }

    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(&mut self, id: u32, x: f64, y: f64, time: f64) -> bool {
        let position = DVec2::new(x, y);
        self.map.input(Input::PointerMove { id, position }, time)
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, id: u32, x: f64, y: f64, time: f64) -> bool {
        let position = DVec2::new(x, y);
        self.map.input(Input::PointerUp { id, position }, time)
    }

    #[wasm_bindgen(js_name = pointerCancel)]
    pub fn pointer_cancel(&mut self, id: u32, time: f64) -> bool {
        self.map.input(Input::PointerCancel { id }, time)
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
        };
        self.map.input(input, time)
    }

    /// A DOM `keydown`'s `key`; returns whether the map used it.
    pub fn key(&mut self, key: &str, modifiers_bits: u32, time: f64) -> bool {
        let key = match key {
            "ArrowLeft" => Key::Left,
            "ArrowRight" => Key::Right,
            "ArrowUp" => Key::Up,
            "ArrowDown" => Key::Down,
            "+" | "=" => Key::ZoomIn,
            "-" | "_" => Key::ZoomOut,
            _ => return false,
        };
        let modifiers = modifiers(modifiers_bits);
        self.map.input(Input::Key { key, modifiers }, time)
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
        let set = |key: &str, value: JsValue| {
            let _ = js_sys::Reflect::set(&object, &key.into(), &value);
        };
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
            position: lng_lat(&get(camera, "position")).unwrap_or(now.position),
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
            padding: padding(&get(options, "padding"), Padding::default()).unwrap_or_default(),
            offset: pair(&get(options, "offset")).map_or(DVec2::ZERO, DVec2::from),
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
        let set = |key: &str, value: JsValue| {
            let _ = js_sys::Reflect::set(&object, &key.into(), &value);
        };
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

    /// Stops the running animation, as interrupted.
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
