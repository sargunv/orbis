//! The ui role (plan §7.1): the page's camera authority. It takes input,
//! runs gestures and camera animations, answers `project`, `unproject` and
//! `hitTest` synchronously, and hands the camera to the map once a frame,
//! on this thread or in the render worker.
//!
//! Cameras cross as `[lon, lat, resolution, bearing, pitch, fov, zoom,
//! elevation]`, elevation optional;
//! camera targets as `[lon, lat, zoom, bearing, pitch, aroundLon,
//! aroundLat, offsetX, offsetY]` with NaN for what stays.

use glam::DVec2;
use orb_camera::{
    Bounds, Camera, CameraEvent, CameraState, CameraTarget, Change, Constraints, EaseOptions,
    Easing, FitOptions, FlyOptions, Padding, Phase,
};
use orb_geo::Geodetic;
use orb_interact::{
    Button, Feel, InteractEvent, Interaction, Key, Modifiers, PointerKind, WheelUnit,
};
use orb_runtime::{HitIndex, Viewport};
use wasm_bindgen::prelude::*;

pub(crate) fn camera_array(camera: &CameraState) -> Vec<f64> {
    vec![
        camera.target.lon,
        camera.target.lat,
        camera.resolution,
        camera.bearing,
        camera.pitch,
        camera.fov,
        camera.zoom(),
        camera.elevation,
    ]
}

pub(crate) fn camera_from(words: &[f64]) -> Result<CameraState, JsError> {
    match *words {
        [lon, lat, resolution, bearing, pitch, fov, ..] => Ok(CameraState {
            target: Geodetic::new(lon, lat),
            resolution,
            bearing,
            pitch,
            fov,
            elevation: words.get(7).copied().unwrap_or(0.0),
        }),
        _ => Err(JsError::new("a camera has 6 numbers or more")),
    }
}

fn set(object: &js_sys::Object, key: &str, value: impl Into<JsValue>) {
    let _ = js_sys::Reflect::set(object, &key.into(), &value.into());
}

fn known(v: f64) -> Option<f64> {
    (!v.is_nan()).then_some(v)
}

fn target_from(words: &[f64]) -> Result<CameraTarget, JsError> {
    let [
        lon,
        lat,
        zoom,
        bearing,
        pitch,
        around_lon,
        around_lat,
        offset_x,
        offset_y,
    ] = *words
    else {
        return Err(JsError::new("a camera target has 9 numbers"));
    };
    let point = |lon: f64, lat: f64| {
        known(lon)
            .zip(known(lat))
            .map(|(lon, lat)| Geodetic::new(lon, lat))
    };
    Ok(CameraTarget {
        center: point(lon, lat),
        zoom: known(zoom),
        bearing: known(bearing),
        pitch: known(pitch),
        around: point(around_lon, around_lat),
        offset: DVec2::new(
            known(offset_x).unwrap_or(0.0),
            known(offset_y).unwrap_or(0.0),
        ),
    })
}

fn target_array(target: &CameraTarget) -> Vec<f64> {
    let (lon, lat) = target
        .center
        .map_or((f64::NAN, f64::NAN), |c| (c.lon, c.lat));
    let (alon, alat) = target
        .around
        .map_or((f64::NAN, f64::NAN), |c| (c.lon, c.lat));
    vec![
        lon,
        lat,
        target.zoom.unwrap_or(f64::NAN),
        target.bearing.unwrap_or(f64::NAN),
        target.pitch.unwrap_or(f64::NAN),
        alon,
        alat,
        target.offset.x,
        target.offset.y,
    ]
}

fn modifiers(bits: u32) -> Modifiers {
    Modifiers {
        shift: bits & 1 != 0,
        ctrl: bits & 2 != 0,
        alt: bits & 4 != 0,
        meta: bits & 8 != 0,
    }
}

fn easing(words: Option<Vec<f64>>) -> Easing {
    match words.as_deref() {
        Some(&[x1, y1, x2, y2]) => Easing::Bezier([x1, y1, x2, y2]),
        Some([]) => Easing::Linear,
        _ => Easing::DEFAULT,
    }
}

/// The page's camera, gestures and published picking index.
#[wasm_bindgen]
#[derive(Debug)]
pub struct UiRole {
    camera: Camera,
    input: Interaction,
    hits: HitIndex,
}

#[wasm_bindgen]
impl UiRole {
    /// A camera for a viewport of `width` × `height` CSS pixels, at the
    /// map's starting camera.
    #[wasm_bindgen(constructor)]
    pub fn new(width: f64, height: f64) -> UiRole {
        let tms = orb_crs::TileMatrixSet::by_id("WebMercatorQuad").expect("built-in");
        UiRole {
            camera: Camera::new(
                CameraState::new(Geodetic::new(0.0, 0.0), 0.0),
                *tms.projection(),
                Viewport { width, height },
            ),
            input: Interaction::default(),
            hits: HitIndex::default(),
        }
    }

    pub fn resize(&mut self, width: f64, height: f64) {
        self.camera.set_viewport(Viewport { width, height });
    }

    /// `"desktop"`, `"ios"` or `"android"`.
    #[wasm_bindgen(js_name = setFeel)]
    pub fn set_feel(&mut self, name: &str) -> Result<(), JsError> {
        self.input.set_feel(match name {
            "desktop" => Feel::DESKTOP,
            "ios" => Feel::IOS,
            "android" => Feel::ANDROID,
            _ => return Err(JsError::new("feel is desktop, ios or android")),
        });
        Ok(())
    }

    /// Zoom and pitch ranges, and `[west, south, east, north]` bounds or
    /// none.
    #[wasm_bindgen(js_name = setConstraints)]
    pub fn set_constraints(
        &mut self,
        min_zoom: f64,
        max_zoom: f64,
        min_pitch: f64,
        max_pitch: f64,
        bounds: Option<Vec<f64>>,
    ) {
        self.camera.set_constraints(Constraints {
            min_zoom,
            max_zoom,
            min_pitch,
            max_pitch,
            bounds: bounds.as_deref().and_then(|b| match *b {
                [west, south, east, north] => Some(Bounds {
                    west,
                    south,
                    east,
                    north,
                }),
                _ => None,
            }),
        });
    }

    /// A pointer pressed: `kind` is the DOM `pointerType`, `button` the DOM
    /// button, `modifiers` bits shift 1, ctrl 2, alt 4, meta 8. Returns
    /// whether the map used it.
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
        now: f64,
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
        let input = orb_interact::Input::PointerDown {
            id,
            kind,
            button,
            position: DVec2::new(x, y),
            modifiers: modifiers(modifiers_bits),
        };
        self.input.handle(&mut self.camera, input, now)
    }

    #[wasm_bindgen(js_name = pointerMove)]
    pub fn pointer_move(&mut self, id: u32, x: f64, y: f64, now: f64) -> bool {
        let input = orb_interact::Input::PointerMove {
            id,
            position: DVec2::new(x, y),
        };
        self.input.handle(&mut self.camera, input, now)
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, id: u32, x: f64, y: f64, now: f64) -> bool {
        let input = orb_interact::Input::PointerUp {
            id,
            position: DVec2::new(x, y),
        };
        self.input.handle(&mut self.camera, input, now)
    }

    #[wasm_bindgen(js_name = pointerCancel)]
    pub fn pointer_cancel(&mut self, id: u32, now: f64) -> bool {
        self.input.handle(
            &mut self.camera,
            orb_interact::Input::PointerCancel { id },
            now,
        )
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
        now: f64,
    ) -> bool {
        let unit = match mode {
            1 => WheelUnit::Line,
            2 => WheelUnit::Page,
            _ => WheelUnit::Pixel,
        };
        let input = orb_interact::Input::Wheel {
            position: DVec2::new(x, y),
            delta: DVec2::new(dx, dy),
            unit,
            modifiers: modifiers(modifiers_bits),
        };
        self.input.handle(&mut self.camera, input, now)
    }

    /// A DOM `keydown`'s `key`; returns whether the map used it.
    pub fn key(&mut self, key: &str, modifiers_bits: u32, now: f64) -> bool {
        let key = match key {
            "ArrowLeft" => Key::Left,
            "ArrowRight" => Key::Right,
            "ArrowUp" => Key::Up,
            "ArrowDown" => Key::Down,
            "+" | "=" => Key::ZoomIn,
            "-" | "_" => Key::ZoomOut,
            _ => return false,
        };
        let input = orb_interact::Input::Key {
            key,
            modifiers: modifiers(modifiers_bits),
        };
        self.input.handle(&mut self.camera, input, now)
    }

    pub fn blur(&mut self, now: f64) {
        self.input
            .handle(&mut self.camera, orb_interact::Input::Blur, now);
    }

    /// Advances gestures and animations to `now`; returns whether they
    /// need more frames.
    pub fn frame(&mut self, now: f64) -> bool {
        let gesture = self.input.frame(&mut self.camera, now);
        self.camera.frame(now) || gesture
    }

    /// Camera and gesture events since the last call, as plain objects:
    /// `{type: "movestart" … "pitchend", user}`, `{type: "animationend",
    /// id, interrupted}`, `{type: "click", x, y, button}`, `{type:
    /// "dblclick", x, y}`.
    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&mut self) -> js_sys::Array {
        let out = js_sys::Array::new();
        for event in self.camera.take_events() {
            let object = js_sys::Object::new();
            match event {
                CameraEvent::Changed {
                    change,
                    phase,
                    user,
                } => {
                    let name = match change {
                        Change::Move => "move",
                        Change::Zoom => "zoom",
                        Change::Rotate => "rotate",
                        Change::Pitch => "pitch",
                    };
                    let suffix = match phase {
                        Phase::Start => "start",
                        Phase::Progress => "",
                        Phase::End => "end",
                    };
                    set(&object, "type", format!("{name}{suffix}"));
                    set(&object, "user", user);
                }
                CameraEvent::AnimationEnd { id, interrupted } => {
                    set(&object, "type", "animationend");
                    set(&object, "id", id.0 as f64);
                    set(&object, "interrupted", interrupted);
                }
            }
            out.push(&object);
        }
        for event in self.input.take_events() {
            let object = js_sys::Object::new();
            match event {
                InteractEvent::Click { position, button } => {
                    set(&object, "type", "click");
                    set(&object, "x", position.x);
                    set(&object, "y", position.y);
                    set(
                        &object,
                        "button",
                        match button {
                            Button::Primary => 0,
                            Button::Middle => 1,
                            Button::Secondary => 2,
                        },
                    );
                }
                InteractEvent::DoubleClick { position } => {
                    set(&object, "type", "dblclick");
                    set(&object, "x", position.x);
                    set(&object, "y", position.y);
                }
            }
            out.push(&object);
        }
        out
    }

    /// The camera: `[lon, lat, resolution, bearing, pitch, fov, zoom]`.
    pub fn camera(&self) -> Vec<f64> {
        camera_array(self.camera.state())
    }

    /// Counts camera changes, so hosts send the camera only when it moved.
    pub fn revision(&self) -> f64 {
        self.camera.revision() as f64
    }

    /// The app or the user has moved the camera; until then the map's
    /// camera follows its style.
    pub fn modified(&self) -> bool {
        self.camera.modified()
    }

    /// Takes a style's camera if nothing moved the camera yet.
    /// Moves the camera onto the map's surface: the `projection` of a
    /// `styleload` event.
    #[wasm_bindgen(js_name = setProjection)]
    pub fn set_projection(&mut self, projection: &str) -> Result<(), JsError> {
        let projection =
            orb_runtime::projection_from_json(projection).map_err(|e| JsError::new(&e))?;
        self.camera.set_projection(projection);
        Ok(())
    }

    #[wasm_bindgen(js_name = adoptStyleCamera)]
    pub fn adopt_style_camera(&mut self, camera: &[f64]) -> Result<bool, JsError> {
        if self.camera.modified() {
            return Ok(false);
        }
        let style = camera_from(camera)?;
        self.camera.jump_to(&CameraTarget {
            center: Some(style.target),
            zoom: Some(style.zoom()),
            bearing: Some(style.bearing),
            pitch: Some(style.pitch),
            ..CameraTarget::default()
        });
        Ok(true)
    }

    #[wasm_bindgen(js_name = jumpTo)]
    pub fn jump_to(&mut self, target: &[f64]) -> Result<(), JsError> {
        self.camera.jump_to(&target_from(target)?);
        Ok(())
    }

    /// GL JS `easeTo`; `easing` is a cubic Bézier's four numbers, empty for
    /// linear, or none for GL JS's default. Returns the animation's id.
    #[wasm_bindgen(js_name = easeTo)]
    pub fn ease_to(
        &mut self,
        target: &[f64],
        duration: f64,
        easing_words: Option<Vec<f64>>,
        now: f64,
    ) -> Result<f64, JsError> {
        let options = EaseOptions {
            duration,
            easing: easing(easing_words),
        };
        Ok(self.camera.ease_to(&target_from(target)?, options, now).0 as f64)
    }

    /// GL JS `flyTo`; `options` is `[curve, speed, screenSpeed, duration,
    /// maxDuration, minZoom]`, NaN for defaults.
    #[wasm_bindgen(js_name = flyTo)]
    pub fn fly_to(
        &mut self,
        target: &[f64],
        options: &[f64],
        easing_words: Option<Vec<f64>>,
        now: f64,
    ) -> Result<f64, JsError> {
        let defaults = FlyOptions::default();
        let at = |i: usize| options.get(i).copied().and_then(known);
        let fly = FlyOptions {
            curve: at(0).unwrap_or(defaults.curve),
            speed: at(1).unwrap_or(defaults.speed),
            screen_speed: at(2),
            duration: at(3),
            max_duration: at(4),
            min_zoom: at(5),
            easing: easing(easing_words),
        };
        Ok(self.camera.fly_to(&target_from(target)?, &fly, now).0 as f64)
    }

    /// GL JS `cameraForBounds`: `bounds` `[west, south, east, north]`,
    /// `options` `[top, right, bottom, left, offsetX, offsetY, maxZoom,
    /// bearing, pitch]` with NaN for defaults. Returns a camera target, or
    /// none when the padding leaves no room.
    #[wasm_bindgen(js_name = cameraForBounds)]
    pub fn camera_for_bounds(&self, bounds: &[f64], options: &[f64]) -> Option<Vec<f64>> {
        let [west, south, east, north] = *bounds else {
            return None;
        };
        let at = |i: usize| options.get(i).copied().and_then(known);
        let fit = FitOptions {
            padding: Padding {
                top: at(0).unwrap_or(0.0),
                right: at(1).unwrap_or(0.0),
                bottom: at(2).unwrap_or(0.0),
                left: at(3).unwrap_or(0.0),
            },
            offset: DVec2::new(at(4).unwrap_or(0.0), at(5).unwrap_or(0.0)),
            max_zoom: at(6),
            bearing: at(7),
            pitch: at(8),
        };
        let bounds = Bounds {
            west,
            south,
            east,
            north,
        };
        self.camera
            .camera_for_bounds(bounds, &fit)
            .map(|t| target_array(&t))
    }

    /// Stops the running animation, as interrupted.
    pub fn stop(&mut self) {
        self.camera.stop();
    }

    #[wasm_bindgen(js_name = isMoving)]
    pub fn is_moving(&self) -> bool {
        self.camera.is_animating() || self.camera.is_held()
    }

    /// The CSS pixel showing a point, or none behind the eye.
    pub fn project(&self, lon: f64, lat: f64) -> Option<Vec<f64>> {
        self.camera
            .project(Geodetic::new(lon, lat))
            .map(|p| vec![p.x, p.y])
    }

    /// The ground point under a CSS pixel, or none above the horizon.
    pub fn unproject(&self, x: f64, y: f64) -> Option<Vec<f64>> {
        self.camera
            .unproject(DVec2::new(x, y))
            .map(|g| vec![g.lon, g.lat])
    }

    /// Projects `[lon, lat, …]` pairs with a given camera (a frame ack's,
    /// so DOM overlays move with the frame on screen) into `[x, y, …]`, NaN
    /// behind the eye.
    #[wasm_bindgen(js_name = projectWith)]
    pub fn project_with(&self, camera: &[f64], points: &[f64]) -> Result<Vec<f64>, JsError> {
        let state = camera_from(camera)?;
        let projection = *self.camera.projection();
        let view = orb_camera::View::new(&state, self.camera.viewport(), &projection);
        let near = view.camera_world();
        let period = projection.wrap_period();
        let mut out = Vec::with_capacity(points.len());
        for &[lon, lat] in points.as_chunks::<2>().0 {
            let mut world = projection.forward(Geodetic::new(lon, lat));
            if let Some(period) = period {
                world.x -= ((world.x - near.x) / period).round() * period;
            }
            let p = view.project(world).unwrap_or(DVec2::NAN);
            out.extend([p.x, p.y]);
        }
        Ok(out)
    }

    /// Stands the camera on the terrain of a frame acknowledgement's
    /// elevation pyramid (empty: no terrain). Projections then go through
    /// the pyramid, approximately (decision 0019).
    #[wasm_bindgen(js_name = updateElevation)]
    pub fn update_elevation(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        let terrain = if bytes.is_empty() {
            None
        } else {
            let pyramid =
                orb_runtime::TerrainElevation::from_pyramid(bytes).map_err(|e| JsError::new(&e))?;
            Some(std::sync::Arc::new(pyramid) as std::sync::Arc<dyn orb_camera::Elevation>)
        };
        self.camera.set_terrain(terrain);
        Ok(())
    }

    /// Takes a picking snapshot from a frame acknowledgement.
    #[wasm_bindgen(js_name = updateHits)]
    pub fn update_hits(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        self.hits.update(bytes).map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = advanceHits)]
    pub fn advance_hits(&mut self, generation: f64) {
        self.hits.set_generation(generation as u64);
    }

    /// The topmost feature under a CSS pixel in the published snapshot,
    /// mapped through the ground when the camera has moved since.
    #[wasm_bindgen(js_name = hitTest)]
    pub fn hit_test(&self, x: f64, y: f64) -> JsValue {
        crate::hit_object(self.hits.hit_test_from(
            DVec2::new(x, y),
            self.camera.state(),
            self.camera.viewport(),
            self.camera.projection(),
        ))
    }
}
