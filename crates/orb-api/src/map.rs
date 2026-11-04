//! A map with its camera store and gestures, driven one frame at a time.

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_camera::{
    AnimationId, Bounds, Camera, CameraEvent, CameraTarget, EaseOptions, FitOptions, FlyOptions,
};
use orb_geo::Geodetic;
use orb_interact::{Button, Feel, Input, InteractEvent, Interaction};
use orb_runtime::{CameraState, FrameReport, Hit, MapEvent, TargetFormat, Transaction, Viewport};

use crate::{Error, Runtime, Shared, Target};

/// How a map starts.
#[derive(Clone, Debug)]
pub struct MapOptions {
    /// The camera; the first style's when `None`, as GL JS does.
    pub camera: Option<CameraState>,
    /// How gestures feel.
    pub feel: Feel,
    /// The backing colour behind transparent style pixels, premultiplied
    /// RGBA; opaque surfaces want their view's background.
    pub clear_color: [f64; 4],
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            camera: None,
            feel: Feel::default(),
            clear_color: [0.0; 4],
        }
    }
}

/// Something a map, its camera or a gesture did.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Map(MapEvent),
    Camera(CameraEvent),
    /// A press and release that didn't move the map, in logical pixels.
    Click {
        position: DVec2,
        button: Button,
    },
    DoubleClick {
        position: DVec2,
    },
    /// A query from [`Map::query_rendered`] finished: a JSON array of
    /// GeoJSON features, topmost first, or why it failed.
    Query {
        id: u64,
        result: Result<String, String>,
    },
}

type WakeHook = Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>;
type Finished = Arc<Mutex<Vec<(u64, Result<String, String>)>>>;

/// A map: style, camera store, gestures and events. Draw it with
/// [`render`](Self::render) whenever [`needs_frame`](Self::needs_frame).
pub struct Map {
    shared: Rc<Shared>,
    runtime: Runtime,
    map: orb_runtime::Map,
    camera: Camera,
    input: Interaction,
    /// The camera revision last given to the map.
    pushed: u64,
    /// Set from any thread when data arrived.
    woken: Arc<AtomicBool>,
    hook: WakeHook,
    /// Marks the map woken and calls the hook, from any thread.
    wake: Arc<dyn Fn() + Send + Sync>,
    events: VecDeque<Event>,
    /// Queries finished on worker threads, not yet events.
    finished: Finished,
    next_query: u64,
    format: Option<TargetFormat>,
    /// The last frame left work: loads, fades or placement.
    busy: bool,
    /// A command or input since the last frame changed what it shows.
    dirty: bool,
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("camera", self.camera.state())
            .field("generation", &self.map.generation())
            .finish_non_exhaustive()
    }
}

impl Map {
    pub fn new(runtime: &Runtime, options: MapOptions) -> Self {
        let woken = Arc::new(AtomicBool::new(false));
        let hook: WakeHook = Arc::default();
        let wake: Arc<dyn Fn() + Send + Sync> = {
            let woken = woken.clone();
            let hook = hook.clone();
            Arc::new(move || {
                woken.store(true, Ordering::Release);
                if let Some(hook) = hook.lock().expect("wake hook poisoned").as_ref() {
                    hook();
                }
            })
        };
        let mut map = orb_runtime::Map::new(runtime.services(wake.clone()), options.camera);
        map.set_clear_color(options.clear_color);
        let camera = Camera::new(
            *map.camera(),
            map.projection(),
            Viewport {
                width: 512.0,
                height: 512.0,
            },
        );
        Self {
            shared: runtime.shared.clone(),
            runtime: runtime.clone(),
            map,
            camera,
            input: Interaction::new(options.feel),
            pushed: 0,
            woken,
            hook,
            wake,
            events: VecDeque::new(),
            finished: Arc::default(),
            next_query: 0,
            format: None,
            busy: true,
            dirty: true,
        }
    }

    /// Calls `wake` from any thread when data arrives and the map needs a
    /// frame that it didn't need before: hosts that stop drawing while the
    /// map is idle resume on it. Commands and input on the host's thread
    /// don't call it; check [`needs_frame`](Self::needs_frame) after them.
    pub fn set_wake(&mut self, wake: impl Fn() + Send + Sync + 'static) {
        *self.hook.lock().expect("wake hook poisoned") = Some(Box::new(wake));
    }

    /// Whether drawing a frame would show something new: data arrived, the
    /// camera moves, a gesture or animation runs, or loads and fades are
    /// under way. An idle map needs no frames.
    pub fn needs_frame(&self) -> bool {
        self.dirty
            || self.busy
            || self.woken.load(Ordering::Acquire)
            || self.camera.is_animating()
            || self.input.is_active()
    }

    /// Loads a style from a URL (`http(s)://`, `file://`, `pmtiles://`).
    pub fn load_style_url(&mut self, url: &str) {
        self.map.load_style_url(url);
        self.dirty = true;
    }

    /// Replaces the style with a v8 style document.
    pub fn set_style_json(&mut self, json: &str) -> Result<(), Error> {
        self.dirty = true;
        self.map
            .set_style_json(json)
            .map_err(Error::InvalidArgument)
    }

    /// Applies a transaction of GL JS map calls as JSON (`[["setPaintProperty",
    /// "water", "fill-color", "red"], …]`) all at once, and returns the
    /// generation from which frames show it.
    pub fn edit(&mut self, commands: &str) -> Result<u64, Error> {
        let tx = Transaction::from_commands(commands).map_err(Error::InvalidArgument)?;
        if self.map.style_doc().is_none() {
            return Err(Error::InvalidState("no style is loaded yet".into()));
        }
        self.dirty = true;
        self.map.edit(&tx).map_err(Error::InvalidArgument)
    }

    /// The style generation, counting loads and edits.
    pub fn generation(&self) -> u64 {
        self.map.generation()
    }

    /// The camera as it is now (it moves during gestures and animations).
    pub fn camera(&self) -> CameraState {
        *self.camera.state()
    }

    /// The viewport the camera and gestures use: the last target's, in
    /// logical pixels.
    pub fn viewport(&self) -> Viewport {
        self.camera.viewport()
    }

    /// Moves the camera at once, ending any animation.
    pub fn jump_to(&mut self, target: &CameraTarget) {
        self.camera.jump_to(target);
        self.dirty = true;
    }

    pub fn ease_to(&mut self, target: &CameraTarget, options: &EaseOptions) -> AnimationId {
        self.dirty = true;
        self.camera.ease_to(target, *options, self.runtime.now())
    }

    pub fn fly_to(&mut self, target: &CameraTarget, options: &FlyOptions) -> AnimationId {
        self.dirty = true;
        self.camera.fly_to(target, options, self.runtime.now())
    }

    /// The camera that shows `bounds` in the current viewport.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        self.camera.camera_for_bounds(bounds, options)
    }

    /// Stops any camera animation where it is.
    pub fn stop(&mut self) {
        self.camera.stop();
    }

    /// Hands the map a pointer, wheel or key input, in logical pixels from
    /// the target's top-left corner. Returns whether the map used it.
    pub fn input(&mut self, input: Input) -> bool {
        let used = self
            .input
            .handle(&mut self.camera, input, self.runtime.now());
        self.dirty |= used;
        used
    }

    /// The topmost feature at a point in logical pixels, from the last
    /// frame drawn.
    pub fn hit_test(&self, point: DVec2) -> Option<Hit> {
        self.map.hit_test(point)
    }

    /// GL JS `queryRenderedFeatures` over the box `min..max` in logical
    /// pixels (a point when they're equal), in every layer of the last
    /// frame drawn. The features arrive as an [`Event::Query`] with the
    /// returned id; the map wakes when they do.
    pub fn query_rendered(&mut self, min: DVec2, max: DVec2) -> u64 {
        self.next_query += 1;
        let id = self.next_query;
        let finished = self.finished.clone();
        let wake = self.wake.clone();
        self.map.query_rendered(
            min,
            max,
            None,
            Box::new(move |result| {
                finished
                    .lock()
                    .expect("queries poisoned")
                    .push((id, result));
                wake();
            }),
        );
        id
    }

    /// Where a point is in the viewport, in logical pixels; `None` when
    /// it's behind the camera.
    pub fn project(&self, point: Geodetic) -> Option<DVec2> {
        self.camera.project(point)
    }

    /// The point under a viewport position in logical pixels; `None` above
    /// the horizon.
    pub fn unproject(&self, point: DVec2) -> Option<Geodetic> {
        self.camera.unproject(point)
    }

    /// Draws a frame: advances gestures and animations, applies arrived
    /// data and renders into `target`. Every frame of a map uses targets of
    /// one format (all surfaces of a platform share one). Returns `None`
    /// when the surface had nothing to draw into this time.
    pub fn render(&mut self, target: &mut Target) -> Result<Option<FrameReport>, Error> {
        if !target.same_runtime(&self.shared) {
            return Err(Error::InvalidArgument(
                "the target belongs to another runtime".into(),
            ));
        }
        let format = target.format();
        if *self.format.get_or_insert(format) != format {
            return Err(Error::InvalidArgument(format!(
                "this map draws {:?} targets, not {:?}",
                self.format.expect("set above").color,
                format.color
            )));
        }
        let viewport = target.viewport();
        if self.camera.viewport() != viewport {
            self.camera.set_viewport(viewport);
        }
        let now = self.runtime.now();
        let gesture = self.input.frame(&mut self.camera, now);
        let animating = self.camera.frame(now);
        self.events
            .extend(self.camera.take_events().into_iter().map(Event::Camera));
        self.events.extend(
            self.input
                .take_events()
                .into_iter()
                .map(|event| match event {
                    InteractEvent::Click { position, button } => Event::Click { position, button },
                    InteractEvent::DoubleClick { position } => Event::DoubleClick { position },
                }),
        );
        // Until the app or the user moves it, the map's camera follows the
        // style.
        if self.camera.modified() && self.camera.revision() != self.pushed {
            self.pushed = self.camera.revision();
            self.map.set_camera(*self.camera.state());
        }
        let Some(texture) = target.begin()? else {
            return Ok(None);
        };
        self.woken.store(false, Ordering::Release);
        let report = self.map.render(
            &mut *self.shared.gpu.borrow_mut(),
            texture,
            format,
            viewport,
            target.scale() as f32,
        );
        target.end();
        let mut jumped = false;
        for event in self.map.take_events() {
            if let MapEvent::StyleLoad { .. } = event {
                self.camera.set_projection(self.map.projection());
            }
            if let MapEvent::StyleLoad {
                camera: Some(style),
            } = &event
                && !self.camera.modified()
            {
                self.camera.jump_to(&CameraTarget {
                    center: Some(style.target),
                    zoom: Some(style.zoom()),
                    bearing: Some(style.bearing),
                    pitch: Some(style.pitch),
                    ..CameraTarget::default()
                });
                jumped = true;
            }
            self.events.push_back(Event::Map(event));
        }
        self.dirty = jumped;
        self.busy = !report.idle || gesture || animating;
        Ok(Some(report))
    }

    /// The oldest event not yet taken.
    pub fn poll_event(&mut self) -> Option<Event> {
        let finished = std::mem::take(&mut *self.finished.lock().expect("queries poisoned"));
        self.events.extend(
            finished
                .into_iter()
                .map(|(id, result)| Event::Query { id, result }),
        );
        self.events.pop_front()
    }

    /// Warnings about the style and its data since the last call.
    pub fn take_warnings(&mut self) -> Vec<String> {
        self.map.take_warnings()
    }

    /// The Rust map underneath, for what this API doesn't cover yet.
    pub fn inner(&self) -> &orb_runtime::Map {
        &self.map
    }
}
