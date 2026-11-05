//! A map with its camera store and gestures, driven one frame at a time.

use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use glam::DVec2;
use orb_camera::{
    AnimationId, Bounds, Camera, CameraEvent, CameraTarget, EaseOptions, FitOptions, FlyOptions,
};
use orb_crs::Geodetic;
use orb_crs::{TileIndex, TileMatrixSet};
use orb_interact::{Button, Feel, Input, InteractEvent, Interaction};
use orb_runtime::{
    CameraState, FrameReport, GridTile, Hit, MapEvent, RenderMode, StyleImage, TargetFormat,
    Transaction, Viewport,
};

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

/// A tile to draw with [`Map::render_tile`].
#[derive(Clone, Debug)]
pub struct TileRequest {
    /// The grid, on the map's plane: its projection must be the scene's.
    pub tile_matrix_set: TileMatrixSet,
    pub z: u8,
    pub col: u32,
    pub row: u32,
    /// Pixels per side, logical; the image is `size × scale` physical
    /// pixels a side, which must be whole.
    pub size: u32,
    pub scale: f64,
}
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
    /// The map's terrain revision the camera stands on.
    terrain: u64,
    /// Set from any thread when data arrived.
    woken: Arc<AtomicBool>,
    /// Notified with `woken`, for still output to wait on.
    signal: Arc<(Mutex<()>, Condvar)>,
    /// What [`render_tile`](Self::render_tile) draws into: the tile with a
    /// tile of its neighbours around it.
    tile_target: Option<Target>,
    hook: WakeHook,
    /// Marks the map woken and calls the hook, from any thread.
    wake: Arc<dyn Fn() + Send + Sync>,
    events: VecDeque<Event>,
    /// Queries finished on worker threads, not yet events.
    finished: Finished,
    next_query: u64,
    format: Option<TargetFormat>,
    /// The runtime's device generation the map's GPU state is of.
    device: u64,
    /// The last frame left work: loads, fades or placement.
    busy: bool,
    /// A command or input since the last frame changed what it shows.
    dirty: bool,
    /// How the map renders now.
    mode: RenderMode,
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("camera", self.camera.state())
            .field("generation", &self.map.generation())
            .finish_non_exhaustive()
    }
}

impl Drop for Map {
    /// Fetches and jobs the map started keep its wake closure, and may
    /// finish after it: the host's callback is dropped first. The closure
    /// holds the hook's mutex while calling it, so this also waits out a
    /// call under way on another thread.
    fn drop(&mut self) {
        let hook = self
            .hook
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        drop(hook);
    }
}

impl Map {
    pub fn new(runtime: &Runtime, options: MapOptions) -> Self {
        let woken = Arc::new(AtomicBool::new(false));
        let signal: Arc<(Mutex<()>, Condvar)> = Arc::default();
        let hook: WakeHook = Arc::default();
        let wake: Arc<dyn Fn() + Send + Sync> = {
            let woken = woken.clone();
            let signal = signal.clone();
            let hook = hook.clone();
            Arc::new(move || {
                woken.store(true, Ordering::Release);
                {
                    let _waiting = signal.0.lock().expect("signal poisoned");
                    signal.1.notify_all();
                }
                if let Some(hook) = hook.lock().expect("wake hook poisoned").as_ref() {
                    hook();
                }
            })
        };
        runtime.add_wake(&wake);
        let mut map = orb_runtime::Map::new(runtime.services(wake.clone()), options.camera);
        // Plan §8's budgets: phones and tablets hold less than desktops.
        if cfg!(any(target_os = "android", target_os = "ios")) {
            map.set_memory_budget(orb_runtime::MOBILE_BUDGET);
        }
        map.set_clear_color(options.clear_color);
        let camera = Camera::new(
            *map.camera(),
            map.surface(),
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
            terrain: 0,
            woken,
            signal,
            tile_target: None,
            hook,
            wake,
            events: VecDeque::new(),
            finished: Arc::default(),
            next_query: 0,
            format: None,
            device: runtime.shared.generation.get(),
            busy: true,
            dirty: true,
            mode: RenderMode::Continuous,
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

    /// Requests the tiles that failed again now: call it when the network
    /// comes back. Until then each is retried after a backoff while frames
    /// are drawn, with its ancestors or children standing in.
    pub fn retry_failed_tiles(&mut self) {
        self.map.retry_failed_tiles();
        self.dirty = true;
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
        self.styled()?;
        self.dirty = true;
        self.map.edit(&tx).map_err(Error::InvalidArgument)
    }

    /// Adds an image to the style for icons, text and patterns (GL JS
    /// `addImage`); an error when it has one by that name.
    pub fn add_image(&mut self, name: &str, image: StyleImage) -> Result<(), Error> {
        self.styled()?;
        self.dirty = true;
        self.map
            .add_image(name, image)
            .map_err(Error::InvalidArgument)
    }

    /// Replaces an image's straight RGBA8 pixels, of the same size (GL JS
    /// `updateImage`).
    pub fn update_image(
        &mut self,
        name: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), Error> {
        self.styled()?;
        self.dirty = true;
        self.map
            .update_image(name, width, height, pixels)
            .map_err(Error::InvalidArgument)
    }

    /// Removes an image from the style (GL JS `removeImage`).
    pub fn remove_image(&mut self, name: &str) -> Result<(), Error> {
        self.styled()?;
        self.dirty = true;
        self.map.remove_image(name).map_err(Error::InvalidArgument)
    }

    /// Whether the style has an image by this name, from its sprites or
    /// added (GL JS `hasImage`).
    pub fn has_image(&self, name: &str) -> bool {
        self.map.has_image(name)
    }

    fn styled(&self) -> Result<(), Error> {
        match self.map.style_doc() {
            Some(_) => Ok(()),
            None => Err(Error::InvalidState("no style is loaded yet".into())),
        }
    }

    /// The style generation, counting loads and edits.
    pub fn generation(&self) -> u64 {
        self.map.generation()
    }

    /// The camera as it is now (it moves during gestures and animations).
    pub fn camera(&self) -> CameraState {
        *self.camera.state()
    }

    /// The view's style zoom on the map's surface (decision 0022).
    pub fn zoom(&self) -> f64 {
        self.camera.zoom()
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

    /// Whether the camera's target stands on the terrain at rest (GL JS
    /// `centerClampedToGround`, on by default). Off, it keeps the
    /// elevation it is given, and the camera orbits a point in the air.
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.camera.set_center_clamped_to_ground(clamped);
        self.map.set_center_clamped_to_ground(clamped);
        self.dirty = true;
    }

    pub fn center_clamped_to_ground(&self) -> bool {
        self.camera.center_clamped_to_ground()
    }

    /// Puts the target this many metres above sea level at once (GL JS
    /// `setCenterElevation`); while clamped, the terrain takes it back.
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.camera.set_center_elevation(elevation);
        self.dirty = true;
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
        self.set_mode(RenderMode::Continuous);
        self.frame(target, true)
    }

    /// Draws what the camera shows once everything in view has loaded and
    /// nothing fades (still output): labels place as if nothing had been
    /// placed before, so the same style, data and camera always draw the
    /// same pixels. Renders as many frames as that takes, waiting for data
    /// in between; an error if it takes longer than `timeout`.
    pub fn render_still(
        &mut self,
        target: &mut Target,
        timeout: Duration,
    ) -> Result<FrameReport, Error> {
        self.set_mode(RenderMode::Still);
        self.settle(target, true, timeout)
    }

    /// Draws one tile of a tile matrix set on the map's plane as still
    /// output, and returns its pixels: RGBA8, premultiplied, rows top to
    /// bottom. The map draws the tile with a tile of its neighbours around
    /// it and places labels in tile scope ([`RenderMode::Tile`]): tiles
    /// drawn apart agree on the labels that cross their edges, those
    /// smaller than half a tile at least, and the same tile always draws
    /// the same bytes. The map's camera store is left as it was.
    pub fn render_tile(&mut self, tile: &TileRequest, timeout: Duration) -> Result<Vec<u8>, Error> {
        let deadline = Instant::now() + timeout;
        let index = TileIndex::new(tile.z, tile.col, tile.row);
        let tms = &tile.tile_matrix_set;
        if !tms.contains(index) {
            return Err(Error::InvalidArgument(format!(
                "{}/{}/{} isn't a tile of {}",
                tile.z,
                tile.col,
                tile.row,
                tms.id()
            )));
        }
        let px = f64::from(tile.size) * tile.scale;
        if tile.size == 0 || px.fract() != 0.0 || px < 1.0 {
            return Err(Error::InvalidArgument(format!(
                "{} pixels at scale {} isn't a whole tile",
                tile.size, tile.scale
            )));
        }
        let px = px as u32;
        let mut target = match self.tile_target.take() {
            Some(t) if t.size() == (3 * px, 3 * px) && t.scale() == tile.scale => t,
            _ => Target::texture(&self.runtime, 3 * px, 3 * px, tile.scale),
        };
        let drawn = self.draw_tile(&mut target, tile, index, deadline);
        let pixels = drawn.and_then(|()| target.read_pixels());
        self.tile_target = Some(target);
        let pixels = pixels?;
        // The middle tile.
        let row_bytes = 3 * px as usize * 4;
        let (start, len) = (px as usize * 4, px as usize * 4);
        Ok((px as usize..2 * px as usize)
            .flat_map(|y| &pixels[y * row_bytes + start..][..len])
            .copied()
            .collect())
    }

    fn draw_tile(
        &mut self,
        target: &mut Target,
        tile: &TileRequest,
        index: TileIndex,
        deadline: Instant,
    ) -> Result<(), Error> {
        // The scene document sets the plane.
        self.set_mode(RenderMode::Still);
        while self.map.style_doc().is_none() {
            self.frame(target, false)?;
            if self.map.style_doc().is_none() {
                self.wait(deadline)?;
            }
        }
        let surface = self.map.surface();
        let tms = &tile.tile_matrix_set;
        if surface.globe.is_some() || !tms.projection().same_plane(&surface.projection) {
            return Err(Error::InvalidArgument(format!(
                "{} isn't on the map's plane: set the scene's projection to its own",
                tms.id()
            )));
        }
        let projection = surface.projection;
        let viewport = Viewport {
            width: f64::from(tile.size),
            height: f64::from(tile.size),
        };
        let world_per_px = tms.tile_span(index.z) / viewport.width;
        let grid = |col: i64, row: i64| -> Option<GridTile> {
            let index = TileIndex::new(index.z, u32::try_from(col).ok()?, u32::try_from(row).ok()?);
            if !tms.contains(index) {
                return None;
            }
            let bounds = tms.tile_bounds(index);
            let center = (bounds.min + bounds.max) / 2.0;
            let target = projection.inverse(center);
            Some(GridTile {
                col,
                row,
                camera: CameraState {
                    target,
                    resolution: world_per_px / projection.scale_factor(target.wrapped()),
                    bearing: 0.0,
                    pitch: 0.0,
                    roll: 0.0,
                    fov: CameraState::DEFAULT_FOV,
                    elevation: 0.0,
                },
                viewport,
                center,
            })
        };
        let (col, row) = (i64::from(index.col), i64::from(index.row));
        let tiles: Vec<GridTile> = (row - 1..=row + 1)
            .flat_map(|r| (col - 1..=col + 1).map(move |c| (c, r)))
            .filter_map(|(c, r)| grid(c, r))
            .collect();
        let camera = grid(col, row).expect("checked above").camera;
        self.map.set_camera(camera);
        // The camera store's camera goes back to the map with the next
        // interactive frame.
        self.pushed = u64::MAX;
        self.set_mode(RenderMode::Tile(tiles));
        self.settle(
            target,
            false,
            deadline.saturating_duration_since(Instant::now()),
        )
        .map(|_| ())
    }

    fn set_mode(&mut self, mode: RenderMode) {
        if self.mode != mode {
            self.map.set_mode(mode.clone());
            self.mode = mode;
            self.dirty = true;
        }
    }

    /// Renders until two frames in a row are idle, the second drawing what
    /// the first's messages brought, waiting for data between frames.
    fn settle(
        &mut self,
        target: &mut Target,
        camera: bool,
        timeout: Duration,
    ) -> Result<FrameReport, Error> {
        let deadline = Instant::now() + timeout;
        let mut was_idle = false;
        loop {
            if let Some(report) = self.frame(target, camera)? {
                if report.idle && was_idle {
                    return Ok(report);
                }
                was_idle = report.idle;
                if was_idle {
                    continue;
                }
            }
            self.wait(deadline)?;
        }
    }

    /// Waits for data, a frame's time at most (fades and transitions run
    /// on the clock), or fails at `deadline`.
    fn wait(&self, deadline: Instant) -> Result<(), Error> {
        let now = Instant::now();
        if now >= deadline {
            return Err(Error::InvalidState(format!(
                "the map didn't settle in time: {:?}",
                self.map.job_counts()
            )));
        }
        let (lock, condvar) = &*self.signal;
        let guard = lock.lock().expect("signal poisoned");
        if !self.woken.load(Ordering::Acquire) {
            let wait = (deadline - now).min(Duration::from_millis(16));
            drop(condvar.wait_timeout(guard, wait).expect("signal poisoned"));
        }
        Ok(())
    }

    /// One frame; with `camera`, gestures and animations advance and the
    /// camera store's camera goes to the map.
    fn frame(&mut self, target: &mut Target, camera: bool) -> Result<Option<FrameReport>, Error> {
        if !target.same_runtime(&self.shared) {
            return Err(Error::InvalidArgument(
                "the target belongs to another runtime".into(),
            ));
        }
        // The runtime replaced its device: draw on the new one, laying the
        // tiles out again (plan §8.8).
        if self.device != self.shared.generation.get() {
            self.device = self.shared.generation.get();
            self.map.reset_gpu();
            self.format = None;
            self.dirty = true;
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
        let (mut gesture, mut animating) = (false, false);
        if camera {
            if self.camera.viewport() != viewport {
                self.camera.set_viewport(viewport);
            }
            let now = self.runtime.now();
            gesture = self.input.frame(&mut self.camera, now);
            animating = self.camera.frame(now);
        }
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
        // After a tile, the camera store's camera goes back to the map.
        let back = self.pushed == u64::MAX;
        if camera && ((self.camera.modified() && self.camera.revision() != self.pushed) || back) {
            self.pushed = self.camera.revision();
            self.map.set_camera(*self.camera.state());
        }
        let Some(texture) = target.begin()? else {
            return Ok(None);
        };
        self.woken.store(false, Ordering::Release);
        if self.shared.sentinel.borrow().is_some() {
            crate::device::before_pipelines();
        }
        let report = self.map.render(
            self.shared.gpu.borrow_mut().gpu(),
            texture,
            format,
            viewport,
            target.scale() as f32,
        );
        target.end();
        // Every pipeline the map asked for compiled, and the driver
        // survived them.
        if report.frame.pipelines_pending == 0 {
            self.runtime.pipelines_made();
        }
        let mut jumped = false;
        for event in self.map.take_events() {
            // Loads and projection edits move the camera's surface.
            if let MapEvent::StyleData { .. } | MapEvent::StyleLoad { .. } = event {
                self.camera.set_surface(self.map.surface());
            }
            if let MapEvent::StyleLoad {
                camera: Some(style),
            } = &event
                && !self.camera.modified()
            {
                self.camera.jump_to(&CameraTarget {
                    center: Some(style.target),
                    zoom: Some(style.zoom(self.camera.projection())),
                    bearing: Some(style.bearing),
                    pitch: Some(style.pitch),
                    roll: Some(style.roll),
                    ..CameraTarget::default()
                });
                jumped = true;
            }
            self.events.push_back(Event::Map(event));
        }
        // The camera stands on the terrain the map has loaded.
        if self.map.elevation_revision() != self.terrain {
            self.terrain = self.map.elevation_revision();
            self.camera.set_terrain(self.map.elevation());
        }
        self.dirty = jumped;
        // A map waiting only for pipelines draws again when one compiles.
        let waiting = report.compiling && self.runtime.compile_wakes();
        self.busy = (!report.idle && !waiting) || gesture || animating;
        if report.idle {
            // A failed write costs the next run its warm start, nothing else.
            let _ = self.runtime.save_pipeline_cache();
        }
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
