//! A map, driven one frame at a time: the runtime map with its wake,
//! still output and the device it draws on.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use glam::DVec2;
use orb_camera::input::{GestureOptions, Input};
use orb_camera::{
    AnimationId, Bounds, CameraTarget, Constraints, EaseOptions, FitOptions, FlyOptions,
    FreeCamera, GestureMomentum, Padding,
};
use orb_crs::Geodetic;
use orb_crs::{TileIndex, TileMatrixSet};
use orb_runtime::{
    CameraState, ClusterQuery, FrameReport, GridTile, Hit, HostTile, MapEvent, RenderMode,
    StyleImage, Transaction, VideoRequest, Viewport,
};

use crate::{Error, Runtime, Shared, Target};

/// How a map starts.
#[derive(Clone, Debug)]
pub struct MapOptions {
    /// The camera; the first style's when `None`, as GL JS does.
    pub camera: Option<CameraState>,
    /// The backing colour behind transparent style pixels, premultiplied
    /// RGBA; opaque surfaces want their view's background.
    pub clear_color: [f64; 4],
    /// How long labels fade in and out, in milliseconds (GL JS
    /// `fadeDuration`); still output doesn't fade.
    pub fade_duration: f64,
    /// Whether labels of different sources collide with each other (GL JS
    /// `crossSourceCollisions`).
    pub cross_source_collisions: bool,
    /// The CSS font family ideographs of the default font stack are drawn
    /// with instead of the style's glyphs (GL JS
    /// `localIdeographFontFamily`); `None` draws the glyphs.
    pub local_ideograph_font_family: Option<String>,
    /// Multisampling for 3D surfaces (GL JS `antialias`, which is off by
    /// default there).
    pub antialias: bool,
    /// Whether terrain tiles hang skirts that hide the cracks between
    /// tiles of different detail; translucent terrain shows them (GL JS
    /// `terrainSkirtLength`).
    pub terrain_skirts: bool,
    /// The previews its styles may use: new style syntax beyond the approved
    /// v8 spec, which warn and fall back to their defaults while off. None
    /// by default.
    pub previews: crate::Previews,
    /// Whether labels keep their place while the camera moves, as
    /// MapLibre Native keeps them: a label tries its previous variable
    /// anchor first (in tilted views, only that anchor), and in tilted views
    /// labels shown before are placed before new and hidden ones. What
    /// labels show then depends on how the camera got there; off, it
    /// depends on the camera alone. On by default; still and tile output
    /// places afresh either way.
    pub stable_labels: bool,
    /// How tiles coarsen with distance in pitched views: MapLibre Native's
    /// tile level of detail (maplibre-native-ffi's tile options), whose
    /// defaults it takes.
    pub tile_lod: orb_camera::TileLod,
    /// What a vector tile the tileset doesn't have (HTTP 404) draws: its
    /// nearest ancestor by default, or nothing, as GL JS and MapLibre
    /// Native draw it. Raster and raster-dem tiles draw their ancestor
    /// either way ([`MissingTiles`](orb_runtime::MissingTiles)).
    pub missing_tiles: orb_runtime::MissingTiles,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            camera: None,
            clear_color: [0.0; 4],
            fade_duration: 300.0,
            cross_source_collisions: true,
            local_ideograph_font_family: Some("sans-serif".into()),
            antialias: true,
            terrain_skirts: true,
            previews: crate::Previews::NONE,
            stable_labels: true,
            tile_lod: orb_camera::TileLod::default(),
            missing_tiles: orb_runtime::MissingTiles::Parent,
        }
    }
}

/// Something a map, its camera or a gesture did.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Map(MapEvent),
    /// A query from [`Map::query_rendered`], [`Map::query_source`] or
    /// [`Map::query_cluster`] finished: a JSON array of GeoJSON features
    /// (rendered ones topmost first) or an expansion zoom, or why it
    /// failed.
    Query {
        id: u64,
        result: Result<String, String>,
    },
}

type WakeHook = Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>;

/// How long still output waits for its data: a minute.
pub const STILL_TIMEOUT: Duration = Duration::from_secs(60);

/// A picture: premultiplied RGBA8, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Image {
    /// The pixels as straight (not premultiplied) RGBA8, for a PNG:
    /// [`orb_runtime::unpremultiply`].
    pub fn into_straight(self) -> Vec<u8> {
        let mut pixels = self.pixels;
        orb_runtime::unpremultiply(&mut pixels);
        pixels
    }
}

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

/// A render node's candidates and those its cull kept
/// ([`Map::cull_counts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CullCount {
    /// The node's family, as `orb-render` names it.
    pub family: String,
    pub candidates: u32,
    pub kept: u32,
}

/// A map: style, camera, gestures and events. Draw it with
/// [`render`](Self::render) whenever [`needs_frame`](Self::needs_frame).
pub struct Map {
    shared: Arc<Shared>,
    runtime: Runtime,
    map: orb_runtime::Map,
    /// Set from any thread when data arrived.
    woken: Arc<AtomicBool>,
    /// Notified with `woken`, for still output to wait on.
    signal: Arc<(Mutex<()>, Condvar)>,
    /// What still output last drew into: a snapshot or a tile.
    offscreen: Option<Target>,
    hook: WakeHook,
    /// Marks the map woken and calls the hook, from any thread.
    wake: Arc<dyn Fn() + Send + Sync>,
    events: VecDeque<Event>,
    /// Queries finished on worker threads, not yet events.
    finished: Finished,
    next_query: u64,
    /// What draws the map, beside its targets: it outlives surfaces, and
    /// a new device gets a new one.
    renderer: Option<orb_runtime::Renderer>,
    /// The runtime's device generation the map's GPU state is of.
    device: u64,
    /// Whether the map has reported that the runtime stopped replacing
    /// lost devices.
    gave_up: bool,
    /// The last frame left work: loads, fades or placement.
    busy: bool,
    /// A command or input since the last frame changed what it shows.
    dirty: bool,
    /// How the map renders now.
    mode: RenderMode,
    /// The last frame's time and when it was drawn: frames the map times
    /// itself (still output, [`render_until_idle`](Self::render_until_idle))
    /// go on from it, on the host's clock.
    last_frame: Option<(f64, Instant)>,
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("camera", self.map.camera())
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
        // What the map drew with goes with it, on the device that made it.
        if let Some(renderer) = self.renderer.take()
            && self.device == self.shared.generation()
        {
            renderer.destroy(&mut *self.shared.gpu());
        }
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
        // Phones and tablets hold less than desktops.
        if cfg!(any(target_os = "android", target_os = "ios")) {
            map.set_memory_budget(orb_runtime::MOBILE_BUDGET);
        }
        map.set_clear_color(options.clear_color);
        map.set_fade_duration(options.fade_duration);
        map.set_cross_source_collisions(options.cross_source_collisions);
        map.set_local_ideograph_font_family(options.local_ideograph_font_family.as_deref());
        map.set_antialias_3d(options.antialias);
        map.set_terrain_skirts(options.terrain_skirts);
        map.set_previews(options.previews);
        map.set_stable_labels(options.stable_labels);
        map.set_tile_lod(options.tile_lod);
        map.set_missing_tiles(options.missing_tiles);
        // Measurements and the fixtures' cull check draw with nothing
        // culled.
        map.set_keep_all_cull(std::env::var_os("ORB_CULL").is_some_and(|v| v == "keep-all"));
        Self {
            shared: runtime.shared.clone(),
            runtime: runtime.clone(),
            map,
            woken,
            signal,
            offscreen: None,
            hook,
            wake,
            events: VecDeque::new(),
            finished: Arc::default(),
            next_query: 0,
            renderer: None,
            device: runtime.shared.generation(),
            gave_up: false,
            busy: true,
            dirty: true,
            mode: RenderMode::Continuous,
            last_frame: None,
        }
    }

    /// Has the cull keep every candidate, culling nothing, from the next
    /// frame (`ORB_CULL=keep-all` sets it for every map): what the cull
    /// saves, for measurements, and that it drops only what draws nothing,
    /// for tests.
    pub fn set_keep_all_cull(&mut self, keep_all: bool) {
        self.map.set_keep_all_cull(keep_all);
        self.dirty = true;
    }

    /// Per render node of the last frame, its family and the candidates
    /// (chunks in tile placements) its cull considered and kept: what the
    /// cull drops. Reads the cull's counters back from the GPU, so it
    /// waits for the frame.
    pub fn cull_counts(&self) -> Vec<CullCount> {
        let Some(renderer) = &self.renderer else {
            return Vec::new();
        };
        let bytes = pollster::block_on(self.shared.gpu().read_buffer(renderer.draw_args().0));
        (renderer.cull_counts(&bytes).into_iter())
            .map(|(family, candidates, kept)| CullCount {
                family: format!("{family:?}"),
                candidates,
                kept,
            })
            .collect()
    }

    /// Calls `wake` from any thread when data arrives and the map needs a
    /// frame that it didn't need before: hosts that stop drawing while the
    /// map is idle resume on it. Commands and input on the host's thread
    /// don't call it; check [`needs_frame`](Self::needs_frame) after them.
    pub fn set_wake(&mut self, wake: impl Fn() + Send + Sync + 'static) {
        *self.hook.lock().expect("wake hook poisoned") = Some(Box::new(wake));
    }

    /// Moves to `camera` at once, exactly, as [`jump_to`](Self::jump_to)
    /// moves to a target: for hosts that hold exact cameras, such as
    /// benches and fixtures.
    pub fn jump_to_state(&mut self, camera: CameraState) {
        self.map.jump_to_state(camera);
    }

    /// Per label text, the greatest opacity it drew with in the last
    /// frame: for benches that watch labels across gestures.
    pub fn shown_labels(&self) -> std::collections::HashMap<Box<str>, f32> {
        self.map.shown_labels()
    }

    /// The style as v8 JSON, with every edit applied and the global state
    /// as its defaults, so that loading it gives this map's style; `None`
    /// before a style loads.
    pub fn style_json(&self) -> Option<String> {
        self.map.style_doc().map(|doc| doc.to_json())
    }

    /// Whether a paint transition runs.
    pub fn transitioning(&self) -> bool {
        self.map.transitioning()
    }

    /// Whether an animation, a glide or a gesture moves the camera.
    pub fn is_moving(&self) -> bool {
        self.map.is_moving()
    }

    /// Whether drawing a frame would show something new: data arrived, the
    /// camera moves, a gesture or animation runs, or loads and fades are
    /// under way. An idle map needs no frames.
    pub fn needs_frame(&self) -> bool {
        self.dirty || self.busy || self.woken.load(Ordering::Acquire) || self.map.is_moving()
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

    /// Applies a transaction's edits all at once, or none if one fails,
    /// and returns the generation from which frames show them. Build it op
    /// by op, or from GL JS map calls as JSON with
    /// [`Transaction::from_commands`] (`[["setPaintProperty", "water",
    /// "fill-color", "red"], …]`).
    pub fn edit(&mut self, tx: &Transaction) -> Result<u64, Error> {
        self.styled()?;
        self.dirty = true;
        self.map.edit(tx).map_err(Error::InvalidArgument)
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

    /// Whether this host plays `video` sources: it then gets
    /// [`MapEvent::Video`] requests and hands frames to
    /// [`source_frame`](Self::source_frame). Without it, video sources draw
    /// nothing, with a warning.
    pub fn set_plays_video(&mut self, plays: bool) {
        self.map.set_plays_video(plays);
    }

    /// Sets how tiles coarsen with distance in pitched views (MapLibre
    /// Native's tile level of detail, as maplibre-native-ffi's
    /// `mln_map_set_tile_options` takes it); the next frame covers with it.
    pub fn set_tile_lod(&mut self, lod: orb_camera::TileLod) {
        self.map.set_tile_lod(lod);
        self.dirty = true;
    }

    /// How tiles coarsen with distance, values out of range brought in.
    pub fn tile_lod(&self) -> orb_camera::TileLod {
        self.map.tile_lod()
    }

    /// The picture the video or canvas source `id` shows now: straight
    /// RGBA8 rows, top first. It replaces any frame not yet drawn.
    pub fn source_frame(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<(), Error> {
        if pixels.len() != width as usize * height as usize * 4 {
            return Err(Error::InvalidArgument(
                "pixels aren't width × height RGBA8".into(),
            ));
        }
        self.dirty = true;
        let frame = orb_runtime::TileImage {
            width,
            height,
            pixels: orb_runtime::ImagePixels::Core(pixels),
        };
        self.map
            .source_frame(id, frame)
            .map_err(Error::InvalidArgument)
    }

    /// The app's answer for `tile` of source `id`, a vector source it
    /// feeds (`"tiles": ["orbis-host:"]`), as a [`MapEvent::TileRequest`]
    /// asked: the tile encoded, its layers as GeoJSON, or empty.
    pub fn host_tile(&mut self, id: &str, tile: TileIndex, answer: HostTile) -> Result<(), Error> {
        self.dirty = true;
        self.map
            .host_tile(id, tile, answer)
            .map_err(Error::InvalidArgument)
    }

    /// Asks for every tile of the app's source `id` again, as its data
    /// changed; drawn tiles stay until the new ones replace them.
    pub fn reload_host_tiles(&mut self, id: &str) -> Result<(), Error> {
        self.dirty = true;
        self.map
            .reload_host_tiles(id)
            .map_err(Error::InvalidArgument)
    }

    /// The video of source `id` can't play on this host: it draws nothing.
    pub fn video_failed(&mut self, id: &str, message: &str) -> Result<(), Error> {
        self.dirty = true;
        self.map
            .video_failed(id, message)
            .map_err(Error::InvalidArgument)
    }

    /// Asks the host to play, pause or seek the video of source `id`, as a
    /// [`MapEvent::Video`] (GL JS's `VideoSource` `play`, `pause`, `seek`).
    pub fn control_video(&mut self, id: &str, request: VideoRequest) -> Result<(), Error> {
        // The request goes out with the next frame's events.
        self.dirty = true;
        self.map
            .control_video(id, request)
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

    /// The loaded style as it is now, edits included: its JSON, its
    /// layers' and sources' ids and each one's JSON (GL JS `getStyle`,
    /// `getLayer`, `getSource`); `None` before a style loads.
    pub fn style(&self) -> Option<&orb_runtime::StyleDoc> {
        self.map.style_doc()
    }

    /// The style generation, counting loads and edits.
    pub fn generation(&self) -> u64 {
        self.map.generation()
    }

    /// The camera as it is now (it moves during gestures and animations).
    pub fn camera(&self) -> CameraState {
        *self.map.camera()
    }

    /// The view's style zoom on the map's surface.
    pub fn zoom(&self) -> f64 {
        self.map.zoom()
    }

    /// The viewport the camera and gestures use: the last target's, in
    /// logical pixels.
    pub fn viewport(&self) -> Viewport {
        self.map.viewport()
    }

    /// Sets the viewport, in logical pixels, before a target draws (each
    /// frame sets its target's): fits, projections and constraints made
    /// before the first frame use the size the map will have.
    pub fn set_viewport(&mut self, width: f64, height: f64) {
        self.map.set_viewport(Viewport { width, height });
        self.dirty = true;
    }

    /// Moves the camera at once, ending any animation.
    pub fn jump_to(&mut self, target: &CameraTarget) {
        self.map.jump_to(target);
        self.dirty = true;
    }

    /// Animates the camera from the next frame on (GL JS `easeTo`).
    pub fn ease_to(&mut self, target: &CameraTarget, options: &EaseOptions) -> AnimationId {
        self.dirty = true;
        self.map.ease_to(target, *options)
    }

    /// Flies along GL JS's zoom-out curve from the next frame on.
    pub fn fly_to(&mut self, target: &CameraTarget, options: &FlyOptions) -> AnimationId {
        self.dirty = true;
        self.map.fly_to(target, options)
    }

    /// Continues a gesture the host recognized itself: each part of
    /// `momentum` glides on from its release velocity and slows by its
    /// decay, from the next frame on, then settles on its snap (mln-ffi's
    /// `mln_gesture_momentum`). Any input or camera move stops it; its
    /// [`CameraEvent::AnimationEnd`](crate::CameraEvent::AnimationEnd) then reports the speed it had left.
    /// `None` when nothing moves or settles over time. [`input`](Self::input)
    /// gestures end with one of their own.
    pub fn start_momentum(&mut self, momentum: &GestureMomentum) -> Option<AnimationId> {
        self.dirty = true;
        self.map.start_momentum(momentum)
    }

    /// Whether the camera's target stands on the terrain at rest (GL JS
    /// `centerClampedToGround`, on by default). Off, it keeps the
    /// elevation it is given, and the camera orbits a point in the air.
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.map.set_center_clamped_to_ground(clamped);
        self.dirty = true;
    }

    pub fn center_clamped_to_ground(&self) -> bool {
        self.map.center_clamped_to_ground()
    }

    /// Puts the target this many metres above sea level at once (GL JS
    /// `setCenterElevation`); while clamped, the terrain takes it back.
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.map.set_center_elevation(elevation);
        self.dirty = true;
    }

    /// The camera's eye and orientation now (GL JS
    /// `getFreeCameraOptions`).
    pub fn free_camera(&self) -> FreeCamera {
        self.map.free_camera()
    }

    /// Puts the eye at `free` at once (GL JS `setFreeCameraOptions`),
    /// ending any animation; the field of view and padding stay.
    pub fn jump_to_free(&mut self, free: &FreeCamera) {
        self.map.jump_to_free(free);
        self.dirty = true;
    }

    /// The camera that shows `bounds` in the current viewport, inside the
    /// camera's padding.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        self.map.camera_for_bounds(bounds, options)
    }

    /// Stops any camera animation where it is.
    pub fn stop(&mut self) {
        self.map.stop();
    }

    /// Zoom and pitch ranges and bounds the camera keeps to.
    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.map.set_constraints(constraints);
        self.dirty = true;
    }

    /// Hands the map a pointer, wheel or key input at `time`, the event's
    /// timestamp in milliseconds on the clock frame times use, in logical
    /// pixels from the target's top-left corner. Returns whether the map
    /// used it.
    pub fn input(&mut self, input: Input, time: f64) -> bool {
        let used = self.map.input(input, time);
        self.dirty |= used;
        used
    }

    /// What gestures may move, how releases continue, and the platform's
    /// thresholds; changing it cancels a gesture under way.
    pub fn set_gestures(&mut self, options: GestureOptions) {
        self.map.set_gestures(options);
        self.dirty = true;
    }

    pub fn gestures(&self) -> &GestureOptions {
        self.map.gestures()
    }

    /// The box a Shift-drag zooms to while it is drawn, corner to corner
    /// in logical pixels: hosts show it.
    pub fn box_zoom(&self) -> Option<[DVec2; 2]> {
        self.map.box_zoom()
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
        let done = self.query_done();
        self.map.query_rendered(min, max, None, done.1);
        done.0
    }

    /// GL JS `querySourceFeatures`: the features of `source` in the tiles
    /// the last frame drew, of `source_layer` (vector sources) or every
    /// layer, that a v8 `filter` as JSON selects (`["within", polygon]`
    /// for an area); a feature crossing tiles comes once a tile. They
    /// arrive as an [`Event::Query`] with the returned id.
    pub fn query_source(
        &mut self,
        source: &str,
        source_layer: Option<&str>,
        filter: Option<&str>,
    ) -> u64 {
        let done = self.query_done();
        self.map.query_source(source, source_layer, filter, done.1);
        done.0
    }

    /// GL JS's cluster queries on a clustered GeoJSON source: a cluster's
    /// children or leaves, as a JSON array of GeoJSON features, or its
    /// expansion zoom, as a number. The answer arrives as an
    /// [`Event::Query`] with the returned id.
    pub fn query_cluster(&mut self, source: &str, query: ClusterQuery) -> u64 {
        let done = self.query_done();
        self.map.query_cluster(source, query, done.1);
        done.0
    }

    /// A new query's id, and what reports its answer as an event.
    fn query_done(&mut self) -> (u64, orb_runtime::QueryDone) {
        self.next_query += 1;
        let id = self.next_query;
        let finished = self.finished.clone();
        let wake = self.wake.clone();
        let done = Box::new(move |result| {
            finished
                .lock()
                .expect("queries poisoned")
                .push((id, result));
            wake();
        });
        (id, done)
    }

    /// Where a point is in the viewport, in logical pixels; `None` when
    /// it's behind the camera.
    pub fn project(&self, point: Geodetic) -> Option<DVec2> {
        self.map.project(point)
    }

    /// The point under a viewport position in logical pixels; `None` above
    /// the horizon.
    pub fn unproject(&self, point: DVec2) -> Option<Geodetic> {
        self.map.unproject(point)
    }

    /// [`project`](Self::project) for the camera a jump to `camera` would
    /// make, in this map's viewport, on its surface and terrain: where a
    /// move would put a point, before making it.
    pub fn project_with(&self, camera: &CameraTarget, point: Geodetic) -> Option<DVec2> {
        self.map.project_with(&self.map.camera_for(camera), point)
    }

    /// [`unproject`](Self::unproject) for the camera a jump to `camera`
    /// would make.
    pub fn unproject_with(&self, camera: &CameraTarget, point: DVec2) -> Option<Geodetic> {
        self.map.unproject_with(&self.map.camera_for(camera), point)
    }

    /// Draws the frame shown at `frame_time` (milliseconds, on the clock
    /// input times use: the display link's): applies arrived data,
    /// advances gestures and animations to it and renders into `target`.
    /// Every frame of a map uses targets of one format (all surfaces of a
    /// platform share one). Returns `None` when the surface had nothing to
    /// draw into this time.
    pub fn render(
        &mut self,
        target: &mut Target,
        frame_time: f64,
    ) -> Result<Option<FrameReport>, Error> {
        self.set_mode(RenderMode::Continuous);
        self.frame(target, frame_time)
    }

    /// Draws frames into `target` until the map is idle (its loads,
    /// pipelines, fades and camera motion done, and a last frame drawing
    /// what the one before brought), waiting for data between frames, and
    /// returns the last frame's report. Frames are timed on the runtime's
    /// clock. For tools and tests: hosts draw on their display link. An
    /// error after `timeout`.
    pub fn render_until_idle(
        &mut self,
        target: &mut Target,
        timeout: Duration,
    ) -> Result<FrameReport, Error> {
        self.set_mode(RenderMode::Continuous);
        self.settle(target, Instant::now() + timeout)
    }

    /// Draws what the camera shows in a `width × height` view (logical
    /// pixels, `scale` physical ones each) once everything in it has loaded
    /// and nothing fades, and returns the picture (still output): labels
    /// place as if nothing had been placed before, so the same style, data
    /// and camera always draw the same bytes. Renders as many frames as that
    /// takes, waiting for data in between; an error after [`STILL_TIMEOUT`].
    pub fn snapshot(&mut self, width: u32, height: u32, scale: f64) -> Result<Image, Error> {
        let deadline = Instant::now() + STILL_TIMEOUT;
        let px = |v: u32| (f64::from(v) * scale).round() as u32;
        let mut target = self.offscreen(px(width), px(height), scale);
        self.set_mode(RenderMode::Still);
        let pixels = self
            .settle(&mut target, deadline)
            .and_then(|_| target.read_pixels());
        let (width, height) = target.size();
        self.offscreen = Some(target);
        Ok(Image {
            width,
            height,
            pixels: pixels?,
        })
    }

    /// Draws one tile of a tile matrix set on the map's plane as still
    /// output, and returns its picture. The map loads the labels of the
    /// tile's neighbours too and places them all in tile scope
    /// ([`RenderMode::Tile`]): tiles drawn apart agree on the labels that
    /// cross their edges, those smaller than half a tile at least, and the
    /// same tile always draws the same bytes. The map's camera is left as it
    /// was. An error after [`STILL_TIMEOUT`].
    pub fn render_tile(&mut self, tile: &TileRequest) -> Result<Image, Error> {
        let deadline = Instant::now() + STILL_TIMEOUT;
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
        let mut target = self.offscreen(px, px, tile.scale);
        let drawn = self.draw_tile(&mut target, tile, index, deadline);
        let pixels = drawn.and_then(|()| target.read_pixels());
        self.offscreen = Some(target);
        Ok(Image {
            width: px,
            height: px,
            pixels: pixels?,
        })
    }

    /// Applies what has arrived, the style among it, without drawing: a
    /// host with nothing to draw into yet (a map before its first target)
    /// still gets its style in and its events ([`MapEvent::StyleLoad`]),
    /// and the requests the style starts go out. Hosts that draw frames
    /// needn't call it. Frames into targets of another format than still
    /// output's (an RGBA8 texture) fail after it, as after
    /// [`snapshot`](Self::snapshot).
    pub fn apply_arrivals(&mut self) -> Result<(), Error> {
        let mut target = self.offscreen(1, 1, 1.0);
        let applied = self.arrive(&mut target);
        self.offscreen = Some(target);
        applied
    }

    /// The texture still output draws into, of this size.
    fn offscreen(&mut self, width: u32, height: u32, scale: f64) -> Target {
        match self.offscreen.take() {
            Some(t) if t.size() == (width, height) && t.scale() == scale => t,
            _ => Target::texture(&self.runtime, width, height, scale),
        }
    }

    fn draw_tile(
        &mut self,
        target: &mut Target,
        tile: &TileRequest,
        index: TileIndex,
        deadline: Instant,
    ) -> Result<(), Error> {
        // The style's projection sets the plane. Until it is in, nothing
        // draws: a frame would load what the interactive camera shows.
        self.set_mode(RenderMode::Still);
        while self.map.style_doc().is_none() {
            self.arrive(target)?;
            self.style_failed()?;
            if self.map.style_doc().is_none() {
                self.wait(deadline)?;
            }
        }
        let surface = self.map.surface();
        let tms = &tile.tile_matrix_set;
        if surface.globe.is_some() || !tms.projection().same_plane(&surface.projection) {
            return Err(Error::InvalidArgument(format!(
                "{} isn't on the map's plane: set the style's projection to its own",
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
                    padding: Padding::default(),
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
        // The middle tile as the middle of a view of the whole grid sees
        // it: from three times as far, so that what stands up (extrusions)
        // leans at the tile's edges as it does at its neighbours'.
        let camera = CameraState {
            fov: 2.0
                * ((CameraState::DEFAULT_FOV / 2.0).to_radians().tan() / 3.0)
                    .atan()
                    .to_degrees(),
            ..grid(col, row).expect("checked above").camera
        };
        let interactive = *self.map.camera();
        self.map.jump_to_state(camera);
        self.set_mode(RenderMode::Tile(tiles));
        let drawn = self.settle(target, deadline).map(|_| ());
        self.map.jump_to_state(interactive);
        drawn
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
    fn settle(&mut self, target: &mut Target, deadline: Instant) -> Result<FrameReport, Error> {
        let mut was_idle = false;
        loop {
            let report = self.frame(target, self.now())?;
            // A style that failed to load never settles: say why at once.
            self.style_failed()?;
            if let Some(report) = report {
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

    /// The latest style request's failure, as an error.
    fn style_failed(&self) -> Result<(), Error> {
        match self.map.style_error() {
            Some(message) => Err(Error::InvalidArgument(message.to_string())),
            None => Ok(()),
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

    /// The time of a frame the map times itself: the host's last frame's
    /// time and the wall time since, so the map's frames keep one clock.
    fn now(&self) -> f64 {
        self.last_frame
            .map_or(0.0, |(time, at)| time + at.elapsed().as_secs_f64() * 1000.0)
    }

    /// One frame, shown at `frame_time`.
    fn frame(
        &mut self,
        target: &mut Target,
        frame_time: f64,
    ) -> Result<Option<FrameReport>, Error> {
        self.last_frame = Some((frame_time, Instant::now()));
        if !self.ready_for(target)? {
            return Ok(None);
        }
        let format = target.format();
        let viewport = target.viewport();
        // The frame borrows the device once: the target begins and ends
        // with it, and it's let go before anything else here borrows it.
        let report = {
            let mut device = self.shared.gpu();
            let Some(texture) = target.begin(&mut device)? else {
                return Ok(None);
            };
            device.start_frame();
            self.woken.store(false, Ordering::Release);
            let draws = orb_hal::Gpu::caps(&*device).gpu_driven();
            let gpu: &mut dyn orb_hal::Gpu = &mut *device;
            let map = &mut self.map;
            let report = if draws {
                let renderer = self
                    .renderer
                    .get_or_insert_with(|| map.create_renderer(gpu, format));
                map.render(
                    gpu,
                    renderer,
                    texture,
                    viewport,
                    target.scale() as f32,
                    frame_time,
                )
            } else {
                // GLES: a flat colour until the CPU frame driver draws (#140).
                orb_hal_wgpu::boot_frame(gpu, texture);
                orb_runtime::FrameReport {
                    idle: true,
                    ..Default::default()
                }
            };
            target.end(&mut device);
            device.finish_frame();
            report
        };
        // The frame found the device lost (a compile the watchdog found
        // hung, a lost context): the next one makes another.
        let lost = self.runtime.lost();
        self.events
            .extend(self.map.take_events().into_iter().map(Event::Map));
        self.dirty = lost;
        // A map waiting only for pipelines draws again when one compiles.
        let waiting = report.compiling;
        self.busy = !report.idle && !waiting;
        if report.idle {
            // A failed write costs the next run its warm start, nothing else.
            let _ = self.runtime.save_pipeline_cache();
        }
        Ok(Some(report))
    }

    /// Checks that this map can draw into `target`, after bringing it to
    /// the runtime's current device: `false` while a lost device's
    /// replacement waits out its backoff (the map is woken when it's due).
    fn ready_for(&mut self, target: &mut Target) -> Result<bool, Error> {
        if !target.same_runtime(&self.shared) {
            return Err(Error::InvalidArgument(
                "the target belongs to another runtime".into(),
            ));
        }
        // A lost GPU is replaced before the frame: texture
        // targets are made again, and window targets fail with
        // `TargetLost` for the host to make on the new one. After too
        // many losses in a row the runtime stops, and the map says why
        // once, as an error event, besides failing each frame.
        if self.runtime.lost() {
            let recovered = self.runtime.recovery_due().and_then(|due| {
                if due {
                    if target.is_window() {
                        target.release();
                    }
                    self.runtime.recover()?;
                }
                Ok(due)
            });
            match recovered {
                Ok(true) => {}
                Ok(false) => {
                    // Nothing draws until the backoff's timer wakes the
                    // map: a host asking `needs_frame` doesn't spin.
                    self.dirty = false;
                    self.busy = false;
                    self.woken.store(false, Ordering::Release);
                    return Ok(false);
                }
                Err(e) => {
                    if !std::mem::replace(&mut self.gave_up, true) {
                        self.events.push_back(Event::Map(MapEvent::Error {
                            message: e.message().to_owned(),
                        }));
                    }
                    return Err(e);
                }
            }
        }
        // The runtime replaced its device: draw on the new one, laying the
        // tiles out again.
        if self.device != self.shared.generation() {
            self.device = self.shared.generation();
            self.renderer = None;
            self.dirty = true;
        }
        let format = target.format();
        if let Some(renderer) = &self.renderer
            && renderer.format() != format
        {
            return Err(Error::InvalidArgument(format!(
                "this map draws {:?} targets, not {:?}",
                renderer.format().color,
                format.color
            )));
        }
        Ok(true)
    }

    /// Applies what has arrived without drawing a frame
    /// ([`orb_runtime::Map::apply_arrivals`]), with the renderer of
    /// `target`'s format.
    fn arrive(&mut self, target: &mut Target) -> Result<(), Error> {
        if !self.ready_for(target)? {
            return Ok(());
        }
        let format = target.format();
        {
            let mut device = self.shared.gpu();
            self.woken.store(false, Ordering::Release);
            if !orb_hal::Gpu::caps(&*device).gpu_driven() {
                return Ok(());
            }
            let gpu: &mut dyn orb_hal::Gpu = &mut *device;
            let map = &mut self.map;
            let renderer = self
                .renderer
                .get_or_insert_with(|| map.create_renderer(gpu, format));
            map.apply_arrivals(gpu, renderer, target.scale() as f32);
        }
        self.events
            .extend(self.map.take_events().into_iter().map(Event::Map));
        Ok(())
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
}
