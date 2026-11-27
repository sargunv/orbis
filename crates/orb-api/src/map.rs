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
    CameraState, ClusterQuery, FeatureTarget, FrameReport, GridTile, Hit, HostTile, MapEvent,
    MapOptions, RenderMode, StyleImage, Transaction, Value, VideoRequest, Viewport,
};

use crate::{Activity, Error, GestureMotion, Move, Runtime, Skew, Target};
use commanded::Commanded;

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

/// How long still output waits with nothing arriving before it gives up:
/// a minute without a wake (no fetch, job, placement or compile coming
/// back). Work that is only slow, on a loaded machine, keeps arriving and
/// never reaches it; a stall does.
pub const STILL_STALL: Duration = Duration::from_secs(60);

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

/// A map: style, camera, gestures and events. Draw it with
/// [`render`](Self::render) whenever [`needs_frame`](Self::needs_frame).
pub struct Map {
    runtime: Runtime,
    map: Commanded,
    /// Set from any thread when data arrived.
    woken: Arc<AtomicBool>,
    /// Set from any thread when the device woke the maps: a pipeline
    /// compiled, or a lost device's replacement is due. News only to a
    /// map whose last frame waited for that.
    device_woken: Arc<AtomicBool>,
    /// Calls the hook for the device's wakes: the runtime holds it weakly.
    _device_wake: Arc<dyn Fn() + Send + Sync>,
    /// Where the last frame, or the last try at one, left the map.
    shown: Shown,
    /// Notified with `woken` and `device_woken`, for still output to
    /// wait on, counting the wakes of each: what still output takes as
    /// progress.
    signal: Arc<(Mutex<Wakes>, Condvar)>,
    /// What still output last drew into: a snapshot or a tile.
    offscreen: Option<Target>,
    /// The format of the target the map's last frame began in, which its
    /// renderer draws in and still output draws into where it's 8-bit.
    format: orb_runtime::TargetFormat,
    /// The pixel ratio the host declared ([`Map::set_viewport`]) or its
    /// last live frame drew at, which a map applying arrivals without a
    /// target loads its sprites at.
    scale: f64,
    hook: WakeHook,
    /// Marks the map woken and calls the hook, from any thread.
    wake: Arc<dyn Fn() + Send + Sync>,
    /// Wakes the map when the data a tile it draws expires, or a failed
    /// tile or glyph range is due to be tried again.
    alarm: crate::alarm::Alarm,
    events: VecDeque<Event>,
    /// Queries finished on worker threads, not yet events.
    finished: Finished,
    next_query: u64,
    /// What draws the map, beside its targets: it outlives surfaces, and
    /// a new device gets a new one.
    renderer: Option<orb_runtime::Renderer>,
    /// The runtime's device generation the map's GPU state is of.
    device: u64,
    /// The last frame's time and when it was drawn: frames the map times
    /// itself (still output, [`render_until_idle`](Self::render_until_idle))
    /// go on from it, on the host's clock.
    last_frame: Option<(f64, Instant)>,
    /// The tiles the last frame drawn waited for, for still output to name
    /// when it stalls.
    tiles_loading: u32,
}

/// Where a map's last frame, or its last try at one, left it: what of
/// the device's wakes and the map's own work is news to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    /// A frame drew: what kept it changing, and whether it asked for
    /// another of its own accord ([`FrameReport::needs_frame`]). Waiting
    /// only for pipelines, it draws again when one compiles.
    Drew { activity: Activity, busy: bool },
    /// The device was lost and its replacement waits out a backoff: the
    /// device's wake when it's due is the only news.
    AwaitingDevice,
    /// The runtime stopped replacing lost devices: the map never draws
    /// again, and has said why once as an error event.
    GaveUp,
}

impl Shown {
    /// No frame drawn yet on the device: as one still loading.
    const FRESH: Self = Self::Drew {
        activity: Activity::Loading,
        busy: true,
    };
}

/// Waits for a map's wakes ([`Map::waiter`]).
#[derive(Clone, Debug)]
pub struct Waiter {
    signal: Arc<(Mutex<Wakes>, Condvar)>,
    woken: Arc<AtomicBool>,
    device_woken: Arc<AtomicBool>,
}

impl Waiter {
    /// Blocks until the map is woken or `timeout` passes, as
    /// [`Map::wait_woken`].
    pub fn wait(&self, timeout: Duration) {
        let (lock, condvar) = &*self.signal;
        let guard = lock.lock().expect("signal poisoned");
        if !self.woken.load(Ordering::Acquire) && !self.device_woken.load(Ordering::Acquire) {
            drop(
                condvar
                    .wait_timeout(guard, timeout)
                    .expect("signal poisoned"),
            );
        }
    }
}

/// A map's wakes so far: its own (its fetches, jobs, placements and
/// alarm) and the device's, which every map on the runtime shares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Wakes {
    own: u64,
    device: u64,
}

/// When still output gives up: once its stall passes with no new wake of
/// its own, however long the map takes while its data keeps arriving.
struct Patience {
    stall: Duration,
    /// The map's wakes so far, as last seen.
    seen: Wakes,
    /// When the map was last seen to make progress, or the wait began.
    since: Instant,
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
            && self.device == self.runtime.shared.generation()
            && let Some(mut device) = self.runtime.shared.opened()
        {
            renderer.destroy(&mut *device);
        }
    }
}

/// A wake and the flag it sets: called from any thread, it sets the flag,
/// counts itself in `signal` (in the count `count` picks) and notifies its
/// waiters, and calls the host's hook.
fn wake_setting(
    signal: &Arc<(Mutex<Wakes>, Condvar)>,
    hook: &WakeHook,
    count: fn(&mut Wakes) -> &mut u64,
) -> (Arc<AtomicBool>, Arc<dyn Fn() + Send + Sync>) {
    let flag = Arc::new(AtomicBool::new(false));
    let (woken, signal, hook) = (flag.clone(), signal.clone(), hook.clone());
    let wake = Arc::new(move || {
        woken.store(true, Ordering::Release);
        {
            *count(&mut signal.0.lock().expect("signal poisoned")) += 1;
            signal.1.notify_all();
        }
        if let Some(hook) = hook.lock().expect("wake hook poisoned").as_ref() {
            hook();
        }
    });
    (flag, wake)
}

impl Map {
    pub fn new(runtime: &Runtime, options: MapOptions) -> Self {
        let signal: Arc<(Mutex<Wakes>, Condvar)> = Arc::default();
        let hook: WakeHook = Arc::default();
        let (woken, wake) = wake_setting(&signal, &hook, |w| &mut w.own);
        let (device_woken, device_wake) = wake_setting(&signal, &hook, |w| &mut w.device);
        runtime.add_wake(&device_wake);
        let clock = orb_host_native::NativeClock::default();
        let services = runtime.services(clock, wake.clone());
        let mut map = orb_runtime::Map::new(services, options.camera);
        // Phones and tablets do less a frame than desktops.
        if cfg!(any(target_os = "android", target_os = "ios")) {
            map.set_frame_budget(orb_runtime::Budget::MOBILE);
        }
        map.set_options(&options);
        Self {
            runtime: runtime.clone(),
            map: Commanded::new(map),
            woken,
            device_woken,
            _device_wake: device_wake,
            shown: Shown::FRESH,
            signal,
            offscreen: None,
            format: crate::target::DEFAULT_FORMAT,
            scale: 1.0,
            hook,
            alarm: crate::alarm::Alarm::new(clock, wake.clone()),
            wake,
            events: VecDeque::new(),
            finished: Arc::default(),
            next_query: 0,
            renderer: None,
            device: runtime.shared.generation(),
            last_frame: None,
            tiles_loading: 0,
        }
    }

    /// Draws frames upside down into their targets when `flipped`, as
    /// maplibre-native-ffi's flipped viewport asks, still output too; off
    /// by default. The picture is the upright one mirrored top to bottom:
    /// the camera, gestures and every viewport position stay the right way
    /// up. The next frame draws so.
    pub fn set_flip_y(&mut self, flipped: bool) {
        self.map.command().set_flip_y(flipped);
    }

    /// The terrain as the last frame drew it, for measuring how still its
    /// cover holds ([`orb_runtime::Map::terrain_probe`]).
    pub fn terrain_probe(&self) -> Option<orb_runtime::TerrainProbe> {
        self.map.terrain_probe(self.map.viewport())
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
        self.map.command().jump_to_state(camera);
    }

    /// Per label text, the greatest opacity it drew with in the last
    /// frame: for benches that watch labels across gestures.
    pub fn shown_labels(&self) -> std::collections::HashMap<Box<str>, f32> {
        self.map.shown_labels().into_iter().collect()
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

    /// Whether an animation or a glide moves the camera on its own: a jump
    /// that ends the last one makes it false at once, and reports the
    /// move's end as it does.
    pub fn is_animating(&self) -> bool {
        self.map.is_animating()
    }

    /// Whether the map wants a frame, one that would show something new:
    /// data arrived, the camera moves, a gesture or animation runs, loads
    /// and fades are under way, a pipeline the last frame waited for
    /// compiled, or the device was lost or replaced. While a lost device's
    /// replacement waits out its backoff only the device's wake when it's
    /// due is news, and once the runtime has stopped replacing lost devices
    /// nothing is, as no frame can draw: the frame that found it failed
    /// with the reason. An idle map needs no frames.
    pub fn needs_frame(&self) -> bool {
        match self.shown {
            Shown::GaveUp => false,
            Shown::AwaitingDevice => self.device_news(),
            Shown::Drew { busy, .. } => {
                self.map.changed()
                    || busy
                    || self.woken.load(Ordering::Acquire)
                    || self.map.is_moving()
                    || self.device_news()
            }
        }
    }

    /// The device's wakes that are news to this map: a compile or a lost
    /// device's replacement its last frame waited for, a loss (the
    /// watchdog's, which a frame must find to recover from), or another
    /// map's replacing the device its picture was on, which went with it.
    fn device_news(&self) -> bool {
        let waiting = matches!(
            self.shown,
            Shown::AwaitingDevice
                | Shown::Drew {
                    activity: Activity::Compiling,
                    ..
                }
        );
        (self.renderer.is_some() && self.device != self.runtime.shared.generation())
            || (self.device_woken.load(Ordering::Acquire) && (waiting || self.runtime.lost()))
    }

    /// As [`needs_frame`](Self::needs_frame), but for the camera: data arrived,
    /// or loads, fades or style transitions go on (MapLibre Native's
    /// `needsRepaint`). A moving camera alone leaves it false.
    pub fn needs_repaint(&self) -> bool {
        match self.shown {
            Shown::GaveUp => false,
            Shown::AwaitingDevice => self.device_news(),
            Shown::Drew { activity, .. } => {
                self.map.changed()
                    || self.woken.load(Ordering::Acquire)
                    || self.device_news()
                    || !matches!(activity, Activity::Idle | Activity::Compiling)
            }
        }
    }

    /// Requests the tiles that failed again now: call it when the network
    /// comes back. Until then each is retried after a backoff, the map
    /// woken for it, with its ancestors or children standing in.
    pub fn retry_failed_tiles(&mut self) {
        self.map.command().retry_failed_tiles();
    }

    /// Whether tiles that leave the view stay cached (the default) or go
    /// at the next frame ([`orb_runtime::Map::set_tile_cache`]).
    pub fn set_tile_cache(&mut self, enabled: bool) {
        self.map.command().set_tile_cache(enabled);
    }

    /// Lets go of the tiles cached out of view at the next frame; they
    /// load again when the view needs them. For low-memory warnings.
    pub fn release_cached_tiles(&mut self) {
        self.map.command().release_cached_tiles();
    }

    /// Drops every loaded tile, which load again for the next frames.
    pub fn clear_tiles(&mut self) {
        self.map.command().clear_tiles();
    }

    /// Gives up the tiles still loading, aborting their fetches: the next
    /// frame that wants them loads them again. For a map that stops
    /// drawing for now, so that it stops using the network.
    pub fn cancel_tile_loads(&mut self) {
        self.map.command().cancel_tile_loads();
    }

    /// Loads a style from a URL (`http(s)://`, `file://`, `pmtiles://`).
    pub fn load_style_url(&mut self, url: &str) {
        self.map.command().load_style_url(url);
    }

    /// Replaces the style with a v8 style document.
    pub fn set_style_json(&mut self, json: &str) -> Result<(), Error> {
        self.map
            .command()
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
        self.map.command().edit(tx).map_err(Error::InvalidArgument)
    }

    /// The state of `target`'s feature (GL JS `getFeatureState`): empty
    /// when it has none, or `target` names no feature.
    pub fn feature_state(&self, target: &FeatureTarget) -> &[(String, Value<'static>)] {
        self.map.feature_state(target)
    }

    /// Merges `state` into the state of `target`'s feature, as MapLibre
    /// Native's `setFeatureState`: whatever its source, which the style
    /// may add later. The state lasts until it's removed, its source
    /// leaves the style, or a style loads (a style URL's arrival too, even
    /// over the same style); state set before any style reads back until
    /// the first style loads.
    /// Returns whether it changed anything; a target without an id sets
    /// nothing. A transaction's `setFeatureState` sets the same state but
    /// checks its target as GL JS does.
    pub fn set_feature_state(
        &mut self,
        target: &FeatureTarget,
        state: &[(String, Value<'static>)],
    ) -> bool {
        self.map
            .command_if(|map| map.set_feature_state(target, state))
    }

    /// Removes `key` of the state of `target`'s feature, or its whole
    /// state; without an id, the whole state of every feature of its
    /// source layer (or of its source's features without one), whatever
    /// `key`, as MapLibre Native does. Returns whether it removed anything.
    pub fn remove_feature_state(&mut self, target: &FeatureTarget, key: Option<&str>) -> bool {
        self.map
            .command_if(|map| map.remove_feature_state(target, key))
    }

    /// Adds an image to the style for icons, text and patterns (GL JS
    /// `addImage`); an error when it has one by that name.
    pub fn add_image(&mut self, name: &str, image: StyleImage) -> Result<(), Error> {
        self.styled()?;
        self.map
            .command()
            .add_image(name, image)
            .map_err(Error::InvalidArgument)
    }

    /// Whether this host plays `video` sources: it then gets
    /// [`MapEvent::Video`] requests and hands frames to
    /// [`source_frame`](Self::source_frame). Without it, video sources draw
    /// nothing, with a warning.
    pub fn set_plays_video(&mut self, plays: bool) {
        self.map.command().set_plays_video(plays);
    }

    /// Sets the colour (premultiplied RGBA) the map draws on, which shows
    /// where nothing in the style covers it, as [`MapOptions::clear_color`]
    /// set it; the next frame draws on it.
    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.map.command().set_clear_color(color);
    }

    /// Sets how tiles coarsen with distance in pitched views (MapLibre
    /// Native's tile level of detail, as maplibre-native-ffi's
    /// `mln_map_set_tile_options` takes it); the next frame covers with it.
    pub fn set_tile_lod(&mut self, lod: orb_camera::TileLod) {
        self.map.command().set_tile_lod(lod);
    }

    /// The options last set, the tile LOD brought into range; those that
    /// apply only to later styles or tiles say so on [`MapOptions`]. Their
    /// camera is `None` ([`Map::camera`] has the map's).
    pub fn options(&self) -> &MapOptions {
        self.map.options()
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
        let frame = orb_runtime::TileImage {
            width,
            height,
            pixels: orb_runtime::ImagePixels::Core(pixels),
        };
        self.map
            .command()
            .source_frame(id, frame)
            .map_err(Error::InvalidArgument)
    }

    /// The app's answer for `tile` of source `id`, a vector source it
    /// feeds (`"tiles": ["orbis-host:"]`), as a [`MapEvent::TileRequest`]
    /// asked: the tile encoded, its layers as GeoJSON, or empty.
    pub fn host_tile(&mut self, id: &str, tile: TileIndex, answer: HostTile) -> Result<(), Error> {
        self.map
            .command()
            .host_tile(id, tile, answer)
            .map_err(Error::InvalidArgument)
    }

    /// Heights in metres for tiles of source `id`, a `raster-dem` source
    /// the app feeds (`"tiles": ["orbis-host:"]`): per tile, `side`² of
    /// them, rows top to bottom. One call is one step of the app's ground:
    /// it answers the tiles a [`MapEvent::TileRequest`] asked for, and
    /// writes over the tiles drawn in place, showing them all in one frame
    /// ([`MapEvent::ElevationsShown`]), terrain, hillshade and the
    /// elevations the camera and labels read together. A call before the
    /// last one shows replaces it. Says which tiles the map doesn't hold.
    pub fn set_host_elevations(
        &mut self,
        id: &str,
        side: u32,
        tiles: Vec<(TileIndex, Vec<f32>)>,
    ) -> Result<orb_runtime::HostElevations, Error> {
        self.map
            .command()
            .set_host_elevations(id, side, tiles)
            .map_err(Error::InvalidArgument)
    }

    /// Asks for every tile of the app's source `id` again, as its data
    /// changed; drawn tiles stay until the new ones replace them.
    pub fn reload_host_tiles(&mut self, id: &str) -> Result<(), Error> {
        self.map
            .command()
            .reload_host_tiles(id)
            .map_err(Error::InvalidArgument)
    }

    /// [`reload_host_tiles`](Self::reload_host_tiles) for the tiles `which`
    /// picks: a drawn tile, or the data tile it draws (an ancestor), and
    /// the kept answer of a data tile.
    pub fn reload_host_tiles_where(
        &mut self,
        id: &str,
        which: &dyn Fn(TileIndex) -> bool,
    ) -> Result<(), Error> {
        self.map
            .command()
            .reload_host_tiles_where(id, which)
            .map_err(Error::InvalidArgument)
    }

    /// The video of source `id` can't play on this host: it draws nothing.
    pub fn video_failed(&mut self, id: &str, message: &str) -> Result<(), Error> {
        self.map
            .command()
            .video_failed(id, message)
            .map_err(Error::InvalidArgument)
    }

    /// Asks the host to play, pause or seek the video of source `id`, as a
    /// [`MapEvent::Video`] (GL JS's `VideoSource` `play`, `pause`, `seek`).
    pub fn control_video(&mut self, id: &str, request: VideoRequest) -> Result<(), Error> {
        self.map
            .command()
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
        self.map
            .command()
            .update_image(name, width, height, pixels)
            .map_err(Error::InvalidArgument)
    }

    /// Removes an image from the style (GL JS `removeImage`).
    pub fn remove_image(&mut self, name: &str) -> Result<(), Error> {
        self.styled()?;
        self.map
            .command()
            .remove_image(name)
            .map_err(Error::InvalidArgument)
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

    /// What the map draws on: its style's plane, or the globe the style's
    /// `projection` wraps it around.
    pub fn surface(&self) -> orb_crs::Surface {
        self.map.surface()
    }

    /// The viewport the camera and gestures use: the last target's, in
    /// logical pixels.
    pub fn viewport(&self) -> Viewport {
        self.map.viewport()
    }

    /// Sets the viewport, in logical pixels, and the pixel ratio
    /// (`scale` physical pixels each) before a target draws (each live
    /// frame sets its target's): fits, projections and constraints made
    /// before the first frame use the size the map will have, and a style
    /// that arrives before it ([`apply_arrivals`](Self::apply_arrivals))
    /// loads its sprites at the ratio it will draw at.
    pub fn set_viewport(&mut self, width: f64, height: f64, scale: f64) {
        self.map.command().set_viewport(Viewport { width, height });
        self.scale = crate::target::sane_scale(scale);
    }

    /// Moves the camera at once. Each move takes only the properties it
    /// sets from the animations running, which go on moving the others,
    /// as in MapLibre Native; one left with nothing to move ends
    /// interrupted. Its camera events come out as it moves, not with the
    /// next frame.
    pub fn jump_to(&mut self, target: &CameraTarget) {
        self.map.command().jump_to(target);
    }

    /// Animates the camera from the next frame on (GL JS `easeTo`).
    pub fn ease_to(&mut self, target: &CameraTarget, options: &EaseOptions) -> AnimationId {
        self.map.command().ease_to(target, *options)
    }

    /// Flies along GL JS's zoom-out curve from the next frame on.
    pub fn fly_to(&mut self, target: &CameraTarget, options: &FlyOptions) -> AnimationId {
        self.map.command().fly_to(target, options)
    }

    /// Continues a gesture the host recognized itself: each part of
    /// `momentum` glides on from its release velocity and slows by its
    /// decay, from the next frame on, then settles on its snap. Any input
    /// or camera move stops it; its
    /// [`CameraEvent::AnimationEnd`](crate::CameraEvent::AnimationEnd) then reports the speed it had left.
    /// `None` when nothing moves or settles over time. [`input`](Self::input)
    /// gestures end with one of their own.
    pub fn start_momentum(&mut self, momentum: &GestureMomentum) -> Option<AnimationId> {
        self.map.command().start_momentum(momentum)
    }

    /// Whether the camera's target stands on the terrain at rest (GL JS
    /// `centerClampedToGround`, on by default). Off, it keeps the
    /// elevation it is given, and the camera orbits a point in the air.
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.map.command().set_center_clamped_to_ground(clamped);
    }

    /// Draws the map in MapLibre Native's axonometric projection, heights
    /// leaning by `skew` and padding ignored, or in perspective (`None`).
    pub fn set_axonometric(&mut self, skew: Option<Skew>) {
        self.map.command().set_axonometric(skew);
    }

    pub fn center_clamped_to_ground(&self) -> bool {
        self.map.center_clamped_to_ground()
    }

    /// Puts the target this many metres above sea level at once (GL JS
    /// `setCenterElevation`); while clamped, the terrain takes it back.
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.map.command().set_center_elevation(elevation);
    }

    /// The camera's eye and orientation now (GL JS
    /// `getFreeCameraOptions`).
    pub fn free_camera(&self) -> FreeCamera {
        self.map.free_camera()
    }

    /// Puts the eye at `free` at once (GL JS `setFreeCameraOptions`),
    /// ending any animation; the field of view and padding stay.
    pub fn jump_to_free(&mut self, free: &FreeCamera) {
        self.map.command().jump_to_free(free);
    }

    /// The camera that shows `bounds` in the current viewport, inside the
    /// camera's padding.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        self.map.camera_for_bounds(bounds, options)
    }

    /// Stops every camera animation where it is.
    pub fn stop(&mut self) {
        self.map.command().stop();
    }

    /// Stops the camera animation `id` where it is, leaving the others.
    pub fn stop_animation(&mut self, id: AnimationId) {
        self.map.command().stop_animation(id);
    }

    /// Zoom and pitch ranges and bounds the camera keeps to.
    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.map.command().set_constraints(constraints);
    }

    /// Whether drags and their glides on the globe turn the bearing with
    /// the planet; off, the default, they hold it.
    pub fn set_globe_free_bearing(&mut self, free: bool) {
        self.map.command().set_globe_free_bearing(free);
    }

    pub fn globe_free_bearing(&self) -> bool {
        self.map.globe_free_bearing()
    }

    /// The camera a drag from `from` to `to` (logical pixels) moves to,
    /// before the constraints: on the globe with a free bearing, turned
    /// with the planet.
    pub fn panned(&self, from: DVec2, to: DVec2) -> CameraState {
        self.map.panned(from, to)
    }

    /// Hands the map a pointer, wheel or key input at `time`, the event's
    /// timestamp in milliseconds on the clock frame times use, in logical
    /// pixels from the target's top-left corner. Returns whether the map
    /// used it.
    pub fn input(&mut self, input: Input, time: f64) -> bool {
        self.map.command_if(|map| map.input(input, time))
    }

    /// What gestures may move, how releases continue, and the platform's
    /// thresholds; changing it cancels a gesture under way.
    pub fn set_gestures(&mut self, options: GestureOptions) {
        self.map.command().set_gestures(options);
    }

    pub fn gestures(&self) -> &GestureOptions {
        self.map.gestures()
    }

    /// Applies the camera moves of a recognizer the host keeps on its own
    /// thread, at `time` on the clock frame times use, in order: the C
    /// API's gesture recognizers' moves. Returns whether one took the
    /// camera from its animations, stopping them.
    pub fn apply_moves(&mut self, moves: impl IntoIterator<Item = Move>, time: f64) -> bool {
        self.map.command().apply_moves(moves, time)
    }

    /// How input moves the camera: holding it, or through a glide, snap or
    /// eased step a gesture started.
    pub fn gesture_motion(&self) -> GestureMotion {
        self.map.gesture_motion()
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
        self.map.query_rendered(min, max, None, None, done.1);
        done.0
    }

    /// [`query_rendered`](Self::query_rendered) in `layers` alone (GL JS's
    /// `layers` option), or in every layer, of the features a v8 `filter`
    /// as JSON selects (GL JS's `filter` option), evaluated at each
    /// feature's tile's zoom as GL JS and MapLibre Native evaluate it.
    pub fn query_rendered_in(
        &mut self,
        min: DVec2,
        max: DVec2,
        layers: Option<&[String]>,
        filter: Option<&str>,
    ) -> u64 {
        let done = self.query_done();
        self.map.query_rendered(min, max, layers, filter, done.1);
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

    /// [`unproject`](Self::unproject) for the camera a jump to `camera`
    /// would make, in this map's viewport, on its surface and terrain:
    /// what a move would show at a point, before making it.
    pub fn unproject_with(&self, camera: &CameraTarget, point: DVec2) -> Option<Geodetic> {
        self.map.unproject_with(&self.map.camera_for(camera), point)
    }

    /// Draws the frame shown at `frame_time` (milliseconds, on the clock
    /// input times use: the display link's): applies arrived data,
    /// advances gestures and animations to it and renders into `target`.
    /// A target of another format than the last frame's has the map's
    /// renderer draw in it from then on, its tiles kept and its pipelines
    /// made again for it. Returns `None` when the surface had nothing to
    /// draw into this time.
    pub fn render(
        &mut self,
        target: &mut Target,
        frame_time: f64,
    ) -> Result<Option<FrameReport>, Error> {
        self.set_mode(RenderMode::Continuous);
        self.scale = target.scale();
        self.frame(target, frame_time)
    }

    /// Advances gestures and animations to `time` (on the clock frames
    /// use) without drawing, for a map nothing draws for now: its camera
    /// moves, its moves end and their events come out as under frames,
    /// and no device is needed. Returns whether the camera moves on; call
    /// it again on the host's beat until it doesn't.
    pub fn advance(&mut self, time: f64) -> bool {
        let moving = self.map.command().advance(time);
        self.events
            .extend(self.map.take_events().into_iter().map(Event::Map));
        moving
    }

    /// Draws frames into `target` until the map is idle (its loads,
    /// pipelines, fades and camera motion done, and a last frame drawing
    /// what the one before brought), or at rest but for moving noise
    /// ([`FrameReport::rests`]), waiting for data between frames, and
    /// returns the last frame's report. Frames are timed on the runtime's
    /// clock. For tools and tests: hosts draw on their display link. An
    /// error once `stall` passes with nothing arriving ([`STILL_STALL`]).
    pub fn render_until_idle(
        &mut self,
        target: &mut Target,
        stall: Duration,
    ) -> Result<FrameReport, Error> {
        let mut patience = self.patience(stall);
        self.set_mode(RenderMode::Continuous);
        self.scale = target.scale();
        self.settle(target, &mut patience)
    }

    /// Draws what the camera shows in a `width × height` view (logical
    /// pixels, `scale` physical ones each) once everything in it has loaded
    /// and nothing fades, and returns the picture (still output): labels
    /// place as if nothing had been placed before, so the same style, data
    /// and camera always draw the same bytes. Renders as many frames as that
    /// takes, waiting for data in between; an error once nothing has
    /// arrived for [`STILL_STALL`]. Tiles that failed before are requested
    /// again first; one that fails again has its ancestors stand in.
    pub fn snapshot(&mut self, width: u32, height: u32, scale: f64) -> Result<Image, Error> {
        let mut patience = self.patience(STILL_STALL);
        self.retry_failed_tiles();
        loop {
            if let Some(image) = self.snapshot_step(width, height, scale)? {
                return Ok(image);
            }
            self.wait(&mut patience)?;
        }
    }

    /// [`snapshot`](Self::snapshot) for a host that can't wait on it:
    /// draws the frames the picture needs now and returns it once the map
    /// has settled, `None` until then. Call it again when the map is woken
    /// or [`needs_frame`](Self::needs_frame); it never times out.
    pub fn snapshot_step(
        &mut self,
        width: u32,
        height: u32,
        scale: f64,
    ) -> Result<Option<Image>, Error> {
        let px = |v: u32| (f64::from(v) * scale).round() as u32;
        let mut target = self.offscreen(px(width), px(height), scale);
        self.set_mode(RenderMode::Still);
        let image = match self.settle_step(&mut target) {
            Ok(Some(_)) => target.read_pixels().map(|pixels| {
                let (width, height) = target.size();
                Some(Image {
                    width,
                    height,
                    pixels,
                })
            }),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        };
        self.offscreen = Some(target);
        image
    }

    /// Draws one tile of a tile matrix set on the map's plane as still
    /// output, and returns its picture. The map loads the labels of the
    /// tile's neighbours too and places them all in tile scope
    /// ([`RenderMode::Tile`]): tiles drawn apart agree on the labels that
    /// cross their edges, those smaller than half a tile at least, and the
    /// same tile always draws the same bytes. The tile draws from a camera
    /// of its own: the map's camera, its constraints, animations and
    /// events, are left as they were. An error once nothing has arrived for
    /// [`STILL_STALL`]. Failed tiles are retried first, as for
    /// [`snapshot`](Self::snapshot).
    pub fn render_tile(&mut self, tile: &TileRequest) -> Result<Image, Error> {
        let mut patience = self.patience(STILL_STALL);
        let (index, px) = Self::tile_pixels(tile)?;
        self.retry_failed_tiles();
        let mut target = self.offscreen(px, px, tile.scale);
        let drawn = self.draw_tile(&mut target, tile, index, &mut patience);
        let pixels = drawn.and_then(|()| target.read_pixels());
        self.offscreen = Some(target);
        Ok(Image {
            width: px,
            height: px,
            pixels: pixels?,
        })
    }

    /// [`render_tile`](Self::render_tile) for a host that can't wait on
    /// it, as [`snapshot_step`](Self::snapshot_step) is
    /// [`snapshot`](Self::snapshot)'s: the picture once the map has
    /// settled, `None` until then. Commands may move the map's camera
    /// meanwhile: the tile draws from its own.
    pub fn render_tile_step(&mut self, tile: &TileRequest) -> Result<Option<Image>, Error> {
        let (index, px) = Self::tile_pixels(tile)?;
        let mut target = self.offscreen(px, px, tile.scale);
        let image = match self.tile_step(&mut target, tile, index) {
            Ok(true) => target.read_pixels().map(|pixels| {
                Some(Image {
                    width: px,
                    height: px,
                    pixels,
                })
            }),
            Ok(false) => Ok(None),
            Err(e) => Err(e),
        };
        self.offscreen = Some(target);
        image
    }

    /// `tile`'s index, once it's checked to be in its grid, and its pixels
    /// a side.
    fn tile_pixels(tile: &TileRequest) -> Result<(TileIndex, u32), Error> {
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
        Ok((index, px as u32))
    }

    /// Applies what has arrived, the style among it, without drawing: a
    /// host with nothing to draw into yet (a map before its first target)
    /// still gets its style in and its events ([`MapEvent::StyleLoad`]),
    /// and the requests the style starts go out, its sprites at the pixel
    /// ratio [`set_viewport`](Self::set_viewport) declared. Hosts that
    /// draw frames needn't call it. It keeps the renderer of the map's
    /// last frame; a map that has drawn none gets one its first frame
    /// takes on, in whatever format that frame's target has.
    pub fn apply_arrivals(&mut self) -> Result<(), Error> {
        self.arrive(self.scale)
    }

    /// Applies what has arrived without drawing a frame
    /// ([`orb_runtime::Map::apply_arrivals`]), starting sprites at
    /// `scale`.
    fn arrive(&mut self, scale: f64) -> Result<(), Error> {
        if !self.ready_for(None)? {
            return Ok(());
        }
        {
            let mut device = self.runtime.shared.gpu()?;
            self.woken.store(false, Ordering::Release);
            let gpu: &mut dyn orb_hal::Gpu = &mut *device;
            let map = self.map.drawing();
            let format = self.format;
            let renderer = self
                .renderer
                .get_or_insert_with(|| map.create_renderer(gpu, format));
            map.apply_arrivals(gpu, renderer, scale as f32);
        }
        self.events
            .extend(self.map.take_events().into_iter().map(Event::Map));
        Ok(())
    }

    /// The texture still output draws into, of this size, in the format
    /// the map draws in, so still output doesn't retarget its renderer;
    /// RGBA8 where that isn't an 8-bit format pixels read back in (a float
    /// window or host texture).
    fn offscreen(&mut self, width: u32, height: u32, scale: f64) -> Target {
        let format = match self.format.color {
            f @ (orb_hal::TextureFormat::Rgba8Unorm | orb_hal::TextureFormat::Bgra8Unorm) => f,
            _ => crate::target::DEFAULT_FORMAT.color,
        };
        match self.offscreen.take() {
            Some(t)
                if t.size() == (width, height)
                    && t.scale() == scale
                    && t.format().color == format =>
            {
                t
            }
            _ => Target::texture_of(&self.runtime, (width, height), scale, format),
        }
    }

    fn draw_tile(
        &mut self,
        target: &mut Target,
        tile: &TileRequest,
        index: TileIndex,
        patience: &mut Patience,
    ) -> Result<(), Error> {
        while !self.tile_style_in(target)? {
            self.wait(patience)?;
        }
        self.set_mode(self.tile_scene(tile, index)?);
        self.settle(target, patience).map(|_| ())
    }

    /// One step of [`draw_tile`](Self::draw_tile): whether the map has
    /// settled on the tile.
    fn tile_step(
        &mut self,
        target: &mut Target,
        tile: &TileRequest,
        index: TileIndex,
    ) -> Result<bool, Error> {
        if !self.tile_style_in(target)? {
            return Ok(false);
        }
        self.set_mode(self.tile_scene(tile, index)?);
        self.settle_step(target).map(|report| report.is_some())
    }

    /// Whether the style is in, after taking in what arrived. The style's
    /// projection sets a tile's plane. Until it is in, nothing draws: a
    /// frame would load what the interactive camera shows.
    fn tile_style_in(&mut self, target: &mut Target) -> Result<bool, Error> {
        if self.map.style_doc().is_some() {
            return Ok(true);
        }
        self.set_mode(RenderMode::Still);
        self.arrive(target.scale())?;
        self.style_failed()?;
        Ok(self.map.style_doc().is_some())
    }

    /// The mode that draws `tile`: its camera, and the grid of it and its
    /// neighbours whose labels it places.
    fn tile_scene(&self, tile: &TileRequest, index: TileIndex) -> Result<RenderMode, Error> {
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
                    axonometric: None,
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
        // In the map's projection: axonometric, it draws tiles so too.
        let camera = CameraState {
            fov: 2.0
                * ((CameraState::DEFAULT_FOV / 2.0).to_radians().tan() / 3.0)
                    .atan()
                    .to_degrees(),
            axonometric: self.map.camera().axonometric,
            ..grid(col, row).expect("checked above").camera
        };
        Ok(RenderMode::Tile {
            camera,
            grid: tiles,
        })
    }

    fn set_mode(&mut self, mode: RenderMode) {
        if *self.map.mode() != mode {
            self.map.command().set_mode(mode);
        }
    }

    /// Renders until two frames in a row are idle, the second drawing what
    /// the first's messages brought, waiting for data between frames.
    fn settle(
        &mut self,
        target: &mut Target,
        patience: &mut Patience,
    ) -> Result<FrameReport, Error> {
        loop {
            if let Some(report) = self.settle_step(target)? {
                return Ok(report);
            }
            self.wait(patience)?;
        }
    }

    /// Renders until two frames in a row are idle, and returns the second
    /// one's report, or `None` at the first frame that leaves work to wait
    /// for.
    fn settle_step(&mut self, target: &mut Target) -> Result<Option<FrameReport>, Error> {
        let mut settle = orb_runtime::Settle::default();
        loop {
            let report = self.frame(target, self.now())?;
            let settled = settle
                .frame(&self.map, report.as_ref())
                .map_err(Error::InvalidArgument)?;
            match report {
                Some(report) if settled => return Ok(Some(report)),
                Some(report) if report.rests() => {}
                _ => return Ok(None),
            }
        }
    }

    /// Whether the latest style request failed because its text didn't
    /// parse as a style, not because its fetch brought none
    /// ([`orb_runtime::Map::style_unparsed`]).
    pub fn style_unparsed(&self) -> bool {
        self.map.style_unparsed()
    }

    /// The latest style request's failure, as an error.
    fn style_failed(&self) -> Result<(), Error> {
        match self.map.style_error() {
            Some(message) => Err(Error::InvalidArgument(message.to_string())),
            None => Ok(()),
        }
    }

    /// Waits for data, a frame's time at most (fades and transitions run
    /// on the clock), or fails once `patience` has seen no progress for
    /// its stall, naming what the map waited for. Progress is a wake of
    /// the map's own, or the device's while the last frame waited for it:
    /// the device wakes every map on the runtime when any one's pipeline
    /// compiles or a region downloads, which is no news to the others.
    fn wait(&self, patience: &mut Patience) -> Result<(), Error> {
        let wakes = self.wakes();
        let device_news = matches!(
            self.shown,
            Shown::AwaitingDevice
                | Shown::Drew {
                    activity: Activity::Compiling,
                    ..
                }
        );
        if wakes.own != patience.seen.own || (device_news && wakes.device != patience.seen.device) {
            patience.since = Instant::now();
        }
        patience.seen = wakes;
        let quiet = patience.since.elapsed();
        if quiet >= patience.stall {
            return Err(Error::InvalidState(format!(
                "the map stalled: nothing arrived for {:?}, and {}",
                patience.stall,
                self.waiting_for()
            )));
        }
        self.wait_woken((patience.stall - quiet).min(Duration::from_millis(16)));
        Ok(())
    }

    /// Still output's patience from now: it gives up once `stall` passes
    /// with no wake.
    fn patience(&self, stall: Duration) -> Patience {
        Patience {
            stall,
            seen: self.wakes(),
            since: Instant::now(),
        }
    }

    /// How many times the map has been woken.
    fn wakes(&self) -> Wakes {
        *self.signal.0.lock().expect("signal poisoned")
    }

    /// What the map waited for when its last frame drew, in words.
    fn waiting_for(&self) -> String {
        let activity = match self.shown {
            Shown::Drew { activity, .. } => activity,
            Shown::AwaitingDevice => return "it waits for a lost device's replacement".into(),
            Shown::GaveUp => return "it has no device".into(),
        };
        match activity {
            Activity::Loading => {
                let awaiting = self.map.awaiting();
                let mut what = Vec::new();
                if awaiting.style {
                    what.push("the style".to_owned());
                }
                if !awaiting.sources.is_empty() {
                    what.push(format!("sources {}", awaiting.sources.join(", ")));
                }
                if self.tiles_loading > 0 {
                    what.push(format!("{} tiles", self.tiles_loading));
                }
                if what.is_empty() {
                    "it was loading labels or the drape".into()
                } else {
                    format!("it was loading {}", what.join(", "))
                }
            }
            Activity::Busy => "its frames were still changing".into(),
            Activity::Compiling => "it was waiting for pipelines to compile".into(),
            Activity::Animating | Activity::Idle => "its last frame was complete".into(),
        }
    }

    /// Blocks until the map is woken (data arrived, or a pipeline a frame
    /// waited for compiled) or `timeout` passes: for a host that must show
    /// a whole frame now, between frames whose report says they waited
    /// for one.
    pub fn wait_woken(&self, timeout: Duration) {
        self.waiter().wait(timeout);
    }

    /// What [`wait_woken`](Self::wait_woken) waits on, apart from the map:
    /// for a host that lets go of a lock on the map while it waits.
    pub fn waiter(&self) -> Waiter {
        Waiter {
            signal: self.signal.clone(),
            woken: self.woken.clone(),
            device_woken: self.device_woken.clone(),
        }
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
        if !self.ready_for(Some(target))? {
            return Ok(None);
        }
        let format = target.format();
        let viewport = target.viewport();
        // The frame borrows the device once: the target begins and ends
        // with it, and it's let go before anything else here borrows it.
        let report = {
            let mut device = self.runtime.shared.gpu()?;
            let Some(texture) = target.begin(&mut device)? else {
                return Ok(None);
            };
            // Only a target that began says what the map draws in: a
            // released window reports a texture's format.
            self.format = format;
            device.start_frame();
            self.woken.store(false, Ordering::Release);
            self.device_woken.store(false, Ordering::Release);
            let gpu: &mut dyn orb_hal::Gpu = &mut *device;
            let map = self.map.drawing();
            let renderer = self
                .renderer
                .get_or_insert_with(|| map.create_renderer(gpu, format));
            renderer.set_format(gpu, format);
            let report = map.render(
                gpu,
                renderer,
                texture,
                viewport,
                target.scale() as f32,
                frame_time,
            );
            target.end(&mut device);
            device.finish_frame();
            report
        };
        // The frame found the device lost (a compile the watchdog found
        // hung, a lost context): the next one makes another.
        let lost = self.runtime.lost();
        self.events
            .extend(self.map.take_events().into_iter().map(Event::Map));
        self.map.drawn();
        if lost {
            self.map.mark();
        }
        self.shown = Shown::Drew {
            activity: report.activity,
            busy: report.needs_frame(),
        };
        self.tiles_loading = report.tiles_loading;
        self.alarm.set(report.wake_at);
        if report.idle() {
            // A failed write costs the next run its warm start, nothing else.
            let _ = self.runtime.save_pipeline_cache();
        }
        Ok(Some(report))
    }

    /// Checks that this map can draw into `target` (or, without one,
    /// apply arrivals), after bringing it to the runtime's current device:
    /// `false` while a lost device's replacement waits out its backoff
    /// (the map is woken when it's due).
    fn ready_for(&mut self, mut target: Option<&mut Target>) -> Result<bool, Error> {
        if target
            .as_ref()
            .is_some_and(|t| !t.same_runtime(&self.runtime.shared))
        {
            return Err(Error::InvalidArgument(
                "the target belongs to another runtime".into(),
            ));
        }
        // The first frame makes the runtime's device.
        self.runtime.open_device()?;
        // A lost GPU is replaced before the frame: texture
        // targets are made again, and window targets fail with
        // `TargetLost` for the host to make on the new one. After too
        // many losses in a row the runtime stops, and the map says why
        // once, as an error event, besides failing each frame.
        if self.runtime.lost() {
            // The wakes so far are spent before the check, so the
            // backoff's wake, landing during it, stays news.
            self.woken.store(false, Ordering::Release);
            self.device_woken.store(false, Ordering::Release);
            let recovered = self.runtime.recovery_due().and_then(|due| {
                if due {
                    if let Some(target) = target.as_mut().filter(|t| t.is_window()) {
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
                    // map: a host asking `needs_frame`, or waiting for a
                    // wake, doesn't spin.
                    self.shown = Shown::AwaitingDevice;
                    return Ok(false);
                }
                Err(e) => {
                    // No device will come: nothing is news again.
                    if std::mem::replace(&mut self.shown, Shown::GaveUp) != Shown::GaveUp {
                        self.events.push_back(Event::Map(MapEvent::Error {
                            message: e.message().to_owned(),
                        }));
                    }
                    return Err(e);
                }
            }
        }
        // The runtime replaced its device: draw on the new one, laying the
        // tiles out again. A map waiting for the replacement waits no more.
        if self.device != self.runtime.shared.generation() {
            self.device = self.runtime.shared.generation();
            self.renderer = None;
            self.shown = Shown::FRESH;
            self.map.mark();
        }
        Ok(true)
    }

    /// The oldest event not yet taken. What the map did between frames
    /// (an animation a command stopped) comes out without one.
    pub fn poll_event(&mut self) -> Option<Event> {
        let finished = std::mem::take(&mut *self.finished.lock().expect("queries poisoned"));
        self.events.extend(
            finished
                .into_iter()
                .map(|(id, result)| Event::Query { id, result }),
        );
        if self.events.is_empty() {
            self.events
                .extend(self.map.take_events().into_iter().map(Event::Map));
        }
        self.events.pop_front()
    }
}

mod commanded {
    use std::ops::Deref;

    use orb_runtime::MapEvent;

    /// The runtime map, and whether something since the last frame
    /// changed what it shows. It reads through `Deref`; a command reaches
    /// it only through [`command`](Self::command) or
    /// [`command_if`](Self::command_if), which mark it changed, so none can
    /// leave the map idle after changing it. Only the frame itself
    /// ([`drawing`](Self::drawing)) and draining its events
    /// ([`take_events`](Self::take_events)), which changes nothing it
    /// shows, reach it unmarked.
    pub(super) struct Commanded {
        map: orb_runtime::Map,
        changed: bool,
    }

    impl Commanded {
        /// A new map, which has drawn nothing yet.
        pub(super) fn new(map: orb_runtime::Map) -> Self {
            Self { map, changed: true }
        }

        /// The map for a command, which the next frame shows.
        pub(super) fn command(&mut self) -> &mut orb_runtime::Map {
            self.changed = true;
            &mut self.map
        }

        /// Runs a command that says whether it changed anything, such as
        /// input the map may not use, marking the map only when it did.
        pub(super) fn command_if(&mut self, f: impl FnOnce(&mut orb_runtime::Map) -> bool) -> bool {
            let changed = f(&mut self.map);
            self.changed |= changed;
            changed
        }

        /// Marks the map changed for what isn't a command: a frame must
        /// draw it again on a new renderer or device.
        pub(super) fn mark(&mut self) {
            self.changed = true;
        }

        /// Whether something since the last frame changed what it shows.
        pub(super) fn changed(&self) -> bool {
            self.changed
        }

        /// The map for drawing a frame or applying arrivals, unmarked:
        /// those show what commands changed rather than changing it.
        pub(super) fn drawing(&mut self) -> &mut orb_runtime::Map {
            &mut self.map
        }

        /// A frame drew everything commands changed.
        pub(super) fn drawn(&mut self) {
            self.changed = false;
        }

        /// The events the map has raised since they were last taken.
        pub(super) fn take_events(&mut self) -> Vec<MapEvent> {
            self.map.take_events()
        }
    }

    impl Deref for Commanded {
        type Target = orb_runtime::Map;

        fn deref(&self) -> &orb_runtime::Map {
            &self.map
        }
    }
}
