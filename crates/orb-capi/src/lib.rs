//! The C ABI, 0.x (plan §14.1, decision 0018): `orb-api` as a Diplomat
//! bridge. This module is the annotated API: Diplomat's C backend writes
//! the header from it, `diplomat-swift-kotlin` the Kotlin and Swift
//! bindings (`cargo xtask bindings`).
//!
//! Conventions:
//! - **Ownership.** `Box` returns are owned and freed with the type's
//!   `*_destroy`; references are borrowed for the call; a returned slice
//!   borrows from its owner until the owner's next mutable call.
//! - **Errors.** Fallible calls return `OrbError`, a status;
//!   [`OrbError::last_message`] has the message of the thread's last error.
//!   Panics abort.
//! - **Contracts.** `#[diplomat::attr(contract, …)]` gives each method its
//!   execution category (`command`, `query`, `snapshot`, `event_batch`,
//!   `render_driver`, `lifecycle`, `immediate`) and each type its thread
//!   rule: `owner`, the thread that created the runtime it belongs to, or
//!   `any`; `error_message` marks the function the bindings call for an
//!   error's message. Code generators skip these; the binding emitters
//!   read them.
//!
//! The bridge isn't Rust API: its functions take raw addresses and keep
//! callbacks across threads on the C caller's word, so Rust can't reach
//! them. A non-`Send` wake callback doesn't compile:
//!
//! ```compile_fail,E0603
//! let wake = std::rc::Rc::new(std::cell::Cell::new(0));
//! let _ = |map: &mut orbis::ffi::OrbMap| map.set_wake(move || wake.set(1));
//! ```
//!
//! Nor does a surface on an arbitrary address:
//!
//! ```compile_fail,E0603
//! let _ = |rt: &orbis::ffi::OrbRuntime| {
//!     orbis::ffi::OrbTarget::create_metal_layer(rt, 0xdead, 1, 1, 1.0)
//! };
//! ```
//!
//! Rust embedders use `orb-api`, whose wake callback must be `Send +
//! Sync` and whose surfaces come from `orb_hal_wgpu`'s `unsafe`
//! constructor.

// Crate-private: the bridge is the C ABI, whose callers keep its documented
// contracts (raw handles live, callbacks callable from any thread). Rust
// embedders use `orb-api`, which states them in its types. Diplomat wants
// its items `pub`; enum variants that only C passes in are never
// constructed here.
#[diplomat::bridge]
#[allow(dead_code, unreachable_pub)]
mod ffi {
    use std::fmt::Write;

    use diplomat_runtime::{DiplomatStr, DiplomatWrite};

    use crate::{fail, text};

    /// Why a call failed; the message is [`OrbError::last_message`].
    #[diplomat::attr(auto, error)]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbError {
        /// An argument is invalid, such as malformed JSON.
        InvalidArgument = 1,
        /// The call doesn't fit the current state, such as editing before
        /// a style is loaded.
        InvalidState = 2,
        /// The platform or the build lacks the feature.
        Unsupported = 3,
        /// The GPU or the platform failed.
        Native = 4,
        /// The target can't be drawn to again; make a new one.
        TargetLost = 5,
        /// The GPU was lost; `OrbRuntime::recover` makes another.
        DeviceLost = 6,
    }

    /// The GPU tier a runtime draws at (plan §4.2).
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbBackend {
        /// Modern where the device policy allows it (the Android
        /// allowlist, the crash sentinel), else Compat.
        Auto = 0,
        /// Vulkan, Metal or D3D12 through wgpu.
        Modern = 1,
        /// GLES 3.0 through EGL.
        Compat = 2,
    }

    impl OrbError {
        /// Writes the message of the last error on the calling thread.
        #[diplomat::attr(contract, immediate)]
        #[diplomat::attr(contract, error_message)]
        pub fn last_message(out: &mut DiplomatWrite) {
            crate::LAST_ERROR.with(|e| out.write_str(&e.borrow()).ok());
        }
    }

    /// A GPU device and host services, shared by the maps made from it.
    #[diplomat::opaque_mut]
    #[diplomat::attr(contract, thread = "owner")]
    #[derive(Debug)]
    pub struct OrbRuntime(pub(crate) orb_api::Runtime);

    impl OrbRuntime {
        /// Makes a device on the default adapter and starts the job and
        /// fetch pools; `job_threads` 0 picks the platform default. Blocks
        /// while the device is made. The calling thread owns the runtime
        /// and everything made from it.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create(job_threads: u32) -> Result<Box<OrbRuntime>, OrbError> {
            Self::create_with(job_threads, OrbBackend::Auto, "".as_bytes(), 0)
        }

        /// As `create`, on `backend`, keeping state between runs (the crash
        /// sentinel, the HTTP cache) in the directory `state_dir` (none when
        /// empty);
        /// `android_api_level` is `Build.VERSION.SDK_INT` on Android, for
        /// the allowlist, and 0 elsewhere.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create_with(
            job_threads: u32,
            backend: OrbBackend,
            state_dir: &DiplomatStr,
            android_api_level: u32,
        ) -> Result<Box<OrbRuntime>, OrbError> {
            let state_dir = text(state_dir);
            orb_api::Runtime::new(&orb_api::RuntimeOptions {
                job_threads: (job_threads > 0).then_some(job_threads as usize),
                backend: match backend {
                    OrbBackend::Auto => orb_api::Backend::Auto,
                    OrbBackend::Modern => orb_api::Backend::Modern,
                    OrbBackend::Compat => orb_api::Backend::Compat,
                },
                state_dir: (!state_dir.is_empty()).then(|| state_dir.as_ref().into()),
                android_api_level,
                egl_library: None,
            })
            .map(|r| Box::new(OrbRuntime(r)))
            .map_err(fail)
        }

        /// Whether the runtime draws at the Compat tier (GLES 3.0).
        #[diplomat::attr(contract, snapshot)]
        pub fn compat(&self) -> bool {
            self.0.tier() == orb_api::Tier::Compat
        }

        /// Why the device is the tier it is: "Modern: …" or "Compat: …".
        #[diplomat::attr(contract, snapshot)]
        pub fn selection(&self, out: &mut DiplomatWrite) {
            out.write_str(&self.0.selection()).ok();
        }

        /// Loses the GPU on purpose, to exercise recovery: frames fail
        /// with `DeviceLost` until `recover`.
        #[diplomat::attr(contract, command)]
        pub fn lose_device(&mut self) {
            self.0.lose_device();
        }

        /// Makes a new device after a loss; maps lay their tiles out again
        /// and window targets report `TargetLost` to be made again.
        #[diplomat::attr(contract, command)]
        pub fn recover(&mut self) -> Result<(), OrbError> {
            self.0.recover().map(|_| ()).map_err(fail)
        }

        /// The C ABI's version: bumped on every incompatible change while
        /// it is 0.x.
        #[diplomat::attr(contract, immediate)]
        pub fn abi_version() -> u32 {
            crate::ABI_VERSION
        }

        /// Render and compute pipelines the device has made so far.
        #[diplomat::attr(contract, snapshot)]
        pub fn pipelines_created(&self) -> u32 {
            self.0.pipelines_created()
        }

        /// Keeps the driver's compiled pipelines in the directory `dir`
        /// between runs, so warm starts skip compiling them (Vulkan;
        /// elsewhere the platform caches and this does nothing). Call
        /// before the first frame; maps save the cache when idle.
        #[diplomat::attr(contract, command)]
        pub fn set_pipeline_cache_dir(&mut self, dir: &DiplomatStr) -> Result<(), OrbError> {
            self.0
                .set_pipeline_cache_dir(std::path::Path::new(&*text(dir)))
                .map_err(fail)
        }

        /// Downloads `style_url`'s map of the area west, south, east,
        /// north (degrees) between map zooms `min_zoom` and `max_zoom` into
        /// the state directory, for maps to open with no network. Events
        /// arrive from `poll_region_event`, with the maps' wakes. Returns
        /// the region's id.
        #[diplomat::attr(contract, command)]
        #[allow(clippy::too_many_arguments)]
        pub fn download_region(
            &mut self,
            name: &DiplomatStr,
            style_url: &DiplomatStr,
            west: f64,
            south: f64,
            east: f64,
            north: f64,
            min_zoom: f64,
            max_zoom: f64,
        ) -> Result<u32, OrbError> {
            self.0
                .download_region(orb_api::RegionDefinition {
                    name: text(name).into_owned(),
                    style_url: text(style_url).into_owned(),
                    bounds: [west, south, east, north],
                    zooms: [min_zoom, max_zoom],
                })
                .map_err(fail)
        }

        /// Deletes a region, stopping its download.
        #[diplomat::attr(contract, command)]
        pub fn delete_region(&mut self, id: u32) -> Result<(), OrbError> {
            self.0.delete_region(id).map_err(fail)
        }

        /// The regions' ids and whether each is complete, one per line:
        /// `id complete bytes name`.
        #[diplomat::attr(contract, snapshot)]
        pub fn regions(&self, out: &mut DiplomatWrite) -> Result<(), OrbError> {
            for region in self.0.regions().map_err(fail)? {
                writeln!(
                    out,
                    "{} {} {} {}",
                    region.id, region.complete, region.bytes, region.definition.name
                )
                .ok();
            }
            Ok(())
        }

        /// The oldest region download event not yet taken.
        #[diplomat::attr(contract, event_batch)]
        pub fn poll_region_event(&mut self) -> Option<Box<OrbRegionEvent>> {
            self.0
                .poll_region_event()
                .map(|e| Box::new(OrbRegionEvent(e)))
        }

        /// Bytes of response bodies the HTTP cache holds (0 without one).
        #[diplomat::attr(contract, snapshot)]
        pub fn http_cache_bytes(&self) -> u64 {
            self.0.http_cache_stats().map_or(0, |s| s.bytes)
        }

        /// Empties the HTTP cache.
        #[diplomat::attr(contract, command)]
        pub fn clear_http_cache(&mut self) {
            self.0.clear_http_cache();
        }

        /// Milliseconds spent compiling pipelines so far, summed over the
        /// threads that compiled them: translation and the driver's
        /// compiles.
        #[diplomat::attr(contract, snapshot)]
        pub fn pipeline_ms(&self) -> f64 {
            self.0.pipeline_time().as_secs_f64() * 1000.0
        }

        /// Each pipeline compiled so far, slowest first: one line each,
        /// milliseconds then its label.
        #[diplomat::attr(contract, snapshot)]
        pub fn pipeline_timings(&self, out: &mut DiplomatWrite) {
            let mut timings = self.0.pipeline_timings();
            timings.sort_by_key(|t| std::cmp::Reverse(t.1));
            for (label, took) in timings {
                let _ = writeln!(out, "{:.1} {label}", took.as_secs_f64() * 1000.0);
            }
        }
    }

    /// What a map draws into: a window surface, or a texture the runtime
    /// owns. Sizes are physical pixels; `scale` is physical pixels per
    /// logical pixel, the unit of every map coordinate.
    #[diplomat::opaque_mut]
    #[diplomat::attr(contract, thread = "owner")]
    #[derive(Debug)]
    pub struct OrbTarget {
        pub(crate) target: orb_api::Target,
        pub(crate) pixels: Vec<u8>,
    }

    impl OrbTarget {
        /// An RGBA8 texture the runtime owns.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create_texture(
            runtime: &OrbRuntime,
            width: u32,
            height: u32,
            scale: f64,
        ) -> Box<OrbTarget> {
            Box::new(OrbTarget {
                target: orb_api::Target::texture(&runtime.0, width, height, scale),
                pixels: Vec::new(),
            })
        }

        /// A surface on a `CAMetalLayer*` (macOS, iOS), passed as an
        /// address. The layer must outlive the target. `Unsupported` on
        /// other platforms.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create_metal_layer(
            runtime: &OrbRuntime,
            layer: usize,
            width: u32,
            height: u32,
            scale: f64,
        ) -> Result<Box<OrbTarget>, OrbError> {
            crate::raw_target(
                runtime,
                layer,
                orb_api::RawSurface::MetalLayer,
                width,
                height,
                scale,
            )
        }

        /// A surface on an `ANativeWindow*` (Android), passed as an
        /// address. The caller keeps a reference to the window until the
        /// target is destroyed. `Unsupported` on other platforms.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create_android_window(
            runtime: &OrbRuntime,
            window: usize,
            width: u32,
            height: u32,
            scale: f64,
        ) -> Result<Box<OrbTarget>, OrbError> {
            crate::raw_target(
                runtime,
                window,
                orb_api::RawSurface::AndroidWindow,
                width,
                height,
                scale,
            )
        }

        /// Resizes the target; a texture's pixels are lost.
        #[diplomat::attr(contract, command)]
        pub fn resize(&mut self, width: u32, height: u32, scale: f64) {
            self.target.resize(width, height, scale);
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn width(&self) -> u32 {
            self.target.size().0
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn height(&self) -> u32 {
            self.target.size().1
        }

        /// Copies a texture target's pixels out for [`pixels`](Self::pixels).
        /// Blocks until the GPU is done.
        #[diplomat::attr(contract, query)]
        pub fn read_pixels(&mut self) -> Result<(), OrbError> {
            self.pixels = self.target.read_pixels().map_err(fail)?;
            Ok(())
        }

        /// The pixels [`read_pixels`](Self::read_pixels) copied: RGBA8, rows
        /// top to bottom without padding.
        #[diplomat::attr(contract, snapshot)]
        #[allow(clippy::needless_lifetimes)] // Diplomat needs them named.
        pub fn pixels<'a>(&'a self) -> &'a [u8] {
            &self.pixels
        }
    }

    /// How gestures feel.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub enum OrbFeel {
        /// GL JS's handlers.
        Desktop = 0,
        /// Pans glide like a `UIScrollView`.
        Ios = 1,
        /// Pans glide like Android's `OverScroller`.
        Android = 2,
    }

    /// The camera: the point at the viewport's centre, the style zoom, the
    /// bearing (degrees clockwise from north), the pitch (degrees from
    /// looking straight down), the roll (degrees about the view axis) and
    /// the centre's elevation (metres above sea level).
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrbCamera {
        pub lon: f64,
        pub lat: f64,
        pub zoom: f64,
        pub bearing: f64,
        pub pitch: f64,
        pub roll: f64,
        pub elevation: f64,
    }

    /// Where a camera move ends; NaN keeps a value as it is. While the
    /// centre is clamped to the ground, the terrain takes the elevation.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrbCameraTarget {
        pub lon: f64,
        pub lat: f64,
        pub zoom: f64,
        pub bearing: f64,
        pub pitch: f64,
        pub roll: f64,
        pub elevation: f64,
    }

    /// A geographic box, degrees.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrbBounds {
        pub west: f64,
        pub south: f64,
        pub east: f64,
        pub north: f64,
    }

    /// A viewport position in logical pixels from the top-left corner.
    /// `valid` is false when there's no answer (behind the camera).
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrbPoint {
        pub x: f64,
        pub y: f64,
        pub valid: bool,
    }

    /// A geographic point, degrees. `valid` is false when there's no
    /// answer (above the horizon).
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrbLngLat {
        pub lon: f64,
        pub lat: f64,
        pub valid: bool,
    }

    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbPointerKind {
        Mouse = 0,
        Touch = 1,
        Pen = 2,
    }

    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbButton {
        Primary = 0,
        Secondary = 1,
        Middle = 2,
    }

    /// The keys the map responds to; hosts map platform key codes to them.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbKey {
        Left = 0,
        Right = 1,
        Up = 2,
        Down = 3,
        ZoomIn = 4,
        ZoomOut = 5,
    }

    /// The unit of a wheel delta, as DOM `deltaMode` gives it.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbWheelUnit {
        Pixel = 0,
        Line = 1,
        Page = 2,
    }

    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct OrbModifiers {
        pub shift: bool,
        pub ctrl: bool,
        pub alt: bool,
        pub meta: bool,
    }

    /// What a frame did.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct OrbFrame {
        /// False when the surface had nothing to draw into this time.
        pub drawn: bool,
        /// The style generation the frame shows in full.
        pub generation: u64,
        /// Nothing is loading.
        pub idle: bool,
        /// Labels are placed for this view; fades may still run.
        pub loaded: bool,
        pub tiles_drawn: u32,
        pub tiles_loading: u32,
        pub draws: u32,
        pub dispatches: u32,
        /// Pipelines still compiling: layers that need them aren't drawn
        /// yet.
        pub pipelines_pending: u32,
        /// Layers not drawn while their pipelines compile.
        pub layers_waiting: u32,
    }

    /// What an event is about.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbEventKind {
        /// A style was installed.
        StyleLoad = 0,
        /// The style changed; frames show it from `generation` on.
        StyleData = 1,
        /// A source's data arrived; `text` is its id, `flag` whether none
        /// of its tiles are loading.
        SourceData = 2,
        /// The first complete frame; once per map.
        Load = 3,
        /// The map has nothing more to draw until something changes.
        Idle = 4,
        /// Something failed to load; `text` says what.
        Error = 5,
        /// The camera started, kept or stopped moving; `change` and
        /// `phase` say how, `flag` whether user input moved it.
        Camera = 6,
        /// An animation stopped; `id` is its id, `flag` whether it was
        /// interrupted.
        AnimationEnd = 7,
        /// A press and release that didn't move the map, at `x`, `y`.
        Click = 8,
        DoubleClick = 9,
        /// A `query_rendered` finished: `id` is its id, `flag` whether it
        /// succeeded, `text` the features as a JSON array of GeoJSON
        /// features, topmost first, or the error.
        QueryResult = 10,
        /// The map renders on a GPU tier: on its first frame and after a
        /// device change; `text` is `modern` or `compat`.
        Capability = 11,
        /// A tile's layers name an image the style doesn't have; `text` is
        /// its name. Once per name until the style gets it.
        StyleImageMissing = 12,
        /// On a host that plays video (`OrbMap::set_plays_video`): play
        /// the first of `urls` it can, muted, looping, from the start, for
        /// the video source `text`, and hand its frames to
        /// `OrbMap::video_frame`.
        VideoOpen = 13,
        /// Play, pause or seek (to `seconds`) the video of source `text`.
        VideoPlay = 14,
        VideoPause = 15,
        VideoSeek = 16,
        /// The video source `text` is gone: stop and let go of its video.
        VideoClose = 17,
    }

    /// For camera events: what changed.
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbChange {
        Move = 0,
        Zoom = 1,
        Rotate = 2,
        Pitch = 3,
    }

    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbPhase {
        Start = 0,
        Progress = 1,
        End = 2,
    }

    /// What an offline region's download did.
    #[derive(Debug, PartialEq, Eq)]
    pub enum OrbRegionEventKind {
        /// `done` of `total` requests so far, `bytes` kept.
        Progress = 0,
        /// Stored and served; `failed` requests are missing from it.
        Complete = 1,
        /// Stopped; see `message`.
        Failed = 2,
    }

    /// An offline region download's progress, completion or failure.
    #[diplomat::opaque]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub struct OrbRegionEvent(pub(crate) orb_api::RegionEvent);

    impl OrbRegionEvent {
        #[diplomat::attr(contract, snapshot)]
        pub fn kind(&self) -> OrbRegionEventKind {
            match self.0 {
                orb_api::RegionEvent::Progress { .. } => OrbRegionEventKind::Progress,
                orb_api::RegionEvent::Complete { .. } => OrbRegionEventKind::Complete,
                orb_api::RegionEvent::Failed { .. } => OrbRegionEventKind::Failed,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn region(&self) -> u32 {
            match self.0 {
                orb_api::RegionEvent::Progress { id, .. }
                | orb_api::RegionEvent::Complete { id, .. }
                | orb_api::RegionEvent::Failed { id, .. } => id,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn done(&self) -> u64 {
            match self.0 {
                orb_api::RegionEvent::Progress { done, .. } => done,
                _ => 0,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn total(&self) -> u64 {
            match self.0 {
                orb_api::RegionEvent::Progress { total, .. } => total,
                _ => 0,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn bytes(&self) -> u64 {
            match self.0 {
                orb_api::RegionEvent::Progress { bytes, .. }
                | orb_api::RegionEvent::Complete { bytes, .. } => bytes,
                orb_api::RegionEvent::Failed { .. } => 0,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn failed(&self) -> u64 {
            match self.0 {
                orb_api::RegionEvent::Complete { failed, .. } => failed,
                _ => 0,
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn message(&self, out: &mut DiplomatWrite) {
            if let orb_api::RegionEvent::Failed { message, .. } = &self.0 {
                out.write_str(message).ok();
            }
        }
    }

    /// Something a map, its camera or a gesture did.
    #[diplomat::opaque]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub struct OrbEvent(pub(crate) orb_api::Event);

    impl OrbEvent {
        #[diplomat::attr(contract, snapshot)]
        pub fn kind(&self) -> OrbEventKind {
            crate::event_kind(&self.0)
        }

        /// Logical pixels, for clicks.
        #[diplomat::attr(contract, snapshot)]
        pub fn x(&self) -> f64 {
            crate::event_position(&self.0).x
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn y(&self) -> f64 {
            crate::event_position(&self.0).y
        }

        /// The generation, for style data.
        #[diplomat::attr(contract, snapshot)]
        pub fn generation(&self) -> u64 {
            match self.0 {
                orb_api::Event::Map(orb_api::MapEvent::StyleData { generation }) => generation,
                _ => 0,
            }
        }

        /// The animation, for animation ends; the query, for query
        /// results.
        #[diplomat::attr(contract, snapshot)]
        pub fn id(&self) -> u64 {
            match self.0 {
                orb_api::Event::Camera(orb_api::CameraEvent::AnimationEnd { id, .. }) => id.0,
                orb_api::Event::Query { id, .. } => id,
                _ => 0,
            }
        }

        /// Camera events: from user input. Animation ends: interrupted.
        /// Source data: loaded. Query results: succeeded.
        #[diplomat::attr(contract, snapshot)]
        pub fn flag(&self) -> bool {
            match self.0 {
                orb_api::Event::Camera(orb_api::CameraEvent::Changed { user, .. }) => user,
                orb_api::Event::Camera(orb_api::CameraEvent::AnimationEnd {
                    interrupted, ..
                }) => interrupted,
                orb_api::Event::Map(orb_api::MapEvent::SourceData { loaded, .. }) => loaded,
                orb_api::Event::Query { ref result, .. } => result.is_ok(),
                _ => false,
            }
        }

        /// Camera events: what changed (`Move` otherwise).
        #[diplomat::attr(contract, snapshot)]
        pub fn change(&self) -> OrbChange {
            match self.0 {
                orb_api::Event::Camera(orb_api::CameraEvent::Changed { change, .. }) => {
                    match change {
                        orb_api::Change::Move => OrbChange::Move,
                        orb_api::Change::Zoom => OrbChange::Zoom,
                        orb_api::Change::Rotate => OrbChange::Rotate,
                        orb_api::Change::Pitch => OrbChange::Pitch,
                    }
                }
                _ => OrbChange::Move,
            }
        }

        /// Camera events: start, progress or end (`End` otherwise).
        #[diplomat::attr(contract, snapshot)]
        pub fn phase(&self) -> OrbPhase {
            match self.0 {
                orb_api::Event::Camera(orb_api::CameraEvent::Changed { phase, .. }) => {
                    match phase {
                        orb_api::Phase::Start => OrbPhase::Start,
                        orb_api::Phase::Progress => OrbPhase::Progress,
                        orb_api::Phase::End => OrbPhase::End,
                    }
                }
                _ => OrbPhase::End,
            }
        }

        /// Clicks: the button (`Primary` otherwise).
        #[diplomat::attr(contract, snapshot)]
        pub fn button(&self) -> OrbButton {
            match self.0 {
                orb_api::Event::Click { button, .. } => match button {
                    orb_api::Button::Primary => OrbButton::Primary,
                    orb_api::Button::Secondary => OrbButton::Secondary,
                    orb_api::Button::Middle => OrbButton::Middle,
                },
                _ => OrbButton::Primary,
            }
        }

        /// Video requests: the URLs to try, one per line.
        #[diplomat::attr(contract, snapshot)]
        pub fn urls(&self, out: &mut DiplomatWrite) {
            if let orb_api::Event::Map(orb_api::MapEvent::Video {
                request: orb_api::VideoRequest::Open { urls },
                ..
            }) = &self.0
            {
                out.write_str(&urls.join("\n")).ok();
            }
        }

        /// Video seeks: the time in seconds (0 otherwise).
        #[diplomat::attr(contract, snapshot)]
        pub fn seconds(&self) -> f64 {
            match self.0 {
                orb_api::Event::Map(orb_api::MapEvent::Video {
                    request: orb_api::VideoRequest::Seek(seconds),
                    ..
                }) => seconds,
                _ => 0.0,
            }
        }

        /// Source data and video requests: the source's id. Errors: the
        /// message. Query results: the features' JSON, or the error.
        /// Capability: the tier. A missing image: its name.
        #[diplomat::attr(contract, snapshot)]
        pub fn text(&self, out: &mut DiplomatWrite) {
            let text = match &self.0 {
                orb_api::Event::Map(orb_api::MapEvent::Capability { tier }) => match tier {
                    orb_api::Tier::Modern => "modern",
                    orb_api::Tier::Compat => "compat",
                },
                orb_api::Event::Map(
                    orb_api::MapEvent::SourceData { source, .. }
                    | orb_api::MapEvent::Video { source, .. },
                ) => source,
                orb_api::Event::Map(orb_api::MapEvent::Error { message }) => message,
                orb_api::Event::Map(orb_api::MapEvent::StyleImageMissing { name }) => name,
                orb_api::Event::Query {
                    result: Ok(text) | Err(text),
                    ..
                } => text,
                _ => "",
            };
            out.write_str(text).ok();
        }
    }

    /// The topmost feature under a point.
    #[diplomat::opaque]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub struct OrbFeature(pub(crate) orb_api::Hit);

    impl OrbFeature {
        /// The style layer's id.
        #[diplomat::attr(contract, snapshot)]
        pub fn layer(&self, out: &mut DiplomatWrite) {
            out.write_str(&self.0.layer).ok();
        }

        /// The source's id.
        #[diplomat::attr(contract, snapshot)]
        pub fn source(&self, out: &mut DiplomatWrite) {
            out.write_str(&self.0.source).ok();
        }

        /// The source layer; empty for sources without layers.
        #[diplomat::attr(contract, snapshot)]
        pub fn source_layer(&self, out: &mut DiplomatWrite) {
            out.write_str(self.0.source_layer.as_deref().unwrap_or_default())
                .ok();
        }

        /// Whether the feature has an id.
        #[diplomat::attr(contract, snapshot)]
        pub fn has_id(&self) -> bool {
            self.0.id.is_some()
        }

        /// The feature's id as text; empty without one.
        #[diplomat::attr(contract, snapshot)]
        pub fn id(&self, out: &mut DiplomatWrite) {
            out.write_str(self.0.id.as_deref().unwrap_or_default()).ok();
        }

        /// The style generation the frame it was drawn in showed.
        #[diplomat::attr(contract, snapshot)]
        pub fn generation(&self) -> u64 {
            self.0.generation
        }
    }

    /// A picture a map drew: premultiplied RGBA8, rows top to bottom.
    #[diplomat::opaque]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub struct OrbImage(pub(crate) orb_api::Image);

    impl OrbImage {
        #[diplomat::attr(contract, snapshot)]
        pub fn width(&self) -> u32 {
            self.0.width
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn height(&self) -> u32 {
            self.0.height
        }

        #[diplomat::attr(contract, snapshot)]
        #[allow(clippy::needless_lifetimes)] // Diplomat needs them named.
        pub fn pixels<'a>(&'a self) -> &'a [u8] {
            &self.0.pixels
        }
    }

    /// A map: style, camera store, gestures and events. Draw it with
    /// `render` whenever `needs_frame`.
    #[diplomat::opaque_mut]
    #[diplomat::attr(contract, thread = "owner")]
    #[derive(Debug)]
    pub struct OrbMap(pub(crate) orb_api::Map);

    impl OrbMap {
        /// A map without a style; it takes the first style's camera unless
        /// moved first. `clear` is the premultiplied RGBA behind
        /// transparent style pixels.
        #[diplomat::attr(contract, lifecycle)]
        pub fn create(
            runtime: &OrbRuntime,
            feel: OrbFeel,
            clear_r: f64,
            clear_g: f64,
            clear_b: f64,
            clear_a: f64,
        ) -> Box<OrbMap> {
            Box::new(OrbMap(orb_api::Map::new(
                &runtime.0,
                orb_api::MapOptions {
                    camera: None,
                    feel: match feel {
                        OrbFeel::Desktop => orb_api::Feel::DESKTOP,
                        OrbFeel::Ios => orb_api::Feel::IOS,
                        OrbFeel::Android => orb_api::Feel::ANDROID,
                    },
                    clear_color: [clear_r, clear_g, clear_b, clear_a],
                },
            )))
        }

        /// Calls `wake` from any thread when data arrives and the map needs
        /// a frame: hosts that stop drawing while idle resume on it.
        /// Commands and input don't call it; check `needs_frame` after
        /// them. Replaces the previous callback.
        #[diplomat::attr(contract, command)]
        pub fn set_wake(&mut self, wake: impl Fn() + 'static) {
            let wake = crate::AnyThread(wake);
            self.0.set_wake(move || wake.call());
        }

        /// Whether a frame would show something new.
        #[diplomat::attr(contract, snapshot)]
        pub fn needs_frame(&self) -> bool {
            self.0.needs_frame()
        }

        /// Requests the tiles that failed again now: call it when the
        /// network comes back. Until then each is retried after a backoff
        /// while frames are drawn, with coarser tiles standing in.
        #[diplomat::attr(contract, command)]
        pub fn retry_failed_tiles(&mut self) {
            self.0.retry_failed_tiles();
        }

        /// Loads a style from a URL (`http(s)://`, `file://`).
        #[diplomat::attr(contract, command)]
        pub fn load_style_url(&mut self, url: &DiplomatStr) {
            self.0.load_style_url(&text(url));
        }

        /// Replaces the style with a v8 style document.
        #[diplomat::attr(contract, command)]
        pub fn set_style_json(&mut self, json: &DiplomatStr) -> Result<(), OrbError> {
            self.0.set_style_json(&text(json)).map_err(fail)
        }

        /// Applies a transaction of GL JS map calls as JSON, all at once;
        /// returns the generation from which frames show it.
        #[diplomat::attr(contract, command)]
        pub fn edit(&mut self, commands: &DiplomatStr) -> Result<u64, OrbError> {
            self.0.edit(&text(commands)).map_err(fail)
        }

        /// Whether this host plays `video` sources: it then gets `Video*`
        /// events and hands frames to `video_frame`. Without it, video
        /// sources draw nothing, with a warning.
        #[diplomat::attr(contract, command)]
        pub fn set_plays_video(&mut self, plays: bool) {
            self.0.set_plays_video(plays);
        }

        /// The frame the video of source `id` shows now: straight
        /// (not premultiplied) RGBA8 rows, top first. It replaces any frame
        /// not yet drawn.
        #[diplomat::attr(contract, command)]
        pub fn video_frame(
            &mut self,
            id: &DiplomatStr,
            width: u32,
            height: u32,
            pixels: &[u8],
        ) -> Result<(), OrbError> {
            self.0
                .video_frame(&text(id), width, height, pixels.to_vec())
                .map_err(fail)
        }

        /// The video of source `id` can't play here: it draws nothing.
        #[diplomat::attr(contract, command)]
        pub fn video_failed(
            &mut self,
            id: &DiplomatStr,
            message: &DiplomatStr,
        ) -> Result<(), OrbError> {
            self.0.video_failed(&text(id), &text(message)).map_err(fail)
        }

        /// Plays the video of source `id` (GL JS `VideoSource.play`): a
        /// `VideoPlay` event for the host.
        #[diplomat::attr(contract, command)]
        pub fn play_video(&mut self, id: &DiplomatStr) -> Result<(), OrbError> {
            self.0
                .control_video(&text(id), orb_api::VideoRequest::Play)
                .map_err(fail)
        }

        /// Pauses the video of source `id`.
        #[diplomat::attr(contract, command)]
        pub fn pause_video(&mut self, id: &DiplomatStr) -> Result<(), OrbError> {
            self.0
                .control_video(&text(id), orb_api::VideoRequest::Pause)
                .map_err(fail)
        }

        /// Seeks the video of source `id` to `seconds`.
        #[diplomat::attr(contract, command)]
        pub fn seek_video(&mut self, id: &DiplomatStr, seconds: f64) -> Result<(), OrbError> {
            self.0
                .control_video(&text(id), orb_api::VideoRequest::Seek(seconds))
                .map_err(fail)
        }

        /// Adds an image to the style for icons, text and patterns (GL JS
        /// `addImage`): straight (not premultiplied) RGBA8 rows, top first,
        /// and GL JS's options as JSON in sprite metadata's spelling
        /// (`{"pixelRatio": 2, "sdf": true}`), empty for the defaults. An
        /// error when the style has an image by that name.
        #[diplomat::attr(contract, command)]
        pub fn add_image(
            &mut self,
            name: &DiplomatStr,
            width: u32,
            height: u32,
            pixels: &[u8],
            options: &DiplomatStr,
        ) -> Result<(), OrbError> {
            let image =
                orb_api::StyleImage::with_options(width, height, pixels.to_vec(), &text(options))
                    .map_err(|e| fail(orb_api::Error::InvalidArgument(e)))?;
            self.0.add_image(&text(name), image).map_err(fail)
        }

        /// Replaces an image's pixels, of the same size (GL JS
        /// `updateImage`); its options stay.
        #[diplomat::attr(contract, command)]
        pub fn update_image(
            &mut self,
            name: &DiplomatStr,
            width: u32,
            height: u32,
            pixels: &[u8],
        ) -> Result<(), OrbError> {
            self.0
                .update_image(&text(name), width, height, pixels.to_vec())
                .map_err(fail)
        }

        /// Removes an image from the style (GL JS `removeImage`).
        #[diplomat::attr(contract, command)]
        pub fn remove_image(&mut self, name: &DiplomatStr) -> Result<(), OrbError> {
            self.0.remove_image(&text(name)).map_err(fail)
        }

        /// Whether the style has an image by this name, from its sprites
        /// or added (GL JS `hasImage`).
        #[diplomat::attr(contract, snapshot)]
        pub fn has_image(&self, name: &DiplomatStr) -> bool {
            self.0.has_image(&text(name))
        }

        /// The style generation, counting loads and edits.
        #[diplomat::attr(contract, snapshot)]
        pub fn generation(&self) -> u64 {
            self.0.generation()
        }

        /// The camera as it is now.
        #[diplomat::attr(contract, snapshot)]
        pub fn camera(&self) -> OrbCamera {
            let c = self.0.camera();
            OrbCamera {
                lon: c.target.lon,
                lat: c.target.lat,
                zoom: self.0.zoom(),
                bearing: c.bearing,
                pitch: c.pitch,
                roll: c.roll,
                elevation: c.elevation,
            }
        }

        /// Whether the centre stands on the terrain at rest (GL JS
        /// `centerClampedToGround`, on by default). Off, it keeps the
        /// elevation it is given and the camera orbits a point in the air.
        #[diplomat::attr(contract, command)]
        pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
            self.0.set_center_clamped_to_ground(clamped);
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn center_clamped_to_ground(&self) -> bool {
            self.0.center_clamped_to_ground()
        }

        /// Puts the centre this many metres above sea level at once (GL JS
        /// `setCenterElevation`); while clamped, the terrain takes it back.
        #[diplomat::attr(contract, command)]
        pub fn set_center_elevation(&mut self, elevation: f64) {
            self.0.set_center_elevation(elevation);
        }

        /// Moves the camera at once.
        #[diplomat::attr(contract, command)]
        pub fn jump_to(&mut self, target: OrbCameraTarget) {
            self.0.jump_to(&crate::camera_target(target));
        }

        /// Eases the camera there over `duration_ms`; returns the
        /// animation's id.
        #[diplomat::attr(contract, command)]
        pub fn ease_to(&mut self, target: OrbCameraTarget, duration_ms: f64) -> u64 {
            let options = orb_api::EaseOptions {
                duration: duration_ms.max(0.0),
                ..Default::default()
            };
            self.0.ease_to(&crate::camera_target(target), &options).0
        }

        /// Flies the camera there, zooming out on the way; NaN
        /// `duration_ms` picks the duration from the distance.
        #[diplomat::attr(contract, command)]
        pub fn fly_to(&mut self, target: OrbCameraTarget, duration_ms: f64) -> u64 {
            let options = orb_api::FlyOptions {
                duration: (!duration_ms.is_nan()).then_some(duration_ms.max(0.0)),
                ..Default::default()
            };
            self.0.fly_to(&crate::camera_target(target), &options).0
        }

        /// Eases the camera to show `bounds` with `padding` logical pixels
        /// around them; returns the animation's id, or 0 when the bounds
        /// can't be shown.
        #[diplomat::attr(contract, command)]
        pub fn fit_bounds(&mut self, bounds: OrbBounds, padding: f64, duration_ms: f64) -> u64 {
            let options = orb_api::FitOptions {
                padding: orb_api::Padding::uniform(padding),
                ..Default::default()
            };
            let bounds = orb_api::Bounds {
                west: bounds.west,
                south: bounds.south,
                east: bounds.east,
                north: bounds.north,
            };
            match self.0.camera_for_bounds(bounds, &options) {
                Some(target) => {
                    let ease = orb_api::EaseOptions {
                        duration: duration_ms.max(0.0),
                        ..Default::default()
                    };
                    self.0.ease_to(&target, &ease).0
                }
                None => 0,
            }
        }

        /// Stops any camera animation.
        #[diplomat::attr(contract, command)]
        pub fn stop(&mut self) {
            self.0.stop();
        }

        /// A pointer went down, in logical pixels. Returns whether the map
        /// used the input.
        #[diplomat::attr(contract, command)]
        pub fn pointer_down(
            &mut self,
            id: u32,
            kind: OrbPointerKind,
            button: OrbButton,
            x: f64,
            y: f64,
            modifiers: OrbModifiers,
        ) -> bool {
            self.0.input(orb_api::Input::PointerDown {
                id,
                kind: match kind {
                    OrbPointerKind::Mouse => orb_api::PointerKind::Mouse,
                    OrbPointerKind::Touch => orb_api::PointerKind::Touch,
                    OrbPointerKind::Pen => orb_api::PointerKind::Pen,
                },
                button: match button {
                    OrbButton::Primary => orb_api::Button::Primary,
                    OrbButton::Secondary => orb_api::Button::Secondary,
                    OrbButton::Middle => orb_api::Button::Middle,
                },
                position: glam::DVec2::new(x, y),
                modifiers: crate::modifiers(modifiers),
            })
        }

        #[diplomat::attr(contract, command)]
        pub fn pointer_move(&mut self, id: u32, x: f64, y: f64) -> bool {
            self.0.input(orb_api::Input::PointerMove {
                id,
                position: glam::DVec2::new(x, y),
            })
        }

        #[diplomat::attr(contract, command)]
        pub fn pointer_up(&mut self, id: u32, x: f64, y: f64) -> bool {
            self.0.input(orb_api::Input::PointerUp {
                id,
                position: glam::DVec2::new(x, y),
            })
        }

        /// The platform took the pointer away: no click, no inertia.
        #[diplomat::attr(contract, command)]
        pub fn pointer_cancel(&mut self, id: u32) -> bool {
            self.0.input(orb_api::Input::PointerCancel { id })
        }

        /// A wheel or trackpad scroll; positive `dy` zooms out.
        #[diplomat::attr(contract, command)]
        pub fn wheel(
            &mut self,
            x: f64,
            y: f64,
            dx: f64,
            dy: f64,
            unit: OrbWheelUnit,
            modifiers: OrbModifiers,
        ) -> bool {
            self.0.input(orb_api::Input::Wheel {
                position: glam::DVec2::new(x, y),
                delta: glam::DVec2::new(dx, dy),
                unit: match unit {
                    OrbWheelUnit::Pixel => orb_api::WheelUnit::Pixel,
                    OrbWheelUnit::Line => orb_api::WheelUnit::Line,
                    OrbWheelUnit::Page => orb_api::WheelUnit::Page,
                },
                modifiers: crate::modifiers(modifiers),
            })
        }

        #[diplomat::attr(contract, command)]
        pub fn key(&mut self, key: OrbKey, modifiers: OrbModifiers) -> bool {
            self.0.input(orb_api::Input::Key {
                key: match key {
                    OrbKey::Left => orb_api::Key::Left,
                    OrbKey::Right => orb_api::Key::Right,
                    OrbKey::Up => orb_api::Key::Up,
                    OrbKey::Down => orb_api::Key::Down,
                    OrbKey::ZoomIn => orb_api::Key::ZoomIn,
                    OrbKey::ZoomOut => orb_api::Key::ZoomOut,
                },
                modifiers: crate::modifiers(modifiers),
            })
        }

        /// The view lost focus: every pointer is released.
        #[diplomat::attr(contract, command)]
        pub fn blur(&mut self) -> bool {
            self.0.input(orb_api::Input::Blur)
        }

        /// The topmost feature at a point in logical pixels, from the last
        /// frame drawn.
        #[diplomat::attr(contract, query)]
        pub fn hit_test(&self, x: f64, y: f64) -> Option<Box<OrbFeature>> {
            self.0
                .hit_test(glam::DVec2::new(x, y))
                .map(|hit| Box::new(OrbFeature(hit)))
        }

        /// GL JS `queryRenderedFeatures` over a box in logical pixels (a
        /// point when the corners are equal), in the last frame drawn.
        /// Returns the query's id; a `QueryResult` event with it brings the
        /// features, and the map wakes when it arrives.
        #[diplomat::attr(contract, query)]
        pub fn query_rendered(&mut self, x0: f64, y0: f64, x1: f64, y1: f64) -> u64 {
            self.0
                .query_rendered(glam::DVec2::new(x0, y0), glam::DVec2::new(x1, y1))
        }

        /// Where a point is in the viewport, in logical pixels.
        #[diplomat::attr(contract, query)]
        pub fn project(&self, lon: f64, lat: f64) -> OrbPoint {
            match self.0.project(orb_api::Geodetic::new(lon, lat)) {
                Some(p) => OrbPoint {
                    x: p.x,
                    y: p.y,
                    valid: true,
                },
                None => OrbPoint {
                    x: f64::NAN,
                    y: f64::NAN,
                    valid: false,
                },
            }
        }

        /// The point under a viewport position in logical pixels.
        #[diplomat::attr(contract, query)]
        pub fn unproject(&self, x: f64, y: f64) -> OrbLngLat {
            match self.0.unproject(glam::DVec2::new(x, y)) {
                Some(p) => OrbLngLat {
                    lon: p.lon,
                    lat: p.lat,
                    valid: true,
                },
                None => OrbLngLat {
                    lon: f64::NAN,
                    lat: f64::NAN,
                    valid: false,
                },
            }
        }

        /// Draws what the camera shows in a `width × height` view (logical
        /// pixels, `scale` physical ones each) once everything in it has
        /// loaded and nothing fades: the same style, data and camera always
        /// draw the same bytes. Blocks until then; an error after a minute.
        #[diplomat::attr(contract, query)]
        pub fn snapshot(
            &mut self,
            width: u32,
            height: u32,
            scale: f64,
        ) -> Result<Box<OrbImage>, OrbError> {
            self.0
                .snapshot(width, height, scale)
                .map(|image| Box::new(OrbImage(image)))
                .map_err(fail)
        }

        /// Draws tile `z`/`col`/`row` of the tile matrix set
        /// `tile_matrix_set` (an OGC identifier, such as `WebMercatorQuad`)
        /// at `size` logical pixels as still output, labels placed so that
        /// tiles drawn apart agree across their edges. The camera is left
        /// as it was.
        #[diplomat::attr(contract, query)]
        #[allow(clippy::too_many_arguments)]
        pub fn render_tile(
            &mut self,
            tile_matrix_set: &DiplomatStr,
            z: u8,
            col: u32,
            row: u32,
            size: u32,
            scale: f64,
        ) -> Result<Box<OrbImage>, OrbError> {
            let id = text(tile_matrix_set);
            let tile_matrix_set = orb_api::TileMatrixSet::by_id(&id).map_err(|_| {
                fail(orb_api::Error::InvalidArgument(format!(
                    "unknown tile matrix set {id}"
                )))
            })?;
            self.0
                .render_tile(&orb_api::TileRequest {
                    tile_matrix_set,
                    z,
                    col,
                    row,
                    size,
                    scale,
                })
                .map(|image| Box::new(OrbImage(image)))
                .map_err(fail)
        }

        /// Draws a frame into `target`: advances gestures and animations,
        /// applies arrived data and renders. A map draws targets of one
        /// format: all texture targets, or all surfaces of a platform.
        #[diplomat::attr(contract, render_driver)]
        pub fn render(&mut self, target: &mut OrbTarget) -> Result<OrbFrame, OrbError> {
            Ok(match self.0.render(&mut target.target).map_err(fail)? {
                None => OrbFrame::default(),
                Some(report) => OrbFrame {
                    drawn: true,
                    generation: report.generation,
                    idle: report.idle,
                    loaded: report.loaded,
                    tiles_drawn: report.tiles_drawn,
                    tiles_loading: report.tiles_loading,
                    draws: report.frame.draws,
                    dispatches: report.frame.dispatches,
                    pipelines_pending: report.frame.pipelines_pending,
                    layers_waiting: report.frame.nodes_waiting,
                },
            })
        }

        /// The oldest event not yet taken.
        #[diplomat::attr(contract, event_batch)]
        pub fn poll_event(&mut self) -> Option<Box<OrbEvent>> {
            self.0.poll_event().map(|e| Box::new(OrbEvent(e)))
        }

        /// Warnings since the last call, one per line.
        #[diplomat::attr(contract, event_batch)]
        pub fn take_warnings(&mut self, out: &mut DiplomatWrite) {
            for warning in self.0.take_warnings() {
                writeln!(out, "{warning}").ok();
            }
        }
    }
}

use std::cell::RefCell;
use std::ptr::NonNull;

use ffi::{OrbError, OrbEventKind};

/// Bumped on every incompatible change to the C ABI while it is 0.x.
const ABI_VERSION: u32 = 3;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Records an error's message for `OrbError::last_message` and returns its
/// status.
fn fail(error: orb_api::Error) -> OrbError {
    let status = match &error {
        orb_api::Error::InvalidArgument(_) => OrbError::InvalidArgument,
        orb_api::Error::InvalidState(_) => OrbError::InvalidState,
        orb_api::Error::Unsupported(_) => OrbError::Unsupported,
        orb_api::Error::Native(_) => OrbError::Native,
        orb_api::Error::TargetLost(_) => OrbError::TargetLost,
        orb_api::Error::DeviceLost => OrbError::DeviceLost,
    };
    LAST_ERROR.with(|e| *e.borrow_mut() = error.message().to_owned());
    status
}

/// UTF-8 from the caller, with invalid sequences replaced.
fn text(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    String::from_utf8_lossy(bytes)
}

/// A callback the caller promises can run on any thread (`OrbMap::set_wake`).
struct AnyThread<F>(F);
// SAFETY: only Diplomat's C entry point builds these (the bridge module is
// crate-private), from a C callback whose contract is that it may be called
// from any thread until it is replaced or the map is destroyed; the
// generated bindings pass thread-safe closures.
unsafe impl<F> Send for AnyThread<F> {}
// SAFETY: as above; the callback takes no arguments and shares no Rust
// state.
unsafe impl<F> Sync for AnyThread<F> {}

impl<F: Fn()> AnyThread<F> {
    fn call(&self) {
        (self.0)()
    }
}

fn raw_target(
    runtime: &ffi::OrbRuntime,
    address: usize,
    kind: fn(NonNull<std::ffi::c_void>) -> orb_api::RawSurface,
    width: u32,
    height: u32,
    scale: f64,
) -> Result<Box<ffi::OrbTarget>, OrbError> {
    let pointer = NonNull::new(address as *mut std::ffi::c_void)
        .ok_or_else(|| fail(orb_api::Error::InvalidArgument("null handle".into())))?;
    // SAFETY: the C ABI's contract is that the handle is a live object of
    // this kind that outlives the target; on Compat, an `ANativeWindow*`
    // is an EGL native window.
    let surface = runtime
        .0
        .with_device(|device| match device {
            orb_api::DeviceRef::Modern(gpu) => unsafe {
                gpu.create_surface_raw(kind(pointer), width, height)
                    .map(orb_api::Surface::Modern)
            },
            orb_api::DeviceRef::Compat(egl) => match kind(pointer) {
                orb_api::RawSurface::AndroidWindow(window) => unsafe {
                    egl.create_window(window.as_ptr())
                        .map(orb_api::Surface::compat)
                },
                orb_api::RawSurface::MetalLayer(_) => {
                    Err("Compat draws into Android windows, not Metal layers".into())
                }
            },
        })
        .map_err(|e| fail(orb_api::Error::Unsupported(e)))?;
    Ok(Box::new(ffi::OrbTarget {
        target: orb_api::Target::surface(&runtime.0, surface, scale),
        pixels: Vec::new(),
    }))
}

fn camera_target(t: ffi::OrbCameraTarget) -> orb_api::CameraTarget {
    let some = |v: f64| (!v.is_nan()).then_some(v);
    orb_api::CameraTarget {
        center: (!t.lon.is_nan() && !t.lat.is_nan()).then(|| orb_api::Geodetic::new(t.lon, t.lat)),
        zoom: some(t.zoom),
        bearing: some(t.bearing),
        pitch: some(t.pitch),
        roll: some(t.roll),
        elevation: some(t.elevation),
        ..Default::default()
    }
}

fn modifiers(m: ffi::OrbModifiers) -> orb_api::Modifiers {
    orb_api::Modifiers {
        shift: m.shift,
        ctrl: m.ctrl,
        alt: m.alt,
        meta: m.meta,
    }
}

fn event_kind(event: &orb_api::Event) -> OrbEventKind {
    use orb_api::{CameraEvent, Event, MapEvent};
    match event {
        Event::Map(MapEvent::StyleLoad { .. }) => OrbEventKind::StyleLoad,
        Event::Map(MapEvent::StyleData { .. }) => OrbEventKind::StyleData,
        Event::Map(MapEvent::SourceData { .. }) => OrbEventKind::SourceData,
        Event::Map(MapEvent::Load) => OrbEventKind::Load,
        Event::Map(MapEvent::Idle) => OrbEventKind::Idle,
        Event::Map(MapEvent::Error { .. }) => OrbEventKind::Error,
        Event::Map(MapEvent::Capability { .. }) => OrbEventKind::Capability,
        Event::Map(MapEvent::StyleImageMissing { .. }) => OrbEventKind::StyleImageMissing,
        Event::Map(MapEvent::Video { request, .. }) => match request {
            orb_api::VideoRequest::Open { .. } => OrbEventKind::VideoOpen,
            orb_api::VideoRequest::Play => OrbEventKind::VideoPlay,
            orb_api::VideoRequest::Pause => OrbEventKind::VideoPause,
            orb_api::VideoRequest::Seek(_) => OrbEventKind::VideoSeek,
            orb_api::VideoRequest::Close => OrbEventKind::VideoClose,
        },
        Event::Camera(CameraEvent::Changed { .. }) => OrbEventKind::Camera,
        Event::Camera(CameraEvent::AnimationEnd { .. }) => OrbEventKind::AnimationEnd,
        Event::Click { .. } => OrbEventKind::Click,
        Event::DoubleClick { .. } => OrbEventKind::DoubleClick,
        Event::Query { .. } => OrbEventKind::QueryResult,
    }
}

fn event_position(event: &orb_api::Event) -> glam::DVec2 {
    match event {
        orb_api::Event::Click { position, .. } | orb_api::Event::DoubleClick { position } => {
            *position
        }
        _ => glam::DVec2::NAN,
    }
}
