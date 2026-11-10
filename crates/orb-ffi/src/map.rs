// Maps: an Orbis map on its runtime's thread, with the state the C API
// reports about it (mode, extent, masks, debug flags, viewport options), its
// generation, and the snapshot the runtime thread publishes after each piece
// of work for `mln_map_get_snapshot` to copy without waiting.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use orb_api::{
    AnimationId, CameraTarget, EaseOptions, Easing, FitOptions, FlyOptions, Geodetic, Phase,
};

use crate::base::{Failure, Result, UserData, boundary, fresh_out, put, sized, text};
use crate::delivery::{self, Completion, Value};
use crate::handles::{self, Object};
use crate::runtime::{Msg, Payload, RuntimeHandle, State};
use crate::sys::*;

/// The caller's side of a map.
pub(crate) struct MapHandle {
    pub(crate) runtime: Arc<RuntimeHandle>,
    pub(crate) id: OnceLock<u64>,
    /// The state the runtime thread published last; `None` until the map
    /// is made.
    snapshot: Mutex<Option<mln_map_snapshot>>,
    /// Whether a render session is attached, for release to refuse at once.
    pub(crate) has_session: AtomicBool,
}

impl MapHandle {
    pub(crate) fn id(&self) -> u64 {
        *self.id.get().expect("map id")
    }

    pub(crate) fn snapshot(&self) -> Option<mln_map_snapshot> {
        *self
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A camera transition's end handler, waiting for its animation.
struct Transition {
    animation: Option<AnimationId>,
    id: Option<u64>,
    handler: EndHandler,
}

#[derive(Clone, Copy)]
pub(crate) struct EndHandler {
    callback: unsafe extern "C" fn(*mut std::ffi::c_void, *const mln_camera_transition_end),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
}

impl EndHandler {
    fn from_c(h: &mln_camera_transition_handler) -> Result<Option<Self>> {
        let Some(callback) = h.callback else {
            if h.release_user_data.is_some() || h.queue.0 != 0 {
                return Err(Failure::argument(
                    "the end handler has no callback but a release or a queue",
                ));
            }
            return Ok(None);
        };
        delivery::check_queue(h.queue.0, "end_handler.queue")?;
        Ok(Some(Self {
            callback,
            user_data: UserData(h.user_data),
            release_user_data: h.release_user_data,
            queue: h.queue.0,
        }))
    }

    fn end(self, outcome: mln_camera_transition_outcome, generation: u64) {
        delivery::deliver(
            self.queue,
            Box::new(move || {
                // SAFETY: plain C data.
                let mut end: mln_camera_transition_end = unsafe { std::mem::zeroed() };
                end.size = std::mem::size_of::<mln_camera_transition_end>() as u32;
                end.outcome = outcome;
                end.generation = generation;
                // SAFETY: the caller's registration; `end` lives until it
                // returns.
                unsafe { (self.callback)(self.user_data.get(), &end) };
            }),
        );
        delivery::release_later(self.queue, self.release_user_data, self.user_data);
    }
}

pub(crate) struct MapEntry {
    pub(crate) map: orb_api::Map,
    handle: Arc<MapHandle>,
    pub(crate) mode: mln_map_mode,
    pub(crate) extent: mln_logical_extent,
    pub(crate) event_mask: u64,
    debug_flags: mln_map_debug_flags,
    stats_view: bool,
    viewport: mln_map_viewport,
    bounds: mln_camera_bounds,
    prefetch_zoom_delta: u32,
    pub(crate) style_url: String,
    /// The style text as loaded, without later edits.
    loaded_style: Option<String>,
    pub(crate) session: Option<u64>,
    /// Grows with each change to the map's state.
    pub(crate) generation: u64,
    pub(crate) render_update_generation: u64,
    /// Data arrived since the map last took it in.
    woken: Arc<AtomicBool>,
    /// A style or source change awaits its loads, for LOADING_FINISHED.
    loading: bool,
    /// The STYLE_LOADED of the current style has been raised.
    style_loaded: bool,
    /// Completions of URL style loads, waiting for the style.
    style_waiters: Vec<Completion>,
    transitions: Vec<Transition>,
    /// What ends the transitions an interruption ends: a cancel, or by
    /// default a replacement.
    ending: mln_camera_transition_outcome,
    /// Command groups open: while any is, a session draws none of the
    /// map's changes, so they show together once the outermost ends.
    pub(crate) groups: u32,
    /// Draws the map's frames while no session does, so animations run.
    idle: Option<orb_api::Target>,
    /// What the C API keeps of the style beyond Orbis's document.
    pub(crate) style: crate::style::StyleState,
    /// Feature queries waiting for Orbis's answer, by its query id, and
    /// the ids of a query's further parts.
    pub(crate) queries: std::collections::HashMap<u64, crate::queries::Query>,
    pub(crate) query_aliases: std::collections::HashMap<u64, u64>,
}

impl MapEntry {
    pub(crate) fn camera(&self) -> mln_camera {
        let c = self.map.camera();
        mln_camera {
            center: mln_lat_lng {
                latitude: c.target.lat,
                longitude: c.target.lon,
            },
            center_altitude: c.elevation,
            padding: mln_edge_insets {
                top: c.padding.top,
                left: c.padding.left,
                bottom: c.padding.bottom,
                right: c.padding.right,
            },
            zoom: self.map.zoom(),
            bearing: c.bearing,
            pitch: c.pitch,
            roll: c.roll,
            field_of_view: c.fov,
        }
    }

    pub(crate) fn view(&self) -> mln_map_view {
        mln_map_view {
            size: std::mem::size_of::<mln_map_view>() as u32,
            camera: self.camera(),
            extent: self.extent,
            // SAFETY: plain C data; zero is Native's default projection.
            projection_mode: unsafe { std::mem::zeroed() },
            viewport: self.viewport,
        }
    }

    /// Publishes the snapshot `mln_map_get_snapshot` copies.
    pub(crate) fn publish(&self) {
        let lod = self.map.tile_lod();
        let snapshot = mln_map_snapshot {
            size: std::mem::size_of::<mln_map_snapshot>() as u32,
            debug_flags: self.debug_flags,
            generation: self.generation,
            fully_loaded: self.style_loaded && !self.loading,
            rendering_stats_view_enabled: self.stats_view,
            gesture_in_progress: false,
            transition_in_progress: self.map.is_moving() || self.map.transitioning(),
            event_mask: self.event_mask,
            latest_render_update_generation: self.render_update_generation,
            tile: mln_map_tile_settings {
                prefetch_zoom_delta: self.prefetch_zoom_delta,
                lod_min_radius: lod.min_radius,
                lod_scale: lod.scale,
                lod_pitch_threshold: lod.pitch_threshold,
                lod_zoom_shift: lod.zoom_shift,
                lod_mode: match lod.mode {
                    orb_api::TileLodMode::Default => MLN_TILE_LOD_MODE_DEFAULT,
                    orb_api::TileLodMode::Distance => MLN_TILE_LOD_MODE_DISTANCE,
                },
            },
            bounds: self.bounds,
            // SAFETY: plain C data.
            free_camera: unsafe { std::mem::zeroed() },
            view: self.view(),
        };
        *self
            .handle
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(snapshot);
    }

    /// Marks whether a session draws the map, on the caller's side.
    pub(crate) fn handle_flag(&self, attached: bool) {
        self.handle.has_session.store(attached, Ordering::Release);
    }

    /// Whether the map moves or loads with nothing drawing it.
    fn busy(&self) -> bool {
        self.session.is_none() && (self.map.is_moving() || self.woken.load(Ordering::Acquire))
    }
}

/// Whether any map needs the runtime thread to step it.
pub(crate) fn any_busy(state: &State) -> bool {
    state.maps.values().any(MapEntry::busy)
}

/// Steps the maps no session draws: takes in what arrived, and draws a
/// frame for a moving camera so its animation runs.
pub(crate) fn advance(state: &mut State) {
    let now = state.now();
    for entry in state.maps.values_mut() {
        if entry.session.is_some() {
            continue;
        }
        if entry.map.is_moving() {
            let (w, h) = (entry.extent.width.max(1), entry.extent.height.max(1));
            let target = entry
                .idle
                .get_or_insert_with(|| orb_api::Target::texture(&state.runtime, w, h, 1.0));
            if target.size() != (w, h) {
                target.resize(w, h, 1.0);
            }
            let _ = entry.map.render(target, now);
        } else if entry.woken.swap(false, Ordering::AcqRel) {
            let _ = entry.map.apply_arrivals();
        }
    }
}

/// Moves Orbis's events for a map into the runtime's events, and ends the
/// transitions and style loads they settle.
pub(crate) fn take_events(state: &mut State, id: u64) {
    let mut out = Vec::new();
    let mut ended = Vec::new();
    let mut waiters = Vec::new();
    {
        let Some(entry) = state.maps.get_mut(&id) else {
            return;
        };
        let mut camera_phases = [false; 3];
        while let Some(event) = entry.map.poll_event() {
            let event = match event {
                orb_api::Event::Map(event) => event,
                orb_api::Event::Query { id, result } => {
                    let id = entry.query_aliases.remove(&id).unwrap_or(id);
                    if let Some(query) = entry.queries.remove(&id)
                        && let Some(waiting) = crate::queries::answer(query, result)
                    {
                        entry.queries.insert(id, waiting);
                    }
                    continue;
                }
            };
            use orb_api::MapEvent as E;
            match event {
                E::StyleLoad => {
                    if !entry.style_loaded {
                        entry.style_loaded = true;
                        out.push((
                            MLN_RUNTIME_EVENT_MAP_STYLE_LOADED,
                            Payload::None,
                            String::new(),
                        ));
                    }
                    entry.loaded_style =
                        entry.loaded_style.take().or_else(|| entry.map.style_json());
                    waiters.extend(entry.style_waiters.drain(..).map(|c| (c, Ok(Value::None))));
                }
                E::Load | E::Idle => {
                    if std::mem::take(&mut entry.loading) {
                        out.push((
                            MLN_RUNTIME_EVENT_MAP_LOADING_FINISHED,
                            Payload::None,
                            String::new(),
                        ));
                    }
                    if matches!(event, E::Idle) {
                        out.push((MLN_RUNTIME_EVENT_MAP_IDLE, Payload::None, String::new()));
                    }
                }
                E::StyleImageMissing { name } => out.push((
                    MLN_RUNTIME_EVENT_MAP_STYLE_IMAGE_MISSING,
                    Payload::StyleImage(name),
                    String::new(),
                )),
                // A failure before the style loaded is the style's; later
                // ones are a source's, a tile's or a sprite's, which
                // MapLibre Native logs.
                E::Error { message } if !entry.style_loaded => {
                    out.push((
                        MLN_RUNTIME_EVENT_MAP_LOADING_FAILED,
                        Payload::LoadingFailed(MLN_MAP_LOAD_ERROR_STYLE_LOAD),
                        message.clone(),
                    ));
                    waiters.extend(entry.style_waiters.drain(..).map(|c| {
                        (
                            c,
                            Err(Failure::native(format!(
                                "the style failed to load: {message}"
                            ))),
                        )
                    }));
                }
                E::Error { message } => crate::logging::log(MLN_LOG_SEVERITY_ERROR, &message),
                E::Warning { message } => crate::logging::log(MLN_LOG_SEVERITY_WARNING, &message),
                E::Camera(orb_api::CameraEvent::Changed { phase, .. }) => {
                    let (kind, i) = match phase {
                        Phase::Start => (MLN_RUNTIME_EVENT_MAP_CAMERA_WILL_CHANGE, 0),
                        Phase::Progress => (MLN_RUNTIME_EVENT_MAP_CAMERA_IS_CHANGING, 1),
                        Phase::End => (MLN_RUNTIME_EVENT_MAP_CAMERA_DID_CHANGE, 2),
                    };
                    if !std::mem::replace(&mut camera_phases[i], true) {
                        let payload = Payload::CameraChange(
                            MLN_CAMERA_CHANGE_MODE_ANIMATED,
                            MLN_CAMERA_CHANGE_SOURCE_COMMAND,
                        );
                        if i == 1 {
                            out.push((kind, Payload::None, String::new()));
                        } else {
                            out.push((kind, payload, String::new()));
                        }
                    }
                }
                E::Camera(orb_api::CameraEvent::AnimationEnd {
                    id, interrupted, ..
                }) => {
                    let outcome = if interrupted {
                        entry.ending
                    } else {
                        MLN_CAMERA_TRANSITION_OUTCOME_COMPLETED
                    };
                    let mut i = 0;
                    while i < entry.transitions.len() {
                        if entry.transitions[i].animation == Some(id) {
                            ended.push((entry.transitions.remove(i).handler, outcome));
                        } else {
                            i += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        // Transitions whose animation never started (a zero duration) end
        // with the camera at rest.
        if !entry.map.is_moving() {
            for t in entry.transitions.drain(..) {
                ended.push((t.handler, MLN_CAMERA_TRANSITION_OUTCOME_COMPLETED));
            }
        }
        entry.ending = MLN_CAMERA_TRANSITION_OUTCOME_REPLACED;
        if !ended.is_empty() || !out.is_empty() {
            entry.generation += 1;
        }
    }
    for (kind, payload, message) in out {
        state.event(id, kind, payload, message);
    }
    let generation = state.maps.get(&id).map_or(0, |m| m.generation);
    for (handler, outcome) in ended {
        handler.end(outcome, generation);
    }
    for (completion, result) in waiters {
        completion.complete(result, generation);
    }
}

/// Posts `job` with the map to its runtime thread, then completes. A
/// command (`required` false) may have no completion; queries,
/// operations and lifecycle calls must. `changes` commands grow the map's
/// generation.
///
/// # Safety
/// `completion` is null or a readable `mln_completion`.
pub(crate) unsafe fn post(
    map: mln_map,
    completion: *const mln_completion,
    required: bool,
    changes: bool,
    job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
) -> Result {
    let handle = handles::map(map.0)?;
    // SAFETY: the caller's contract.
    let completion = unsafe {
        if required {
            Some(Completion::required(completion)?)
        } else {
            Completion::from_c(completion)?
        }
    };
    let id = handle.id();
    handle.runtime.post(move |state| {
        let result = if state.maps.contains_key(&id) {
            job(state, id)
        } else {
            Err(Failure::state(
                "the map isn't made: its creation failed or it was released",
            ))
        };
        if changes
            && result.is_ok()
            && let Some(entry) = state.maps.get_mut(&id)
        {
            entry.generation += 1;
        }
        state.settle();
        let generation = state.maps.get(&id).map_or(0, |m| m.generation);
        state.complete(completion, result, generation);
    })
}

/// The map's entry, inside a job.
pub(crate) fn entry(state: &mut State, id: u64) -> &mut MapEntry {
    state.maps.get_mut(&id).expect("the job checked the map")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_create_map(
    runtime: mln_runtime,
    options: *const mln_map_options,
    out_map: *mut mln_map,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let runtime = handles::runtime(runtime.0)?;
            let options = *sized(options, "options")?;
            crate::base::physical(options.initial_extent)?;
            if options.map_mode > MLN_MAP_MODE_TILE {
                return Err(Failure::argument(format!(
                    "{} names no map mode",
                    options.map_mode
                )));
            }
            let event_mask = if options.fields & MLN_MAP_OPTION_EVENT_MASK != 0 {
                check_mask(options.event_mask)?;
                options.event_mask
            } else {
                MLN_RUNTIME_EVENT_FLAG_ALL
            };
            let fonts = text(options.fallback_font_faces, "fallback_font_faces")?;
            if !fonts.trim().is_empty() && !fonts.trim_start().starts_with('[') {
                return Err(Failure::argument(
                    "fallback_font_faces isn't an array of font files",
                ));
            }
            fresh_out(out_map, "out_map")?;
            let completion = Completion::required(completion)?;
            let handle = Arc::new(MapHandle {
                runtime: runtime.clone(),
                id: OnceLock::new(),
                snapshot: Mutex::new(None),
                has_session: AtomicBool::new(false),
            });
            let id = handles::insert(Object::Map(handle.clone()));
            let _ = handle.id.set(id);
            runtime.maps.fetch_add(1, Ordering::AcqRel);
            let wake_tx = Mutex::new(runtime.sender());
            runtime.post(move |state| {
                let mut map = orb_api::Map::new(&state.runtime, orb_api::MapOptions::default());
                let extent = options.initial_extent;
                map.set_viewport(f64::from(extent.width), f64::from(extent.height));
                let woken = Arc::new(AtomicBool::new(false));
                let flag = woken.clone();
                map.set_wake(move || {
                    flag.store(true, Ordering::Release);
                    let tx = wake_tx
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let _ = tx.send(Msg::Pump);
                });
                state.maps.insert(
                    id,
                    MapEntry {
                        map,
                        handle,
                        mode: options.map_mode,
                        extent,
                        event_mask,
                        debug_flags: 0,
                        stats_view: false,
                        viewport: mln_map_viewport {
                            north_orientation: MLN_NORTH_ORIENTATION_UP,
                            constrain_mode: MLN_CONSTRAIN_MODE_HEIGHT_ONLY,
                            viewport_mode: MLN_VIEWPORT_MODE_DEFAULT,
                            frustum_offset: mln_edge_insets {
                                top: 0.0,
                                left: 0.0,
                                bottom: 0.0,
                                right: 0.0,
                            },
                        },
                        bounds: default_bounds(),
                        prefetch_zoom_delta: 4,
                        style_url: String::new(),
                        loaded_style: None,
                        session: None,
                        generation: 1,
                        render_update_generation: 0,
                        woken,
                        loading: false,
                        style_loaded: false,
                        style_waiters: Vec::new(),
                        transitions: Vec::new(),
                        ending: MLN_CAMERA_TRANSITION_OUTCOME_REPLACED,
                        groups: 0,
                        idle: None,
                        style: Default::default(),
                        queries: Default::default(),
                        query_aliases: Default::default(),
                    },
                );
                // Native's map moves its camera into place as it's made.
                for kind in [
                    MLN_RUNTIME_EVENT_MAP_CAMERA_WILL_CHANGE,
                    MLN_RUNTIME_EVENT_MAP_CAMERA_DID_CHANGE,
                ] {
                    state.event(
                        id,
                        kind,
                        Payload::CameraChange(
                            MLN_CAMERA_CHANGE_MODE_IMMEDIATE,
                            MLN_CAMERA_CHANGE_SOURCE_COMMAND,
                        ),
                        String::new(),
                    );
                }
                let generation = state.maps[&id].generation;
                state.complete(Some(completion), Ok(Value::None), generation);
            })?;
            put(out_map, mln_map(id), "out_map")
        })
    }
}

fn default_bounds() -> mln_camera_bounds {
    mln_camera_bounds {
        bounds: mln_lat_lng_bounds {
            southwest: mln_lat_lng {
                latitude: -90.0,
                longitude: -180.0,
            },
            northeast: mln_lat_lng {
                latitude: 90.0,
                longitude: 180.0,
            },
        },
        unbounded: true,
        min_zoom: 0.0,
        max_zoom: 25.5,
        min_pitch: 0.0,
        max_pitch: 60.0,
    }
}

fn check_mask(mask: u64) -> Result {
    if mask & !(MLN_RUNTIME_EVENT_FLAG_ALL) != 0 {
        return Err(Failure::argument(format!(
            "the event mask {mask:#x} has bits that name no event"
        )));
    }
    Ok(())
}

fn release(map: mln_map, completion: Option<Completion>, force: bool) -> Result {
    let handle = handles::map(map.0)?;
    if !force && handle.has_session.load(Ordering::Acquire) {
        return Err(Failure::state(
            "the map still has a render session: release it first",
        ));
    }
    handles::remove(map.0);
    // The runtime's release can follow at once: the map's teardown runs
    // before it on the runtime thread.
    handle.runtime.maps.fetch_sub(1, Ordering::AcqRel);
    let id = handle.id();
    let runtime = handle.runtime.clone();
    runtime.post(move |state| {
        if let Some(session) = state.maps.get(&id).and_then(|m| m.session) {
            crate::session::drop_session(state, session);
        }
        if let Some(mut entry) = state.maps.remove(&id) {
            for t in entry.transitions.drain(..) {
                t.handler
                    .end(MLN_CAMERA_TRANSITION_OUTCOME_CLOSED, entry.generation);
            }
            let queries = entry.queries.drain().map(|(_, q)| q.completion);
            for c in entry.style_waiters.drain(..).chain(queries) {
                c.complete(
                    Err(Failure::new(MLN_STATUS_CANCELLED, "the map was released")),
                    0,
                );
            }
        }
        drop(handle);
        state.complete(completion, Ok(Value::None), 0);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_release(
    map: mln_map,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            handles::map(map.0)?;
            release(map, Some(Completion::required(completion)?), false)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_dispose(
    map: mln_map,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe { boundary(out_diagnostic, || release(map, None, true)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_get_snapshot(
    map: mln_map,
    out_snapshot: *mut mln_map_snapshot,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::map(map.0)?;
            if out_snapshot.is_null() {
                return Err(Failure::argument("out_snapshot is null"));
            }
            if ((*out_snapshot).size as usize) < std::mem::size_of::<mln_map_snapshot>() {
                return Err(Failure::argument("out_snapshot.size is too small"));
            }
            let snapshot = handle
                .snapshot()
                .ok_or_else(|| Failure::state("the map isn't made yet"))?;
            out_snapshot.write(snapshot);
            Ok(())
        })
    }
}

macro_rules! command {
    ($(#[$doc:meta])* $name:ident($($arg:ident: $ty:ty),* $(,)?), $required:expr, $changes:expr,
     |$state:ident, $id:ident| $body:expr) => {
        $(#[$doc])*
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            map: mln_map,
            $($arg: $ty,)*
            completion: *const mln_completion,
            out_diagnostic: *mut mln_diagnostic,
        ) -> mln_status {
            // SAFETY: the caller's pointers.
            unsafe {
                boundary(out_diagnostic, || {
                    $body
                    // The body returns a job for `post`.
                    .and_then(|job| post(map, completion, $required, $changes, job))
                })
            }
        }
    };
}

pub(crate) use command;

/// A job that only changes the map entry.
pub(crate) fn job(
    f: impl FnOnce(&mut MapEntry) -> Result + Send + 'static,
) -> impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static {
    move |state, id| f(entry(state, id)).map(|()| Value::None)
}

command!(
    mln_map_set_debug_flags(flags: mln_map_debug_flags),
    false,
    true,
    |state, id| {
        Ok(job(move |entry| {
            entry.debug_flags = flags;
            Ok(())
        }))
    }
);

command!(
    mln_map_set_rendering_stats_view_enabled(enabled: bool),
    false,
    true,
    |state, id| {
        Ok(job(move |entry| {
            entry.stats_view = enabled;
            Ok(())
        }))
    }
);

command!(mln_map_dump_debug_logs(), false, false, |state, id| {
    Ok(job(move |entry| {
        crate::logging::log(
            MLN_LOG_SEVERITY_INFO,
            &format!(
                "map: generation {}, camera {:?}, {} transitions",
                entry.generation,
                entry.map.camera(),
                entry.transitions.len()
            ),
        );
        Ok(())
    }))
});

command!(
    mln_map_set_event_mask(mask: mln_runtime_event_flags),
    false,
    true,
    |state, id| {
        check_mask(mask)?;
        Ok(job(move |entry| {
            entry.event_mask = mask;
            Ok(())
        }))
    }
);

command!(
    mln_map_resize(size: mln_logical_size),
    false,
    true,
    |state, id| {
        if size.width == 0 || size.height == 0 {
            return Err(Failure::argument("the size is empty"));
        }
        Ok(job(move |entry| {
            entry.extent.width = size.width;
            entry.extent.height = size.height;
            entry
                .map
                .set_viewport(f64::from(size.width), f64::from(size.height));
            Ok(())
        }))
    }
);

command!(mln_map_request_repaint(), false, false, |state, id| {
    Ok(move |state: &mut State, id: u64| {
        if let Some(session) = state.maps[&id].session {
            crate::session::force_redraw(state, session);
        }
        Ok(Value::None)
    })
});

command!(mln_map_begin_command_group(), false, false, |state, id| {
    Ok(job(|entry| {
        entry.groups += 1;
        Ok(())
    }))
});

command!(mln_map_end_command_group(), false, true, |state, id| {
    Ok(job(|entry| {
        entry.groups = (entry.groups.checked_sub(1))
            .ok_or_else(|| Failure::state("the map has no open command group"))?;
        Ok(())
    }))
});

command!(mln_map_get_style_url(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| Ok(Value::Text(state.maps[&id].style_url.clone())))
});

command!(mln_map_get_loaded_style_json(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| {
        let entry = &state.maps[&id];
        // Before any style loads, the document is empty.
        Ok(Value::Text(entry.loaded_style.clone().unwrap_or_default()))
    })
});

command!(
    mln_map_set_tile_options(options: *const mln_map_tile_options),
    false,
    true,
    |state, id| {
        let o = *sized(options, "options")?;
        if o.fields & MLN_MAP_TILE_OPTION_LOD_MODE != 0 && o.lod_mode > MLN_TILE_LOD_MODE_DISTANCE {
            return Err(Failure::argument("lod_mode names no mode"));
        }
        Ok(job(move |entry| {
            let mut lod = entry.map.tile_lod();
            let has = |bit| o.fields & bit != 0;
            if has(MLN_MAP_TILE_OPTION_PREFETCH_ZOOM_DELTA) {
                entry.prefetch_zoom_delta = o.prefetch_zoom_delta;
            }
            if has(MLN_MAP_TILE_OPTION_LOD_MIN_RADIUS) {
                lod.min_radius = o.lod_min_radius;
            }
            if has(MLN_MAP_TILE_OPTION_LOD_SCALE) {
                lod.scale = o.lod_scale;
            }
            if has(MLN_MAP_TILE_OPTION_LOD_PITCH_THRESHOLD) {
                lod.pitch_threshold = o.lod_pitch_threshold;
            }
            if has(MLN_MAP_TILE_OPTION_LOD_ZOOM_SHIFT) {
                lod.zoom_shift = o.lod_zoom_shift;
            }
            if has(MLN_MAP_TILE_OPTION_LOD_MODE) {
                lod.mode = if o.lod_mode == MLN_TILE_LOD_MODE_DISTANCE {
                    orb_api::TileLodMode::Distance
                } else {
                    orb_api::TileLodMode::Default
                };
            }
            entry.map.set_tile_lod(lod);
            Ok(())
        }))
    }
);

command!(
    mln_map_set_viewport_options(options: *const mln_map_viewport_options),
    false,
    true,
    |state, id| {
        let o = *sized(options, "options")?;
        if o.fields & MLN_MAP_VIEWPORT_OPTION_CONSTRAIN_MODE != 0
            && o.constrain_mode > MLN_CONSTRAIN_MODE_SCREEN
        {
            return Err(Failure::argument("constrain_mode names no mode"));
        }
        Ok(job(move |entry| {
            let has = |bit| o.fields & bit != 0;
            if has(MLN_MAP_VIEWPORT_OPTION_NORTH_ORIENTATION) {
                entry.viewport.north_orientation = o.north_orientation;
            }
            if has(MLN_MAP_VIEWPORT_OPTION_CONSTRAIN_MODE) {
                entry.viewport.constrain_mode = o.constrain_mode;
            }
            if has(MLN_MAP_VIEWPORT_OPTION_VIEWPORT_MODE) {
                entry.viewport.viewport_mode = o.viewport_mode;
            }
            if has(MLN_MAP_VIEWPORT_OPTION_FRUSTUM_OFFSET) {
                entry.viewport.frustum_offset = o.frustum_offset;
            }
            Ok(())
        }))
    }
);

command!(
    mln_map_set_bounds(options: *const mln_bound_options),
    false,
    true,
    |state, id| {
        let o = *sized(options, "options")?;
        Ok(job(move |entry| {
            let has = |bit| o.fields & bit != 0;
            if has(MLN_BOUND_OPTION_BOUNDS) {
                entry.bounds.bounds = o.bounds;
                entry.bounds.unbounded = false;
            }
            if has(MLN_BOUND_OPTION_UNBOUNDED) {
                entry.bounds.unbounded = true;
                entry.bounds.bounds = default_bounds().bounds;
            }
            if has(MLN_BOUND_OPTION_MIN_ZOOM) {
                entry.bounds.min_zoom = o.min_zoom;
            }
            if has(MLN_BOUND_OPTION_MAX_ZOOM) {
                entry.bounds.max_zoom = o.max_zoom;
            }
            if has(MLN_BOUND_OPTION_MIN_PITCH) {
                entry.bounds.min_pitch = o.min_pitch;
            }
            if has(MLN_BOUND_OPTION_MAX_PITCH) {
                entry.bounds.max_pitch = o.max_pitch;
            }
            let b = entry.bounds;
            if b.min_zoom > b.max_zoom || b.min_pitch > b.max_pitch {
                return Err(Failure::argument("the minimum is above the maximum"));
            }
            entry.map.set_constraints(orb_api::Constraints {
                min_zoom: b.min_zoom,
                max_zoom: b.max_zoom,
                min_pitch: b.min_pitch,
                max_pitch: b.max_pitch,
                bounds: (!b.unbounded).then(|| bounds(b.bounds)),
            });
            Ok(())
        }))
    }
);

/// Starts loading the style at `url`; the completion waits for it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_set_style_url(
    map: mln_map,
    url: mln_buffer_view,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::map(map.0)?;
            if url.data.is_null() {
                return Err(Failure::argument("url is null"));
            }
            let url = text(url, "url")?;
            if url.is_empty() {
                return Err(Failure::argument("the URL is empty"));
            }
            let completion = Completion::from_c(completion)?;
            let id = handle.id();
            handle.runtime.post(move |state| {
                let Some(entry) = state.maps.get_mut(&id) else {
                    state.complete(completion, Err(Failure::state("the map isn't made")), 0);
                    return;
                };
                entry.map.load_style_url(&url);
                entry.style.reset();
                entry.style_url = url;
                entry.style_loaded = false;
                entry.loaded_style = None;
                entry.loading = true;
                entry.generation += 1;
                // A load still under way is superseded.
                for c in entry.style_waiters.drain(..) {
                    c.complete(
                        Err(Failure::new(
                            MLN_STATUS_CANCELLED,
                            "a newer style replaced it",
                        )),
                        0,
                    );
                }
                entry.style_waiters.extend(completion);
                entry.woken.store(true, Ordering::Release);
                state.event(
                    id,
                    MLN_RUNTIME_EVENT_MAP_LOADING_STARTED,
                    Payload::None,
                    String::new(),
                );
            })
        })
    }
}

command!(
    mln_map_set_style_json(json: mln_buffer_view),
    false,
    true,
    |state, id| {
        if json.data.is_null() {
            return Err(Failure::argument("json is null"));
        }
        let json = text(json, "json")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            entry.style_url.clear();
            entry.style_loaded = false;
            entry.loading = true;
            for c in entry.style_waiters.drain(..) {
                c.complete(
                    Err(Failure::new(
                        MLN_STATUS_CANCELLED,
                        "a newer style replaced it",
                    )),
                    0,
                );
            }
            let result = entry.map.set_style_json(&json);
            if result.is_ok() {
                entry.style.reset();
                entry.loaded_style = Some(json);
                // The document is in: take its StyleLoad now, as Native
                // notifies inside the command.
                let _ = entry.map.apply_arrivals();
            }
            state.event(
                id,
                MLN_RUNTIME_EVENT_MAP_LOADING_STARTED,
                Payload::None,
                String::new(),
            );
            if let Err(e) = &result {
                state.event(
                    id,
                    MLN_RUNTIME_EVENT_MAP_LOADING_FAILED,
                    Payload::LoadingFailed(MLN_MAP_LOAD_ERROR_STYLE_PARSE),
                    e.message().into(),
                );
            }
            // A document that doesn't parse leaves the style as it was.
            result
                .map(|()| Value::None)
                .map_err(|e| Failure::native(e.message()))
        })
    }
);

/// An Orbis camera move from the options' fields.
fn target(entry: &MapEntry, camera: &mln_camera_options) -> Result<CameraTarget> {
    let f = camera.fields;
    let has = |bit| f & bit != 0;
    let mut t = CameraTarget::default();
    if has(MLN_CAMERA_OPTION_CENTER) {
        let c = camera.center;
        if !(c.latitude.is_finite() && c.longitude.is_finite()) || c.latitude.abs() > 90.0 {
            return Err(Failure::argument("the center isn't a coordinate"));
        }
        t.center = Some(Geodetic::new(c.longitude, c.latitude));
    }
    let finite = |v: f64, what: &str| {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(Failure::argument(format!("{what} isn't finite")))
        }
    };
    if has(MLN_CAMERA_OPTION_ZOOM) {
        t.zoom = Some(finite(camera.zoom, "zoom")?);
    }
    if has(MLN_CAMERA_OPTION_BEARING) {
        t.bearing = Some(finite(camera.bearing, "bearing")?);
    }
    if has(MLN_CAMERA_OPTION_PITCH) {
        t.pitch = Some(finite(camera.pitch, "pitch")?);
    }
    if has(MLN_CAMERA_OPTION_ROLL) {
        t.roll = Some(finite(camera.roll, "roll")?);
    }
    if has(MLN_CAMERA_OPTION_FOV) {
        t.fov = Some(finite(camera.field_of_view, "field_of_view")?);
    }
    if has(MLN_CAMERA_OPTION_CENTER_ALTITUDE) {
        t.elevation = Some(finite(camera.center_altitude, "center_altitude")?);
    }
    if has(MLN_CAMERA_OPTION_PADDING) {
        let p = camera.padding;
        t.padding = Some(orb_api::Padding {
            top: p.top,
            right: p.right,
            bottom: p.bottom,
            left: p.left,
        });
    }
    if has(MLN_CAMERA_OPTION_ANCHOR) && t.center.is_none() {
        let a = camera.anchor;
        t.around = entry.map.unproject(glam::DVec2::new(a.x, a.y));
    }
    Ok(t)
}

/// The animation fields Orbis reads, and the end handler.
#[derive(Clone, Copy)]
struct Animation {
    fields: mln_animation_option_fields,
    duration_ms: f64,
    velocity: f64,
    min_zoom: f64,
    easing: mln_unit_bezier,
    transition_id: u64,
    handler: Option<EndHandler>,
}

impl Animation {
    fn from_c(a: &mln_animation_options) -> Result<Self> {
        let has = |bit| a.fields & bit != 0;
        if has(MLN_ANIMATION_OPTION_DURATION)
            && !(a.duration_ms.is_finite() && a.duration_ms >= 0.0)
        {
            return Err(Failure::argument("duration_ms isn't a duration"));
        }
        if has(MLN_ANIMATION_OPTION_VELOCITY) && !(a.velocity.is_finite() && a.velocity > 0.0) {
            return Err(Failure::argument("velocity isn't positive"));
        }
        Ok(Self {
            fields: a.fields,
            duration_ms: a.duration_ms,
            velocity: a.velocity,
            min_zoom: a.min_zoom,
            easing: a.easing,
            transition_id: a.transition_id,
            handler: EndHandler::from_c(&a.end_handler)?,
        })
    }
}

fn easing(animation: &Animation) -> Option<Easing> {
    (animation.fields & MLN_ANIMATION_OPTION_EASING != 0).then(|| {
        let e = animation.easing;
        Easing::Bezier([e.x1, e.y1, e.x2, e.y2])
    })
}

/// Moves the camera as an update says, at once or animated, and keeps its
/// end handler for when it ends.
fn update_camera(
    entry: &mut MapEntry,
    mode: mln_camera_update_mode,
    t: &CameraTarget,
    animation: &Animation,
) -> Result {
    let duration =
        (animation.fields & MLN_ANIMATION_OPTION_DURATION != 0).then_some(animation.duration_ms);
    // The transitions running now end replaced.
    entry.ending = MLN_CAMERA_TRANSITION_OUTCOME_REPLACED;
    let animation_id = match mode {
        MLN_CAMERA_UPDATE_MODE_JUMP => {
            entry.map.jump_to(t);
            None
        }
        MLN_CAMERA_UPDATE_MODE_EASE => {
            let mut options = EaseOptions::default();
            if let Some(d) = duration {
                options.duration = d;
            }
            if let Some(e) = easing(animation) {
                options.easing = e;
            }
            Some(entry.map.ease_to(t, &options))
        }
        MLN_CAMERA_UPDATE_MODE_FLY => {
            let mut options = FlyOptions {
                duration,
                ..Default::default()
            };
            if animation.fields & MLN_ANIMATION_OPTION_MIN_ZOOM != 0 {
                options.min_zoom = Some(animation.min_zoom);
            }
            if animation.fields & MLN_ANIMATION_OPTION_VELOCITY != 0 {
                options.screen_speed = Some(animation.velocity);
            }
            if let Some(e) = easing(animation) {
                options.easing = e;
            }
            Some(entry.map.fly_to(t, &options))
        }
        _ => return Err(Failure::argument("unknown camera update mode")),
    };
    // A jump ends the transitions it interrupts now.
    if animation_id.is_none() {
        for t in entry.transitions.drain(..) {
            t.handler
                .end(MLN_CAMERA_TRANSITION_OUTCOME_REPLACED, entry.generation);
        }
    }
    if let Some(handler) = animation.handler {
        entry.transitions.push(Transition {
            animation: animation_id,
            id: (animation.fields & MLN_ANIMATION_OPTION_TRANSITION_ID != 0)
                .then_some(animation.transition_id),
            handler,
        });
    }
    Ok(())
}

command!(
    mln_map_update_camera(update: *const mln_camera_update),
    false,
    true,
    |state, id| {
        let update = *sized(update, "update")?;
        sized(&update.camera, "update.camera")?;
        if update.mode > MLN_CAMERA_UPDATE_MODE_FLY {
            return Err(Failure::argument("update.mode names no mode"));
        }
        let (mode, camera) = (update.mode, update.camera);
        let animation = Animation::from_c(&update.animation)?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let t = target(entry, &camera)?;
            update_camera(entry, mode, &t, &animation)?;
            Ok(Value::None)
        })
    }
);

/// An `mln_camera_delta`'s fields.
#[derive(Clone, Copy)]
struct Delta {
    fields: mln_camera_delta_fields,
    offset: mln_screen_point,
    scale: f64,
    bearing: f64,
    pitch: f64,
    anchor: mln_screen_point,
    animation: Animation,
    gesture_phase: mln_gesture_phase,
}

/// A host's gesture step on Orbis's camera: the pan moves the point under
/// the offset's start to its end, then zoom and rotation turn about the
/// anchor.
fn apply_delta(entry: &mut MapEntry, d: &Delta) -> Result {
    let has = |bit| d.fields & bit != 0;
    if d.gesture_phase == MLN_GESTURE_PHASE_BEGIN {
        entry.map.stop();
    }
    let viewport = entry.map.viewport();
    let middle = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
    if has(MLN_CAMERA_DELTA_OFFSET) && (d.offset.x != 0.0 || d.offset.y != 0.0) {
        let offset = glam::DVec2::new(d.offset.x, d.offset.y);
        if let Some(center) = entry.map.unproject(middle - offset) {
            entry.map.jump_to(&CameraTarget {
                center: Some(center),
                ..Default::default()
            });
        }
    }
    let camera = entry.map.camera();
    let mut t = CameraTarget::default();
    if has(MLN_CAMERA_DELTA_SCALE) && d.scale > 0.0 && d.scale.is_finite() {
        t.zoom = Some(entry.map.zoom() + d.scale.log2());
    }
    if has(MLN_CAMERA_DELTA_BEARING) {
        t.bearing = Some(camera.bearing + d.bearing);
    }
    if has(MLN_CAMERA_DELTA_PITCH) {
        t.pitch = Some(camera.pitch + d.pitch);
    }
    if t.zoom.is_none() && t.bearing.is_none() && t.pitch.is_none() {
        return Ok(());
    }
    if has(MLN_CAMERA_DELTA_ANCHOR) {
        t.around = entry
            .map
            .unproject(glam::DVec2::new(d.anchor.x, d.anchor.y));
    }
    let mode = if d.animation.fields & MLN_ANIMATION_OPTION_DURATION != 0 {
        MLN_CAMERA_UPDATE_MODE_EASE
    } else {
        MLN_CAMERA_UPDATE_MODE_JUMP
    };
    update_camera(entry, mode, &t, &d.animation)
}

command!(
    mln_map_apply_camera_delta(delta: *const mln_camera_delta),
    false,
    true,
    |state, id| {
        let d = *sized(delta, "delta")?;
        if d.gesture_phase > MLN_GESTURE_PHASE_CANCEL {
            return Err(Failure::argument("gesture_phase names no phase"));
        }
        let delta = Delta {
            fields: d.fields,
            offset: d.offset,
            scale: d.scale,
            bearing: d.bearing,
            pitch: d.pitch,
            anchor: d.anchor,
            animation: Animation::from_c(&d.animation)?,
            gesture_phase: d.gesture_phase,
        };
        Ok(job(move |entry| apply_delta(entry, &delta)))
    }
);

command!(mln_map_get_camera(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| {
        let entry = &state.maps[&id];
        Ok(Value::of(mln_camera_query_result {
            generation: entry.generation,
            camera: entry.camera(),
        }))
    })
});

/// Ends the transitions `which` names as cancelled, and stops the camera
/// if they were animating it.
fn cancel(entry: &mut MapEntry, which: Option<u64>) {
    let generation = entry.generation;
    let mut stop = false;
    let mut i = 0;
    while i < entry.transitions.len() {
        if which.is_none() || entry.transitions[i].id == which {
            let t = entry.transitions.remove(i);
            stop |= t.animation.is_some();
            t.handler
                .end(MLN_CAMERA_TRANSITION_OUTCOME_CANCELLED, generation);
        } else {
            i += 1;
        }
    }
    if which.is_none() || stop {
        entry.ending = MLN_CAMERA_TRANSITION_OUTCOME_CANCELLED;
        entry.map.stop();
    }
}

command!(
    mln_map_cancel_camera_transitions(),
    false,
    true,
    |state, id| {
        Ok(job(|entry| {
            cancel(entry, None);
            Ok(())
        }))
    }
);

command!(
    mln_map_cancel_camera_transitions_with_id(transition_id: u64),
    false,
    true,
    |state, id| {
        Ok(job(move |entry| {
            cancel(entry, Some(transition_id));
            Ok(())
        }))
    }
);

fn bounds(b: mln_lat_lng_bounds) -> orb_api::Bounds {
    orb_api::Bounds {
        west: b.southwest.longitude,
        south: b.southwest.latitude,
        east: b.northeast.longitude,
        north: b.northeast.latitude,
    }
}

fn check_bounds(b: &mln_lat_lng_bounds) -> Result {
    let ok = |l: &mln_lat_lng| {
        l.latitude.is_finite() && l.longitude.is_finite() && l.latitude.abs() <= 90.0
    };
    if !ok(&b.southwest) || !ok(&b.northeast) || b.southwest.latitude > b.northeast.latitude {
        return Err(Failure::argument("the bounds aren't a box of coordinates"));
    }
    Ok(())
}

fn fit_options(o: &mln_camera_fit_options) -> FitOptions {
    let mut fit = FitOptions::default();
    if o.fields & MLN_CAMERA_FIT_OPTION_PADDING != 0 {
        let p = o.padding;
        fit.padding = orb_api::Padding {
            top: p.top,
            right: p.right,
            bottom: p.bottom,
            left: p.left,
        };
    }
    if o.fields & MLN_CAMERA_FIT_OPTION_BEARING != 0 {
        fit.bearing = Some(o.bearing);
    }
    if o.fields & MLN_CAMERA_FIT_OPTION_PITCH != 0 {
        fit.pitch = Some(o.pitch);
    }
    fit
}

/// # Safety
/// `options` is null or a readable `mln_camera_fit_options`.
unsafe fn fit_from(options: *const mln_camera_fit_options) -> Result<FitOptions> {
    if options.is_null() {
        return Ok(FitOptions::default());
    }
    // SAFETY: the caller's contract.
    Ok(fit_options(unsafe { sized(options, "fit_options")? }))
}

/// The box around coordinates.
///
/// # Safety
/// `coordinates` holds `count` readable coordinates.
unsafe fn box_of(coordinates: *const mln_lat_lng, count: usize) -> Result<mln_lat_lng_bounds> {
    if coordinates.is_null() || count == 0 {
        return Err(Failure::argument("no coordinates"));
    }
    // SAFETY: the caller's contract.
    let points = unsafe { std::slice::from_raw_parts(coordinates, count) };
    let mut b = mln_lat_lng_bounds {
        southwest: points[0],
        northeast: points[0],
    };
    for p in points {
        b.southwest.latitude = b.southwest.latitude.min(p.latitude);
        b.southwest.longitude = b.southwest.longitude.min(p.longitude);
        b.northeast.latitude = b.northeast.latitude.max(p.latitude);
        b.northeast.longitude = b.northeast.longitude.max(p.longitude);
    }
    check_bounds(&b)?;
    Ok(b)
}

/// The camera that fits `b`, as options with every member set.
fn camera_for(
    entry: &MapEntry,
    b: mln_lat_lng_bounds,
    fit: &FitOptions,
) -> Result<mln_camera_options> {
    let t = entry
        .map
        .camera_for_bounds(bounds(b), fit)
        .ok_or_else(|| Failure::argument("no camera shows these bounds"))?;
    let camera = entry.map.camera();
    let center = t.center.unwrap_or(camera.target);
    let mut options = mln_camera_options {
        size: std::mem::size_of::<mln_camera_options>() as u32,
        fields: MLN_CAMERA_OPTION_CENTER
            | MLN_CAMERA_OPTION_ZOOM
            | MLN_CAMERA_OPTION_BEARING
            | MLN_CAMERA_OPTION_PITCH,
        ..Default::default()
    };
    options.center = mln_lat_lng {
        latitude: center.lat,
        longitude: center.lon,
    };
    options.zoom = t.zoom.unwrap_or_else(|| entry.map.zoom());
    options.bearing = t.bearing.unwrap_or(camera.bearing);
    options.pitch = t.pitch.unwrap_or(camera.pitch);
    Ok(options)
}

command!(
    mln_map_camera_for_lat_lng_bounds(b: mln_lat_lng_bounds, fit: *const mln_camera_fit_options),
    true,
    false,
    |state, id| {
        check_bounds(&b)?;
        let fit = fit_from(fit)?;
        Ok(move |state: &mut State, id: u64| camera_for(&state.maps[&id], b, &fit).map(Value::of))
    }
);

command!(
    mln_map_camera_for_lat_lngs(
        coordinates: *const mln_lat_lng,
        count: usize,
        fit: *const mln_camera_fit_options,
    ),
    true,
    false,
    |state, id| {
        let b = box_of(coordinates, count)?;
        let fit = fit_from(fit)?;
        Ok(move |state: &mut State, id: u64| camera_for(&state.maps[&id], b, &fit).map(Value::of))
    }
);

/// # Safety
/// `update` is a readable `mln_camera_fit_update`.
unsafe fn fit_update(
    update: *const mln_camera_fit_update,
) -> Result<(mln_camera_update_mode, FitOptions, Animation)> {
    // SAFETY: the caller's contract.
    let u = *unsafe { sized(update, "update")? };
    if u.mode > MLN_CAMERA_UPDATE_MODE_FLY {
        return Err(Failure::argument("update.mode names no mode"));
    }
    Ok((
        u.mode,
        fit_options(&u.fit),
        Animation::from_c(&u.animation)?,
    ))
}

fn fit_job(
    b: mln_lat_lng_bounds,
    (mode, fit, animation): (mln_camera_update_mode, FitOptions, Animation),
) -> impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static {
    job(move |entry| {
        let camera = camera_for(entry, b, &fit)?;
        let t = target(entry, &camera)?;
        update_camera(entry, mode, &t, &animation)
    })
}

command!(
    mln_map_fit_camera_to_lat_lng_bounds(
        b: mln_lat_lng_bounds,
        update: *const mln_camera_fit_update,
    ),
    false,
    true,
    |state, id| {
        check_bounds(&b)?;
        Ok(fit_job(b, fit_update(update)?))
    }
);

command!(
    mln_map_fit_camera_to_lat_lngs(
        coordinates: *const mln_lat_lng,
        count: usize,
        update: *const mln_camera_fit_update,
    ),
    false,
    true,
    |state, id| {
        let b = box_of(coordinates, count)?;
        Ok(fit_job(b, fit_update(update)?))
    }
);

command!(
    mln_map_lat_lng_bounds_for_camera(camera: *const mln_camera_options),
    true,
    false,
    |state, id| {
        let camera = *sized(camera, "camera")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = &state.maps[&id];
            let mut t = target(entry, &camera)?;
            let c = entry.map.camera();
            t.center.get_or_insert(c.target);
            t.zoom.get_or_insert(entry.map.zoom());
            t.bearing.get_or_insert(c.bearing);
            t.pitch.get_or_insert(c.pitch);
            let v = entry.map.viewport();
            let corners = [
                glam::DVec2::new(0.0, 0.0),
                glam::DVec2::new(v.width, 0.0),
                glam::DVec2::new(0.0, v.height),
                glam::DVec2::new(v.width, v.height),
            ];
            let mut b: Option<mln_lat_lng_bounds> = None;
            for p in corners {
                let Some(g) = entry.map.unproject_with(&t, p) else {
                    continue;
                };
                let l = mln_lat_lng {
                    latitude: g.lat,
                    longitude: g.lon,
                };
                let b = b.get_or_insert(mln_lat_lng_bounds {
                    southwest: l,
                    northeast: l,
                });
                b.southwest.latitude = b.southwest.latitude.min(l.latitude);
                b.southwest.longitude = b.southwest.longitude.min(l.longitude);
                b.northeast.latitude = b.northeast.latitude.max(l.latitude);
                b.northeast.longitude = b.northeast.longitude.max(l.longitude);
            }
            b.map(Value::of)
                .ok_or_else(|| Failure::argument("the camera shows no ground"))
        })
    }
);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_camera_for_tile(
    tile: mln_canonical_tile_id,
    logical_size: mln_logical_size,
    out_camera: *mut mln_camera_options,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if tile.z > 30 || u64::from(tile.x) >= 1 << tile.z || u64::from(tile.y) >= 1 << tile.z {
                return Err(Failure::argument("no such tile"));
            }
            if logical_size.width == 0 || logical_size.width != logical_size.height {
                return Err(Failure::argument("a tile's map is square"));
            }
            let n = f64::from(1u32 << tile.z);
            let lon = (f64::from(tile.x) + 0.5) / n * 360.0 - 180.0;
            let y = std::f64::consts::PI * (1.0 - 2.0 * (f64::from(tile.y) + 0.5) / n);
            let lat = y.sinh().atan().to_degrees();
            if out_camera.is_null() {
                return Err(Failure::argument("out_camera is null"));
            }
            let size = (*out_camera).size;
            let mut camera = mln_camera_options {
                size,
                fields: MLN_CAMERA_OPTION_CENTER
                    | MLN_CAMERA_OPTION_ZOOM
                    | MLN_CAMERA_OPTION_BEARING
                    | MLN_CAMERA_OPTION_PITCH,
                ..Default::default()
            };
            camera.center = mln_lat_lng {
                latitude: lat,
                longitude: lon,
            };
            camera.zoom = f64::from(tile.z) + (f64::from(logical_size.width) / 512.0).log2();
            out_camera.write(camera);
            Ok(())
        })
    }
}
