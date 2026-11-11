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
    /// Fixed when the map is made.
    pub(crate) mode: mln_map_mode,
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

/// A camera command's animations, and its end handler, which runs once
/// the last of them ends.
struct Transition {
    animations: Vec<AnimationId>,
    id: Option<u64>,
    handler: Option<EndHandler>,
    /// How it ends, once a command cancelled it: its animation's end
    /// reports the speed it had left.
    outcome: Option<mln_camera_transition_outcome>,
    /// A later command took the last property one of its animations
    /// moved.
    replaced: bool,
}

impl Transition {
    fn new(animations: Vec<AnimationId>, animation: &Animation) -> Self {
        Self {
            animations,
            id: (animation.fields & MLN_ANIMATION_OPTION_TRANSITION_ID != 0)
                .then_some(animation.transition_id),
            handler: animation.handler,
            outcome: None,
            replaced: false,
        }
    }

    /// How it ends once nothing of it runs.
    fn outcome(&self) -> mln_camera_transition_outcome {
        self.outcome.unwrap_or(match self.replaced {
            true => MLN_CAMERA_TRANSITION_OUTCOME_REPLACED,
            false => MLN_CAMERA_TRANSITION_OUTCOME_COMPLETED,
        })
    }
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
        self.end_moving(outcome, generation, orb_api::MomentumVelocity::default());
    }

    /// Ends the transition, reporting the speed its motion had left.
    fn end_moving(
        self,
        outcome: mln_camera_transition_outcome,
        generation: u64,
        velocity: orb_api::MomentumVelocity,
    ) {
        delivery::deliver(
            self.queue,
            Box::new(move || {
                // SAFETY: plain C data.
                let mut end: mln_camera_transition_end = unsafe { std::mem::zeroed() };
                end.size = std::mem::size_of::<mln_camera_transition_end>() as u32;
                end.outcome = outcome;
                end.generation = generation;
                end.velocity = mln_camera_velocity {
                    pan: mln_screen_point {
                        x: velocity.pan.x,
                        y: velocity.pan.y,
                    },
                    zoom: velocity.zoom,
                    bearing: velocity.bearing,
                    pitch: velocity.pitch,
                };
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
    projection_mode: mln_projection_mode,
    bounds: mln_camera_bounds,
    prefetch_zoom_delta: u32,
    pub(crate) style_url: String,
    /// The style text as loaded, without later edits.
    loaded_style: Option<String>,
    pub(crate) session: Option<u64>,
    /// Grows with each change to the map's state.
    pub(crate) generation: u64,
    /// The render updates the map has published for its session: each
    /// command that changes what it draws publishes one, and with a session
    /// so does each change the map makes on its own (data arriving, an
    /// animation's next step).
    pub(crate) render_update_generation: u64,
    /// A change inside a command group, which its outermost end publishes.
    pub(crate) held_update: bool,
    /// Set by a command's job that changed nothing the map draws, so it
    /// publishes no render update.
    pub(crate) unchanged: bool,
    /// Data arrived since the map last took it in.
    woken: Arc<AtomicBool>,
    /// A style or source change awaits its loads, for LOADING_FINISHED.
    loading: bool,
    /// The STYLE_LOADED of the current style has been raised.
    style_loaded: bool,
    /// Completions of URL style loads, waiting for the style.
    style_waiters: Vec<Completion>,
    /// Camera commands whose animations run.
    transitions: Vec<Transition>,
    /// Command groups open: while any is, a session draws none of the
    /// map's changes, so they show together once the outermost ends.
    pub(crate) groups: u32,
    /// A gesture runs: from its begin to its end or cancel, or to the end
    /// of the glide its release starts.
    gesture: bool,
    /// The glide a gesture's release started.
    momentum: Option<AnimationId>,
    /// Whether the gesture under way changed the zoom, the bearing and
    /// the pitch: only those settle on a release's snaps.
    gesture_changed: [bool; 3],
    /// How the camera's latest change was made, for its camera events.
    change: (mln_camera_change_mode, mln_camera_change_source),
    /// An immediate move raised its camera events with its command, so
    /// the frame that shows it reports none: Orbis's report of it is
    /// dropped up to its end. Changes the map makes on its own (a style's
    /// camera, constraints) report as they show.
    jumped: bool,
    /// Transitions a command ended, whose handlers run after its change
    /// is published, with its generation.
    ended: Vec<(EndHandler, mln_camera_transition_outcome)>,
    /// Camera events of immediate moves, for the next settle.
    camera_events: Vec<(mln_runtime_event_type, Payload, String)>,
    /// Draws the map's frames while no session does, so animations run.
    idle: Option<orb_api::Target>,
    /// What the C API keeps of the style beyond Orbis's document.
    pub(crate) style: crate::style::StyleState,
    /// Feature queries waiting for Orbis's answer, by its query id, and
    /// the ids of a query's further parts.
    pub(crate) queries: std::collections::HashMap<u64, crate::queries::Query>,
    pub(crate) query_aliases: std::collections::HashMap<u64, u64>,
    /// The completion of the still image the map's session draws next, a
    /// frame at a time, as Native renders stills asynchronously: the
    /// runtime's other work goes on meanwhile.
    pub(crate) still: Option<Completion>,
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
            projection_mode: self.projection_mode,
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
            gesture_in_progress: self.gesture,
            // Native's `inTransition`: an animation runs. A jump that took the
            // last one's fields ends it at once, though the move's end
            // waits for the runtime thread's next frame to report it.
            transition_in_progress: self.map.is_animating() || self.map.transitioning(),
            event_mask: self.event_mask,
            latest_render_update_generation: self.render_update_generation,
            tile: mln_map_tile_settings {
                prefetch_zoom_delta: self.prefetch_zoom_delta,
                lod_min_radius: lod.min_radius,
                lod_scale: lod.scale,
                lod_pitch_threshold: lod.pitch_threshold.to_degrees(),
                lod_zoom_shift: lod.zoom_shift,
                lod_mode: match lod.mode {
                    orb_api::TileLodMode::Default => MLN_TILE_LOD_MODE_DEFAULT,
                    orb_api::TileLodMode::Distance => MLN_TILE_LOD_MODE_DISTANCE,
                },
            },
            bounds: self.bounds,
            free_camera: free_camera(&self.map),
            view: self.view(),
        };
        *self
            .handle
            .snapshot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(snapshot);
    }

    /// Publishes a render update for a change, or holds it for the end of
    /// the open command group.
    pub(crate) fn update(&mut self) {
        if self.groups > 0 {
            self.held_update = true;
        } else {
            self.render_update_generation += 1;
        }
    }

    /// The render update that first shows the changes made so far.
    pub(crate) fn next_update(&self) -> u64 {
        self.render_update_generation + u64::from(self.held_update)
    }

    /// Marks whether a session draws the map, on the caller's side.
    pub(crate) fn handle_flag(&self, attached: bool) {
        self.handle.has_session.store(attached, Ordering::Release);
    }

    /// Whether the map moves or loads with nothing drawing it, or its
    /// session's still image needs another frame.
    fn busy(&self) -> bool {
        match self.session {
            None => self.map.is_moving() || self.woken.load(Ordering::Acquire),
            Some(_) => self.still.is_some() && self.map.needs_frame(),
        }
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
///
/// Returns what ends the transitions and style loads, for the caller to
/// run once the events are sent: a handler sees its events queued.
pub(crate) fn take_events(state: &mut State, id: u64) -> Vec<Box<dyn FnOnce()>> {
    let mut out = Vec::new();
    let mut ended = Vec::new();
    let mut moving = Vec::new();
    let mut waiters = Vec::new();
    {
        let Some(entry) = state.maps.get_mut(&id) else {
            return Vec::new();
        };
        let mut camera_phases = [false; 3];
        // The report of an immediate move ends with its end phases.
        let mut jump_ended = false;
        // The camera events of immediate moves, raised by their commands,
        // and the transitions they ended.
        out.append(&mut entry.camera_events);
        ended.append(&mut entry.ended);
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
                E::TileRequest { source, tile } => {
                    if let Some(custom) = entry.style.custom(&source) {
                        custom.fetch(id, &source, tile);
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
                // An immediate move raised its events with its command; Orbis
                // reports it again with the frame that shows it.
                E::Camera(orb_api::CameraEvent::Changed { phase, .. })
                    if entry.jumped && !(jump_ended && phase != Phase::End) =>
                {
                    jump_ended |= phase == Phase::End;
                }
                E::Camera(orb_api::CameraEvent::Changed { phase, .. }) => {
                    if std::mem::take(&mut jump_ended) {
                        entry.jumped = false;
                    }
                    let (kind, i) = match phase {
                        Phase::Start => (MLN_RUNTIME_EVENT_MAP_CAMERA_WILL_CHANGE, 0),
                        Phase::Progress => (MLN_RUNTIME_EVENT_MAP_CAMERA_IS_CHANGING, 1),
                        Phase::End => (MLN_RUNTIME_EVENT_MAP_CAMERA_DID_CHANGE, 2),
                    };
                    if !std::mem::replace(&mut camera_phases[i], true) {
                        let payload = Payload::CameraChange(entry.change.0, entry.change.1);
                        if i == 1 {
                            out.push((kind, Payload::None, String::new()));
                        } else {
                            out.push((kind, payload, String::new()));
                        }
                    }
                }
                E::Camera(orb_api::CameraEvent::AnimationEnd {
                    id,
                    interrupted,
                    velocity,
                }) => {
                    // A gesture's glide ends the gesture.
                    if entry.momentum == Some(id) {
                        entry.momentum = None;
                        entry.gesture = false;
                    }
                    // An animation interrupted without a cancel lost what
                    // it moved to a later command.
                    if let Some(i) = entry
                        .transitions
                        .iter()
                        .position(|t| t.animations.contains(&id))
                    {
                        let t = &mut entry.transitions[i];
                        t.animations.retain(|a| *a != id);
                        t.replaced |= interrupted;
                        if t.animations.is_empty() {
                            let t = entry.transitions.remove(i);
                            if let Some(handler) = t.handler {
                                moving.push((handler, t.outcome(), velocity));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if jump_ended {
            entry.jumped = false;
        }
        // Transitions whose animations ended unreported end with the
        // camera at rest.
        if !entry.map.is_moving() {
            for t in entry.transitions.drain(..) {
                let outcome = t.outcome();
                ended.extend(t.handler.map(|h| (h, outcome)));
            }
        }
        if !ended.is_empty() || !out.is_empty() {
            entry.generation += 1;
        }
    }
    for (kind, payload, message) in out {
        state.event(id, kind, payload, message);
    }
    let generation = state.maps.get(&id).map_or(0, |m| m.generation);
    let mut after: Vec<Box<dyn FnOnce()>> = Vec::new();
    for (handler, outcome) in ended {
        after.push(Box::new(move || handler.end(outcome, generation)));
    }
    for (handler, outcome, velocity) in moving {
        after.push(Box::new(move || {
            handler.end_moving(outcome, generation, velocity);
        }));
    }
    for (completion, result) in waiters {
        after.push(Box::new(move || completion.complete(result, generation)));
    }
    after
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
        if let Some(entry) = state.maps.get_mut(&id) {
            let unchanged = std::mem::take(&mut entry.unchanged);
            if changes && result.is_ok() {
                entry.generation += 1;
                if !unchanged {
                    entry.update();
                }
            }
        }
        state.settle();
        let (generation, update) = state
            .maps
            .get(&id)
            .map_or((0, 0), |m| (m.generation, m.next_update()));
        state.complete_update(completion, result, generation, update);
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
            let options = *sized(options, "mln_map_options")?;
            if options.fields & !(MLN_MAP_OPTION_EVENT_MASK | MLN_MAP_OPTION_TILE_CACHE_ENABLED)
                != 0
            {
                return Err(Failure::argument("mln_map_options.fields has unknown bits"));
            }
            crate::base::physical(options.initial_extent).map_err(|f| {
                Failure::argument(format!("the initial extent is invalid: {}", f.message))
            })?;
            if options.map_mode > MLN_MAP_MODE_TILE {
                return Err(Failure::argument(format!(
                    "mln_map_options.map_mode {} names no map mode",
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
                mode: options.map_mode,
            });
            let id = handles::insert(Object::Map(handle.clone()));
            let _ = handle.id.set(id);
            runtime.maps.fetch_add(1, Ordering::AcqRel);
            let wake_tx = Mutex::new(runtime.sender());
            runtime.post(move |state| {
                let mut map = orb_api::Map::new(&state.runtime, orb_api::MapOptions::default());
                let extent = options.initial_extent;
                map.set_viewport(f64::from(extent.width), f64::from(extent.height));
                // Native's limits, not orb-camera's GL JS ones (zoom from
                // -2 to 22, pitch to 85°), which a map whose bounds were
                // never set would otherwise keep.
                map.set_constraints(constraints(&default_bounds()));
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
                        // Perspective, with Native's skews: heights rise
                        // straight up once axonometric.
                        projection_mode: {
                            let skew = orb_api::Skew::default();
                            mln_projection_mode {
                                axonometric: false,
                                x_skew: skew.x,
                                y_skew: skew.y,
                            }
                        },
                        bounds: default_bounds(),
                        prefetch_zoom_delta: 4,
                        style_url: String::new(),
                        loaded_style: None,
                        session: None,
                        generation: 1,
                        render_update_generation: 0,
                        held_update: false,
                        unchanged: false,
                        woken,
                        loading: false,
                        style_loaded: false,
                        style_waiters: Vec::new(),
                        transitions: Vec::new(),
                        groups: 0,
                        gesture: false,
                        momentum: None,
                        gesture_changed: [false; 3],
                        camera_events: Vec::new(),
                        ended: Vec::new(),
                        change: (
                            MLN_CAMERA_CHANGE_MODE_IMMEDIATE,
                            MLN_CAMERA_CHANGE_SOURCE_COMMAND,
                        ),
                        jumped: false,
                        idle: None,
                        style: Default::default(),
                        queries: Default::default(),
                        query_aliases: Default::default(),
                        still: None,
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
        return Err(Failure::state("map still has an attached render session"));
    }
    handles::remove(map.0);
    // The runtime's release can follow at once: the map's teardown runs
    // before it on the runtime thread.
    handle.runtime.child_released();
    let id = handle.id();
    let runtime = handle.runtime.clone();
    runtime.post(move |state| {
        if let Some(session) = state.maps.get(&id).and_then(|m| m.session) {
            crate::session::drop_session(state, session);
        }
        if let Some(mut entry) = state.maps.remove(&id) {
            for handler in entry.transitions.drain(..).filter_map(|t| t.handler) {
                // A closed transition reports no generation: its map is gone.
                handler.end(MLN_CAMERA_TRANSITION_OUTCOME_CLOSED, 0);
            }
            let queries = entry.queries.drain().map(|(_, q)| q.completion);
            let still = entry.still.take();
            for c in entry.style_waiters.drain(..).chain(queries).chain(still) {
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
                    // The body reads the map's handle by the name `$state`.
                    #[allow(unused_variables)]
                    let $state = map;
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
        let known = MLN_MAP_DEBUG_TILE_BORDERS
            | MLN_MAP_DEBUG_PARSE_STATUS
            | MLN_MAP_DEBUG_TIMESTAMPS
            | MLN_MAP_DEBUG_COLLISION
            | MLN_MAP_DEBUG_OVERDRAW
            | MLN_MAP_DEBUG_STENCIL_CLIP
            | MLN_MAP_DEBUG_DEPTH_BUFFER;
        if flags & !known != 0 {
            return Err(Failure::argument("debug flags contain unknown bits"));
        }
        Ok(job(move |entry| {
            entry.unchanged = entry.debug_flags == flags;
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
            entry.unchanged = entry.stats_view == enabled;
            entry.stats_view = enabled;
            Ok(())
        }))
    }
);

command!(mln_map_dump_debug_logs(), false, false, |state, id| {
    Ok(job(move |entry| {
        // Native's dump: the style's URL and its sources between rules.
        let rule = "-".repeat(80);
        let info = |message: &str| crate::logging::log(MLN_LOG_SEVERITY_INFO, message);
        info(&rule);
        info(&format!("styleURL: {}", entry.style_url));
        if let Some(style) = entry.map.style() {
            for source in style.source_ids() {
                info(&format!("Source::id: {source}"));
            }
        }
        info(&rule);
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
            entry.unchanged = true;
            Ok(())
        }))
    }
);

command!(
    mln_map_resize(size: mln_logical_size),
    false,
    true,
    |map, id| {
        if size.width == 0 || size.height == 0 {
            return Err(Failure::argument("the size is empty"));
        }
        if handles::map(map.0)?.has_session.load(Ordering::Acquire) {
            return Err(Failure::state(
                "the map has an attached render session: resize the session instead",
            ));
        }
        Ok(job(move |entry| {
            entry.unchanged =
                (entry.extent.width, entry.extent.height) == (size.width, size.height);
            entry.extent.width = size.width;
            entry.extent.height = size.height;
            let before = entry.map.camera();
            entry
                .map
                .set_viewport(f64::from(size.width), f64::from(size.height));
            // A resize that moves the camera back within its limits
            // cancels its transitions, as in Native.
            if entry.map.camera() != before {
                cancel(entry, None);
            }
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
        // The outermost end publishes what the groups held, as one update.
        entry.unchanged = true;
        if entry.groups == 0 && std::mem::take(&mut entry.held_update) {
            entry.render_update_generation += 1;
        }
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
        let known = MLN_MAP_TILE_OPTION_PREFETCH_ZOOM_DELTA
            | MLN_MAP_TILE_OPTION_LOD_MIN_RADIUS
            | MLN_MAP_TILE_OPTION_LOD_SCALE
            | MLN_MAP_TILE_OPTION_LOD_PITCH_THRESHOLD
            | MLN_MAP_TILE_OPTION_LOD_ZOOM_SHIFT
            | MLN_MAP_TILE_OPTION_LOD_MODE;
        if o.fields & !known != 0 {
            return Err(Failure::argument("the tile options have unknown bits"));
        }
        let has = |bit| o.fields & bit != 0;
        if has(MLN_MAP_TILE_OPTION_PREFETCH_ZOOM_DELTA) && o.prefetch_zoom_delta > 255 {
            return Err(Failure::argument("prefetch_zoom_delta is at most 255"));
        }
        for (bit, value, what) in [
            (
                MLN_MAP_TILE_OPTION_LOD_MIN_RADIUS,
                o.lod_min_radius,
                "lod_min_radius",
            ),
            (MLN_MAP_TILE_OPTION_LOD_SCALE, o.lod_scale, "lod_scale"),
            (
                MLN_MAP_TILE_OPTION_LOD_PITCH_THRESHOLD,
                o.lod_pitch_threshold,
                "lod_pitch_threshold",
            ),
            (
                MLN_MAP_TILE_OPTION_LOD_ZOOM_SHIFT,
                o.lod_zoom_shift,
                "lod_zoom_shift",
            ),
        ] {
            if has(bit) && !value.is_finite() {
                return Err(Failure::argument(format!("{what} must be finite")));
            }
        }
        if has(MLN_MAP_TILE_OPTION_LOD_MODE) && o.lod_mode > MLN_TILE_LOD_MODE_DISTANCE {
            return Err(Failure::argument("lod_mode is invalid"));
        }
        Ok(job(move |entry| {
            let before = (entry.map.tile_lod(), entry.prefetch_zoom_delta);
            let mut lod = before.0;
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
                lod.pitch_threshold = o.lod_pitch_threshold.to_radians();
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
            entry.unchanged = (lod, entry.prefetch_zoom_delta) == before;
            if lod != before.0 {
                entry.map.set_tile_lod(lod);
            }
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
        let known = MLN_MAP_VIEWPORT_OPTION_NORTH_ORIENTATION
            | MLN_MAP_VIEWPORT_OPTION_CONSTRAIN_MODE
            | MLN_MAP_VIEWPORT_OPTION_VIEWPORT_MODE
            | MLN_MAP_VIEWPORT_OPTION_FRUSTUM_OFFSET;
        if o.fields & !known != 0 {
            return Err(Failure::argument("the viewport options have unknown bits"));
        }
        let has = |bit| o.fields & bit != 0;
        if has(MLN_MAP_VIEWPORT_OPTION_NORTH_ORIENTATION)
            && o.north_orientation > MLN_NORTH_ORIENTATION_LEFT
        {
            return Err(Failure::argument("north_orientation is invalid"));
        }
        if has(MLN_MAP_VIEWPORT_OPTION_CONSTRAIN_MODE)
            && o.constrain_mode > MLN_CONSTRAIN_MODE_SCREEN
        {
            return Err(Failure::argument("constrain_mode is invalid"));
        }
        if has(MLN_MAP_VIEWPORT_OPTION_VIEWPORT_MODE)
            && o.viewport_mode > MLN_VIEWPORT_MODE_FLIPPED_Y
        {
            return Err(Failure::argument("viewport_mode is invalid"));
        }
        if has(MLN_MAP_VIEWPORT_OPTION_FRUSTUM_OFFSET) {
            let f = o.frustum_offset;
            let sides = [f.top, f.left, f.bottom, f.right];
            if !sides.iter().all(|v| v.is_finite()) {
                return Err(Failure::argument(
                    "the frustum offset's sides must be finite",
                ));
            }
            if sides.iter().any(|v| *v < 0.0) {
                return Err(Failure::argument(
                    "the frustum offset's sides must be greater than or equal to 0",
                ));
            }
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
    mln_map_set_projection_mode(options: *const mln_projection_mode_options),
    false,
    true,
    |state, id| {
        let o = *sized(options, "options")?;
        let known = MLN_PROJECTION_MODE_OPTION_AXONOMETRIC
            | MLN_PROJECTION_MODE_OPTION_X_SKEW
            | MLN_PROJECTION_MODE_OPTION_Y_SKEW;
        if o.fields & !known != 0 {
            return Err(Failure::argument(
                "the projection mode options have unknown bits",
            ));
        }
        let has = |bit| o.fields & bit != 0;
        for (bit, value, what) in [
            (MLN_PROJECTION_MODE_OPTION_X_SKEW, o.x_skew, "x_skew"),
            (MLN_PROJECTION_MODE_OPTION_Y_SKEW, o.y_skew, "y_skew"),
        ] {
            if has(bit) && !value.is_finite() {
                return Err(Failure::argument(format!("{what} must be finite")));
            }
        }
        Ok(job(move |entry| {
            let has = |bit| o.fields & bit != 0;
            let mut mode = entry.projection_mode;
            if has(MLN_PROJECTION_MODE_OPTION_AXONOMETRIC) {
                mode.axonometric = o.axonometric;
            }
            if has(MLN_PROJECTION_MODE_OPTION_X_SKEW) {
                mode.x_skew = o.x_skew;
            }
            if has(MLN_PROJECTION_MODE_OPTION_Y_SKEW) {
                mode.y_skew = o.y_skew;
            }
            let before = entry.projection_mode;
            entry.unchanged = (mode.axonometric, mode.x_skew, mode.y_skew)
                == (before.axonometric, before.x_skew, before.y_skew);
            entry.projection_mode = mode;
            entry
                .map
                .set_axonometric(mode.axonometric.then_some(orb_api::Skew {
                    x: mode.x_skew,
                    y: mode.y_skew,
                }));
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
        let known = MLN_BOUND_OPTION_BOUNDS
            | MLN_BOUND_OPTION_MIN_ZOOM
            | MLN_BOUND_OPTION_MAX_ZOOM
            | MLN_BOUND_OPTION_MIN_PITCH
            | MLN_BOUND_OPTION_MAX_PITCH
            | MLN_BOUND_OPTION_UNBOUNDED;
        if o.fields & !known != 0 {
            return Err(Failure::argument("the bound options have unknown bits"));
        }
        let has = |bit| o.fields & bit != 0;
        if has(MLN_BOUND_OPTION_BOUNDS) && has(MLN_BOUND_OPTION_UNBOUNDED) {
            return Err(Failure::argument(
                "bounds and unbounded don't combine: set one",
            ));
        }
        if has(MLN_BOUND_OPTION_BOUNDS) {
            check_bounds(&o.bounds)?;
        }
        for (bit, value, what) in [
            (MLN_BOUND_OPTION_MIN_ZOOM, o.min_zoom, "min_zoom"),
            (MLN_BOUND_OPTION_MAX_ZOOM, o.max_zoom, "max_zoom"),
            (MLN_BOUND_OPTION_MIN_PITCH, o.min_pitch, "min_pitch"),
            (MLN_BOUND_OPTION_MAX_PITCH, o.max_pitch, "max_pitch"),
        ] {
            if has(bit) && !value.is_finite() {
                return Err(Failure::argument(format!("{what} must be finite")));
            }
        }
        if has(MLN_BOUND_OPTION_MIN_ZOOM)
            && has(MLN_BOUND_OPTION_MAX_ZOOM)
            && o.min_zoom > o.max_zoom
        {
            return Err(Failure::argument("min_zoom is above max_zoom"));
        }
        if has(MLN_BOUND_OPTION_MIN_PITCH)
            && has(MLN_BOUND_OPTION_MAX_PITCH)
            && o.min_pitch > o.max_pitch
        {
            return Err(Failure::argument("min_pitch is above max_pitch"));
        }
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
            // A new limit the camera is past moves it there at once, as
            // a jump, which takes that property from its animations.
            let (zoom, pitch) = (entry.map.zoom(), entry.map.camera().pitch);
            let limited = CameraTarget {
                zoom: (zoom < b.min_zoom || zoom > b.max_zoom)
                    .then(|| zoom.clamp(b.min_zoom, b.max_zoom)),
                pitch: (pitch < b.min_pitch || pitch > b.max_pitch)
                    .then(|| pitch.clamp(b.min_pitch, b.max_pitch)),
                ..CameraTarget::default()
            };
            entry.map.set_constraints(constraints(&b));
            if limited != CameraTarget::default() {
                entry.map.jump_to(&limited);
            }
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
        if json.data.is_null() || json.size == 0 {
            return Err(Failure::argument("the style JSON must not be empty"));
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

/// Checks camera options as mln-ffi does, before the command is accepted.
/// The options' own `size` isn't read: the struct that holds them
/// versions them.
fn check_camera(c: &mln_camera_options) -> Result {
    let known = MLN_CAMERA_OPTION_CENTER
        | MLN_CAMERA_OPTION_ZOOM
        | MLN_CAMERA_OPTION_BEARING
        | MLN_CAMERA_OPTION_PITCH
        | MLN_CAMERA_OPTION_CENTER_ALTITUDE
        | MLN_CAMERA_OPTION_PADDING
        | MLN_CAMERA_OPTION_ANCHOR
        | MLN_CAMERA_OPTION_ROLL
        | MLN_CAMERA_OPTION_FOV;
    if c.fields & !known != 0 {
        return Err(Failure::argument("the camera options have unknown bits"));
    }
    let has = |bit| c.fields & bit != 0;
    if has(MLN_CAMERA_OPTION_CENTER) {
        check_coordinate(&c.center, "the center")?;
    }
    for (bit, value, what) in [
        (MLN_CAMERA_OPTION_ZOOM, c.zoom, "zoom"),
        (MLN_CAMERA_OPTION_BEARING, c.bearing, "bearing"),
        (MLN_CAMERA_OPTION_PITCH, c.pitch, "pitch"),
        (MLN_CAMERA_OPTION_ROLL, c.roll, "roll"),
        (MLN_CAMERA_OPTION_FOV, c.field_of_view, "field_of_view"),
        (
            MLN_CAMERA_OPTION_CENTER_ALTITUDE,
            c.center_altitude,
            "center_altitude",
        ),
    ] {
        if has(bit) && !value.is_finite() {
            return Err(Failure::argument(format!("{what} must be finite")));
        }
    }
    if has(MLN_CAMERA_OPTION_PADDING) {
        check_insets(&c.padding)?;
    }
    if has(MLN_CAMERA_OPTION_ANCHOR) {
        check_point(&c.anchor)?;
    }
    Ok(())
}

pub(crate) fn check_coordinate(l: &mln_lat_lng, what: &str) -> Result {
    if !(l.latitude.is_finite() && l.longitude.is_finite()) {
        return Err(Failure::argument(format!(
            "{what}'s latitude and longitude must be finite"
        )));
    }
    if l.latitude.abs() > 90.0 {
        return Err(Failure::argument(format!(
            "{what}'s latitude must be within [-90, 90]"
        )));
    }
    Ok(())
}

fn check_insets(p: &mln_edge_insets) -> Result {
    let values = [p.top, p.left, p.bottom, p.right];
    if !values.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return Err(Failure::argument(
            "padding values must be finite and greater than or equal to 0",
        ));
    }
    Ok(())
}

fn check_point(p: &mln_screen_point) -> Result {
    if !(p.x.is_finite() && p.y.is_finite()) {
        return Err(Failure::argument("screen point values must be finite"));
    }
    Ok(())
}

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
        // An unbounded map wraps the center into the world; bounds, the
        // world's too, hold it within them.
        let lon = if entry.bounds.unbounded {
            (c.longitude + 180.0).rem_euclid(360.0) - 180.0
        } else {
            c.longitude
        };
        t.center = Some(Geodetic::new(lon, c.latitude));
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
        let known = MLN_ANIMATION_OPTION_DURATION
            | MLN_ANIMATION_OPTION_VELOCITY
            | MLN_ANIMATION_OPTION_MIN_ZOOM
            | MLN_ANIMATION_OPTION_EASING
            | MLN_ANIMATION_OPTION_TRANSITION_ID;
        if a.fields & !known != 0 {
            return Err(Failure::argument("the animation options have unknown bits"));
        }
        let has = |bit| a.fields & bit != 0;
        if has(MLN_ANIMATION_OPTION_DURATION)
            && !(a.duration_ms.is_finite()
                && a.duration_ms >= 0.0
                && a.duration_ms <= orb_camera::MAX_DURATION_MS)
        {
            return Err(Failure::argument(
                "duration_ms must be a non-negative duration within the native duration range",
            ));
        }
        if has(MLN_ANIMATION_OPTION_VELOCITY) && !(a.velocity.is_finite() && a.velocity > 0.0) {
            return Err(Failure::argument("velocity must be positive and finite"));
        }
        if has(MLN_ANIMATION_OPTION_MIN_ZOOM) && !a.min_zoom.is_finite() {
            return Err(Failure::argument("min_zoom must be finite"));
        }
        if has(MLN_ANIMATION_OPTION_EASING) {
            let e = a.easing;
            let unit = |v: f64| (0.0..=1.0).contains(&v);
            if !(unit(e.x1) && unit(e.x2) && e.y1.is_finite() && e.y2.is_finite()) {
                return Err(Failure::argument(
                    "the easing's x control points must be within [0, 1]",
                ));
            }
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
    phase: mln_gesture_phase,
) -> Result {
    let duration =
        (animation.fields & MLN_ANIMATION_OPTION_DURATION != 0).then_some(animation.duration_ms);
    gesture_phase(entry, phase);
    if phase != MLN_GESTURE_PHASE_NONE {
        // A gesture's update can repeat a value it doesn't change.
        let camera = entry.map.camera();
        let now = [entry.map.zoom(), camera.bearing, camera.pitch];
        for ((changed, to), now) in entry
            .gesture_changed
            .iter_mut()
            .zip([t.zoom, t.bearing, t.pitch])
            .zip(now)
        {
            *changed |= to.is_some_and(|v| v != now);
        }
    }
    // Native eases without a duration at once, and a flight of no duration
    // jumps.
    let immediate = match mode {
        MLN_CAMERA_UPDATE_MODE_EASE => duration.is_none_or(|d| d == 0.0),
        MLN_CAMERA_UPDATE_MODE_FLY => duration == Some(0.0),
        _ => true,
    };
    let mode = if immediate {
        MLN_CAMERA_UPDATE_MODE_JUMP
    } else {
        mode
    };
    entry.change = (
        if immediate {
            MLN_CAMERA_CHANGE_MODE_IMMEDIATE
        } else {
            MLN_CAMERA_CHANGE_MODE_ANIMATED
        },
        if phase == MLN_GESTURE_PHASE_NONE {
            MLN_CAMERA_CHANGE_SOURCE_COMMAND
        } else {
            MLN_CAMERA_CHANGE_SOURCE_GESTURE
        },
    );
    if immediate {
        // Native reports an immediate move as it makes it.
        for kind in [
            MLN_RUNTIME_EVENT_MAP_CAMERA_WILL_CHANGE,
            MLN_RUNTIME_EVENT_MAP_CAMERA_DID_CHANGE,
        ] {
            let payload = Payload::CameraChange(entry.change.0, entry.change.1);
            entry.camera_events.push((kind, payload, String::new()));
        }
    }
    // An animation reports its own changes, and an earlier jump's with
    // them.
    entry.jumped &= immediate;
    let animation_id = match mode {
        MLN_CAMERA_UPDATE_MODE_JUMP => {
            let before = entry.map.camera();
            entry.map.jump_to(t);
            entry.jumped |= entry.map.camera() != before;
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
    track(entry, animation_id, animation);
    Ok(())
}

/// Keeps a command's animation for its end handler and its transition
/// ID; a command that moved at once ends now.
fn track(entry: &mut MapEntry, animation_id: Option<AnimationId>, animation: &Animation) {
    match animation_id {
        Some(a) => entry.transitions.push(Transition::new(vec![a], animation)),
        None => entry.ended.extend(
            animation
                .handler
                .map(|h| (h, MLN_CAMERA_TRANSITION_OUTCOME_COMPLETED)),
        ),
    }
}

/// A gesture's phase: a begin cancels the camera's transitions, and the
/// snapshot shows a gesture from its begin to its end or cancel, which
/// cancels the transitions started during it.
fn gesture_phase(entry: &mut MapEntry, phase: mln_gesture_phase) {
    // A begin, or the first phase after the input ended, starts a gesture.
    let input = entry.gesture && entry.momentum.is_none();
    if phase == MLN_GESTURE_PHASE_BEGIN || (phase != MLN_GESTURE_PHASE_NONE && !input) {
        entry.gesture_changed = [false; 3];
    }
    match phase {
        MLN_GESTURE_PHASE_BEGIN => {
            // The glide a new gesture stops no longer ends a gesture.
            entry.momentum = None;
            cancel(entry, None);
            entry.gesture = true;
        }
        MLN_GESTURE_PHASE_UPDATE => entry.gesture = true,
        MLN_GESTURE_PHASE_END => entry.gesture = false,
        MLN_GESTURE_PHASE_CANCEL => {
            cancel(entry, None);
            entry.gesture = false;
        }
        _ => {}
    }
}

command!(
    mln_map_update_camera(update: *const mln_camera_update),
    false,
    true,
    |state, id| {
        let update = *sized(update, "update")?;
        if update.mode > MLN_CAMERA_UPDATE_MODE_FLY
            || update.gesture_phase > MLN_GESTURE_PHASE_CANCEL
        {
            return Err(Failure::argument(
                "the update's mode or gesture phase is invalid",
            ));
        }
        check_camera(&update.camera)?;
        let (mode, camera, phase) = (update.mode, update.camera, update.gesture_phase);
        let animation = Animation::from_c(&update.animation)?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let t = target(entry, &camera)?;
            update_camera(entry, mode, &t, &animation, phase)?;
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

/// The camera `t` sets (its zoom, bearing and pitch) moved so that `keep`
/// stays at screen point `at`: zoom, rotation and tilt about a point.
/// Panning by a screen offset is exact only on a flat view, so the pan
/// is refined until the point sits within a billionth of a pixel.
fn keeping(entry: &MapEntry, mut t: CameraTarget, keep: Geodetic, at: glam::DVec2) -> CameraTarget {
    let camera = entry.map.camera();
    t.center = Some(t.center.unwrap_or(camera.target));
    t.zoom = Some(t.zoom.unwrap_or_else(|| entry.map.zoom()));
    t.bearing = Some(t.bearing.unwrap_or(camera.bearing));
    t.pitch = Some(t.pitch.unwrap_or(camera.pitch));
    for _ in 0..32 {
        let (Some(p), Some(middle)) = (
            entry.map.project_with(&t, keep),
            t.center.and_then(|c| entry.map.project_with(&t, c)),
        ) else {
            break;
        };
        let off = at - p;
        if off.length() < 1e-9 {
            break;
        }
        match entry.map.unproject_with(&t, middle - off) {
            Some(center) => t.center = Some(center),
            None => break,
        }
    }
    t
}

/// A host's gesture step on Orbis's camera: the pan moves the map's
/// content by the offset, then zoom, rotation and tilt turn about the
/// anchor, or the center.
fn apply_delta(entry: &mut MapEntry, d: &Delta) -> Result {
    let has = |bit| d.fields & bit != 0;
    let animated = d.animation.fields & MLN_ANIMATION_OPTION_DURATION != 0;
    let mut t = CameraTarget::default();
    if has(MLN_CAMERA_DELTA_OFFSET) && (d.offset.x != 0.0 || d.offset.y != 0.0) {
        let offset = glam::DVec2::new(d.offset.x, d.offset.y);
        let center = entry.map.camera().target;
        let middle = entry.map.project(center).unwrap_or_default();
        t.center = entry.map.unproject(middle - offset);
    }
    let camera = entry.map.camera();
    if has(MLN_CAMERA_DELTA_SCALE) {
        t.zoom = Some(entry.map.zoom() + d.scale.log2());
    }
    if has(MLN_CAMERA_DELTA_BEARING) {
        t.bearing = Some(camera.bearing + d.bearing);
    }
    if has(MLN_CAMERA_DELTA_PITCH) {
        t.pitch = Some(camera.pitch + d.pitch);
    }
    if has(MLN_CAMERA_DELTA_ANCHOR) {
        // The pan comes first: the anchor keeps what the pan put under it.
        let at = glam::DVec2::new(d.anchor.x, d.anchor.y);
        let panned = CameraTarget {
            center: t.center,
            ..Default::default()
        };
        if let Some(keep) = entry.map.unproject_with(&panned, at) {
            t = keeping(entry, t, keep, at);
        }
    }
    let mode = if animated {
        MLN_CAMERA_UPDATE_MODE_EASE
    } else {
        MLN_CAMERA_UPDATE_MODE_JUMP
    };
    update_camera(entry, mode, &t, &d.animation, d.gesture_phase)
}

/// A part's decay over its defaults, checked as mln-ffi checks it.
fn decay(
    d: &mln_momentum_decay,
    defaults: orb_api::MomentumDecay,
) -> Result<orb_api::MomentumDecay> {
    let has = |bit| d.fields & bit != 0;
    let mut out = defaults;
    let limit = |v: f64| {
        if v.is_nan() || v < 0.0 {
            Err(Failure::argument(
                "a momentum decay's speeds and distance are numbers that isn't negative",
            ))
        } else {
            Ok(v)
        }
    };
    if has(MLN_MOMENTUM_DECAY_MIN_SPEED) {
        out.min_speed = limit(d.min_speed)?;
    }
    if has(MLN_MOMENTUM_DECAY_MAX_SPEED) {
        out.max_speed = limit(d.max_speed)?;
    }
    if has(MLN_MOMENTUM_DECAY_MAX_DISTANCE) {
        out.max_distance = limit(d.max_distance)?;
    }
    let ease = has(MLN_MOMENTUM_DECAY_MODEL) && d.model == MLN_MOMENTUM_DECAY_MODEL_EASE;
    if has(MLN_MOMENTUM_DECAY_MODEL) && d.model > MLN_MOMENTUM_DECAY_MODEL_EASE {
        return Err(Failure::argument("the momentum decay model is invalid"));
    }
    if ease {
        if !(has(MLN_MOMENTUM_DECAY_DISTANCE) && has(MLN_MOMENTUM_DECAY_DURATION_MS)) {
            return Err(Failure::argument(
                "an eased glide needs a distance and a duration",
            ));
        }
        if !(d.distance.is_finite() && d.duration_ms.is_finite() && d.duration_ms > 0.0) {
            return Err(Failure::argument(
                "an eased glide needs a distance and a duration that are finite",
            ));
        }
        let mut easing = Easing::Bezier([0.175, 0.5, 0.35, 1.0]);
        if has(MLN_MOMENTUM_DECAY_EASING) {
            let e = d.easing;
            let unit = |v: f64| (0.0..=1.0).contains(&v);
            if !(unit(e.x1) && unit(e.y1) && unit(e.x2) && unit(e.y2)) {
                return Err(Failure::argument(
                    "an eased glide's control points are within [0, 1] only",
                ));
            }
            easing = Easing::Bezier([e.x1, e.y1, e.x2, e.y2]);
        }
        out.model = orb_api::DecayModel::Ease {
            distance: d.distance,
            duration: d.duration_ms,
            easing,
        };
    } else if let orb_api::DecayModel::Friction {
        mut half_life,
        mut stop_speed,
    } = defaults.model
    {
        if has(MLN_MOMENTUM_DECAY_HALF_LIFE_MS) {
            if !(d.half_life_ms.is_finite() && d.half_life_ms > 0.0) {
                return Err(Failure::argument("friction needs a positive half-life"));
            }
            half_life = d.half_life_ms;
        }
        if has(MLN_MOMENTUM_DECAY_STOP_SPEED) {
            if !(d.stop_speed.is_finite() && d.stop_speed > 0.0) {
                return Err(Failure::argument("friction needs a positive stop speed"));
            }
            stop_speed = d.stop_speed;
        }
        out.model = orb_api::DecayModel::Friction {
            half_life,
            stop_speed,
        };
    }
    Ok(out)
}

/// A zoom, bearing or pitch part of a release, over its defaults.
fn anchored(
    m: &mln_anchored_momentum,
    part: orb_api::AnchoredMomentum,
    snap_defaults: orb_api::MomentumSnap,
) -> Result<Option<orb_api::AnchoredMomentum>> {
    let has = |bit| m.fields & bit != 0;
    // The decay is checked even for a part that doesn't move.
    let checked = decay(&m.decay, part.decay)?;
    if has(MLN_ANCHORED_MOMENTUM_VELOCITY) && !m.velocity.is_finite() {
        return Err(Failure::argument("momentum velocities must be finite"));
    }
    let mut out = part;
    out.velocity = has(MLN_ANCHORED_MOMENTUM_VELOCITY).then_some(m.velocity);
    out.decay = checked;
    if has(MLN_ANCHORED_MOMENTUM_ANCHOR) {
        check_point(&m.anchor)?;
        out.anchor = Some(glam::DVec2::new(m.anchor.x, m.anchor.y));
    }
    if has(MLN_ANCHORED_MOMENTUM_SNAP) {
        let s = &m.snap;
        let set = |bit| s.fields & bit != 0;
        let mut snap = snap_defaults;
        if set(MLN_MOMENTUM_SNAP_SPACING) {
            if !(s.spacing.is_finite() && s.spacing > 0.0) {
                return Err(Failure::argument("a snap needs a positive spacing"));
            }
            snap.spacing = s.spacing;
        }
        if set(MLN_MOMENTUM_SNAP_OFFSET) {
            if !s.offset.is_finite() {
                return Err(Failure::argument("a snap's offset must be finite"));
            }
            snap.offset = s.offset;
        }
        if set(MLN_MOMENTUM_SNAP_TOLERANCE) {
            if !(s.tolerance.is_finite() && s.tolerance >= 0.0) {
                return Err(Failure::argument("a snap's tolerance isn't negative"));
            }
            snap.tolerance = s.tolerance;
        }
        if set(MLN_MOMENTUM_SNAP_DURATION_MS) {
            if !(s.duration_ms.is_finite() && s.duration_ms >= 0.0) {
                return Err(Failure::argument("a snap's duration isn't negative"));
            }
            snap.duration = s.duration_ms;
        }
        out.snap = Some(snap);
    }
    // Every part is checked; only one with a velocity or a snap moves.
    Ok((out.velocity.is_some() || out.snap.is_some()).then_some(out))
}

/// A release's momentum as Orbis's camera takes it.
fn momentum(m: &mln_gesture_momentum) -> Result<orb_api::GestureMomentum> {
    let pan_decay = decay(&m.pan.decay, orb_api::MomentumDecay::PAN)?;
    let pan = if m.pan.fields & MLN_PAN_MOMENTUM_VELOCITY != 0 {
        let v = m.pan.velocity;
        if !(v.x.is_finite() && v.y.is_finite()) {
            return Err(Failure::argument("momentum velocities must be finite"));
        }
        let mut pan = orb_api::PanMomentum::new(glam::DVec2::new(v.x, v.y));
        pan.decay = pan_decay;
        Some(pan)
    } else {
        None
    };
    Ok(orb_api::GestureMomentum {
        pan,
        zoom: anchored(
            &m.zoom,
            orb_api::AnchoredMomentum::zoom(None),
            orb_api::MomentumSnap::ZOOM,
        )?,
        bearing: anchored(
            &m.bearing,
            orb_api::AnchoredMomentum::bearing(None),
            orb_api::MomentumSnap::BEARING,
        )?,
        pitch: anchored(
            &m.pitch,
            orb_api::AnchoredMomentum::pitch(None),
            orb_api::MomentumSnap::PITCH,
        )?,
    })
}

/// A gesture's release: its last step, then the glide its momentum
/// starts, which ends the gesture when it ends. A release that moves
/// nothing over time ends the gesture at once.
fn release_gesture(entry: &mut MapEntry, d: &Delta, glide: &orb_api::GestureMomentum) -> Result {
    let step = Delta {
        animation: Animation {
            handler: None,
            ..d.animation
        },
        gesture_phase: MLN_GESTURE_PHASE_UPDATE,
        ..*d
    };
    apply_delta(entry, &step)?;
    // Only what the gesture changed settles.
    let mut glide = *glide;
    for (part, changed) in [&mut glide.zoom, &mut glide.bearing, &mut glide.pitch]
        .into_iter()
        .zip(entry.gesture_changed)
    {
        if let Some(part) = part.as_mut().filter(|_| !changed) {
            part.snap = None;
        }
    }
    let started = entry.map.start_momentum(&glide);
    entry.momentum = started;
    entry.gesture = started.is_some();
    if started.is_some() {
        entry.jumped = false;
        entry.change = (
            MLN_CAMERA_CHANGE_MODE_ANIMATED,
            MLN_CAMERA_CHANGE_SOURCE_MOMENTUM,
        );
    }
    track(entry, started, &d.animation);
    Ok(())
}

/// Checks a delta as mln-ffi does: only the fields it selects are read.
fn check_delta(d: &mln_camera_delta) -> Result {
    let known = MLN_CAMERA_DELTA_OFFSET
        | MLN_CAMERA_DELTA_SCALE
        | MLN_CAMERA_DELTA_BEARING
        | MLN_CAMERA_DELTA_PITCH
        | MLN_CAMERA_DELTA_ANCHOR
        | MLN_CAMERA_DELTA_MOMENTUM;
    if d.fields & !known != 0 {
        return Err(Failure::argument("the delta's fields have unknown bits"));
    }
    if d.gesture_phase > MLN_GESTURE_PHASE_CANCEL {
        return Err(Failure::argument("the delta's gesture phase is invalid"));
    }
    let has = |bit| d.fields & bit != 0;
    if has(MLN_CAMERA_DELTA_OFFSET) && !(d.offset.x.is_finite() && d.offset.y.is_finite()) {
        return Err(Failure::argument("the offset's values must be finite"));
    }
    if has(MLN_CAMERA_DELTA_SCALE) && !(d.scale.is_finite() && d.scale > 0.0) {
        return Err(Failure::argument("scale must be finite and positive"));
    }
    if (has(MLN_CAMERA_DELTA_BEARING) && !d.bearing.is_finite())
        || (has(MLN_CAMERA_DELTA_PITCH) && !d.pitch.is_finite())
    {
        return Err(Failure::argument(
            "the delta's bearing and pitch must be finite",
        ));
    }
    if has(MLN_CAMERA_DELTA_ANCHOR) {
        if !(has(MLN_CAMERA_DELTA_SCALE)
            || has(MLN_CAMERA_DELTA_BEARING)
            || has(MLN_CAMERA_DELTA_PITCH))
        {
            return Err(Failure::argument(
                "the anchor requires scale, bearing, or pitch",
            ));
        }
        check_point(&d.anchor)?;
        if has(MLN_CAMERA_DELTA_OFFSET) && d.animation.fields & MLN_ANIMATION_OPTION_DURATION != 0 {
            return Err(Failure::argument(
                "an anchor combines with an offset only in an immediate delta",
            ));
        }
    }
    Ok(())
}

command!(
    mln_map_apply_camera_delta(delta: *const mln_camera_delta),
    false,
    true,
    |state, id| {
        let d = *sized(delta, "delta")?;
        check_delta(&d)?;
        let glide = if d.fields & MLN_CAMERA_DELTA_MOMENTUM != 0 {
            if d.gesture_phase != MLN_GESTURE_PHASE_END {
                return Err(Failure::argument(
                    "only a delta in the end phase carries momentum",
                ));
            }
            if d.animation.fields & !MLN_ANIMATION_OPTION_TRANSITION_ID != 0 {
                return Err(Failure::argument(
                    "a delta with momentum sets only the transition ID of its animation",
                ));
            }
            Some(momentum(&d.momentum)?)
        } else {
            None
        };
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
        Ok(job(move |entry| match glide {
            None => apply_delta(entry, &delta),
            Some(glide) => release_gesture(entry, &delta, &glide),
        }))
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

/// Ends the transitions `which` names, or all of them, as cancelled, and
/// stops their animations where they are. Each ends with its animations,
/// whose ends report the speed they had left.
fn cancel(entry: &mut MapEntry, which: Option<u64>) {
    let mut stopping = Vec::new();
    for t in &mut entry.transitions {
        if which.is_none() || t.id == which {
            t.outcome = Some(MLN_CAMERA_TRANSITION_OUTCOME_CANCELLED);
            stopping.extend_from_slice(&t.animations);
        }
    }
    match which {
        None => entry.map.stop(),
        Some(_) => {
            for a in stopping {
                entry.map.stop_animation(a);
            }
        }
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

/// The Earth's circumference at the equator, metres: Native's free
/// camera measures altitude in Web Mercator units of it.
const EARTH_CIRCUMFERENCE_M: f64 = 2.0 * std::f64::consts::PI * 6_378_137.0;

/// Native's free camera orientation: the bearing turns about the up axis,
/// then the pitch about the view's right axis, then the roll about the
/// view axis.
fn orientation(bearing: f64, pitch: f64, roll: f64) -> glam::DQuat {
    glam::DQuat::from_rotation_z(bearing.to_radians())
        * glam::DQuat::from_rotation_x(-pitch.to_radians())
        * glam::DQuat::from_rotation_z(roll.to_radians())
}

/// The bearing, pitch and roll of a unit orientation, in degrees, as
/// Native reads them back from its view's frame.
fn bearing_pitch_roll(q: glam::DQuat) -> (f64, f64, f64) {
    use glam::DVec3;
    let forward = q * DVec3::new(0.0, 0.0, -1.0);
    let up = q * DVec3::new(0.0, -1.0, 0.0);
    let right = up.cross(forward);
    let flat = forward.x.hypot(forward.y);
    let pitch = flat.atan2(-forward.z);
    // Looking straight down, the frame's right gives the bearing.
    let (bearing, roll) = if flat > 1e-3 {
        (forward.x.atan2(-forward.y), -right.z.atan2(up.z))
    } else {
        (-(-right.y).atan2(right.x), 0.0)
    };
    (bearing.to_degrees(), pitch.to_degrees(), roll.to_degrees())
}

/// The map's free camera in Native's terms: the eye in Web Mercator units,
/// and its orientation.
pub(crate) fn free_camera(map: &orb_api::Map) -> mln_free_camera {
    let free = map.free_camera();
    let (lon, lat) = (free.position.lon, free.position.lat);
    let q = orientation(free.bearing, free.pitch, free.roll);
    mln_free_camera {
        position: mln_vec3 {
            x: (lon + 180.0) / 360.0,
            y: 0.5
                - (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0)
                    .tan()
                    .ln()
                    / (2.0 * std::f64::consts::PI),
            z: free.altitude / (lat.to_radians().cos() * EARTH_CIRCUMFERENCE_M),
        },
        orientation: mln_quaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        },
    }
}

// Puts the camera's eye where the options say at once, cancelling its
// transitions; what they leave out stays.
command!(
    mln_map_set_free_camera_options(options: *const mln_free_camera_options),
    false,
    true,
    |state, id| {
        let o = *sized(options, "options")?;
        let known = MLN_FREE_CAMERA_OPTION_POSITION | MLN_FREE_CAMERA_OPTION_ORIENTATION;
        if o.fields & !known != 0 {
            return Err(Failure::argument(
                "the free camera options have unknown bits",
            ));
        }
        let fields = o.fields;
        let has = move |bit| fields & bit != 0;
        let p = o.position;
        if has(MLN_FREE_CAMERA_OPTION_POSITION)
            && !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
        {
            return Err(Failure::argument("the position must be finite"));
        }
        let q = o.orientation;
        let q = glam::DQuat::from_xyzw(q.x, q.y, q.z, q.w);
        if has(MLN_FREE_CAMERA_OPTION_ORIENTATION) {
            if !q.is_finite() {
                return Err(Failure::argument("the orientation must be finite"));
            }
            if q.length() == 0.0 {
                return Err(Failure::argument(
                    "the orientation is a quaternion of zero length",
                ));
            }
        }
        Ok(job(move |entry| {
            cancel(entry, None);
            let mut free = entry.map.free_camera();
            if has(MLN_FREE_CAMERA_OPTION_POSITION) {
                let lat = (std::f64::consts::PI * (1.0 - 2.0 * p.y))
                    .sinh()
                    .atan()
                    .to_degrees();
                free.position = Geodetic::new(p.x * 360.0 - 180.0, lat);
                free.altitude = p.z * lat.to_radians().cos() * EARTH_CIRCUMFERENCE_M;
            }
            if has(MLN_FREE_CAMERA_OPTION_ORIENTATION) {
                (free.bearing, free.pitch, free.roll) = bearing_pitch_roll(q.normalize());
            }
            entry.change = (
                MLN_CAMERA_CHANGE_MODE_IMMEDIATE,
                MLN_CAMERA_CHANGE_SOURCE_COMMAND,
            );
            let before = entry.map.camera();
            entry.map.jump_to_free(&free);
            if entry.map.camera() != before {
                // Native reports an immediate move as it makes it.
                for kind in [
                    MLN_RUNTIME_EVENT_MAP_CAMERA_WILL_CHANGE,
                    MLN_RUNTIME_EVENT_MAP_CAMERA_DID_CHANGE,
                ] {
                    let payload = Payload::CameraChange(entry.change.0, entry.change.1);
                    entry.camera_events.push((kind, payload, String::new()));
                }
                entry.jumped = true;
            }
            Ok(())
        }))
    }
);

/// The camera's limits for `b`.
fn constraints(b: &mln_camera_bounds) -> orb_api::Constraints {
    orb_api::Constraints {
        min_zoom: b.min_zoom,
        max_zoom: b.max_zoom,
        min_pitch: b.min_pitch,
        max_pitch: b.max_pitch,
        bounds: (!b.unbounded).then(|| bounds(b.bounds)),
    }
}

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
    if !ok(&b.southwest) || !ok(&b.northeast) {
        return Err(Failure::argument(
            "the bounds' corners must be finite coordinates within [-90, 90]",
        ));
    }
    if b.southwest.latitude > b.northeast.latitude {
        return Err(Failure::argument(
            "the bounds' southwest corner is north of their northeast corner",
        ));
    }
    Ok(())
}

/// Checks fit options as mln-ffi does.
fn check_fit(o: &mln_camera_fit_options) -> Result {
    let known =
        MLN_CAMERA_FIT_OPTION_PADDING | MLN_CAMERA_FIT_OPTION_BEARING | MLN_CAMERA_FIT_OPTION_PITCH;
    if o.fields & !known != 0 {
        return Err(Failure::argument("the fit options have unknown bits"));
    }
    if o.fields & MLN_CAMERA_FIT_OPTION_PADDING != 0 {
        check_insets(&o.padding)?;
    }
    if (o.fields & MLN_CAMERA_FIT_OPTION_BEARING != 0 && !o.bearing.is_finite())
        || (o.fields & MLN_CAMERA_FIT_OPTION_PITCH != 0 && !o.pitch.is_finite())
    {
        return Err(Failure::argument(
            "the fit's bearing and pitch must be finite",
        ));
    }
    Ok(())
}

/// The box around a GeoJSON geometry's coordinates.
fn geometry_box(geometry: mln_buffer_view) -> Result<mln_lat_lng_bounds> {
    if geometry.data.is_null() {
        return Err(Failure::argument("the geometry is null"));
    }
    // SAFETY: the caller's view.
    let text = unsafe { text(geometry, "the geometry")? };
    let json = orb_expr::json::parse(&text)
        .map_err(|e| Failure::argument(format!("the geometry is not valid JSON: {e}")))?;
    fn walk(json: &orb_expr::json::Json, out: &mut Vec<mln_lat_lng>) -> Result {
        let Some(items) = json.as_array() else {
            return Err(Failure::argument(
                "the geometry's coordinates aren't arrays",
            ));
        };
        if let [lon, lat, ..] = items
            && let (Some(lon), Some(lat)) = (lon.as_f64(), lat.as_f64())
        {
            out.push(mln_lat_lng {
                latitude: lat,
                longitude: lon,
            });
            return Ok(());
        }
        items.iter().try_for_each(|item| walk(item, out))
    }
    fn coordinates(json: &orb_expr::json::Json, out: &mut Vec<mln_lat_lng>) -> Result {
        let kind = json
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| Failure::argument("the geometry has no type"))?;
        match kind {
            "GeometryCollection" => json
                .get("geometries")
                .and_then(|g| g.as_array())
                .ok_or_else(|| Failure::argument("the collection has no geometries"))?
                .iter()
                .try_for_each(|g| coordinates(g, out)),
            "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon"
            | "MultiPolygon" => walk(
                json.get("coordinates")
                    .ok_or_else(|| Failure::argument("the geometry has no coordinates"))?,
                out,
            ),
            other => Err(Failure::argument(format!(
                "\"{other}\" isn't a GeoJSON geometry type"
            ))),
        }
    }
    let mut points = Vec::new();
    coordinates(&json, &mut points)?;
    if points.is_empty() {
        return Err(Failure::argument(
            "the geometry needs at least one coordinate",
        ));
    }
    // SAFETY: `points` holds `len` coordinates.
    unsafe { box_of(points.as_ptr(), points.len()) }
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
    let o = unsafe { sized(options, "fit_options")? };
    check_fit(o)?;
    Ok(fit_options(o))
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
    let v = entry.map.viewport();
    let p = fit.padding;
    if p.left + p.right >= v.width || p.top + p.bottom >= v.height {
        return Err(Failure::argument(
            "the padding leaves no room for the bounds in the map",
        ));
    }
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
            | MLN_CAMERA_OPTION_PITCH
            | MLN_CAMERA_OPTION_PADDING,
        ..Default::default()
    };
    options.center = mln_lat_lng {
        latitude: center.lat,
        longitude: center.lon,
    };
    // The fitted camera keeps the padding it was fitted within.
    options.padding = mln_edge_insets {
        top: p.top,
        left: p.left,
        bottom: p.bottom,
        right: p.right,
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
        return Err(Failure::argument("the update's mode is invalid"));
    }
    // The update versions the fit it holds: the fit's own size isn't read.
    check_fit(&u.fit)?;
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
        // A fit that fails leaves the camera, and ends its handler.
        let camera = match camera_for(entry, b, &fit) {
            Ok(camera) => camera,
            Err(failure) => {
                if let Some(handler) = animation.handler {
                    entry
                        .ended
                        .push((handler, MLN_CAMERA_TRANSITION_OUTCOME_CANCELLED));
                }
                return Err(failure);
            }
        };
        let t = target(entry, &camera)?;
        update_camera(entry, mode, &t, &animation, MLN_GESTURE_PHASE_NONE)
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
    mln_map_camera_for_geometry(geometry: mln_buffer_view, fit: *const mln_camera_fit_options),
    true,
    false,
    |state, id| {
        let b = geometry_box(geometry)?;
        let fit = fit_from(fit)?;
        Ok(move |state: &mut State, id: u64| camera_for(&state.maps[&id], b, &fit).map(Value::of))
    }
);

command!(
    mln_map_fit_camera_to_geometry(geometry: mln_buffer_view, update: *const mln_camera_fit_update),
    false,
    true,
    |state, id| {
        let b = geometry_box(geometry)?;
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
            if logical_size.width == 0 || logical_size.width != logical_size.height {
                return Err(Failure::argument("a tile's map is square and not empty"));
            }
            // A vector tile is 512 pixels wide at its zoom: a wider map
            // shows several.
            if logical_size.width > 512 {
                return Err(Failure::argument("a tile's map is at most 512 pixels wide"));
            }
            if tile.z > 30 || u64::from(tile.x) >= 1 << tile.z || u64::from(tile.y) >= 1 << tile.z {
                return Err(Failure::argument("no such tile"));
            }
            // Just past the tile's zoom, so the map covers it with that
            // tile and no other, as Native's camera does.
            let zoom = f64::from(tile.z) + (f64::from(logical_size.width) / 512.0).log2() + 1e-4;
            if !(0.0..=25.5).contains(&zoom) {
                return Err(Failure::argument(
                    "the tile's zoom on that map is outside 0 to 25.5",
                ));
            }
            let out = sized(out_camera, "out_camera")?;
            let n = f64::from(1u32 << tile.z);
            let lon = (f64::from(tile.x) + 0.5) / n * 360.0 - 180.0;
            let y = std::f64::consts::PI * (1.0 - 2.0 * (f64::from(tile.y) + 0.5) / n);
            let lat = y.sinh().atan().to_degrees();
            // Every field that moves the view, so the camera replaces a
            // map's earlier one.
            let mut camera = mln_camera_options {
                size: out.size,
                fields: MLN_CAMERA_OPTION_CENTER
                    | MLN_CAMERA_OPTION_ZOOM
                    | MLN_CAMERA_OPTION_BEARING
                    | MLN_CAMERA_OPTION_PITCH
                    | MLN_CAMERA_OPTION_ROLL
                    | MLN_CAMERA_OPTION_PADDING
                    | MLN_CAMERA_OPTION_CENTER_ALTITUDE,
                ..Default::default()
            };
            camera.center = mln_lat_lng {
                latitude: lat,
                longitude: lon,
            };
            camera.zoom = zoom;
            out_camera.write(camera);
            Ok(())
        })
    }
}
