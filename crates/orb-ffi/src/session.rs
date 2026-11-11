// Render sessions: an Orbis target (a CAMetalLayer surface, an RGBA8
// texture to read back, or a ring of the host's textures) that a map draws
// into on its runtime's thread, the frame demands that drive it, and the
// frame results it reports.
//
// Orbis's runtime owns its GPU device, so a session renders on that device:
// an attach that names another device (an MTLDevice, an offscreen device of
// another adapter) isn't supported, and a ring's textures must be of it.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::base::{Failure, Result, UserData, boundary, put, sized};
use crate::delivery::{Completion, Value, Wake};
use crate::handles::{self, Object};
use crate::runtime::{Msg, Payload, RuntimeHandle, State};
use crate::sys::*;

/// What the caller's threads share with the runtime thread.
pub(crate) struct SessionHandle {
    pub(crate) runtime: Arc<RuntimeHandle>,
    id: OnceLock<u64>,
    frame_wake: Option<Wake>,
    results: Mutex<Vec<mln_render_frame_result>>,
    /// Whether the frame wake was called since the last drain.
    notified: AtomicBool,
    /// A pacer's refresh not yet seen by the runtime thread.
    pub(crate) tick: AtomicBool,
    /// The host draws at a redraw wake, so takes no pacer.
    redraw_wake: bool,
    driver: mln_render_driver_kind,
    readback: bool,
    /// The host's textures, when it renders into a borrowed ring.
    pub(crate) ring: Mutex<crate::frames::Ring>,
    /// Views of its acquired frames that are open.
    pub(crate) open_views: AtomicUsize,
    /// Fixed once it fails.
    failure: Mutex<Option<Failure>>,
    state: Mutex<mln_render_session_snapshot>,
}

impl SessionHandle {
    fn id(&self) -> u64 {
        *self.id.get().expect("session id")
    }

    pub(crate) fn session_id(&self) -> u64 {
        self.id()
    }

    /// The textures of its ring, or 0 when it renders into none.
    pub(crate) fn ring_depth(&self) -> u32 {
        self.ring.lock().unwrap().depth()
    }

    fn push(&self, result: mln_render_frame_result) {
        self.results.lock().unwrap().push(result);
        if !self.notified.swap(true, Ordering::AcqRel)
            && let Some(wake) = &self.frame_wake
        {
            wake.signal();
        }
    }

    fn failed(&self) -> Result {
        match &*self.failure.lock().unwrap() {
            Some(f) => Err(f.clone()),
            None => Ok(()),
        }
    }

    /// The session's lifecycle state.
    pub(crate) fn lifecycle(&self) -> mln_render_session_state {
        self.state.lock().unwrap().state
    }

    /// Whether the session still renders: not detached, abandoned or
    /// failed.
    pub(crate) fn usable(&self) -> Result {
        if matches!(
            self.lifecycle(),
            MLN_RENDER_SESSION_STATE_DETACHING
                | MLN_RENDER_SESSION_STATE_DETACHED
                | MLN_RENDER_SESSION_STATE_ABANDONED
        ) {
            return Err(Failure::state(
                "the session was detached, abandoned, released or disposed",
            ));
        }
        self.failed()
    }

    /// Refuses a call that needs the host's frames back first.
    fn no_frames_held(&self) -> Result {
        if self.state.lock().unwrap().acquired_frame_count > 0 {
            return Err(Failure::state(
                "the host holds acquired frames of the session: release them first",
            ));
        }
        Ok(())
    }

    /// Changes the session's published state.
    pub(crate) fn state_mut(&self, f: impl FnOnce(&mut mln_render_session_snapshot)) {
        f(&mut self.state.lock().unwrap());
    }

    /// Changes the count of frames the host holds.
    pub(crate) fn count_acquired(&self, delta: i32) {
        let mut s = self.state.lock().unwrap();
        s.acquired_frame_count = s.acquired_frame_count.saturating_add_signed(delta);
    }

    /// Wakes the runtime thread to render demands waiting for a slot.
    pub(crate) fn wake_runtime(&self) {
        let _ = self.runtime.sender().send(Msg::Pump);
    }

    /// Posts a change to the session's state: its generation grows once
    /// the change is made.
    fn command(
        &self,
        completion: Option<Completion>,
        job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
    ) -> Result {
        self.post(completion, true, job)
    }

    /// Posts work that leaves the session's state as it is.
    fn operation(
        &self,
        completion: Option<Completion>,
        job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
    ) -> Result {
        self.post(completion, false, job)
    }

    fn post(
        &self,
        completion: Option<Completion>,
        changes: bool,
        job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
    ) -> Result {
        let id = self.id();
        self.runtime.post(move |state| {
            let result = if state.sessions.contains_key(&id) {
                // The demands sent before the call render first.
                pump_one(state, id);
                job(state, id)
            } else {
                Err(Failure::new(
                    MLN_STATUS_TARGET_LOST,
                    "the session lost its target before the work ran",
                ))
            };
            if changes
                && result.is_ok()
                && let Some(entry) = state.sessions.get_mut(&id)
            {
                entry.generation += 1;
            }
            state.complete(completion, result, 0);
        })
    }
}

pub(crate) struct SessionEntry {
    handle: Arc<SessionHandle>,
    map: u64,
    /// None once the target is released.
    target: Option<Target>,
    logical: mln_logical_size,
    scale: f64,
    demands: VecDeque<mln_frame_demand>,
    redraw_wake: Option<Wake>,
    redraw_signaled: bool,
    pacer: Option<Arc<crate::pacer::FramePacer>>,
    frame_generation: u64,
    /// The render update of the map that the session drew last, and what
    /// that frame showed.
    drawn_update: u64,
    shown: Option<Shown>,
    /// Draw calls of every frame the session drew.
    draws: i64,
    /// Grows with each change to the session's state.
    generation: u64,
    /// The last still image, which a readback returns.
    still: Option<orb_api::Image>,
}

impl Drop for SessionEntry {
    fn drop(&mut self) {
        if let Some(pacer) = self.pacer.take() {
            pacer.remove(&self.handle);
        }
        if let Some(wake) = self.redraw_wake.take() {
            wake.release();
        }
    }
}

impl SessionEntry {
    pub(crate) fn publish(&self) {
        self.sync_state();
    }

    fn sync_state(&self) {
        let mut s = self.handle.state.lock().unwrap();
        s.generation = self.generation;
        s.extent = mln_logical_extent {
            width: self.logical.width,
            height: self.logical.height,
            scale_factor: self.scale,
        };
        s.frame_generation = self.frame_generation;
        s.has_target = self.target.is_some();
    }
}

fn physical(logical: mln_logical_size, scale: f64) -> Result<(u32, u32)> {
    crate::base::physical(mln_logical_extent {
        width: logical.width,
        height: logical.height,
        scale_factor: scale,
    })
}

/// A retain of the caller's layer, which an accepted attach takes so the
/// layer outlives the surface on it.
struct Retained(
    #[cfg_attr(
        not(target_vendor = "apple"),
        expect(dead_code, reason = "only Apple layers take a retain")
    )]
    UserData,
);

#[cfg(target_vendor = "apple")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRetain(object: *const std::ffi::c_void) -> *const std::ffi::c_void;
    fn CFRelease(object: *const std::ffi::c_void);
}

impl Retained {
    fn new(object: UserData) -> Self {
        // SAFETY: a live Objective-C object, by the C API's contract.
        #[cfg(target_vendor = "apple")]
        unsafe {
            CFRetain(object.get());
        }
        Self(object)
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        // SAFETY: the retain that `new` took.
        #[cfg(target_vendor = "apple")]
        unsafe {
            CFRelease(self.0.get());
        }
    }
}

/// A session's target.
enum Target {
    /// A surface, with the layer it draws into released after it, or an
    /// offscreen texture.
    One {
        target: orb_api::Target,
        _layer: Option<Retained>,
    },
    /// The host's textures, one per slot of the session's ring.
    Ring(Vec<orb_api::Target>),
}

/// A CAMetalLayer as an Orbis window target.
fn metal_target(
    state: &State,
    layer: UserData,
    logical: mln_logical_size,
    scale: f64,
) -> Result<Target> {
    let (width, height) = physical(logical, scale)?;
    let layer = NonNull::new(layer.get()).ok_or_else(|| Failure::argument("the layer is null"))?;
    let surface = state.runtime.with_device(|gpu| {
        // SAFETY: the C API's contract: the layer is a live CAMetalLayer that
        // the caller keeps until the session's target is released.
        unsafe { gpu.create_surface_raw(orb_api::RawSurface::MetalLayer(layer), width, height) }
            .map_err(Failure::native)
    })?;
    Ok(Target::One {
        target: orb_api::Target::surface(&state.runtime, surface, scale),
        _layer: Some(Retained::new(UserData(layer.as_ptr()))),
    })
}

/// Refuses an MTLDevice other than the runtime's: Orbis renders on its
/// own device. A null one is the runtime's.
fn same_device(state: &State, device: UserData) -> Result {
    if device.get().is_null() {
        return Ok(());
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if state
        .runtime
        .with_device(|gpu| gpu.metal_device())
        .map(|d| d.as_ptr())
        == Some(device.get())
    {
        return Ok(());
    }
    let _ = state;
    Err(Failure::unsupported(
        "Orbis renders on its runtime's own MTLDevice, not the one the descriptor names",
    ))
}

/// The host's textures as targets of the runtime's device.
fn ring_target(state: &State, slots: &[crate::frames::SlotInfo]) -> Result<Target> {
    let targets = slots
        .iter()
        .map(|slot| crate::frames::import(&state.runtime, slot))
        .collect::<Result<_>>()?;
    Ok(Target::Ring(targets))
}

pub(crate) enum TargetSpec {
    Offscreen,
    /// A CAMetalLayer, and the MTLDevice the host names for it, if any.
    Metal(UserData, UserData),
    Ring(Vec<crate::frames::SlotInfo>),
}

/// The attach of every target kind.
///
/// # Safety
/// The caller's pointers.
pub(crate) unsafe fn attach(
    map: mln_map,
    logical: mln_logical_size,
    spec: TargetSpec,
    options: *const mln_render_session_attach_options,
    out_session: *mut mln_render_session,
    completion: *const mln_completion,
) -> Result {
    let map_handle = handles::map(map.0)?;
    // SAFETY: the caller's pointers.
    let options = unsafe { sized(options, "options")? };
    if !matches!(
        options.driver,
        MLN_RENDER_DRIVER_CORE_WORKER | MLN_RENDER_DRIVER_CALLER_GRAPHICS_THREAD
    ) {
        return Err(Failure::argument(
            "options.driver names no render driver kind",
        ));
    }
    if options.driver != MLN_RENDER_DRIVER_CORE_WORKER {
        return Err(Failure::unsupported(
            "Orbis sessions render on the runtime's thread (MLN_RENDER_DRIVER_CORE_WORKER)",
        ));
    }
    let lock = &options.queue_lock;
    if lock.lock.is_some() != lock.unlock.is_some() {
        return Err(Failure::argument(
            "options.queue_lock sets one of lock and unlock: set both or neither",
        ));
    }
    if lock.lock.is_none() && (lock.release_user_data.is_some() || !lock.user_data.is_null()) {
        return Err(Failure::argument(
            "options.queue_lock names user data but no lock and unlock",
        ));
    }
    if lock.lock.is_some() {
        return Err(Failure::unsupported(
            "Orbis sessions submit to their runtime's own queue, which no caller shares",
        ));
    }
    // A target's pixels follow from its logical size at the map's scale.
    let scale = (map_handle.snapshot()).map_or(1.0, |s| s.view.extent.scale_factor);
    physical(logical, scale)?;
    let known_sync = (MLN_GPU_SYNC_KIND_FLAG_D3D12_FENCE << 1) - 1;
    if options.producer_sync_kinds & !known_sync != 0 {
        return Err(Failure::argument(
            "options.producer_sync_kinds sets a bit that names no gpu sync kind",
        ));
    }
    if options.producer_sync_kinds != 0
        && options.producer_sync_kinds & MLN_GPU_SYNC_KIND_FLAG_CPU_COMPLETE == 0
    {
        return Err(Failure::unsupported(
            "Orbis sessions have no GPU sync objects to share",
        ));
    }
    // SAFETY: as above.
    unsafe { crate::base::fresh_out(out_session, "out_session")? };
    let frame_wake = Wake::from_c(&options.frame_wake, "options.frame_wake")?;
    let redraw_wake = Wake::from_c(&options.redraw_wake, "options.redraw_wake")?;
    Wake::from_c(&options.driver_work_wake, "options.driver_work_wake")?;
    // SAFETY: as above.
    let completion = unsafe { Completion::required(completion)? };
    if map_handle.snapshot().is_none() {
        return Err(Failure::state("the map isn't made yet"));
    }
    if map_handle
        .has_session
        .swap(true, std::sync::atomic::Ordering::AcqRel)
    {
        return Err(Failure::state("map already has an attached render session"));
    }
    let readback = matches!(spec, TargetSpec::Offscreen);
    let ring = match &spec {
        TargetSpec::Ring(slots) => crate::frames::Ring::new(slots.clone()),
        _ => crate::frames::Ring::default(),
    };
    let handle = Arc::new(SessionHandle {
        runtime: map_handle.runtime.clone(),
        id: OnceLock::new(),
        frame_wake,
        results: Mutex::default(),
        notified: AtomicBool::new(false),
        tick: AtomicBool::new(false),
        redraw_wake: redraw_wake.is_some(),
        driver: options.driver,
        readback,
        ring: Mutex::new(ring),
        open_views: AtomicUsize::new(0),
        failure: Mutex::new(None),
        state: Mutex::new(mln_render_session_snapshot {
            size: std::mem::size_of::<mln_render_session_snapshot>() as u32,
            state: MLN_RENDER_SESSION_STATE_ATTACHING,
            ..Default::default()
        }),
    });
    let id = handles::insert(Object::Session(handle.clone()));
    let _ = handle.id.set(id);
    // A disposed runtime stops once its maps and sessions are gone.
    handle.runtime.maps.fetch_add(1, Ordering::AcqRel);
    let map_id = map.0;
    let map_flag = map_handle.clone();
    let map_flag = MapFlag(map_flag);
    map_handle.runtime.post(move |state| {
        let result = (|| {
            let entry = state
                .maps
                .get(&map_id)
                .ok_or_else(|| Failure::state("the map was released"))?;
            let scale = entry.extent.scale_factor;
            let target = match spec {
                TargetSpec::Offscreen => {
                    let (w, h) = physical(logical, scale)?;
                    Target::One {
                        target: orb_api::Target::texture(&state.runtime, w, h, scale),
                        _layer: None,
                    }
                }
                TargetSpec::Metal(layer, device) => {
                    same_device(state, device)?;
                    metal_target(state, layer, logical, scale)?
                }
                TargetSpec::Ring(slots) => ring_target(state, &slots)?,
            };
            Ok((target, scale))
        })();
        match result {
            Ok((target, scale)) => {
                let map = state.maps.get_mut(&map_id).expect("map");
                map.session = Some(id);
                map.extent.width = logical.width;
                map.extent.height = logical.height;
                let entry = SessionEntry {
                    handle: handle.clone(),
                    map: map_id,
                    target: Some(target),
                    logical,
                    scale,
                    demands: VecDeque::new(),
                    redraw_wake,
                    redraw_signaled: false,
                    pacer: None,
                    frame_generation: 0,
                    drawn_update: 0,
                    shown: None,
                    draws: 0,
                    generation: 1,
                    still: None,
                };
                handle.state.lock().unwrap().state = MLN_RENDER_SESSION_STATE_ATTACHED;
                entry.sync_state();
                state.sessions.insert(id, entry);
                state.complete(Some(completion), Ok(Value::None), 0);
            }
            Err(failure) => {
                handles::remove(id);
                handle.runtime.child_released();
                map_flag.store(false, std::sync::atomic::Ordering::Release);
                state.complete(Some(completion), Err(failure), 0);
            }
        }
    })?;
    // SAFETY: as above.
    unsafe { put(out_session, mln_render_session(id), "out_session") }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_attach_offscreen(
    map: mln_map,
    descriptor: *const mln_offscreen_descriptor,
    options: *const mln_render_session_attach_options,
    device: mln_offscreen_device,
    out_session: *mut mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if device.0 != 0 {
                handles::offscreen_device(device.0)?;
            }
            let d = sized(descriptor, "descriptor")?;
            attach(
                map,
                d.logical_size,
                TargetSpec::Offscreen,
                options,
                out_session,
                completion,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_attach_metal_surface(
    map: mln_map,
    descriptor: *const mln_metal_surface_descriptor,
    options: *const mln_render_session_attach_options,
    out_session: *mut mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if !cfg!(target_vendor = "apple") {
                return Err(Failure::unsupported("Metal is only on Apple platforms"));
            }
            let d = sized(descriptor, "descriptor")?;
            if d.layer.is_null() {
                return Err(Failure::argument("descriptor.layer is null"));
            }
            attach(
                map,
                d.logical_size,
                TargetSpec::Metal(UserData(d.layer), UserData(d.context.device)),
                options,
                out_session,
                completion,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_set_metal_surface_target(
    session: mln_render_session,
    replacement: *const mln_metal_surface_replacement,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let r = sized(replacement, "replacement")?;
            if r.layer.is_null() {
                return Err(Failure::argument("replacement.layer is null"));
            }
            physical(r.logical_size, 1.0)?;
            let handle = handles::session(session.0)?;
            handle.usable()?;
            handle.no_frames_held()?;
            if handle.ring_depth() > 0 {
                return Err(Failure::unsupported(
                    "a borrowed ring takes caller-owned textures, not a native surface",
                ));
            }
            if handle.readback {
                return Err(Failure::unsupported(
                    "an offscreen session renders into a texture of its own and takes no target",
                ));
            }
            let (logical, layer) = (r.logical_size, UserData(r.layer));
            handle.command(Completion::from_c(completion)?, move |state, id| {
                let scale = state.sessions[&id].scale;
                // The old surface goes before the new one is made on its layer.
                state.sessions.get_mut(&id).expect("session").target = None;
                let target = metal_target(state, layer, logical, scale)?;
                let entry = state.sessions.get_mut(&id).expect("session");
                entry.target = Some(target);
                entry.logical = logical;
                entry.sync_state();
                let map = entry.map;
                if let Some(m) = state.maps.get_mut(&map) {
                    (m.extent.width, m.extent.height) = (logical.width, logical.height);
                }
                force_redraw(state, id);
                Ok(Value::None)
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_release_target(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            if handle.readback {
                return Err(Failure::unsupported(
                    "an offscreen session keeps its texture: detach it instead",
                ));
            }
            handle.no_frames_held()?;
            handle.command(Completion::from_c(completion)?, |state, id| {
                let entry = state.sessions.get_mut(&id).expect("session");
                entry.target = None;
                entry.handle.ring.lock().unwrap().release_target();
                entry.sync_state();
                Ok(Value::None)
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_resize(
    session: mln_render_session,
    logical_size: mln_logical_size,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            physical(logical_size, 1.0)?;
            if handle.ring_depth() > 0 {
                return Err(Failure::unsupported(
                    "a borrowed ring's textures are sized by its owner: give it new textures \
                     of the new size instead",
                ));
            }
            handle.no_frames_held()?;
            handle.command(Completion::from_c(completion)?, move |state, id| {
                let entry = state.sessions.get_mut(&id).expect("session");
                let (w, h) = physical(logical_size, entry.scale)?;
                if let Some(Target::One { target, .. }) = &mut entry.target {
                    target.resize(w, h, entry.scale);
                }
                entry.logical = logical_size;
                entry.sync_state();
                let map = entry.map;
                if let Some(m) = state.maps.get_mut(&map) {
                    (m.extent.width, m.extent.height) = (logical_size.width, logical_size.height);
                }
                force_redraw(state, id);
                Ok(Value::None)
            })
        })
    }
}

/// [`SessionHandle::command`], for other modules.
pub(crate) fn command(
    handle: &SessionHandle,
    completion: Option<Completion>,
    job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
) -> Result {
    handle.command(completion, job)
}

/// Whether the session renders into a texture of its own.
pub(crate) fn renders_offscreen(handle: &SessionHandle) -> bool {
    handle.readback
}

/// [`SessionHandle::no_frames_held`], for other modules.
pub(crate) fn no_frames_held(handle: &SessionHandle) -> Result {
    handle.no_frames_held()
}

/// Gives a ring session new textures, whose slots are all free.
pub(crate) fn retarget_ring(
    state: &mut State,
    id: u64,
    logical: mln_logical_size,
    slots: Vec<crate::frames::SlotInfo>,
) -> Result<Value> {
    let entry = &state.sessions[&id];
    let (same, format) = {
        let ring = entry.handle.ring.lock().unwrap();
        (
            entry.target.is_some()
                && (entry.logical.width, entry.logical.height) == (logical.width, logical.height)
                && ring.holds(&slots),
            ring.pixel_format(),
        )
    };
    // The textures it has already: nothing changes, nothing to draw.
    if same {
        return Ok(Value::None);
    }
    if format.is_some_and(|f| Some(f) != slots.first().map(crate::frames::SlotInfo::pixel_format)) {
        return Err(Failure::unsupported(
            "the replacement's textures have another pixel format than the ring's",
        ));
    }
    let target = ring_target(state, &slots)?;
    let entry = state.sessions.get_mut(&id).expect("session");
    entry.target = Some(target);
    entry.handle.ring.lock().unwrap().replace(slots);
    entry.logical = logical;
    entry.sync_state();
    let map = entry.map;
    if let Some(m) = state.maps.get_mut(&map) {
        (m.extent.width, m.extent.height) = (logical.width, logical.height);
    }
    force_redraw(state, id);
    Ok(Value::None)
}

/// Marks the session's next demand as having something to draw.
pub(crate) fn force_redraw(state: &mut State, session: u64) {
    if let Some(entry) = state.sessions.get_mut(&session)
        && let Some(map) = state.maps.get_mut(&entry.map)
    {
        // A zero-length move dirties the map without moving its camera.
        map.map.jump_to(&orb_api::CameraTarget::default());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_demand_frame(
    session: mln_render_session,
    demand: *const mln_frame_demand,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            let demand = *sized(demand, "demand")?;
            if demand.flags & MLN_FRAME_DEMAND_FORCE != 0
                && demand.flags & MLN_FRAME_DEMAND_WAIT_FOR_UPDATE != 0
            {
                return Err(Failure::argument(
                    "MLN_FRAME_DEMAND_FORCE and MLN_FRAME_DEMAND_WAIT_FOR_UPDATE don't combine",
                ));
            }
            if demand.flags & !(MLN_FRAME_DEMAND_FORCE | MLN_FRAME_DEMAND_WAIT_FOR_UPDATE) != 0 {
                return Err(Failure::argument("demand.flags sets an unknown flag"));
            }
            if demand.token == MLN_FRAME_DEMAND_TOKEN_STILL_IMAGE {
                return Err(Failure::argument(
                    "the still image token is reserved for the library's still images",
                ));
            }
            handle.ring.lock().unwrap().usable()?;
            let id = handle.id();
            handle.runtime.post(move |state| {
                if let Some(entry) = state.sessions.get_mut(&id) {
                    queue_demand(entry, demand);
                }
            })
        })
    }
}

fn queue_demand(entry: &mut SessionEntry, demand: mln_frame_demand) {
    entry.redraw_signaled = false;
    if demand.flags & MLN_FRAME_DEMAND_FORCE == 0
        && let Some(i) = entry.demands.iter().position(|d| d.flags == demand.flags)
    {
        let old = std::mem::replace(&mut entry.demands[i], demand);
        entry.handle.push(result(
            old.token,
            MLN_RENDER_RESULT_SUPERSEDED,
            entry.frame_generation,
            None,
        ));
        return;
    }
    entry.demands.push_back(demand);
}

/// What a rendered frame shows.
#[derive(Clone, Copy)]
struct Shown {
    view: mln_map_view,
    /// The map's generation, and the render update the frame draws.
    generation: u64,
    update: u64,
    needs_repaint: bool,
    /// Everything the view needs had loaded.
    full: bool,
}

fn result(
    token: u64,
    disposition: mln_render_result,
    frame_generation: u64,
    shown: Option<Shown>,
) -> mln_render_frame_result {
    let mut r = mln_render_frame_result {
        size: std::mem::size_of::<mln_render_frame_result>() as u32,
        disposition,
        token,
        frame_generation,
        mode: MLN_RENDER_MODE_FULL,
        ..Default::default()
    };
    if let Some(shown) = shown {
        r.view = shown.view;
        r.generation = shown.generation;
        r.map_update_generation = shown.update;
        r.needs_repaint = shown.needs_repaint;
        if !shown.full {
            r.mode = MLN_RENDER_MODE_PARTIAL;
        }
    }
    r
}

/// The map a live session draws.
pub(crate) fn map_of(state: &State, session: u64) -> Option<u64> {
    state
        .sessions
        .get(&session)
        .map(|s| s.map)
        .filter(|m| state.maps.contains_key(m))
}

/// How long a frame waits for the pipelines it needs.
const PIPELINE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Draws a frame of everything the map holds. Orbis compiles pipelines
/// off the frame and draws what has none yet in a later frame; a frame of
/// the C API shows the map's state, as Native's synchronous compiles do,
/// so one that left something out for a compile is drawn again once the
/// compile is done.
fn render_whole(
    map: &mut orb_api::Map,
    target: &mut orb_api::Target,
    now: f64,
) -> std::result::Result<Option<orb_api::FrameReport>, orb_api::Error> {
    let deadline = std::time::Instant::now() + PIPELINE_WAIT;
    loop {
        let report = map.render(target, now)?;
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match &report {
            // A compile wakes the map when it's done. A frame that waited
            // with none pending (a compile's arrival) draws it in the next.
            Some(r) if r.frame.waited && !left.is_zero() => {
                if r.frame.pipelines_pending > 0 {
                    map.wait_woken(left.min(std::time::Duration::from_millis(16)));
                }
            }
            _ => return Ok(report),
        }
    }
}

/// A frame's `needs_repaint`, as MapLibre Native reports it: the map's
/// loads, fades and transitions go on. A moving camera places labels
/// again, whose placement fades over the style's transition while placement
/// transitions are on, so it counts too, unless they're instant.
fn needs_repaint(map: &crate::map::MapEntry) -> bool {
    map.map.needs_repaint()
        || (map.map.is_moving()
            && map.style.placement_transitions != Some(false)
            && map
                .map
                .style()
                .is_none_or(|doc| doc.style().transition.duration > 0.0))
}

/// Draws the frames that the sessions' demands and pacers ask for, and
/// reports them; then takes the maps' events.
pub(crate) fn pump(state: &mut State) {
    let ids: Vec<u64> = state.sessions.keys().copied().collect();
    for id in ids {
        pump_one(state, id);
    }
}

/// A frame a session drew: its map, and what the frame event reports.
type Drawn = (u64, bool, bool, mln_rendering_stats);

/// [`pump`] for one session.
pub(crate) fn pump_one(state: &mut State, id: u64) {
    let now = state.now();
    let State { sessions, maps, .. } = state;
    let Some(entry) = sessions.get_mut(&id) else {
        return;
    };
    let Some(map) = maps.get_mut(&entry.map) else {
        return;
    };
    let (drawn, still) = step(entry, map, now);
    for (map, needs_repaint, full, stats) in drawn {
        state.event(
            map,
            MLN_RUNTIME_EVENT_MAP_RENDER_FRAME_FINISHED,
            Payload::RenderFrame {
                needs_repaint,
                full,
                stats,
            },
            String::new(),
        );
    }
    if let Some((completion, result, generation)) = still {
        state.complete(Some(completion), result, generation);
    }
}

/// Runs a session's demands: draws the frames they ask for and reports
/// each. Returns the frames it drew, and the still image it finished.
#[allow(clippy::type_complexity)]
fn step(
    entry: &mut SessionEntry,
    map: &mut crate::map::MapEntry,
    now: f64,
) -> (Vec<Drawn>, Option<(Completion, Result<Value>, u64)>) {
    let mut drawn = Vec::new();
    let mut still = None;
    // An open command group holds the map's changes back from every
    // frame, forced ones too, until the outermost group ends.
    let held = map.groups > 0;
    if !held && entry.target.is_some() && entry.handle.failed().is_ok() {
        still = step_still(map, entry);
    }
    // A continuous map with something new to draw that its session
    // hasn't drawn yet (data arrived, an animation's next step)
    // publishes an update for it, once until the session draws it.
    let continuous = map.mode == MLN_MAP_MODE_CONTINUOUS;
    if continuous
        && !held
        && map.map.has_news()
        && map.render_update_generation <= entry.drawn_update
    {
        map.render_update_generation = entry.drawn_update + 1;
    }
    if entry.handle.tick.swap(false, Ordering::AcqRel)
        && entry.demands.is_empty()
        && !held
        && map.map.has_news()
    {
        entry.demands.push_back(mln_frame_demand {
            size: 0,
            flags: 0,
            token: 0,
        });
    }
    // Demands that wait for the map's next update, set aside so the
    // ones behind them run: a static or tile map's next update comes
    // only with its next still image, and a barrier ends the wait.
    let mut waiting = VecDeque::new();
    while let Some(demand) = entry.demands.pop_front() {
        if entry.handle.failure.lock().unwrap().is_some() {
            entry.handle.push(result(
                demand.token,
                MLN_RENDER_RESULT_FAILED,
                entry.frame_generation,
                None,
            ));
            continue;
        }
        let Some(target) = &mut entry.target else {
            entry.handle.push(result(
                demand.token,
                MLN_RENDER_RESULT_TARGET_NOT_READY,
                entry.frame_generation,
                None,
            ));
            continue;
        };
        // A static or tile map publishes updates only with its still
        // images, so only a forced demand draws it between them.
        let force = demand.flags & MLN_FRAME_DEMAND_FORCE != 0;
        let update = continuous && map.render_update_generation > entry.drawn_update;
        // Inside a command group a forced frame shows the update the
        // session drew last again, which its target still holds.
        if held
            && force
            && let Some(shown) = entry.shown
        {
            entry.frame_generation += 1;
            entry.handle.push(result(
                demand.token,
                MLN_RENDER_RESULT_RENDERED,
                entry.frame_generation,
                Some(shown),
            ));
            continue;
        }
        if held || !(force || update) {
            if demand.flags & MLN_FRAME_DEMAND_WAIT_FOR_UPDATE != 0 {
                waiting.push_back(demand);
                continue;
            }
            entry.handle.push(result(
                demand.token,
                MLN_RENDER_RESULT_NO_UPDATE,
                entry.frame_generation,
                None,
            ));
            continue;
        }
        // A ring draws into a slot the host doesn't hold; with none, the
        // demand waits for the host to give one back.
        let (target, slot) = match target {
            Target::One { target, .. } => (target, None),
            Target::Ring(targets) => match entry.handle.ring.lock().unwrap().begin() {
                crate::frames::Begin::Slot(slot) => (&mut targets[slot], Some(slot)),
                crate::frames::Begin::Wait => {
                    entry.demands.push_front(demand);
                    break;
                }
                crate::frames::Begin::Quarantined => {
                    entry.handle.push(result(
                        demand.token,
                        MLN_RENDER_RESULT_TARGET_NOT_READY,
                        entry.frame_generation,
                        None,
                    ));
                    continue;
                }
            },
        };
        let mut full = true;
        let mut stats = mln_rendering_stats {
            encoding_time_ms: 0.0,
            rendering_time_ms: 0.0,
            frame_count: 0,
            draw_call_count: 0,
            total_draw_call_count: 0,
        };
        let disposition = match render_whole(&mut map.map, target, now) {
            Ok(Some(report)) => {
                entry.frame_generation += 1;
                full = report.loaded;
                let draws = i64::from(report.frame.draws);
                entry.draws += draws;
                let t = &report.times;
                let r = &t.render;
                let encoding = r.cull + r.bundles + r.drape + r.encode + r.submit;
                stats = mln_rendering_stats {
                    encoding_time_ms: f64::from(encoding) / 1000.0,
                    rendering_time_ms: f64::from(t.total) / 1000.0,
                    frame_count: entry.frame_generation as i64,
                    draw_call_count: draws,
                    total_draw_call_count: entry.draws,
                };
                MLN_RENDER_RESULT_RENDERED
            }
            Ok(None) => MLN_RENDER_RESULT_TARGET_NOT_READY,
            Err(e) => {
                let failure = Failure::from(e);
                let status = if failure.status == MLN_STATUS_TARGET_LOST {
                    MLN_STATUS_TARGET_LOST
                } else {
                    MLN_STATUS_NATIVE_ERROR
                };
                {
                    let mut s = entry.handle.state.lock().unwrap();
                    s.state = MLN_RENDER_SESSION_STATE_FAILED;
                    s.failure_status = status;
                }
                *entry.handle.failure.lock().unwrap() = Some(Failure::new(status, failure.message));
                MLN_RENDER_RESULT_FAILED
            }
        };
        let needs_repaint = needs_repaint(map);
        let shown = (disposition == MLN_RENDER_RESULT_RENDERED).then(|| Shown {
            view: map.view(),
            generation: entry.generation,
            update: map.render_update_generation,
            needs_repaint,
            full,
        });
        let frame = result(demand.token, disposition, entry.frame_generation, shown);
        // The ring holds the frame before the host hears of it.
        if let Some(slot) = slot {
            entry
                .handle
                .ring
                .lock()
                .unwrap()
                .end(slot, shown.is_some().then_some(frame));
        }
        entry.handle.push(frame);
        if shown.is_some() {
            // A readback now reads the frame, not an earlier still.
            entry.still = None;
            entry.drawn_update = map.render_update_generation;
            entry.shown = shown;
            drawn.push((entry.map, needs_repaint, full, stats));
        }
    }
    waiting.append(&mut entry.demands);
    entry.demands = waiting;
    entry.sync_state();
    // The host draws at its next refresh when told there's something new.
    if entry.target.is_some()
        && entry.demands.is_empty()
        && !entry.redraw_signaled
        && !held
        && map.map.has_news()
        && let Some(wake) = &entry.redraw_wake
    {
        entry.redraw_signaled = true;
        wake.signal();
    }
    // Upkeep shows nothing new, so it publishes no update: what of it
    // draws nothing runs now, and a resize waits for the next frame.
    if continuous && !held && !map.map.has_news() {
        map.map.upkeep();
    }
    (drawn, still)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_drain_frame_results(
    session: mln_render_session,
    out_batch: *mut mln_render_frame_batch,
    out_view: *mut mln_render_frame_batch_view,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            if out_batch.is_null() || out_view.is_null() {
                return Err(Failure::argument("out_batch or out_view is null"));
            }
            // The view is checked before the results are looked at.
            sized(out_view.cast_const(), "out_view")?;
            let results = {
                let mut results = handle.results.lock().unwrap();
                handle.notified.store(false, Ordering::Release);
                std::mem::take(&mut *results)
            };
            let mut view = mln_render_frame_batch_view {
                size: std::mem::size_of::<mln_render_frame_batch_view>() as u32,
                result_size: std::mem::size_of::<mln_render_frame_result>() as u32,
                results: std::ptr::null(),
                result_count: 0,
            };
            if results.is_empty() {
                out_batch.write(mln_render_frame_batch(0));
            } else {
                let results = Arc::new(results);
                view.results = results.as_ptr();
                view.result_count = results.len();
                out_batch.write(mln_render_frame_batch(handles::insert(Object::Batch(
                    results,
                ))));
            }
            out_view.write(view);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_render_frame_batch_release(batch: mln_render_frame_batch) {
    handles::remove(batch.0);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_snapshot(
    session: mln_render_session,
    out_snapshot: *mut mln_render_session_snapshot,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            put(out_snapshot, *handle.state.lock().unwrap(), "out_snapshot")
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_capabilities(
    session: mln_render_session,
    out_capabilities: *mut mln_render_session_capabilities,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            let depth = handle.ring_depth();
            let flags = if handle.readback {
                MLN_RENDER_SESSION_CAPABILITY_READBACK
            } else if depth > 0 {
                MLN_RENDER_SESSION_CAPABILITY_FRAME_ACQUISITION
            } else {
                MLN_RENDER_SESSION_CAPABILITY_PRESENTATION
            };
            put(
                out_capabilities,
                mln_render_session_capabilities {
                    size: std::mem::size_of::<mln_render_session_capabilities>() as u32,
                    driver: handle.driver,
                    texture_ring_depth: depth.max(1),
                    flags,
                    producer_sync_kind: MLN_GPU_SYNC_CPU_COMPLETE,
                    consumer_sync_kinds: MLN_GPU_SYNC_KIND_FLAG_CPU_COMPLETE,
                },
                "out_capabilities",
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_get_failure(
    session: mln_render_session,
    out_message: *mut std::ffi::c_char,
    message_capacity: usize,
    out_message_length: *mut usize,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            let message = handle
                .failure
                .lock()
                .unwrap()
                .as_ref()
                .map(|f| f.message.clone())
                .unwrap_or_default();
            put(out_message_length, message.len(), "out_message_length")?;
            if !out_message.is_null() && message_capacity > message.len() {
                std::ptr::copy_nonoverlapping(message.as_ptr(), out_message.cast(), message.len());
                *out_message.add(message.len()) = 0;
            }
            Ok(())
        })
    }
}

/// The pixels a readback returns: the last still image, or the texture.
fn readback(state: &mut State, id: u64) -> Result<Value> {
    let entry = state.sessions.get_mut(&id).expect("session");
    if entry.still.is_none() && entry.frame_generation == 0 {
        return Err(Failure::state("the session has no rendered frame to read"));
    }
    let (width, height, pixels) = match &entry.still {
        Some(image) => (image.width, image.height, image.pixels.clone()),
        None => {
            let Some(Target::One { target, .. }) = &entry.target else {
                return Err(Failure::state("the session has no target"));
            };
            let (w, h) = target.size();
            (w, h, target.read_pixels()?)
        }
    };
    let info = mln_texture_image_info {
        width,
        height,
        stride: width * 4,
        byte_length: pixels.len(),
    };
    Ok(Value::Readback { pixels, info })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_read_texture(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            if !handle.readback {
                return Err(Failure::unsupported(
                    "the session lacks MLN_RENDER_SESSION_CAPABILITY_READBACK",
                ));
            }
            handle.command(Completion::from_c(completion)?, readback)
        })
    }
}

/// Posts one of the session's maintenance operations, which run in order
/// with its frames on the runtime's thread.
///
/// # Safety
/// The caller's pointers.
unsafe fn maintain(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
    job: impl FnOnce(&mut State, u64) -> Result<Value> + Send + 'static,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            handle.operation(Completion::from_c(completion)?, move |state, id| {
                state.sessions[&id].handle.failed()?;
                job(state, id)
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_barrier(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        maintain(session, completion, out_diagnostic, |state, id| {
            // The runtime's thread runs work in the order it's posted, so
            // what came before the barrier has run; demands still waiting
            // end.
            let entry = state.sessions.get_mut(&id).expect("session");
            for demand in std::mem::take(&mut entry.demands) {
                entry.handle.push(result(
                    demand.token,
                    MLN_RENDER_RESULT_NO_UPDATE,
                    entry.frame_generation,
                    None,
                ));
            }
            Ok(Value::None)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_reduce_memory_use(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        maintain(session, completion, out_diagnostic, |state, id| {
            if let Some(map) = map_of(state, id) {
                let map = &mut state.maps.get_mut(&map).expect("map").map;
                map.release_cached_tiles();
            }
            Ok(Value::None)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_clear_data(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        maintain(session, completion, out_diagnostic, |state, id| {
            if let Some(map) = map_of(state, id) {
                state.maps.get_mut(&map).expect("map").map.clear_tiles();
            }
            Ok(Value::None)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_dump_debug_logs(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        maintain(session, completion, out_diagnostic, |state, id| {
            let entry = &state.sessions[&id];
            let (width, height) = physical(entry.logical, entry.scale)?;
            let target = if entry.target.is_some() {
                "attached"
            } else {
                "released"
            };
            let message = format!(
                "render session: {width}x{height} px at scale {}, {} frames drawn, \
                 {} demands waiting, target {target}, on {}",
                entry.scale,
                entry.frame_generation,
                entry.demands.len(),
                state.runtime.stats().adapter,
            );
            crate::logging::log(MLN_LOG_SEVERITY_INFO, &message);
            Ok(Value::None)
        })
    }
}

/// The tile a TILE map's camera shows: its zoom fits one tile to the map.
fn tile_of(camera: &mln_camera, logical: mln_logical_size) -> Result<orb_api::TileRequest> {
    let z = (camera.zoom - (f64::from(logical.width) / 512.0).log2()).round();
    if !(0.0..=30.0).contains(&z) {
        return Err(Failure::argument("the camera's zoom names no tile"));
    }
    let n = 2f64.powi(z as i32);
    let x = ((camera.center.longitude + 180.0) / 360.0 * n).floor();
    let lat = camera.center.latitude.to_radians();
    let y = ((1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor();
    Ok(orb_api::TileRequest {
        tile_matrix_set: orb_api::TileMatrixSet::web_mercator().clone(),
        z: z as u8,
        col: x.clamp(0.0, n - 1.0) as u32,
        row: y.clamp(0.0, n - 1.0) as u32,
        size: logical.width,
        scale: 1.0,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_map_request_still_image(
    map: mln_map,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::map(map.0)?;
            // A refused call leaves the completion with the caller.
            let completion = Completion::required(completion)?;
            if !matches!(handle.mode, MLN_MAP_MODE_STATIC | MLN_MAP_MODE_TILE) {
                return Err(Failure::state("the map is not in static or tile mode"));
            }
            let id = map.0;
            handle.runtime.post(move |state| {
                let refused = match state.maps.get_mut(&id) {
                    None => Failure::state("the map was released"),
                    Some(map) if map.still.is_some() => {
                        Failure::state("the map already has a still image pending")
                    }
                    Some(map) if map.groups > 0 => {
                        Failure::state("a still image can't be drawn inside a command group")
                    }
                    // The map's session draws it from its next step on,
                    // with the tiles that failed before requested again.
                    Some(map) => {
                        map.map.retry_failed_tiles();
                        map.still = Some(completion);
                        return;
                    }
                };
                state.complete(Some(completion), Err(refused), 0);
            })
        })
    }
}

/// Draws the next frames of the still image a STATIC or TILE map waits on
/// (Orbis's Still and Tile modes), and once it is done reports it as a
/// frame with the still image token and returns its completion's outcome.
fn step_still(
    map: &mut crate::map::MapEntry,
    entry: &mut SessionEntry,
) -> Option<(Completion, Result<Value>, u64)> {
    map.still.as_ref()?;
    let step = match map.mode {
        MLN_MAP_MODE_STATIC => map
            .map
            .snapshot_step(entry.logical.width, entry.logical.height, entry.scale)
            .map_err(Failure::from),
        _ => tile_of(&map.camera(), entry.logical).and_then(|mut tile| {
            tile.scale = entry.scale;
            map.map.render_tile_step(&tile).map_err(Failure::from)
        }),
    };
    let image = match step {
        Ok(None) => return None,
        Ok(Some(image)) => image,
        Err(failure) => return Some((map.still.take()?, Err(failure), map.generation)),
    };
    entry.still = Some(image);
    entry.frame_generation += 1;
    // A still image publishes the update it draws.
    map.render_update_generation += 1;
    let generation = map.generation;
    let shown = Shown {
        view: map.view(),
        generation: entry.generation,
        update: map.render_update_generation,
        needs_repaint: false,
        full: true,
    };
    entry.drawn_update = shown.update;
    entry.shown = Some(shown);
    entry.handle.push(result(
        MLN_FRAME_DEMAND_TOKEN_STILL_IMAGE,
        MLN_RENDER_RESULT_RENDERED,
        entry.frame_generation,
        Some(shown),
    ));
    entry.sync_state();
    Some((map.still.take()?, Ok(Value::None), generation))
}

fn end(
    session: mln_render_session,
    completion: Option<Completion>,
    wait: bool,
    state_after: mln_render_session_state,
    retire: bool,
) -> Result {
    let handle = handles::session(session.0)?;
    let lifecycle = handle.lifecycle();
    if !retire
        && matches!(
            lifecycle,
            MLN_RENDER_SESSION_STATE_DETACHING
                | MLN_RENDER_SESSION_STATE_DETACHED
                | MLN_RENDER_SESSION_STATE_ABANDONED
        )
    {
        return Err(Failure::state(
            "the session was already detached or abandoned",
        ));
    }
    // A dispose ends the frames the host holds; a detach or a release
    // waits for them.
    if wait {
        crate::frames::end_frames(&handle);
    } else {
        handle.no_frames_held()?;
    }
    // A detach ends the session's rendering and keeps its handle for the
    // release, which retires it.
    if retire {
        handles::remove(session.0);
    }
    if lifecycle != MLN_RENDER_SESSION_STATE_ABANDONED {
        handle.state.lock().unwrap().state = MLN_RENDER_SESSION_STATE_DETACHING;
    }
    let id = handle.id();
    let runtime = handle.runtime.clone();
    let job = move |state: &mut State| {
        if let Some(entry) = state.sessions.get(&id) {
            entry.handle.state.lock().unwrap().state = state_after;
        }
        drop_session(state, id);
        Ok(())
    };
    if wait {
        // A runtime that stopped took its sessions with it.
        let ended = handle.runtime.call(job);
        if retire {
            runtime.child_released();
        }
        match ended {
            Err(f) if f.status == MLN_STATUS_INVALID_STATE => Ok(()),
            ended => ended,
        }
    } else {
        handle.runtime.post(move |state| {
            let _ = job(state);
            state.complete(completion, Ok(Value::None), 0);
            if retire {
                runtime.child_released();
            }
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_release(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let completion = Some(Completion::required(completion)?);
            end(
                session,
                completion,
                false,
                MLN_RENDER_SESSION_STATE_DETACHED,
                true,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_detach(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let completion = Some(Completion::required(completion)?);
            end(
                session,
                completion,
                false,
                MLN_RENDER_SESSION_STATE_DETACHED,
                false,
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_dispose(
    session: mln_render_session,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            end(session, None, true, MLN_RENDER_SESSION_STATE_DETACHED, true)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_abandon(
    session: mln_render_session,
    out_result: *mut mln_render_abandon_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_result.is_null() {
                return Err(Failure::argument("out_result is null"));
            }
            sized(out_result.cast_const(), "out_result")?;
            let handle = handles::session(session.0)?;
            if matches!(
                handle.lifecycle(),
                MLN_RENDER_SESSION_STATE_DETACHING
                    | MLN_RENDER_SESSION_STATE_DETACHED
                    | MLN_RENDER_SESSION_STATE_ABANDONED
            ) {
                return Err(Failure::state(
                    "the session was already detached or abandoned",
                ));
            }
            if handle.open_views.load(Ordering::Acquire) > 0 {
                return Err(Failure::new(
                    MLN_STATUS_BUSY,
                    "a view of one of the session's frames is open",
                ));
            }
            abandon(&handle, false)?;
            put(out_result, abandoned(0), "out_result")
        })
    }
}

fn abandoned(kept: u32) -> mln_render_abandon_result {
    mln_render_abandon_result {
        size: std::mem::size_of::<mln_render_abandon_result>() as u32,
        disposition: if kept > 0 {
            MLN_RENDER_ABANDON_DISPOSITION_QUARANTINED
        } else {
            MLN_RENDER_ABANDON_DISPOSITION_CLEAN
        },
        quarantined_resource_count: kept,
        reserved: 0,
    }
}

/// An abandon for a process that exits: it keeps the session's graphics
/// objects, which the exit frees, rather than destroy them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_binding_render_session_abandon_at_exit(
    session: mln_render_session,
    out_result: *mut mln_render_abandon_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if out_result.is_null() {
                return Err(Failure::argument("out_result is null"));
            }
            sized(out_result.cast_const(), "out_result")?;
            let handle = handles::session(session.0)?;
            if matches!(
                handle.lifecycle(),
                MLN_RENDER_SESSION_STATE_DETACHED | MLN_RENDER_SESSION_STATE_ABANDONED
            ) {
                return Err(Failure::state(
                    "the session was already detached or abandoned",
                ));
            }
            let kept = abandon(&handle, true)?;
            put(out_result, abandoned(kept), "out_result")
        })
    }
}

/// Orbis's sessions are driven by their runtime's thread
/// (MLN_RENDER_DRIVER_CORE_WORKER), so the host has nothing to service.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_service_driver_work(
    session: mln_render_session,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            handles::session(session.0)?;
            Err(Failure::state(
                "the session's driver is the core worker: it has no driver work to service",
            ))
        })
    }
}

/// Stops a session at once: once it returns, the runtime no longer uses
/// the session's graphics objects. Metal and Vulkan objects of Orbis's own
/// device go with it, so nothing is kept.
pub(crate) fn abandon(handle: &Arc<SessionHandle>, keep: bool) -> Result<u32> {
    if handle.runtime.on_thread() {
        return Err(Failure::new(
            MLN_STATUS_BUSY,
            "a session can't be abandoned from inside the runtime's own work",
        ));
    }
    let id = handle.id();
    handle
        .runtime
        .call(move |state| {
            if let Some(entry) = state.sessions.get(&id) {
                entry.handle.state.lock().unwrap().state = MLN_RENDER_SESSION_STATE_ABANDONED;
            }
            // At exit, nothing of the session's is destroyed: its graphics
            // objects stay until the process ends.
            let kept = match state.sessions.get_mut(&id).and_then(|e| e.target.take()) {
                Some(target) if keep => {
                    let count = match &target {
                        Target::One { .. } => 1,
                        Target::Ring(targets) => targets.len() as u32,
                    };
                    std::mem::forget(target);
                    count
                }
                _ => 0,
            };
            drop_session(state, id);
            Ok(kept)
        })
        .map_err(|f| match f.status {
            MLN_STATUS_WRONG_THREAD => Failure::new(
                MLN_STATUS_BUSY,
                "a session can't be abandoned from inside the runtime's own work",
            ),
            _ => f,
        })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_offscreen_device_create(
    options: *const mln_offscreen_device_options,
    out_device: *mut mln_offscreen_device,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            if !options.is_null() {
                let o = sized(options, "options")?;
                let known = MLN_OFFSCREEN_DEVICE_OPTION_ADAPTER_INDEX
                    | MLN_OFFSCREEN_DEVICE_OPTION_SINGLE_CONTEXT
                    | MLN_OFFSCREEN_DEVICE_OPTION_CLIENT_API;
                if o.fields & !known != 0 {
                    return Err(Failure::argument("options.fields sets an unknown field"));
                }
                if o.fields & MLN_OFFSCREEN_DEVICE_OPTION_CLIENT_API != 0 {
                    match o.client_api {
                        MLN_OPEN_GL_CLIENT_API_GL | MLN_OPEN_GL_CLIENT_API_GLES => {
                            return Err(Failure::unsupported(
                                "Orbis renders through Metal or Vulkan, with no OpenGL API",
                            ));
                        }
                        _ => return Err(Failure::argument("options.client_api names no API")),
                    }
                }
                // The runtime's device is on the default adapter, the one
                // GPU its sessions can render on.
                if o.fields & MLN_OFFSCREEN_DEVICE_OPTION_ADAPTER_INDEX != 0 && o.adapter_index != 0
                {
                    return Err(Failure::native(format!(
                        "adapter {} names no GPU: Orbis renders on its runtime's adapter, 0",
                        o.adapter_index
                    )));
                }
            }
            crate::base::fresh_out(out_device, "out_device")?;
            put(
                out_device,
                mln_offscreen_device(handles::insert(Object::OffscreenDevice)),
                "out_device",
            )
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_offscreen_device_release(device: mln_offscreen_device) {
    handles::remove(device.0);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_render_session_set_frame_pacer(
    session: mln_render_session,
    pacer: mln_frame_pacer,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            if handle.redraw_wake {
                return Err(Failure::state(
                    "the session was attached with a redraw wake, which paces it instead",
                ));
            }
            let pacer = if pacer.0 == 0 {
                None
            } else {
                Some(handles::pacer(pacer.0)?)
            };
            let id = handle.id();
            handle.runtime.post(move |state| {
                let Some(entry) = state.sessions.get_mut(&id) else {
                    return;
                };
                if entry.redraw_wake.is_some() {
                    return;
                }
                if let Some(old) = entry.pacer.take() {
                    old.remove(&entry.handle);
                }
                if let Some(pacer) = &pacer {
                    pacer.add(&entry.handle);
                }
                entry.pacer = pacer;
            })
        })
    }
}

/// Ends a session on the runtime thread: its map can attach another.
pub(crate) fn drop_session(state: &mut State, id: u64) {
    if let Some(mut entry) = state.sessions.remove(&id) {
        // Its graphics objects go first; what it was asked for ends.
        entry.target = None;
        entry.handle.ring.lock().unwrap().clear();
        for demand in std::mem::take(&mut entry.demands) {
            entry.handle.push(result(
                demand.token,
                MLN_RENDER_RESULT_TARGET_NOT_READY,
                entry.frame_generation,
                None,
            ));
        }
        if let Some(map) = state.maps.get_mut(&entry.map) {
            map.session = None;
            map.handle_flag(false);
            // Native's renderer takes its tile requests with it; the next
            // session's frames ask again.
            map.map.cancel_tile_loads();
            // The map's still image waits for its next session.
            if map.still.is_some() {
                entry.handle.push(result(
                    MLN_FRAME_DEMAND_TOKEN_STILL_IMAGE,
                    MLN_RENDER_RESULT_TARGET_NOT_READY,
                    entry.frame_generation,
                    None,
                ));
            }
        }
        entry.publish();
        if let Some(wake) = entry.handle.frame_wake {
            wake.release();
        }
    }
}

/// The map handle, for its session flag, carried into the attach job.
struct MapFlag(Arc<crate::map::MapHandle>);

impl MapFlag {
    fn store(&self, value: bool, order: std::sync::atomic::Ordering) {
        self.0.has_session.store(value, order);
    }
}
