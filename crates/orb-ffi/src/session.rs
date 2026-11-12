// Render sessions: an Orbis target (a CAMetalLayer surface, an RGBA8
// texture to read back, or a ring of the host's textures) that a map draws
// into, the frame demands that drive it, and the frame results it reports.
//
// Each session draws on a thread of its own, holding its map's lock for a
// frame: the runtime thread runs the map's commands meanwhile, and a frame
// or a detach never waits behind another map's work or a callback that
// parks the runtime thread. The session's demands and calls run in the
// order they were sent, each once the map has applied the commands sent
// before it, so a frame shows them; an end (detach, release, dispose,
// abandon) doesn't wait, and ends what still waits.
//
// Orbis's runtime owns its GPU device, so a session renders on that device:
// an attach that names another device (an MTLDevice, an offscreen device of
// another adapter) isn't supported, and a ring's textures must be of it.
// The device is made when the first session makes its target.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;
use std::time::Duration;

use crate::base::{Failure, Result, UserData, boundary, put, sized};
use crate::delivery::{self, Completion, Value, Wake};
use crate::handles::{self, Object};
use crate::map::{MapCell, MapEntry, MapHandle, settle};
use crate::runtime::{Core, Payload, RuntimeHandle};
use crate::sys::*;

/// What a session's thread is asked to do.
enum Msg {
    /// A frame demand, once the map applied command `after`.
    Demand {
        demand: mln_frame_demand,
        after: u64,
    },
    /// A call on the session, once the map applied command `after`: a
    /// `changes` call grows the session's generation.
    Run {
        job: Job,
        completion: Option<Completion>,
        changes: bool,
        after: u64,
    },
    /// The session ends, whatever waits.
    End(Ending),
    /// Something may be new: a command applied, data arrived, a pacer
    /// ticked, the host gave a frame back.
    Pump,
}

impl Msg {
    /// The map command it waits for.
    fn after(&self) -> u64 {
        match self {
            Self::Demand { after, .. } | Self::Run { after, .. } => *after,
            Self::End(_) | Self::Pump => 0,
        }
    }
}

type Job = Box<dyn FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send>;

/// How a session ends.
enum Ending {
    /// A detach or a release: the session's state becomes `state_after`,
    /// and a release also retires its handle.
    Detach {
        state_after: mln_render_session_state,
        retire: bool,
        completion: Option<Completion>,
        /// A dispose waits for it.
        done: Option<mpsc::SyncSender<()>>,
    },
    /// An abandon, which keeps the graphics objects at exit (`keep`) and
    /// reports how many it kept.
    Abandon {
        keep: bool,
        done: mpsc::SyncSender<u32>,
    },
    /// Its map was released or the runtime stops.
    Closed(Option<mpsc::SyncSender<()>>),
}

/// A session, as its map and runtime reach it.
#[derive(Clone)]
pub(crate) struct Link(Arc<SessionHandle>);

impl Link {
    pub(crate) fn id(&self) -> u64 {
        self.0.id()
    }

    /// Has the session look at its map again.
    pub(crate) fn wake(&self) {
        self.0.wake();
    }

    /// Ends the session as its map goes.
    pub(crate) fn map_released(&self) {
        self.0.send(Msg::End(Ending::Closed(None)));
    }

    /// Ends the session as its runtime stops, waiting a while for its
    /// graphics objects to go.
    pub(crate) fn stop(&self) {
        let (tx, rx) = mpsc::sync_channel(1);
        self.0.send(Msg::End(Ending::Closed(Some(tx))));
        let _ = rx.recv_timeout(Duration::from_secs(1));
    }
}

/// What the caller's threads share with the session's thread.
pub(crate) struct SessionHandle {
    pub(crate) runtime: Arc<RuntimeHandle>,
    map: Arc<MapHandle>,
    id: OnceLock<u64>,
    /// The session's thread, until the session ends.
    tx: Mutex<Option<Sender<Msg>>>,
    thread: OnceLock<ThreadId>,
    /// A pump is on its way to the thread.
    pumped: AtomicBool,
    frame_wake: Option<Wake>,
    results: Mutex<Vec<mln_render_frame_result>>,
    /// Whether the frame wake was called since the last drain.
    notified: AtomicBool,
    /// A pacer's refresh not yet seen by the session's thread.
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

    /// Wakes the session's thread, to look at its map and demands again.
    pub(crate) fn wake(&self) {
        if !self.pumped.swap(true, Ordering::AcqRel) {
            self.send(Msg::Pump);
        }
    }

    /// Sends `msg` to the session's thread, or once the session ended
    /// answers it as an ended session does.
    fn send(&self, msg: Msg) {
        let unsent = match &*self.tx.lock().unwrap() {
            Some(tx) => tx.send(msg).err().map(|e| e.0),
            None => Some(msg),
        };
        if let Some(msg) = unsent {
            self.after_end(msg);
        }
    }

    /// Stops taking messages, once the session ended, and answers those
    /// that came meanwhile.
    fn close(&self, rx: &Receiver<Msg>) {
        self.tx.lock().unwrap().take();
        for msg in rx.try_iter() {
            self.after_end(msg);
        }
    }

    /// Answers what reaches a session that ended: calls fail, a further
    /// end has nothing left to end.
    fn after_end(&self, msg: Msg) {
        match msg {
            Msg::Run { completion, .. } => lost_target(completion),
            Msg::End(Ending::Detach {
                state_after,
                retire,
                completion,
                done,
            }) => {
                let mut state = self.state.lock().unwrap();
                if state.state != MLN_RENDER_SESSION_STATE_ABANDONED {
                    state.state = state_after;
                }
                drop(state);
                crate::runtime::complete_update(completion, Ok(Value::None), 0, 0);
                if retire {
                    self.runtime.child_released();
                }
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            Msg::End(Ending::Abandon { done, .. }) => {
                self.state.lock().unwrap().state = MLN_RENDER_SESSION_STATE_ABANDONED;
                let _ = done.send(0);
            }
            Msg::End(Ending::Closed(done)) => {
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            Msg::Demand { .. } | Msg::Pump => {}
        }
    }

    /// The map the session draws.
    pub(crate) fn map_handle(&self) -> &Arc<MapHandle> {
        &self.map
    }

    /// Whether this is the session's own thread.
    fn on_thread(&self) -> bool {
        self.thread.get() == Some(&std::thread::current().id())
    }

    /// Posts a change to the session's state: its generation grows once
    /// the change is made.
    fn command(
        &self,
        completion: Option<Completion>,
        job: impl FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send + 'static,
    ) -> Result {
        self.post(completion, true, job)
    }

    /// Posts work that leaves the session's state as it is.
    fn operation(
        &self,
        completion: Option<Completion>,
        job: impl FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send + 'static,
    ) -> Result {
        self.post(completion, false, job)
    }

    /// Runs `job` on the session's thread after the demands and calls sent
    /// before it, once the map applied the commands sent before it.
    fn post(
        &self,
        completion: Option<Completion>,
        changes: bool,
        job: impl FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send + 'static,
    ) -> Result {
        let msg = Msg::Run {
            job: Box::new(job),
            completion,
            changes,
            after: self.map.posted(),
        };
        self.send(msg);
        Ok(())
    }
}

/// Fails a call that reached a session after it ended.
fn lost_target(completion: Option<Completion>) {
    crate::runtime::complete_update(
        completion,
        Err(Failure::new(
            MLN_STATUS_TARGET_LOST,
            "the session lost its target before the work ran",
        )),
        0,
        0,
    );
}

pub(crate) struct SessionEntry {
    handle: Arc<SessionHandle>,
    runtime: orb_api::Runtime,
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
    runtime: &orb_api::Runtime,
    layer: UserData,
    logical: mln_logical_size,
    scale: f64,
) -> Result<Target> {
    let (width, height) = physical(logical, scale)?;
    let layer = NonNull::new(layer.get()).ok_or_else(|| Failure::argument("the layer is null"))?;
    let surface = runtime.with_device(|gpu| {
        // SAFETY: the C API's contract: the layer is a live CAMetalLayer that
        // the caller keeps until the session's target is released.
        unsafe { gpu.create_surface_raw(orb_api::RawSurface::MetalLayer(layer), width, height) }
            .map_err(Failure::native)
    })??;
    Ok(Target::One {
        target: orb_api::Target::surface(runtime, surface, scale),
        _layer: Some(Retained::new(UserData(layer.as_ptr()))),
    })
}

/// Refuses an MTLDevice other than the runtime's: Orbis renders on its
/// own device. A null one is the runtime's.
fn same_device(runtime: &orb_api::Runtime, device: UserData) -> Result {
    if device.get().is_null() {
        return Ok(());
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if runtime
        .with_device(|gpu| gpu.metal_device())?
        .map(|d| d.as_ptr())
        == Some(device.get())
    {
        return Ok(());
    }
    let _ = runtime;
    Err(Failure::unsupported(
        "Orbis renders on its runtime's own MTLDevice, not the one the descriptor names",
    ))
}

/// The host's textures as targets of the runtime's device.
fn ring_target(runtime: &orb_api::Runtime, slots: &[crate::frames::SlotInfo]) -> Result<Target> {
    let targets = slots
        .iter()
        .map(|slot| crate::frames::import(runtime, slot))
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
            "Orbis sessions render on threads of their own (MLN_RENDER_DRIVER_CORE_WORKER)",
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
    let (tx, rx) = mpsc::channel();
    let handle = Arc::new(SessionHandle {
        runtime: map_handle.runtime.clone(),
        map: map_handle.clone(),
        id: OnceLock::new(),
        tx: Mutex::new(Some(tx)),
        thread: OnceLock::new(),
        pumped: AtomicBool::new(false),
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
    map_handle.runtime.post(move |state| {
        let Some(cell) = state.maps.get(&map_id).cloned() else {
            handles::remove(id);
            handle.runtime.child_released();
            handle.map.has_session.store(false, Ordering::Release);
            let failure = Failure::state("the map was released");
            return state.complete(Some(completion), Err(failure), 0);
        };
        state.sessions.insert(id, Link(handle.clone()));
        let start = Start {
            spec,
            logical,
            redraw_wake,
            completion,
        };
        let core = state.core.clone();
        let spawned = std::thread::Builder::new()
            .name("orbis-session".into())
            .spawn(move || run(handle, core, cell, start, rx));
        if let Err(e) = spawned {
            crate::logging::log(
                MLN_LOG_SEVERITY_ERROR,
                &format!("can't start a render session's thread: {e}"),
            );
        }
    })?;
    // SAFETY: as above.
    unsafe { put(out_session, mln_render_session(id), "out_session") }
}

/// What a session's thread starts from.
struct Start {
    spec: TargetSpec,
    logical: mln_logical_size,
    redraw_wake: Option<Wake>,
    completion: Completion,
}

/// A session's thread: it makes the session's target, then draws its
/// demands and runs its calls until it ends.
fn run(
    handle: Arc<SessionHandle>,
    core: Arc<Core>,
    cell: Arc<MapCell>,
    start: Start,
    rx: Receiver<Msg>,
) {
    delivery::start_outbox();
    let _ = handle.thread.set(std::thread::current().id());
    let session = Session::start(handle.clone(), core, cell, start);
    let Some(mut session) = session else {
        handle.close(&rx);
        delivery::flush();
        handle.ended();
        return;
    };
    // The new session has the map to draw: it tells a host that waits for
    // its redraw wake.
    session.work();
    delivery::flush();
    loop {
        let first = if session.steps {
            match rx.recv_timeout(crate::runtime::STEP) {
                Ok(msg) => Some(msg),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(msg) => Some(msg),
                Err(_) => break,
            }
        };
        for msg in first.into_iter().chain(rx.try_iter()) {
            match msg {
                Msg::End(ending) => {
                    session.end(ending);
                    handle.close(&rx);
                    delivery::flush();
                    handle.ended();
                    return;
                }
                Msg::Pump => handle.pumped.store(false, Ordering::Release),
                msg => session.pending.push_back(msg),
            }
        }
        session.work();
        delivery::flush();
    }
    session.end(Ending::Closed(None));
    handle.close(&rx);
    delivery::flush();
    handle.ended();
}

impl SessionHandle {
    /// The session's thread ends: the runtime stops keeping it.
    fn ended(self: &Arc<Self>) {
        let id = self.id();
        let _ = self.runtime.post(move |state| {
            state.sessions.remove(&id);
        });
    }
}

/// A session on its thread.
struct Session {
    entry: SessionEntry,
    cell: Arc<MapCell>,
    core: Arc<Core>,
    /// Demands and calls that wait for the map's commands.
    pending: VecDeque<Msg>,
    /// It steps on time: its map's still image needs frames for more than
    /// data, which wakes it.
    steps: bool,
}

impl Session {
    /// Makes the session's target and attaches it to its map, completing
    /// the attach; `None` if it failed.
    fn start(
        handle: Arc<SessionHandle>,
        core: Arc<Core>,
        cell: Arc<MapCell>,
        start: Start,
    ) -> Option<Self> {
        let runtime = core.runtime.clone();
        let scale = cell.lock().extent.scale_factor;
        let target = match start.spec {
            TargetSpec::Offscreen => physical(start.logical, scale).map(|(w, h)| Target::One {
                target: orb_api::Target::texture(&runtime, w, h, scale),
                _layer: None,
            }),
            TargetSpec::Metal(layer, device) => same_device(&runtime, device)
                .and_then(|()| metal_target(&runtime, layer, start.logical, scale)),
            TargetSpec::Ring(slots) => ring_target(&runtime, &slots),
        };
        let mut map = cell.lock();
        let target = target.and_then(|target| match map.released {
            true => Err(Failure::state("the map was released")),
            false => Ok(target),
        });
        let target = match target {
            Ok(target) => target,
            Err(failure) => {
                drop(map);
                handles::remove(handle.id());
                handle.runtime.child_released();
                handle.map.has_session.store(false, Ordering::Release);
                start.completion.complete(Err(failure), 0);
                return None;
            }
        };
        map.set_session(Some(Link(handle.clone())));
        map.extent.width = start.logical.width;
        map.extent.height = start.logical.height;
        settle(&cell, &mut map);
        drop(map);
        let entry = SessionEntry {
            handle: handle.clone(),
            runtime,
            target: Some(target),
            logical: start.logical,
            scale,
            demands: VecDeque::new(),
            redraw_wake: start.redraw_wake,
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
        start.completion.complete(Ok(Value::None), 0);
        Some(Self {
            entry,
            cell,
            core,
            pending: VecDeque::new(),
            steps: false,
        })
    }

    /// Runs the demands and calls whose commands the map has applied, in
    /// order, then draws what the demands ask for and reports it.
    fn work(&mut self) {
        let applied = self.cell.applied.load(Ordering::Acquire);
        let cell = self.cell.clone();
        let mut map = cell.lock();
        while self.pending.front().is_some_and(|m| m.after() <= applied) {
            match self.pending.pop_front() {
                Some(Msg::Demand { demand, .. }) => queue_demand(&mut self.entry, demand),
                Some(Msg::Run {
                    job,
                    completion,
                    changes,
                    ..
                }) => {
                    // The demands sent before the call render first.
                    map = self.pump(&cell, map);
                    let result = job(&mut self.entry, &mut map);
                    if changes && result.is_ok() {
                        self.entry.generation += 1;
                    }
                    settle(&cell, &mut map);
                    self.entry.sync_state();
                    crate::runtime::complete_update(completion, result, 0, 0);
                }
                _ => {}
            }
        }
        map = self.pump(&cell, map);
        settle(&cell, &mut map);
        self.steps = map.still.is_some() && map.map.needs_frame();
    }

    /// Draws the frames the session's demands and pacer ask for, and its
    /// map's still image, and raises their events.
    fn pump<'a>(
        &mut self,
        cell: &'a MapCell,
        map: MutexGuard<'a, MapEntry>,
    ) -> MutexGuard<'a, MapEntry> {
        let now = self.core.now();
        let (mut map, drawn, still) = step(&mut self.entry, cell, map, now);
        for (needs_repaint, full, stats) in drawn {
            // The map's first frame starts rendering it, and the first that
            // shows everything after one that didn't finishes that, as
            // Native's renderer reports them.
            let rendered = map.rendered.replace(full);
            if rendered.is_none() {
                map.event(
                    MLN_RUNTIME_EVENT_MAP_RENDER_MAP_STARTED,
                    Payload::None,
                    String::new(),
                );
            }
            map.event(
                MLN_RUNTIME_EVENT_MAP_RENDER_FRAME_STARTED,
                Payload::None,
                String::new(),
            );
            map.event(
                MLN_RUNTIME_EVENT_MAP_RENDER_FRAME_FINISHED,
                Payload::RenderFrame {
                    needs_repaint,
                    full,
                    stats,
                },
                String::new(),
            );
            if full && matches!(rendered, None | Some(false)) {
                map.event(
                    MLN_RUNTIME_EVENT_MAP_RENDER_MAP_FINISHED,
                    Payload::None,
                    String::new(),
                );
            }
        }
        if let Some((completion, result, generation)) = still {
            settle(&self.cell, &mut map);
            completion.complete(result, generation);
        }
        map
    }

    /// Ends the session: what waits for the map's commands ends unrun, the
    /// target goes, and the map can attach another.
    fn end(mut self, ending: Ending) {
        // What is ready runs first; a detach doesn't wait for the rest.
        if !matches!(ending, Ending::Closed(_)) {
            self.work();
        }
        for msg in std::mem::take(&mut self.pending) {
            match msg {
                Msg::Demand { demand, .. } => self.entry.handle.push(result(
                    demand.token,
                    MLN_RENDER_RESULT_TARGET_NOT_READY,
                    self.entry.frame_generation,
                    None,
                )),
                Msg::Run { completion, .. } => lost_target(completion),
                Msg::End(_) | Msg::Pump => {}
            }
        }
        let handle = self.entry.handle.clone();
        match ending {
            Ending::Detach {
                state_after,
                retire,
                completion,
                done,
            } => {
                handle.state.lock().unwrap().state = state_after;
                self.close();
                crate::runtime::complete_update(completion, Ok(Value::None), 0, 0);
                if retire {
                    handle.runtime.child_released();
                }
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            Ending::Abandon { keep, done } => {
                handle.state.lock().unwrap().state = MLN_RENDER_SESSION_STATE_ABANDONED;
                // At exit, nothing of the session's is destroyed: its
                // graphics objects stay until the process ends.
                let kept = match self.entry.target.take() {
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
                self.close();
                let _ = done.send(kept);
            }
            Ending::Closed(done) => {
                self.close();
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
        }
    }

    /// Lets go of the session's target and map: its demands end, and the
    /// map can attach another.
    fn close(mut self) {
        let entry = &mut self.entry;
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
        let mut map = self.cell.lock();
        if map
            .session
            .as_ref()
            .is_some_and(|s| Arc::ptr_eq(&s.0, &entry.handle))
        {
            map.set_session(None);
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
        settle(&self.cell, &mut map);
        drop(map);
        entry.sync_state();
        if let Some(wake) = entry.handle.frame_wake {
            wake.release();
        }
    }
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
            handle.command(Completion::from_c(completion)?, move |entry, map| {
                // The old surface goes before the new one is made on its layer.
                entry.target = None;
                entry.target = Some(metal_target(&entry.runtime, layer, logical, entry.scale)?);
                entry.logical = logical;
                entry.sync_state();
                (map.extent.width, map.extent.height) = (logical.width, logical.height);
                force_redraw(map);
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
            handle.command(Completion::from_c(completion)?, |entry, _| {
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
            handle.command(Completion::from_c(completion)?, move |entry, map| {
                let (w, h) = physical(logical_size, entry.scale)?;
                if let Some(Target::One { target, .. }) = &mut entry.target {
                    target.resize(w, h, entry.scale);
                }
                entry.logical = logical_size;
                entry.sync_state();
                (map.extent.width, map.extent.height) = (logical_size.width, logical_size.height);
                force_redraw(map);
                Ok(Value::None)
            })
        })
    }
}

/// [`SessionHandle::command`], for other modules.
pub(crate) fn command(
    handle: &SessionHandle,
    completion: Option<Completion>,
    job: impl FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send + 'static,
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
    entry: &mut SessionEntry,
    map: &mut MapEntry,
    logical: mln_logical_size,
    slots: Vec<crate::frames::SlotInfo>,
) -> Result<Value> {
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
    entry.target = Some(ring_target(&entry.runtime, &slots)?);
    entry.handle.ring.lock().unwrap().replace(slots);
    entry.logical = logical;
    entry.sync_state();
    (map.extent.width, map.extent.height) = (logical.width, logical.height);
    force_redraw(map);
    Ok(Value::None)
}

/// Marks the session's next demand as having something to draw.
pub(crate) fn force_redraw(map: &mut MapEntry) {
    // A zero-length move dirties the map without moving its camera.
    map.map.jump_to(&orb_api::CameraTarget::default());
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
            // It shows the commands sent before it; one that reaches an
            // ended session has nobody to draw it.
            handle.send(Msg::Demand {
                demand,
                after: handle.map.posted(),
            });
            Ok(())
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

/// How long a frame waits for the pipelines it needs.
const PIPELINE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Draws a frame of everything the map holds. Orbis compiles pipelines
/// off the frame and draws what has none yet in a later frame; a frame of
/// the C API shows the map's state, as Native's synchronous compiles do,
/// so one that left something out for a compile is drawn again once the
/// compile is done.
///
/// The map's lock is let go while the compile runs, so its commands go on.
fn render_whole<'a>(
    cell: &'a MapCell,
    mut map: MutexGuard<'a, MapEntry>,
    target: &mut orb_api::Target,
    now: f64,
) -> (
    MutexGuard<'a, MapEntry>,
    std::result::Result<Option<orb_api::FrameReport>, orb_api::Error>,
) {
    let deadline = std::time::Instant::now() + PIPELINE_WAIT;
    loop {
        let report = match map.map.render(target, now) {
            Ok(report) => report,
            Err(e) => return (map, Err(e)),
        };
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match &report {
            // A compile wakes the map when it's done. A frame that waited
            // with none pending (a compile's arrival) draws it in the next.
            Some(r) if r.frame.waited && !left.is_zero() => {
                if r.frame.pipelines_pending > 0 {
                    let waiter = map.map.waiter();
                    drop(map);
                    waiter.wait(left.min(Duration::from_millis(16)));
                    map = cell.lock();
                }
            }
            _ => return (map, Ok(report)),
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

/// A frame a session drew: what its frame event reports.
type Drawn = (bool, bool, mln_rendering_stats);

/// A still image a session finished: its completion, outcome and the
/// map's generation.
type Still = (Completion, Result<Value>, u64);

/// Runs a session's demands: draws the frames they ask for and reports
/// each. Returns the map's lock, the frames it drew, and the still image
/// it finished.
fn step<'a>(
    entry: &mut SessionEntry,
    cell: &'a MapCell,
    mut map: MutexGuard<'a, MapEntry>,
    now: f64,
) -> (MutexGuard<'a, MapEntry>, Vec<Drawn>, Option<Still>) {
    let mut drawn = Vec::new();
    let mut still = None;
    // An open command group holds the map's changes back from every
    // frame, forced ones too, until the outermost group ends.
    let held = map.groups > 0;
    if !held && entry.target.is_some() && entry.handle.failed().is_ok() {
        still = step_still(&mut map, entry);
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
        // Commands sent and not yet applied are an update on its way: a
        // burst of them shows in one frame, once the last is in.
        let in_flight = cell.applied.load(Ordering::Acquire) < entry.handle.map.posted();
        let update = continuous && !in_flight && map.render_update_generation > entry.drawn_update;
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
        let (guard, rendered) = render_whole(cell, map, target, now);
        map = guard;
        let disposition = match rendered {
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
        let needs_repaint = needs_repaint(&map);
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
            drawn.push((needs_repaint, full, stats));
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
    (map, drawn, still)
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
fn readback(entry: &mut SessionEntry, _: &mut MapEntry) -> Result<Value> {
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
/// with its frames on the session's thread.
///
/// # Safety
/// The caller's pointers.
unsafe fn maintain(
    session: mln_render_session,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
    job: impl FnOnce(&mut SessionEntry, &mut MapEntry) -> Result<Value> + Send + 'static,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let handle = handles::session(session.0)?;
            handle.usable()?;
            handle.operation(Completion::from_c(completion)?, move |entry, map| {
                entry.handle.failed()?;
                job(entry, map)
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
        maintain(session, completion, out_diagnostic, |entry, _| {
            // The session's thread runs work in the order it's posted,
            // each after the map's commands sent before it, so what came
            // before the barrier has run; demands still waiting end.
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
        maintain(session, completion, out_diagnostic, |_, map| {
            map.map.release_cached_tiles();
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
        maintain(session, completion, out_diagnostic, |_, map| {
            map.map.clear_tiles();
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
        maintain(session, completion, out_diagnostic, |entry, _| {
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
                entry.runtime.stats().adapter,
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
            handle.command(None, false, move |map| {
                let refused = if map.still.is_some() {
                    Failure::state("the map already has a still image pending")
                } else if map.groups > 0 {
                    Failure::state("a still image can't be drawn inside a command group")
                } else {
                    // The map's session draws it from its next step on,
                    // with the tiles that failed before requested again.
                    map.map.retry_failed_tiles();
                    map.still = Some(completion);
                    return Ok(Value::None);
                };
                completion.complete(Err(refused), 0);
                Ok(Value::None)
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
    // The session's thread ends it at once, whatever waits; a dispose
    // waits for that, but for on the thread itself (from a callback).
    let (done, ended) = match wait && !handle.on_thread() {
        true => {
            let (tx, rx) = mpsc::sync_channel(1);
            (Some(tx), Some(rx))
        }
        false => (None, None),
    };
    let ending = Ending::Detach {
        state_after,
        retire,
        completion,
        done,
    };
    handle.send(Msg::End(ending));
    if let Some(ended) = ended {
        let _ = ended.recv();
    }
    Ok(())
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

/// Orbis's sessions are driven by threads of their own
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
    if handle.runtime.on_thread() || handle.on_thread() {
        return Err(Failure::new(
            MLN_STATUS_BUSY,
            "a session can't be abandoned from inside the runtime's own work",
        ));
    }
    let (done, kept) = mpsc::sync_channel(1);
    handle.send(Msg::End(Ending::Abandon { keep, done }));
    Ok(kept.recv().unwrap_or(0))
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
            handle.operation(None, move |entry, _| {
                if entry.redraw_wake.is_none() {
                    if let Some(old) = entry.pacer.take() {
                        old.remove(&entry.handle);
                    }
                    if let Some(pacer) = &pacer {
                        pacer.add(&entry.handle);
                    }
                    entry.pacer = pacer;
                }
                Ok(Value::None)
            })
        })
    }
}
