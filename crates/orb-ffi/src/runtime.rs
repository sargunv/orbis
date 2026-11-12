// A runtime is mln-ffi's core worker: one thread that runs every map's
// commands in the order they were sent, and steps the maps no render
// session draws. Each session draws its map on a thread of its own
// (session.rs), so frames don't wait behind commands, and a command's
// callback that parks the worker holds back no frame, no detach.
//
// A map is its state behind a lock (map.rs's `MapCell`): the thread that
// works on it, this one for a command or its session's for a frame, holds
// the lock while it does, then publishes the map's snapshot and takes its
// events before letting go. Calls from any thread check their arguments on
// the caller's thread and post the work; what it reports waits in the
// working thread's outbox until the lock is let go, the events first, so a
// completion follows its events, both follow the state they report, and a
// callback can call the library back.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crate::base::{Failure, Result, UserData, boundary, put, sized, text};
use crate::delivery::{self, Completion, Value};
use crate::handles::{self, Object};
use crate::map::MapCell;
use crate::sys::*;

type Job = Box<dyn FnOnce(&mut State) + Send>;

pub(crate) enum Msg {
    Run(Job),
    /// Data arrived for a map, or a pacer ticked: look for work.
    Pump,
    Stop(Option<Completion>),
}

/// What every thread of a runtime shares: the runtime thread, its
/// sessions' threads.
pub(crate) struct Core {
    pub(crate) runtime: orb_api::Runtime,
    /// Frame and input times are milliseconds since this.
    epoch: Instant,
    handler: Mutex<Option<EventHandler>>,
    /// Wakes the runtime thread to step its maps.
    tx: Sender<Msg>,
}

impl Core {
    pub(crate) fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }

    /// Sends a batch of events to the handler, if there is one.
    pub(crate) fn send(&self, events: Vec<Event>) {
        if events.is_empty() {
            return;
        }
        if let Some(handler) = &*self
            .handler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            handler.send(events);
        }
    }

    /// Asks the runtime thread to look for maps to step.
    pub(crate) fn wake(&self) {
        let _ = self.tx.send(Msg::Pump);
    }

    fn replace_handler(&self, handler: Option<EventHandler>) {
        let old = std::mem::replace(
            &mut *self
                .handler
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            handler,
        );
        if let Some(old) = old {
            old.release();
        }
    }
}

/// What the runtime thread owns.
pub(crate) struct State {
    pub(crate) core: Arc<Core>,
    pub(crate) maps: HashMap<u64, Arc<MapCell>>,
    /// The render sessions' threads, for the runtime's end to stop.
    pub(crate) sessions: HashMap<u64, crate::session::Link>,
    pub(crate) hook: Arc<crate::resources::Hook>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// The handle's side, shared with every thread.
pub(crate) struct RuntimeHandle {
    tx: Sender<Msg>,
    thread: ThreadId,
    join: Mutex<Option<JoinHandle<()>>>,
    /// Maps and render sessions alive on the runtime, counted on the
    /// caller's side so a release can refuse at once.
    pub(crate) maps: std::sync::atomic::AtomicUsize,
    /// The resource provider and transforms, where the runtime's fetches
    /// would reach the network.
    pub(crate) hook: Arc<crate::resources::Hook>,
    /// Disposed while maps lived: it stops once the last one is released.
    disposed: std::sync::atomic::AtomicBool,
}

impl RuntimeHandle {
    /// Runs `job` on the runtime thread later, in call order.
    pub(crate) fn post(&self, job: impl FnOnce(&mut State) + Send + 'static) -> Result {
        self.tx
            .send(Msg::Run(Box::new(job)))
            .map_err(|_| Failure::state("the runtime was released"))
    }

    /// A map or session of the runtime went: a disposed runtime stops
    /// after the last.
    pub(crate) fn child_released(&self) {
        if self.maps.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            self.last_map_released();
        }
    }

    pub(crate) fn on_thread(&self) -> bool {
        std::thread::current().id() == self.thread
    }

    /// Finishes a disposal that waited for the runtime's maps.
    pub(crate) fn last_map_released(&self) {
        if self
            .disposed
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            let _ = self.tx.send(Msg::Stop(None));
        }
    }
}

impl State {
    pub(crate) fn runtime(&self) -> &orb_api::Runtime {
        &self.core.runtime
    }

    /// Delivers `completion` after what the work so far reported.
    pub(crate) fn complete(
        &self,
        completion: Option<Completion>,
        result: Result<Value>,
        generation: u64,
    ) {
        complete_update(completion, result, generation, generation);
    }
}

/// Delivers a command's `completion`, for a change the render update
/// `render_update` first shows; a command posted without one has nobody to
/// tell of its failure but the log, as in Native.
pub(crate) fn complete_update(
    completion: Option<Completion>,
    result: Result<Value>,
    generation: u64,
    render_update: u64,
) {
    match (completion, result) {
        (Some(c), result) => c.complete_update(result, generation, render_update),
        (None, Err(failure)) => crate::logging::log(
            MLN_LOG_SEVERITY_WARNING,
            &format!(
                "a discarded command failed (status {}): {}",
                failure.status, failure.message
            ),
        ),
        (None, Ok(_)) => {}
    }
}

/// A runtime event, owned until its batch is delivered.
pub(crate) struct Event {
    pub(crate) kind: mln_runtime_event_type,
    pub(crate) source_type: mln_runtime_event_source_type,
    pub(crate) source: u64,
    pub(crate) generation: u64,
    pub(crate) payload: Payload,
    pub(crate) message: String,
}

pub(crate) enum Payload {
    None,
    CameraChange(mln_camera_change_mode, mln_camera_change_source),
    LoadingFailed(mln_map_load_error),
    RenderFrame {
        needs_repaint: bool,
        full: bool,
        stats: mln_rendering_stats,
    },
    StyleImage(String),
    SourceChanged(String),
    TileAction {
        source: String,
        operation: mln_tile_operation,
        tile: orb_api::TileIndex,
    },
    Sprite {
        id: String,
        url: String,
    },
}

/// A registered `mln_runtime_event_handler`.
pub(crate) struct EventHandler {
    callback: unsafe extern "C" fn(*mut std::ffi::c_void, *const mln_event_batch_view),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
}

impl EventHandler {
    fn send(&self, events: Vec<Event>) {
        let (callback, user_data) = (self.callback, self.user_data);
        delivery::deliver(
            self.queue,
            Box::new(move || {
                let mut events = events;
                let c: Vec<mln_runtime_event> = events.iter_mut().map(c_event).collect();
                let batch = mln_event_batch_view {
                    size: std::mem::size_of::<mln_event_batch_view>() as u32,
                    event_size: std::mem::size_of::<mln_runtime_event>() as u32,
                    events: c.as_ptr(),
                    event_count: c.len(),
                };
                // SAFETY: the caller's registration; the batch lives until it
                // returns.
                unsafe { callback(user_data.get(), &batch) };
            }),
        );
    }

    fn release(self) {
        delivery::release_later(self.queue, self.release_user_data, self.user_data);
    }
}

/// An event as C reads it, lending its text.
fn c_event(e: &mut Event) -> mln_runtime_event {
    let mut event = mln_runtime_event {
        type_: e.kind,
        source_type: e.source_type,
        source: e.source,
        generation: e.generation,
        message: crate::base::lend(&mut e.message),
        ..Default::default()
    };
    match &mut e.payload {
        Payload::None => {}
        Payload::CameraChange(mode, source) => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_CAMERA_CHANGE;
            event.payload.camera_change = mln_runtime_event_camera_change {
                mode: *mode,
                source: *source,
            };
        }
        Payload::LoadingFailed(error) => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_LOADING_FAILED;
            event.payload.loading_failed = mln_runtime_event_loading_failed { error: *error };
        }
        Payload::RenderFrame {
            needs_repaint,
            full,
            stats,
        } => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_RENDER_FRAME;
            event.payload.render_frame = mln_runtime_event_render_frame {
                mode: if *full {
                    MLN_RENDER_MODE_FULL
                } else {
                    MLN_RENDER_MODE_PARTIAL
                },
                needs_repaint: *needs_repaint,
                placement_changed: false,
                stats: *stats,
            };
        }
        Payload::StyleImage(id) => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_STYLE_IMAGE;
            event.payload.style_image = mln_runtime_event_style_image {
                image_id: crate::base::lend(id),
            };
        }
        Payload::SourceChanged(id) => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_SOURCE_CHANGED;
            event.payload.source_changed = mln_runtime_event_source_changed {
                source_id: crate::base::lend(id),
            };
        }
        Payload::TileAction {
            source,
            operation,
            tile,
        } => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_TILE_ACTION;
            let z = u32::from(tile.z);
            event.payload.tile_action = mln_runtime_event_tile_action {
                source_id: crate::base::lend(source),
                operation: *operation,
                tile_id: mln_tile_id {
                    overscaled_z: z,
                    wrap: 0,
                    canonical_z: z,
                    canonical_x: tile.col,
                    canonical_y: tile.row,
                },
            };
        }
        Payload::Sprite { id, url } => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_SPRITE;
            event.payload.sprite = mln_runtime_event_sprite {
                id: crate::base::lend(id),
                url: crate::base::lend(url),
            };
        }
    }
    event
}

/// How often a map steps that nothing paces: a camera that moves with no
/// session drawing it, a still image waiting on time. Data, compiles and
/// pacers wake their threads themselves.
pub(crate) const STEP: Duration = Duration::from_millis(16);

/// A fresh directory for the regions of a runtime without a cache file,
/// removed with it.
fn temporary_state_dir() -> Result<std::path::PathBuf> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!(
        "orbis-runtime-{}-{n}-{stamp:x}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir)
        .map_err(|e| Failure::native(format!("can't make the runtime's regions directory: {e}")))?;
    Ok(dir)
}

/// Runs the runtime thread until it's stopped; `temporary` is its regions
/// directory to remove then, before the stop completes.
fn run(rx: Receiver<Msg>, temporary: Option<&std::path::Path>) {
    loop {
        let steps = STATE.with(|s| s.borrow().as_ref().is_some_and(crate::map::any_steps));
        let first = if steps {
            match rx.recv_timeout(STEP) {
                Ok(msg) => Some(msg),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match rx.recv() {
                Ok(msg) => Some(msg),
                Err(_) => return,
            }
        };
        let mut stop = None;
        for msg in first.into_iter().chain(rx.try_iter()) {
            match msg {
                Msg::Run(job) => STATE.with(|s| {
                    let mut s = s.borrow_mut();
                    job(s.as_mut().expect("runtime state"));
                }),
                Msg::Pump => {}
                Msg::Stop(completion) => {
                    stop = Some(completion);
                    break;
                }
            }
            delivery::flush();
        }
        if let Some(completion) = stop {
            let state = STATE.with(|s| s.borrow_mut().take());
            if let Some(mut state) = state {
                // Sessions before maps before the runtime, as their devices
                // and surfaces need.
                for (_, session) in state.sessions.drain() {
                    session.stop();
                }
                state.maps.clear();
                state.core.replace_handler(None);
                state.runtime().set_network_hook(None);
                state.hook.clear();
            }
            if let Some(dir) = temporary {
                let _ = std::fs::remove_dir_all(dir);
            }
            if let Some(completion) = completion {
                completion.complete(Ok(Value::None), 0);
            }
            delivery::flush();
            return;
        }
        STATE.with(|s| crate::map::advance(s.borrow_mut().as_mut().expect("runtime state")));
        delivery::flush();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_create(
    options: *const mln_runtime_options,
    out_runtime: *mut mln_runtime,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            crate::base::fresh_out(out_runtime, "out_runtime")?;
            let mut state_dir = None;
            if !options.is_null() {
                let options = sized(options, "options")?;
                let cache = text(options.cache_path, "cache_path")?;
                text(options.asset_path, "asset_path")?;
                if !cache.is_empty() {
                    state_dir = Some(state_dir_for(&cache)?);
                }
            }
            // Without a cache file, as in MapLibre Native's in-memory
            // database, offline regions last as long as the runtime.
            let temporary = match state_dir {
                Some(_) => None,
                None => Some(temporary_state_dir()?),
            };
            // The device waits for the first session that needs it.
            let runtime = orb_api::Runtime::new(&orb_api::RuntimeOptions {
                state_dir,
                regions_dir: temporary.clone(),
                ..Default::default()
            })
            .map_err(|e| {
                if let Some(dir) = &temporary {
                    let _ = std::fs::remove_dir_all(dir);
                }
                Failure::from(e)
            })?;
            let hook = crate::resources::hook();
            runtime.set_online(crate::network::online());
            runtime.set_network_hook(Some(hook.clone()));
            let (tx, rx) = mpsc::channel();
            let core = Arc::new(Core {
                runtime,
                epoch: Instant::now(),
                handler: Mutex::new(None),
                tx: tx.clone(),
            });
            let state = State {
                core: core.clone(),
                maps: HashMap::new(),
                sessions: HashMap::new(),
                hook: hook.clone(),
            };
            let join = std::thread::Builder::new()
                .name("orbis-runtime".into())
                .spawn(move || {
                    delivery::start_outbox();
                    STATE.with(|s| *s.borrow_mut() = Some(state));
                    run(rx, temporary.as_deref());
                    if let Some(dir) = &temporary {
                        let _ = std::fs::remove_dir_all(dir);
                    }
                })
                .map_err(|e| Failure::native(format!("can't start the runtime thread: {e}")))?;
            let handle = Arc::new(RuntimeHandle {
                tx,
                thread: join.thread().id(),
                join: Mutex::new(Some(join)),
                maps: Default::default(),
                hook,
                disposed: Default::default(),
            });
            let id = handles::insert(Object::Runtime(handle.clone()));
            crate::network::register(&handle);
            put(out_runtime, mln_runtime(id), "out_runtime")
        })
    }
}

/// How the file Orbis writes at a cache path starts.
pub(crate) const CACHE_MARKER: &str = "An Orbis cache:";

/// Where Orbis keeps a runtime's state for MapLibre Native's cache file:
/// a directory beside the file, which names it. Both are made if need be,
/// in a directory that must exist and be writable, as Native's database
/// file is.
fn state_dir_for(cache: &str) -> Result<std::path::PathBuf> {
    let file = std::path::Path::new(cache);
    let name = file
        .file_name()
        .ok_or_else(|| Failure::argument(format!("can't open the cache at {cache}: no file name")))?
        .to_string_lossy()
        .into_owned();
    let open =
        |e: std::io::Error| Failure::argument(format!("can't open the cache at {cache}: {e}"));
    if !file.exists() {
        std::fs::write(
            file,
            format!("{CACHE_MARKER} its resources and offline regions are in {name}.orbis.\n"),
        )
        .map_err(open)?;
    }
    let path = std::path::PathBuf::from(format!("{cache}.orbis"));
    std::fs::create_dir_all(&path).map_err(open)?;
    Ok(path)
}

fn stop(runtime: mln_runtime, completion: Option<Completion>, wait: bool) -> Result {
    let handle = handles::runtime(runtime.0)?;
    if handle.maps.load(std::sync::atomic::Ordering::Acquire) > 0 {
        if completion.is_some() {
            return Err(Failure::state(
                "the runtime still has maps or render sessions: release them first",
            ));
        }
        // A disposal finishes once the last map is released.
        handles::remove(runtime.0);
        crate::network::unregister(&handle);
        handle
            .disposed
            .store(true, std::sync::atomic::Ordering::Release);
        if handle.maps.load(std::sync::atomic::Ordering::Acquire) == 0 {
            handle.last_map_released();
        }
        return Ok(());
    }
    handles::remove(runtime.0);
    crate::network::unregister(&handle);
    let _ = handle.tx.send(Msg::Stop(completion));
    if wait
        && !handle.on_thread()
        && let Some(join) = handle
            .join
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    {
        let _ = join.join();
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_release(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let completion = Completion::required(completion)?;
            handles::runtime(runtime.0)?;
            stop(runtime, Some(completion), false)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_dispose(
    runtime: mln_runtime,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe { boundary(out_diagnostic, || stop(runtime, None, false)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_set_event_handler(
    runtime: mln_runtime,
    handler: *const mln_runtime_event_handler,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let runtime = handles::runtime(runtime.0)?;
            let h = sized(handler, "handler")?;
            let callback = h
                .callback
                .ok_or_else(|| Failure::argument("handler.callback is null"))?;
            delivery::check_queue(h.queue.0, "handler.queue")?;
            let completion = Completion::from_c(completion)?;
            let handler = EventHandler {
                callback,
                user_data: UserData(h.user_data),
                release_user_data: h.release_user_data,
                queue: h.queue.0,
            };
            runtime.post(move |state| {
                state.core.replace_handler(Some(handler));
                state.complete(completion, Ok(Value::None), 0);
            })
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_runtime_clear_event_handler(
    runtime: mln_runtime,
    completion: *const mln_completion,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let runtime = handles::runtime(runtime.0)?;
            let completion = Completion::from_c(completion)?;
            runtime.post(move |state| {
                state.core.replace_handler(None);
                state.complete(completion, Ok(Value::None), 0);
            })
        })
    }
}
