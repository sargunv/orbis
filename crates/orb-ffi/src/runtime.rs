// A runtime is one thread that owns an Orbis runtime (its GPU device and
// services) and the runtime's maps and render sessions: mln-ffi's core
// worker. Calls from any thread check their arguments on the caller's
// thread, post the work to the runtime thread, and report through their
// completion. After each piece of work the thread publishes the maps'
// snapshots, then sends the events the work raised, then its completions,
// so a completion follows its events and both follow the state they
// report.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use crate::base::{Failure, Result, UserData, boundary, put, sized, text};
use crate::delivery::{self, Completion, Value};
use crate::handles::{self, Object};
use crate::map::MapEntry;
use crate::session::SessionEntry;
use crate::sys::*;

type Job = Box<dyn FnOnce(&mut State) + Send>;

pub(crate) enum Msg {
    Run(Job),
    /// Data arrived for a map, or a pacer ticked: look for work.
    Pump,
    Stop(Option<Completion>),
}

/// What the runtime thread owns.
pub(crate) struct State {
    pub(crate) runtime: orb_api::Runtime,
    pub(crate) id: u64,
    pub(crate) maps: HashMap<u64, MapEntry>,
    pub(crate) sessions: HashMap<u64, SessionEntry>,
    pub(crate) handler: Option<EventHandler>,
    /// Frame and input times are milliseconds since this.
    pub(crate) epoch: Instant,
    /// Events raised and not yet sent.
    pub(crate) events: Vec<Event>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// The handle's side, shared with every thread.
pub(crate) struct RuntimeHandle {
    tx: Mutex<Sender<Msg>>,
    thread: ThreadId,
    join: Mutex<Option<JoinHandle<()>>>,
    /// Maps alive on the runtime, counted on the caller's side so a
    /// release can refuse at once.
    pub(crate) maps: std::sync::atomic::AtomicUsize,
}

impl RuntimeHandle {
    /// Runs `job` on the runtime thread later, in call order.
    pub(crate) fn post(&self, job: impl FnOnce(&mut State) + Send + 'static) -> Result {
        self.sender()
            .send(Msg::Run(Box::new(job)))
            .map_err(|_| Failure::state("the runtime was released"))
    }

    /// Runs `job` on the runtime thread and waits for its result. On the
    /// runtime thread itself (in a callback), it runs in place.
    pub(crate) fn call<R: Send + 'static>(
        &self,
        job: impl FnOnce(&mut State) -> Result<R> + Send + 'static,
    ) -> Result<R> {
        if std::thread::current().id() == self.thread {
            return STATE.with(|state| match state.try_borrow_mut() {
                Ok(mut state) => match state.as_mut() {
                    Some(state) => job(state),
                    None => Err(Failure::state("the runtime was released")),
                },
                Err(_) => Err(Failure::new(
                    MLN_STATUS_WRONG_THREAD,
                    "the call can't run inside the runtime's own work",
                )),
            });
        }
        let (tx, rx) = mpsc::sync_channel(1);
        self.post(move |state| {
            let _ = tx.send(job(state));
        })?;
        rx.recv()
            .map_err(|_| Failure::state("the runtime was released"))?
    }

    pub(crate) fn sender(&self) -> Sender<Msg> {
        self.tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(crate) fn on_thread(&self) -> bool {
        std::thread::current().id() == self.thread
    }
}

impl State {
    pub(crate) fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() * 1000.0
    }

    /// Queues an event of a map for the handler, if the map's mask takes
    /// it, at the map's current generation.
    pub(crate) fn event(
        &mut self,
        map: u64,
        kind: mln_runtime_event_type,
        payload: Payload,
        message: String,
    ) {
        let Some(entry) = self.maps.get(&map) else {
            return;
        };
        if entry.event_mask & (1u64 << kind) == 0 {
            return;
        }
        let generation = entry.generation;
        self.events.push(Event {
            kind,
            source_type: MLN_RUNTIME_EVENT_SOURCE_MAP,
            source: map,
            generation,
            payload,
            message,
        });
    }

    /// Publishes the maps' state, then sends the events raised so far:
    /// what a completion delivered after it can rely on.
    pub(crate) fn settle(&mut self) {
        let ids: Vec<u64> = self.maps.keys().copied().collect();
        for id in ids {
            crate::map::take_events(self, id);
        }
        for entry in self.maps.values() {
            entry.publish();
        }
        for entry in self.sessions.values() {
            entry.publish();
        }
        let events = std::mem::take(&mut self.events);
        if !events.is_empty()
            && let Some(handler) = &self.handler
        {
            handler.send(events);
        }
    }

    /// Delivers `completion` after the state and events of the work so
    /// far.
    pub(crate) fn complete(
        &mut self,
        completion: Option<Completion>,
        result: Result<Value>,
        generation: u64,
    ) {
        self.settle();
        match (completion, result) {
            (Some(c), result) => c.complete(result, generation),
            // A command posted without a completion has nobody to tell of
            // its failure but the log, as in Native.
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
    RenderFrame { needs_repaint: bool },
    StyleImage(String),
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
        Payload::RenderFrame { needs_repaint } => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_RENDER_FRAME;
            // SAFETY: plain C data.
            let mut frame: mln_runtime_event_render_frame = unsafe { std::mem::zeroed() };
            frame.mode = MLN_RENDER_MODE_FULL;
            frame.needs_repaint = *needs_repaint;
            event.payload.render_frame = frame;
        }
        Payload::StyleImage(id) => {
            event.payload_type = MLN_RUNTIME_EVENT_PAYLOAD_STYLE_IMAGE;
            event.payload.style_image = mln_runtime_event_style_image {
                image_id: crate::base::lend(id),
            };
        }
    }
    event
}

/// How long the thread waits for work while a map still loads or moves
/// without a frame to draw it: Orbis advances maps in steps.
const IDLE_STEP: Duration = Duration::from_millis(16);

fn run(rx: Receiver<Msg>) {
    loop {
        let busy = STATE.with(|s| s.borrow().as_ref().is_some_and(crate::map::any_busy));
        let first = if busy {
            match rx.recv_timeout(IDLE_STEP) {
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
                    let state = s.as_mut().expect("runtime state");
                    job(state);
                    state.settle();
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
                state.sessions.clear();
                state.maps.clear();
                state.settle();
                if let Some(handler) = state.handler.take() {
                    handler.release();
                }
            }
            if let Some(completion) = completion {
                completion.complete(Ok(Value::None), 0);
            }
            delivery::flush();
            return;
        }
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            let state = s.as_mut().expect("runtime state");
            crate::map::advance(state);
            crate::session::pump(state);
            state.settle();
        });
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
            if out_runtime.is_null() {
                return Err(Failure::argument("out_runtime is null"));
            }
            let mut state_dir = None;
            if !options.is_null() {
                let options = sized(options, "options")?;
                let cache = text(options.cache_path, "cache_path")?;
                text(options.asset_path, "asset_path")?;
                if !cache.is_empty() {
                    state_dir = Some(state_dir_for(&cache)?);
                }
            }
            let (tx, rx) = mpsc::channel();
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let join = std::thread::Builder::new()
                .name("orbis-runtime".into())
                .spawn(move || {
                    delivery::start_outbox();
                    let options = orb_api::RuntimeOptions {
                        state_dir,
                        ..Default::default()
                    };
                    match orb_api::Runtime::new(&options) {
                        Ok(runtime) => {
                            runtime.set_online(crate::network::online());
                            STATE.with(|s| {
                                *s.borrow_mut() = Some(State {
                                    runtime,
                                    id: 0,
                                    maps: HashMap::new(),
                                    sessions: HashMap::new(),
                                    handler: None,
                                    epoch: Instant::now(),
                                    events: Vec::new(),
                                })
                            });
                            let _ = ready_tx.send(Ok(()));
                            run(rx);
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(Failure::from(e)));
                        }
                    }
                })
                .map_err(|e| Failure::native(format!("can't start the runtime thread: {e}")))?;
            let thread = join.thread().id();
            ready_rx
                .recv()
                .map_err(|_| Failure::native("the runtime thread stopped"))??;
            let handle = Arc::new(RuntimeHandle {
                tx: Mutex::new(tx),
                thread,
                join: Mutex::new(Some(join)),
                maps: Default::default(),
            });
            let id = handles::insert(Object::Runtime(handle.clone()));
            crate::network::register(&handle);
            handle.post(move |state| state.id = id)?;
            put(out_runtime, mln_runtime(id), "out_runtime")
        })
    }
}

/// Where Orbis keeps a runtime's state for MapLibre Native's cache file:
/// a directory beside it, made if need be, so an unwritable place fails
/// the create as an unopenable database does.
fn state_dir_for(cache: &str) -> Result<std::path::PathBuf> {
    let path = std::path::PathBuf::from(format!("{cache}.orbis"));
    std::fs::create_dir_all(&path)
        .map_err(|e| Failure::argument(format!("can't open the cache at {cache}: {e}")))?;
    Ok(path)
}

fn stop(runtime: mln_runtime, completion: Option<Completion>, wait: bool) -> Result {
    let handle = handles::runtime(runtime.0)?;
    if handle.maps.load(std::sync::atomic::Ordering::Acquire) > 0 {
        return Err(Failure::state(
            "the runtime still has maps: release them first",
        ));
    }
    handles::remove(runtime.0);
    crate::network::unregister(&handle);
    let _ = handle.sender().send(Msg::Stop(completion));
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
                if let Some(old) = state.handler.replace(handler) {
                    old.release();
                }
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
                state.settle();
                if let Some(old) = state.handler.take() {
                    old.release();
                }
                state.complete(completion, Ok(Value::None), 0);
            })
        })
    }
}
