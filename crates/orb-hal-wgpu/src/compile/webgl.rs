//! Compiles on WebGL 2, on the page's thread. WebGL 2 has no async
//! pipeline creation, and each compile (wgpu translating to GLSL, then
//! the driver) takes milliseconds: jobs wait in a queue, front first (a
//! request the view needs moves there), and each poll starts them for a
//! frame's share of time. Where the browser links programs in the
//! background (`KHR_parallel_shader_compile`), wgpu leaves each link to
//! finish there and polls say when it did; elsewhere starting a compile
//! waits for its link. Background jobs compile in the page's idle time, a
//! few links at a time.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use orb_hal::PipelineId;
use wasm_bindgen::{JsCast, JsValue, closure::Closure};

use super::{Built, Done, Job};

/// How long one poll starts compiles for, at least one: a frame's share,
/// so the map keeps drawing while the rest wait.
const BUDGET_MS: f64 = 6.0;

/// The most an idle callback starts compiles for, at least one.
const IDLE_BUDGET_MS: f64 = 4.0;

/// Background links in flight at once, where they link in the background:
/// a link a frame asks for later waits behind those the browser has.
const BACKGROUND_LANES: usize = 2;

type Wake = Rc<RefCell<Option<Rc<dyn Fn()>>>>;

#[derive(Default)]
pub(crate) struct Compiler {
    state: Rc<RefCell<State>>,
    wake: Wake,
}

#[derive(Default)]
struct State {
    /// Waiting for a frame's poll, front first.
    queue: VecDeque<Job>,
    /// Background jobs waiting for idle time, front first.
    background: VecDeque<Job>,
    /// Pipelines made whose programs are still linking.
    linking: Vec<Linking>,
    /// Background compiles finished in idle time, for the next take.
    finished: Vec<Done>,
    /// The device, once a job came, and whether its links are deferred.
    device: Option<(wgpu::Device, bool)>,
    /// An idle callback is scheduled.
    idle: bool,
}

struct Linking {
    id: PipelineId,
    label: String,
    built: Built,
    started: f64,
    background: bool,
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.borrow();
        f.debug_struct("Compiler")
            .field("queued", &state.queue.len())
            .field("background", &state.background.len())
            .field("linking", &state.linking.len())
            .finish_non_exhaustive()
    }
}

fn now() -> f64 {
    js_sys::Date::now()
}

impl Compiler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Calls `wake` while pipelines a frame waits for compile, so the map
    /// polls again.
    pub(crate) fn set_wake(&self, wake: Rc<dyn Fn()>) {
        *self.wake.borrow_mut() = Some(wake);
    }

    pub(crate) fn submit(&mut self, device: &wgpu::Device, job: Job, background: bool) {
        let mut state = self.state.borrow_mut();
        state
            .device
            .get_or_insert_with(|| (device.clone(), defer_links(device)));
        if background {
            state.background.push_back(job);
            drop(state);
            schedule_idle(&self.state, &self.wake);
        } else {
            state.queue.push_back(job);
            drop(state);
            self.wake_soon();
        }
    }

    pub(crate) fn prioritize(&mut self, id: PipelineId) {
        let mut state = self.state.borrow_mut();
        if let Some(at) = state.queue.iter().position(|job| job.id() == id) {
            let job = state.queue.remove(at).expect("found above");
            state.queue.push_front(job);
        }
    }

    /// Makes a background compile one a frame waits for: compiled first
    /// if it waits, polled with the frame's if it links.
    pub(crate) fn promote(&mut self, id: PipelineId) {
        let mut state = self.state.borrow_mut();
        if let Some(at) = state.background.iter().position(|job| job.id() == id) {
            let job = state.background.remove(at).expect("found above");
            state.queue.push_front(job);
        } else if let Some(link) = state.linking.iter_mut().find(|l| l.id == id) {
            link.background = false;
        }
        drop(state);
        self.wake_soon();
    }

    /// Compiles finished since the last call; `compile` (a frame's poll)
    /// starts the next for [`BUDGET_MS`] and takes the links that
    /// finished first.
    pub(crate) fn take(&mut self, device: &wgpu::Device, compile: bool) -> Vec<Done> {
        let mut state = self.state.borrow_mut();
        let mut done = std::mem::take(&mut state.finished);
        if !compile {
            return done;
        }
        let deferred = state.device.as_ref().is_some_and(|(_, d)| *d);
        let start = now();
        while let Some(job) = state.queue.pop_front() {
            start_compile(&mut state, device, deferred, job, false, &mut done);
            if now() - start >= BUDGET_MS {
                break;
            }
        }
        poll_links(&mut state, device, &mut done, |l| !l.background);
        let waiting = !state.queue.is_empty() || state.linking.iter().any(|l| !l.background);
        drop(state);
        if waiting {
            self.wake_soon();
        }
        done
    }

    /// Wakes the map once the current task is done.
    fn wake_soon(&self) {
        if let Some(wake) = self.wake.borrow().clone() {
            let wake = Closure::once(move |_: JsValue| wake());
            let _ = js_sys::Promise::resolve(&JsValue::UNDEFINED).then(&wake);
            wake.forget();
        }
    }
}

/// Makes `job`'s pipeline: done, or (links `deferred`) linking.
fn start_compile(
    state: &mut State,
    device: &wgpu::Device,
    deferred: bool,
    job: Job,
    background: bool,
    done: &mut Vec<Done>,
) {
    let began = now();
    let (id, label) = (job.id(), job.label().to_owned());
    match job.compile(device) {
        Ok(built @ Built::Render(..)) if deferred => state.linking.push(Linking {
            id,
            label,
            built,
            started: began,
            background,
        }),
        built => done.push(Done {
            id,
            built,
            label,
            took: std::time::Duration::from_secs_f64((now() - began) / 1000.0),
        }),
    }
}

/// Takes the links `which` picks that finished.
fn poll_links(
    state: &mut State,
    device: &wgpu::Device,
    done: &mut Vec<Done>,
    which: impl Fn(&Linking) -> bool,
) {
    let mut at = 0;
    while at < state.linking.len() {
        let link = &state.linking[at];
        let polled = match &link.built {
            Built::Render(_, pipeline) if which(link) => poll_link(device, pipeline),
            _ => None,
        };
        let Some(result) = polled else {
            at += 1;
            continue;
        };
        let link = state.linking.swap_remove(at);
        done.push(Done {
            id: link.id,
            built: result.map(|()| link.built),
            label: link.label,
            took: std::time::Duration::from_secs_f64((now() - link.started) / 1000.0),
        });
    }
}

/// Defers the links of `device`'s later render pipelines, where programs
/// link in the background: whether they are.
#[allow(unsafe_code)]
fn defer_links(device: &wgpu::Device) -> bool {
    // SAFETY: wgpu's GL device only sets a flag its pipeline creation
    // reads.
    unsafe { device.as_hal::<wgpu::hal::api::Gles>() }.is_some_and(|d| d.defer_links(true))
}

/// `None` while `pipeline`'s program links, then whether it did.
#[allow(unsafe_code)]
fn poll_link(device: &wgpu::Device, pipeline: &wgpu::RenderPipeline) -> Option<Result<(), String>> {
    // SAFETY: `pipeline` is `device`'s, and neither is destroyed through
    // the guards.
    let device = unsafe { device.as_hal::<wgpu::hal::api::Gles>() }?;
    match unsafe { pipeline.as_hal::<wgpu::hal::api::Gles>() } {
        Some(pipeline) => unsafe { device.poll_link(&pipeline) },
        None => Some(Ok(())),
    }
}

/// Runs [`idle_step`] in the page's next idle time, unless it is
/// scheduled.
fn schedule_idle(state: &Rc<RefCell<State>>, wake: &Wake) {
    if std::mem::replace(&mut state.borrow_mut().idle, true) {
        return;
    }
    let (shared, wake) = (state.clone(), wake.clone());
    let step = Closure::once_into_js(move |deadline: JsValue| idle_step(&shared, &wake, &deadline));
    let global = js_sys::global();
    let idle = js_sys::Reflect::get(&global, &"requestIdleCallback".into())
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
    let scheduled = match idle {
        Some(request) => request.call1(&global, &step),
        // Workers have no idle callbacks: a task after the frame's.
        None => js_sys::Reflect::get(&global, &"setTimeout".into())
            .and_then(|f| f.dyn_into::<js_sys::Function>())
            .and_then(|set| set.call2(&global, &step, &16.into())),
    };
    if scheduled.is_err() {
        state.borrow_mut().idle = false;
    }
}

/// Takes the background links that finished and starts the next
/// background compiles while lanes are free (one at a time where links
/// wait), once no compile a frame waits for is left.
fn idle_step(state: &Rc<RefCell<State>>, wake: &Wake, deadline: &JsValue) {
    let mut s = state.borrow_mut();
    s.idle = false;
    let Some((device, deferred)) = s.device.clone() else {
        return;
    };
    let mut done = Vec::new();
    poll_links(&mut s, &device, &mut done, |l| l.background);
    let frame_waits = !s.queue.is_empty() || s.linking.iter().any(|l| !l.background);
    if !frame_waits {
        let left = js_sys::Reflect::get(deadline, &"timeRemaining".into())
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .and_then(|f| f.call0(deadline).ok())
            .and_then(|v| v.as_f64())
            .unwrap_or(IDLE_BUDGET_MS)
            .min(IDLE_BUDGET_MS);
        let start = now();
        let lanes = if deferred { BACKGROUND_LANES } else { 1 };
        while s.linking.iter().filter(|l| l.background).count() < lanes
            && let Some(job) = s.background.pop_front()
        {
            start_compile(&mut s, &device, deferred, job, true, &mut done);
            if !deferred || now() - start >= left {
                break;
            }
        }
    }
    s.finished.extend(done);
    let more = !s.background.is_empty() || s.linking.iter().any(|l| l.background);
    drop(s);
    if more {
        schedule_idle(state, wake);
    }
}

impl Job {
    /// Makes the pipeline, catching validation errors: wgpu's GL backend
    /// validates on this thread, so its error scope is ready at once.
    fn compile(&self, device: &wgpu::Device) -> Result<Built, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = self.build(device);
        let mut scope = std::pin::pin!(scope.pop());
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(scope.as_mut(), &mut context) {
            std::task::Poll::Ready(Some(error)) => Err(error.to_string()),
            _ => Ok(built),
        }
    }
}
