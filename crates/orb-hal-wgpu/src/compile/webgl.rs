//! Compiles on WebGL 2, on the page's thread under a time budget. WebGL 2
//! has no async pipeline creation, and each compile (wgpu translating to
//! GLSL, then the driver) takes milliseconds: jobs wait in a queue, front
//! first (a request the view needs moves there), and each poll compiles
//! them for a frame's share of time.

use std::collections::VecDeque;
use std::rc::Rc;

use orb_hal::PipelineId;

use super::{Built, Done, Job};

/// How long one poll compiles for, at least one pipeline: a frame's share,
/// so the map keeps drawing while the rest wait.
const BUDGET_MS: f64 = 6.0;

#[derive(Default)]
pub(crate) struct Compiler {
    queue: VecDeque<Job>,
    wake: std::cell::RefCell<Option<Rc<dyn Fn()>>>,
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler")
            .field("queued", &self.queue.len())
            .finish_non_exhaustive()
    }
}

impl Compiler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Calls `wake` while pipelines wait, so the map polls again.
    pub(crate) fn set_wake(&self, wake: Rc<dyn Fn()>) {
        *self.wake.borrow_mut() = Some(wake);
    }

    pub(crate) fn submit(&mut self, job: Job) {
        self.queue.push_back(job);
        self.wake_soon();
    }

    pub(crate) fn prioritize(&mut self, id: PipelineId) {
        if let Some(at) = self.queue.iter().position(|job| job.id() == id) {
            let job = self.queue.remove(at).expect("found above");
            self.queue.push_front(job);
        }
    }

    /// Compiles front first for [`BUDGET_MS`] when `compile`; otherwise
    /// nothing has finished.
    pub(crate) fn take(&mut self, device: &wgpu::Device, compile: bool) -> Vec<Done> {
        let mut done = Vec::new();
        if !compile {
            return done;
        }
        let start = js_sys::Date::now();
        while let Some(job) = self.queue.pop_front() {
            let began = js_sys::Date::now();
            let built = job.compile(device);
            let now = js_sys::Date::now();
            done.push(Done {
                built,
                label: job.label().to_owned(),
                took: std::time::Duration::from_secs_f64((now - began) / 1000.0),
            });
            if now - start >= BUDGET_MS {
                break;
            }
        }
        if !self.queue.is_empty() {
            self.wake_soon();
        }
        done
    }

    /// Wakes the map once the current task is done.
    fn wake_soon(&self) {
        if let Some(wake) = self.wake.borrow().clone() {
            let wake = wasm_bindgen::closure::Closure::once(move |_: wasm_bindgen::JsValue| wake());
            let _ = js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED).then(&wake);
            wake.forget();
        }
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
