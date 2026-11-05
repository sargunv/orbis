//! Pipelines compile off the thread that draws (plan §8.4, decision 0038).
//!
//! A pipeline request becomes a [`Job`] that owns everything it needs.
//! Natively, the host's job executor takes jobs from a queue, front first,
//! and makes the pipelines with wgpu, whose device is shared between
//! threads; a request the view needs moves to the front. Each task notes
//! when it started its job, so a compile that never returns (a driver hung
//! in its compiler) shows as [`Compiler::hung`]. In the browser, jobs compile with
//! `createRenderPipelineAsync` / `createComputePipelineAsync`; once the
//! promise resolves, wgpu makes the same pipeline, which the browser's
//! pipeline cache (Dawn's, keyed by content) returns without compiling it
//! again. wgpu has no async creation yet (gfx-rs/wgpu#3794).

use std::sync::OnceLock;

/// How modules are shared between requests: across compile threads
/// natively, on the one thread in the browser.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Shared<T> = std::sync::Arc<T>;
#[cfg(target_arch = "wasm32")]
pub(crate) type Shared<T> = std::rc::Rc<T>;

use orb_hal::*;

use crate::convert;

/// A shader module, made when the first pipeline that uses it compiles:
/// natively, on the worker (naga parses and validates there); in the
/// browser, when the pipeline is made.
#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) label: String,
    pub(crate) source: String,
    made: OnceLock<wgpu::ShaderModule>,
    /// The browser's module for async compiles.
    #[cfg(target_arch = "wasm32")]
    raw: std::cell::OnceCell<wasm_bindgen::JsValue>,
}

impl Module {
    pub(crate) fn new(label: &str, source: String) -> Shared<Self> {
        Shared::new(Self {
            label: label.to_owned(),
            source,
            made: OnceLock::new(),
            #[cfg(target_arch = "wasm32")]
            raw: Default::default(),
        })
    }

    fn get(&self, device: &wgpu::Device) -> &wgpu::ShaderModule {
        self.made.get_or_init(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&self.label),
                source: wgpu::ShaderSource::Wgsl(self.source.as_str().into()),
            })
        })
    }
}

#[derive(Debug)]
pub(crate) struct Stage {
    pub(crate) module: Shared<Module>,
    pub(crate) entry_point: String,
    pub(crate) constants: Vec<(String, f64)>,
}

impl Stage {
    pub(crate) fn new(stage: &ShaderStage<'_>, module: Shared<Module>) -> Self {
        Self {
            module,
            entry_point: stage.entry_point.to_owned(),
            constants: stage
                .constants
                .iter()
                .map(|&(k, v)| (k.to_owned(), v))
                .collect(),
        }
    }

    fn constants(&self) -> Vec<(&str, f64)> {
        self.constants
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect()
    }
}

/// A render pipeline request, owned.
#[derive(Debug)]
pub(crate) struct RenderJob {
    pub(crate) label: String,
    pub(crate) layout: wgpu::PipelineLayout,
    pub(crate) vertex: Stage,
    pub(crate) fragment: Stage,
    pub(crate) targets: Vec<ColorTargetState>,
    pub(crate) depth: Option<DepthState>,
    pub(crate) topology: PrimitiveTopology,
    pub(crate) cull_mode: Option<Face>,
    pub(crate) sample_count: u32,
    pub(crate) cache: Option<wgpu::PipelineCache>,
}

/// A compute pipeline request, owned.
#[derive(Debug)]
pub(crate) struct ComputeJob {
    pub(crate) label: String,
    pub(crate) layout: wgpu::PipelineLayout,
    pub(crate) compute: Stage,
    pub(crate) cache: Option<wgpu::PipelineCache>,
}

#[derive(Debug)]
pub(crate) enum Job {
    Render(u32, RenderJob),
    Compute(u32, ComputeJob),
}

/// A compiled pipeline for its slot.
#[derive(Debug)]
pub(crate) enum Built {
    Render(u32, wgpu::RenderPipeline),
    Compute(u32, wgpu::ComputePipeline),
}

/// A finished compile: the pipeline, or why it failed.
#[derive(Debug)]
pub(crate) struct Done {
    pub(crate) built: Result<Built, String>,
    pub(crate) label: String,
    pub(crate) took: std::time::Duration,
}

impl Job {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn id(&self) -> PipelineId {
        match self {
            Self::Render(slot, _) => PipelineId::Render(RenderPipelineId(*slot)),
            Self::Compute(slot, _) => PipelineId::Compute(ComputePipelineId(*slot)),
        }
    }

    pub(crate) fn label(&self) -> &str {
        match self {
            Self::Render(_, job) => &job.label,
            Self::Compute(_, job) => &job.label,
        }
    }

    /// Makes the pipeline with wgpu.
    fn build(&self, device: &wgpu::Device) -> Built {
        match self {
            Self::Render(slot, job) => Built::Render(*slot, job.build(device)),
            Self::Compute(slot, job) => {
                let constants = job.compute.constants();
                Built::Compute(
                    *slot,
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&job.label),
                        layout: Some(&job.layout),
                        module: job.compute.module.get(device),
                        entry_point: Some(&job.compute.entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &constants,
                            ..Default::default()
                        },
                        cache: job.cache.as_ref(),
                    }),
                )
            }
        }
    }

    /// Makes the pipeline, catching validation errors and panics.
    #[cfg(not(target_arch = "wasm32"))]
    fn compile(&self, device: &wgpu::Device) -> Result<Built, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.build(device)));
        let error = pollster::block_on(scope.pop());
        match (built, error) {
            (Ok(built), None) => Ok(built),
            (_, Some(error)) => Err(error.to_string()),
            (Err(panic), None) => Err(panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "panicked".into())),
        }
    }
}

impl Job {
    /// Makes the pipeline the browser compiled: errors already rejected
    /// its promise.
    #[cfg(target_arch = "wasm32")]
    fn compile(&self, device: &wgpu::Device) -> Result<Built, String> {
        Ok(self.build(device))
    }
}

impl RenderJob {
    fn build(&self, device: &wgpu::Device) -> wgpu::RenderPipeline {
        let targets: Vec<_> = self
            .targets
            .iter()
            .map(|t| Some(convert::color_target(t)))
            .collect();
        let vertex_constants = self.vertex.constants();
        let fragment_constants = self.fragment.constants();
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&self.label),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: self.vertex.module.get(device),
                entry_point: Some(&self.vertex.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &vertex_constants,
                    ..Default::default()
                },
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: self.cull_mode.map(|face| match face {
                    Face::Front => wgpu::Face::Front,
                    Face::Back => wgpu::Face::Back,
                }),
                topology: match self.topology {
                    PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
                    PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                    PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                },
                ..Default::default()
            },
            depth_stencil: self.depth.map(|d| wgpu::DepthStencilState {
                format: convert::texture_format(d.format),
                depth_write_enabled: Some(d.write),
                depth_compare: Some(convert::compare(d.compare)),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: self.fragment.module.get(device),
                entry_point: Some(&self.fragment.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &fragment_constants,
                    ..Default::default()
                },
                targets: &targets,
            }),
            multiview_mask: None,
            cache: self.cache.as_ref(),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::Compiler;
#[cfg(target_arch = "wasm32")]
pub(crate) use web::Compiler;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use orb_hal::PipelineId;
    use orb_jobs::Executor;

    use super::{Done, Job};

    /// Compiles on the host's job executor, front of the queue first: each
    /// request spawns one task, which compiles the job at the front when
    /// it runs.
    pub(crate) struct Compiler {
        shared: Arc<Shared>,
        done_tx: mpsc::Sender<Done>,
        done: mpsc::Receiver<Done>,
        /// Without one, each job compiles on the calling thread (tools
        /// that want every pipeline ready when it is asked for).
        executor: Option<Arc<dyn Executor>>,
        finished: VecDeque<Done>,
        watched: bool,
    }

    impl std::fmt::Debug for Compiler {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Compiler").finish_non_exhaustive()
        }
    }

    #[derive(Default)]
    struct Shared {
        queue: Mutex<Queue>,
        /// Fault injection: tasks stall in their job until the compiler
        /// closes, as in a driver hung in its compiler.
        hang: std::sync::atomic::AtomicBool,
        /// Called from a task when a compile finishes.
        compiled: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
        /// The jobs compiling and since when, by task.
        busy: Mutex<Vec<(u64, String, Instant)>>,
        /// Signalled when a task takes a job, and when the compiler
        /// closes: the watchdog and injected hangs wait on it.
        changed: Condvar,
        /// How long a compile may run before its driver is taken to have
        /// hung.
        limit: Mutex<Duration>,
    }

    #[derive(Default)]
    struct Queue {
        jobs: VecDeque<Job>,
        next_task: u64,
        closed: bool,
    }

    impl Compiler {
        pub(crate) fn new() -> Self {
            let (done_tx, done) = mpsc::channel();
            let shared = Shared {
                limit: Mutex::new(orb_hal::device::COMPILE_WATCHDOG),
                ..Default::default()
            };
            Self {
                shared: Arc::new(shared),
                done_tx,
                done,
                executor: None,
                finished: VecDeque::new(),
                watched: false,
            }
        }

        /// Compiles every later request on `executor`, or on the calling
        /// thread, before `create_*_pipeline` returns.
        pub(crate) fn set_executor(&mut self, executor: Option<Arc<dyn Executor>>) {
            self.executor = executor;
        }

        pub(crate) fn submit(&mut self, device: &wgpu::Device, job: Job) {
            let Some(executor) = &self.executor else {
                let started = Instant::now();
                let built = job.compile(device);
                self.finished.push_back(Done {
                    built,
                    label: job.label().to_owned(),
                    took: started.elapsed(),
                });
                return;
            };
            let task = {
                let mut queue = self.shared.queue.lock().expect("compile queue poisoned");
                queue.jobs.push_back(job);
                queue.next_task += 1;
                queue.next_task
            };
            if !self.watched {
                self.watched = true;
                let shared = self.shared.clone();
                std::thread::Builder::new()
                    .name("orbis pipeline watchdog".into())
                    .spawn(move || watchdog(&shared))
                    .expect("spawn the pipeline watchdog");
            }
            let (shared, device, done) =
                (self.shared.clone(), device.clone(), self.done_tx.clone());
            executor.spawn(Box::new(move || {
                compile_front(&shared, &device, task, &done)
            }));
        }

        /// How long a compile may run before its driver is taken to have
        /// hung.
        pub(crate) fn set_watchdog(&self, limit: Duration) {
            *self.shared.limit.lock().expect("compile watch poisoned") = limit;
        }

        /// Calls `wake` from a task whenever a compile finishes, and from
        /// the watchdog when one runs past its limit.
        pub(crate) fn set_wake(&self, wake: Arc<dyn Fn() + Send + Sync>) {
            *self.shared.compiled.lock().expect("compile wake poisoned") = Some(wake);
        }

        /// Makes tasks stall in their next job (fault injection for the
        /// watchdog's tests).
        pub(crate) fn inject_hang(&self) {
            self.shared
                .hang
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        /// Moves a queued job to the front.
        pub(crate) fn prioritize(&mut self, id: PipelineId) {
            let mut queue = self.shared.queue.lock().expect("compile queue poisoned");
            if let Some(at) = queue.jobs.iter().position(|j| j.id() == id)
                && at > 0
            {
                let job = queue.jobs.remove(at).expect("found");
                queue.jobs.push_front(job);
            }
        }

        /// Compiles finished since the last call.
        pub(crate) fn take(&mut self) -> impl Iterator<Item = Done> + '_ {
            self.finished.extend(self.done.try_iter());
            self.finished.drain(..)
        }

        /// Blocks until `pending` compiles have finished, for tools and
        /// tests.
        pub(crate) fn wait(&mut self, pending: u32) {
            while (self.finished.len() as u32) < pending {
                match self.done.recv() {
                    Ok(done) => self.finished.push_back(done),
                    Err(_) => return,
                }
            }
        }

        /// A compile that has run past the watchdog's limit: its label.
        pub(crate) fn hung(&self) -> Option<String> {
            let limit = *self.shared.limit.lock().expect("compile watch poisoned");
            let busy = self.shared.busy.lock().expect("compile watch poisoned");
            busy.iter()
                .find(|(.., since)| since.elapsed() > limit)
                .map(|(_, label, since)| {
                    format!("{label} ({:.1} s)", since.elapsed().as_secs_f64())
                })
        }
    }

    impl Shared {
        fn closed(&self) -> bool {
            self.queue.lock().expect("compile queue poisoned").closed
        }

        fn wake(&self) {
            let wake = self.compiled.lock().expect("compile wake poisoned").clone();
            if let Some(wake) = wake {
                wake();
            }
        }
    }

    impl Drop for Compiler {
        /// Drops queued jobs: their tasks find none. A task stuck in a
        /// driver stays stuck.
        fn drop(&mut self) {
            let mut queue = self.shared.queue.lock().expect("compile queue poisoned");
            queue.closed = true;
            queue.jobs.clear();
            drop(queue);
            let _busy = self.shared.busy.lock().expect("compile watch poisoned");
            self.shared.changed.notify_all();
        }
    }

    /// One request's task: compiles the job at the front of the queue.
    fn compile_front(shared: &Shared, device: &wgpu::Device, task: u64, done: &mpsc::Sender<Done>) {
        let Some(job) = shared
            .queue
            .lock()
            .expect("compile queue poisoned")
            .jobs
            .pop_front()
        else {
            return;
        };
        let started = Instant::now();
        let mut busy = shared.busy.lock().expect("compile watch poisoned");
        busy.push((task, job.label().to_owned(), started));
        shared.changed.notify_all();
        while shared.hang.load(std::sync::atomic::Ordering::Relaxed) && !shared.closed() {
            busy = shared.changed.wait(busy).expect("compile watch poisoned");
        }
        drop(busy);
        let built = job.compile(device);
        shared
            .busy
            .lock()
            .expect("compile watch poisoned")
            .retain(|(t, ..)| *t != task);
        let finished = Done {
            built,
            label: job.label().to_owned(),
            took: started.elapsed(),
        };
        if done.send(finished).is_ok() {
            shared.wake();
        }
    }

    /// Wakes the host once when a compile runs past the limit: a map that
    /// waits only for pipelines draws no frames, and the frame after the
    /// wake finds the compile hung. The device is lost then, so once is
    /// enough.
    fn watchdog(shared: &Shared) {
        let mut busy = shared.busy.lock().expect("compile watch poisoned");
        loop {
            if shared.closed() {
                return;
            }
            let limit = *shared.limit.lock().expect("compile watch poisoned");
            let Some(since) = busy.iter().map(|(.., since)| *since).min() else {
                busy = shared.changed.wait(busy).expect("compile watch poisoned");
                continue;
            };
            let left = limit.saturating_sub(since.elapsed());
            if left.is_zero() {
                drop(busy);
                shared.wake();
                return;
            }
            busy = shared
                .changed
                .wait_timeout(busy, left)
                .expect("compile watch poisoned")
                .0;
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use js_sys::{Array, Function, Object, Promise, Reflect};
    use orb_hal::*;
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};

    use super::{Done, Job, Module, RenderJob, Stage};

    /// A job whose promise settled (`Ok` resolved, `Err` the reason), and
    /// how long after its request, milliseconds.
    type Settled = (Job, Result<(), String>, f64);

    /// Compiles with the browser's async pipeline creation.
    #[derive(Debug, Default)]
    pub(crate) struct Compiler {
        settled: Rc<RefCell<VecDeque<Settled>>>,
        finished: VecDeque<Done>,
    }

    fn now() -> f64 {
        js_sys::Date::now()
    }

    impl Compiler {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        pub(crate) fn submit(&mut self, device: &wgpu::Device, layouts: Vec<JsValue>, job: Job) {
            let raw = device.as_webgpu().expect("the web uses the WebGPU backend");
            let raw: &JsValue = raw.as_ref();
            let layout = call(
                raw,
                "createPipelineLayout",
                &dict(&[("bindGroupLayouts", layouts.iter().collect::<Array>().into())]),
            );
            let (method, descriptor) = match &job {
                Job::Render(_, job) => ("createRenderPipelineAsync", render(raw, &layout, job)),
                Job::Compute(_, job) => (
                    "createComputePipelineAsync",
                    dict(&[
                        ("label", job.label.as_str().into()),
                        ("layout", layout.clone()),
                        ("compute", stage(raw, &job.compute, None)),
                    ]),
                ),
            };
            let promise: Promise = call(raw, method, &descriptor).unchecked_into();
            let started = now();
            let job = Rc::new(RefCell::new(Some(job)));
            let settled = self.settled.clone();
            let (ok_job, ok_settled) = (job.clone(), settled.clone());
            // The browser's pipeline stays alive until wgpu made its own:
            // its cache holds only live pipelines.
            let ok = Closure::once(move |pipeline: JsValue| {
                if let Some(job) = ok_job.borrow_mut().take() {
                    ok_settled
                        .borrow_mut()
                        .push_back((job, Ok(()), now() - started));
                }
                drop(pipeline);
            });
            let err = Closure::once(move |reason: JsValue| {
                if let Some(job) = job.borrow_mut().take() {
                    let message = Reflect::get(&reason, &"message".into())
                        .ok()
                        .and_then(|m| m.as_string())
                        .unwrap_or_else(|| format!("{reason:?}"));
                    settled
                        .borrow_mut()
                        .push_back((job, Err(message), now() - started));
                }
            });
            let _ = promise.then2(&ok, &err);
            ok.forget();
            err.forget();
        }

        pub(crate) fn prioritize(&mut self, _: PipelineId) {}

        /// Compiles finished since the last call: wgpu makes each
        /// pipeline the browser compiled.
        pub(crate) fn take(&mut self, device: &wgpu::Device) -> impl Iterator<Item = Done> + '_ {
            let settled: Vec<_> = self.settled.borrow_mut().drain(..).collect();
            for (job, result, ms) in settled {
                let built = result.and_then(|()| job.compile(device));
                self.finished.push_back(Done {
                    built,
                    label: job.label().to_owned(),
                    took: std::time::Duration::from_secs_f64(ms / 1000.0),
                });
            }
            self.finished.drain(..)
        }
    }

    /// A plain JS object of these fields.
    pub(crate) fn dict(fields: &[(&str, JsValue)]) -> JsValue {
        let object = Object::new();
        for (key, value) in fields {
            Reflect::set(&object, &(*key).into(), value).expect("set a field");
        }
        object.into()
    }

    /// `target.method(argument)`.
    pub(crate) fn call(target: &JsValue, method: &str, argument: &JsValue) -> JsValue {
        let function: Function = Reflect::get(target, &method.into())
            .expect("a GPUDevice method")
            .unchecked_into();
        function
            .call1(target, argument)
            .unwrap_or_else(|e| panic!("{method}: {e:?}"))
    }

    fn module(device: &JsValue, module: &Module) -> JsValue {
        module
            .raw
            .get_or_init(|| {
                call(
                    device,
                    "createShaderModule",
                    &dict(&[
                        ("label", module.label.as_str().into()),
                        ("code", module.source.as_str().into()),
                    ]),
                )
            })
            .clone()
    }

    fn stage(device: &JsValue, stage: &Stage, targets: Option<JsValue>) -> JsValue {
        let constants = Object::new();
        for (key, value) in &stage.constants {
            Reflect::set(&constants, &key.as_str().into(), &(*value).into()).expect("constant");
        }
        let mut fields = vec![
            ("module", module(device, &stage.module)),
            ("entryPoint", stage.entry_point.as_str().into()),
            ("constants", constants.into()),
        ];
        if let Some(targets) = targets {
            fields.push(("targets", targets));
        }
        dict(&fields)
    }

    /// wgpu's mapping of the same request, as a WebGPU dictionary: the
    /// browser's cache finds wgpu's pipeline equal to this one.
    fn render(device: &JsValue, layout: &JsValue, job: &RenderJob) -> JsValue {
        let component = |src: &str, dst: &str| {
            dict(&[
                ("srcFactor", src.into()),
                ("dstFactor", dst.into()),
                ("operation", "add".into()),
            ])
        };
        let targets: Array = job
            .targets
            .iter()
            .map(|t| {
                let mut fields = vec![("format", format(t.format).into())];
                match t.blend {
                    Some(BlendState::Additive) => fields.push((
                        "blend",
                        dict(&[
                            ("color", component("one", "one")),
                            ("alpha", component("one", "one")),
                        ]),
                    )),
                    Some(BlendState::PremultipliedAlpha) => fields.push((
                        "blend",
                        dict(&[
                            ("color", component("one", "one-minus-src-alpha")),
                            ("alpha", component("one", "one-minus-src-alpha")),
                        ]),
                    )),
                    Some(BlendState::Multiply) => fields.push((
                        "blend",
                        dict(&[
                            ("color", component("zero", "src")),
                            ("alpha", component("zero", "one")),
                        ]),
                    )),
                    None => {}
                }
                dict(&fields)
            })
            .collect();
        let mut primitive = vec![(
            "topology",
            match job.topology {
                PrimitiveTopology::LineList => "line-list",
                PrimitiveTopology::TriangleList => "triangle-list",
                PrimitiveTopology::TriangleStrip => "triangle-strip",
            }
            .into(),
        )];
        if let Some(face) = job.cull_mode {
            primitive.push((
                "cullMode",
                match face {
                    Face::Front => "front",
                    Face::Back => "back",
                }
                .into(),
            ));
        }
        let mut fields = vec![
            ("label", job.label.as_str().into()),
            ("layout", layout.clone()),
            ("vertex", stage(device, &job.vertex, None)),
            (
                "fragment",
                stage(device, &job.fragment, Some(targets.into())),
            ),
            ("primitive", dict(&primitive)),
            ("multisample", dict(&[("count", job.sample_count.into())])),
        ];
        if let Some(depth) = job.depth {
            fields.push((
                "depthStencil",
                dict(&[
                    ("format", format(depth.format).into()),
                    ("depthWriteEnabled", depth.write.into()),
                    (
                        "depthCompare",
                        match depth.compare {
                            CompareFunction::Always => "always",
                            CompareFunction::Less => "less",
                            CompareFunction::LessEqual => "less-equal",
                            CompareFunction::Greater => "greater",
                            CompareFunction::GreaterEqual => "greater-equal",
                        }
                        .into(),
                    ),
                ]),
            ));
        }
        dict(&fields)
    }

    pub(crate) fn format(format: TextureFormat) -> &'static str {
        match format {
            TextureFormat::R8Unorm => "r8unorm",
            TextureFormat::Rgba16Float => "rgba16float",
            TextureFormat::Rgba8Unorm => "rgba8unorm",
            TextureFormat::Rgba8UnormSrgb => "rgba8unorm-srgb",
            TextureFormat::Bgra8Unorm => "bgra8unorm",
            TextureFormat::Bgra8UnormSrgb => "bgra8unorm-srgb",
            TextureFormat::R32Float => "r32float",
            TextureFormat::Depth32Float => "depth32float",
        }
    }

    /// A bind group layout as a WebGPU dictionary, as wgpu maps it.
    pub(crate) fn bind_group_layout(device: &JsValue, desc: &BindGroupLayoutDesc<'_>) -> JsValue {
        let dimension = |d: TextureViewDimension| match d {
            TextureViewDimension::D2 => "2d",
            TextureViewDimension::D2Array => "2d-array",
        };
        let entries: Array = desc
            .entries
            .iter()
            .map(|entry| {
                let (key, value) = match entry.ty {
                    BindingType::Uniform => ("buffer", dict(&[("type", "uniform".into())])),
                    BindingType::Storage { read_only } => (
                        "buffer",
                        dict(&[(
                            "type",
                            if read_only {
                                "read-only-storage"
                            } else {
                                "storage"
                            }
                            .into(),
                        )]),
                    ),
                    BindingType::Texture {
                        sample,
                        dimension: d,
                    } => (
                        "texture",
                        dict(&[
                            (
                                "sampleType",
                                match sample {
                                    TextureSampleType::Float { filterable: true } => "float",
                                    TextureSampleType::Float { filterable: false } => {
                                        "unfilterable-float"
                                    }
                                    TextureSampleType::Uint => "uint",
                                    TextureSampleType::Depth
                                    | TextureSampleType::DepthMultisampled => "depth",
                                }
                                .into(),
                            ),
                            ("viewDimension", dimension(d).into()),
                            (
                                "multisampled",
                                (sample == TextureSampleType::DepthMultisampled).into(),
                            ),
                        ]),
                    ),
                    BindingType::Sampler { filtering } => (
                        "sampler",
                        dict(&[(
                            "type",
                            if filtering {
                                "filtering"
                            } else {
                                "non-filtering"
                            }
                            .into(),
                        )]),
                    ),
                    BindingType::ComparisonSampler => {
                        ("sampler", dict(&[("type", "comparison".into())]))
                    }
                    BindingType::StorageTexture {
                        format: f,
                        dimension: d,
                    } => (
                        "storageTexture",
                        dict(&[
                            ("access", "write-only".into()),
                            ("format", format(f).into()),
                            ("viewDimension", dimension(d).into()),
                        ]),
                    ),
                };
                dict(&[
                    ("binding", entry.binding.into()),
                    ("visibility", entry.visibility.bits().into()),
                    (key, value),
                ])
            })
            .collect();
        call(
            device,
            "createBindGroupLayout",
            &dict(&[("label", desc.label.into()), ("entries", entries.into())]),
        )
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::bind_group_layout as raw_bind_group_layout;
