//! Pipelines compile off the thread that draws.
//!
//! A pipeline request becomes a [`Job`] that owns everything it needs.
//! Natively, the host's job executor takes jobs from a queue, front first,
//! and makes the pipelines with wgpu, whose device is shared between
//! threads; a request the view needs moves to the front. Each task notes
//! when it started its job, so a compile that never returns (a driver hung
//! in its compiler) shows as [`Compiler::hung`]. On WebGPU, jobs compile with
//! `createRenderPipelineAsync` / `createComputePipelineAsync`, and wgpu
//! wraps the pipeline the promise resolves to
//! (`Device::create_render_pipeline_from_webgpu_handle`, the fork's, #35):
//! wgpu has no async creation of its own (gfx-rs/wgpu#3794), and making the
//! pipeline again through wgpu compiled it a second time, synchronously, in
//! a browser without a content-keyed pipeline cache (Firefox). WebGL 2's:
//! `webgl`.

#[cfg(not(webgpu))]
use std::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

/// How modules are shared between requests: across compile threads
/// natively, on the one thread in the browser.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Shared<T> = std::sync::Arc<T>;
#[cfg(target_arch = "wasm32")]
pub(crate) type Shared<T> = std::rc::Rc<T>;

use orb_hal::*;

#[cfg(not(webgpu))]
use crate::convert;

/// A shader module, made when the first pipeline that uses it compiles:
/// natively, on the worker (naga parses and validates there); in the
/// browser, when the pipeline is made.
#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) label: String,
    #[cfg(not(webgl))]
    pub(crate) source: String,
    /// Natively, `source` as one naga module per entry point,
    /// postcard-encoded, and each made when a stage first uses it.
    #[cfg(not(target_arch = "wasm32"))]
    entries: Vec<EntryModule>,
    /// On a GL ES device, the build's GLSL of its stages, and whether the
    /// device clips depth from 0 to 1: each stage is a module of its own,
    /// its constants written in, which wgpu compiles as it is.
    #[cfg(not(webgpu))]
    glsl: Option<(Glsl, bool)>,
    #[cfg(not(target_arch = "wasm32"))]
    made: OnceLock<wgpu::ShaderModule>,
    /// The browser's module for async compiles.
    #[cfg(webgpu)]
    raw: std::cell::OnceCell<wasm_bindgen::JsValue>,
}

impl Module {
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    pub(crate) fn new(
        label: &str,
        source: String,
        ir: Option<IrModules>,
        glsl: Option<(Glsl, bool)>,
    ) -> Shared<Self> {
        Shared::new(Self {
            label: label.to_owned(),
            #[cfg(not(webgl))]
            source,
            #[cfg(not(target_arch = "wasm32"))]
            entries: (ir.into_iter().flatten())
                .map(|&(entry, bytes)| EntryModule {
                    entry,
                    bytes,
                    made: OnceLock::new(),
                })
                .collect(),
            #[cfg(not(webgpu))]
            glsl,
            #[cfg(not(target_arch = "wasm32"))]
            made: OnceLock::new(),
            #[cfg(webgpu)]
            raw: Default::default(),
        })
    }

    /// The module for a stage at `entry`: natively the build's module of
    /// that entry point when there is one, which skips parsing the WGSL
    /// (most of a module's cost) and wgpu's split of a module per stage;
    /// the WGSL's otherwise.
    #[cfg(not(target_arch = "wasm32"))]
    fn get(&self, device: &wgpu::Device, entry: &str) -> &wgpu::ShaderModule {
        if let Some(one) = self.entries.iter().find(|e| e.entry == entry)
            && let Some(module) = one
                .made
                .get_or_init(|| {
                    let module = postcard::from_bytes::<naga::Module>(one.bytes).ok()?;
                    Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&self.label),
                        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                    }))
                })
                .as_ref()
        {
            return module;
        }
        self.made.get_or_init(|| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&self.label),
                source: wgpu::ShaderSource::Wgsl(self.source.as_str().into()),
            })
        })
    }
}

/// A shader's modules by entry point, postcard-encoded
/// ([`ShaderModuleDesc::ir`](orb_hal::ShaderModuleDesc::ir)).
pub(crate) type IrModules = &'static [(&'static str, &'static [u8])];

/// One entry point's module, made when a stage first uses it (`None`
/// if it doesn't decode: the WGSL's is made instead).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
struct EntryModule {
    entry: &'static str,
    bytes: &'static [u8],
    made: OnceLock<Option<wgpu::ShaderModule>>,
}

/// A stage's module, and the override constants its pipeline sets.
#[cfg(not(webgpu))]
type StageModule<'a> = (Cow<'a, wgpu::ShaderModule>, Vec<(&'a str, f64)>);

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

    /// Its module, and the override constants the pipeline sets: none for
    /// the build's GLSL, which has them written in.
    #[cfg(not(webgpu))]
    fn module(&self, device: &wgpu::Device) -> Result<StageModule<'_>, String> {
        let constants: Vec<(&str, f64)> = (self.constants.iter())
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        #[cfg(not(webgpu))]
        if let Module {
            label,
            glsl: Some((glsl, zero_to_one_depth)),
            ..
        } = &*self.module
        {
            let (entry, zero_to_one_depth) = (&self.entry_point, *zero_to_one_depth);
            let source = glsl(label, entry, &constants, zero_to_one_depth)
                .ok_or_else(|| format!("{label}: no GLSL of {entry} with {constants:?}"))?;
            // SAFETY: the build wrote and validated the GLSL as wgpu's GL
            // back end writes it, its bindings named for the pipeline's
            // layout (orb-shaders' build.rs).
            #[allow(unsafe_code)]
            let module = unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some(label),
                    entry_points: Cow::Borrowed(&[wgpu::PassthroughShaderEntryPoint {
                        name: Cow::Borrowed(entry),
                        workgroup_size: (0, 0, 0),
                    }]),
                    glsl: Some(source.into()),
                    ..Default::default()
                })
            };
            return Ok((Cow::Owned(module), Vec::new()));
        }
        #[cfg(webgl)]
        return Err(format!(
            "{}: WebGL 2 compiles the build's GLSL alone",
            self.module.label
        ));
        #[cfg(not(target_arch = "wasm32"))]
        Ok((
            Cow::Borrowed(self.module.get(device, &self.entry_point)),
            constants,
        ))
    }
}

/// A render pipeline request, owned.
#[derive(Debug)]
pub(crate) struct RenderJob {
    pub(crate) label: String,
    /// wgpu's layout. WebGPU's async call takes the browser's, made from
    /// the bind group layouts' own handles.
    #[cfg(not(webgpu))]
    pub(crate) layout: wgpu::PipelineLayout,
    pub(crate) vertex: Stage,
    pub(crate) fragment: Stage,
    pub(crate) targets: Vec<ColorTargetState>,
    pub(crate) depth: Option<DepthState>,
    pub(crate) topology: PrimitiveTopology,
    pub(crate) cull_back: bool,
    pub(crate) sample_count: u32,
    /// Whether the vertex stage reads WebGL 2's attribute zero
    /// ([`Caps::gl_shaders`]).
    #[cfg(not(webgpu))]
    pub(crate) attribute_zero: bool,
    #[cfg(not(webgpu))]
    pub(crate) cache: Option<wgpu::PipelineCache>,
}

/// WebGL 2's attribute zero: a `u32` an instance (#179).
#[cfg(not(webgpu))]
const ATTRIBUTE_ZERO: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 4,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: 0,
    }],
};

/// A compute pipeline request, owned.
#[derive(Debug)]
pub(crate) struct ComputeJob {
    pub(crate) label: String,
    #[cfg(not(webgpu))]
    pub(crate) layout: wgpu::PipelineLayout,
    pub(crate) compute: Stage,
    #[cfg(not(webgpu))]
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
    pub(crate) id: PipelineId,
    pub(crate) built: Result<Built, String>,
    pub(crate) label: String,
    pub(crate) took: std::time::Duration,
}

impl Job {
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
    #[cfg(not(webgpu))]
    fn build(&self, device: &wgpu::Device) -> Result<Built, String> {
        #[cfg(not(target_arch = "wasm32"))]
        dump::job(self);
        Ok(match self {
            Self::Render(slot, job) => Built::Render(*slot, job.build(device)?),
            Self::Compute(slot, job) => {
                let (module, constants) = job.compute.module(device)?;
                Built::Compute(
                    *slot,
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&job.label),
                        layout: Some(&job.layout),
                        module: &module,
                        entry_point: Some(&job.compute.entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &constants,
                            ..Default::default()
                        },
                        cache: job.cache.as_ref(),
                    }),
                )
            }
        })
    }

    /// Makes the pipeline, catching validation and internal errors and
    /// panics, and a device lost in the making: a driver that fails to compile a
    /// pipeline (Vulkan's `VK_ERROR_INITIALIZATION_FAILED`, which wgpu
    /// takes as "driver implementation is at fault") loses the device
    /// without an error to catch, and would on any other device too.
    #[cfg(not(target_arch = "wasm32"))]
    fn compile(&self, device: &wgpu::Device, loss: Option<&Loss>) -> Result<Built, String> {
        // wgpu loses the device in the call that failed, so a loss this
        // thread saw while it built is this pipeline's; one from elsewhere
        // (another compile, `WgpuGpu::lose`) isn't.
        let lost_here =
            || loss.and_then(|l| *l.thread.lock().ok()?) == Some(std::thread::current().id());
        let lost_before = lost_here();
        // A driver that fails to compile or link a shader (GLES) is an
        // internal error.
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.build(device)));
        // Popped innermost first; both must be.
        let invalid = pollster::block_on(validation.pop());
        let error = invalid.or(pollster::block_on(internal.pop()));
        match (built, error) {
            (Ok(_), None) if !lost_before && lost_here() => {
                let reason = (loss.and_then(|l| l.reason.lock().ok()?.clone()))
                    .map(|r| format!(" ({r})"))
                    .unwrap_or_default();
                Err(format!("the device was lost making it{reason}"))
            }
            (Ok(built), None) => built,
            (_, Some(error)) => Err(error.to_string()),
            (Err(panic), None) => Err(panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "panicked".into())),
        }
    }
}

#[cfg(not(webgpu))]
impl RenderJob {
    fn build(&self, device: &wgpu::Device) -> Result<wgpu::RenderPipeline, String> {
        let targets: Vec<_> = self
            .targets
            .iter()
            .map(|&t| Some(convert::color_target(t)))
            .collect();
        let (vertex, vertex_constants) = self.vertex.module(device)?;
        let (fragment, fragment_constants) = self.fragment.module(device)?;
        Ok(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&self.label),
                layout: Some(&self.layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some(&self.vertex.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &vertex_constants,
                        ..Default::default()
                    },
                    buffers: if self.attribute_zero {
                        &[Some(ATTRIBUTE_ZERO)]
                    } else {
                        &[]
                    },
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: self.cull_back.then_some(wgpu::Face::Back),
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
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some(&self.fragment.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &fragment_constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: self.cache.as_ref(),
            }),
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::Compiler;

/// Where `WgpuGpu` records its device's loss, which compiles watch.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct Loss {
    /// The thread wgpu told of the loss.
    pub(crate) thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::ThreadId>>>,
    /// What the driver said.
    pub(crate) reason: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}
#[cfg(webgpu)]
pub(crate) use webgpu::Compiler;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex, mpsc};
    use std::time::{Duration, Instant};

    use orb_hal::PipelineId;
    use orb_host::jobs::Executor;

    use super::{Done, Job, Loss};

    /// Background compiles running at once: each runs as a task of its
    /// own, queued behind whatever the executor has, so tiles keep
    /// flowing while a style's pipelines compile ahead of need.
    const BACKGROUND_LANES: usize = 2;

    /// How long one pipeline may compile before its driver is taken to
    /// have hung in its compiler: the device is then lost, and the runtime
    /// reports the hang rather than make another. Set from the slowest cold
    /// compile on an Adreno 619 with a wide margin.
    pub(crate) const WATCHDOG: Duration = Duration::from_secs(60);

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
        /// Fault injection: compiles of pipelines of this label take this
        /// much longer, as on a device compiling cold.
        slow: Mutex<Option<(String, Duration)>>,
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
        /// The device's loss, once watched.
        loss: std::sync::OnceLock<Loss>,
        /// How many pipelines may compile at once ([`COMPILES`]).
        at_once: usize,
        background: Mutex<Background>,
        /// Background compiles may start ([`Compiler::open_background`]).
        open: std::sync::atomic::AtomicBool,
    }

    /// Background compiles ([`Compiler::submit_background`]).
    #[derive(Default)]
    struct Background {
        /// Waiting, front first.
        jobs: VecDeque<Job>,
        /// Compiling now.
        running: usize,
        /// Background jobs waiting or compiling that no frame asked for:
        /// their compiles don't wake the host.
        quiet: std::collections::HashSet<PipelineId>,
    }

    #[derive(Default)]
    struct Queue {
        jobs: VecDeque<Job>,
        next_task: u64,
        closed: bool,
        /// Tasks compiling, each taking jobs from the front until none
        /// are left.
        compiling: usize,
    }

    /// Caps how many pipelines compile at once (by default as many as the
    /// host's job threads run): `1` makes them one at a time, for drivers
    /// suspected of failing compiles that overlap (Qualcomm's, #141).
    const COMPILES: &str = "ORB_COMPILES_AT_ONCE";

    impl Compiler {
        pub(crate) fn new() -> Self {
            let (done_tx, done) = mpsc::channel();
            let at_once = (std::env::var(COMPILES).ok())
                .and_then(|n| n.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(usize::MAX);
            let shared = Shared {
                limit: Mutex::new(WATCHDOG),
                at_once,
                open: true.into(),
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

        /// Takes a device lost during a compile as that pipeline's
        /// failure.
        pub(crate) fn watch_loss(&self, loss: Loss) {
            let _ = self.shared.loss.set(loss);
        }

        pub(crate) fn submit(&mut self, device: &wgpu::Device, job: Job) {
            self.submit_at(device, job, false);
        }

        fn submit_at(&mut self, device: &wgpu::Device, job: Job, front: bool) {
            let Some(executor) = self.executor.clone() else {
                let started = Instant::now();
                let built = job.compile(device, self.shared.loss.get());
                self.finished.push_back(Done {
                    id: job.id(),
                    built,
                    label: job.label().to_owned(),
                    took: started.elapsed(),
                });
                return;
            };
            let task = {
                let mut queue = self.shared.queue.lock().expect("compile queue poisoned");
                if front {
                    queue.jobs.push_front(job);
                } else {
                    queue.jobs.push_back(job);
                }
                queue.next_task += 1;
                queue.next_task
            };
            self.watch();
            let (shared, device, done) =
                (self.shared.clone(), device.clone(), self.done_tx.clone());
            executor.spawn(Box::new(move || {
                compile_front(&shared, &device, task, &done)
            }));
        }

        fn watch(&mut self) {
            if !self.watched {
                self.watched = true;
                let shared = self.shared.clone();
                std::thread::Builder::new()
                    .name("orbis pipeline watchdog".into())
                    .spawn(move || watchdog(&shared))
                    .expect("spawn the pipeline watchdog");
            }
        }

        /// Queues a background compile: [`BACKGROUND_LANES`] at a time,
        /// each a task behind what the executor has queued. Without an
        /// executor, or under a cap on compiles at once ([`COMPILES`]), it
        /// waits until [`promote`](Self::promote)d.
        pub(crate) fn submit_background(&mut self, device: &wgpu::Device, job: Job) {
            {
                let mut background = self.shared.background.lock().expect("compile poisoned");
                background.quiet.insert(job.id());
                background.jobs.push_back(job);
            }
            if let Some(executor) = self.executor.clone() {
                self.watch();
                let lane = Lane {
                    shared: self.shared.clone(),
                    device: device.clone(),
                    done: self.done_tx.clone(),
                    executor,
                };
                lane.feed();
            }
        }

        /// Lets background compiles start (`open`), or holds those that
        /// haven't: lanes finish the compile they run.
        pub(crate) fn open_background(&mut self, device: &wgpu::Device, open: bool) {
            use std::sync::atomic::Ordering;
            let was = self.shared.open.swap(open, Ordering::Relaxed);
            if open
                && !was
                && let Some(executor) = self.executor.clone()
            {
                for _ in 0..BACKGROUND_LANES {
                    let lane = Lane {
                        shared: self.shared.clone(),
                        device: device.clone(),
                        done: self.done_tx.clone(),
                        executor: executor.clone(),
                    };
                    lane.feed();
                }
            }
        }

        /// Makes a background compile an ordinary one, compiled next: a
        /// frame waits for it.
        pub(crate) fn promote(&mut self, device: &wgpu::Device, id: PipelineId) {
            let job = {
                let mut background = self.shared.background.lock().expect("compile poisoned");
                // Its compile, finishing now, wakes the host.
                background.quiet.remove(&id);
                let at = background.jobs.iter().position(|j| j.id() == id);
                at.and_then(|at| background.jobs.remove(at))
            };
            if let Some(job) = job {
                self.submit_at(device, job, true);
            }
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

        /// Makes compiles of pipelines labelled `label` take `by` longer
        /// (fault injection for tests of what waits for them).
        pub(crate) fn inject_slow(&self, label: &str, by: Duration) {
            *self.shared.slow.lock().expect("compile watch poisoned") = Some((label.into(), by));
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

        /// Blocks until a compile has finished that [`take`](Self::take)
        /// hasn't taken, for tools and tests.
        pub(crate) fn wait(&mut self) {
            if self.finished.is_empty()
                && let Ok(done) = self.done.recv()
            {
                self.finished.push_back(done);
            }
        }

        /// Waits up to `limit` for the compiles running now to finish:
        /// after a loss, so one that lost the device reports as failed.
        pub(crate) fn settle(&self, limit: Duration) {
            let deadline = Instant::now() + limit;
            let mut busy = self.shared.busy.lock().expect("compile watch poisoned");
            while !busy.is_empty() {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return;
                }
                busy = (self.shared.changed.wait_timeout(busy, left))
                    .expect("compile watch poisoned")
                    .0;
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
            (self.shared.background.lock().expect("compile poisoned"))
                .jobs
                .clear();
            let _busy = self.shared.busy.lock().expect("compile watch poisoned");
            self.shared.changed.notify_all();
        }
    }

    /// Background compiles' tasks: each takes the next background job and,
    /// once compiled, queues the task for the one after.
    struct Lane {
        shared: Arc<Shared>,
        device: wgpu::Device,
        done: mpsc::Sender<Done>,
        executor: Arc<dyn Executor>,
    }

    impl Lane {
        /// Starts a task for the next waiting job if a lane is free and
        /// background compiles may start.
        fn feed(self) {
            if !self.shared.open.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            {
                let mut background = self.shared.background.lock().expect("compile poisoned");
                // Under a cap on compiles at once ([`COMPILES`]), they
                // wait until a frame asks for them.
                let lanes = match self.shared.at_once {
                    usize::MAX => BACKGROUND_LANES,
                    _ => 0,
                };
                if background.running >= lanes || background.jobs.is_empty() {
                    return;
                }
                background.running += 1;
            }
            let executor = self.executor.clone();
            executor.spawn(Box::new(move || self.run()));
        }

        fn run(self) {
            let job = {
                let mut background = self.shared.background.lock().expect("compile poisoned");
                background.jobs.pop_front()
            };
            let closed = self.shared.closed();
            if let Some(job) = job.filter(|_| !closed) {
                let task = {
                    let mut queue = self.shared.queue.lock().expect("compile queue poisoned");
                    queue.next_task += 1;
                    queue.next_task
                };
                compile(&self.shared, &self.device, task, &self.done, job);
            }
            self.shared
                .background
                .lock()
                .expect("compile poisoned")
                .running -= 1;
            if !closed {
                // Spawned from the executor's thread, behind what it has.
                let lane = Lane {
                    shared: self.shared.clone(),
                    device: self.device.clone(),
                    done: self.done.clone(),
                    executor: self.executor.clone(),
                };
                lane.feed();
            }
        }
    }

    /// One request's task: compiles the job at the front of the queue, and
    /// the next while any are left, unless as many tasks as may compile at
    /// once already do, which then take its job. A task finds the queue
    /// empty and stops under the queue's lock, so no job waits without one.
    fn compile_front(shared: &Shared, device: &wgpu::Device, task: u64, done: &mpsc::Sender<Done>) {
        let mut queue = shared.queue.lock().expect("compile queue poisoned");
        if queue.compiling >= shared.at_once {
            return;
        }
        queue.compiling += 1;
        while let Some(job) = queue.jobs.pop_front() {
            drop(queue);
            compile(shared, device, task, done, job);
            queue = shared.queue.lock().expect("compile queue poisoned");
        }
        queue.compiling -= 1;
    }

    fn compile(
        shared: &Shared,
        device: &wgpu::Device,
        task: u64,
        done: &mpsc::Sender<Done>,
        job: Job,
    ) {
        let started = Instant::now();
        let mut busy = shared.busy.lock().expect("compile watch poisoned");
        busy.push((task, job.label().to_owned(), started));
        shared.changed.notify_all();
        while shared.hang.load(std::sync::atomic::Ordering::Relaxed) && !shared.closed() {
            busy = shared.changed.wait(busy).expect("compile watch poisoned");
        }
        drop(busy);
        let slow = shared.slow.lock().expect("compile watch poisoned").clone();
        if let Some((_, by)) = slow.filter(|(label, _)| job.label() == label) {
            std::thread::sleep(by);
        }
        let built = job.compile(device, shared.loss.get());
        let id = job.id();
        let finished = Done {
            id,
            built,
            label: job.label().to_owned(),
            took: started.elapsed(),
        };
        // Sent before it leaves `busy`, so `settle` finds it.
        let sent = done.send(finished).is_ok();
        let mut busy = shared.busy.lock().expect("compile watch poisoned");
        busy.retain(|(t, ..)| *t != task);
        shared.changed.notify_all();
        drop(busy);
        // A background compile no frame asked for wakes nothing. Read after
        // the send: a promotion before it is woken for, and one after it
        // finds the pipeline compiled.
        let quiet = (shared.background.lock().expect("compile poisoned"))
            .quiet
            .remove(&id);
        if sent && !quiet {
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

#[cfg(webgpu)]
mod webgpu {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use js_sys::{Array, Function, Object, Promise, Reflect};
    use orb_hal::*;
    use wasm_bindgen::{JsCast, JsValue};

    use super::{Built, Done, Job, Module, RenderJob, Stage};

    /// Called when a promise settles: the map draws the pipeline then.
    type Wake = Rc<RefCell<Option<Rc<dyn Fn()>>>>;

    /// Background compiles in flight at once: the browser compiles async
    /// pipelines on a few threads in turn, and one a frame asks for later
    /// waits behind those it already has.
    const BACKGROUND_LANES: usize = 2;

    /// Compiles with the browser's async pipeline creation.
    #[derive(Default)]
    pub(crate) struct Compiler {
        state: Rc<State>,
    }

    #[derive(Default)]
    struct State {
        /// The device, once a job came: wgpu wraps each pipeline as its
        /// promise settles.
        device: std::cell::OnceCell<wgpu::Device>,
        /// Pipelines made since the last [`Compiler::take`].
        settled: RefCell<VecDeque<Done>>,
        wake: Wake,
        background: RefCell<Background>,
    }

    /// Background compiles ([`Compiler::submit`]).
    #[derive(Default)]
    struct Background {
        /// Waiting, front first, with their bind group layouts.
        jobs: VecDeque<(Job, Vec<JsValue>)>,
        /// Promises not yet settled.
        running: usize,
        /// Waiting or compiling, and no frame asked for them: their
        /// promises wake nothing.
        quiet: std::collections::HashSet<PipelineId>,
        /// The device, once a job came.
        device: Option<JsValue>,
        /// They may start ([`Compiler::open_background`]).
        closed: bool,
    }

    impl std::fmt::Debug for Compiler {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Compiler").finish_non_exhaustive()
        }
    }

    fn wake_up(wake: &Wake) {
        let wake = wake.borrow().clone();
        if let Some(wake) = wake {
            wake();
        }
    }

    fn now() -> f64 {
        js_sys::Date::now()
    }

    impl Compiler {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        /// Calls `wake` whenever a pipeline's promise settles.
        pub(crate) fn set_wake(&self, wake: Rc<dyn Fn()>) {
            *self.state.wake.borrow_mut() = Some(wake);
        }

        /// Asks the browser for a pipeline; a `background` one waits its
        /// turn ([`BACKGROUND_LANES`]).
        pub(crate) fn submit(
            &mut self,
            device: &wgpu::Device,
            layouts: Vec<JsValue>,
            job: Job,
            background: bool,
        ) {
            self.state.device.get_or_init(|| device.clone());
            let raw = device.as_webgpu().expect("the web uses the WebGPU backend");
            let raw: &JsValue = raw.as_ref();
            if background {
                let mut b = self.state.background.borrow_mut();
                b.device.get_or_insert_with(|| raw.clone());
                b.quiet.insert(job.id());
                b.jobs.push_back((job, layouts));
                drop(b);
                feed(&self.state);
            } else {
                dispatch(&self.state, raw, &layouts, job, false);
            }
        }

        /// Lets background compiles be asked for (`open`), or holds those
        /// not asked for yet.
        pub(crate) fn open_background(&mut self, _: &wgpu::Device, open: bool) {
            let was = !std::mem::replace(&mut self.state.background.borrow_mut().closed, !open);
            if open && !was {
                feed(&self.state);
            }
        }

        /// Makes a background compile one a frame waits for: asked for now
        /// if it waits its turn, woken for when it settles.
        pub(crate) fn promote(&mut self, id: PipelineId) {
            let mut b = self.state.background.borrow_mut();
            b.quiet.remove(&id);
            let Some(at) = b.jobs.iter().position(|(j, _)| j.id() == id) else {
                return;
            };
            let (job, layouts) = b.jobs.remove(at).expect("found above");
            let device = b.device.clone().expect("set with the job");
            drop(b);
            dispatch(&self.state, &device, &layouts, job, false);
        }
    }

    /// Asks for waiting background compiles while lanes are free.
    fn feed(state: &Rc<State>) {
        loop {
            let mut b = state.background.borrow_mut();
            if b.closed || b.running >= BACKGROUND_LANES {
                return;
            }
            let Some((job, layouts)) = b.jobs.pop_front() else {
                return;
            };
            b.running += 1;
            let device = b.device.clone().expect("set with the job");
            drop(b);
            dispatch(state, &device, &layouts, job, true);
        }
    }

    /// Asks the browser to compile `job`; once its promise settles, the
    /// next [`Compiler::take`] takes it. A `background` one's settling
    /// starts the next.
    fn dispatch(state: &Rc<State>, raw: &JsValue, layouts: &[JsValue], job: Job, background: bool) {
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
        let state = state.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = (wasm_bindgen_futures::JsFuture::from(promise).await).map_err(|reason| {
                Reflect::get(&reason, &"message".into())
                    .ok()
                    .and_then(|m| m.as_string())
                    .unwrap_or_else(|| format!("{reason:?}"))
            });
            settle(&state, job, result, now() - started, background);
        });
    }

    /// A promise settled: wgpu wraps the pipeline the browser compiled, and
    /// the host is woken unless no frame asked for it.
    fn settle(
        state: &Rc<State>,
        job: Job,
        result: Result<JsValue, String>,
        ms: f64,
        background: bool,
    ) {
        let id = job.id();
        let device = state.device.get().expect("set with the job");
        let built = result.map(|pipeline| match &job {
            Job::Render(slot, _) => Built::Render(
                *slot,
                device.create_render_pipeline_from_webgpu_handle(pipeline.unchecked_into()),
            ),
            Job::Compute(slot, _) => Built::Compute(
                *slot,
                device.create_compute_pipeline_from_webgpu_handle(pipeline.unchecked_into()),
            ),
        });
        state.settled.borrow_mut().push_back(Done {
            id,
            built,
            label: job.label().to_owned(),
            took: std::time::Duration::from_secs_f64(ms / 1000.0),
        });
        let quiet = state.background.borrow_mut().quiet.remove(&id);
        if background {
            state.background.borrow_mut().running -= 1;
            feed(state);
        }
        if !quiet {
            wake_up(&state.wake);
        }
    }

    impl Compiler {
        pub(crate) fn prioritize(&mut self, _: PipelineId) {}

        /// Compiles finished since the last call.
        pub(crate) fn take(&mut self) -> Vec<Done> {
            self.state.settled.borrow_mut().drain(..).collect()
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

    /// The request as a WebGPU dictionary, as wgpu would map it.
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
        if job.cull_back {
            primitive.push(("cullMode", "back".into()));
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
                    ("depthCompare", "greater-equal".into()),
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
            TextureFormat::Bgra8Unorm => "bgra8unorm",
            TextureFormat::R32Float => "r32float",
            TextureFormat::R32Uint => "r32uint",
            TextureFormat::Rgba32Uint => "rgba32uint",
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
                        filterable,
                        dimension: d,
                    } => (
                        "texture",
                        dict(&[
                            (
                                "sampleType",
                                if filterable {
                                    "float"
                                } else {
                                    "unfilterable-float"
                                }
                                .into(),
                            ),
                            ("viewDimension", dimension(d).into()),
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
                    BindingType::UintTexture { dimension: d } => (
                        "texture",
                        dict(&[
                            ("sampleType", "uint".into()),
                            ("viewDimension", dimension(d).into()),
                        ]),
                    ),
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

#[cfg(webgpu)]
pub(crate) use webgpu::bind_group_layout as raw_bind_group_layout;

#[cfg(all(target_arch = "wasm32", not(any(webgpu, webgl))))]
compile_error!("orb-hal-wgpu on the web needs one of the `webgpu` and `gl` features");
#[cfg(all(webgpu, webgl))]
compile_error!("orb-hal-wgpu on the web takes one of the `webgpu` and `gl` features, not both");

#[cfg(not(target_arch = "wasm32"))]
mod dump;
#[cfg(webgl)]
mod webgl;
#[cfg(webgl)]
pub(crate) use webgl::Compiler;
