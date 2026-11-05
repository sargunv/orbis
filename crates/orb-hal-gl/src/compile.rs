//! Programs compile off the thread that draws (plan §8.4, decision 0038).
//!
//! Natively compile threads link them, each on a context that shares the
//! executor's objects (`egl::CompileThread`); the executor then sets each
//! linked program's uniform bindings on its own context. In the browser,
//! `KHR_parallel_shader_compile` lets the executor compile and link without
//! waiting and ask whether a program is done. Without either, one program
//! links per frame: the frame waits for one, not for all.

use std::collections::{HashMap, VecDeque};
use std::sync::{Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use glow::HasContext;

/// A program to link: its stages' sources, or a binary of it from an
/// earlier run.
pub(crate) struct Job {
    pub(crate) slot: u32,
    pub(crate) label: String,
    pub(crate) vertex: String,
    pub(crate) fragment: String,
    pub(crate) binary: Option<glow::ProgramBinary>,
    /// Whether to read the linked program's binary back for the cache.
    pub(crate) keep_binary: bool,
}

/// A linked program (its GL name, shared with the executor's context) and
/// its binary, or why it failed.
pub(crate) struct Done {
    pub(crate) slot: u32,
    pub(crate) result: Result<(glow::Program, Option<glow::ProgramBinary>), String>,
    pub(crate) took: Duration,
}

/// The compile threads' queue, front first, and what each works on.
#[derive(Default)]
pub(crate) struct Queue {
    jobs: Mutex<(VecDeque<Job>, bool)>,
    wake: Condvar,
    busy: Mutex<HashMap<usize, (String, Instant)>>,
    /// Called from the thread when a program links.
    linked: Mutex<Option<std::sync::Arc<dyn Fn() + Send + Sync>>>,
}

impl Queue {
    pub(crate) fn push(&self, job: Job) {
        self.jobs
            .lock()
            .expect("compile queue poisoned")
            .0
            .push_back(job);
        self.wake.notify_one();
    }

    pub(crate) fn prioritize(&self, slot: u32) {
        let mut jobs = self.jobs.lock().expect("compile queue poisoned");
        if let Some(at) = jobs.0.iter().position(|j| j.slot == slot)
            && at > 0
        {
            let job = jobs.0.remove(at).expect("found");
            jobs.0.push_front(job);
        }
    }

    pub(crate) fn set_wake(&self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
        *self.linked.lock().expect("compile wake poisoned") = Some(wake);
    }

    pub(crate) fn close(&self) {
        let mut jobs = self.jobs.lock().expect("compile queue poisoned");
        jobs.0.clear();
        jobs.1 = true;
        self.wake.notify_all();
    }

    pub(crate) fn hung(&self, limit: Duration) -> Option<String> {
        let busy = self.busy.lock().expect("compile watch poisoned");
        busy.values()
            .find(|(_, since)| since.elapsed() > limit)
            .map(|(label, since)| format!("{label} ({:.1} s)", since.elapsed().as_secs_f64()))
    }

    fn next(&self) -> Option<Job> {
        let mut jobs = self.jobs.lock().expect("compile queue poisoned");
        loop {
            if jobs.1 {
                return None;
            }
            if let Some(job) = jobs.0.pop_front() {
                return Some(job);
            }
            jobs = self.wake.wait(jobs).expect("compile queue poisoned");
        }
    }
}

/// Compile thread `worker`'s loop, on its own context.
pub(crate) fn work(gl: &glow::Context, queue: &Queue, done: &mpsc::Sender<Done>, worker: usize) {
    let mut stages = HashMap::new();
    while let Some(job) = queue.next() {
        let started = Instant::now();
        queue
            .busy
            .lock()
            .expect("compile watch poisoned")
            .insert(worker, (job.label.clone(), started));
        // SAFETY: objects of the context current on this thread, which
        // shares them with the executor's.
        let result = unsafe { link(gl, &mut stages, &job) };
        queue
            .busy
            .lock()
            .expect("compile watch poisoned")
            .remove(&worker);
        let finished = Done {
            slot: job.slot,
            result,
            took: started.elapsed(),
        };
        if done.send(finished).is_err() {
            break;
        }
        let wake = queue.linked.lock().expect("compile wake poisoned").clone();
        if let Some(wake) = wake {
            wake();
        }
    }
    // SAFETY: as above.
    unsafe {
        for (_, shader) in stages {
            gl.delete_shader(shader);
        }
    }
}

/// Links a job's program on the current context and waits for it, so
/// another context of the share group can use it.
///
/// # Safety
///
/// `gl` is current on this thread, and `stages` holds its shaders.
unsafe fn link(
    gl: &glow::Context,
    stages: &mut HashMap<(u32, u64), glow::Shader>,
    job: &Job,
) -> Result<(glow::Program, Option<glow::ProgramBinary>), String> {
    // SAFETY: the caller's context.
    unsafe {
        let program = gl.create_program()?;
        if let Some(binary) = &job.binary {
            gl.program_binary(program, binary);
            if gl.get_program_link_status(program) {
                gl.finish();
                return Ok((program, None));
            }
            // The driver turned it down (it changed): compile.
        }
        for (kind, source) in [
            (glow::VERTEX_SHADER, &job.vertex),
            (glow::FRAGMENT_SHADER, &job.fragment),
        ] {
            let key = (kind, crate::cache::key(source, ""));
            let shader = match stages.get(&key) {
                Some(&shader) => shader,
                None => {
                    let shader = gl.create_shader(kind)?;
                    gl.shader_source(shader, source);
                    gl.compile_shader(shader);
                    if !gl.get_shader_compile_status(shader) {
                        let log = gl.get_shader_info_log(shader);
                        gl.delete_shader(shader);
                        gl.delete_program(program);
                        return Err(format!("GLSL compile failed: {log}"));
                    }
                    stages.insert(key, shader);
                    shader
                }
            };
            gl.attach_shader(program, shader);
        }
        if job.keep_binary {
            gl.program_binary_retrievable_hint(program, true);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(format!("GLSL link failed: {log}"));
        }
        let binary = job
            .keep_binary
            .then(|| gl.get_program_binary(program))
            .flatten();
        gl.finish();
        Ok((program, binary))
    }
}
