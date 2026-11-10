//! Every offline scene's frames, settled and edited as the golden streams
//! draw them, under the graphics API's own validation: Vulkan's Khronos
//! layer and D3D12's debug layer, which wgpu turns on in debug builds
//! (or with `WGPU_VALIDATION=1`) where they're installed and whose
//! findings wgpu-hal logs as errors. Any error fails the test. Metal's and
//! GL's validation don't report through wgpu-hal, so there it skips, as it
//! does where validation is off or its layer isn't found (it says which
//! under `--nocapture`); on Apple it runs on MoltenVK with
//! `ORB_BACKEND=vulkan cargo test -p orb-runtime -p orb-hal-wgpu
//! --features orb-hal-wgpu/vulkan --test validation` (AGENTS.md).
//!
//! Its own test binary: the log is the process's.

mod common;

use std::sync::Mutex;

use common::WgpuGpu;
use common::scenes::{ALL, scenes, settle};
use orb_hal_wgpu::wgpu::{self, Backend};

/// What wgpu-hal logs when it can't turn the layer on: Vulkan's
/// (`debug`) and D3D12's (`warn`).
const NO_LAYER: [&str; 2] = ["unable to find layer", "Failed to get debug interface"];

/// What wgpu-hal logged: errors, and why validation is off, if it is.
struct Log {
    errors: Mutex<Vec<String>>,
    no_layer: Mutex<Option<String>>,
}

impl log::Log for Log {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.target().starts_with("wgpu_hal")
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let message = record.args().to_string();
        if record.level() == log::Level::Error {
            let mut errors = self.errors.lock().unwrap_or_else(|e| e.into_inner());
            errors.push(message);
        } else if NO_LAYER.iter().any(|s| message.contains(s)) {
            *self.no_layer.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
        }
    }

    fn flush(&self) {}
}

static LOG: Log = Log {
    errors: Mutex::new(Vec::new()),
    no_layer: Mutex::new(None),
};

#[test]
fn scenes_validate() {
    // The instance offscreen devices make: whether it validates.
    if !(wgpu::InstanceFlags::default().with_env()).contains(wgpu::InstanceFlags::VALIDATION) {
        eprintln!("skipped: wgpu's validation is off (a release build without WGPU_VALIDATION=1)");
        return;
    }
    log::set_logger(&LOG).unwrap();
    // Only while the first instance is made: wgpu-hal says at `debug`
    // that Vulkan's layer is missing, and the layer's messages it asks for
    // follow the level set when an instance is made.
    log::set_max_level(log::LevelFilter::Debug);
    let gpu = pollster::block_on(WgpuGpu::offscreen()).unwrap();
    log::set_max_level(log::LevelFilter::Error);
    let backend = gpu.adapter_info().expect("a native adapter").backend;
    drop(gpu);
    if !matches!(backend, Backend::Vulkan | Backend::Dx12) {
        eprintln!("skipped: {backend:?}'s validation doesn't report through wgpu-hal");
        return;
    }
    if let Some(message) = LOG.no_layer.lock().unwrap().take() {
        eprintln!("skipped: {backend:?}'s validation layer isn't available ({message})");
        return;
    }
    let mut failed = Vec::new();
    let first = std::mem::take(&mut *LOG.errors.lock().unwrap());
    if !first.is_empty() {
        failed.push(format!("making a device:\n  {}", first.join("\n  ")));
    }
    for scene in scenes(ALL) {
        // Validation runs as commands are recorded and submitted; dropping
        // the device reports what it leaked.
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).unwrap();
        settle(&mut gpu, &scene, |_| {});
        drop(gpu);
        let errors = std::mem::take(&mut *LOG.errors.lock().unwrap());
        if !errors.is_empty() {
            failed.push(format!("{}:\n  {}", scene.name, errors.join("\n  ")));
        }
    }
    assert!(
        failed.is_empty(),
        "validation errors in\n{}",
        failed.join("\n")
    );
}
