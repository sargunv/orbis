//! Every offline scene's frames, settled and edited as the golden streams
//! draw them, under the graphics API's own validation: Vulkan's Khronos
//! layer and D3D12's debug layer, which wgpu turns on in debug builds
//! where they're installed and whose findings wgpu-hal logs as errors.
//! Any error fails the test. Metal's and GL's validation don't report
//! through wgpu-hal, so there it skips; on Apple it runs on MoltenVK with
//! `ORB_BACKEND=vulkan cargo test -p orb-runtime -p orb-hal-wgpu
//! --features orb-hal-wgpu/vulkan --test validation` (AGENTS.md).
//!
//! Its own test binary: the log is the process's.

mod common;

use std::sync::Mutex;

use common::WgpuGpu;
use common::scenes::{ALL, scenes, settle};
use orb_hal_wgpu::wgpu::Backend;

/// What wgpu-hal logged as errors.
struct Errors(Mutex<Vec<String>>);

impl log::Log for Errors {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Error && metadata.target().starts_with("wgpu_hal")
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let mut errors = self.0.lock().unwrap_or_else(|e| e.into_inner());
            errors.push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static ERRORS: Errors = Errors(Mutex::new(Vec::new()));

#[test]
fn scenes_validate() {
    let gpu = pollster::block_on(WgpuGpu::offscreen()).unwrap();
    let backend = gpu.adapter_info().expect("a native adapter").backend;
    drop(gpu);
    if !matches!(backend, Backend::Vulkan | Backend::Dx12) {
        eprintln!("skipped: {backend:?}'s validation doesn't report through wgpu-hal");
        return;
    }
    log::set_logger(&ERRORS).unwrap();
    log::set_max_level(log::LevelFilter::Error);
    let mut failed = Vec::new();
    for scene in scenes(ALL) {
        // Validation runs as commands are recorded and submitted; dropping
        // the device reports what it leaked.
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).unwrap();
        settle(&mut gpu, &scene, |_| {});
        drop(gpu);
        let errors = std::mem::take(&mut *ERRORS.0.lock().unwrap());
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
