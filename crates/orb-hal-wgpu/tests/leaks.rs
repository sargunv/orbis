//! A pipeline that fails to compile leaves nothing of its own on the
//! device. On the Adreno X1's Vulkan (#141) the driver failed `symbol`'s
//! pipelines, and Vulkan's validation layer then found the four shader
//! modules wgpu-hal had made from naga's SPIR-V for them still alive when
//! the device went (VUID-vkDestroyDevice-device-05137). Those modules leak
//! on every way a pipeline can fail after its first stage is made: here
//! the fragment stage fails in naga, after the vertex stage's module.
//!
//! Runs where Vulkan's validation layer reports through wgpu-hal: on Apple
//! through MoltenVK, `ORB_BACKEND=vulkan WGPU_VALIDATION=1 cargo test -p
//! orb-hal-wgpu --features vulkan --test leaks` (with Homebrew's
//! `molten-vk` and `vulkan-validationlayers`, also
//! `VK_ICD_FILENAMES=$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json`,
//! `VK_LAYER_PATH=$(brew --prefix)/share/vulkan/explicit_layer.d` and
//! `DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib`). Elsewhere it skips,
//! saying why under `--nocapture`.
//!
//! Its own test binary: the log is the process's.

use std::sync::Mutex;

use orb_hal_wgpu::WgpuGpu;
use orb_hal_wgpu::wgpu::{self, Backend};

/// What wgpu-hal logged: its errors, and whether Vulkan's layer was
/// missing.
struct Log {
    errors: Mutex<Vec<String>>,
    no_layer: Mutex<bool>,
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
            (self.errors.lock().unwrap_or_else(|e| e.into_inner())).push(message);
        } else if message.contains("unable to find layer") {
            *self.no_layer.lock().unwrap_or_else(|e| e.into_inner()) = true;
        }
    }

    fn flush(&self) {}
}

static LOG: Log = Log {
    errors: Mutex::new(Vec::new()),
    no_layer: Mutex::new(false),
};

/// The vertex stage compiles; the fragment stage's division by an override
/// set to zero fails in naga's constant evaluation, after the vertex
/// stage's module was made.
const SHADER: &str = "
override divisor: i32 = 1;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(f32(7 / divisor));
}
";

#[test]
fn a_failed_pipeline_leaves_no_shader_modules() {
    if !(wgpu::InstanceFlags::default().with_env()).contains(wgpu::InstanceFlags::VALIDATION) {
        eprintln!("skipped: wgpu's validation is off (a release build without WGPU_VALIDATION=1)");
        return;
    }
    // wgpu-hal hears the layer through debug utils, which offscreen
    // devices leave off on loaders that race on them unless asked.
    // SAFETY: no other thread runs yet.
    unsafe { std::env::set_var("WGPU_DEBUG", "1") };
    log::set_logger(&LOG).unwrap();
    // wgpu-hal says at `debug` that the layer is missing, and asks the
    // layer for messages at the level set when the instance is made.
    log::set_max_level(log::LevelFilter::Debug);
    let gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    log::set_max_level(log::LevelFilter::Error);
    let backend = gpu.adapter_info().expect("a native adapter").backend;
    if backend != Backend::Vulkan {
        eprintln!("skipped: {backend:?}'s validation doesn't report leaks through wgpu-hal");
        return;
    }
    if *LOG.no_layer.lock().unwrap() {
        eprintln!("skipped: Vulkan's validation layer isn't available");
        return;
    }
    let device = gpu.device().clone();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("leaks"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fails in its fragment stage"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("divisor", 0.0)],
                ..Default::default()
            },
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let failed = pollster::block_on(scope.pop());
    assert!(
        failed.is_some(),
        "the pipeline compiled: its fragment stage should fail"
    );
    drop(module);
    drop(device);
    drop(gpu);
    let errors = std::mem::take(&mut *LOG.errors.lock().unwrap());
    assert!(
        !errors.iter().any(|e| e.contains("leaked")),
        "the device outlived objects:\n  {}",
        errors.join("\n  ")
    );
}
