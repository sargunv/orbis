//! `ORB_COMPILES_AT_ONCE` caps how many pipelines compile at the same
//! time, however many job threads ask.
//!
//! Its own test binary: the variable is the process's.

use std::sync::Arc;
use std::time::{Duration, Instant};

use orb_hal::*;
use orb_hal_wgpu::WgpuGpu;

const SHADER: &str = "
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
";

#[test]
fn one_at_a_time_compiles_one_at_a_time() {
    const PIPELINES: u32 = 4;
    const SLOW: Duration = Duration::from_millis(250);
    // SAFETY: no other thread runs yet.
    unsafe { std::env::set_var("ORB_COMPILES_AT_ONCE", "1") };
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    gpu.set_compile_executor(Some(Arc::new(orb_host::jobs::ThreadPool::new(
        "pipelines",
        PIPELINES as usize,
    ))));
    gpu.inject_slow_compile("slow", SLOW);
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: "slow",
        source: SHADER.into(),
        ir: None,
        glsl: None,
        fold: &[],
    });
    let stage = |entry_point| ShaderStage {
        module,
        entry_point,
        constants: &[],
    };
    let started = Instant::now();
    let pipelines: Vec<_> = (0..PIPELINES)
        .map(|_| {
            gpu.create_render_pipeline(&RenderPipelineDesc {
                label: "slow",
                bind_group_layouts: &[],
                vertex: stage("vs"),
                fragment: stage("fs"),
                targets: &[ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                }],
                depth: None,
                topology: PrimitiveTopology::TriangleList,
                cull_back: false,
                sample_count: 1,
            })
        })
        .collect();
    gpu.finish_pipelines();
    let took = started.elapsed();
    for pipeline in pipelines {
        assert!(gpu.pipeline_ready(pipeline.into()));
    }
    // Four threads would have slept through their compiles together.
    assert!(
        took >= SLOW * PIPELINES,
        "{PIPELINES} compiles of {SLOW:?} each took {took:?}: they overlapped"
    );
}
