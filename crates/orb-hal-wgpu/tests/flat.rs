//! Flat varyings on the default adapter (`ORB_BACKEND` picks another).

use orb_hal::*;
use orb_hal_wgpu::WgpuGpu;

const SHADER: &str = "
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat, either) vertex: u32,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    // Vertices 3, 7 and 11 are the corners of a triangle over the target.
    let corner = (i / 4u) % 3u;
    let p = vec2<f32>(f32(corner == 1u), f32(corner == 2u)) * 4.0 - 1.0;
    return Out(vec4<f32>(p, 0.0, 1.0), i);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(in.vertex) / 255.0, 0.0, 0.0, 1.0);
}
";

/// A flat varying takes the value of the vertex `Caps::flat_from_last`
/// names, in an indexed draw of an instance past the first, as the
/// terrain draws its tiles: its mesh orders each triangle by it, so that
/// a cell's footprint comes from the cell's own first grid point.
#[test]
fn a_flat_varying_takes_the_vertex_the_caps_name() {
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let indices = gpu.create_buffer(&BufferDesc {
        label: "indices",
        size: 12,
        usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
    });
    gpu.write_buffer(indices, 0, &[3u32, 7, 11].map(u32::to_ne_bytes).concat());
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: "flat",
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
    let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_back: false,
        label: "flat",
        bind_group_layouts: &[],
        vertex: stage("vs"),
        fragment: stage("fs"),
        targets: &[ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
        }],
        depth: None,
        topology: PrimitiveTopology::TriangleList,
        sample_count: 1,
    });
    let target = gpu.create_texture(&TextureDesc {
        label: "target",
        width: 1,
        height: 1,
        array_layers: 1,
        dimension: TextureViewDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    });
    gpu.finish_pipelines();
    assert!(gpu.pipeline_ready(pipeline.into()));
    gpu.submit(&[Pass::Render(RenderPass {
        label: "flat",
        colors: vec![ColorAttachment {
            store: StoreOp::Store,
            target,
            resolve: None,
            load: LoadOp::Clear([0.0; 4]),
        }],
        depth: None,
        commands: vec![
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetIndexBuffer {
                buffer: indices,
                offset: 0,
            },
            RenderCommand::DrawIndexed {
                indices: 0..3,
                base_vertex: 0,
                instances: 1..2,
            },
        ],
    })]);
    let pixels = pollster::block_on(gpu.read_texture(target));
    let expected = if gpu.caps().flat_from_last { 11 } else { 3 };
    assert_eq!(
        pixels[..2],
        [expected, 0],
        "flat_from_last {}",
        gpu.caps().flat_from_last
    );
}
