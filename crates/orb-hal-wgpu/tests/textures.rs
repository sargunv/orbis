//! Textures through orb-hal on the default adapter: writes, array views,
//! samplers in bind groups.

use orb_hal::*;
use orb_hal_wgpu::WgpuGpu;

const SHADER: &str = r#"
@group(0) @binding(0) var pages: texture_2d_array<f32>;
@group(0) @binding(1) var nearest: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    // Left half samples layer 0, right half layer 1.
    let layer = select(0, 1, p.x >= 2.0);
    return textureSample(pages, nearest, fract(p.xy / 2.0), layer);
}
"#;

#[test]
fn samples_written_layers_of_a_texture_array() {
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let pages = gpu.create_texture(&TextureDesc {
        label: "pages",
        width: 2,
        height: 2,
        array_layers: 2,
        dimension: orb_hal::TextureViewDimension::D2Array,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        sample_count: 1,
        mip_levels: 1,
    });
    let red = [255, 0, 0, 255].repeat(4);
    let blue = [0, 0, 255, 255].repeat(4);
    for (layer, texels) in [(0, &red), (1, &blue)] {
        gpu.write_texture(
            &TextureRegion {
                texture: pages,
                x: 0,
                y: 0,
                layer,
                width: 2,
                height: 2,
            },
            texels,
        );
    }
    let sampler = gpu.create_sampler(&SamplerDesc {
        label: "nearest",
        filter: FilterMode::Nearest,
        address: AddressMode::ClampToEdge,
        mipmap_filter: FilterMode::Nearest,
    });
    let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
        label: "pages",
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample: TextureSampleType::Float { filterable: true },
                    dimension: orb_hal::TextureViewDimension::D2Array,
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler { filtering: true },
            },
        ],
    });
    let group = gpu.create_bind_group(&BindGroupDesc {
        label: "pages",
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Texture {
                    texture: pages,
                    dimension: orb_hal::TextureViewDimension::D2Array,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
        ],
    });
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: "sample",
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let stage = |entry_point| ShaderStage {
        module,
        entry_point,
        constants: &[],
    };
    let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: "sample",
        bind_group_layouts: &[layout],
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
        width: 4,
        height: 1,
        array_layers: 1,
        dimension: orb_hal::TextureViewDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    });
    gpu.submit(&[Pass::Render(RenderPass {
        label: "sample".into(),
        color: ColorAttachment {
            store: StoreOp::Store,
            target,
            resolve: None,
            load: LoadOp::Clear([0.0; 4]),
        },
        depth: None,
        commands: vec![
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup { index: 0, group },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ],
        timestamps: None,
    })]);
    let pixels = pollster::block_on(gpu.read_texture(target));
    assert_eq!(&pixels[..4], [255, 0, 0, 255], "layer 0");
    assert_eq!(&pixels[12..], [0, 0, 255, 255], "layer 1");
}
