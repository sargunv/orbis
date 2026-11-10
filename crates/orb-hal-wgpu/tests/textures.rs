//! Textures through orb-hal on the default adapter: writes, array views,
//! samplers in bind groups.

use orb_hal::TextureViewDimension::D2Array;
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
        view_formats: &[],
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
                level: 0,
            },
            texels,
        );
    }
    let sampler = gpu.create_sampler(&SamplerDesc {
        label: "nearest",
        filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        anisotropy: 1,
    });
    let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
        label: "pages",
        entries: &[
            BindGroupLayoutEntry::texture(0, ShaderStages::FRAGMENT, true, D2Array),
            BindGroupLayoutEntry::sampler(1, ShaderStages::FRAGMENT, true),
        ],
    });
    let group = gpu.create_bind_group(&BindGroupDesc {
        label: "pages",
        layout,
        entries: &[
            BindGroupEntry::texture(0, pages, D2Array),
            BindGroupEntry::sampler(1, sampler),
        ],
    });
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: "sample",
        source: SHADER.into(),
        ir: None,
    });
    let stage = |entry_point| ShaderStage {
        module,
        entry_point,
        constants: &[],
    };
    let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_back: false,
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
        view_formats: &[],
    });
    // Pipelines compile off this thread; commands use them once compiled.
    gpu.finish_pipelines();
    assert!(gpu.pipeline_ready(pipeline.into()));
    gpu.submit(&[Pass::Render(RenderPass {
        label: "sample",
        colors: vec![ColorAttachment {
            store: StoreOp::Store,
            target,
            resolve: None,
            load: LoadOp::Clear([0.0; 4]),
        }],
        depth: None,
        commands: vec![
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup { index: 0, group },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ],
    })]);
    let pixels = pollster::block_on(gpu.read_texture(target));
    assert_eq!(&pixels[..4], [255, 0, 0, 255], "layer 0");
    assert_eq!(&pixels[12..], [0, 0, 255, 255], "layer 1");
}

/// Bitmaps packed side by side in an R8 array, as the glyph atlas packs
/// them, written in one submission: every texel lands. On Apple GPUs
/// writes in one submission that share a block of a texture raced, and
/// under load one now and then lost a block's worth of another's texels
/// (#156).
#[test]
fn packed_writes_in_one_submission_all_land() {
    const SIZE: u32 = 512;
    const LAYERS: u32 = 2;
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let mut seed: u64 = 7;
    let mut rnd = |n: u32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % u64::from(n)) as u32
    };
    // Glyph-sized bitmaps, then small ones, many to a block.
    for (min, spread) in [(40, 30), (4, 20)] {
        let page = gpu.create_texture(&TextureDesc {
            label: "page",
            width: SIZE,
            height: SIZE,
            array_layers: LAYERS,
            dimension: D2Array,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        let mut expected = vec![0u8; (SIZE * SIZE * LAYERS) as usize];
        for layer in 0..LAYERS {
            let (mut x, mut y, mut shelf) = (0, 0, 0);
            loop {
                let (w, h) = (min + rnd(spread), min + rnd(spread));
                if x + w > SIZE {
                    (x, y, shelf) = (0, y + shelf, 0);
                }
                if y + h > SIZE {
                    break;
                }
                let value = 1 + rnd(255) as u8;
                let texels = vec![value; (w * h) as usize];
                let region = TextureRegion {
                    texture: page,
                    x,
                    y,
                    layer,
                    width: w,
                    height: h,
                    level: 0,
                };
                gpu.write_texture(&region, &texels);
                for row in y..y + h {
                    let at = ((layer * SIZE + row) * SIZE + x) as usize;
                    expected[at..at + w as usize].fill(value);
                }
                x += w;
                shelf = shelf.max(h);
            }
        }
        gpu.submit(&[]);
        let read = read_layers(&gpu, page);
        let wrong = read.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "texels wrong of bitmaps {min}+{spread} wide");
        gpu.destroy_texture(page);
    }
}

/// Every layer of an R8 array texture, read back.
fn read_layers(gpu: &WgpuGpu, id: TextureId) -> Vec<u8> {
    let texture = gpu.texture(id).clone();
    let device = gpu.device().clone();
    let size = texture.size();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layers"),
        size: u64::from(size.width * size.height * size.depth_or_array_layers),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );
    gpu.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    buffer.slice(..).get_mapped_range().unwrap().to_vec()
}
