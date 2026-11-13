//! The device's per-pixel floor: full-screen passes of plain wgpu, timed
//! with timestamp queries, for telling what a map's full-screen layer
//! costs from what the GPU charges any pass that size.
//!
//! ```text
//! cargo run --release -p orb-hal-wgpu --example fill_floor -- [--size WxH] [--frames N] [--filter s]
//! ```
//!
//! Each case is a render pass into a target of `--size` (default 5120×2880,
//! 2560×1440 at 2×), cleared and stored, drawing what the case names. A
//! submission draws the pass eight times back to back, so the GPU stays at
//! its clock; on a tiled GPU every pass's first timestamp lands as its
//! vertex work starts, all together, so a pass's time is the span of the
//! last seven over seven. Prints each case's median over `--frames`
//! submissions (default 50). `ORB_BACKEND` and wgpu's variables choose the
//! adapter.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::borrow::Cow;

    const SHADER: &str = r"
struct Out { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@group(0) @binding(0) var<storage, read> quads: array<vec4<f32>>;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct Tile { a: vec4<f32>, b: vec4<f32>, c: vec4<u32>, d: vec4<u32> };
@group(0) @binding(3) var<storage, read> tiles: array<Tile>;

// Instance i covers quads[i] (NDC min, max), as a tile would.
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let r = quads[i];
    let corner = vec2<f32>(f32(v & 1u), f32((v >> 1u) & 1u));
    var o: Out;
    o.pos = vec4<f32>(mix(r.xy, r.zw, corner), 0.2 + 0.6 * corner.x, 1.0);
    o.uv = corner;
    return o;
}
@fragment fn flat(in: Out) -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.3, 0.4, 0.5); }
@fragment fn flat_discard(in: Out) -> @location(0) vec4<f32> {
    if in.uv.x < -1.0 { discard; }
    return vec4<f32>(0.2, 0.3, 0.4, 0.5);
}
// A tile's clip read per fragment and tested, as a map's fragments do.
@fragment fn storage_read(in: Out) -> @location(0) vec4<f32> {
    let t = tiles[u32(in.uv.x * 0.1)];
    if any(in.uv < t.a.xy) || any(in.uv > t.a.zw) { discard; }
    return vec4<f32>(0.2, 0.3, 0.4, 0.5) * t.b;
}
@fragment fn textured(in: Out) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
// The texture repeated at its own scale: noise, as incompressible as a map.
@fragment fn textured_noise(in: Out) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.pos.xy / 512.0);
}
@fragment fn textured9(in: Out) -> @location(0) vec4<f32> {
    var s = vec4<f32>(0.0);
    let d = 1.0 / 512.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            s += textureSample(tex, samp, in.uv + vec2<f32>(f32(x), f32(y)) * d);
        }
    }
    return s / 9.0;
}
// About the globe's scattering: twenty steps of three exps.
@fragment fn exp20(in: Out) -> @location(0) vec4<f32> {
    var acc = vec3<f32>(0.0);
    var p = in.uv.x;
    for (var i = 0; i < 20; i++) {
        p = p * 1.01 + 0.01;
        acc += exp(-vec3<f32>(p, p * 0.5, p * 0.25));
    }
    return vec4<f32>(acc * 0.05, 1.0);
}
";

    /// Passes a submission draws, back to back.
    const PASSES: u32 = 8;

    struct Case {
        name: &'static str,
        fragment: Option<&'static str>,
        blend: bool,
        layers: u32,
        /// Tiles per side the screen is split into, each drawn as a quad.
        grid: u32,
        format: wgpu::TextureFormat,
        samples: u32,
        /// The pass loads what the one before stored rather than clearing.
        load: bool,
        /// A Depth32Float attachment, cleared and depth-tested, and whether
        /// the pass stores it.
        depth: Option<bool>,
    }

    const fn case(name: &'static str, fragment: Option<&'static str>, layers: u32) -> Case {
        Case {
            name,
            fragment,
            blend: true,
            layers,
            grid: 1,
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            load: false,
            depth: None,
        }
    }

    struct Resources {
        module: wgpu::ShaderModule,
        texture: wgpu::TextureView,
        sampler: wgpu::Sampler,
        tiles: wgpu::Buffer,
    }

    pub(super) fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let arg = |name: &str| {
            (args.iter().position(|a| a == name)).and_then(|i| args.get(i + 1).cloned())
        };
        let size = arg("--size")
            .and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?))
            })
            .unwrap_or((5120u32, 2880u32));
        let frames: usize = arg("--frames").and_then(|s| s.parse().ok()).unwrap_or(50);
        let filter = arg("--filter");

        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle().with_env();
        if let Ok(names) = std::env::var("ORB_BACKEND") {
            descriptor.backends = wgpu::Backends::from_comma_list(&names);
        }
        let instance = wgpu::Instance::new(descriptor);
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("an adapter");
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fill floor"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: orb_hal_wgpu::device_limits(&adapter),
            ..Default::default()
        }))
        .expect("a device with timestamp queries");
        println!(
            "{} ({:?}), {}×{} ({:.1} Mpx), median of {frames} submissions, ms a pass",
            info.name,
            info.backend,
            size.0,
            size.1,
            f64::from(size.0 * size.1) / 1e6
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut seed = 0x2545_f491_u32;
        let texels: Vec<u8> = (0..512 * 512 * 4)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 24) as u8
            })
            .collect();
        queue.write_texture(
            texture.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512 * 4),
                rows_per_image: None,
            },
            texture.size(),
        );
        let tiles = buffer(
            &device,
            &queue,
            &[-1.0, -1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0].repeat(2048),
        );
        let resources = Resources {
            module: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
            }),
            texture: texture.create_view(&Default::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            tiles,
        };

        let cases = [
            case("clear only", None, 0),
            Case {
                blend: false,
                ..case("flat, opaque", Some("flat"), 1)
            },
            case("flat, blended", Some("flat"), 1),
            case("flat, blended ×4", Some("flat"), 4),
            case("flat, blended ×16", Some("flat"), 16),
            case("flat, blended, discard ×4", Some("flat_discard"), 4),
            case("tile read + discard ×4", Some("storage_read"), 4),
            case("textured ×4", Some("textured"), 4),
            case("textured, 9 taps", Some("textured9"), 1),
            case("20 exp3", Some("exp20"), 1),
            Case {
                grid: 9,
                ..case("flat, blended ×4, as 9×9 tiles", Some("flat"), 4)
            },
            Case {
                format: wgpu::TextureFormat::Rgba16Float,
                ..case("flat, blended ×4, rgba16float", Some("flat"), 4)
            },
            Case {
                samples: 4,
                ..case("flat, blended ×4, 4× MSAA + resolve", Some("flat"), 4)
            },
            // What a pass costs that stores the screen for another to
            // load (a split main pass, an offscreen layer): the
            // texture's noise, which compresses as a map does, not flat.
            case("noise", Some("textured_noise"), 1),
            Case {
                load: true,
                ..case(
                    "noise, loading the pass before's",
                    Some("textured_noise"),
                    1,
                )
            },
            Case {
                depth: Some(false),
                ..case("noise, depth32float discarded", Some("textured_noise"), 1)
            },
            Case {
                depth: Some(true),
                ..case("noise, depth32float stored", Some("textured_noise"), 1)
            },
        ];
        for c in cases {
            if filter
                .as_ref()
                .is_some_and(|f| !c.name.contains(f.as_str()))
            {
                continue;
            }
            let ms = run(&device, &queue, &resources, &c, size, frames);
            println!("| {} | {ms:.3} |", c.name);
        }
    }

    fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, words: &[f32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &bytes);
        buffer
    }

    fn run(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        r: &Resources,
        c: &Case,
        (width, height): (u32, u32),
        frames: usize,
    ) -> f64 {
        let target = |samples| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: c.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let resolved = target(1);
        let depth = c.depth.map(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: c.samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let multisampled = (c.samples > 1).then(|| target(c.samples));

        let s = 2.0 / c.grid as f32;
        let rects: Vec<f32> = (0..c.grid * c.grid)
            .flat_map(|i| {
                let (x0, y0) = (
                    -1.0 + (i % c.grid) as f32 * s,
                    -1.0 + (i / c.grid) as f32 * s,
                );
                [x0, y0, x0 + s, y0 + s]
            })
            .collect();
        let quads = buffer(device, queue, &rects);
        let storage = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage(0, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: quads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&r.texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&r.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: r.tiles.as_entire_binding(),
                },
            ],
        });
        let pipeline = c.fragment.map(|entry| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(c.name),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &r.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: c.depth.map(|_| wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: c.samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &r.module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: c.format,
                        blend: c
                            .blend
                            .then_some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        });

        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2 * PASSES,
        });
        let bytes = 16 * u64::from(PASSES);
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = f64::from(queue.get_timestamp_period());
        let warmup = 5;
        let mut times = Vec::with_capacity(frames);
        let wait = || {
            (device.poll(wgpu::PollType::wait_indefinitely())).expect("the device answers");
        };
        for f in 0..warmup + frames {
            let mut encoder = device.create_command_encoder(&Default::default());
            for k in 0..PASSES {
                let (view, resolve_target) = match &multisampled {
                    Some(ms) => (ms, Some(&resolved)),
                    None => (&resolved, None),
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(c.name),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: match c.load {
                                true => wgpu::LoadOp::Load,
                                false => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            },
                            store: match multisampled {
                                Some(_) => wgpu::StoreOp::Discard,
                                None => wgpu::StoreOp::Store,
                            },
                        },
                    })],
                    depth_stencil_attachment: depth.as_ref().zip(c.depth).map(|(view, store)| {
                        wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: match store {
                                    true => wgpu::StoreOp::Store,
                                    false => wgpu::StoreOp::Discard,
                                },
                            }),
                            stencil_ops: None,
                        }
                    }),
                    timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(2 * k),
                        end_of_pass_write_index: Some(2 * k + 1),
                    }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if let Some(p) = &pipeline {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..c.layers {
                        pass.draw(0..4, 0..c.grid * c.grid);
                    }
                }
            }
            queue.submit([encoder.finish()]);
            // A pass's last sample lands once its fragment work finishes,
            // which on a tiled GPU can outlast a resolve in the same
            // submission: resolve once the passes are done.
            wait();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(&queries, 0..2 * PASSES, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, bytes);
            queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            wait();
            let ticks: Vec<u64> = (readback.slice(..).get_mapped_range())
                .expect("mapped")
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_ne_bytes(*b))
                .collect();
            readback.unmap();
            if f >= warmup {
                let span = ticks[ticks.len() - 1].saturating_sub(ticks[1]);
                times.push(span as f64 * period / 1e6 / f64::from(PASSES - 1));
            }
        }
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    }
}
