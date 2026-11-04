//! Extrusion surfaces resolve opaque nearest samples, then apply layer opacity
//! once. The shared depth attachment preserves occlusion between 3D layers.
use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_scene::SceneBuffers;
use orb_shaders::{EXTRUSION_COMPOSITE_SHADER, EXTRUSION_SHADER};

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// At one sample and at four.
    pipelines: [Option<RenderPipelineId>; 2],
    layouts: [BindGroupLayoutId; 2],
    pub(crate) samples: u32,
    pub(crate) composite: RenderPipelineId,
    layout: BindGroupLayoutId,
    depth: Option<TextureId>,
    size: [u32; 2],
    single_layer: bool,
}
#[derive(Debug, Default)]
pub(crate) struct NodeExtrusion {
    color: Option<TextureId>,
    multisample: Option<TextureId>,
    size: [u32; 2],
    samples: u32,
    pub(crate) group: Option<BindGroupId>,
    bundle: Option<RenderBundleId>,
}
impl NodeExtrusion {
    /// Drops the recorded draw, so the next frame records it over the
    /// current bind groups.
    pub(crate) fn forget_bundle(&mut self, gpu: &mut dyn Gpu) {
        if let Some(bundle) = self.bundle.take() {
            gpu.destroy_render_bundle(bundle);
        }
    }
}

impl Extrusions {
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
    ) -> Self {
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "extrusion composite",
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Uniform,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });
        let composite = draw_pipeline(
            gpu,
            &EXTRUSION_COMPOSITE_SHADER,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[scene, layout],
        );
        Self {
            pipelines: [None; 2],
            layouts: [scene, node],
            samples: 4,
            composite,
            layout,
            depth: None,
            size: [0; 2],
            single_layer: false,
        }
    }
    fn pipeline(&mut self, gpu: &mut dyn Gpu) {
        let slot = usize::from(self.samples == 4);
        if self.pipelines[slot].is_some() {
            return;
        }
        let shader = EXTRUSION_SHADER;
        let module = gpu.create_shader_module(&ShaderModuleDesc {
            label: shader.label,
            wgsl: shader.source().into(),
        });
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            cull_mode: Some(Face::Back),
            label: shader.label,
            bind_group_layouts: &self.layouts,
            vertex: ShaderStage {
                module,
                entry_point: shader.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module,
                entry_point: shader.fragment,
                constants: &[],
            },
            targets: &[ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
            }],
            depth: Some(DepthState {
                format: TextureFormat::Depth32Float,
                write: true,
                compare: CompareFunction::GreaterEqual,
            }),
            topology: PrimitiveTopology::TriangleList,
            sample_count: self.samples,
        });
        self.pipelines[slot] = Some(pipeline);
    }
    pub(crate) fn set_antialias(&mut self, antialias: bool) -> bool {
        let samples = if antialias { 4 } else { 1 };
        if samples == self.samples {
            return false;
        }
        self.samples = samples;
        self.size = [0; 2];
        true
    }
    /// The depth 3D passes share, once a resize made it, and how its
    /// passes store it: a single pass's is transient.
    pub(crate) fn depth(&self) -> Option<(TextureId, StoreOp)> {
        let store = if self.single_layer {
            StoreOp::Discard
        } else {
            StoreOp::Store
        };
        self.depth.map(|d| (d, store))
    }

    /// Sizes the depth `count` 3D passes share: extrusion layers and
    /// terrain stacks.
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], count: usize) -> bool {
        if count == 0 || (self.size == size && self.single_layer == (count == 1)) {
            return false;
        }
        self.pipeline(gpu);
        if let Some(depth) = self.depth.take() {
            gpu.destroy_texture(depth);
        }
        self.depth = Some(gpu.create_texture(&TextureDesc {
            label: "3d depth",
            width: size[0],
            height: size[1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | if count == 1 {
                    TextureUsages::TRANSIENT_ATTACHMENT
                } else {
                    TextureUsages::empty()
                },
            sample_count: self.samples,
            mip_levels: 1,
        }));
        self.size = size;
        self.single_layer = count == 1;
        true
    }
    pub(crate) fn release(gpu: &mut dyn Gpu, node: NodeExtrusion) {
        if let Some(bundle) = node.bundle {
            gpu.destroy_render_bundle(bundle);
        }
        if let Some(group) = node.group {
            gpu.destroy_bind_group(group);
        }
        if let Some(color) = node.color {
            gpu.destroy_texture(color);
        }
        if let Some(msaa) = node.multisample {
            gpu.destroy_texture(msaa);
        }
    }
    pub(crate) fn resize_node(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        uniform: BindingResource,
        size: [u32; 2],
    ) -> bool {
        if node.size == size && node.samples == self.samples {
            return false;
        }
        Self::release(gpu, std::mem::take(node));
        let color = gpu.create_texture(&TextureDesc {
            label: "extrusion color",
            width: size[0],
            height: size[1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        let multisample = (self.samples > 1).then(|| {
            gpu.create_texture(&TextureDesc {
                label: "extrusion multisample",
                width: size[0],
                height: size[1],
                array_layers: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                sample_count: self.samples,
                mip_levels: 1,
            })
        });
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "extrusion composite",
            layout: self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Texture {
                        texture: color,
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });
        *node = NodeExtrusion {
            color: Some(color),
            multisample,
            samples: self.samples,
            size,
            group: Some(group),
            bundle: None,
        };
        true
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "one recorded draw and its attachments"
    )]
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        scene: BindGroupId,
        group: BindGroupId,
        buffers: &SceneBuffers,
        index: u32,
        first: bool,
    ) -> RenderPass {
        let bundle = *node.bundle.get_or_insert_with(|| {
            let pipeline = self.pipelines[usize::from(self.samples == 4)];
            let commands = [
                RenderCommand::SetPipeline(pipeline.expect("created by resize")),
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetBindGroup { index: 1, group },
                RenderCommand::SetIndexBuffer {
                    buffer: buffers.compacted,
                    format: IndexFormat::Uint32,
                    offset: 0,
                },
                RenderCommand::DrawIndexedIndirect {
                    buffer: buffers.args,
                    offset: crate::args_offset(index),
                },
            ];
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "extrusion",
                    color_formats: &[TextureFormat::Rgba8Unorm],
                    depth_format: Some(TextureFormat::Depth32Float),
                    sample_count: self.samples,
                },
                &commands,
            )
        });
        RenderPass {
            label: "extrusion".into(),
            color: ColorAttachment {
                store: if node.multisample.is_some() {
                    StoreOp::Discard
                } else {
                    StoreOp::Store
                },
                target: node.multisample.or(node.color).unwrap(),
                resolve: node.multisample.and(node.color),
                load: LoadOp::Clear([0.0; 4]),
            },
            depth: Some(DepthAttachment {
                store: if self.single_layer {
                    StoreOp::Discard
                } else {
                    StoreOp::Store
                },
                target: self.depth.unwrap(),
                load: if first {
                    LoadOp::Clear(0.0)
                } else {
                    LoadOp::Load
                },
            }),
            timestamps: None,
            commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
        }
    }
}
